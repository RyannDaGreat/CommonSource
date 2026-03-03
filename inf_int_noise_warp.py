# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "taichi",
#   "torch",
#   "torchvision",
#   "fire",
#   "rp",
# ]
# ///

"""
Infinite-Resolution Integral Noise Warping (ICLR 2025)

Implements the particle-based variant (Algorithm 3) from the paper. The paper
also describes a grid-based variant (Algorithm 2) that models deformed pixel
regions as octagons — that variant is not implemented here because it breaks
with non-diffeomorphic (non-injective) deformation maps. The particle-based
approach treats pixels as particles distributed to grid cells via bilinear
weighting kernels, making it agnostic to non-injective maps and ~42x faster
on GPU.

Warps Gaussian white noise through optical flow fields while preserving
spatial uncorrelation via Brownian bridge stochastic sampling. Uses Taichi
for GPU/CPU compute.

Public API:
    warp_noise(init_noise, flows)        -> generator of [H, W, C] noise arrays
    warp_video_noise(video, flow_fn=...) -> generator of EasyDict(frame, flow, noise)
    demo_video_file(video_path=...)      -> saves side-by-side MP4
    demo_webcam(...)                     -> live webcam display

Reference:
    Deng et al., "Infinite-Resolution Integral Noise Warping for Diffusion Models", ICLR 2025
    https://openreview.net/forum?id=Y6LPWBo2HP
    https://github.com/yitongdeng-projects/infinite_resolution_integral_noise_warping_code

Examples:
    >>> import numpy as np
    >>> flows = [np.zeros((64, 64, 2)) for _ in range(5)]
    >>> init = np.random.randn(64, 64, 4)
    >>> noises = list(warp_noise(init, flows))
    >>> len(noises)
    6
"""

import itertools
import math
import platform

import numpy as np

# ---------------------------------------------------------------------------
# Taichi lazy init
# ---------------------------------------------------------------------------

_taichi_initialized = False


def _ensure_taichi_init():
    """
    Initializes Taichi once per process.
    macOS: CPU backend (Metal is unreliable for scatter-pattern float atomics).
    Linux/Windows: CUDA GPU backend.
    """
    global _taichi_initialized
    if _taichi_initialized:
        return

    import taichi as ti

    if platform.system() == "Darwin":
        ti.init(arch=ti.cpu, debug=False, default_fp=ti.f64, random_seed=0)
    else:
        ti.init(
            arch=ti.gpu,
            device_memory_GB=4.0,
            debug=False,
            default_fp=ti.f64,
            random_seed=0,
        )
    _taichi_initialized = True


# ---------------------------------------------------------------------------
# Taichi kernels & funcs (inlined from the ICLR 2025 reference code)
# ---------------------------------------------------------------------------

_EPS = 1.0e-6


def _define_taichi_kernels():
    """
    Defines all Taichi kernels/funcs. Must be called after ti.init().
    Returns a dict of {name: callable}.

    Not pure: creates Taichi IR objects as a side effect of decoration.
    """
    import taichi as ti

    @ti.func
    def _get_randn_like(noise):
        return ti.Vector([ti.randn() for _ in range(noise.n)])

    @ti.kernel
    def _fill_noise(result: ti.template()):
        for I in ti.grouped(result):
            result[I] = ti.Vector([ti.randn() for _ in range(result.n)])

    @ti.func
    def _unravel_index(index, n, m):
        s = index // m
        t = index % m
        return s, t

    @ti.func
    def _ravel_index(s, t, n, m):
        return s * m + t

    @ti.func
    def _sample_brownian_bridge(x, t1, t2, q):
        """B(t) = W(t) - tW(1) + tx; sample B(t2) | B(t1) = q."""
        sample = x
        denom = 1.0 - t1
        if t2 < 1.0 and denom > _EPS:
            mu = (1.0 - t2) / denom * q + (t2 - t1) / denom * x
            var = (t2 - t1) * (1 - t2) / denom
            z = _get_randn_like(x)
            sample = z * ti.math.sqrt(var) + mu
        return sample

    @ti.func
    def _is_in_bound(u, v, n, m):
        result = False
        if u >= 0 and v >= 0 and u < n and v < m:
            result = True
        return result

    @ti.kernel
    def _particle_warp_kernel(
        map_field: ti.template(),
        noise_field: ti.template(),
        buffer_field: ti.template(),
        ticket_serial_field: ti.template(),
        master_field: ti.template(),
        area_field: ti.template(),
        pixel_area_field: ti.template(),
    ):
        img_n = noise_field.shape[0]
        img_m = noise_field.shape[1]

        # Phase 1: clear
        for i, j in noise_field:
            pixel_area_field[i, j] *= 0
            buffer_field[i, j] *= 0
            ticket_serial_field[i, j] *= 0

        # Phase 2: backward map — distribute weighted requests via atomic tickets
        for i, j in noise_field:
            raveled_index = _ravel_index(i, j, img_n, img_m)
            warped_pos = map_field[i, j] - 0.5
            lower_corner = ti.math.floor(warped_pos)
            frac = warped_pos - lower_corner
            lower_x, lower_y = int(lower_corner.x), int(lower_corner.y)
            upper_x, upper_y = lower_x + 1, lower_y + 1

            if _is_in_bound(lower_x, lower_y, img_n, img_m):
                w = (1.0 - frac.x) * (1.0 - frac.y)
                if w > 0.0:
                    t = ti.atomic_add(ticket_serial_field[lower_x, lower_y], 1)
                    master_field[lower_x, lower_y, t] = raveled_index
                    area_field[lower_x, lower_y, t] = w

            if _is_in_bound(upper_x, upper_y, img_n, img_m):
                w = frac.x * frac.y
                if w > 0.0:
                    t = ti.atomic_add(ticket_serial_field[upper_x, upper_y], 1)
                    master_field[upper_x, upper_y, t] = raveled_index
                    area_field[upper_x, upper_y, t] = w

            if _is_in_bound(lower_x, upper_y, img_n, img_m):
                w = (1.0 - frac.x) * frac.y
                if w > 0.0:
                    t = ti.atomic_add(ticket_serial_field[lower_x, upper_y], 1)
                    master_field[lower_x, upper_y, t] = raveled_index
                    area_field[lower_x, upper_y, t] = w

            if _is_in_bound(upper_x, lower_y, img_n, img_m):
                w = frac.x * (1.0 - frac.y)
                if w > 0.0:
                    t = ti.atomic_add(ticket_serial_field[upper_x, lower_y], 1)
                    master_field[upper_x, lower_y, t] = raveled_index
                    area_field[upper_x, lower_y, t] = w

        # Phase 3: Brownian bridge sample + scatter back to source pixels
        for u, v in noise_field:
            total_request = 0.0
            k_idx = 0
            access_record = area_field[u, v, k_idx]
            while access_record > 0.0:
                total_request += access_record
                k_idx += 1
                access_record = area_field[u, v, k_idx]

            if total_request > 0.0:
                k_idx = 0
                access_record = area_field[u, v, k_idx]
                access_source = master_field[u, v, k_idx]
                past_range = 0.0
                past_value = ti.Vector(
                    [0.0 for _ in ti.static(range(noise_field.n))]
                )
                while access_record > 0.0:
                    curr_normalized_request = access_record / total_request
                    source_i, source_j = _unravel_index(
                        access_source, img_n, img_m
                    )
                    next_range = past_range + curr_normalized_request
                    next_value = _sample_brownian_bridge(
                        noise_field[u, v], past_range, next_range, past_value
                    )
                    curr_value = next_value - past_value
                    past_range = next_range
                    past_value = next_value
                    buffer_field[source_i, source_j] += curr_value
                    pixel_area_field[source_i, source_j] += (
                        curr_normalized_request
                    )
                    area_field[u, v, k_idx] *= 0
                    k_idx += 1
                    access_record = area_field[u, v, k_idx]
                    access_source = master_field[u, v, k_idx]

        # Phase 4: normalize — preserve variance; unassigned pixels get fresh noise
        for i, j in noise_field:
            pixel_area = pixel_area_field[i, j]
            if pixel_area > 0.0:
                noise_field[i, j] = (
                    1.0 / ti.math.sqrt(pixel_area) * buffer_field[i, j]
                )
            else:
                noise_field[i, j] = _get_randn_like(noise_field)

    return dict(
        fill_noise=_fill_noise,
        particle_warp_kernel=_particle_warp_kernel,
    )


_kernels = None


def _get_kernels():
    """Returns the dict of compiled Taichi kernels, creating them on first call."""
    global _kernels
    if _kernels is None:
        _kernels = _define_taichi_kernels()
    return _kernels


# ---------------------------------------------------------------------------
# _ParticleWarper (private)
# ---------------------------------------------------------------------------


class _ParticleWarper:
    """
    Taichi-backed particle warper. Allocates fields and runs the 4-phase kernel.

    Not pure: mutates internal Taichi fields.

    Args:
        im_height (int): Image height.
        im_width  (int): Image width.
        num_noise_channel (int): Number of noise channels.
        fp: Taichi float type (ti.f32 or ti.f64).

    Examples:
        >>> # w = _ParticleWarper(64, 64, 4, ti.f64)
    """

    def __init__(self, im_height, im_width, num_noise_channel, fp):
        import taichi as ti

        kernels = _get_kernels()
        self._particle_warp_kernel = kernels["particle_warp_kernel"]
        self._np_dtype = np.float32 if fp == ti.f32 else np.float64

        self.master_field = ti.field(ti.i32)
        self.area_field = ti.field(fp)
        dense_size = 8
        max_entries = 10000

        dims = (
            math.ceil(im_height / dense_size),
            math.ceil(im_width / dense_size),
            math.ceil(max_entries / dense_size),
        )
        try:
            block = ti.root.pointer(ti.ijk, dims)
        except RuntimeError:
            max_entries = 512
            dims = (
                math.ceil(im_height / dense_size),
                math.ceil(im_width / dense_size),
                math.ceil(max_entries / dense_size),
            )
            print(
                "Sparse SNodes not supported; using dense layout"
                " (max_entries=%d)" % max_entries
            )
            block = ti.root.dense(ti.ijk, dims)

        pixel = block.dense(ti.ijk, (dense_size, dense_size, dense_size))
        pixel.place(self.master_field, self.area_field)

        self.noise_field = ti.Vector.field(
            num_noise_channel, fp, shape=(im_height, im_width)
        )
        kernels["fill_noise"](self.noise_field)
        self.buffer_field = ti.Vector.field(
            num_noise_channel, fp, shape=(im_height, im_width)
        )
        self.pixel_area_field = ti.field(fp, shape=(im_height, im_width))
        self.ticket_serial_field = ti.field(
            ti.i32, shape=(im_height, im_width)
        )
        self.map_field = ti.Vector.field(2, fp, shape=(im_height, im_width))

    def set_noise(self, noise_array):
        self.noise_field.from_numpy(noise_array.astype(self._np_dtype))

    def set_deformation(self, map_array):
        self.map_field.from_numpy(map_array.astype(self._np_dtype))

    def run(self):
        self._particle_warp_kernel(
            self.map_field,
            self.noise_field,
            self.buffer_field,
            self.ticket_serial_field,
            self.master_field,
            self.area_field,
            self.pixel_area_field,
        )


# ---------------------------------------------------------------------------
# Shared helpers (private)
# ---------------------------------------------------------------------------


def _make_warper(H, W, C):
    """
    Create a _ParticleWarper and its cell-center identity map.

    Not pure: calls _ensure_taichi_init(), allocates Taichi fields.

    Args:
        H (int): Image height.
        W (int): Image width.
        C (int): Number of noise channels.

    Returns:
        tuple: (_ParticleWarper, identity_cc [H, W, 2] (row, col))

    Examples:
        >>> # warper, identity_cc = _make_warper(64, 64, 4)
    """
    import taichi as ti

    _ensure_taichi_init()
    ii, jj = np.meshgrid(
        np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij"
    )
    identity_cc = np.stack((ii, jj), axis=-1)  # [H, W, 2] (row, col)
    warper = _ParticleWarper(H, W, C, fp=ti.f64)
    return warper, identity_cc


def _warp_step(warper, identity_cc, prev_noise, flow_dxdy):
    """
    Run one warp step: apply flow to prev_noise, return new noise.

    Not pure: mutates warper's internal Taichi fields.

    Args:
        warper (_ParticleWarper): The warper instance.
        identity_cc (np.ndarray): [H, W, 2] cell-center identity map (row, col).
        prev_noise (np.ndarray): [H, W, C] previous noise frame.
        flow_dxdy (np.ndarray): [H, W, 2] optical flow (dx, dy).

    Returns:
        np.ndarray: [H, W, C] warped noise.

    Examples:
        >>> # new_noise = _warp_step(warper, identity_cc, old_noise, flow)
    """
    flow_rc = flow_dxdy[:, :, ::-1]  # (dx, dy) -> (row, col) = (dy, dx)
    warper.set_deformation(identity_cc - flow_rc)
    warper.set_noise(prev_noise)
    warper.run()
    return warper.noise_field.to_numpy()


def _default_flow_fn(device):
    """
    Returns a flow function using RAFT. Lazily imports and instantiates the model.

    Not pure: imports modules, allocates GPU model.

    Args:
        device: Torch device string for RAFT (e.g. 'cuda', 'cpu').

    Returns:
        callable: (img_a, img_b) -> np.ndarray [H, W, 2] (dx, dy).
                  img_a, img_b are images as defined by rp.is_image.

    Examples:
        >>> # fn = _default_flow_fn('cpu')
        >>> # flow = fn(frame0, frame1)  # [H, W, 2] (dx, dy)
    """
    import rp
    from rp.git.CommonSource.raft import RaftOpticalFlow

    raft = RaftOpticalFlow(device)

    def _flow_fn(img_a, img_b):
        dx, dy = raft(img_a, img_b)  # each [H, W] torch tensor
        dx = rp.as_numpy_array(dx)  # [H, W]
        dy = rp.as_numpy_array(dy)  # [H, W]
        return np.stack([dx, dy], axis=-1)  # [H, W, 2] (dx, dy)

    return _flow_fn


def _resolve_video_frames(video):
    """
    Convert a video source into an iterator of frames.

    Not pure: may load images from disk or stream video.

    Args:
        video: Video path (str), URL, folder, glob, or iterable of images.

    Returns:
        iterator: Yields frames (as rp.is_image).

    Examples:
        >>> # frames = _resolve_video_frames("bear.mp4")
        >>> # frames = _resolve_video_frames([img1, img2, img3])
    """
    import rp

    if isinstance(video, str):
        if rp.is_video_file(video) or rp.is_valid_url(video):
            return rp.load_video_stream(video)
        elif rp.is_a_folder(video):
            paths = rp.get_all_image_files(video, sort_by="number")
            return (rp.load_image(p) for p in paths)
        else:
            import glob

            paths = sorted(sorted(glob.glob(video)), key=len)
            assert paths, "%s matched no files" % video
            return (rp.load_image(p) for p in paths)
    return iter(video)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_VIDEO_URL = (
    "https://www.shutterstock.com/shutterstock/videos/1100085499/preview/"
    "stock-footage-bremen-germany-october-old-style-carousel-moving-on-"
    "square-in-city-horses-on-traditional.webm"
)


def warp_noise(init_noise, flows):
    """
    Warp noise through a sequence of optical flow fields.

    Uses the particle-based warping algorithm (Algorithm 3 from the paper):
    bilinear backward mapping with Brownian bridge stochastic sampling.
    Each output frame is spatially-uncorrelated white Gaussian noise whose
    motion matches the input flows.

    Both inputs and outputs are generators/iterables, so flows are consumed
    one at a time — avoiding storing a full [T, H, W, 2] flow tensor in RAM.
    This matters because flow fields are large and often come from an optical
    flow model (e.g. RAFT) that can produce them lazily.

    Pure generator (no side effects beyond Taichi's one-time init).

    Args:
        init_noise (np.ndarray): Shape [H, W, C], initial Gaussian noise.
        flows: Iterable (or generator) of np.ndarray, each [H, W, 2] (dx, dy).
               Convention: (dx, dy) where dx=horizontal, dy=vertical.
               Same as RAFT and noise_warp.py.

    Yields:
        np.ndarray: Shape [H, W, C]. First yield is init_noise unchanged.

    Examples:
        >>> import numpy as np
        >>> flows = [np.zeros((64, 64, 2)) for _ in range(3)]
        >>> init = np.random.randn(64, 64, 4)
        >>> noises = list(warp_noise(init, flows))
        >>> len(noises)
        4
        >>> noises[0] is init
        True

        >>> # Streaming from an optical flow model (no THW2 tensor in RAM):
        >>> # def raft_flows(frames):
        >>> #     for i in range(len(frames) - 1):
        >>> #         yield compute_flow(frames[i], frames[i+1])
        >>> # for noise in warp_noise(init, raft_flows(video)):
        >>> #     process(noise)
    """
    H, W, C = init_noise.shape
    warper, identity_cc = _make_warper(H, W, C)

    yield init_noise

    prev_noise = init_noise
    for flow in flows:
        prev_noise = _warp_step(warper, identity_cc, prev_noise, flow)
        yield prev_noise


def warp_video_noise(
    video,
    flow_fn=None,
    noise_channels=4,
    init_noise=None,
    device=None,
    show_progress=True,
):
    """
    Compute optical flow from video frames and warp noise through it.

    Fully streaming: frames, flows, and noise are produced one at a time —
    no full video/flow/noise tensor needs to live in RAM at once.

    Not pure: may load video from disk, may instantiate flow model.

    Args:
        video: Video source. Can be:
               - str: video file path, URL, folder of images, or glob pattern
               - iterable of images as defined by rp.is_image (HWC numpy, PIL, etc)
        flow_fn: callable(img_a, img_b) -> np.ndarray [H, W, 2] (dx, dy).
                 None uses RAFT (requires torch, torchvision).
        noise_channels (int): Number of noise channels.
        init_noise (np.ndarray or None): [H, W, C] initial noise.
                   None = np.random.randn(H, W, noise_channels).
        device: Torch device for default flow_fn (RAFT). None uses
                rp.select_torch_device(prefer_used=True, reserve=True).
                Ignored if flow_fn is provided.
        show_progress (bool): Show progress bar via rp.eta.

    Yields:
        EasyDict with keys:
            frame: input video frame (as received from video iterable)
            flow:  np.ndarray [H, W, 2] (dx, dy), or None for first frame
            noise: np.ndarray [H, W, C] warped noise

    Examples:
        >>> # for out in warp_video_noise("bear.mp4"):
        >>> #     print(out.noise.shape, out.flow is None)
        >>> # for out in warp_video_noise(webcam_gen(), flow_fn=my_flow):
        >>> #     display(out.noise)
    """
    import rp

    # --- Resolve video to frame iterator ---
    frames = _resolve_video_frames(video)

    if show_progress:
        frames = rp.eta(frames, title="warp_video_noise")

    # --- Resolve flow function (lazy RAFT import) ---
    if flow_fn is None:
        if device is None:
            device = rp.select_torch_device(prefer_used=True, reserve=True)
        flow_fn = _default_flow_fn(device)

    # --- First frame ---
    frames = iter(frames)
    first_frame = next(frames)
    H, W = rp.get_image_dimensions(first_frame)

    if init_noise is None:
        init_noise = np.random.randn(H, W, noise_channels)

    warper, identity_cc = _make_warper(H, W, noise_channels)

    yield rp.as_easydict(frame=first_frame, flow=None, noise=init_noise)

    # --- Subsequent frames: flow -> warp -> yield ---
    prev_frame = first_frame
    prev_noise = init_noise
    for frame in frames:
        flow = flow_fn(prev_frame, frame)
        prev_noise = _warp_step(warper, identity_cc, prev_noise, flow)
        yield rp.as_easydict(frame=frame, flow=flow, noise=prev_noise)
        prev_frame = frame


def demo_video_file(
    video_path=_DEFAULT_VIDEO_URL,
    noise_channels=4,
    output_folder=None,
    resize=None,
):
    """
    Run optical flow on a video file and save a side-by-side MP4
    of [input frame | warped noise visualization].

    Fully streaming — video frames are never all loaded into RAM.

    Not pure: loads video from disk/URL, writes MP4, prints progress.

    Args:
        video_path (str): Path/URL to input video. Defaults to a carousel demo.
        noise_channels (int): Number of noise channels.
        output_folder (str): Where to save output. Auto-chosen if None.
        resize (tuple or float): Resize input frames before computing flow.
                                 Tuple (H, W) or float scale factor.

    Returns:
        str: Path to the output folder.

    Examples:
        >>> # demo_video_file()  # uses default carousel video
        >>> # demo_video_file("bear.mp4", resize=0.5)
    """
    import rp

    # --- Stream video frames ---
    frames = _resolve_video_frames(video_path)

    if resize is not None:
        frames = (rp.cv_resize_image(f, resize) for f in frames)

    # Peek at first frame for H, W, then chain it back
    frames = iter(frames)
    first = next(frames)
    H, W = rp.get_image_dimensions(first)
    frames = itertools.chain([first], frames)

    # --- Output folder ---
    if output_folder is None:
        base = rp.get_file_name(video_path, include_file_extension=False)
        output_folder = rp.get_unique_copy_path("outputs/" + base)
    output_folder = rp.make_directory(output_folder)
    print("Output folder:", output_folder)

    # --- Stream side-by-side visualization into MP4 ---
    def _vis_gen():
        for out in warp_video_noise(frames, noise_channels=noise_channels):
            frame_rgb = rp.as_float_image(rp.as_rgb_image(out.frame))
            noise_rgb = (out.noise[:, :, :3] / 5 + 0.5).clip(0, 1)
            yield rp.horizontally_concatenated_images(frame_rgb, noise_rgb)

    mp4_path = rp.path_join(output_folder, "noise_video.mp4")
    rp.save_video_mp4(
        _vis_gen(),
        mp4_path,
        video_bitrate="max",
        framerate=30,
        height=H,
        width=W * 2,
    )
    print("Saved", mp4_path)
    return output_folder


def demo_webcam(noise_channels=3, height=128):
    """
    Live webcam noise warping using Farneback optical flow (OpenCV).
    No torch/RAFT needed — runs on Mac CPU.
    Press 'q' to quit.

    Not pure: captures webcam, displays live window via cv_imshow.

    Args:
        noise_channels (int): Number of noise channels.
        height (int): Resize webcam frames to this height (preserves aspect).

    Examples:
        >>> # demo_webcam()
        >>> # demo_webcam(height=64, noise_channels=4)
    """
    import cv2
    import rp

    def _webcam_frames():
        """
        Not pure: reads from webcam via rp.load_webcam_stream().

        Yields:
            np.ndarray: RGB HWC frames resized to target height.

        Examples:
            >>> # for frame in _webcam_frames(): display(frame)
        """
        for frame in rp.load_webcam_stream():
            yield rp.resize_image_to_fit(frame, height=height)

    def _farneback_flow(img_a, img_b):
        """
        Pure. Computes Farneback optical flow between two images.

        Args:
            img_a: First image (as rp.is_image).
            img_b: Second image (as rp.is_image).

        Returns:
            np.ndarray: [H, W, 2] flow in (dx, dy).

        Examples:
            >>> # flow = _farneback_flow(frame0, frame1)
        """
        flow_2hw = rp.cv_optical_flow(
            img_a, img_b, algorithm="Farneback"
        )  # (2, H, W) (dx, dy)
        return np.transpose(flow_2hw, (1, 2, 0))  # [H, W, 2] (dx, dy)

    try:
        for out in warp_video_noise(
            _webcam_frames(),
            flow_fn=_farneback_flow,
            noise_channels=noise_channels,
            show_progress=False,
        ):
            frame_rgb = rp.as_float_image(rp.as_rgb_image(out.frame))
            noise_rgb = (out.noise[:, :, :3] / 5 + 0.5).clip(0, 1)
            vis = rp.horizontally_concatenated_images(frame_rgb, noise_rgb)
            rp.cv_imshow(vis, label="Infinite Integral Noise Warp", wait=None)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import fire

    fire.Fire(
        dict(demo_video_file=demo_video_file, demo_webcam=demo_webcam)
    )
