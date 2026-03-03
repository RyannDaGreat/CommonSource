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

Warps Gaussian white noise through optical flow fields while preserving
spatial uncorrelation. Uses Taichi for GPU/CPU compute with Brownian bridge
stochastic sampling.

Public API:
    warp_noise(init_noise, flows) -> generator of [H, W, C] noise arrays
    demo(video_path, ...)         -> runs RAFT optical flow + noise warping on a video

Reference:
    Deng et al., "Infinite-Resolution Integral Noise Warping for Diffusion Models", ICLR 2025
    https://openreview.net/forum?id=Y6LPWBo2HP

Examples:
    >>> import numpy as np
    >>> flows = [np.zeros((64, 64, 2)) for _ in range(5)]
    >>> init = np.random.randn(64, 64, 4)
    >>> noises = list(warp_noise(init, flows))
    >>> len(noises)
    6

    >>> # Full demo: video -> RAFT flow -> warped noise -> MP4
    >>> demo("path/to/video.mp4")
"""

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
# ParticleWarper (private)
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
        >>> # (requires Taichi init first)
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
# Public API
# ---------------------------------------------------------------------------


def warp_noise(init_noise, flows):
    """
    Warp noise through a sequence of optical flow fields.

    Each output frame is spatially-uncorrelated white Gaussian noise whose
    motion matches the input flows, via Brownian bridge sampling.

    Both inputs and outputs are generators/iterables, so flows are consumed
    one at a time — avoiding storing a full [T, H, W, 2] flow tensor in RAM.
    This matters because flow fields are large and often come from an optical
    flow model (e.g. RAFT) that can produce them lazily.

    Pure generator (no side effects beyond Taichi's one-time init).

    Args:
        init_noise (np.ndarray): Shape [H, W, C], initial Gaussian noise.
        flows: Iterable (or generator) of np.ndarray, each [H, W, 2].
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
    import taichi as ti

    _ensure_taichi_init()

    H, W, C = init_noise.shape
    ii, jj = np.meshgrid(
        np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij"
    )
    identity_cc = np.stack((ii, jj), axis=-1)  # (row, col)

    warper = _ParticleWarper(H, W, C, fp=ti.f64)

    yield init_noise

    prev_noise = init_noise
    for flow_dxdy in flows:
        # Convert (dx, dy) -> (row, col) for the kernel: row=dy, col=dx
        flow_rc = flow_dxdy[:, :, ::-1]
        warper.set_deformation(identity_cc - flow_rc)
        warper.set_noise(prev_noise)
        warper.run()
        prev_noise = warper.noise_field.to_numpy()
        yield prev_noise


def demo(
    video_path,
    noise_channels=4,
    output_folder=None,
    resize=None,
):
    """
    Run RAFT optical flow on a video, then warp noise through the resulting
    flow fields. Saves a noise video MP4.

    Streams end-to-end: RAFT flows are computed lazily, warp_noise yields one
    frame at a time, and save_video_mp4 accepts a generator — so neither the
    full flow tensor nor the full noise tensor needs to live in RAM at once.

    Not pure: reads video from disk, writes output files, prints progress.

    Args:
        video_path (str): Path to input video (MP4), folder of images,
                          glob pattern, or URL.
        noise_channels (int): Number of noise channels. Default 4.
        output_folder (str): Where to save outputs. Auto-chosen if None.
        resize (tuple or float): Resize input frames before computing flow.
                                 Tuple (H, W) or float scale factor.

    Returns:
        str: Path to the output folder.

    Examples:
        >>> # demo("bear.mp4")
        >>> # demo("path/to/frames/", noise_channels=3, resize=0.5)
    """
    import rp
    from rp.git.CommonSource.raft import RaftOpticalFlow

    # --- Load video frames ---
    if rp.is_video_file(video_path) or rp.is_valid_url(video_path):
        video_frames = rp.load_video(video_path)
    elif rp.is_a_folder(video_path):
        frame_paths = rp.get_all_image_files(video_path, sort_by="number")
        video_frames = rp.load_images(frame_paths, show_progress=True)
    else:
        import glob

        frame_paths = sorted(sorted(glob.glob(video_path)), key=len)
        assert frame_paths, "%s matched no files" % video_path
        video_frames = rp.load_images(frame_paths, show_progress=True)

    if resize is not None:
        video_frames = rp.resize_images(video_frames, size=resize, interp="area")

    video_frames = rp.as_rgb_images(video_frames)
    video_frames = np.stack(video_frames).astype(np.float32) / 255
    T, H, W, _ = video_frames.shape
    print("Input video: %d frames at %dx%d" % (T, H, W))

    # --- Setup output folder ---
    if output_folder is None:
        base = rp.get_file_name(video_path, include_file_extension=False)
        output_folder = rp.get_unique_copy_path("outputs/" + base)
    output_folder = rp.make_directory(output_folder)
    print("Output folder:", output_folder)

    # --- Compute RAFT optical flow (lazily) ---
    raft_model = RaftOpticalFlow()

    def _flow_generator():
        """
        Yields [H, W, 2] flow arrays in (dx, dy) convention from consecutive
        video frame pairs via RAFT. One at a time — no THW2 tensor in RAM.
        """
        for t in range(T - 1):
            dx, dy = raft_model(video_frames[t], video_frames[t + 1])
            dx = rp.as_numpy_array(dx)  # [H, W]
            dy = rp.as_numpy_array(dy)  # [H, W]
            print("  flow %d/%d" % (t + 1, T - 1))
            yield np.stack([dx, dy], axis=-1)  # [H, W, 2] in (dx, dy)

    # --- Warp noise -> video (fully streaming, nothing accumulated) ---
    init_noise = np.random.randn(H, W, noise_channels)

    def _noise_to_rgb(noise):
        """Pure. Converts [H, W, C] noise to [H, W, 3] clipped RGB for video."""
        return (noise[:, :, :3] / 4 + 0.5).clip(0, 1)

    mp4_path = rp.path_join(output_folder, "noise_video.mp4")
    rp.save_video_mp4(
        (_noise_to_rgb(n) for n in warp_noise(init_noise, _flow_generator())),
        mp4_path,
        video_bitrate="max", framerate=30,
        height=H, width=W,
    )
    print("Saved", mp4_path)

    return output_folder


if __name__ == "__main__":
    import fire

    fire.Fire(dict(demo=demo))
