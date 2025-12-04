# Added December 4, 2025 by Clara Burgert using Claude

"""
DELTA: Dense Efficient Long-range 3D Tracking for Any video (ICLR 2025)

This module provides simple functions to work with Snap Research's DELTA model,
which performs dense point tracking across video sequences in both 2D and 3D.

Capabilities:
- Dense 2D tracking: Track all pixels across frames (RGB-only input)
- Sparse 2D tracking: Track specific points across frames (RGB-only input)
- Dense 3D tracking: Track all pixels with depth (RGB-D input)
- Sparse 3D tracking: Track specific points with depth (RGB-D input)

Functions automatically download the model on first use. No class initialization needed.

Input formats:
- Videos: List of frames (numpy arrays), path, or URL
- Points: Nx2 numpy array of (x, y) coordinates for sparse tracking
- Depth: Optional depth maps for 3D tracking

Example:
    # Dense 2D tracking - track all pixels
    coords, visibility, confidence = track_video_dense_2d("video.mp4")

    # Sparse 2D tracking - track specific points
    coords, visibility, confidence = track_video_sparse_2d("video.mp4", points=[[100, 200], [300, 400]])

    # Dense 3D tracking (requires depth)
    trajectories_3d = track_video_dense_3d("video.mp4", depth_maps)

See: https://github.com/snap-research/DELTA_densetrack3d
"""
import rp
import torch
import numpy as np
from einops import rearrange
from typing import Optional, Union, List, Tuple
from pathlib import Path


__all__ = [
    "track_video_dense_2d",
    "track_video_sparse_2d",
    "track_video_dense_3d",
    "track_video_sparse_3d",
    "download_model",
    "visualize_tracks_2d",
]

PIP_REQUIREMENTS = [
    "torch",
    "einops",
    "timm",
    "kornia",
    "scipy",
    "tqdm",
    "jaxtyping",
    "decord",
    "omegaconf",
    "mediapy",
    "gdown",
    "gitpython",
    "opencv-python",
    "numpy<2",
]

# Model versions dict - 2D and 3D variants
MODELS = {
    "2d": {
        "gdrive_id": "1S_T7DzqBXMtr0voRC_XUGn1VTnPk_7Rm",
        "filename": "densetrack2d.pth",
    },
    "3d": {
        "gdrive_id": "18d5M3nl3AxbG4ZkT7wssvMXZXbmXrnjz",
        "filename": "densetrack3d.pth",
    },
}

# Default paths - network drive (shared) and local fallback
DEFAULT_MODEL_DIR = Path("/root/models/delta")
LOCAL_MODEL_DIR = Path("/models/delta")

# Global device tracking
_delta_device = None


def _ensure_all_gpus_visible():
    """
    Reset CUDA_VISIBLE_DEVICES to make all GPUs visible.
    This ensures that device="cuda:N" always refers to the physical GPU N,
    regardless of any CUDA_VISIBLE_DEVICES setting in the environment.
    """
    rp.set_cuda_visible_devices()  # No args = reset to all GPUs


def _default_delta_device():
    """
    Get or initialize the default device for DELTA model.
    Uses rp.select_torch_device() to pick the best available device.
    """
    global _delta_device
    if _delta_device is None:
        _delta_device = rp.select_torch_device(reserve=True)
    return _delta_device


def _get_model_path(model_type: str = "2d", model_dir: Optional[Path] = None) -> Path:
    """Get the path where the model should be stored/loaded from."""
    if model_dir is not None:
        return Path(model_dir) / MODELS[model_type]["filename"]

    # Try local first, then network
    local_path = LOCAL_MODEL_DIR / MODELS[model_type]["filename"]
    if local_path.exists():
        return local_path

    return DEFAULT_MODEL_DIR / MODELS[model_type]["filename"]


def download_model(model_type: str = "2d", model_dir: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download a DELTA model checkpoint from Google Drive.

    Args:
        model_type: "2d" or "3d" - which model variant to download
        model_dir: Directory to save the model. Defaults to /root/models/delta
        force: If True, re-download even if file exists

    Returns:
        Path to the downloaded model file
    """
    rp.pip_import("gdown")
    import gdown

    if model_type not in MODELS:
        raise ValueError(f"model_type must be '2d' or '3d', got '{model_type}'")

    model_info = MODELS[model_type]

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    output_path = model_dir / model_info["filename"]

    if output_path.exists() and not force:
        print(f"Model already exists at {output_path}")
        return output_path

    gdrive_url = f"https://drive.google.com/uc?id={model_info['gdrive_id']}"
    print(f"Downloading DELTA {model_type} model to {output_path}...")

    gdown.download(gdrive_url, str(output_path), quiet=False)

    print(f"Download complete: {output_path}")
    return output_path


def _ensure_delta_repo():
    """
    Ensure the DELTA repository is cloned and available.
    Returns the path to the repo.
    """
    rp.pip_import("git", "gitpython")
    import git

    repo_dir = Path("/root/models/delta/DELTA_densetrack3d")

    if not repo_dir.exists():
        print("Cloning DELTA repository...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(
            "https://github.com/snap-research/DELTA_densetrack3d.git",
            str(repo_dir),
            recursive=True
        )
        print(f"Repository cloned to {repo_dir}")

    return repo_dir


def _install_delta_deps():
    """Install DELTA dependencies."""
    rp.pip_import("einops")
    rp.pip_import("timm")
    rp.pip_import("kornia")
    rp.pip_import("scipy")
    rp.pip_import("tqdm")
    rp.pip_import("jaxtyping")
    rp.pip_import("decord")
    rp.pip_import("omegaconf")
    rp.pip_import("mediapy")


def _add_delta_to_path():
    """Add DELTA repo to Python path."""
    import sys
    repo_dir = _ensure_delta_repo()
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


@rp.memoized
def _get_delta_model_2d(model_path: str, device: torch.device):
    """
    Load and cache the DELTA 2D model.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, predictor)
    """
    _install_delta_deps()
    _add_delta_to_path()

    from densetrack3d.models.densetrack3d.densetrack2d import DenseTrack2D
    from densetrack3d.models.predictor.dense_predictor2d import DensePredictor2D

    # Create model with default config
    model = DenseTrack2D(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        upsample_factor=4,
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load with strict=False to handle minor version differences
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"DELTA 2D: Missing keys (may be OK): {len(missing)} keys")
    if unexpected:
        print(f"DELTA 2D: Unexpected keys (may be OK): {len(unexpected)} keys")

    model = model.to(device)
    model.eval()

    predictor = DensePredictor2D(model=model)
    predictor = predictor.to(device)
    predictor.eval()

    return model, predictor


@rp.memoized
def _get_delta_model_3d(model_path: str, device: torch.device):
    """
    Load and cache the DELTA 3D model.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, predictor)
    """
    _install_delta_deps()
    _add_delta_to_path()

    from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
    from densetrack3d.models.predictor.dense_predictor import DensePredictor3D

    # Create model with default config
    model = DenseTrack3D(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        upsample_factor=4,
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load with strict=False to handle minor version differences
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"DELTA 3D: Missing keys (may be OK): {len(missing)} keys")
    if unexpected:
        print(f"DELTA 3D: Unexpected keys (may be OK): {len(unexpected)} keys")

    model = model.to(device)
    model.eval()

    predictor = DensePredictor3D(model=model)
    predictor = predictor.to(device)
    predictor.eval()

    return model, predictor


def _get_delta_2d(model_path: Optional[str] = None, device=None):
    """Get the DELTA 2D model and predictor."""
    # Ensure all GPUs are visible so device="cuda:N" always means physical GPU N
    _ensure_all_gpus_visible()

    if model_path is None:
        model_path = _get_model_path("2d")
        if not Path(model_path).exists():
            model_path = download_model("2d")

    model_path = str(model_path)

    if device is None:
        device = _default_delta_device()
    else:
        global _delta_device
        _delta_device = device
    device = torch.device(device)

    return _get_delta_model_2d(model_path, device)


def _get_delta_3d(model_path: Optional[str] = None, device=None):
    """Get the DELTA 3D model and predictor."""
    # Ensure all GPUs are visible so device="cuda:N" always means physical GPU N
    _ensure_all_gpus_visible()

    if model_path is None:
        model_path = _get_model_path("3d")
        if not Path(model_path).exists():
            model_path = download_model("3d")

    model_path = str(model_path)

    if device is None:
        device = _default_delta_device()
    else:
        global _delta_device
        _delta_device = device
    device = torch.device(device)

    return _get_delta_model_3d(model_path, device)


def _load_video(video, num_frames: Optional[int] = None) -> np.ndarray:
    """
    Load and preprocess a video for DELTA model input.

    Args:
        video: Path, URL, or list of frames
        num_frames: Number of frames to sample (None = use all)

    Returns:
        numpy array of shape (T, H, W, 3) with uint8 values
    """
    if isinstance(video, str):
        video = rp.load_video(video)

    video = rp.as_numpy_images(video)

    if num_frames is not None:
        video = rp.resize_list(video, num_frames)

    video = np.stack([rp.as_rgb_image(rp.as_byte_image(f)) for f in video])

    return video


def _video_to_tensor(video: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert video numpy array to tensor for model input.

    Args:
        video: (T, H, W, 3) uint8 numpy array
        device: Target device

    Returns:
        (1, T, 3, H, W) float tensor normalized to [0, 1]
    """
    # THWC -> TCHW
    video_tensor = torch.from_numpy(video).float() / 255.0
    video_tensor = rearrange(video_tensor, 't h w c -> 1 t c h w')
    video_tensor = video_tensor.to(device)

    return video_tensor


def track_video_dense_2d(
    video,
    *,
    query_frame: int = 0,
    num_frames: Optional[int] = None,
    device=None,
    model_path: Optional[str] = None,
    use_fp16: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform dense 2D tracking on a video - tracks all pixels from a query frame.

    Args:
        video: Video as path, URL, or list of frames (numpy arrays)
        query_frame: Frame index to use as reference (default 0 = first frame)
        num_frames: Number of frames to process (None = all frames)
        device: Device to run inference on (e.g., 'cuda:0', 'cpu')
        model_path: Optional path to model checkpoint
        use_fp16: Use half precision for lower memory usage

    Returns:
        Tuple of:
            - coords: (T, H, W, 2) float32 array of (x, y) coordinates for each pixel
            - visibility: (T, H, W) bool array indicating if pixel is visible
            - confidence: (T, H, W) float32 array of confidence scores
    """
    model, predictor = _get_delta_2d(model_path, device)
    actual_device = next(model.parameters()).device

    # Load and prepare video
    video_np = _load_video(video, num_frames)
    T, H, W, _ = video_np.shape

    video_tensor = _video_to_tensor(video_np, actual_device)

    if use_fp16:
        video_tensor = video_tensor.half()

    # Run inference
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        result = predictor(
            video_tensor,
            grid_query_frame=query_frame,
        )

    # Extract outputs
    # trajs_uv is (B, T, N, 2) where N = H_dense * W_dense
    coords = result["trajs_uv"][0].cpu().numpy()  # (T, N, 2)
    vis = result["vis"][0].cpu().numpy()  # (T, N)
    conf = result["conf"][0].cpu().numpy()  # (T, N)

    # Get dense resolution from model
    dense_reso = result.get("dense_reso", None)
    if dense_reso is not None:
        H_d, W_d = dense_reso
    else:
        # Estimate from number of points
        N = coords.shape[1]
        # Assume roughly same aspect ratio
        H_d = int(np.sqrt(N * H / W))
        W_d = N // H_d

    # Reshape to spatial grid
    coords = coords.reshape(T, H_d, W_d, 2)
    vis = vis.reshape(T, H_d, W_d)
    conf = conf.reshape(T, H_d, W_d)

    # Scale coordinates to original resolution if needed
    if H_d != H or W_d != W:
        scale_y, scale_x = H / H_d, W / W_d
        coords = coords * np.array([scale_x, scale_y])
        # Resize vis and conf to original resolution using rp.resize_image with (H, W) tuple
        vis_resized = np.zeros((T, H, W), dtype=bool)
        conf_resized = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            vis_resized[t] = rp.resize_image(vis[t].astype(np.float32), (H, W)) > 0.5
            conf_resized[t] = rp.resize_image(conf[t], (H, W))
        vis = vis_resized
        conf = conf_resized
        coords_resized = np.zeros((T, H, W, 2), dtype=np.float32)
        for t in range(T):
            coords_resized[t, ..., 0] = rp.resize_image(coords[t, ..., 0], (H, W))
            coords_resized[t, ..., 1] = rp.resize_image(coords[t, ..., 1], (H, W))
        coords = coords_resized

    return coords.astype(np.float32), vis.astype(bool), conf.astype(np.float32)


def track_video_sparse_2d(
    video,
    points: Optional[np.ndarray] = None,
    *,
    query_frame: int = 0,
    num_frames: Optional[int] = None,
    grid_size: int = 20,
    device=None,
    model_path: Optional[str] = None,
    use_fp16: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform sparse 2D tracking on a video - tracks specific points.

    Args:
        video: Video as path, URL, or list of frames
        points: Nx2 array of (x, y) coordinates to track. If None, uses a grid.
        query_frame: Frame index to use as reference (default 0)
        num_frames: Number of frames to process (None = all)
        grid_size: If points is None, create a grid_size x grid_size grid of points
        device: Device to run inference on
        model_path: Optional path to model checkpoint
        use_fp16: Use half precision for lower memory

    Returns:
        Tuple of:
            - coords: (T, N, 2) float32 array of tracked coordinates
            - visibility: (T, N) bool array indicating visibility
            - confidence: (T, N) float32 array of confidence scores
    """
    model, predictor = _get_delta_2d(model_path, device)
    actual_device = next(model.parameters()).device

    # Load and prepare video
    video_np = _load_video(video, num_frames)
    T, H, W, _ = video_np.shape

    # Generate grid points if not provided
    if points is None:
        xs = np.linspace(0, W - 1, grid_size)
        ys = np.linspace(0, H - 1, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    points = np.asarray(points, dtype=np.float32)
    N = len(points)

    video_tensor = _video_to_tensor(video_np, actual_device)

    if use_fp16:
        video_tensor = video_tensor.half()

    # For sparse tracking, we run dense and then sample
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        result = predictor(
            video_tensor,
            grid_query_frame=query_frame,
        )

    # Get dense results
    coords_dense = result["trajs_uv"][0].cpu().numpy()  # (T, N_dense, 2)
    vis_dense = result["vis"][0].cpu().numpy()  # (T, N_dense)
    conf_dense = result["conf"][0].cpu().numpy()  # (T, N_dense)

    dense_reso = result.get("dense_reso", None)
    if dense_reso is not None:
        H_d, W_d = dense_reso
    else:
        N_dense = coords_dense.shape[1]
        H_d = int(np.sqrt(N_dense * H / W))
        W_d = N_dense // H_d

    # Reshape to grid
    coords_dense = coords_dense.reshape(T, H_d, W_d, 2)
    vis_dense = vis_dense.reshape(T, H_d, W_d)
    conf_dense = conf_dense.reshape(T, H_d, W_d)

    # Sample at query points using bilinear interpolation
    # Map points to dense grid coordinates
    scale_y, scale_x = H_d / H, W_d / W
    query_pts_scaled = points * np.array([scale_x, scale_y])

    coords_out = np.zeros((T, N, 2), dtype=np.float32)
    vis_out = np.zeros((T, N), dtype=bool)
    conf_out = np.zeros((T, N), dtype=np.float32)

    for i, (qx, qy) in enumerate(query_pts_scaled):
        # Bilinear interpolation
        x0, y0 = int(qx), int(qy)
        x1, y1 = min(x0 + 1, W_d - 1), min(y0 + 1, H_d - 1)
        x0, y0 = max(0, x0), max(0, y0)

        wx = qx - x0
        wy = qy - y0

        for t in range(T):
            # Interpolate coordinates
            c00 = coords_dense[t, y0, x0]
            c01 = coords_dense[t, y0, x1]
            c10 = coords_dense[t, y1, x0]
            c11 = coords_dense[t, y1, x1]

            coords_out[t, i] = (
                c00 * (1 - wx) * (1 - wy) +
                c01 * wx * (1 - wy) +
                c10 * (1 - wx) * wy +
                c11 * wx * wy
            )

            # Interpolate confidence
            conf_out[t, i] = (
                conf_dense[t, y0, x0] * (1 - wx) * (1 - wy) +
                conf_dense[t, y0, x1] * wx * (1 - wy) +
                conf_dense[t, y1, x0] * (1 - wx) * wy +
                conf_dense[t, y1, x1] * wx * wy
            )

            # Visibility: use nearest neighbor
            vis_out[t, i] = vis_dense[t, round(qy), round(qx)]

    # Scale coordinates back to original resolution
    scale_back_x, scale_back_y = W / W_d, H / H_d
    # Actually the coords are already in original space since model scales them
    # Just return as-is

    return coords_out, vis_out, conf_out


def track_video_dense_3d(
    video,
    depth: Optional[np.ndarray] = None,
    *,
    query_frame: int = 0,
    num_frames: Optional[int] = None,
    device=None,
    model_path: Optional[str] = None,
    use_fp16: bool = False,
    intrinsics: Optional[np.ndarray] = None,
) -> dict:
    """
    Perform dense 3D tracking on a video with depth information.

    Args:
        video: Video as path, URL, or list of frames
        depth: (T, H, W) depth maps. If None, will estimate using UniDepth.
        query_frame: Frame index to use as reference (default 0)
        num_frames: Number of frames to process (None = all)
        device: Device to run inference on
        model_path: Optional path to model checkpoint
        use_fp16: Use half precision for lower memory
        intrinsics: Optional (3, 3) camera intrinsics matrix

    Returns:
        Dictionary containing:
            - coords_2d: (T, H, W, 2) 2D coordinates
            - coords_3d: (T, H, W, 3) 3D coordinates (x, y, z)
            - depth: (T, H, W) depth values
            - visibility: (T, H, W) visibility mask
            - confidence: (T, H, W) confidence scores
    """
    model, predictor = _get_delta_3d(model_path, device)
    actual_device = next(model.parameters()).device

    # Load and prepare video
    video_np = _load_video(video, num_frames)
    T, H, W, _ = video_np.shape

    video_tensor = _video_to_tensor(video_np, actual_device)

    # Handle depth
    if depth is None:
        # Would need to estimate depth - for now raise error
        raise ValueError(
            "Depth maps are required for 3D tracking. "
            "Please provide depth maps or use track_video_dense_2d for RGB-only tracking."
        )

    depth = np.asarray(depth, dtype=np.float32)
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
    depth_tensor = depth_tensor.to(actual_device)

    if use_fp16:
        video_tensor = video_tensor.half()
        depth_tensor = depth_tensor.half()

    # Prepare intrinsics
    if intrinsics is not None:
        intrinsics_tensor = torch.from_numpy(intrinsics).float().unsqueeze(0).to(actual_device)
    else:
        intrinsics_tensor = None

    # Run inference
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        result = predictor(
            video_tensor,
            depth_tensor,
            grid_query_frame=query_frame,
            predefined_intrs=intrinsics_tensor,
        )

    # Extract outputs
    coords_2d = result["trajs_uv"][0].cpu().numpy()
    coords_depth = result["trajs_depth"][0].cpu().numpy()
    vis = result["vis"][0].cpu().numpy()
    conf = result["conf"][0].cpu().numpy()

    dense_reso = result.get("dense_reso", None)
    if dense_reso is not None:
        H_d, W_d = dense_reso
    else:
        N = coords_2d.shape[1]
        H_d = int(np.sqrt(N * H / W))
        W_d = N // H_d

    # Reshape to spatial grid
    coords_2d = coords_2d.reshape(T, H_d, W_d, 2)
    coords_depth = coords_depth.reshape(T, H_d, W_d)
    vis = vis.reshape(T, H_d, W_d)
    conf = conf.reshape(T, H_d, W_d)

    # Get 3D coordinates if available
    coords_3d = None
    if "trajs_3d_dict" in result:
        trajs_3d = result["trajs_3d_dict"]
        if "xyz" in trajs_3d:
            coords_3d = trajs_3d["xyz"][0].cpu().numpy()
            coords_3d = coords_3d.reshape(T, H_d, W_d, 3)

    return {
        "coords_2d": coords_2d.astype(np.float32),
        "coords_3d": coords_3d,
        "depth": coords_depth.astype(np.float32),
        "visibility": vis.astype(bool),
        "confidence": conf.astype(np.float32),
        "dense_resolution": (H_d, W_d),
    }


def track_video_sparse_3d(
    video,
    depth: np.ndarray,
    points: Optional[np.ndarray] = None,
    *,
    query_frame: int = 0,
    num_frames: Optional[int] = None,
    grid_size: int = 20,
    device=None,
    model_path: Optional[str] = None,
    use_fp16: bool = False,
    intrinsics: Optional[np.ndarray] = None,
) -> dict:
    """
    Perform sparse 3D tracking on a video with depth.

    Args:
        video: Video as path, URL, or list of frames
        depth: (T, H, W) depth maps
        points: Nx2 array of (x, y) coordinates to track. If None, uses a grid.
        query_frame: Frame index to use as reference
        num_frames: Number of frames to process
        grid_size: If points is None, create a grid_size x grid_size grid
        device: Device to run inference on
        model_path: Optional path to model checkpoint
        use_fp16: Use half precision
        intrinsics: Optional camera intrinsics matrix

    Returns:
        Dictionary containing tracked 3D coordinates and metadata
    """
    # Get dense results first
    result = track_video_dense_3d(
        video,
        depth,
        query_frame=query_frame,
        num_frames=num_frames,
        device=device,
        model_path=model_path,
        use_fp16=use_fp16,
        intrinsics=intrinsics,
    )

    T = result["coords_2d"].shape[0]
    H_d, W_d = result["dense_resolution"]

    # Load video to get original dimensions
    video_np = _load_video(video, num_frames)
    _, H, W, _ = video_np.shape

    # Generate grid points if not provided
    if points is None:
        xs = np.linspace(0, W - 1, grid_size)
        ys = np.linspace(0, H - 1, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    points = np.asarray(points, dtype=np.float32)
    N = len(points)

    # Map points to dense grid coordinates
    scale_y, scale_x = H_d / H, W_d / W
    query_pts_scaled = points * np.array([scale_x, scale_y])

    # Sample results at query points
    coords_2d_out = np.zeros((T, N, 2), dtype=np.float32)
    depth_out = np.zeros((T, N), dtype=np.float32)
    vis_out = np.zeros((T, N), dtype=bool)
    conf_out = np.zeros((T, N), dtype=np.float32)

    for i, (qx, qy) in enumerate(query_pts_scaled):
        x0, y0 = int(qx), int(qy)
        x1, y1 = min(x0 + 1, W_d - 1), min(y0 + 1, H_d - 1)
        x0, y0 = max(0, x0), max(0, y0)

        wx = qx - x0
        wy = qy - y0

        for t in range(T):
            # Bilinear interpolation
            c00 = result["coords_2d"][t, y0, x0]
            c01 = result["coords_2d"][t, y0, x1]
            c10 = result["coords_2d"][t, y1, x0]
            c11 = result["coords_2d"][t, y1, x1]

            coords_2d_out[t, i] = (
                c00 * (1 - wx) * (1 - wy) +
                c01 * wx * (1 - wy) +
                c10 * (1 - wx) * wy +
                c11 * wx * wy
            )

            depth_out[t, i] = (
                result["depth"][t, y0, x0] * (1 - wx) * (1 - wy) +
                result["depth"][t, y0, x1] * wx * (1 - wy) +
                result["depth"][t, y1, x0] * (1 - wx) * wy +
                result["depth"][t, y1, x1] * wx * wy
            )

            conf_out[t, i] = (
                result["confidence"][t, y0, x0] * (1 - wx) * (1 - wy) +
                result["confidence"][t, y0, x1] * wx * (1 - wy) +
                result["confidence"][t, y1, x0] * (1 - wx) * wy +
                result["confidence"][t, y1, x1] * wx * wy
            )

            vis_out[t, i] = result["visibility"][t, round(qy), round(qx)]

    output = {
        "coords_2d": coords_2d_out,
        "depth": depth_out,
        "visibility": vis_out,
        "confidence": conf_out,
        "query_points": points,
    }

    # Handle 3D coordinates if available
    if result["coords_3d"] is not None:
        coords_3d_out = np.zeros((T, N, 3), dtype=np.float32)
        for i, (qx, qy) in enumerate(query_pts_scaled):
            x0, y0 = int(qx), int(qy)
            x1, y1 = min(x0 + 1, W_d - 1), min(y0 + 1, H_d - 1)
            x0, y0 = max(0, x0), max(0, y0)
            wx = qx - x0
            wy = qy - y0

            for t in range(T):
                c00 = result["coords_3d"][t, y0, x0]
                c01 = result["coords_3d"][t, y0, x1]
                c10 = result["coords_3d"][t, y1, x0]
                c11 = result["coords_3d"][t, y1, x1]

                coords_3d_out[t, i] = (
                    c00 * (1 - wx) * (1 - wy) +
                    c01 * wx * (1 - wy) +
                    c10 * (1 - wx) * wy +
                    c11 * wx * wy
                )
        output["coords_3d"] = coords_3d_out

    return output


def visualize_tracks_2d(
    video,
    coords: np.ndarray,
    visibility: np.ndarray,
    *,
    downsample: int = 16,
    trail_length: int = 10,
    point_size: int = 3,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Visualize 2D tracking results on video frames.

    Args:
        video: Original video (path, URL, or frames)
        coords: (T, H, W, 2) or (T, N, 2) tracking coordinates
        visibility: (T, H, W) or (T, N) visibility mask
        downsample: Downsample factor for visualization (only show every Nth point)
        trail_length: Number of frames to show trailing path
        point_size: Size of tracked points
        show_progress: Show progress bar

    Returns:
        (T, H, W, 3) visualization video as numpy array
    """
    if isinstance(video, str):
        video = rp.load_video(video)

    video = np.stack([rp.as_rgb_image(rp.as_byte_image(f)) for f in video])
    T, H, W, _ = video.shape

    # Handle both dense (T, H, W, 2) and sparse (T, N, 2) formats
    if coords.ndim == 4:
        # Dense format - downsample to sparse
        coords = coords[:, ::downsample, ::downsample, :]
        visibility = visibility[:, ::downsample, ::downsample]
        coords = coords.reshape(T, -1, 2)
        visibility = visibility.reshape(T, -1)

    N = coords.shape[1]

    # Generate colors for each point (seeded for reproducibility)
    np.random.seed(42)
    colors = rp.random_rgb_byte_colors(N)

    output = video.copy()

    iterator = range(T)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Visualizing tracks")

    for t in iterator:
        frame = output[t].copy()

        for i in range(N):
            if not visibility[t, i]:
                continue

            x, y = coords[t, i]
            x, y = int(round(x)), int(round(y))

            if 0 <= x < W and 0 <= y < H:
                color = tuple(int(c) for c in colors[i])

                # Import cv2 for drawing
                import cv2

                # Draw trail
                curr_x, curr_y = x, y
                for dt in range(1, min(trail_length, t + 1)):
                    if visibility[t - dt, i]:
                        x_prev, y_prev = coords[t - dt, i]
                        x_prev, y_prev = int(round(x_prev)), int(round(y_prev))
                        if 0 <= x_prev < W and 0 <= y_prev < H:
                            # Draw line segment using cv2
                            cv2.line(frame, (curr_x, curr_y), (x_prev, y_prev), color, 1)
                            curr_x, curr_y = x_prev, y_prev

                # Draw current point
                x, y = coords[t, i]
                x, y = int(round(x)), int(round(y))
                cv2.circle(frame, (x, y), point_size, color, -1)

        output[t] = frame

    return output


def demo():
    """
    Run a demo of DELTA 2D tracking in a Jupyter notebook.
    Shows the model's capabilities on a sample video.
    """
    import rp

    # Update CommonSource
    rp.git_import('CommonSource', pull=True)
    import rp.git.CommonSource.delta as delta

    # Load a sample video
    video_url = 'https://videos.pexels.com/video-files/6507082/6507082-hd_1920_1080_25fps.mp4'
    video = rp.load_video(video_url, use_cache=True)
    video = rp.resize_video_to_fit(video, height=384, width=512, allow_growth=False)
    video = rp.resize_list_to_fit(video, 50)  # Limit frames for demo

    print("Running DELTA 2D tracking...")
    coords, vis, conf = delta.track_video_dense_2d(video)

    print("Visualizing results...")
    viz_video = delta.visualize_tracks_2d(video, coords, vis, downsample=32)

    print("Displaying video...")
    rp.display_video(viz_video, framerate=15)

    return coords, vis, conf


if __name__ == "__main__":
    import fire
    fire.Fire({name: globals()[name] for name in __all__})
