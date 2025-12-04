# Added December 3, 2025 by Clara Burgert using Claude

"""
Video-Depth-Anything: Consistent depth estimation for super-long videos (CVPR 2025 Highlight)

This module provides simple functions to work with ByteDance's Video-Depth-Anything models,
which perform temporally consistent monocular depth estimation on videos.

Features:
- Relative depth estimation (normalized depth maps)
- Metric depth estimation (real-world scale in meters)
- Temporally consistent results across arbitrarily long videos
- Support for multiple model sizes: small (28M), base (113M), large (382M)

Functions automatically download models on first use. No class initialization needed.

Input formats:
- Videos: np.ndarray (THWC uint8 or float 0-1), list of frames, path, or URL
- Single images: np.ndarray (HWC), PIL Image, or path/URL

Output:
- Depth maps: np.ndarray with shape (T, H, W) for videos or (H, W) for single images
- Values are relative depth (higher = farther) or metric depth in meters

Example:
    # Get depth for a video
    depths = estimate_video_depth("video.mp4")

    # Get depth for a single image
    depth = estimate_image_depth("photo.jpg")

    # Use metric depth model
    depths = estimate_video_depth("video.mp4", metric=True)

See: https://github.com/DepthAnything/Video-Depth-Anything
"""
import rp
from typing import Union, Optional, List
import os

# numpy and torch are imported lazily via _ensure_dependencies()

__all__ = [
    "estimate_video_depth",
    "estimate_image_depth",
    "download_model",
    "get_available_models",
    "demo",
]

PIP_REQUIREMENTS = [
    "torch",
    "torchvision",
    "opencv-python",
    "einops",
    "tqdm",
    "xformers",
    "gitpython",
    "numpy<2",
]

# Model configurations
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# HuggingFace download URLs
MODEL_URLS = {
    "vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
    "metric_vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth",
    "metric_vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth",
    "metric_vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth",
}

# Default paths - network drive first, can override with local path
DEFAULT_MODEL_DIR = "/root/models/video_depth_anything"
LOCAL_MODEL_DIR = "/models/video_depth_anything"

# Global device tracking
_video_depth_device = None


def _default_device():
    """
    Get or initialize the default device for Video-Depth-Anything.
    Uses rp.select_torch_device() to pick the best available device.
    """
    global _video_depth_device
    if _video_depth_device is None:
        _ensure_dependencies()
        _video_depth_device = rp.select_torch_device(reserve=True)
    return _video_depth_device


def get_available_models() -> dict:
    """
    Returns a dictionary describing all available model variants.

    Returns:
        dict: Model configurations with parameter counts and descriptions
    """
    return {
        "vits": {"params": "28.4M", "description": "Small model - fastest, lower quality"},
        "vitb": {"params": "113.1M", "description": "Base model - balanced speed/quality"},
        "vitl": {"params": "381.8M", "description": "Large model - best quality, slowest"},
    }


def _get_model_path(encoder: str = "vitl", metric: bool = False, model_dir: Optional[str] = None) -> str:
    """
    Get the local path where a model checkpoint should be stored.

    Args:
        encoder: Model encoder type ('vits', 'vitb', or 'vitl')
        metric: Whether to use metric depth model
        model_dir: Override default model directory

    Returns:
        str: Full path to the model checkpoint file
    """
    if model_dir is None:
        # Check if local drive exists and use it if so
        if os.path.isdir("/models"):
            model_dir = LOCAL_MODEL_DIR
        else:
            model_dir = DEFAULT_MODEL_DIR

    prefix = "metric_video_depth_anything" if metric else "video_depth_anything"
    filename = f"{prefix}_{encoder}.pth"
    return os.path.join(model_dir, filename)


def download_model(
    encoder: str = "vitl",
    metric: bool = False,
    model_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download a Video-Depth-Anything model checkpoint.

    Args:
        encoder: Model encoder type ('vits', 'vitb', or 'vitl')
        metric: Whether to download metric depth model
        model_dir: Directory to save the model (default: /root/models/video_depth_anything)
        force: Re-download even if file exists

    Returns:
        str: Path to the downloaded model file
    """
    assert encoder in MODEL_CONFIGS, f"Invalid encoder: {encoder}. Choose from: {list(MODEL_CONFIGS.keys())}"

    model_path = _get_model_path(encoder, metric, model_dir)
    model_dir_actual = os.path.dirname(model_path)

    # Create directory if needed
    os.makedirs(model_dir_actual, exist_ok=True)

    if os.path.exists(model_path) and not force:
        print(f"Model already exists at: {model_path}")
        return model_path

    # Get URL
    url_key = f"metric_{encoder}" if metric else encoder
    url = MODEL_URLS[url_key]

    print(f"Downloading {'metric ' if metric else ''}Video-Depth-Anything {encoder.upper()} model...")
    print(f"URL: {url}")
    print(f"Destination: {model_path}")

    # Download with progress bar using rp
    rp.download_url(url, model_path)

    print(f"Download complete: {model_path}")
    return model_path


def _ensure_dependencies():
    """Install required dependencies if not present."""
    rp.pip_import("numpy", auto_yes=True)
    rp.pip_import("torch", auto_yes=True)
    rp.pip_import("torchvision", auto_yes=True)
    rp.pip_import("cv2", "opencv-python", auto_yes=True)
    rp.pip_import("einops", auto_yes=True)
    rp.pip_import("tqdm", auto_yes=True)
    rp.pip_import("xformers", auto_yes=True)


def _clone_video_depth_anything_repo():
    """Clone the Video-Depth-Anything repo if not already present."""
    import sys

    repo_path = "/tmp/Video-Depth-Anything"
    if not os.path.exists(repo_path):
        print("Cloning Video-Depth-Anything repository...")
        rp.pip_import("git", "gitpython")
        import git
        git.Repo.clone_from(
            "https://github.com/DepthAnything/Video-Depth-Anything.git",
            repo_path,
            depth=1,
        )

    # Add to path if not already there
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    return repo_path


@rp.memoized
def _get_model_helper(model_path: str, encoder: str, metric: bool, device_str: str):
    """
    Load and cache a Video-Depth-Anything model.
    Results are memoized for efficiency.

    Args:
        model_path: Path to model checkpoint
        encoder: Encoder type
        metric: Whether this is a metric depth model
        device_str: Device string (e.g., 'cuda:0', 'cpu') for cache key

    Returns:
        Loaded model in eval mode
    """
    _ensure_dependencies()
    import torch
    _clone_video_depth_anything_repo()

    from video_depth_anything.video_depth import VideoDepthAnything

    config = MODEL_CONFIGS[encoder]
    model = VideoDepthAnything(
        encoder=config["encoder"],
        features=config["features"],
        out_channels=config["out_channels"],
        metric=metric,
    )

    # Load checkpoint
    if not os.path.exists(model_path):
        # Download if not present
        download_model(encoder=encoder, metric=metric, model_dir=os.path.dirname(model_path))

    device = torch.device(device_str)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    return model


def _get_model(
    encoder: str = "vitl",
    metric: bool = False,
    model_path: Optional[str] = None,
    device = None,
):
    """
    Get a Video-Depth-Anything model, downloading if necessary.

    Args:
        encoder: Model encoder type ('vits', 'vitb', or 'vitl')
        metric: Whether to use metric depth model
        model_path: Override model path
        device: Device to load model on

    Returns:
        Loaded model
    """
    _ensure_dependencies()
    import torch

    if model_path is None:
        model_path = _get_model_path(encoder, metric)

    if device is None:
        device = _default_device()
    else:
        global _video_depth_device
        _video_depth_device = device

    # Convert to torch.device and then to string for memoization key
    device = torch.device(device)
    device_str = str(device)

    return _get_model_helper(model_path, encoder, metric, device_str)


def _load_video(video, max_frames: Optional[int] = None, target_fps: Optional[int] = None):
    """
    Load and preprocess a video for Video-Depth-Anything.

    Args:
        video: Video path, URL, or array of frames (THWC)
        max_frames: Maximum number of frames to process
        target_fps: Target FPS to sample at

    Returns:
        np.ndarray: Video frames as THWC uint8 array
    """
    import numpy as np

    # Load from path/URL if needed
    if isinstance(video, str):
        video = rp.load_video(video)

    # Convert to numpy array
    video = rp.as_numpy_images(video)
    video = rp.as_rgb_images(video)
    video = rp.as_byte_images(video)

    # Stack if list
    if isinstance(video, list):
        video = np.stack(video, axis=0)

    # Subsample frames if needed
    if max_frames is not None and len(video) > max_frames:
        video = rp.resize_list(list(video), max_frames)
        video = np.stack(video, axis=0)

    return video


def _load_image(image):
    """
    Load and preprocess a single image.

    Args:
        image: Image path, URL, np.ndarray, PIL Image, or torch tensor

    Returns:
        np.ndarray: Image as HWC uint8 array
    """
    if isinstance(image, str):
        image = rp.load_image(image)

    image = rp.as_numpy_image(image)
    image = rp.as_rgb_image(image)
    image = rp.as_byte_image(image)

    return image


def estimate_video_depth(
    video,
    encoder: str = "vitl",
    metric: bool = False,
    device = None,
    model_path: Optional[str] = None,
    input_size: int = 518,
    max_res: int = 1280,
    max_frames: Optional[int] = None,
    target_fps: Optional[int] = None,
    fp32: bool = False,
    show_progress: bool = True,
):
    """
    Estimate depth for a video with temporal consistency.

    Args:
        video: Video as path, URL, THWC array, or list of frames
        encoder: Model size - 'vits' (small), 'vitb' (base), or 'vitl' (large)
        metric: Use metric depth model (outputs real-world meters)
        device: Device to run inference on. Can be:
            - None: Auto-select best available GPU
            - int: CUDA device index (e.g., 0, 1, 2)
            - str: Device string (e.g., 'cuda:0', 'cpu')
            - torch.device: PyTorch device object
        model_path: Override default model path
        input_size: Processing resolution (default 518)
        max_res: Maximum resolution (default 1280)
        max_frames: Maximum frames to process (None = all)
        target_fps: Target FPS for sampling (None = original)
        fp32: Use float32 instead of float16
        show_progress: Show progress bar during inference

    Returns:
        np.ndarray: Depth maps with shape (T, H, W), float32
            - Relative depth: higher values = farther from camera
            - Metric depth: values in meters
    """
    _ensure_dependencies()
    import numpy as np
    import torch
    import einops

    # Handle device specification
    if isinstance(device, int):
        device = f"cuda:{device}"

    # Validate encoder
    assert encoder in MODEL_CONFIGS, f"Invalid encoder: {encoder}. Choose from: {list(MODEL_CONFIGS.keys())}"

    # Load video
    frames = _load_video(video, max_frames=max_frames, target_fps=target_fps)

    # Validate shapes using rp
    dims = rp.validate_tensor_shapes(
        frames="T H W C",
        C=3,
    )

    if show_progress:
        print(f"Processing {dims.T} frames at {dims.H}x{dims.W}...")

    # Get model
    model = _get_model(encoder=encoder, metric=metric, model_path=model_path, device=device)

    # Get actual device from model - use full device string for correct GPU targeting
    actual_device = next(model.parameters()).device
    device_str = str(actual_device)

    # Run inference using the model's infer_video_depth method
    # This handles all the temporal consistency logic internally
    _clone_video_depth_anything_repo()

    depths, _ = model.infer_video_depth(
        frames,
        target_fps=-1,  # Already handled in _load_video
        input_size=input_size,
        device=device_str,  # Use full device string (e.g., 'cuda:0') not just 'cuda'
        fp32=fp32,
    )

    # Validate output shape
    rp.validate_tensor_shapes(
        depths="T H W",
        T=dims.T,
    )

    return depths.astype(np.float32)


def estimate_image_depth(
    image,
    encoder: str = "vitl",
    metric: bool = False,
    device = None,
    model_path: Optional[str] = None,
    input_size: int = 518,
    fp32: bool = False,
):
    """
    Estimate depth for a single image.

    This is a convenience wrapper that processes a single image as a 1-frame video.
    For batch processing of independent images, consider using estimate_video_depth
    with a list of frames (though temporal consistency won't apply).

    Args:
        image: Image as path, URL, HWC array, PIL Image, or torch tensor
        encoder: Model size - 'vits' (small), 'vitb' (base), or 'vitl' (large)
        metric: Use metric depth model (outputs real-world meters)
        device: Device to run inference on (see estimate_video_depth for options)
        model_path: Override default model path
        input_size: Processing resolution (default 518)
        fp32: Use float32 instead of float16

    Returns:
        np.ndarray: Depth map with shape (H, W), float32
    """
    import numpy as np

    # Load single image
    frame = _load_image(image)

    # Create 1-frame "video"
    frames = frame[np.newaxis, ...]

    # Process
    depths = estimate_video_depth(
        frames,
        encoder=encoder,
        metric=metric,
        device=device,
        model_path=model_path,
        input_size=input_size,
        fp32=fp32,
        show_progress=False,
    )

    # Return single depth map
    return depths[0]


def visualize_depth(
    depth,
    colormap: str = "inferno",
    normalize: bool = True,
):
    """
    Visualize a depth map as a colored image.

    Args:
        depth: Depth map with shape (H, W) or (T, H, W)
        colormap: Matplotlib colormap name (default: 'inferno')
        normalize: Normalize depth to 0-1 range (default: True)

    Returns:
        np.ndarray: Colored visualization as uint8 HWC or THWC array
    """
    import numpy as np
    rp.pip_import("matplotlib")
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)

    is_video = depth.ndim == 3
    if not is_video:
        depth = depth[np.newaxis, ...]

    if normalize:
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

    # Apply colormap
    depth_uint8 = (depth * 255).astype(np.uint8)
    colormap_lut = (np.array(cmap.colors) * 255).astype(np.uint8)
    colored = colormap_lut[depth_uint8]

    if not is_video:
        return colored[0]
    return colored


def demo():
    """
    Run a demo of Video-Depth-Anything in a Jupyter notebook.
    Shows depth estimation on a sample video.
    """
    import rp

    # Update CommonSource
    rp.git_import('CommonSource', pull=True)
    import rp.git.CommonSource.video_depth_anything as vda

    # Get a sample video
    video_url = 'https://videos.pexels.com/video-files/6507082/6507082-hd_1920_1080_25fps.mp4'
    video = rp.load_video(video_url, use_cache=True)
    video = rp.resize_video_to_fit(video, height=480, width=640, allow_growth=False)
    video = rp.resize_list_to_fit(video, 60)  # Keep up to 60 frames

    print("Estimating depth...")
    depths = vda.estimate_video_depth(video, encoder='vitl')

    # Visualize
    depth_vis = vda.visualize_depth(depths)

    # Create side-by-side comparison
    comparison = rp.tiled_videos(
        rp.labeled_videos(
            [video, depth_vis],
            ["Input Video", "Estimated Depth"],
            font="R:Futura",
            show_progress=True,
        ),
        length=2,
        show_progress=True,
    )

    comparison = rp.labeled_images(
        comparison,
        "Video-Depth-Anything Demo",
        size=30,
        font="R:Futura",
        text_color="yellow",
        show_progress=True,
    )

    rp.display_video(comparison, framerate=15)

    print(f"Depth range: {depths.min():.3f} to {depths.max():.3f}")
    print(f"Depth shape: {depths.shape}")


if __name__ == "__main__":
    import fire
    fire.Fire({
        "estimate_video_depth": estimate_video_depth,
        "estimate_image_depth": estimate_image_depth,
        "download_model": download_model,
        "get_available_models": get_available_models,
        "demo": demo,
    })
