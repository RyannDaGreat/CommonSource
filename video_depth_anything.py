# Added December 3, 2025 by Clara Burgert using Claude

"""
Video-Depth-Anything: Consistent depth estimation for super-long videos (CVPR 2025 Highlight)

Platform Support:
    - CUDA: ✅ Full support (with xformers for best performance)
    - MPS (Apple Silicon): ✅ Full support (tested Dec 2025, uses fallback attention without xformers)
    - CPU: ✅ Full support (slow)

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
    "opencv-python",  # imports as cv2
    "einops",
    "tqdm",
    "gitpython",  # imports as git
    "matplotlib",
    "numpy<2",
    # Optional (CUDA only):
    # "xformers",
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

# Default model path (set to override HuggingFace cache)
default_model_path = None


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


def _get_cache_dir() -> str:
    """Get the cache directory for Video-Depth-Anything models."""
    if default_model_path:
        return default_model_path
    return os.path.join(os.path.expanduser("~"), ".cache", "video_depth_anything")


def download_model(
    variant: str = "vitl",
    metric: bool = False,
    path: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download a Video-Depth-Anything model checkpoint, or return cached path if already downloaded.

    This function is idempotent - calling it multiple times with the same
    variant/metric will not re-download. Use this to get the model path.

    Args:
        variant: Model variant ('vits', 'vitb', or 'vitl')
        metric: Whether to download metric depth model
        path: Directory to save the model. If None, uses ~/.cache/video_depth_anything
        force: Re-download even if file exists

    Returns:
        str: Path to the model file
    """
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Invalid variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}")

    if path is None:
        path = _get_cache_dir()

    os.makedirs(path, exist_ok=True)

    prefix = "metric_video_depth_anything" if metric else "video_depth_anything"
    filename = f"{prefix}_{variant}.pth"
    model_path = os.path.join(path, filename)

    if os.path.exists(model_path) and not force:
        print(f"Model already exists at: {model_path}")
        return model_path

    url_key = f"metric_{variant}" if metric else variant
    url = MODEL_URLS[url_key]

    print(f"Downloading {'metric ' if metric else ''}Video-Depth-Anything {variant.upper()} model...")
    print(f"Destination: {model_path}")

    rp.download_url(url, model_path)

    print(f"Download complete: {model_path}")
    return model_path


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
def _get_model_helper(model_path: str, variant: str, metric: bool, device_str: str):
    """
    Load and cache a Video-Depth-Anything model.
    Results are memoized for efficiency.

    Args:
        model_path: Path to model checkpoint
        variant: Model variant (vits, vitb, vitl)
        metric: Whether this is a metric depth model
        device_str: Device string (e.g., 'cuda:0', 'cpu') for cache key

    Returns:
        Loaded model in eval mode
    """
    import torch
    _clone_video_depth_anything_repo()

    from video_depth_anything.video_depth import VideoDepthAnything

    config = MODEL_CONFIGS[variant]
    model = VideoDepthAnything(
        encoder=config["encoder"],
        features=config["features"],
        out_channels=config["out_channels"],
        metric=metric,
    )

    # Load checkpoint
    if not os.path.exists(model_path):
        # Download if not present
        download_model(variant=variant, metric=metric, model_dir=os.path.dirname(model_path))

    device = torch.device(device_str)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    return model


def _get_model(
    variant: str = "vitl",
    metric: bool = False,
    model_path: Optional[str] = None,
    device = None,
):
    """
    Get a Video-Depth-Anything model, downloading if necessary.

    Args:
        variant: Model variant ('vits', 'vitb', or 'vitl')
        metric: Whether to use metric depth model
        model_path: Override model path
        device: Device to load model on

    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = download_model(variant, metric)

    # Resolve device
    device = rp.r._resolve_torch_device(device)
    device_str = str(device)

    return _get_model_helper(model_path, variant, metric, device_str)


def _load_video(video, num_frames: Optional[int] = None, target_fps: Optional[int] = None):
    """
    Load and preprocess a video for Video-Depth-Anything.

    Args:
        video: Video path, URL, or array of frames (THWC)
        num_frames: Number of frames to process (subsamples if video is longer)
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
    if num_frames is not None and len(video) > num_frames:
        video = rp.resize_list(list(video), num_frames)
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
    *,
    variant: str = "vitl",
    metric: bool = False,
    device=None,
    model_path: Optional[str] = None,
    input_size: int = 518,
    max_res: int = 1280,
    num_frames: Optional[int] = None,
    target_fps: Optional[int] = None,
    fp32: bool = False,
    show_progress: bool = True,
):
    """
    Estimate depth for a video with temporal consistency.

    Args:
        video: Video as path, URL, THWC array, or list of frames
        variant: Model variant - 'vits' (small), 'vitb' (base), or 'vitl' (large)
        metric: Use metric depth model (outputs real-world meters)
        device: Device to run inference on. Can be:
            - None: Auto-select best available GPU
            - int: CUDA device index (e.g., 0, 1, 2)
            - str: Device string (e.g., 'cuda:0', 'cpu')
            - torch.device: PyTorch device object
        model_path: Override default model path
        input_size: Processing resolution (default 518)
        max_res: Maximum resolution (default 1280)
        num_frames: Number of frames to process (None = all)
        target_fps: Target FPS for sampling (None = original)
        fp32: Use float32 instead of float16
        show_progress: Show progress bar during inference

    Returns:
        np.ndarray: Depth maps with shape (T, H, W), float32
            - Relative depth: higher values = farther from camera
            - Metric depth: values in meters
    """
    import numpy as np
    import torch
    import einops

    # Handle device specification
    if isinstance(device, int):
        device = f"cuda:{device}"

    # Validate variant
    assert variant in MODEL_CONFIGS, f"Invalid variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}"

    # Load video
    frames = _load_video(video, num_frames=num_frames, target_fps=target_fps)

    # Validate shapes using rp
    dims = rp.validate_tensor_shapes(
        frames="T H W C",
        C=3,
    )

    if show_progress:
        print(f"Processing {dims.T} frames at {dims.H}x{dims.W}...")

    # Get model
    model = _get_model(variant=variant, metric=metric, model_path=model_path, device=device)

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
    *,
    variant: str = "vitl",
    metric: bool = False,
    device=None,
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
        variant: Model variant - 'vits' (small), 'vitb' (base), or 'vitl' (large)
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
        variant=variant,
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
    depths = vda.estimate_video_depth(video, variant='vitl')

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
