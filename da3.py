# Added December 3, 2025 by Clara Burgert using Claude

"""
Depth Anything 3: State-of-the-art monocular and multi-view depth estimation (late 2025)

This module provides simple functions to work with ByteDance's Depth Anything 3 model,
which handles both images and videos for:
- Monocular depth estimation (single image -> depth map)
- Multi-view depth estimation (multiple images -> consistent depths)
- Metric depth estimation (real-world scale depth)
- Camera pose estimation

Functions automatically download the model on first use. No class initialization needed.

Input formats:
- Images: np.ndarray (HW3 uint8 or float 0-1), PIL Image, or path/URL
- Videos: List of frames, path, or URL

Output formats:
- Depth maps: HW float32 np.ndarray (depth values in arbitrary or metric units)
- Confidence maps: HW float32 np.ndarray (0-1 confidence)

Example:
    # Get depth map from a single image
    depth = estimate_depth("photo.jpg")

    # Get metric depth (real-world scale)
    depth = estimate_depth_metric("photo.jpg")

    # Get depth maps from video frames
    depths = estimate_depth_video("video.mp4")

    # Get multi-view consistent depths with camera poses
    result = estimate_depth_multiview(["img1.jpg", "img2.jpg", "img3.jpg"])

See: https://huggingface.co/depth-anything
     https://github.com/ByteDance-Seed/Depth-Anything-3
"""
import rp

# Use pip_import for lazy dependency loading (auto_yes=True for non-interactive use)
np = rp.pip_import("numpy", auto_yes=True)


__all__ = [
    "estimate_depth",
    "estimate_depth_metric",
    "estimate_depth_video",
    "estimate_depth_multiview",
    "download_model",
    "get_model_path",
]


# Model variants and their HuggingFace paths
MODEL_VARIANTS = {
    # Main series - unified depth-ray representation
    "base": "depth-anything/DA3-BASE",
    "small": "depth-anything/DA3-SMALL",
    "large": "depth-anything/DA3-LARGE",
    "giant": "depth-anything/DA3-GIANT",
    # Specialized models
    "mono-large": "depth-anything/DA3MONO-LARGE",  # Best for monocular relative depth
    "metric-large": "depth-anything/DA3METRIC-LARGE",  # Metric scale depth
    "nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",  # Best overall (combines capabilities)
}

# Default model for different tasks
DEFAULT_MONO_MODEL = "mono-large"  # Best for single image relative depth
DEFAULT_METRIC_MODEL = "metric-large"  # Best for metric depth
DEFAULT_MULTIVIEW_MODEL = "nested-giant-large"  # Best for multi-view

# Default paths for local model storage
DEFAULT_NETWORK_MODEL_PATH = "/root/models/depth_anything_3"
DEFAULT_LOCAL_MODEL_PATH = "/models/depth_anything_3"

# Global device state
_da3_device = None


def _default_da3_device():
    """
    Get or initialize the default device for Depth Anything 3 model.
    Uses rp.select_torch_device() to pick the best available device.

    Returns:
        The default torch device for DA3
    """
    global _da3_device
    if _da3_device is None:
        _da3_device = rp.select_torch_device(reserve=True)
    return _da3_device


def _normalize_device(device):
    """Normalize device specification to torch.device."""
    torch = rp.pip_import("torch", auto_yes=True)
    global _da3_device

    if device is None:
        device = _default_da3_device()
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        if device.lower() == "cpu":
            device = torch.device("cpu")
        elif device.isdigit():
            device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

    _da3_device = device
    return device


def _get_variant_path(variant):
    """Get HuggingFace path for a model variant."""
    if variant in MODEL_VARIANTS:
        return MODEL_VARIANTS[variant]
    # If it's already a full path, return as-is
    if "/" in variant:
        return variant
    raise ValueError(f"Unknown model variant: {variant}. Available: {list(MODEL_VARIANTS.keys())}")


def get_model_path(variant="mono-large", prefer_local=True):
    """
    Get the local path for a DA3 model variant if it exists.

    Args:
        variant: Model variant name or full HuggingFace path
        prefer_local: If True, prefer local path over network path

    Returns:
        str: Path to model directory or None if needs download
    """
    import os

    # Normalize variant name
    if variant in MODEL_VARIANTS:
        folder_name = variant.replace("-", "_")
    else:
        folder_name = variant.split("/")[-1].lower().replace("-", "_")

    paths_to_check = []
    if prefer_local:
        paths_to_check = [
            os.path.join(DEFAULT_LOCAL_MODEL_PATH, folder_name),
            os.path.join(DEFAULT_NETWORK_MODEL_PATH, folder_name),
        ]
    else:
        paths_to_check = [
            os.path.join(DEFAULT_NETWORK_MODEL_PATH, folder_name),
            os.path.join(DEFAULT_LOCAL_MODEL_PATH, folder_name),
        ]

    for path in paths_to_check:
        if os.path.isdir(path):
            # Check for model files
            if any(f.endswith(('.safetensors', '.bin', '.pt')) for f in os.listdir(path)):
                return path

    return None


def download_model(variant="mono-large", path=None, force=False):
    """
    Download a DA3 model variant to specified path.

    Args:
        variant: Model variant name (e.g., "mono-large", "metric-large", "base")
        path: Directory to save model. Defaults to DEFAULT_NETWORK_MODEL_PATH
        force: If True, re-download even if model exists

    Returns:
        str: Path where model was saved
    """
    import os

    rp.pip_import("huggingface_hub")
    from huggingface_hub import snapshot_download

    hf_path = _get_variant_path(variant)

    if path is None:
        folder_name = variant.replace("-", "_") if variant in MODEL_VARIANTS else variant.split("/")[-1].lower().replace("-", "_")
        path = os.path.join(DEFAULT_NETWORK_MODEL_PATH, folder_name)

    if os.path.exists(path) and not force:
        print(f"Model already exists at {path}. Use force=True to re-download.")
        return path

    os.makedirs(path, exist_ok=True)

    print(f"Downloading Depth Anything 3 model '{variant}' to {path}...")
    downloaded = snapshot_download(
        repo_id=hf_path,
        local_dir=path,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to {downloaded}")
    return downloaded


def _ensure_da3_package():
    """Ensure depth_anything_3 package is installed."""
    import sys
    import os

    # Try importing first
    try:
        import depth_anything_3
        return
    except ImportError:
        pass

    # Need to install from GitHub
    print("Installing depth_anything_3 package and dependencies...")

    # Install core dependencies first (auto_yes=True for non-interactive)
    rp.pip_import("torch", auto_yes=True)
    rp.pip_import("einops", auto_yes=True)
    rp.pip_import("timm", auto_yes=True)
    rp.pip_import("safetensors", auto_yes=True)
    rp.pip_import("huggingface_hub", auto_yes=True)
    rp.pip_import("omegaconf", auto_yes=True)
    rp.pip_import("addict", auto_yes=True)
    rp.pip_import("cv2", "opencv-python", auto_yes=True)
    rp.pip_import("scipy", auto_yes=True)
    rp.pip_import("matplotlib", auto_yes=True)
    rp.pip_import("imageio", auto_yes=True)
    rp.pip_import("plyfile", auto_yes=True)
    rp.pip_import("trimesh", auto_yes=True)

    # moviepy 2.x removed moviepy.editor, depth_anything_3 needs 1.0.3
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "moviepy==1.0.3", "-q"],
                   capture_output=True)

    # Try to import xformers (optional, for faster attention)
    try:
        rp.pip_import("xformers", auto_yes=True)
    except:
        print("xformers not available, will use standard attention")

    # Install the package from GitHub without dependencies (we've handled them above)
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps",
         "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to install depth_anything_3: {result.stderr}")

    # Install remaining dependencies that may be needed
    try:
        rp.pip_import("evo", auto_yes=True)  # For camera pose evaluation
        rp.pip_import("pycolmap", auto_yes=True)  # For COLMAP export
    except:
        pass  # These are optional

    # Clear import cache
    import importlib
    importlib.invalidate_caches()


@rp.memoized
def _get_da3_model_helper(model_path_or_id, device_str):
    """
    Helper function to load a DA3 model.
    Results are memoized for efficiency.

    Args:
        model_path_or_id: Local path or HuggingFace model ID
        device_str: Device string for loading

    Returns:
        model: The loaded DA3 model
    """
    torch = rp.pip_import("torch", auto_yes=True)

    _ensure_da3_package()

    from depth_anything_3.api import DepthAnything3

    print(f"Loading Depth Anything 3 model from {model_path_or_id}...")
    print(f"Using device: {device_str}")

    model = DepthAnything3.from_pretrained(model_path_or_id)
    model = model.to(device=torch.device(device_str))

    return model


def _get_da3_model(variant=None, model_path=None, device=None):
    """
    Get a DA3 model, downloading if needed.

    Args:
        variant: Model variant name (e.g., "mono-large", "metric-large")
        model_path: Explicit local path to model
        device: Device to load model on

    Returns:
        model: The loaded DA3 model
    """
    device = _normalize_device(device)
    device_str = str(device)

    # Determine what to load
    if model_path is not None:
        # Use explicit path
        load_path = model_path
    elif variant is not None:
        # Try local path first
        local_path = get_model_path(variant)
        if local_path is not None:
            load_path = local_path
        else:
            # Use HuggingFace path (will download automatically)
            load_path = _get_variant_path(variant)
    else:
        # Use default
        local_path = get_model_path(DEFAULT_MONO_MODEL)
        if local_path is not None:
            load_path = local_path
        else:
            load_path = MODEL_VARIANTS[DEFAULT_MONO_MODEL]

    return _get_da3_model_helper(load_path, device_str)


def _load_image(image):
    """
    Load and preprocess an image for DA3 model input.

    Args:
        image: An image as path, URL, np.ndarray, or PIL.Image

    Returns:
        PIL.Image object ready for the model
    """
    if isinstance(image, str):
        image = rp.load_image(image)

    image = rp.as_numpy_image(image)
    image = rp.as_rgb_image(image)
    image = rp.as_byte_image(image)
    image = rp.as_pil_image(image)

    return image


def _load_images(images):
    """Load multiple images."""
    return [_load_image(img) for img in images]


def _load_video(video, num_frames=None):
    """
    Load and preprocess a video for DA3 model input.

    Args:
        video: A video as path, URL, or list of frames
        num_frames: Number of frames to sample from the video

    Returns:
        List of PIL.Image objects ready for the model
    """
    if isinstance(video, str):
        video = rp.load_video(video)

    if num_frames is not None:
        video = rp.resize_list(video, num_frames)

    video = rp.as_numpy_images(video)
    video = rp.as_rgb_images(video)
    video = rp.as_byte_images(video)
    video = rp.as_pil_images(video)

    return video


def estimate_depth(image, *, device=None, model_path=None, variant=None, normalize=True):
    """
    Estimate relative depth from a single image.

    This uses the monocular model which predicts relative depth values.
    For real-world scale depth, use estimate_depth_metric().

    Args:
        image: np.ndarray, PIL Image, or path/URL
        device: Optional device to run inference on (str like "cuda:0", int like 0, or torch.device)
        model_path: Optional explicit path to model weights
        variant: Model variant to use (default: "mono-large")
        normalize: If True, normalize depth to 0-1 range

    Returns:
        depth: HW float32 np.ndarray with relative depth values
    """
    torch = rp.pip_import("torch", auto_yes=True)

    if variant is None:
        variant = DEFAULT_MONO_MODEL

    model = _get_da3_model(variant=variant, model_path=model_path, device=device)

    # Load and process image
    pil_image = _load_image(image)

    # Run inference
    with torch.no_grad():
        prediction = model.inference([pil_image])

    # Extract depth
    depth = prediction.depth[0]  # [H, W]

    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()

    depth = depth.astype(np.float32)

    if normalize:
        dmin, dmax = depth.min(), depth.max()
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin)

    return depth


def estimate_depth_metric(image, *, device=None, model_path=None, variant=None):
    """
    Estimate metric depth from a single image.

    This uses the metric model which predicts depth in real-world units (meters).

    Args:
        image: np.ndarray, PIL Image, or path/URL
        device: Optional device to run inference on
        model_path: Optional explicit path to model weights
        variant: Model variant to use (default: "metric-large")

    Returns:
        depth: HW float32 np.ndarray with depth values in meters
    """
    torch = rp.pip_import("torch", auto_yes=True)

    if variant is None:
        variant = DEFAULT_METRIC_MODEL

    model = _get_da3_model(variant=variant, model_path=model_path, device=device)

    pil_image = _load_image(image)

    with torch.no_grad():
        prediction = model.inference([pil_image])

    depth = prediction.depth[0]

    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()

    return depth.astype(np.float32)


def estimate_depth_video(video, *, device=None, model_path=None, variant=None,
                         num_frames=None, normalize=True):
    """
    Estimate depth maps for video frames.

    Processes each frame independently for relative depth.
    For temporally consistent multi-view depth, use estimate_depth_multiview().

    Args:
        video: List of frames, path, or URL
        device: Optional device to run inference on
        model_path: Optional explicit path to model weights
        variant: Model variant to use (default: "mono-large")
        num_frames: Number of frames to process (None = all frames)
        normalize: If True, normalize each depth map to 0-1 range

    Returns:
        depths: THW float32 np.ndarray with depth values for each frame
    """
    torch = rp.pip_import("torch", auto_yes=True)

    if variant is None:
        variant = DEFAULT_MONO_MODEL

    model = _get_da3_model(variant=variant, model_path=model_path, device=device)

    frames = _load_video(video, num_frames)

    depths = []
    for frame in frames:
        with torch.no_grad():
            prediction = model.inference([frame])

        depth = prediction.depth[0]
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()

        depth = depth.astype(np.float32)

        if normalize:
            dmin, dmax = depth.min(), depth.max()
            if dmax > dmin:
                depth = (depth - dmin) / (dmax - dmin)

        depths.append(depth)

    return np.stack(depths, axis=0)


def estimate_depth_multiview(images, *, device=None, model_path=None, variant=None,
                             return_confidence=False, return_poses=False):
    """
    Estimate spatially consistent depth from multiple views.

    This uses the full multi-view model which produces consistent geometry
    across all input images, along with camera pose estimates.

    Args:
        images: List of images (np.ndarray, PIL Image, or path/URL)
        device: Optional device to run inference on
        model_path: Optional explicit path to model weights
        variant: Model variant to use (default: "nested-giant-large")
        return_confidence: If True, also return confidence maps
        return_poses: If True, also return estimated camera poses

    Returns:
        If return_confidence=False and return_poses=False:
            depths: NHW float32 np.ndarray with depth values
        If return_confidence=True or return_poses=True:
            dict with keys:
                - "depth": NHW float32 np.ndarray
                - "confidence": NHW float32 np.ndarray (if return_confidence)
                - "extrinsics": Nx3x4 float32 np.ndarray (if return_poses)
                - "intrinsics": Nx3x3 float32 np.ndarray (if return_poses)
    """
    torch = rp.pip_import("torch", auto_yes=True)

    if variant is None:
        variant = DEFAULT_MULTIVIEW_MODEL

    model = _get_da3_model(variant=variant, model_path=model_path, device=device)

    pil_images = _load_images(images)

    with torch.no_grad():
        prediction = model.inference(pil_images)

    depth = prediction.depth  # [N, H, W]
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    depth = depth.astype(np.float32)

    if not return_confidence and not return_poses:
        return depth

    result = {"depth": depth}

    if return_confidence:
        conf = prediction.conf
        if isinstance(conf, torch.Tensor):
            conf = conf.cpu().numpy()
        result["confidence"] = conf.astype(np.float32)

    if return_poses:
        extrinsics = prediction.extrinsics
        intrinsics = prediction.intrinsics
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.cpu().numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()
        result["extrinsics"] = extrinsics.astype(np.float32)
        result["intrinsics"] = intrinsics.astype(np.float32)

    return result


def demo(image=None, output="/tmp/da3_demo.jpg", device=None, variant="small"):
    """
    Run a quick demo of Depth Anything 3's depth estimation.

    Args:
        image: Path/URL to image (default: COCO test image)
        output: Path to save visualization
        device: Device to use (e.g., "cuda:0", "cpu")
        variant: Model variant (small, base, large, mono-large, etc.)
    """
    if image is None:
        image = "http://images.cocodataset.org/val2017/000000039769.jpg"

    img = rp.load_image(image, use_cache=True)
    print(f"Input image shape: {img.shape}")

    print(f"Estimating depth with variant={variant}...")
    depth = estimate_depth(img, device=device, variant=variant)
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: {depth.min():.3f} to {depth.max():.3f}")

    depth_colored = rp.apply_colormap(depth, 'turbo')
    comparison = rp.horizontally_concatenated_images([
        rp.labeled_image(img, "Input Image"),
        rp.labeled_image(depth_colored, "Depth Map"),
    ])

    rp.save_image(comparison, output)
    print(f"Saved visualization to: {output}")

    try:
        rp.display_image(comparison)
    except:
        pass

    return depth


if __name__ == "__main__":
    import fire
    fire.Fire({name: globals()[name] for name in __all__ + ["demo"]})
