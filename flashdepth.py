# Added December 3, 2025 by Clara Burgert using Claude

"""
FlashDepth: Real-time streaming video depth estimation at 2K resolution (ICCV 2025 Highlight)

Platform Support:
    - CUDA: ✅ Full support (requires flash_attn)
    - MPS (Apple Silicon): ❌ BROKEN - requires flash_attn which is CUDA-only (Dec 2025)
    - CPU: ❌ BROKEN - requires flash_attn which is CUDA-only

This module provides simple functions to work with Eyeline-Labs' FlashDepth models,
which perform real-time depth estimation on video streams with temporal consistency.

Features:
- Real-time depth estimation at 2K resolution (~24 FPS on A100)
- Temporally consistent results via Mamba temporal modeling
- Support for multiple model sizes: small (S), large (L), and full version
- Both single image and video inference

Functions automatically download models on first use. No class initialization needed.

Input formats:
- Videos: np.ndarray (THWC uint8 or float 0-1), list of frames, path, or URL
- Single images: np.ndarray (HWC), PIL Image, or path/URL

Output:
- Depth maps: np.ndarray with shape (T, H, W) for videos or (H, W) for single images
- Values are relative depth (higher = farther from camera)

Example:
    # Get depth for a video
    depths = estimate_video_depth("video.mp4")

    # Get depth for a single image
    depth = estimate_image_depth("photo.jpg")

    # Use specific model variant
    depths = estimate_video_depth("video.mp4", variant="large")

See: https://github.com/Eyeline-Labs/FlashDepth
"""
import rp
import torch
import numpy as np
from typing import Union, Optional, List
import os

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
    "timm",
    "gitpython",
    "huggingface_hub",
    "matplotlib",
    "numpy<2",
    # Optional for temporal consistency (requires matching CUDA):
    # "mamba-ssm",
    # "causal-conv1d",
]

# Model configurations
# FlashDepth uses vits for full/small variants, vitl for large variant
# Note: use_mamba can be set to False for easier installation (no Mamba compilation needed)
# but temporal consistency will be lost for videos
MODEL_CONFIGS = {
    "full": {
        "vit_size": "vits",
        "checkpoint": "iter_43002.pth",
        "hf_path": "flashdepth/iter_43002.pth",
        "description": "Full model - fastest at high resolution",
        "use_mamba": True,
        "mamba_in_dpt_layer": [1],  # Moved to after first DPT layer in repo
        "downsample_mamba": [0.1],
    },
    "large": {
        "vit_size": "vitl",
        "checkpoint": "iter_10001.pth",
        "hf_path": "flashdepth-l/iter_10001.pth",
        "description": "Large model - most accurate, recommended for low-res inputs",
        "use_mamba": True,
        "mamba_in_dpt_layer": [3],
        "downsample_mamba": [0.1],
    },
    "small": {
        "vit_size": "vits",
        "checkpoint": "iter_14001.pth",
        "hf_path": "flashdepth-s/iter_14001.pth",
        "description": "Small model - lightweight option",
        "use_mamba": True,
        "mamba_in_dpt_layer": [3],
        "downsample_mamba": [0.1],
    },
}

# Global flag to skip Mamba if it fails to install
_MAMBA_AVAILABLE = None

# HuggingFace repository
HF_REPO = "Eyeline-Labs/FlashDepth"

# Default paths - network drive first, can override with local path
DEFAULT_MODEL_DIR = "/root/models/flashdepth"
LOCAL_MODEL_DIR = "/models/flashdepth"

# Global device tracking
_flashdepth_device = None


def _resolve_device(device: Optional[Union[str, torch.device, int]]) -> torch.device:
    """
    Resolve device specification to a torch.device.

    Handles the complexity of GPU selection when CUDA_VISIBLE_DEVICES may be set.
    When user specifies an int (physical GPU index), we ensure that GPU is used
    regardless of CUDA_VISIBLE_DEVICES.

    Args:
        device: Device specification:
            - None: Auto-select best available GPU via rp.select_torch_device
            - int: Physical GPU index (e.g., 3 means GPU 3)
            - str: Device string (e.g., 'cuda:0', 'cpu')
            - torch.device: PyTorch device object

    Returns:
        torch.device: Resolved device
    """
    global _flashdepth_device

    if device is None:
        # Auto-select using rp
        if _flashdepth_device is None:
            _flashdepth_device = rp.select_torch_device(reserve=True)
        return torch.device(_flashdepth_device)

    if isinstance(device, int):
        # User specified physical GPU index
        # Check if CUDA_VISIBLE_DEVICES is restricting our view
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        if cuda_visible is not None:
            # CUDA_VISIBLE_DEVICES is set - need to map physical to logical
            visible_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]

            if device in visible_gpus:
                # The requested GPU is visible - find its logical index
                logical_idx = visible_gpus.index(device)
                resolved = torch.device(f"cuda:{logical_idx}")
            else:
                # The requested GPU is NOT in the visible list
                # Add it to CUDA_VISIBLE_DEVICES (modifies environment)
                new_visible = f"{cuda_visible},{device}"
                os.environ["CUDA_VISIBLE_DEVICES"] = new_visible
                # Logical index is at the end
                logical_idx = len(visible_gpus)
                resolved = torch.device(f"cuda:{logical_idx}")
                print(f"Note: Added GPU {device} to CUDA_VISIBLE_DEVICES (now: {new_visible})")
        else:
            # No restriction - direct mapping
            resolved = torch.device(f"cuda:{device}")

        _flashdepth_device = resolved
        return resolved

    if isinstance(device, str):
        _flashdepth_device = device
        return torch.device(device)

    if isinstance(device, torch.device):
        _flashdepth_device = device
        return device

    raise ValueError(f"Invalid device type: {type(device)}. Expected int, str, or torch.device.")


def _default_device():
    """
    Get or initialize the default device for FlashDepth.
    Uses rp.select_torch_device() to pick the best available device.
    """
    global _flashdepth_device
    if _flashdepth_device is None:
        _flashdepth_device = rp.select_torch_device(reserve=True)
    return _flashdepth_device


def get_available_models() -> dict:
    """
    Returns a dictionary describing all available model variants.

    Returns:
        dict: Model configurations with descriptions
    """
    return {
        "full": {
            "vit": "ViT-S",
            "description": "Fastest at 2K resolution (~24 FPS on A100)",
            "recommended_for": "High resolution videos (short side >= 518)",
        },
        "large": {
            "vit": "ViT-L",
            "description": "Most accurate, slower",
            "recommended_for": "Low resolution videos (short side < 518)",
        },
        "small": {
            "vit": "ViT-S",
            "description": "Lightweight, balanced speed/quality",
            "recommended_for": "General purpose use",
        },
    }


def _get_model_path(variant: str = "large", model_dir: Optional[str] = None) -> str:
    """
    Get the local path where a model checkpoint should be stored.

    Args:
        variant: Model variant ('full', 'large', or 'small')
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

    config = MODEL_CONFIGS[variant]
    return os.path.join(model_dir, variant, config["checkpoint"])


def download_model(
    variant: str = "large",
    model_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download a FlashDepth model checkpoint from HuggingFace.

    Args:
        variant: Model variant ('full', 'large', or 'small')
        model_dir: Directory to save the model (default: /root/models/flashdepth)
        force: Re-download even if file exists

    Returns:
        str: Path to the downloaded model file
    """
    assert variant in MODEL_CONFIGS, f"Invalid variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}"

    model_path = _get_model_path(variant, model_dir)
    model_dir_actual = os.path.dirname(model_path)

    # Create directory if needed
    os.makedirs(model_dir_actual, exist_ok=True)

    if os.path.exists(model_path) and not force:
        print(f"Model already exists at: {model_path}")
        return model_path

    # Get HuggingFace path
    config = MODEL_CONFIGS[variant]
    hf_path = config["hf_path"]

    print(f"Downloading FlashDepth {variant} model...")
    print(f"From: {HF_REPO}/{hf_path}")
    print(f"To: {model_path}")

    # Download from HuggingFace using huggingface_hub
    rp.pip_import("huggingface_hub")
    from huggingface_hub import hf_hub_download

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=hf_path,
        local_dir=model_dir if model_dir else (LOCAL_MODEL_DIR if os.path.isdir("/models") else DEFAULT_MODEL_DIR),
        local_dir_use_symlinks=False,
    )

    # Move to expected location if needed
    if downloaded_path != model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import shutil
        shutil.move(downloaded_path, model_path)

    print(f"Download complete: {model_path}")
    return model_path


def _ensure_dependencies():
    """Install required dependencies if not present."""
    rp.pip_import("torch")
    rp.pip_import("torchvision")
    rp.pip_import("cv2", "opencv-python")
    rp.pip_import("einops")
    rp.pip_import("tqdm")
    rp.pip_import("timm")


def _clone_flashdepth_repo():
    """Clone the FlashDepth repo if not already present."""
    import sys

    repo_path = "/tmp/FlashDepth"
    if not os.path.exists(repo_path):
        print("Cloning FlashDepth repository...")
        rp.pip_import("git", "gitpython")
        import git
        git.Repo.clone_from(
            "https://github.com/Eyeline-Labs/FlashDepth.git",
            repo_path,
            depth=1,
        )

    # Add to path if not already there
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    return repo_path


def _check_mamba_available():
    """Check if Mamba is available, caching the result."""
    global _MAMBA_AVAILABLE
    if _MAMBA_AVAILABLE is not None:
        return _MAMBA_AVAILABLE

    try:
        import mamba_ssm
        _MAMBA_AVAILABLE = True
        return True
    except ImportError:
        return False


def _setup_mamba(force_install: bool = False):
    """
    Set up the Mamba package from the FlashDepth repo.

    Args:
        force_install: If True, attempt installation even if previous attempts failed

    Returns:
        bool: True if Mamba is available, False otherwise
    """
    global _MAMBA_AVAILABLE

    # Check if already available
    if _check_mamba_available():
        return True

    # If we've already tried and failed, don't retry unless forced
    if _MAMBA_AVAILABLE is False and not force_install:
        return False

    repo_path = _clone_flashdepth_repo()
    mamba_path = os.path.join(repo_path, "mamba")

    # Try to install mamba from the repo using subprocess (NOT rp.pip_import which prompts)
    print("Installing Mamba from FlashDepth repo...")
    print("Note: This requires matching CUDA versions. If it fails, we'll run without Mamba.")
    import subprocess
    import sys

    env = os.environ.copy()
    env["MAMBA_FORCE_BUILD"] = "TRUE"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "."],
            cwd=mamba_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"Mamba installation from repo failed (CUDA version mismatch likely).")
            # Try alternative: install from PyPI with pre-built wheels
            print("Trying to install mamba-ssm from PyPI (may need matching CUDA)...")
            try:
                result2 = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "mamba-ssm", "causal-conv1d"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except Exception:
                pass
    except Exception as e:
        print(f"Mamba installation error: {e}")
        # Try PyPI fallback
        print("Trying to install mamba-ssm from PyPI...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "mamba-ssm", "causal-conv1d"],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception:
            pass

    # Check if it worked
    if _check_mamba_available():
        _MAMBA_AVAILABLE = True
        return True
    else:
        _MAMBA_AVAILABLE = False
        print("WARNING: Mamba not available. FlashDepth will run without temporal consistency.")
        print("         Single-image and video depth estimation will still work correctly.")
        print("         For temporal consistency, ensure CUDA version matches PyTorch CUDA version.")
        return False


@rp.memoized
def _get_model_helper(model_path: str, variant: str, device: torch.device):
    """
    Load and cache a FlashDepth model.
    Results are memoized for efficiency.

    Args:
        model_path: Path to model checkpoint
        variant: Model variant
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    _ensure_dependencies()
    repo_path = _clone_flashdepth_repo()

    # Try to set up Mamba, but continue without it if unavailable
    mamba_available = _setup_mamba()

    import sys
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from flashdepth.model import FlashDepth

    config = MODEL_CONFIGS[variant]

    # Determine if we should use mamba based on config and availability
    use_mamba = config["use_mamba"] and mamba_available

    if config["use_mamba"] and not mamba_available:
        print(f"Note: Running {variant} model without Mamba temporal consistency.")

    # Create model with appropriate config
    model = FlashDepth(
        vit_size=config["vit_size"],
        patch_size=14,
        use_mamba=use_mamba,
        downsample_mamba=config["downsample_mamba"] if use_mamba else [1.0],
        mamba_in_dpt_layer=config["mamba_in_dpt_layer"] if use_mamba else [],
        training=False,
        mamba_type="add",
        num_mamba_layers=4,
        mamba_d_conv=4,
        mamba_d_state=256,
    )

    # Load checkpoint
    if not os.path.exists(model_path):
        download_model(variant=variant, model_dir=os.path.dirname(os.path.dirname(model_path)))

    print(f"Loading checkpoint from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Remove DDP prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load with strict=False to allow missing mamba weights when not using mamba
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device).eval()

    return model


def _get_model(
    variant: str = "large",
    model_path: Optional[str] = None,
    device: Optional[Union[str, torch.device, int]] = None,
):
    """
    Get a FlashDepth model, downloading if necessary.

    Args:
        variant: Model variant ('full', 'large', or 'small')
        model_path: Override model path
        device: Device to load model on (int for physical GPU index, str, or torch.device)

    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = _get_model_path(variant)

    # Resolve device using our helper that handles CUDA_VISIBLE_DEVICES
    resolved_device = _resolve_device(device)

    return _get_model_helper(model_path, variant, resolved_device)


def _load_video(video, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Load and preprocess a video for FlashDepth.

    Args:
        video: Video path, URL, or array of frames (THWC)
        max_frames: Maximum number of frames to process

    Returns:
        np.ndarray: Video frames as THWC uint8 array
    """
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


def _load_image(image) -> np.ndarray:
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


def _preprocess_frames(frames: np.ndarray, target_size: int = 518) -> torch.Tensor:
    """
    Preprocess frames for FlashDepth inference.

    Args:
        frames: THWC uint8 numpy array
        target_size: Target resolution (default 518)

    Returns:
        torch.Tensor: Preprocessed frames as BTCHW tensor
    """
    rp.pip_import("torchvision")
    import torchvision.transforms as transforms
    from einops import rearrange

    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Process each frame
    processed = []
    for frame in frames:
        # Resize maintaining aspect ratio
        h, w = frame.shape[:2]
        if h < w:
            new_h = target_size
            new_w = int(w * target_size / h)
            new_w = (new_w // 14) * 14  # Must be multiple of patch_size
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
            new_h = (new_h // 14) * 14

        frame_resized = rp.resize_image(frame, (new_h, new_w))

        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = rearrange(frame_tensor, "h w c -> c h w")
        frame_tensor = normalize(frame_tensor)
        processed.append(frame_tensor)

    # Stack into batch: (T, C, H, W)
    video_tensor = torch.stack(processed, dim=0)

    # Add batch dimension: (1, T, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor


def estimate_video_depth(
    video,
    variant: str = "large",
    device: Optional[Union[str, torch.device, int]] = None,
    model_path: Optional[str] = None,
    input_size: int = 518,
    max_frames: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Estimate depth for a video with temporal consistency.

    Args:
        video: Video as path, URL, THWC array, or list of frames
        variant: Model variant - 'full' (fastest), 'large' (most accurate), or 'small'
        device: Device to run inference on. Can be:
            - None: Auto-select best available GPU via rp.select_torch_device
            - int: Physical GPU index (e.g., 3 means GPU 3, works regardless of CUDA_VISIBLE_DEVICES)
            - str: Device string (e.g., 'cuda:0', 'cpu') - uses logical indexing
            - torch.device: PyTorch device object
        model_path: Override default model path
        input_size: Processing resolution (default 518)
        max_frames: Maximum frames to process (None = all)
        show_progress: Show progress during inference

    Returns:
        np.ndarray: Depth maps with shape (T, H, W), float32
            Higher values = farther from camera
    """
    from einops import rearrange

    # Validate variant
    assert variant in MODEL_CONFIGS, f"Invalid variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}"

    # Load video
    frames = _load_video(video, max_frames=max_frames)

    # Validate shapes using rp
    dims = rp.validate_tensor_shapes(
        frames="T H W C",
        C=3,
    )

    if show_progress:
        print(f"Processing {dims.T} frames at {dims.H}x{dims.W}...")

    # Get model
    model = _get_model(variant=variant, model_path=model_path, device=device)

    # Get actual device from model
    actual_device = next(model.parameters()).device

    # Preprocess frames
    video_tensor = _preprocess_frames(frames, target_size=input_size)
    video_tensor = video_tensor.to(actual_device)

    # Validate input tensor shape
    rp.validate_tensor_shapes(
        video_tensor="torch: B T C H W",
        B=1,
        C=3,
    )

    # Run inference
    depths = []

    # Use autocast for GPU, skip for CPU
    use_autocast = actual_device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_autocast else torch.no_grad()

    with torch.no_grad(), autocast_ctx:
        # Initialize temporal state (only if Mamba is available and enabled)
        if hasattr(model, 'use_mamba') and model.use_mamba and hasattr(model, 'mamba'):
            model.mamba.start_new_sequence()

        iterator = range(video_tensor.shape[1])
        if show_progress:
            iterator = rp.eta(iterator, title="FlashDepth inference")

        for i in iterator:
            frame = video_tensor[:, i, :, :, :]
            B, C, H, W = frame.shape

            patch_h, patch_w = H // model.patch_size, W // model.patch_size

            # Extract features and predict depth
            dpt_features = model.get_dpt_features(frame, input_shape=(B, C, H, W))
            pred_depth = model.final_head(dpt_features, patch_h, patch_w)
            pred_depth = torch.clip(pred_depth, min=0)

            # Resize to original frame size
            pred_depth_resized = torch.nn.functional.interpolate(
                pred_depth.unsqueeze(1),
                size=(dims.H, dims.W),
                mode="bilinear",
                align_corners=True
            ).squeeze(1)

            depths.append(pred_depth_resized.cpu().float().numpy())

    # Stack into array
    depths = np.concatenate(depths, axis=0)

    # Validate output shape
    rp.validate_tensor_shapes(
        depths="T H W",
        T=dims.T,
    )

    return depths.astype(np.float32)


def estimate_image_depth(
    image,
    variant: str = "large",
    device: Optional[Union[str, torch.device, int]] = None,
    model_path: Optional[str] = None,
    input_size: int = 518,
) -> np.ndarray:
    """
    Estimate depth for a single image.

    This processes a single image without temporal modeling.
    For videos, use estimate_video_depth which maintains temporal consistency.

    Args:
        image: Image as path, URL, HWC array, PIL Image, or torch tensor
        variant: Model variant - 'full', 'large', or 'small'
        device: Device to run inference on (see estimate_video_depth for options)
        model_path: Override default model path
        input_size: Processing resolution (default 518)

    Returns:
        np.ndarray: Depth map with shape (H, W), float32
    """
    # Load single image
    frame = _load_image(image)

    # Create 1-frame "video"
    frames = frame[np.newaxis, ...]

    # Process (temporal modeling will have no effect with single frame)
    depths = estimate_video_depth(
        frames,
        variant=variant,
        device=device,
        model_path=model_path,
        input_size=input_size,
        show_progress=False,
    )

    # Return single depth map
    return depths[0]


def visualize_depth(
    depth: np.ndarray,
    colormap: str = "inferno",
    normalize: bool = True,
) -> np.ndarray:
    """
    Visualize a depth map as a colored image.

    Args:
        depth: Depth map with shape (H, W) or (T, H, W)
        colormap: Matplotlib colormap name (default: 'inferno')
        normalize: Normalize depth to 0-1 range (default: True)

    Returns:
        np.ndarray: Colored visualization as uint8 HWC or THWC array
    """
    rp.pip_import("matplotlib")
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)

    is_video = depth.ndim == 3
    if not is_video:
        depth = depth[np.newaxis, ...]

    if normalize:
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

    # Apply colormap frame by frame
    colored_frames = []
    for d in depth:
        colored = cmap(d)[:, :, :3]  # Remove alpha channel
        colored = (colored * 255).astype(np.uint8)
        colored_frames.append(colored)

    colored = np.stack(colored_frames, axis=0)

    if not is_video:
        return colored[0]
    return colored


def demo():
    """
    Run a demo of FlashDepth in a Jupyter notebook.
    Shows depth estimation on a sample video.
    """
    import rp

    # Update CommonSource
    rp.git_import('CommonSource', pull=True)
    import rp.git.CommonSource.flashdepth as fd

    # Get a sample video
    video_url = 'https://videos.pexels.com/video-files/6507082/6507082-hd_1920_1080_25fps.mp4'
    video = rp.load_video(video_url, use_cache=True)
    video = rp.resize_video_to_fit(video, height=480, width=640, allow_growth=False)
    video = rp.resize_list_to_fit(video, 30)  # Keep up to 30 frames

    print("Estimating depth with FlashDepth...")
    depths = fd.estimate_video_depth(video, variant='large')

    # Visualize
    depth_vis = fd.visualize_depth(depths)

    # Create side-by-side comparison
    comparison = rp.tiled_videos(
        rp.labeled_videos(
            [video, depth_vis],
            ["Input Video", "FlashDepth Output"],
            font="R:Futura",
            show_progress=True,
        ),
        length=2,
        show_progress=True,
    )

    comparison = rp.labeled_images(
        comparison,
        "FlashDepth Demo",
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
