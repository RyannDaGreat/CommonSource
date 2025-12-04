# Added December 3, 2025 by Clara Burgert using Claude

"""
Kandinsky 5.0 Video/Image Generation Module

A simple interface for Kandinsky 5.0 models following rp CommonSource conventions.
Supports text-to-video (T2V), image-to-video (I2V), text-to-image (T2I), and image-to-image (I2I) generation.

Model variants:
    - lite: Smaller model (~2B params, ~4.5GB) - faster, less VRAM
    - pro: Larger model (~19B params, ~43GB) - higher quality, more VRAM

Usage:
    import rp
    rp.git_import('CommonSource')
    from rp.git.CommonSource.kandinsky import text_to_video, image_to_video

    # Generate a video from text
    video = text_to_video("A cat walking on a sunny beach")
    rp.save_video_mp4(video, "cat_beach.mp4")

    # Generate a video from an image
    video = image_to_video("input.jpg", "The scene comes alive with motion")
    rp.save_video_mp4(video, "animated.mp4")

Model paths:
    Network storage (source): ./weights/ or custom path
    Local cache: /models/kandinsky5/ (for faster loading)
"""

import rp
import os
import warnings
import logging
from typing import Union, Optional, List

__all__ = [
    "text_to_video",
    "image_to_video",
    "text_to_image",
    "image_to_image",
    "get_t2v_pipeline",
    "get_i2v_pipeline",
    "get_t2i_pipeline",
    "get_i2i_pipeline",
    "download_models",
    "set_device",
    "get_device",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_LOCAL_CACHE",
]

PIP_REQUIREMENTS = [
    "torch==2.8.0",
    "torchvision",
    "transformers>=4.56.2",
    "diffusers",
    "accelerate",
    "einops",
    "numpy",
    "pillow",
    "omegaconf",
    "safetensors",
    "sentencepiece",
    "imageio",
    "imageio-ffmpeg",
    "opencv-python",
    "peft",
]

# Default paths
DEFAULT_MODEL_PATH = "/root/CleanCode/Github/kandinsky-5/weights"
DEFAULT_LOCAL_CACHE = "/models/kandinsky5"

# Global device variable (like SA2VA pattern)
_kandinsky_device = None


def _default_device():
    """Get the default device, selecting one if not already set.

    Handles CUDA_VISIBLE_DEVICES properly - if only one GPU is visible,
    use cuda:0 rather than trying to use a saved device index that may
    not exist in the restricted environment.
    """
    global _kandinsky_device
    if _kandinsky_device is None:
        # Ensure torch is installed before selecting device (auto_yes for non-interactive)
        rp.pip_import("torch", auto_yes=True)
        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus == 1:
                # Only one GPU visible (likely CUDA_VISIBLE_DEVICES set)
                # Use cuda:0 directly instead of rp.select_torch_device
                _kandinsky_device = "cuda:0"
            else:
                # Multiple GPUs available, let user select
                _kandinsky_device = rp.select_torch_device(reserve=True)
        else:
            _kandinsky_device = "cpu"
    return _kandinsky_device


def set_device(device):
    """Set the global device for Kandinsky models."""
    global _kandinsky_device
    _kandinsky_device = device


def get_device():
    """Get the current device for Kandinsky models."""
    return _default_device()


def _disable_warnings():
    """Disable verbose warnings from torch and transformers."""
    warnings.filterwarnings("ignore")
    rp.pip_import("torch")
    import torch
    logging.getLogger("torch").setLevel(logging.ERROR)
    try:
        torch._logging.set_logs(
            dynamo=logging.ERROR,
            dynamic=logging.ERROR,
            aot=logging.ERROR,
            inductor=logging.ERROR,
            guards=False,
            recompiles=False
        )
    except:
        pass


def _ensure_kandinsky_installed():
    """Ensure kandinsky package and dependencies are installed."""
    # Core dependencies - auto_yes=True allows non-interactive install
    rp.pip_import("torch", auto_yes=True)
    rp.pip_import("numpy", auto_yes=True)
    rp.pip_import("PIL", "pillow", auto_yes=True)
    rp.pip_import("omegaconf", auto_yes=True)
    rp.pip_import("safetensors", auto_yes=True)
    rp.pip_import("transformers", auto_yes=True)
    rp.pip_import("huggingface_hub", auto_yes=True)
    rp.pip_import("einops", auto_yes=True)
    rp.pip_import("flash_attn", auto_yes=True)

    # Check if kandinsky is available
    try:
        from kandinsky import get_T2V_pipeline
    except ImportError:
        # Try to install from the local repo
        kandinsky_path = "/root/CleanCode/Github/kandinsky-5"
        if os.path.exists(kandinsky_path):
            import sys
            if kandinsky_path not in sys.path:
                sys.path.insert(0, kandinsky_path)


def _load_image(image):
    """
    Load and normalize an image to PIL format.
    Accepts: file path, numpy array, PIL image, torch tensor
    Returns: PIL Image
    """
    if isinstance(image, str):
        image = rp.load_image(image)

    image = rp.as_numpy_image(image)
    image = rp.as_rgb_image(image)
    image = rp.as_byte_image(image)
    image = rp.as_pil_image(image)

    return image


def _video_to_numpy(video_frames):
    """Convert video output to numpy THWC format.

    Kandinsky pipelines return tensors in (B, C, T, H, W) format.
    We convert to (T, H, W, C) format for rp compatibility.
    """
    if video_frames is None:
        return None

    # Handle list of numpy arrays
    if isinstance(video_frames, list):
        import numpy as np
        return np.stack([rp.as_numpy_image(f) for f in video_frames])

    # Handle torch tensor - Kandinsky returns (B, C, T, H, W) format
    if rp.is_torch_tensor(video_frames):
        import einops
        # Check if 5D tensor (B, C, T, H, W)
        if video_frames.ndim == 5:
            # Validate shape - Kandinsky returns (B=1, C=3, T, H, W)
            rp.validate_tensor_shapes(
                video_frames='torch: B C T H W',
                B=1, C=3,  # Kandinsky always produces batch=1, RGB channels
            )
            # Rearrange from (B, C, T, H, W) to (T, H, W, C), taking first batch
            video_frames = einops.rearrange(video_frames[0], 'C T H W -> T H W C')
            video_frames = video_frames.cpu().numpy()
            # Convert to uint8 range [0, 255]
            if video_frames.max() <= 1.0:
                video_frames = (video_frames * 255).astype('uint8')
            else:
                video_frames = video_frames.astype('uint8')
            return video_frames
        # Standard 4D tensor (T, C, H, W) - validate and convert
        rp.validate_tensor_shapes(
            video_frames='torch: T C H W',
            C=3,  # RGB channels
        )
        return rp.as_numpy_video(video_frames)

    # Already numpy - validate it's THWC format
    if hasattr(video_frames, 'ndim') and video_frames.ndim == 4:
        rp.validate_tensor_shapes(
            video_frames='numpy: T H W C',
            C=3,  # RGB channels
        )
    return rp.as_numpy_array(video_frames)


def download_models(
    model_type: str = "lite",
    source_path: str = None,
    local_cache: str = None,
    include_t2v: bool = True,
    include_i2v: bool = True,
    include_t2i: bool = True,
    include_i2i: bool = True,
):
    """
    Download/cache Kandinsky models from network storage to local storage.

    Args:
        model_type: "lite" or "pro" - which model size to download
        source_path: Network storage path (default: DEFAULT_MODEL_PATH)
        local_cache: Local cache path (default: DEFAULT_LOCAL_CACHE)
        include_t2v: Download text-to-video model
        include_i2v: Download image-to-video model
        include_t2i: Download text-to-image model
        include_i2i: Download image-to-image model

    Returns:
        dict: Paths to downloaded model components
    """
    if source_path is None:
        source_path = DEFAULT_MODEL_PATH
    if local_cache is None:
        local_cache = DEFAULT_LOCAL_CACHE

    os.makedirs(local_cache, exist_ok=True)

    # Components to sync
    items = {
        "vae": "dir",
        "text_encoder": "dir",
        "text_encoder2": "dir",
    }

    # Add model checkpoints based on selection
    if model_type == "lite":
        if include_t2v:
            items["model/kandinsky5lite_t2v_sft_5s.safetensors"] = "file"
        if include_i2v:
            items["model/kandinsky5lite_i2v_sft_5s.safetensors"] = "file"
        if include_t2i:
            items["model/kandinsky5lite_t2i.safetensors"] = "file"
        if include_i2i:
            items["model/kandinsky5lite_i2i.safetensors"] = "file"
    elif model_type == "pro":
        if include_t2v:
            items["model/kandinsky5pro_t2v_sft_5s.safetensors"] = "file"

    # Sync each item
    for item, item_type in items.items():
        src = os.path.join(source_path, item)
        dst = os.path.join(local_cache, item)

        if item_type == "file":
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                print(f"Copying {item}...")
                import shutil
                shutil.copy2(src, dst)
        else:
            if not os.path.exists(dst):
                print(f"Copying {item}/...")
                import shutil
                shutil.copytree(src, dst)

    return {
        "model_path": local_cache,
        "vae_path": os.path.join(local_cache, "vae"),
        "text_encoder_path": os.path.join(local_cache, "text_encoder"),
        "text_encoder2_path": os.path.join(local_cache, "text_encoder2"),
    }


@rp.memoized
def _get_t2v_pipeline_helper(config_path: str, device: str, offload: bool):
    """Helper to load and cache T2V pipeline."""
    _ensure_kandinsky_installed()
    _disable_warnings()

    from kandinsky import get_T2V_pipeline

    device_map = {"dit": device, "vae": device, "text_embedder": device}

    return get_T2V_pipeline(
        device_map=device_map,
        conf_path=config_path,
        offload=offload,
    )


def get_t2v_pipeline(
    model_type: str = "lite",
    config_path: str = None,
    model_path: str = None,
    device: str = None,
    offload: bool = False,
):
    """
    Get a text-to-video pipeline.

    Args:
        model_type: "lite" or "pro"
        config_path: Path to config YAML (auto-generated if None)
        model_path: Path to model weights
        device: Device to use (auto-selected if None)
        offload: Whether to offload models to CPU to save VRAM

    Returns:
        Kandinsky5T2VPipeline
    """
    if device is None:
        device = str(_default_device())

    if config_path is None:
        # Generate config path based on model type
        if model_path is None:
            model_path = DEFAULT_LOCAL_CACHE
            if not os.path.exists(os.path.join(model_path, "model")):
                model_path = DEFAULT_MODEL_PATH

        # Check for local config
        config_dir = "/root/CleanCode/Github/kandinsky-5/configs_local"
        config_name = f"k5_{model_type}_t2v_5s_sft_sd_local.yaml"
        config_path = os.path.join(config_dir, config_name)

        if not os.path.exists(config_path):
            # Fall back to standard config
            config_dir = "/root/CleanCode/Github/kandinsky-5/configs"
            config_name = f"k5_{model_type}_t2v_5s_sft_sd.yaml"
            config_path = os.path.join(config_dir, config_name)

    return _get_t2v_pipeline_helper(config_path, device, offload)


@rp.memoized
def _get_i2v_pipeline_helper(config_path: str, device: str, offload: bool):
    """Helper to load and cache I2V pipeline."""
    _ensure_kandinsky_installed()
    _disable_warnings()

    from kandinsky import get_I2V_pipeline

    device_map = {"dit": device, "vae": device, "text_embedder": device}

    return get_I2V_pipeline(
        device_map=device_map,
        conf_path=config_path,
        offload=offload,
    )


def get_i2v_pipeline(
    model_type: str = "lite",
    config_path: str = None,
    model_path: str = None,
    device: str = None,
    offload: bool = False,
):
    """
    Get an image-to-video pipeline.

    Args:
        model_type: "lite" (pro not available for i2v)
        config_path: Path to config YAML
        model_path: Path to model weights
        device: Device to use (auto-selected if None)
        offload: Whether to offload models to CPU

    Returns:
        Kandinsky5I2VPipeline
    """
    if device is None:
        device = str(_default_device())

    if config_path is None:
        config_dir = "/root/CleanCode/Github/kandinsky-5/configs_local"
        config_name = f"k5_{model_type}_i2v_5s_sft_sd_local.yaml"
        config_path = os.path.join(config_dir, config_name)

        if not os.path.exists(config_path):
            config_dir = "/root/CleanCode/Github/kandinsky-5/configs"
            config_name = f"k5_{model_type}_i2v_5s_sft_sd.yaml"
            config_path = os.path.join(config_dir, config_name)

    return _get_i2v_pipeline_helper(config_path, device, offload)


@rp.memoized
def _get_t2i_pipeline_helper(config_path: str, device: str, offload: bool):
    """Helper to load and cache T2I pipeline."""
    _ensure_kandinsky_installed()
    _disable_warnings()

    from kandinsky import get_T2I_pipeline

    device_map = {"dit": device, "vae": device, "text_embedder": device}

    return get_T2I_pipeline(
        device_map=device_map,
        conf_path=config_path,
        offload=offload,
    )


def get_t2i_pipeline(
    config_path: str = None,
    model_path: str = None,
    device: str = None,
    offload: bool = False,
):
    """
    Get a text-to-image pipeline.

    Args:
        config_path: Path to config YAML
        model_path: Path to model weights
        device: Device to use (auto-selected if None)
        offload: Whether to offload models to CPU

    Returns:
        Kandinsky5T2IPipeline
    """
    if device is None:
        device = str(_default_device())

    if config_path is None:
        config_dir = "/root/CleanCode/Github/kandinsky-5/configs_local"
        config_name = "k5_lite_t2i_sft_hd_local.yaml"
        config_path = os.path.join(config_dir, config_name)

        if not os.path.exists(config_path):
            config_dir = "/root/CleanCode/Github/kandinsky-5/configs"
            config_name = "k5_lite_t2i_sft_hd.yaml"
            config_path = os.path.join(config_dir, config_name)

    return _get_t2i_pipeline_helper(config_path, device, offload)


@rp.memoized
def _get_i2i_pipeline_helper(config_path: str, device: str, offload: bool):
    """Helper to load and cache I2I pipeline."""
    _ensure_kandinsky_installed()
    _disable_warnings()

    from kandinsky import get_I2I_pipeline

    device_map = {"dit": device, "vae": device, "text_embedder": device}

    return get_I2I_pipeline(
        device_map=device_map,
        conf_path=config_path,
        offload=offload,
    )


def get_i2i_pipeline(
    config_path: str = None,
    model_path: str = None,
    device: str = None,
    offload: bool = False,
):
    """
    Get an image-to-image pipeline.

    Args:
        config_path: Path to config YAML
        model_path: Path to model weights
        device: Device to use (auto-selected if None)
        offload: Whether to offload models to CPU

    Returns:
        Kandinsky5I2IPipeline
    """
    if device is None:
        device = str(_default_device())

    if config_path is None:
        config_dir = "/root/CleanCode/Github/kandinsky-5/configs_local"
        config_name = "k5_lite_i2i_sft_hd_local.yaml"
        config_path = os.path.join(config_dir, config_name)

        if not os.path.exists(config_path):
            config_dir = "/root/CleanCode/Github/kandinsky-5/configs"
            config_name = "k5_lite_i2i_sft_hd.yaml"
            config_path = os.path.join(config_dir, config_name)

    return _get_i2i_pipeline_helper(config_path, device, offload)


def text_to_video(
    prompt: str,
    model_type: str = "lite",
    width: int = 768,
    height: int = 512,
    duration: int = 5,
    num_steps: int = None,
    guidance_weight: float = None,
    seed: int = None,
    device: str = None,
    offload: bool = False,
    negative_prompt: str = None,
    expand_prompt: bool = True,
    save_path: str = None,
) -> "np.ndarray":
    """
    Generate a video from a text prompt.

    Args:
        prompt: Text description of the video to generate
        model_type: "lite" or "pro"
        width: Video width in pixels (512, 640, 768, 896, 1024, 1152, 1280)
        height: Video height in pixels (512, 640, 768, 896, 1024, 1152, 1280)
        duration: Video duration in seconds (default 5)
        num_steps: Number of diffusion steps (default from config, ~50)
        guidance_weight: CFG weight (default from config, ~5.0)
        seed: Random seed for reproducibility
        device: Device to use (auto-selected if None)
        offload: Offload models to CPU to save VRAM
        negative_prompt: What to avoid in the video
        expand_prompt: Whether to use prompt expansion
        save_path: If provided, save the video to this path

    Returns:
        numpy.ndarray: Video frames in THWC format (time, height, width, channels)

    Example:
        >>> video = text_to_video("A cat walking on a sunny beach")
        >>> rp.save_video_mp4(video, "cat.mp4", framerate=24)
    """
    _ensure_kandinsky_installed()
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    pipe = get_t2v_pipeline(
        model_type=model_type,
        device=device,
        offload=offload,
    )

    kwargs = {
        "time_length": duration,
        "width": width,
        "height": height,
        "expand_prompts": 1 if expand_prompt else 0,
    }

    if num_steps is not None:
        kwargs["num_steps"] = num_steps
    if guidance_weight is not None:
        kwargs["guidance_weight"] = guidance_weight
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if save_path is not None:
        kwargs["save_path"] = save_path
    if seed is not None:
        kwargs["seed"] = seed

    result = pipe(prompt, **kwargs)

    return _video_to_numpy(result)


def image_to_video(
    image,
    prompt: str,
    model_type: str = "lite",
    duration: int = 5,
    num_steps: int = None,
    guidance_weight: float = None,
    seed: int = None,
    device: str = None,
    offload: bool = False,
    expand_prompt: bool = True,
    save_path: str = None,
) -> "np.ndarray":
    """
    Generate a video from an image and text prompt.

    Args:
        image: Input image (path, numpy array, PIL image, or torch tensor)
        prompt: Text description of how to animate the image
        model_type: "lite" (pro not available for i2v)
        duration: Video duration in seconds (default 5)
        num_steps: Number of diffusion steps
        guidance_weight: CFG weight
        seed: Random seed for reproducibility
        device: Device to use
        offload: Offload models to CPU
        expand_prompt: Whether to use prompt expansion
        save_path: If provided, save the video to this path

    Returns:
        numpy.ndarray: Video frames in THWC format

    Example:
        >>> video = image_to_video("photo.jpg", "The scene comes alive with gentle motion")
        >>> rp.save_video_mp4(video, "animated.mp4", framerate=24)
    """
    _ensure_kandinsky_installed()
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Load and prepare image
    image_pil = _load_image(image)

    pipe = get_i2v_pipeline(
        model_type=model_type,
        device=device,
        offload=offload,
    )

    kwargs = {
        "image": image_pil,
        "time_length": duration,
        "expand_prompts": 1 if expand_prompt else 0,
    }

    if num_steps is not None:
        kwargs["num_steps"] = num_steps
    if guidance_weight is not None:
        kwargs["guidance_weight"] = guidance_weight
    if save_path is not None:
        kwargs["save_path"] = save_path
    if seed is not None:
        kwargs["seed"] = seed

    result = pipe(prompt, **kwargs)

    return _video_to_numpy(result)


def text_to_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = None,
    guidance_weight: float = None,
    seed: int = None,
    device: str = None,
    offload: bool = False,
    expand_prompt: bool = True,
    save_path: str = None,
) -> "np.ndarray":
    """
    Generate an image from a text prompt.

    Args:
        prompt: Text description of the image to generate
        width: Image width (supported: 640, 768, 896, 1024, 1152, 1280, 1408)
        height: Image height (supported: 640, 768, 896, 1024, 1152, 1280, 1408)
        num_steps: Number of diffusion steps
        guidance_weight: CFG weight
        seed: Random seed
        device: Device to use
        offload: Offload models to CPU
        expand_prompt: Whether to use prompt expansion
        save_path: If provided, save image to this path

    Returns:
        numpy.ndarray: Image in HWC format

    Example:
        >>> image = text_to_image("A beautiful sunset over mountains")
        >>> rp.save_image(image, "sunset.png")
    """
    _ensure_kandinsky_installed()
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    pipe = get_t2i_pipeline(device=device, offload=offload)

    kwargs = {
        "width": width,
        "height": height,
        "expand_prompts": 1 if expand_prompt else 0,
    }

    if num_steps is not None:
        kwargs["num_steps"] = num_steps
    if guidance_weight is not None:
        kwargs["guidance_weight"] = guidance_weight
    if save_path is not None:
        kwargs["save_path"] = save_path
    if seed is not None:
        kwargs["seed"] = seed

    result = pipe(prompt, **kwargs)

    if result is not None:
        return rp.as_numpy_image(result)
    return result


def image_to_image(
    image,
    prompt: str,
    num_steps: int = None,
    guidance_weight: float = None,
    seed: int = None,
    device: str = None,
    offload: bool = False,
    expand_prompt: bool = True,
    save_path: str = None,
) -> "np.ndarray":
    """
    Transform an image based on a text prompt.

    Args:
        image: Input image (path, numpy array, PIL image, or torch tensor)
        prompt: Text description of how to transform the image
        num_steps: Number of diffusion steps
        guidance_weight: CFG weight
        seed: Random seed
        device: Device to use
        offload: Offload models to CPU
        expand_prompt: Whether to use prompt expansion
        save_path: If provided, save image to this path

    Returns:
        numpy.ndarray: Transformed image in HWC format

    Example:
        >>> result = image_to_image("photo.jpg", "Make it look like a painting")
        >>> rp.save_image(result, "painting.png")
    """
    _ensure_kandinsky_installed()
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Load and prepare image
    image_pil = _load_image(image)

    pipe = get_i2i_pipeline(device=device, offload=offload)

    kwargs = {
        "image": image_pil,
        "expand_prompts": 1 if expand_prompt else 0,
    }

    if num_steps is not None:
        kwargs["num_steps"] = num_steps
    if guidance_weight is not None:
        kwargs["guidance_weight"] = guidance_weight
    if save_path is not None:
        kwargs["save_path"] = save_path
    if seed is not None:
        kwargs["seed"] = seed

    result = pipe(prompt, **kwargs)

    if result is not None:
        return rp.as_numpy_image(result)
    return result


if __name__ == "__main__":
    rp.pip_import("fire")
    import fire
    fire.Fire({name: globals()[name] for name in __all__})
