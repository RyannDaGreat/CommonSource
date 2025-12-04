# Added December 3, 2025 by Clara Burgert using Claude

"""
ChronoEdit: NVIDIA's 14B parameter image editing model (2025)

This module provides simple functions to edit images using text prompts with NVIDIA's
ChronoEdit model. It works by treating image editing as video generation - the input
image is the first frame and the output image is the last frame.

Functions automatically download the model on first use. No class initialization needed.

Input formats:
- Images: np.ndarray (HW3 or HWC, uint8 or float 0-1), PIL Image, or path/URL
- Prompts: String describing the desired edit

Output:
- Edited image as np.ndarray (HWC uint8)

Example:
    # Edit an image
    edited = edit_image("photo.jpg", "Add sunglasses to the face")

    # Edit with specific GPU
    edited = edit_image("photo.jpg", "Make it look like sunset", device="cuda:1")

    # Multi-GPU inference (distribute across GPUs)
    edited = edit_image("photo.jpg", "Add a hat", devices=[0, 1, 2, 3])

See: https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers
"""
import rp
import os
import sys

# Use pip_import for numpy to ensure it's installed
rp.pip_import("numpy", auto_yes=True)
import numpy as np

__all__ = [
    "edit_image",
    "download_model",
    "default_model_path",
]

# Default paths - network drive location and HuggingFace fallback
default_model_path = "/root/CleanCode/Github/ChronoEdit/checkpoints/ChronoEdit-14B-Diffusers"
huggingface_model_id = "nvidia/ChronoEdit-14B-Diffusers"

# Global device tracking
_chronoedit_device = None


def _default_chronoedit_device():
    """
    Get or initialize the default device for ChronoEdit model.
    Uses rp.select_torch_device() to pick the best available device.

    Returns:
        The default torch device for ChronoEdit
    """
    global _chronoedit_device
    if _chronoedit_device is None:
        _chronoedit_device = rp.select_torch_device(reserve=True)
    return _chronoedit_device


def _install_dependencies():
    """Install required dependencies for ChronoEdit."""
    # Core dependencies
    rp.pip_import("torch", auto_yes=True)
    rp.pip_import("diffusers", auto_yes=True)
    rp.pip_import("transformers", auto_yes=True)
    rp.pip_import("accelerate", auto_yes=True)
    rp.pip_import("peft", auto_yes=True)
    # Note: flash_attn requires compilation; model works without it (just slower)
    try:
        rp.pip_import("flash_attn", auto_yes=True)
    except Exception:
        pass  # flash_attn optional, will fall back to standard attention


def download_model(path=None, force=False):
    """
    Download the ChronoEdit model from HuggingFace.

    Args:
        path: Local path to download the model to. If None, uses default_model_path.
        force: If True, re-download even if model exists.

    Returns:
        Path to the downloaded model
    """
    _install_dependencies()

    path = path or default_model_path

    # Check if model already exists
    model_index = os.path.join(path, "model_index.json")
    if os.path.exists(model_index) and not force:
        print(f"Model already exists at {path}")
        return path

    print(f"Downloading ChronoEdit model to {path}...")
    os.makedirs(path, exist_ok=True)

    # Use huggingface_hub to download
    rp.pip_import("huggingface_hub", auto_yes=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=huggingface_model_id,
        local_dir=path,
        local_dir_use_symlinks=False,
    )

    print(f"Model downloaded to {path}")
    return path


def _get_chronoedit_source_path():
    """Get the path to the ChronoEdit source code."""
    return "/root/CleanCode/Github/ChronoEdit"


@rp.memoized
def _get_chronoedit_pipeline_helper(model_path, device, offload_model):
    """
    Helper function to load the ChronoEdit pipeline.
    Results are memoized for efficiency.

    Args:
        model_path: Path to the model
        device: Device to load the model on
        offload_model: Whether to enable model offloading for memory savings

    Returns:
        ChronoEdit pipeline
    """
    import torch

    # Add ChronoEdit source to path for imports
    chronoedit_source = _get_chronoedit_source_path()
    if chronoedit_source not in sys.path:
        sys.path.insert(0, chronoedit_source)

    # Set environment variable to skip guardrails (they require gated nvidia model)
    os.environ["CHRONOEDIT_SKIP_GUARDRAILS"] = "1"

    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import UniPCMultistepScheduler
    from transformers import CLIPVisionModel
    from chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
    from chronoedit_diffusers.transformer_chronoedit import ChronoEditTransformer3DModel

    print(f"Loading ChronoEdit from {model_path}...")

    # Load components
    image_encoder = CLIPVisionModel.from_pretrained(
        model_path,
        subfolder="image_encoder",
        torch_dtype=torch.float32
    )

    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )

    transformer = ChronoEditTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    pipe = ChronoEditPipeline.from_pretrained(
        model_path,
        image_encoder=image_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
        disable_guardrails=True
    )

    # Configure scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=5.0
    )

    # Move to device
    pipe.to(device)
    pipe._offload_model = offload_model

    print(f"ChronoEdit loaded on {device}")
    return pipe


def _get_chronoedit_pipeline(model_path=None, device=None, offload_model=True):
    """
    Get the ChronoEdit pipeline. Downloads from HuggingFace if not available locally.

    Args:
        model_path: Path to the model. If None, uses default_model_path.
        device: Device to load the model on. If None, auto-selects.
        offload_model: Whether to offload model to CPU after each forward pass.

    Returns:
        ChronoEdit pipeline
    """
    _install_dependencies()

    # Resolve model path
    if model_path is None:
        model_path = default_model_path
        # Check if default exists, otherwise download
        if not os.path.exists(os.path.join(model_path, "model_index.json")):
            print(f"Model not found at {model_path}, downloading...")
            download_model(model_path)

    # Resolve device
    if device is None:
        device = _default_chronoedit_device()
    else:
        global _chronoedit_device
        _chronoedit_device = device

    import torch
    device = torch.device(device)

    return _get_chronoedit_pipeline_helper(model_path, device, offload_model)


def _load_image(image):
    """
    Load and preprocess an image for ChronoEdit input.

    Args:
        image: An image as path, URL, np.ndarray, or PIL.Image

    Returns:
        PIL.Image object ready for the model
    """
    # Load the image if its a path or URL
    if isinstance(image, str):
        image = rp.load_image(image)

    # Convert to PIL image (HWC uint8 RGB)
    image = rp.as_numpy_image(image)
    image = rp.as_rgb_image(image)
    image = rp.as_byte_image(image)
    image = rp.as_pil_image(image)

    return image


def _calculate_dimensions(image, mod_value):
    """
    Calculate output dimensions based on target resolution.
    Maintains aspect ratio while fitting to ~720p area.

    Args:
        image: PIL Image
        mod_value: Modulo value for dimension alignment

    Returns:
        Tuple of (width, height)
    """
    target_area = 720 * 1280

    aspect_ratio = image.height / image.width
    height = round(np.sqrt(target_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(target_area / aspect_ratio)) // mod_value * mod_value

    return width, height


def edit_image(
    image,
    prompt,
    *,
    device=None,
    devices=None,
    model_path=None,
    seed=None,
    num_inference_steps=50,
    guidance_scale=5.0,
    offload_model=True,
    height=None,
    width=None,
) -> np.ndarray:
    """
    Edit an image using a text prompt.

    Args:
        image: Input image as np.ndarray, PIL Image, or path/URL
        prompt: Text prompt describing the desired edit
        device: Device to run inference on (e.g., "cuda:0", "cuda:1").
                If None, auto-selects best available GPU.
        devices: List of GPU IDs for multi-GPU inference (e.g., [0, 1, 2, 3]).
                 If provided, overrides the device parameter.
                 Currently uses first GPU but reserves all for pipeline parallelism.
        model_path: Path to the ChronoEdit model. If None, uses default network path.
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps (default 50)
        guidance_scale: Classifier-free guidance scale (default 5.0)
        offload_model: Whether to offload model to CPU after each forward pass.
                       Reduces GPU memory but slower. Default True.
        height: Output height. If None, calculated from input aspect ratio.
        width: Output width. If None, calculated from input aspect ratio.

    Returns:
        Edited image as np.ndarray (HWC uint8 RGB)

    Example:
        # Basic usage
        edited = edit_image("cat.jpg", "Add sunglasses to the cat's face")

        # With specific GPU
        edited = edit_image("cat.jpg", "Make it sunset", device="cuda:2")

        # Multi-GPU (for future pipeline parallelism)
        edited = edit_image("cat.jpg", "Add a hat", devices=[0, 1, 2, 3])
    """
    # Make sure dependencies are installed first
    _install_dependencies()
    import torch

    # Handle multi-GPU specification
    if devices is not None:
        if isinstance(devices, (list, tuple)) and len(devices) > 0:
            # For now, use first GPU. Multi-GPU pipeline parallelism can be added later.
            # The devices list reserves all specified GPUs for potential future use.
            device = f"cuda:{devices[0]}"
            if len(devices) > 1:
                print(f"Multi-GPU specified: {devices}. Using cuda:{devices[0]} (pipeline parallelism planned for future)")
        else:
            device = f"cuda:{devices}" if isinstance(devices, int) else None

    # Get the pipeline
    pipe = _get_chronoedit_pipeline(model_path=model_path, device=device, offload_model=offload_model)

    # Load and preprocess image
    pil_image = _load_image(image)

    # Calculate dimensions
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    if height is None or width is None:
        calc_width, calc_height = _calculate_dimensions(pil_image, mod_value)
        width = width or calc_width
        height = height or calc_height

    # Resize image to target dimensions
    pil_image = pil_image.resize((width, height))

    # Setup generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Run inference (5 frames for editing, 29 for temporal reasoning)
    print(f"Editing image: {prompt}")
    print(f"Output dimensions: {width}x{height}, Steps: {num_inference_steps}")

    output = pipe(
        image=pil_image,
        prompt=prompt,
        negative_prompt=None,
        height=height,
        width=width,
        num_frames=5,  # 5 frames for editing (first=input, last=output)
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        enable_temporal_reasoning=False,
        generator=generator,
        offload_model=pipe._offload_model,
    ).frames[0]

    # Get the last frame (the edited image)
    last_frame = output[-1]

    # Convert to uint8 numpy array
    result = (last_frame * 255).clip(0, 255).astype(np.uint8)

    return result


def demo():
    """
    Run this demo to test ChronoEdit functionality.
    """
    import rp

    # Download and update this RP extension
    rp.git_import('CommonSource', pull=True)
    import rp.git.CommonSource.chronoedit as chronoedit

    # Get test image
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    image = rp.load_image(test_image_url, use_cache=True)
    image = rp.resize_image_to_fit(image, height=720, width=1280, allow_growth=False)

    # Edit the image
    prompt = "Add sunglasses to the cat's face"
    edited = chronoedit.edit_image(image, prompt, seed=42)

    # Display results
    comparison = rp.tiled_images([image, edited], length=2)
    comparison = rp.labeled_images(
        comparison,
        f"ChronoEdit: {prompt}",
        font="R:Futura",
        size=30,
    )
    rp.display_image(comparison)

    return edited


if __name__ == "__main__":
    import fire
    fire.Fire({name: globals()[name] for name in __all__})
