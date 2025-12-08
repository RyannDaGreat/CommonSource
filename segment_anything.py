"""
SAM3: Segment Anything Model 3 - Meta AI's unified foundation model for promptable segmentation

This module provides simple functions to work with Meta's SAM3 model,
which handles both images and videos for:
- Text-prompted segmentation (segment all instances of a concept)
- Point/box-prompted segmentation
- Video object tracking with text prompts

Functions automatically download the model on first use. No class initialization needed.

Input formats:
- Images: np.ndarray (HW3 uint8 or float 0-1), PIL Image, or path/URL
- Videos: List of frames, path, or URL
- Text: String prompts for concept to segment (e.g., "person", "car", "dog")

All segment functions return an EasyDict with:
    - mask: HW (or THW for video) bool - Combined mask (OR of all instances)
    - masks: NHW (or TNHW for video) bool - Per-instance binary masks
    - boxes: Nx4 np.ndarray - Bounding boxes in XYXY format (images only)
    - scores: N np.ndarray - Confidence scores (images only)

Python API Example:
    # Segment all instances of a concept in an image
    result = segment_image("photo.jpg", "person")
    result.mask   # HW combined mask
    result.masks  # NHW per-instance masks
    result.boxes  # Nx4 bounding boxes
    result.scores # N confidence scores

    # Segment with point prompts
    result = segment_image_points("photo.jpg", points=[[100, 200]], labels=[1])

    # Segment with box prompts
    result = segment_image_boxes("photo.jpg", boxes=[[10, 10, 200, 200]])

    # Segment and track objects in video
    result = segment_video("video.mp4", "person")
    result.mask   # THW combined mask per frame
    result.masks  # TNHW per-frame per-instance masks

Module settings:
    sam3.MODEL_ID = "1038lab/sam3"  # Community mirror (default), or "facebook/sam3" (requires auth)
    sam3.default_model_path = None  # Override HuggingFace cache location

See: https://huggingface.co/facebook/sam3
     https://github.com/facebookresearch/sam3
"""
import rp


__all__ = [
    "segment_image",
    "segment_image_points",
    "segment_image_boxes",
    "segment_video",
    "visualize_segmentation",
    "download_model",
    "default_model_path",
    "demo",
]

PIP_REQUIREMENTS = [
    "torch",
    "huggingface_hub",
    "sam3",
]

# Model configuration
MODEL_ID = "1038lab/sam3"        # Community mirror (no auth required)
# MODEL_ID = "facebook/sam3"    # Official (requires gated HuggingFace access)
WEIGHTS_FILE = "sam3.pt"

# Default model path (set to override HuggingFace cache)
default_model_path = None


def download_model(path=None, force=False):
    """
    Download SAM3 model, or return cached path if already downloaded.

    This function is idempotent - calling it multiple times will not
    re-download. Use this to get the model path.

    Args:
        path: Directory to save model. If None, uses HuggingFace cache.
        force: If True, re-download even if model exists

    Returns:
        str: Path to the model file
    """
    import os

    rp.pip_import("huggingface_hub")
    from huggingface_hub import hf_hub_download

    # Use global override if set
    if path is None:
        path = default_model_path

    # Check if already exists when path is specified
    if path:
        os.makedirs(path, exist_ok=True)
        weights_file = os.path.join(path, WEIGHTS_FILE)
        if os.path.exists(weights_file) and not force:
            print("Model already exists at %s" % weights_file)
            return weights_file
        print("Downloading SAM3 model to %s..." % path)

    # Single download call - local_dir is None when using HuggingFace cache
    return hf_hub_download(
        repo_id=MODEL_ID,
        filename=WEIGHTS_FILE,
        local_dir=path,
        force_download=force,
    )


def _get_checkpoint_path(model_path=None):
    """Get the checkpoint path, downloading if necessary."""
    import os

    if model_path is not None and os.path.exists(model_path):
        return model_path

    return download_model()


def _get_bpe_vocab_path():
    """
    Get path to the BPE vocabulary file, downloading if needed.

    The sam3 package requires the CLIP BPE vocabulary file for text encoding.
    This file is from OpenAI's CLIP repository.
    """
    import os

    # Store in the sam3 package's expected location
    sam3_dir = os.path.dirname(__import__("sam3").__file__)
    assets_dir = os.path.join(os.path.dirname(sam3_dir), "assets")
    bpe_path = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")

    if os.path.exists(bpe_path):
        return bpe_path

    # Download from OpenAI's CLIP repository
    print(f"Downloading BPE vocabulary file to {bpe_path}...")
    os.makedirs(assets_dir, exist_ok=True)

    url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    rp.download_url(url, bpe_path)

    print(f"Downloaded BPE vocabulary to {bpe_path}")
    return bpe_path


@rp.memoized
def _get_sam3_model_helper(checkpoint_path, device):
    """Load the SAM3 model using the official sam3 package."""
    rp.pip_import("torch")
    import torch
    rp.pip_import("sam3")

    # Import from official sam3 package
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"Loading SAM3 model from {checkpoint_path}...")
    print(f"Using device: {device}")

    # For specific GPU selection (cuda:N), set the default device first
    # This ensures all tensors created during model building go to the right GPU
    if device.startswith("cuda:"):
        gpu_idx = int(device.split(":")[1])
        torch.cuda.set_device(gpu_idx)
        sam3_device = "cuda"  # Now "cuda" refers to the selected GPU
    else:
        sam3_device = device

    # Get the BPE vocabulary path (downloads if needed)
    bpe_path = _get_bpe_vocab_path()

    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device=sam3_device,
        eval_mode=True,
        bpe_path=bpe_path,
    )

    # Create processor with the correct device
    processor = Sam3Processor(model, device=device)

    return model, processor


def _get_sam3_model(model_path=None, device=None):
    """Get the SAM3 model, downloading if needed."""
    device = str(rp.r._resolve_torch_device(device))
    checkpoint_path = _get_checkpoint_path(model_path)
    return _get_sam3_model_helper(checkpoint_path, device)


@rp.memoized
def _get_sam3_video_predictor_helper(checkpoint_path, bpe_path, gpu_ids):
    """Load the SAM3 video predictor (cached per GPU configuration).

    Args:
        checkpoint_path: Path to model checkpoint
        bpe_path: Path to BPE vocab file
        gpu_ids: Tuple of GPU indices to use (must be tuple for memoization)
    """
    rp.pip_import("sam3")
    from sam3.model_builder import build_sam3_video_predictor

    return build_sam3_video_predictor(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        gpus_to_use=list(gpu_ids),
    )


def _get_sam3_video_predictor(model_path=None, device=None):
    """
    Get the SAM3 video predictor, downloading if needed.

    Args:
        model_path: Override model path
        device: Device(s) to run on. Can be:
            - None: Auto-select (cuda:0)
            - int: Single GPU index (e.g., 0, 1, 7)
            - str: Device string (e.g., 'cuda:0', 'cuda:1')
            - list/set: Multiple GPUs (e.g., [0, 1, 2] or {4, 5, 6})

    Returns:
        SAM3 video predictor loaded on the specified GPU(s)
    """
    checkpoint_path = _get_checkpoint_path(model_path)
    bpe_path = _get_bpe_vocab_path()
    gpu_ids = _resolve_gpu_ids(device)
    return _get_sam3_video_predictor_helper(checkpoint_path, bpe_path, gpu_ids)


def _resolve_gpu_ids(device):
    """Convert device specification(s) to a tuple of GPU index integers."""
    if rp.is_iterable(device) and not isinstance(device, str):
        return tuple(sorted(set(rp.r._resolve_torch_device(d).index for d in device)))
    return (rp.r._resolve_torch_device(device).index,)


def _load_image(image):
    """Load and preprocess an image for SAM3 model input."""
    if isinstance(image, str):
        image = rp.load_image(image)

    image = rp.as_numpy_image(image,copy=False)
    image = rp.as_rgb_image(image,copy=False)
    image = rp.as_byte_image(image,copy=False)
    image = rp.as_pil_image(image,copy=False)

    return image



def _load_video(video):
    """Load and preprocess a video for SAM3 model input."""
    if isinstance(video, str):
        video = rp.load_video(video)
    return [_load_image(f) for f in rp.eta(video, "Loading frames")]


def visualize_segmentation(image, masks, boxes=None, scores=None):
    """
    Create a visualization of segmentation results overlaid on an image.

    Args:
        image: Original image (np.ndarray, PIL Image, or path/URL)
        masks: NHW bool np.ndarray of binary masks
        boxes: Optional Nx4 np.ndarray of bounding boxes in XYXY format
        scores: Optional N np.ndarray of confidence scores (for annotation)

    Returns:
        np.ndarray: HW3 uint8 image with mask overlays and optional boxes
    """
    import numpy as np

    if isinstance(image, str):
        image = rp.load_image(image)

    image = rp.as_numpy_image(image)
    vis = rp.as_float_image(image).copy()

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    h, w = vis.shape[:2]
    for i, mask in enumerate(masks):
        # Skip masks with mismatched dimensions
        if mask.shape != (h, w):
            continue
        color = colors[i % len(colors)]
        vis[mask] = vis[mask] * 0.5 + np.array(color) * 0.5

    vis = rp.as_byte_image(vis)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            color = [int(c * 255) for c in colors[i % len(colors)]]
            thickness = 2
            h, w = vis.shape[:2]
            vis[max(0, y1):min(h, y1 + thickness), max(0, x1):min(w, x2)] = color
            vis[max(0, y2 - thickness):min(h, y2), max(0, x1):min(w, x2)] = color
            vis[max(0, y1):min(h, y2), max(0, x1):min(w, x1 + thickness)] = color
            vis[max(0, y1):min(h, y2), max(0, x2 - thickness):min(w, x2)] = color

    return vis


def _to_numpy(x, empty_shape=(0,)):
    """Convert tensor/list to numpy array."""
    import numpy as np
    rp.pip_import("torch")
    import torch

    if isinstance(x, torch.Tensor):
        return x.cpu().float().numpy()
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return torch.stack(x).cpu().float().numpy()
    if isinstance(x, list) and len(x) > 0:
        return np.array(x)
    return np.zeros(empty_shape)


def _make_result(masks, boxes, scores):
    """
    Create a SegmentResult EasyDict from masks, boxes, scores arrays.

    Returns:
        EasyDict with:
            - mask: HW bool np.ndarray - Combined mask (OR of all instance masks)
            - masks: NHW bool np.ndarray - Binary masks for each detected instance
            - boxes: Nx4 np.ndarray - Bounding boxes in XYXY format
            - scores: N np.ndarray - Confidence scores for each detection
    """
    import numpy as np
    height, width = masks.shape[1:3] if len(masks) > 0 else (0, 0)
    mask = np.any(masks, axis=0) if len(masks) > 0 else np.zeros((height, width), dtype=bool)
    return rp.EasyDict(mask=mask, masks=masks, boxes=boxes, scores=scores)


def _empty_result(height, width):
    """Return empty segmentation result."""
    import numpy as np
    return _make_result(
        masks=np.zeros((0, height, width), dtype=bool),
        boxes=np.zeros((0, 4)),
        scores=np.array([]),
    )


def _process_sam3_result(result, height, width, threshold):
    """Convert SAM3 result dict to SegmentResult EasyDict."""
    masks = _to_numpy(result.get("masks", []), (0, height, width))
    boxes = _to_numpy(result.get("boxes", []), (0, 4))
    scores = _to_numpy(result.get("scores", []))

    # Filter by threshold
    if len(scores) > 0:
        keep = scores >= threshold
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]

    # Squeeze extra dimensions from masks (e.g., (N, 1, H, W) -> (N, H, W))
    while len(masks) > 0 and masks.ndim > 3:
        masks = masks.squeeze(1)

    # Ensure masks are boolean
    if len(masks) > 0 and masks.dtype != bool:
        masks = masks > 0.5

    return _make_result(masks, boxes, scores)


def segment_image(image, prompt, *, device=None, model_path=None, threshold=0.5):
    """
    Segment all instances of a concept in an image using text prompt.

    Args:
        image: np.ndarray, PIL Image, or path/URL
        prompt: Text prompt describing what to segment (e.g., "person", "car")
        device: Device to run inference on (str, int, or torch.device)
        model_path: Model path to use a specific SAM3 model
        threshold: Detection confidence threshold (default 0.5)

    Returns:
        EasyDict with:
            - mask: HW bool np.ndarray - Combined mask (OR of all instance masks)
            - masks: NHW bool np.ndarray - Binary masks for each detected instance
            - boxes: Nx4 np.ndarray - Bounding boxes in XYXY format
            - scores: N np.ndarray - Confidence scores for each detection
    """
    model, processor = _get_sam3_model(model_path=model_path, device=device)

    pil_image = _load_image(image)

    inference_state = processor.set_image(pil_image)
    result = processor.set_text_prompt(state=inference_state, prompt=prompt)

    return _process_sam3_result(result, pil_image.height, pil_image.width, threshold)


def segment_image_points(image, points, labels, *, device=None, model_path=None,
                         threshold=0.5, point_box_size=0.02):
    """
    Segment objects in an image using point prompts.

    Note: SAM3's processor uses box prompts internally. Points are converted
    to tiny boxes centered at each point location.

    Args:
        image: np.ndarray, PIL Image, or path/URL
        points: List of [x, y] coordinates or Nx2 array (in pixel coordinates)
        labels: List of labels (1=foreground, 0=background) or N array
        device: Device to run inference on
        model_path: Model path to use a specific SAM3 model
        threshold: Detection confidence threshold (default 0.5)
        point_box_size: Size of the box around each point as fraction of image (default 0.02)

    Returns:
        EasyDict with:
            - mask: HW bool np.ndarray - Combined mask (OR of all instance masks)
            - masks: NHW bool np.ndarray - Binary masks for each detected instance
            - boxes: Nx4 np.ndarray - Bounding boxes in XYXY format
            - scores: N np.ndarray - Confidence scores for each detection
    """
    import numpy as np

    model, processor = _get_sam3_model(model_path=model_path, device=device)

    pil_image = _load_image(image)
    width, height = pil_image.size

    points = np.array(points) if not isinstance(points, np.ndarray) else points
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)

    # Convert points to tiny boxes in [center_x, center_y, width, height] normalized format
    result = None
    for point, label in zip(points, labels):
        cx = point[0] / width
        cy = point[1] / height
        box = [cx, cy, point_box_size, point_box_size]
        result = processor.add_geometric_prompt(
            box=box,
            label=bool(label == 1),
            state=inference_state
        )

    if result is None:
        return _empty_result(height, width)

    return _process_sam3_result(result, height, width, threshold)


def segment_image_boxes(image, boxes, labels=None, prompt=None, *, device=None, model_path=None,
                        threshold=0.5):
    """
    Segment objects in an image using bounding box prompts.

    Args:
        image: np.ndarray, PIL Image, or path/URL
        boxes: List of [x1, y1, x2, y2] boxes or Nx4 array (in pixel coordinates)
        labels: Optional list of labels (1=positive, 0=negative). Default: all positive.
        prompt: Optional text prompt to combine with box prompts
        device: Device to run inference on
        model_path: Model path to use a specific SAM3 model
        threshold: Detection confidence threshold (default 0.5)

    Returns:
        EasyDict with:
            - mask: HW bool np.ndarray - Combined mask (OR of all instance masks)
            - masks: NHW bool np.ndarray - Binary masks for each detected instance
            - boxes: Nx4 np.ndarray - Bounding boxes in XYXY format
            - scores: N np.ndarray - Confidence scores for each detection
    """
    import numpy as np

    model, processor = _get_sam3_model(model_path=model_path, device=device)

    pil_image = _load_image(image)
    width, height = pil_image.size

    boxes_arr = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes

    if labels is None:
        labels = [1] * len(boxes_arr)
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)

    if prompt is not None:
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # Convert boxes from [x1, y1, x2, y2] to [center_x, center_y, width, height] normalized format
    result = None
    for box, label in zip(boxes_arr, labels):
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        result = processor.add_geometric_prompt(
            box=[cx, cy, w, h],
            label=bool(label == 1),
            state=inference_state
        )

    if result is None:
        return _empty_result(height, width)

    return _process_sam3_result(result, height, width, threshold)


def segment_video(video, prompt, *, device=None, model_path=None, threshold=0.5):
    """
    Segment and track all instances of a concept throughout a video.

    Args:
        video: List of frames, path, or URL
        prompt: Text prompt describing what to segment (e.g., "person")
        device: Device(s) to run inference on. Can be:
            - None: Auto-select (cuda:0)
            - int: Single GPU index (e.g., 0, 1, 7)
            - str: Device string (e.g., 'cuda:0', 'cuda:1')
            - list/set: Multiple GPUs (e.g., [0, 1, 2] or {4, 5, 6})
        model_path: Model path to use a specific SAM3 model
        threshold: Mask confidence threshold (default 0.5)

    Returns:
        EasyDict with:
            - mask: THW bool np.ndarray - Combined mask per frame (OR of all instances)
            - masks: TNHW bool np.ndarray - Per-frame per-instance binary masks
            - boxes: Empty Nx4 array (not available for video)
            - scores: Empty N array (not available for video)
    """
    import numpy as np

    rp.pip_import("torch")
    import torch

    # Load video as list of PIL images (in-memory, no disk I/O)
    frames = _load_video(video)

    if len(frames) == 0:
        return rp.EasyDict(
            mask=np.zeros((0, 0, 0), dtype=bool),
            masks=np.zeros((0, 0, 0, 0), dtype=bool),
            boxes=np.zeros((0, 4)),
            scores=np.array([]),
        )

    video_predictor = _get_sam3_video_predictor(model_path, device)

    # Start session with PIL images directly (no disk I/O)
    response = video_predictor.handle_request(
        request=dict(type="start_session", resource_path=frames)
    )
    session_id = response["session_id"]

    # Add text prompt
    video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt
        )
    )

    # Propagate through video
    all_frame_masks = []
    num_frames_to_process = len(frames)

    # Propagate through all frames - returns a generator
    propagation = video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=0,
            max_frame_num_to_track=num_frames_to_process,
            propagation_direction="forward",
        )
    )

    # Get frame dimensions from first PIL image
    height, width = frames[0].size[1], frames[0].size[0]

    # Process results for each frame
    for result in rp.eta(propagation, "Tracking", length=num_frames_to_process):
        outputs = result.get("outputs", {})
        masks = outputs.get("out_binary_masks", None)

        if masks is not None and len(masks) > 0:
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            # Apply threshold and convert to bool
            masks = masks > threshold
            all_frame_masks.append(masks)
        else:
            # No objects detected in this frame
            all_frame_masks.append(np.zeros((0, height, width), dtype=bool))

    # Pad all frames to same number of objects (max across all frames)
    max_objects = max((m.shape[0] for m in all_frame_masks), default=0)
    masks_tnhw = np.stack([rp.crop_tensor(m, (max_objects, height, width)) for m in all_frame_masks])

    # Combined mask per frame: THW
    mask_thw = np.any(masks_tnhw, axis=1)

    return rp.EasyDict(
        mask=mask_thw,
        masks=masks_tnhw,
        boxes=np.zeros((0, 4)),
        scores=np.array([]),
    )


def demo():
    """Run this demo to test SAM3 capabilities."""
    rp.git_import('CommonSource', pull=False)
    import rp.git.CommonSource.segment_anything as sam

    print("=== SAM3 Image Segmentation Demo ===")
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = rp.load_image(image_url, use_cache=True)
    print(f"Image shape: {image.shape}")

    result = sam.segment_image(image, "cat")
    print(f"Found {len(result.masks)} cats")
    print(f"Scores: {result.scores}")

    if len(result.masks) > 0:
        output_path = "/tmp/sam3_demo_output.jpg"
        rp.save_image(sam.visualize_segmentation(image, result.masks, result.boxes), output_path)
        print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    import fire
    fire.Fire({name: globals()[name] for name in __all__})
