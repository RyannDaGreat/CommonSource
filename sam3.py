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

Python API Example:
    # Segment all instances of a concept in an image
    masks, boxes, scores = segment_image("photo.jpg", "person")

    # Segment with point prompts
    masks, boxes, scores = segment_image_points("photo.jpg", points=[[100, 200]], labels=[1])

    # Segment with box prompts
    masks, boxes, scores = segment_image_boxes("photo.jpg", boxes=[[10, 10, 200, 200]])

    # Segment and track objects in video
    results = segment_video("video.mp4", "person")

CLI Usage:
    # Segment with text prompt and save visualization
    python segment_anything.py segment_image photo.jpg "cat" --output=result.jpg --device=cuda:0

    # Segment with point prompts
    python segment_anything.py segment_image_points photo.jpg "[[100,200]]" "[1]" --output=points.jpg

    # Segment with box prompts
    python segment_anything.py segment_image_boxes photo.jpg "[[10,10,200,200]]" --output=boxes.jpg

    # Download model to default location
    python segment_anything.py download_model

    # Get model path
    python segment_anything.py get_model_path

See: https://huggingface.co/facebook/sam3
     https://github.com/facebookresearch/sam3

Note: This module uses the community mirror at 1038lab/sam3 for model weights
      since the official facebook/sam3 requires gated access approval.
"""
import rp


__all__ = [
    "segment_image",
    "segment_image_points",
    "segment_image_boxes",
    "segment_video",
    "download_model",
    "get_model_path",
]


# Model paths and identifiers
MIRROR_MODEL_ID = "1038lab/sam3"
MIRROR_WEIGHTS_FILE = "sam3.pt"  # Use .pt format for compatibility with sam3 package
OFFICIAL_MODEL_ID = "facebook/sam3"
OFFICIAL_WEIGHTS_FILE = "sam3.pt"

# Default paths for local model storage
DEFAULT_NETWORK_MODEL_PATH = "/root/models/sam3"
DEFAULT_LOCAL_MODEL_PATH = "/models/sam3"

# Global state
_sam3_device = None
_sam3_model = None
_sam3_processor = None


def get_model_path(prefer_local=True, use_mirror=True):
    """
    Get the path to use for SAM3 model weights.

    Args:
        prefer_local: If True, prefer local path over network path
        use_mirror: If True, use community mirror; if False, use official (requires access)

    Returns:
        str: Path to model weights file or None if needs download
    """
    import os

    paths_to_check = []
    if prefer_local:
        paths_to_check = [DEFAULT_LOCAL_MODEL_PATH, DEFAULT_NETWORK_MODEL_PATH]
    else:
        paths_to_check = [DEFAULT_NETWORK_MODEL_PATH, DEFAULT_LOCAL_MODEL_PATH]

    for path in paths_to_check:
        safetensors_file = os.path.join(path, "sam3.safetensors")
        if os.path.exists(safetensors_file):
            return safetensors_file
        pt_file = os.path.join(path, "sam3.pt")
        if os.path.exists(pt_file):
            return pt_file

    return None


def download_model(path=None, use_mirror=True, force=False):
    """
    Download SAM3 model to specified path.

    Args:
        path: Directory to save model. Defaults to DEFAULT_NETWORK_MODEL_PATH
        use_mirror: If True, download from community mirror (no auth required)
        force: If True, re-download even if model exists

    Returns:
        str: Path where model was saved
    """
    import os

    if path is None:
        path = DEFAULT_NETWORK_MODEL_PATH

    rp.pip_import("huggingface_hub")
    from huggingface_hub import hf_hub_download

    os.makedirs(path, exist_ok=True)

    if use_mirror:
        weights_file = os.path.join(path, "sam3.safetensors")
        if os.path.exists(weights_file) and not force:
            print(f"Model already exists at {weights_file}. Use force=True to re-download.")
            return weights_file

        print(f"Downloading SAM3 model from mirror to {path}...")
        downloaded = hf_hub_download(
            repo_id=MIRROR_MODEL_ID,
            filename=MIRROR_WEIGHTS_FILE,
            local_dir=path,
        )
        print(f"Model downloaded to {downloaded}")
        return downloaded
    else:
        weights_file = os.path.join(path, "sam3.pt")
        if os.path.exists(weights_file) and not force:
            print(f"Model already exists at {weights_file}. Use force=True to re-download.")
            return weights_file

        print(f"Downloading SAM3 model from official repo to {path}...")
        print("Note: You must have access to facebook/sam3 on HuggingFace.")
        downloaded = hf_hub_download(
            repo_id=OFFICIAL_MODEL_ID,
            filename=OFFICIAL_WEIGHTS_FILE,
            local_dir=path,
        )
        print(f"Model downloaded to {downloaded}")
        return downloaded


def _default_sam3_device():
    """Get or initialize the default device for SAM3 model."""
    global _sam3_device
    if _sam3_device is None:
        _sam3_device = rp.select_torch_device(reserve=True)
    return _sam3_device


def _normalize_device(device):
    """Normalize device specification."""
    rp.pip_import("torch")
    import torch
    global _sam3_device

    if device is None:
        device = _default_sam3_device()

    # Convert torch.device to string
    if hasattr(device, 'type'):
        device = str(device)
    elif isinstance(device, int):
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, str) and device.lower() == "cpu":
        device = "cpu"

    _sam3_device = device
    return device


def _get_checkpoint_path(model_path=None, use_mirror=True):
    """Get the checkpoint path, downloading if necessary."""
    import os
    rp.pip_import("huggingface_hub")
    from huggingface_hub import hf_hub_download

    if model_path is not None and os.path.exists(model_path):
        return model_path

    # Check local paths
    local_path = get_model_path(use_mirror=use_mirror)
    if local_path is not None:
        return local_path

    # Download from HuggingFace
    if use_mirror:
        return hf_hub_download(repo_id=MIRROR_MODEL_ID, filename=MIRROR_WEIGHTS_FILE)
    else:
        return hf_hub_download(repo_id=OFFICIAL_MODEL_ID, filename=OFFICIAL_WEIGHTS_FILE)


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

    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device=sam3_device,
        eval_mode=True,
    )

    # Create processor with the correct device
    processor = Sam3Processor(model, device=device)

    return model, processor


def _get_sam3_model(model_path=None, device=None, use_mirror=True):
    """Get the SAM3 model, downloading if needed."""
    device = _normalize_device(device)
    checkpoint_path = _get_checkpoint_path(model_path, use_mirror)
    return _get_sam3_model_helper(checkpoint_path, device)


def _load_image(image):
    """Load and preprocess an image for SAM3 model input."""
    if isinstance(image, str):
        image = rp.load_image(image)

    image = rp.as_numpy_image(image)
    image = rp.as_rgb_image(image)
    image = rp.as_byte_image(image)
    image = rp.as_pil_image(image)

    return image


def _save_visualization(image, masks, boxes, scores, output_path):
    """Save a visualization with mask overlays and bounding boxes."""
    import numpy as np

    # Load original image if needed
    if isinstance(image, str):
        image = rp.load_image(image)

    vis = rp.as_float_image(image).copy()

    # Overlay masks with different colors
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        vis[mask] = vis[mask] * 0.5 + np.array(color) * 0.5

    # Draw bounding boxes
    vis_byte = rp.as_byte_image(vis)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = [int(c * 255) for c in colors[i % len(colors)]]
        thickness = 2
        vis_byte[max(0,y1):min(vis_byte.shape[0],y1+thickness), max(0,x1):min(vis_byte.shape[1],x2)] = color
        vis_byte[max(0,y2-thickness):min(vis_byte.shape[0],y2), max(0,x1):min(vis_byte.shape[1],x2)] = color
        vis_byte[max(0,y1):min(vis_byte.shape[0],y2), max(0,x1):min(vis_byte.shape[1],x1+thickness)] = color
        vis_byte[max(0,y1):min(vis_byte.shape[0],y2), max(0,x2-thickness):min(vis_byte.shape[1],x2)] = color

    rp.save_image(vis_byte, output_path)
    print(f"Saved visualization to: {output_path}")
    print(f"  Found {len(masks)} objects, scores: {scores.tolist() if hasattr(scores, 'tolist') else list(scores)}")


def _load_video(video, num_frames=None):
    """Load and preprocess a video for SAM3 model input."""
    if isinstance(video, str):
        video = rp.load_video(video)

    if num_frames is not None:
        video = rp.resize_list(video, num_frames)

    video = rp.as_numpy_images(video)
    video = rp.as_rgb_images(video)
    video = rp.as_byte_images(video)
    video = rp.as_pil_images(video)

    return video


def segment_image(image, text, *, device=None, model_path=None,
                  threshold=0.5, use_mirror=True, output=None):
    """
    Segment all instances of a concept in an image using text prompt.

    Args:
        image: np.ndarray, PIL Image, or path/URL
        text: Text prompt describing what to segment (e.g., "person", "car")
        device: Optional device to run inference on (str, int, or torch.device)
        model_path: Optional model path to use a specific SAM3 model
        threshold: Detection confidence threshold (default 0.5)
        use_mirror: Whether to use community mirror for model download
        output: Optional path to save visualization image

    Returns:
        tuple: (masks, boxes, scores)
            - masks: NHW bool np.ndarray - Binary masks for each detected instance
            - boxes: Nx4 np.ndarray - Bounding boxes in XYXY format
            - scores: N np.ndarray - Confidence scores for each detection
    """
    import numpy as np

    model, processor = _get_sam3_model(model_path=model_path, device=device, use_mirror=use_mirror)

    # Load and process image
    pil_image = _load_image(image)

    # Run inference using Sam3Processor
    inference_state = processor.set_image(pil_image)
    result = processor.set_text_prompt(state=inference_state, prompt=text)

    import torch

    # Extract results
    masks = result.get("masks", [])
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    # Convert to numpy arrays
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    elif isinstance(masks, list) and len(masks) > 0:
        if isinstance(masks[0], torch.Tensor):
            masks = torch.stack(masks).cpu().numpy()
        else:
            masks = np.array(masks)
    else:
        masks = np.zeros((0, pil_image.height, pil_image.width), dtype=bool)

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    elif isinstance(boxes, list) and len(boxes) > 0:
        boxes = np.array(boxes)
    else:
        boxes = np.zeros((0, 4))

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    elif isinstance(scores, list):
        scores = np.array(scores)
    else:
        scores = np.array([])

    # Filter by threshold
    if len(scores) > 0:
        mask_thresh = scores >= threshold
        masks = masks[mask_thresh]
        boxes = boxes[mask_thresh]
        scores = scores[mask_thresh]

    # Squeeze extra dimensions from masks (e.g., (N, 1, H, W) -> (N, H, W))
    while len(masks) > 0 and masks.ndim > 3:
        masks = masks.squeeze(1)

    # Ensure masks are boolean
    if len(masks) > 0 and masks.dtype != bool:
        masks = masks > 0.5

    # Save visualization if output path provided
    if output is not None:
        _save_visualization(image, masks, boxes, scores, output)

    return masks, boxes, scores


def segment_image_points(image, points, labels, *, device=None, model_path=None,
                         threshold=0.5, use_mirror=True, point_box_size=0.02, output=None):
    """
    Segment objects in an image using point prompts.

    Note: SAM3's processor uses box prompts internally. Points are converted
    to tiny boxes centered at each point location.

    Args:
        image: np.ndarray, PIL Image, or path/URL
        points: List of [x, y] coordinates or Nx2 array (in pixel coordinates)
        labels: List of labels (1=foreground, 0=background) or N array
        device: Optional device to run inference on
        model_path: Optional model path to use a specific SAM3 model
        threshold: Detection confidence threshold (default 0.5)
        use_mirror: Whether to use community mirror
        point_box_size: Size of the box around each point as fraction of image (default 0.02)
        output: Optional path to save visualization image

    Returns:
        tuple: (masks, boxes, scores)
    """
    import numpy as np

    model, processor = _get_sam3_model(model_path=model_path, device=device, use_mirror=use_mirror)

    import torch

    pil_image = _load_image(image)
    width, height = pil_image.size

    points = np.array(points) if not isinstance(points, np.ndarray) else points
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

    # Set image and prepare for prompts
    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)

    # Convert points to tiny boxes in [center_x, center_y, width, height] normalized format
    # Sam3Processor.add_geometric_prompt expects boxes in [cx, cy, w, h] normalized to [0, 1]
    result = None
    for i, (point, label) in enumerate(zip(points, labels)):
        # Normalize point coordinates to [0, 1]
        cx = point[0] / width
        cy = point[1] / height
        # Create tiny box around point
        box = [cx, cy, point_box_size, point_box_size]
        is_positive = bool(label == 1)
        result = processor.add_geometric_prompt(
            box=box,
            label=is_positive,
            state=inference_state
        )

    if result is None:
        # No points provided
        return (
            np.zeros((0, height, width), dtype=bool),
            np.zeros((0, 4)),
            np.array([])
        )

    masks = result.get("masks", [])
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    elif isinstance(masks, list) and len(masks) > 0:
        masks = np.array(masks)
    else:
        masks = np.zeros((0, height, width), dtype=bool)

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    else:
        boxes = np.array(boxes) if len(boxes) > 0 else np.zeros((0, 4))

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    else:
        scores = np.array(scores) if len(scores) > 0 else np.array([])

    # Filter by threshold
    if len(scores) > 0:
        mask_thresh = scores >= threshold
        masks = masks[mask_thresh]
        boxes = boxes[mask_thresh]
        scores = scores[mask_thresh]

    # Squeeze extra dimensions from masks
    while len(masks) > 0 and masks.ndim > 3:
        masks = masks.squeeze(1)

    if len(masks) > 0 and masks.dtype != bool:
        masks = masks > 0.5

    if output is not None:
        _save_visualization(image, masks, boxes, scores, output)

    return masks, boxes, scores


def segment_image_boxes(image, boxes, labels=None, text=None, *, device=None, model_path=None,
                        threshold=0.5, use_mirror=True, output=None):
    """
    Segment objects in an image using bounding box prompts.

    Args:
        image: np.ndarray, PIL Image, or path/URL
        boxes: List of [x1, y1, x2, y2] boxes or Nx4 array (in pixel coordinates)
        labels: Optional list of labels (1=positive, 0=negative). Default: all positive.
        text: Optional text prompt to combine with box prompts
        device: Optional device to run inference on
        model_path: Optional model path to use a specific SAM3 model
        threshold: Detection confidence threshold (default 0.5)
        use_mirror: Whether to use community mirror
        output: Optional path to save visualization image

    Returns:
        tuple: (masks, boxes, scores)
    """
    import numpy as np

    model, processor = _get_sam3_model(model_path=model_path, device=device, use_mirror=use_mirror)

    import torch

    pil_image = _load_image(image)
    width, height = pil_image.size

    boxes_arr = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes

    if labels is None:
        labels = [1] * len(boxes_arr)
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

    # Set image and prepare for prompts
    inference_state = processor.set_image(pil_image)
    processor.reset_all_prompts(inference_state)

    # Optionally set text prompt first
    if text is not None:
        inference_state = processor.set_text_prompt(state=inference_state, prompt=text)

    # Convert boxes from [x1, y1, x2, y2] to [center_x, center_y, width, height] normalized format
    result = None
    for i, (box, label) in enumerate(zip(boxes_arr, labels)):
        x1, y1, x2, y2 = box
        # Convert to center format and normalize
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        box_cxcywh = [cx, cy, w, h]
        is_positive = bool(label == 1)
        result = processor.add_geometric_prompt(
            box=box_cxcywh,
            label=is_positive,
            state=inference_state
        )

    if result is None:
        # No boxes provided
        return (
            np.zeros((0, height, width), dtype=bool),
            np.zeros((0, 4)),
            np.array([])
        )

    masks = result.get("masks", [])
    out_boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    elif isinstance(masks, list) and len(masks) > 0:
        masks = np.array(masks)
    else:
        masks = np.zeros((0, height, width), dtype=bool)

    if isinstance(out_boxes, torch.Tensor):
        out_boxes = out_boxes.cpu().numpy()
    else:
        out_boxes = np.array(out_boxes) if len(out_boxes) > 0 else np.zeros((0, 4))

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    else:
        scores = np.array(scores) if len(scores) > 0 else np.array([])

    # Filter by threshold
    if len(scores) > 0:
        mask_thresh = scores >= threshold
        masks = masks[mask_thresh]
        out_boxes = out_boxes[mask_thresh]
        scores = scores[mask_thresh]

    # Squeeze extra dimensions from masks
    while len(masks) > 0 and masks.ndim > 3:
        masks = masks.squeeze(1)

    if len(masks) > 0 and masks.dtype != bool:
        masks = masks > 0.5

    if output is not None:
        _save_visualization(image, masks, out_boxes, scores, output)

    return masks, out_boxes, scores


def segment_video(video, text, *, device=None, model_path=None,
                  num_frames=None, max_frames_to_track=None, use_mirror=True):
    """
    Segment and track all instances of a concept throughout a video.

    Args:
        video: List of frames, path, or URL
        text: Text prompt describing what to segment (e.g., "person")
        device: Optional device to run inference on
        model_path: Optional model path to use a specific SAM3 model
        num_frames: Number of frames to process. If None, processes all frames.
        max_frames_to_track: Maximum frames to track. If None, tracks all frames.
        use_mirror: Whether to use community mirror

    Returns:
        dict: Results with 'masks', 'boxes', 'scores', 'object_ids' keys
    """
    import numpy as np
    import tempfile
    import os

    rp.pip_import("torch")
    import torch
    rp.pip_import("sam3")
    from sam3.model_builder import build_sam3_video_predictor

    device = _normalize_device(device)
    checkpoint_path = _get_checkpoint_path(model_path, use_mirror)

    frames = _load_video(video, num_frames)

    # SAM3 video predictor requires frames saved to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frame_dir)

        for i, frame in enumerate(frames):
            frame_path = os.path.join(frame_dir, f"{i:06d}.jpg")
            frame.save(frame_path)

        video_predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )

        response = video_predictor.handle_request(
            request=dict(type="start_session", resource_path=frame_dir)
        )
        session_id = response["session_id"]

        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=text
            )
        )

        # Propagate through video
        all_masks = []
        all_boxes = []
        all_scores = []
        object_ids = []

        if max_frames_to_track is None:
            max_frames_to_track = len(frames)

        for frame_idx in range(min(max_frames_to_track, len(frames))):
            response = video_predictor.handle_request(
                request=dict(
                    type="propagate",
                    session_id=session_id,
                    frame_index=frame_idx
                )
            )

            outputs = response.get("outputs", {})
            masks = outputs.get("masks", [])
            boxes = outputs.get("boxes", [])
            scores = outputs.get("scores", [])

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            all_masks.append(masks)
            all_boxes.append(boxes)
            all_scores.append(scores)

            if frame_idx == 0:
                object_ids = outputs.get("object_ids", [])

        # Reorganize by object
        num_objects = len(object_ids) if object_ids else (len(all_masks[0]) if all_masks else 0)

        masks_per_obj = [[] for _ in range(num_objects)]
        boxes_per_obj = [[] for _ in range(num_objects)]
        scores_per_obj = [[] for _ in range(num_objects)]

        for frame_masks, frame_boxes, frame_scores in zip(all_masks, all_boxes, all_scores):
            for obj_idx in range(num_objects):
                if obj_idx < len(frame_masks):
                    masks_per_obj[obj_idx].append(frame_masks[obj_idx])
                    boxes_per_obj[obj_idx].append(frame_boxes[obj_idx] if obj_idx < len(frame_boxes) else np.zeros(4))
                    scores_per_obj[obj_idx].append(frame_scores[obj_idx] if obj_idx < len(frame_scores) else 0.0)

        for obj_idx in range(num_objects):
            if masks_per_obj[obj_idx]:
                masks_per_obj[obj_idx] = np.stack(masks_per_obj[obj_idx], axis=0)
                boxes_per_obj[obj_idx] = np.stack(boxes_per_obj[obj_idx], axis=0)
                scores_per_obj[obj_idx] = np.array(scores_per_obj[obj_idx])

        return {
            'masks': masks_per_obj,
            'boxes': boxes_per_obj,
            'scores': scores_per_obj,
            'object_ids': list(object_ids) if object_ids else list(range(num_objects)),
        }


def demo():
    """Run this demo to test SAM3 capabilities."""
    import numpy as np

    rp.git_import('CommonSource', pull=True)
    import rp.git.CommonSource.segment_anything as sam

    print("=== SAM3 Image Segmentation Demo ===")
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = rp.load_image(image_url, use_cache=True)
    print(f"Image shape: {image.shape}")

    masks, boxes, scores = sam.segment_image(image, "cat")
    print(f"Found {len(masks)} cats")
    print(f"Scores: {scores}")

    if len(masks) > 0:
        overlay = rp.as_float_image(image).copy()
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        output_path = "/tmp/sam3_demo_output.jpg"
        rp.save_image(overlay, output_path)
        print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    import fire
    fire.Fire({name: globals()[name] for name in __all__})
