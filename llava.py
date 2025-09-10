"""
llava16: Minimal video captioning/VQA wrapper (LLaVA-Next Video v1.6, 34B)

This module exposes two simple functions:
- describe_video(video, *, num_frames=16, model_id=None) -> str
- chat_video(video, prompt, *, num_frames=16, model_id=None) -> str

It mirrors a proven working flow:
    - Model:  llava-hf/llava-v1.6-34b-hf
    - Class:  transformers.LlavaNextVideoForConditionalGeneration
    - Proc :  transformers.AutoProcessor
    - Frames: rp.load_video_via_decord(..., indices=<num_frames>) -> rp.as_pil_images(...)

Inputs:
    - video: path/URL or list of frames
    - prompt: string, e.g. "Please describe this video in detail."

Example:
    import llava16 as L
    print(L.describe_video("clip.mp4", num_frames=16))
    print(L.chat_video("clip.mp4", "How many people are speaking?", num_frames=16))
"""

import torch
import rp

from typing import Optional, List


__all__ = ["describe_video", "chat_video"]

# Just for reference
known_model_ids = [
    "llava-hf/llava-v1.6-7b-hf",
    "llava-hf/llava-v1.6-34b-hf",  # Uses 65GB Vram at 16 frames
]

# Default HF model id (override via function args if desired)
_DEFAULT_MODEL_ID = "llava-hf/llava-v1.6-34b-hf"


@rp.memoized
def _get_llava16(model_id: str):
    """
    Download and initialize the LLaVA-Next Video v1.6 model + processor.
    Memoized so the heavy initialization happens only once per model_id.

    Args:
        model_id: HuggingFace model id

    Returns:
        (model, processor)
    """
    rp.pip_import("transformers")
    from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # memory-friendly
        low_cpu_mem_usage=True,  # reduce CPU RAM during load
        device_map="auto",  # choose GPU(s) if available
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _load_video_frames(video, num_frames: Optional[int]) -> List:
    """
    Load frames with decord (fast path) and convert to PIL frames.

    Args:
        video: path/URL or list-of-frames
        num_frames: if int, evenly sample that many frames

    Returns:
        list of PIL.Image
    """
    if isinstance(video, str):
        frames = rp.load_video_via_decord(video, indices=num_frames)
    frames = rp.as_numpy_images(frames, copy=False)
    frames = rp.as_rgb_images(frames, copy=False)
    frames = rp.as_byte_images(frames, copy=False)
    frames = rp.as_pil_images(video, copy=False)
    return list(frames)


def _run(video, prompt: str, *, num_frames: int, model_id: Optional[str]) -> str:
    """
    Internal helper shared by describe_video / chat_video.
    """

    prompt = f"USER: <video>\n{prompt} ASSISTANT:"

    model_id = model_id or _DEFAULT_MODEL_ID
    model, processor = _get_llava16(model_id)

    frames = _load_video_frames(video, num_frames=num_frames)

    # One processor call; same pattern as your working snippet
    inputs = processor(text=prompt, videos=[frames], return_tensors="pt")

    # move tensors to cuda if available (keeps it simple & robust)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=100)

    # Decode; keep tokenizer cleanup disabled like your snippet
    result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    assert "ASSISTANT:" in result
    result[result.find("ASSISTANT:") + len("ASSISTANT:") :].strip()

    return result


def describe_video(video, *, num_frames: int = 16, model_id: Optional[str] = None) -> str:
    """
    Video captioning: generates a detailed description of the given video.

    Args:
        video: path/URL or list-of-frames
        num_frames: how many frames to sample (default 16)
        model_id: optional HF model id override

    Returns:
        str: caption
    """
    return _run(video, prompt, num_frames=num_frames, model_id=model_id)


def chat_video(video, prompt, *, num_frames: int = 16, model_id: Optional[str] = None) -> str:
    """
    Video Q&A: asks a question about the video.

    Args:
        video: path/URL or list-of-frames
        prompt: user question or instruction (no need to add special tokens)
        num_frames: how many frames to sample (default 16)
        model_id: optional HF model id override

    Returns:
        str: answer text
    """
    return _run(video, prompt, num_frames=num_frames, model_id=model_id)