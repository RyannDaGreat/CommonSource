"""
DWPose: Whole-body pose estimation (body + hands + face, 133 COCO-WholeBody keypoints).

Uses rtmlib (ONNX Runtime backend) for fast inference. Auto-downloads models on first use.
No class initialization needed — just call the functions.

Input: BGR uint8 numpy images (single or batch).
Output: Per-person keypoints as (N_people, 133, 2) float32 + confidence scores (N_people, 133).

Keypoint breakdown (133 total, COCO-WholeBody):
    Body:  17 keypoints (0-16)   — nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
    Feet:   6 keypoints (17-22)  — big toes, small toes, heels
    Face:  68 keypoints (23-90)  — jaw, eyebrows, nose, eyes, mouth (iBUG 68-point)
    Hands: 42 keypoints (91-132) — 21 per hand (wrist, thumb, index, middle, ring, pinky)

Example:
    # Detect poses in a single image
    keypoints, scores = detect_poses(image)

    # Process a video's sprite sheet (25 frames)
    all_kpts, all_scores = detect_poses_batch(frames)

    # Draw skeletons on black background
    viz = draw_poses_on_black(keypoints, scores, height=512, width=512)

    # Compute statistics
    stats = compute_pose_stats(all_kpts, all_scores)

See: https://github.com/IDEA-Research/DWPose
     https://github.com/Tau-J/rtmlib
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import Doc

import numpy as np


__all__ = [
    "detect_poses",
    "detect_poses_batch",
    "draw_poses_on_black",
    "compute_pose_stats",
    "demo",
]

# ── Constants ──

N_BODY = 17
N_FEET = 6
N_FACE = 68
N_HANDS = 42
N_KEYPOINTS = N_BODY + N_FEET + N_FACE + N_HANDS  # 133

# COCO body skeleton edges (0-indexed, body keypoints only)
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),     # head
    (5, 6),                               # shoulders
    (5, 7), (7, 9),                       # left arm
    (6, 8), (8, 10),                      # right arm
    (5, 11), (6, 12),                     # torso
    (11, 12),                             # hips
    (11, 13), (13, 15),                   # left leg
    (12, 14), (14, 16),                   # right leg
]

# OpenPose-style limb colors (BGR for cv2)
LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]

JOINT_COLOR = (255, 255, 255)
CONFIDENCE_THRESHOLD = 0.3

# ── Model singleton ──

_model = None


def _ensure_cuda_libs():
    """Command, general. Add nvidia pip package lib dirs to LD_LIBRARY_PATH for onnxruntime CUDA."""
    import os
    try:
        import nvidia
        nv_dir = os.path.dirname(nvidia.__file__)
        paths = []
        for d in os.listdir(nv_dir):
            sublib = os.path.join(nv_dir, d, "lib")
            if os.path.isdir(sublib):
                paths.append(sublib)
        if paths:
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            new_paths = ":".join(paths)
            if new_paths not in existing:
                os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{existing}" if existing else new_paths
                # Also add to ctypes search path for already-loaded process
                import ctypes
                for p in paths:
                    try:
                        ctypes.CDLL(os.path.join(p, "libcublasLt.so.12"))
                    except OSError:
                        pass
    except ImportError:
        pass


def _get_model(mode: str = "balanced", backend: str = "onnxruntime", device: str = "cuda"):
    """Query (loads model on first call), general. Get or create the DWPose model."""
    global _model
    if _model is None:
        _ensure_cuda_libs()
        from rtmlib import Wholebody
        _model = Wholebody(
            to_openpose=False,
            mode=mode,
            backend=backend,
            device=device,
        )
    return _model


# ── Core functions ──


def detect_poses(
    image: Annotated[np.ndarray, Doc("BGR uint8 image, shape (H, W, 3)")],
    confidence_threshold: Annotated[float, Doc("Min confidence to count a person as detected")] = CONFIDENCE_THRESHOLD,
) -> Annotated[tuple[np.ndarray, np.ndarray], Doc("(keypoints (N, 133, 2), scores (N, 133))")]:
    """
    Query (runs model), general. Detect all people and their poses in one image.

    Returns keypoints and confidence scores for each detected person.
    Keypoints are in pixel coordinates. Scores are 0-1 per keypoint.

    Examples:
        >>> import numpy as np
        >>> # Black image — no people expected
        >>> img = np.zeros((256, 256, 3), dtype=np.uint8)
        >>> kpts, scores = detect_poses(img)
        >>> kpts.shape[1:] == (133, 2)
        True
        >>> scores.shape[1] == 133
        True
    """
    model = _get_model()
    keypoints, scores = model(image)
    if keypoints is None or len(keypoints) == 0:
        return np.empty((0, N_KEYPOINTS, 2), dtype=np.float32), np.empty((0, N_KEYPOINTS), dtype=np.float32)
    return np.asarray(keypoints, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def detect_poses_batch(
    frames: Annotated[list[np.ndarray], Doc("List of BGR uint8 frames")],
) -> Annotated[tuple[list[np.ndarray], list[np.ndarray]], Doc("(list of keypoints, list of scores) per frame")]:
    """
    Query (runs model), general. Detect poses in a batch of frames.

    Processes frame-by-frame (rtmlib doesn't support native batching).
    Returns per-frame results as lists.

    Examples:
        >>> import numpy as np
        >>> frames = [np.zeros((128, 128, 3), dtype=np.uint8)] * 3
        >>> kpts_list, scores_list = detect_poses_batch(frames)
        >>> len(kpts_list) == 3
        True
    """
    all_kpts = []
    all_scores = []
    for frame in frames:
        kpts, scores = detect_poses(frame)
        all_kpts.append(kpts)
        all_scores.append(scores)
    return all_kpts, all_scores


def draw_poses_on_black(
    keypoints: Annotated[np.ndarray, Doc("(N_people, 133, 2) keypoint coordinates")],
    scores: Annotated[np.ndarray, Doc("(N_people, 133) confidence scores")],
    height: Annotated[int, Doc("Output image height")],
    width: Annotated[int, Doc("Output image width")],
    threshold: Annotated[float, Doc("Min confidence to draw a keypoint")] = CONFIDENCE_THRESHOLD,
    draw_face: Annotated[bool, Doc("Draw 68 face landmark points")] = True,
    draw_hands: Annotated[bool, Doc("Draw 42 hand landmark points")] = True,
    line_thickness: Annotated[int, Doc("Skeleton line thickness")] = 2,
    joint_radius: Annotated[int, Doc("Joint circle radius")] = 3,
) -> Annotated[np.ndarray, Doc("BGR uint8 image with skeletons on black background")]:
    """
    Pure function, general. Draw whole-body pose on a black canvas.

    Draws body skeleton (17 keypoints, colored limbs), face landmarks (68 cyan
    dots), and hand landmarks (42 magenta dots). Face/hand only drawn if their
    mean confidence exceeds threshold (avoids garbage points from occluded parts).

    Examples:
        >>> import numpy as np
        >>> kpts = np.zeros((1, 133, 2), dtype=np.float32)
        >>> scores = np.ones((1, 133), dtype=np.float32)
        >>> viz = draw_poses_on_black(kpts, scores, 256, 256)
        >>> viz.shape == (256, 256, 3)
        True
        >>> viz.dtype == np.uint8
        True
    """
    import cv2

    face_start = N_BODY + N_FEET  # 23
    face_end = face_start + N_FACE  # 91
    hand_l_start = face_end  # 91
    hand_l_end = hand_l_start + 21  # 112
    hand_r_start = hand_l_end  # 112
    hand_r_end = hand_r_start + 21  # 133

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for person_idx in range(len(keypoints)):
        kpts = keypoints[person_idx]
        conf = scores[person_idx]

        # Body skeleton edges
        for edge_idx, (i, j) in enumerate(BODY_SKELETON):
            if conf[i] >= threshold and conf[j] >= threshold:
                pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                color = LIMB_COLORS[edge_idx % len(LIMB_COLORS)]
                cv2.line(canvas, pt1, pt2, color, line_thickness)

        # Body joint circles
        for i in range(N_BODY):
            if conf[i] >= threshold:
                pt = (int(kpts[i, 0]), int(kpts[i, 1]))
                cv2.circle(canvas, pt, joint_radius, JOINT_COLOR, -1)

        # Face landmarks (cyan dots) — only if mean face confidence is good
        if draw_face and conf[face_start:face_end].mean() >= threshold:
            for i in range(face_start, face_end):
                if conf[i] >= threshold:
                    pt = (int(kpts[i, 0]), int(kpts[i, 1]))
                    cv2.circle(canvas, pt, max(1, joint_radius // 2), (255, 255, 0), -1)

        # Hand landmarks (magenta dots) — only if mean hand confidence is good
        if draw_hands:
            for start, end in [(hand_l_start, hand_l_end), (hand_r_start, hand_r_end)]:
                if conf[start:end].mean() >= threshold:
                    for i in range(start, end):
                        if conf[i] >= threshold:
                            pt = (int(kpts[i, 0]), int(kpts[i, 1]))
                            cv2.circle(canvas, pt, max(1, joint_radius // 2), (255, 0, 255), -1)

    return canvas


def draw_poses_overlay(
    keypoints: Annotated[np.ndarray, Doc("(N_people, 133, 2) keypoint coordinates")],
    scores: Annotated[np.ndarray, Doc("(N_people, 133) confidence scores")],
    background: Annotated[np.ndarray, Doc("BGR uint8 source frame to overlay on")],
    threshold: Annotated[float, Doc("Min confidence to draw")] = CONFIDENCE_THRESHOLD,
    blend: Annotated[float, Doc("Skeleton opacity (0=invisible, 1=opaque)")] = 0.7,
    **kwargs,
) -> Annotated[np.ndarray, Doc("BGR uint8 image with skeleton overlaid on source")]:
    """
    Pure function, general. Draw skeleton overlaid on source frame.

    Examples:
        >>> import numpy as np
        >>> bg = np.zeros((256, 256, 3), dtype=np.uint8)
        >>> kpts = np.zeros((1, 133, 2), dtype=np.float32)
        >>> scores = np.ones((1, 133), dtype=np.float32)
        >>> viz = draw_poses_overlay(kpts, scores, bg)
        >>> viz.shape == (256, 256, 3)
        True
    """
    import cv2
    h, w = background.shape[:2]
    skel = draw_poses_on_black(keypoints, scores, h, w, threshold=threshold, **kwargs)
    result = background.copy()
    mask = skel.sum(axis=2) > 0
    result[mask] = cv2.addWeighted(background, 1.0 - blend, skel, blend, 0)[mask]
    return result


def compute_pose_stats(
    kpts_per_frame: Annotated[list[np.ndarray], Doc("Per-frame keypoints from detect_poses_batch")],
    scores_per_frame: Annotated[list[np.ndarray], Doc("Per-frame scores from detect_poses_batch")],
    threshold: Annotated[float, Doc("Min mean body confidence to count as a person")] = CONFIDENCE_THRESHOLD,
) -> Annotated[dict, Doc("Stats dict: max_seen_people, first_human_frame, last_human_frame")]:
    """
    Pure function, general. Compute pose statistics across video frames.

    A person is "detected" if their mean body keypoint confidence exceeds threshold.

    Examples:
        >>> import numpy as np
        >>> # No people in any frame
        >>> empty = [np.empty((0, 133, 2), dtype=np.float32)] * 5
        >>> scores = [np.empty((0, 133), dtype=np.float32)] * 5
        >>> stats = compute_pose_stats(empty, scores)
        >>> stats['max_seen_people']
        0
        >>> stats['first_human_frame']
        -1
    """
    people_per_frame = []
    for kpts, scores in zip(kpts_per_frame, scores_per_frame):
        if len(kpts) == 0:
            people_per_frame.append(0)
            continue
        # Count people with mean body confidence above threshold
        body_conf = scores[:, :N_BODY].mean(axis=1)
        n_people = int((body_conf >= threshold).sum())
        people_per_frame.append(n_people)

    max_seen = max(people_per_frame) if people_per_frame else 0
    human_frames = [i for i, n in enumerate(people_per_frame) if n > 0]

    return {
        "max_seen_people": max_seen,
        "first_human_frame": human_frames[0] if human_frames else -1,
        "last_human_frame": human_frames[-1] if human_frames else -1,
    }


def keypoints_to_numpy(
    kpts_per_frame: Annotated[list[np.ndarray], Doc("Per-frame (N_i, 133, 2) arrays")],
    scores_per_frame: Annotated[list[np.ndarray], Doc("Per-frame (N_i, 133) arrays")],
) -> Annotated[np.ndarray, Doc("(T, P, 133, 3) float32 array, NaN-padded")]:
    """
    Pure function, general. Stack per-frame pose results into a dense (T, P, 133, 3) array.

    P = max people across all frames. Frames with fewer people are NaN-padded.
    Channel 3 = (x, y, confidence).

    Examples:
        >>> import numpy as np
        >>> kpts = [np.array([[[10, 20]] * 133], dtype=np.float32)]
        >>> scores = [np.array([[0.9] * 133], dtype=np.float32)]
        >>> arr = keypoints_to_numpy(kpts, scores)
        >>> arr.shape
        (1, 1, 133, 3)
        >>> arr[0, 0, 0, :2].tolist()
        [10.0, 20.0]
    """
    T = len(kpts_per_frame)
    P = max((len(k) for k in kpts_per_frame), default=0)
    if P == 0:
        return np.full((T, 0, N_KEYPOINTS, 3), np.nan, dtype=np.float32)

    result = np.full((T, P, N_KEYPOINTS, 3), np.nan, dtype=np.float32)
    for t, (kpts, scores) in enumerate(zip(kpts_per_frame, scores_per_frame)):
        n = len(kpts)
        if n > 0:
            result[t, :n, :, :2] = kpts
            result[t, :n, :, 2] = scores
    return result


def demo():
    """Command, general. Quick demo: detect poses in a test image and show."""
    import cv2
    print("DWPose demo — detecting poses on a test image...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "DWPose Test", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    kpts, scores = detect_poses(img)
    print(f"Detected {len(kpts)} people")
    print(f"Keypoints shape: {kpts.shape}")
    viz = draw_poses_on_black(kpts, scores, 480, 640)
    print(f"Visualization shape: {viz.shape}")
    stats = compute_pose_stats([kpts], [scores])
    print(f"Stats: {stats}")
    return kpts, scores, viz, stats


if __name__ == "__main__":
    demo()
