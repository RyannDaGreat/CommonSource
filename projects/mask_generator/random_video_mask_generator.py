"""
Ryan Burgert 2025
Random video mask generator for video-inpainting training.

This module generates synthetic binary (0/255) mask videos. Masks are built up
by activating random shape plugins frame-by-frame; the output gets occasional
warp/persistence/zoom/pulse post-processing.

Scale-aware: works at any (T, H, W) — every velocity, thickness, and size is
expressed as a fraction of `scale = (W + H) / 2` so visual character is preserved
from small (T=25, H=60, W=90) up to large (T=200, H=480, W=720) and beyond.

Public API:
    get_random_video_mask(T=25, H=60, W=90)  -> uint8 (T, H, W) mask video
    demo(output_folder=...)                  -> saves demo grids
"""

__all__ = ['get_random_video_mask', 'demo']

from collections import deque
from fractions import Fraction
import math
import random

import cv2
import numpy as np

import rp


# =============================================================================
# Core scale helpers
# =============================================================================

def _scale(frame_width, frame_height):
    """
    Pure function. Returns the reference scale used for all velocity/size math.

    Args:
        frame_width  (int): Frame width in pixels.
        frame_height (int): Frame height in pixels.

    Returns:
        float: (frame_width + frame_height) / 2

    Examples:
        >>> _scale(90, 60)
        75.0
        >>> _scale(360, 240)
        300.0
    """
    return (frame_width + frame_height) / 2.0


def _rand_delta(scale, proportion=0.03):
    """
    Pure function. Returns a random signed velocity scaled to the frame.

    Velocity ≈ ±proportion * scale, so motion looks the same regardless of resolution.

    Args:
        scale      (float): Reference scale from _scale().
        proportion (float): Fraction of scale for max speed.

    Returns:
        float

    Examples:
        >>> random.seed(0); abs(_rand_delta(75.0)) <= 0.03 * 75.0
        True
    """
    max_delta = proportion * scale
    return random.uniform(-max_delta, max_delta)


def _ease_in_out_cubic(t):
    """
    Pure function. Cubic ease-in-out for smooth animation interpolation.

    Args:
        t (float): Input in [0, 1].

    Returns:
        float: Eased value in [0, 1].

    Examples:
        >>> _ease_in_out_cubic(0.0)
        0.0
        >>> _ease_in_out_cubic(1.0)
        1.0
        >>> round(_ease_in_out_cubic(0.5), 6)
        0.5
    """
    return (4 * t * t * t) if t < 0.5 else (1 - pow(-2 * t + 2, 3) / 2)


# =============================================================================
# Base plugin class
# =============================================================================

class _ShapePlugin:
    """Base class for shape plugins. Subclasses override draw + randomize."""

    def __init__(self, name):
        self.name = name
        self.hyperparameters = {}
        self.duration = 1
        self.remaining_duration = 0
        self.is_animated = False

    def draw(self, frame, frame_width, frame_height):
        """Command. Subclasses must implement. Mutates frame."""
        raise NotImplementedError("Subclasses must implement draw.")

    def randomize_hyperparameters(self, frame_width, frame_height):
        """Command. Default no-op. Subclasses set self.hyperparameters here."""
        pass

    def update_hyperparameters(self, frame_width, frame_height):
        """Command. Called per frame to advance animation state."""
        if self.is_animated:
            self._animate_movement(frame_width, frame_height)

    def start_drawing(self, frame_width, frame_height, T=25):
        """
        Command. Called when the plugin enters the active set. Mutates self.

        Args:
            frame_width, frame_height: pixels.
            T (int): total number of video frames (for scaling duration).
        """
        self.randomize_hyperparameters(frame_width, frame_height)
        if random.random() < 0.2:
            self.duration = random.randint(2, max(3, T // 5))
        else:
            self.duration = 1
        self.remaining_duration = self.duration
        self.is_animated = True

    def is_active(self):
        """Query."""
        return self.remaining_duration > 0

    def decrement_duration(self):
        """Command."""
        self.remaining_duration -= 1

    def _animate_movement(self, frame_width, frame_height):
        """Command. Default bouncing logic for animated center+size plugins."""
        sc = _scale(frame_width, frame_height)
        min_size = 0.02 * sc

        for param, delta_param in self._get_animated_parameters():
            self.hyperparameters[param] += self.hyperparameters[delta_param]

            lower_bound = 0
            upper_bound = frame_width if "x" in param else frame_height
            size_param = None
            if "width" in self.hyperparameters and param in ("center_x", "center_y"):
                size_param = "width" if "x" in param else "height"

            if size_param:
                lower_bound = int(self.hyperparameters[size_param] / 2)
                upper_bound -= int(self.hyperparameters[size_param] / 2)

            if (self.hyperparameters[param] < lower_bound
                    or self.hyperparameters[param] > upper_bound):
                self.hyperparameters[delta_param] *= -1

            if "width" in param or "height" in param:
                if (self.hyperparameters[param] < min_size
                        or self.hyperparameters[param] > frame_width) and "width" in param:
                    self.hyperparameters[delta_param] *= -1
                if (self.hyperparameters[param] < min_size
                        or self.hyperparameters[param] > frame_height) and "height" in param:
                    self.hyperparameters[delta_param] *= -1

    def _get_animated_parameters(self):
        """Query. Returns list of (param, delta_param) tuples. Subclasses override."""
        return []


# =============================================================================
# Refactored "classic" plugins (scale-aware versions of the originals)
# =============================================================================

class _RectanglePlugin(_ShapePlugin):
    """
    Command. Draws a filled rectangle that bounces around the frame. Mutates frame.
    Velocities scale as ±0.03 * scale pixels/frame.
    """

    def __init__(self):
        super().__init__("rectangle")
        self.hyperparameters = {
            "center_x": 0, "center_y": 0, "width": 0, "height": 0,
            "delta_x": 0, "delta_y": 0, "delta_width": 0, "delta_height": 0,
        }

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        x1 = max(0, int(h["center_x"] - h["width"] / 2))
        y1 = max(0, int(h["center_y"] - h["height"] / 2))
        x2 = min(frame_width, int(h["center_x"] + h["width"] / 2))
        y2 = min(frame_height, int(h["center_y"] + h["height"] / 2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), 255, -1)
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        h = self.hyperparameters
        h["center_x"] = random.randint(0, frame_width - 1)
        h["center_y"] = random.randint(0, frame_height - 1)
        h["width"] = int(random.random() * frame_width)
        h["height"] = int(random.random() * frame_height)
        h["delta_x"] = _rand_delta(sc, 0.03)
        h["delta_y"] = _rand_delta(sc, 0.03)
        h["delta_width"] = _rand_delta(sc, 0.03)
        h["delta_height"] = _rand_delta(sc, 0.03)

    def _get_animated_parameters(self):
        return [("center_x", "delta_x"), ("center_y", "delta_y"),
                ("width", "delta_width"), ("height", "delta_height")]


class _EllipsePlugin(_ShapePlugin):
    """
    Command. Draws a filled ellipse that moves, resizes, and rotates. Mutates frame.
    Position/size velocities ∝ scale; rotation velocity is dimensionless degrees/frame.
    """

    def __init__(self):
        super().__init__("ellipse")
        self.hyperparameters = {
            "center_x": 0, "center_y": 0, "width": 0, "height": 0, "angle": 0,
            "delta_x": 0, "delta_y": 0, "delta_width": 0, "delta_height": 0, "delta_angle": 0,
        }

    def draw(self, frame, frame_width, frame_height):
        # Animation deltas can drive width/height negative; OpenCV's ellipse
        # asserts axes >= 0, so clamp them to a valid range.
        axes = (
            max(0, int(self.hyperparameters["width"] / 2)),
            max(0, int(self.hyperparameters["height"] / 2)),
        )
        cv2.ellipse(
            frame,
            (
                int(self.hyperparameters["center_x"]),
                int(self.hyperparameters["center_y"]),
            ),
            axes,
            int(self.hyperparameters["angle"]),
            0,
            360,
            (255, 255, 255),
            -1,
        )
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        h = self.hyperparameters
        h["center_x"] = random.randint(0, frame_width - 1)
        h["center_y"] = random.randint(0, frame_height - 1)
        h["width"] = int(random.random() * frame_width)
        h["height"] = int(random.random() * frame_height)
        h["angle"] = random.randint(0, 359)
        h["delta_x"] = _rand_delta(sc, 0.03)
        h["delta_y"] = _rand_delta(sc, 0.03)
        h["delta_width"] = _rand_delta(sc, 0.03)
        h["delta_height"] = _rand_delta(sc, 0.03)
        h["delta_angle"] = random.randint(-15, 15)

    def _get_animated_parameters(self):
        return [("center_x", "delta_x"), ("center_y", "delta_y"),
                ("width", "delta_width"), ("height", "delta_height"),
                ("angle", "delta_angle")]

    def _animate_movement(self, frame_width, frame_height):
        super()._animate_movement(frame_width, frame_height)
        self.hyperparameters["angle"] %= 360


class _ScribblePlugin(_ShapePlugin):
    """
    Command. Polyline scribble whose points dance. Mutates frame.
    Thickness ∝ scale; point velocities ∝ scale.
    """

    def __init__(self, num_points_range=(3, 13), max_points=50):
        super().__init__("scribble")
        self.num_points_range = num_points_range
        self.max_points = max_points
        self.hyperparameters = {
            "center_x": 0, "center_y": 0, "width": 0, "height": 0,
            "num_points": 0, "thickness": 0, "points": [], "delta_points": [],
        }

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        pts = np.array(h["points"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=255,
                      thickness=max(1, int(h["thickness"])))
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        h = self.hyperparameters
        h["center_x"] = random.randint(0, frame_width - 1)
        h["center_y"] = random.randint(0, frame_height - 1)
        h["width"] = int(random.random() * frame_width)
        h["height"] = int(random.random() * frame_height)
        h["num_points"] = min(random.randint(*self.num_points_range), self.max_points)
        h["thickness"] = random.uniform(max(1, 0.005 * sc), max(2, 0.05 * sc))
        h["points"] = []
        h["delta_points"] = []
        max_dp = 0.04 * sc

        for _ in range(h["num_points"]):
            x = random.randint(max(0, h["center_x"] - h["width"]),
                               min(frame_width - 1, h["center_x"] + h["width"]))
            y = random.randint(max(0, h["center_y"] - h["height"]),
                               min(frame_height - 1, h["center_y"] + h["height"]))
            h["points"].append((x, y))
            h["delta_points"].append((random.uniform(-max_dp, max_dp),
                                       random.uniform(-max_dp, max_dp)))

    def _animate_movement(self, frame_width, frame_height):
        h = self.hyperparameters
        updated = []
        for i, (x, y) in enumerate(h["points"]):
            dx, dy = h["delta_points"][i]
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= frame_width:
                dx *= -1
                nx = max(0, min(frame_width - 1, nx))
            if ny < 0 or ny >= frame_height:
                dy *= -1
                ny = max(0, min(frame_height - 1, ny))
            h["delta_points"][i] = (dx, dy)
            updated.append((nx, ny))
        h["points"] = updated


class _TrianglePlugin(_ShapePlugin):
    """
    Command. Equilateral triangle that moves and rotates. Mutates frame.
    Position velocities ∝ scale; rotation is dimensionless deg/frame.
    """

    def __init__(self):
        super().__init__("triangle")
        self.hyperparameters = {
            "center_x": 0, "center_y": 0, "size": 0, "angle": 0,
            "delta_x": 0, "delta_y": 0, "delta_size": 0, "delta_angle": 0,
        }

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        s = h["size"]
        cx, cy = h["center_x"], h["center_y"]
        a = math.radians(h["angle"])
        pts = np.array([
            [int(cx + s * math.cos(a)), int(cy + s * math.sin(a))],
            [int(cx + s * math.cos(a + 2 * math.pi / 3)), int(cy + s * math.sin(a + 2 * math.pi / 3))],
            [int(cx + s * math.cos(a + 4 * math.pi / 3)), int(cy + s * math.sin(a + 4 * math.pi / 3))],
        ], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], 255)
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        h = self.hyperparameters
        h["center_x"] = random.randint(0, frame_width - 1)
        h["center_y"] = random.randint(0, frame_height - 1)
        h["size"] = int(random.random() * min(frame_width, frame_height) / 2)
        h["angle"] = random.randint(0, 359)
        h["delta_x"] = _rand_delta(sc, 0.03)
        h["delta_y"] = _rand_delta(sc, 0.03)
        h["delta_size"] = _rand_delta(sc, 0.015)
        h["delta_angle"] = random.randint(-10, 10)

    def _get_animated_parameters(self):
        return [("center_x", "delta_x"), ("center_y", "delta_y"),
                ("size", "delta_size"), ("angle", "delta_angle")]

    def _animate_movement(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        min_size = max(1, 0.02 * sc)
        super()._animate_movement(frame_width, frame_height)
        h = self.hyperparameters
        h["angle"] %= 360
        h["size"] = max(min_size, min(h["size"], min(frame_width, frame_height) / 2))


class _SaltPlugin(_ShapePlugin):
    """Command. Random white salt pixels. Resolution-independent (amount = fraction of pixels)."""

    def __init__(self, amount=0.05):
        super().__init__("salt")
        self.amount = amount

    def draw(self, frame, frame_width, frame_height):
        n = frame_width * frame_height
        k = int(n * self.amount)
        coords = [random.randint(0, n - 1) for _ in range(k)]
        for c in coords:
            frame[c // frame_width, c % frame_width] = 255
        return frame


class _BoopySaltPlugin(_ShapePlugin):
    """
    Command. Swarm of small bouncing circles. Mutates frame.
    Radii and velocities ∝ scale.
    """

    def __init__(self, num_dots_range=(0, 50)):
        super().__init__("boopy_salt")
        self.num_dots_range = num_dots_range
        self.hyperparameters = {"dots": []}

    def draw(self, frame, frame_width, frame_height):
        for d in self.hyperparameters["dots"]:
            cv2.circle(frame, (int(d["center_x"]), int(d["center_y"])),
                       max(1, int(d["radius"])), 255, -1)
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        r_min = max(1, 0.005 * sc)
        r_max = max(2, 0.05 * sc)
        v_max = 0.015 * sc
        n = random.randint(*self.num_dots_range)
        self.hyperparameters["dots"] = [
            {"center_x": random.randint(0, frame_width - 1),
             "center_y": random.randint(0, frame_height - 1),
             "radius": random.uniform(r_min, r_max),
             "delta_x": random.uniform(-v_max, v_max),
             "delta_y": random.uniform(-v_max, v_max)}
            for _ in range(n)
        ]

    def _animate_movement(self, frame_width, frame_height):
        for d in self.hyperparameters["dots"]:
            d["center_x"] += d["delta_x"]
            d["center_y"] += d["delta_y"]
            r = d["radius"]
            if d["center_x"] < r or d["center_x"] > frame_width - 1 - r:
                d["delta_x"] *= -1
                d["center_x"] = max(r, min(frame_width - 1 - r, d["center_x"]))
            if d["center_y"] < r or d["center_y"] > frame_height - 1 - r:
                d["delta_y"] *= -1
                d["center_y"] = max(r, min(frame_height - 1 - r, d["center_y"]))


# =============================================================================
# NEW PLUGIN: 3D plane rotation (agent 2)
# =============================================================================

def _rotation_matrix_x(angle_rad):
    """Pure function. 3x3 rotation matrix around X axis.

    Examples:
        >>> np.allclose(_rotation_matrix_x(0.0), np.eye(3))
        True
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_matrix_y(angle_rad):
    """Pure function. 3x3 rotation matrix around Y axis.

    Examples:
        >>> np.allclose(_rotation_matrix_y(0.0), np.eye(3))
        True
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_matrix_z(angle_rad):
    """Pure function. 3x3 rotation matrix around Z axis.

    Examples:
        >>> np.allclose(_rotation_matrix_z(0.0), np.eye(3))
        True
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _ngon_vertices_3d(n, radius):
    """
    Pure function. Regular n-gon in z=0 plane. (n, 3) float64 [x, y, 0].

    Examples:
        >>> v = _ngon_vertices_3d(4, 1.0)
        >>> v.shape
        (4, 3)
        >>> np.allclose(v[:, 2], 0.0)
        True
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)], axis=1)


def _project_perspective(vertices_3d, focal, center_x, center_y):
    """
    Pure function. Project (n, 3) 3D vertices to (n, 2) int32 screen pixels.
    Uses screen = focal * world / (focal + z). Denominator clamped to avoid flips.

    Examples:
        >>> p = _project_perspective(np.array([[10., 5., 0.]]), focal=100., center_x=45., center_y=30.)
        >>> p.shape
        (1, 2)
    """
    xs, ys, zs = vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2]
    denom = np.maximum(focal + zs, focal * 0.1)
    sx = focal * xs / denom + center_x
    sy = focal * ys / denom + center_y
    return np.stack([sx, sy], axis=1).astype(np.int32)


class _3DPlanePlugin(_ShapePlugin):
    """
    Command. Flat polygon (3..7 vertices) tumbling in 3D, perspective-projected. Mutates frame.
    Sizes proportional to min(W, H); angular velocities scale-proportional.
    """

    _SPIN_MODES = ("dominant_x", "dominant_y", "dominant_z", "chaotic")

    def __init__(self):
        super().__init__("3d_plane")

    def randomize_hyperparameters(self, frame_width, frame_height):
        scale = min(frame_width, frame_height)
        spin = random.choice(self._SPIN_MODES)
        base_d = (scale / 60)

        if spin == "dominant_x":
            dx, dy, dz = random.uniform(0.06, 0.14) * base_d, random.uniform(0, 0.015) * base_d, random.uniform(0, 0.015) * base_d
        elif spin == "dominant_y":
            dx, dy, dz = random.uniform(0, 0.015) * base_d, random.uniform(0.06, 0.14) * base_d, random.uniform(0, 0.015) * base_d
        elif spin == "dominant_z":
            dx, dy, dz = random.uniform(0, 0.015) * base_d, random.uniform(0, 0.015) * base_d, random.uniform(0.06, 0.14) * base_d
        else:
            dx = random.uniform(0.03, 0.12) * base_d * random.choice([-1, 1])
            dy = random.uniform(0.03, 0.12) * base_d * random.choice([-1, 1])
            dz = random.uniform(0.03, 0.12) * base_d * random.choice([-1, 1])

        radius = scale * random.uniform(0.15, 0.50)
        self.hyperparameters = {
            "center_x": random.uniform(frame_width * 0.1, frame_width * 0.9),
            "center_y": random.uniform(frame_height * 0.1, frame_height * 0.9),
            "radius": radius,
            "angle_x": random.uniform(0, 2 * np.pi),
            "angle_y": random.uniform(0, 2 * np.pi),
            "angle_z": random.uniform(0, 2 * np.pi),
            "delta_x": dx, "delta_y": dy, "delta_z": dz,
            "n_vertices": random.choice([3, 4, 5, 6, 7]),
            "focal": radius * random.uniform(1.5, 3.0),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        h["angle_x"] = (h["angle_x"] + h["delta_x"]) % (2 * np.pi)
        h["angle_y"] = (h["angle_y"] + h["delta_y"]) % (2 * np.pi)
        h["angle_z"] = (h["angle_z"] + h["delta_z"]) % (2 * np.pi)

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        verts = _ngon_vertices_3d(h["n_vertices"], h["radius"])
        R = _rotation_matrix_x(h["angle_x"]) @ _rotation_matrix_y(h["angle_y"]) @ _rotation_matrix_z(h["angle_z"])
        rotated = verts @ R.T
        screen = _project_perspective(rotated, h["focal"], h["center_x"], h["center_y"])
        cv2.fillPoly(frame, [screen.reshape((-1, 1, 2))], 255)
        return frame


# =============================================================================
# NEW PLUGIN: Parametric curves (agent 3)
# =============================================================================

def _gcd_period(R, r):
    """
    Pure function. Period multiplier (radians) for a hypocycloid/epicycloid
    with radii R, r, approximating R/r as a small rational.

    Examples:
        >>> abs(_gcd_period(1.0, 0.25) - 2 * math.pi) < 1e-6
        True
    """
    try:
        frac = Fraction(R / r).limit_denominator(20)
        return 2 * np.pi * frac.denominator
    except Exception:
        return 2 * np.pi


def _spiral_points(a, b, t_max, n_points):
    """
    Pure function. Archimedean spiral r(t)=a+b*t, t in [0, t_max]. Returns (n, 2) float32.

    Examples:
        >>> _spiral_points(0, 1, 2*math.pi, 10).shape
        (10, 2)
    """
    t = np.linspace(0.0, t_max, n_points, dtype=np.float32)
    r = a + b * t
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _lissajous_points(A, B, p, q, phi, n_points):
    """
    Pure function. Lissajous x=A sin(p t+phi), y=B sin(q t). Returns (n, 2) float32.

    Examples:
        >>> _lissajous_points(1, 1, 3, 2, 0, 200).shape
        (200, 2)
    """
    t = np.linspace(0.0, 2 * np.pi, n_points, dtype=np.float32)
    return np.stack([A * np.sin(p * t + phi), B * np.sin(q * t)], axis=1)


def _rose_points(radius, k, n_points):
    """
    Pure function. Rose r = radius * cos(k * theta). Returns (n, 2) float32.

    Examples:
        >>> _rose_points(1.0, 3, 200).shape
        (200, 2)
    """
    theta = np.linspace(0.0, 4 * np.pi, n_points, dtype=np.float32)
    r = radius * np.cos(k * theta)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def _hypocycloid_points(R, r, n_points):
    """
    Pure function. Hypocycloid. (n, 2) float32. 0 < r < R.

    Examples:
        >>> _hypocycloid_points(1.0, 0.25, 200).shape
        (200, 2)
    """
    r = max(r, 1e-4)
    t = np.linspace(0.0, _gcd_period(R, r), n_points, dtype=np.float32)
    ratio = (R - r) / r
    return np.stack([(R - r) * np.cos(t) + r * np.cos(ratio * t),
                     (R - r) * np.sin(t) - r * np.sin(ratio * t)], axis=1)


def _epicycloid_points(R, r, n_points):
    """
    Pure function. Epicycloid. (n, 2) float32. r > 0.

    Examples:
        >>> _epicycloid_points(1.0, 0.3, 200).shape
        (200, 2)
    """
    r = max(r, 1e-4)
    t = np.linspace(0.0, 2 * np.pi, n_points, dtype=np.float32)
    ratio = (R + r) / r
    return np.stack([(R + r) * np.cos(t) - r * np.cos(ratio * t),
                     (R + r) * np.sin(t) - r * np.sin(ratio * t)], axis=1)


def _points_to_polyline(points, center_x, center_y, scale_factor):
    """
    Pure function. Convert normalised (N,2) float points to (N,1,2) int32 pixel coords.

    Examples:
        >>> _points_to_polyline(np.array([[0.,0.],[1.,0.]], dtype=np.float32), 45, 30, 20.).shape
        (2, 1, 2)
    """
    scaled = points * scale_factor
    px = (scaled[:, 0] + center_x).astype(np.int32)
    py = (scaled[:, 1] + center_y).astype(np.int32)
    return np.stack([px, py], axis=1).reshape(-1, 1, 2)


class _ParametricCurvePlugin(_ShapePlugin):
    """
    Command. Animated parametric curves: spiral, lissajous, rose, hypocycloid, epicycloid.
    Mutates frame. All sizes/velocities ∝ min(W, H) * 0.4 ("inscribed radius").
    """

    _TYPES = ("spiral", "lissajous", "rose", "hypocycloid", "epicycloid")

    def __init__(self):
        super().__init__("parametric_curve")

    def randomize_hyperparameters(self, frame_width, frame_height):
        base_scale = min(frame_width, frame_height)
        base_radius = base_scale * 0.4
        drift_speed = base_scale * random.uniform(0.005, 0.02)
        angle = random.uniform(0, 2 * np.pi)

        self.hyperparameters = {
            "type": random.choice(self._TYPES),
            "n_points": random.randint(100, 300),
            "thickness": max(1, int(base_scale * random.uniform(0.01, 0.04))),
            "cx": random.uniform(0, frame_width),
            "cy": random.uniform(0, frame_height),
            "dcx": drift_speed * np.cos(angle),
            "dcy": drift_speed * np.sin(angle),
            "scale_px": base_radius,
        }
        h = self.hyperparameters
        t = h["type"]

        if t == "spiral":
            h.update(sp_a=random.uniform(0.0, 0.3), sp_b=random.uniform(0.05, 0.25),
                     sp_t_max=random.uniform(2, 5) * np.pi, sp_phase=random.uniform(0, 2 * np.pi),
                     sp_da=random.uniform(-0.005, 0.005), sp_db=random.uniform(-0.003, 0.003),
                     sp_dt_max=random.uniform(-0.05, 0.1), sp_dphase=random.uniform(0.02, 0.1))
        elif t == "lissajous":
            h.update(li_A=random.uniform(0.5, 1.0), li_B=random.uniform(0.5, 1.0),
                     li_p=float(random.randint(1, 5)), li_q=float(random.randint(1, 5)),
                     li_phi=random.uniform(0, np.pi),
                     li_dp=random.uniform(-0.02, 0.02), li_dq=random.uniform(-0.02, 0.02),
                     li_dphi=random.uniform(0.01, 0.08))
        elif t == "rose":
            h.update(ro_k=float(random.randint(2, 7)), ro_dk=random.uniform(-0.03, 0.03),
                     ro_phase=random.uniform(0, 2 * np.pi), ro_dphase=random.uniform(0.02, 0.1))
        elif t == "hypocycloid":
            h.update(cy_R=1.0, cy_r=1.0 / random.randint(2, 5),
                     cy_dr=random.uniform(-0.005, 0.005),
                     cy_phase=random.uniform(0, 2 * np.pi), cy_dphase=random.uniform(0.01, 0.06))
        elif t == "epicycloid":
            R = 0.6
            h.update(cy_R=R, cy_r=R / random.randint(2, 5),
                     cy_dr=random.uniform(-0.003, 0.003),
                     cy_phase=random.uniform(0, 2 * np.pi), cy_dphase=random.uniform(0.01, 0.06))

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        margin = min(frame_width, frame_height) * 0.05
        h["cx"] += h["dcx"]; h["cy"] += h["dcy"]
        if h["cx"] < margin or h["cx"] > frame_width - margin:
            h["dcx"] *= -1
            h["cx"] = max(margin, min(frame_width - margin, h["cx"]))
        if h["cy"] < margin or h["cy"] > frame_height - margin:
            h["dcy"] *= -1
            h["cy"] = max(margin, min(frame_height - margin, h["cy"]))

        t = h["type"]
        if t == "spiral":
            h["sp_a"] = max(0.0, min(0.6, h["sp_a"] + h["sp_da"]))
            h["sp_b"] = max(0.02, min(0.4, h["sp_b"] + h["sp_db"]))
            h["sp_t_max"] = max(np.pi, min(8 * np.pi, h["sp_t_max"] + h["sp_dt_max"]))
            h["sp_phase"] = (h["sp_phase"] + h["sp_dphase"]) % (2 * np.pi)
            if h["sp_a"] <= 0.0 or h["sp_a"] >= 0.6: h["sp_da"] *= -1
            if h["sp_b"] <= 0.02 or h["sp_b"] >= 0.4: h["sp_db"] *= -1
            if h["sp_t_max"] <= np.pi or h["sp_t_max"] >= 8 * np.pi: h["sp_dt_max"] *= -1
        elif t == "lissajous":
            h["li_p"] = max(0.5, min(6.0, h["li_p"] + h["li_dp"]))
            h["li_q"] = max(0.5, min(6.0, h["li_q"] + h["li_dq"]))
            h["li_phi"] = (h["li_phi"] + h["li_dphi"]) % (2 * np.pi)
            if h["li_p"] <= 0.5 or h["li_p"] >= 6.0: h["li_dp"] *= -1
            if h["li_q"] <= 0.5 or h["li_q"] >= 6.0: h["li_dq"] *= -1
        elif t == "rose":
            h["ro_k"] = max(1.0, min(8.0, h["ro_k"] + h["ro_dk"]))
            h["ro_phase"] = (h["ro_phase"] + h["ro_dphase"]) % (2 * np.pi)
            if h["ro_k"] <= 1.0 or h["ro_k"] >= 8.0: h["ro_dk"] *= -1
        elif t in ("hypocycloid", "epicycloid"):
            r_min = 0.05
            r_max = h["cy_R"] * 0.9 if t == "hypocycloid" else h["cy_R"]
            h["cy_r"] = max(r_min, min(r_max, h["cy_r"] + h["cy_dr"]))
            h["cy_phase"] = (h["cy_phase"] + h["cy_dphase"]) % (2 * np.pi)
            if h["cy_r"] <= r_min or h["cy_r"] >= r_max: h["cy_dr"] *= -1

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        pts = self._compute_points()
        if pts is None or len(pts) < 2:
            return frame
        poly = _points_to_polyline(pts, h["cx"], h["cy"], h["scale_px"])
        cv2.polylines(frame, [poly], isClosed=False, color=255, thickness=h["thickness"])
        return frame

    def _compute_points(self):
        h = self.hyperparameters
        t = h["type"]
        if t == "spiral":
            pts = _spiral_points(h["sp_a"], h["sp_b"], h["sp_t_max"], h["n_points"])
            c, s = np.cos(h["sp_phase"]), np.sin(h["sp_phase"])
            pts = pts @ np.array([[c, -s], [s, c]], dtype=np.float32).T
            max_r = h["sp_a"] + h["sp_b"] * h["sp_t_max"]
            return pts / max_r if max_r > 1e-4 else pts
        if t == "lissajous":
            pts = _lissajous_points(h["li_A"], h["li_B"], h["li_p"], h["li_q"], h["li_phi"], h["n_points"])
            pts[:, 0] /= max(abs(h["li_A"]), 1e-4)
            pts[:, 1] /= max(abs(h["li_B"]), 1e-4)
            return pts
        if t == "rose":
            pts = _rose_points(1.0, h["ro_k"], h["n_points"])
            c, s = np.cos(h["ro_phase"]), np.sin(h["ro_phase"])
            return pts @ np.array([[c, -s], [s, c]], dtype=np.float32).T
        if t in ("hypocycloid", "epicycloid"):
            fn = _hypocycloid_points if t == "hypocycloid" else _epicycloid_points
            pts = fn(h["cy_R"], h["cy_r"], h["n_points"])
            m = max(np.abs(pts).max(), 1e-4)
            pts = pts / m
            c, s = np.cos(h["cy_phase"]), np.sin(h["cy_phase"])
            return pts @ np.array([[c, -s], [s, c]], dtype=np.float32).T
        return None


# =============================================================================
# NEW PLUGINS: Bezier swarm (agent 4)
# =============================================================================

def _sample_cubic_bezier(p0, p1, p2, p3, n_samples=75):
    """
    Pure function. Sample cubic Bezier B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t)t^2 P2 + t^3 P3.

    Examples:
        >>> _sample_cubic_bezier([0,0],[1,2],[2,2],[3,0], n_samples=3).shape
        (3, 2)
    """
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    t = np.linspace(0.0, 1.0, n_samples)[:, None]
    u = 1.0 - t
    return (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3


def _bounce_point(x, y, vx, vy, w, h):
    """
    Pure function. Advance (x,y) by (vx,vy) with wall reflection. Returns (x, y, vx, vy).
    """
    x += vx; y += vy
    if x < 0:
        x = -x; vx = abs(vx)
    elif x >= w:
        x = 2 * w - x - 2; vx = -abs(vx)
    if y < 0:
        y = -y; vy = abs(vy)
    elif y >= h:
        y = 2 * h - y - 2; vy = -abs(vy)
    return x, y, vx, vy


def _make_control_points(frame_width, frame_height, n_points, speed_scale):
    """Pure function. Returns list of n dicts {x, y, vx, vy} with speeds ≈ speed_scale."""
    out = []
    for _ in range(n_points):
        a = random.uniform(0, 2 * np.pi)
        s = random.uniform(0.5 * speed_scale, 1.5 * speed_scale)
        out.append({"x": random.uniform(0, frame_width - 1),
                    "y": random.uniform(0, frame_height - 1),
                    "vx": s * math.cos(a), "vy": s * math.sin(a)})
    return out


def _advance_control_points(cps, frame_width, frame_height):
    """Command. Advance all control points one step, bouncing in place."""
    for cp in cps:
        cp["x"], cp["y"], cp["vx"], cp["vy"] = _bounce_point(
            cp["x"], cp["y"], cp["vx"], cp["vy"], frame_width, frame_height
        )


def _draw_bezier_strip(frame, cps, thickness, closed=False):
    """Command. Draw cubic Bezier (or chained closed loop). Mutates frame."""
    pts = [(cp["x"], cp["y"]) for cp in cps]
    n = len(pts)
    if n < 4:
        return frame
    if closed:
        all_seg = [_sample_cubic_bezier(pts[i % n], pts[(i + 1) % n],
                                         pts[(i + 2) % n], pts[(i + 3) % n], 30)
                   for i in range(n)]
        curve = np.vstack(all_seg).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [curve], 255)
    else:
        curve = _sample_cubic_bezier(pts[0], pts[1], pts[2], pts[3], 80)
        poly = curve.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [poly], isClosed=False, color=255,
                      thickness=max(1, int(thickness)))
    return frame


class _BezierWormPlugin(_ShapePlugin):
    """Command. Fat cubic Bezier with 4 bouncing control points → wriggling worm."""

    def __init__(self):
        super().__init__("bezier_worm")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        self.hyperparameters = {
            "cps": _make_control_points(frame_width, frame_height, 4,
                                          random.uniform(0.02, 0.06) * sc),
            "thickness": max(1, int(random.uniform(0.03, 0.07) * sc)),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        _advance_control_points(self.hyperparameters["cps"], frame_width, frame_height)

    def draw(self, frame, frame_width, frame_height):
        return _draw_bezier_strip(frame, self.hyperparameters["cps"],
                                   self.hyperparameters["thickness"], closed=False)


class _BezierCalligraphyPlugin(_ShapePlugin):
    """Command. 3..6 thin Beziers, calligraphic flourishes."""

    def __init__(self):
        super().__init__("bezier_calligraphy")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        n = random.randint(3, 6)
        speed = random.uniform(0.015, 0.05) * sc
        thin = max(1, int(random.uniform(0.005, 0.015) * sc))
        self.hyperparameters = {
            "curves": [{"cps": _make_control_points(frame_width, frame_height, 4, speed),
                        "thickness": max(1, thin + random.randint(-1, 1))}
                       for _ in range(n)]
        }

    def update_hyperparameters(self, frame_width, frame_height):
        for c in self.hyperparameters["curves"]:
            _advance_control_points(c["cps"], frame_width, frame_height)

    def draw(self, frame, frame_width, frame_height):
        for c in self.hyperparameters["curves"]:
            _draw_bezier_strip(frame, c["cps"], c["thickness"], closed=False)
        return frame


class _BezierLoopPlugin(_ShapePlugin):
    """Command. Closed Bezier loop (5..6 cps) filled solid — amoeba/blob."""

    def __init__(self):
        super().__init__("bezier_loop")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        self.hyperparameters = {
            "cps": _make_control_points(frame_width, frame_height, random.choice([5, 6]),
                                          random.uniform(0.02, 0.055) * sc),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        _advance_control_points(self.hyperparameters["cps"], frame_width, frame_height)

    def draw(self, frame, frame_width, frame_height):
        return _draw_bezier_strip(frame, self.hyperparameters["cps"], thickness=1, closed=True)


# =============================================================================
# NEW PLUGINS: Polygon, star, gear (agent 5)
# =============================================================================

def _regular_polygon_vertices(n, radius, angle_offset=0.0):
    """Pure function. Regular n-gon vertices in local coords. (n, 2) float."""
    angles = [angle_offset + 2 * math.pi * k / n for k in range(n)]
    return np.array([[radius * math.cos(a), radius * math.sin(a)] for a in angles], dtype=np.float32)


def _star_vertices(n_points, outer, inner, angle_offset=0.0):
    """Pure function. n-pointed star alternating outer/inner radii. (2n, 2) float."""
    verts = []
    for k in range(2 * n_points):
        r = outer if k % 2 == 0 else inner
        a = angle_offset + math.pi * k / n_points
        verts.append([r * math.cos(a), r * math.sin(a)])
    return np.array(verts, dtype=np.float32)


def _gear_vertices(n_teeth, outer, inner, tooth_width_frac=0.5, angle_offset=0.0):
    """Pure function. Gear with rectangular teeth. (4n, 2) float."""
    verts = []
    sector = 2 * math.pi / n_teeth
    half_tooth = tooth_width_frac * sector / 2.0
    for k in range(n_teeth):
        base = angle_offset + k * sector
        a0 = base - sector / 2
        a1 = base - half_tooth
        a2 = base + half_tooth
        a3 = base + sector / 2
        verts.append([inner * math.cos(a0), inner * math.sin(a0)])
        verts.append([outer * math.cos(a1), outer * math.sin(a1)])
        verts.append([outer * math.cos(a2), outer * math.sin(a2)])
        verts.append([inner * math.cos(a3), inner * math.sin(a3)])
    return np.array(verts, dtype=np.float32)


def _to_cv2_points(local_verts, cx, cy):
    """Pure function. Translate local verts to frame coords; return (N,1,2) int32."""
    translated = local_verts + np.array([cx, cy], dtype=np.float32)
    return translated.reshape(-1, 1, 2).astype(np.int32)


class _PolygonPlugin(_ShapePlugin):
    """Command. Regular n-gon (n=3..12) that rotates and pulses sinusoidally."""

    def __init__(self):
        super().__init__("polygon")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        min_dim = min(frame_width, frame_height)
        self.hyperparameters = {
            "n_sides": random.randint(3, 12),
            "center_x": random.uniform(0, frame_width),
            "center_y": random.uniform(0, frame_height),
            "base_size": random.uniform(0.05 * min_dim, 0.35 * min_dim),
            "angle": random.uniform(0, 2 * math.pi),
            "phase": random.uniform(0, 1),
            "amplitude": random.uniform(0.1, 0.4),
            "rot_vel": random.uniform(-0.12, 0.12),
            "phase_vel": random.uniform(0.02, 0.08),
            "drift_x": random.uniform(-0.02 * sc, 0.02 * sc),
            "drift_y": random.uniform(-0.02 * sc, 0.02 * sc),
            "filled": random.random() < 0.6,
            "thickness": max(1, int(random.uniform(0.005 * sc, 0.04 * sc))),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        h["angle"] = (h["angle"] + h["rot_vel"]) % (2 * math.pi)
        h["phase"] = (h["phase"] + h["phase_vel"]) % 1.0
        h["center_x"] += h["drift_x"]; h["center_y"] += h["drift_y"]
        if h["center_x"] < 0 or h["center_x"] > frame_width:
            h["drift_x"] *= -1
            h["center_x"] = max(0.0, min(float(frame_width), h["center_x"]))
        if h["center_y"] < 0 or h["center_y"] > frame_height:
            h["drift_y"] *= -1
            h["center_y"] = max(0.0, min(float(frame_height), h["center_y"]))

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        size = h["base_size"] * (1.0 + h["amplitude"] * math.sin(2 * math.pi * h["phase"]))
        verts = _regular_polygon_vertices(h["n_sides"], size, h["angle"])
        pts = _to_cv2_points(verts, h["center_x"], h["center_y"])
        if h["filled"]:
            cv2.fillPoly(frame, [pts], 255)
        else:
            cv2.polylines(frame, [pts], isClosed=True, color=255, thickness=h["thickness"])
        return frame


class _StarPlugin(_ShapePlugin):
    """Command. n-pointed star (n=3..8) that rotates and pulses."""

    def __init__(self):
        super().__init__("star")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        min_dim = min(frame_width, frame_height)
        self.hyperparameters = {
            "n_points": random.randint(3, 8),
            "center_x": random.uniform(0, frame_width),
            "center_y": random.uniform(0, frame_height),
            "outer": random.uniform(0.05 * min_dim, 0.35 * min_dim),
            "inner_ratio": random.uniform(0.2, 0.6),
            "angle": random.uniform(0, 2 * math.pi),
            "phase": random.uniform(0, 1),
            "amplitude": random.uniform(0.1, 0.45),
            "rot_vel": random.uniform(-0.10, 0.10),
            "phase_vel": random.uniform(0.02, 0.08),
            "drift_x": random.uniform(-0.02 * sc, 0.02 * sc),
            "drift_y": random.uniform(-0.02 * sc, 0.02 * sc),
            "filled": random.random() < 0.6,
            "thickness": max(1, int(random.uniform(0.005 * sc, 0.04 * sc))),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        h["angle"] = (h["angle"] + h["rot_vel"]) % (2 * math.pi)
        h["phase"] = (h["phase"] + h["phase_vel"]) % 1.0
        h["center_x"] += h["drift_x"]; h["center_y"] += h["drift_y"]
        if h["center_x"] < 0 or h["center_x"] > frame_width:
            h["drift_x"] *= -1
            h["center_x"] = max(0.0, min(float(frame_width), h["center_x"]))
        if h["center_y"] < 0 or h["center_y"] > frame_height:
            h["drift_y"] *= -1
            h["center_y"] = max(0.0, min(float(frame_height), h["center_y"]))

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        factor = 1.0 + h["amplitude"] * math.sin(2 * math.pi * h["phase"])
        outer = h["outer"] * factor
        inner = outer * h["inner_ratio"]
        verts = _star_vertices(h["n_points"], outer, inner, h["angle"])
        pts = _to_cv2_points(verts, h["center_x"], h["center_y"])
        if h["filled"]:
            cv2.fillPoly(frame, [pts], 255)
        else:
            cv2.polylines(frame, [pts], isClosed=True, color=255, thickness=h["thickness"])
        return frame


class _GearPlugin(_ShapePlugin):
    """Command. Gear with rectangular teeth, rotating and pulsing."""

    def __init__(self):
        super().__init__("gear")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        min_dim = min(frame_width, frame_height)
        self.hyperparameters = {
            "n_teeth": random.randint(4, 12),
            "center_x": random.uniform(0, frame_width),
            "center_y": random.uniform(0, frame_height),
            "outer": random.uniform(0.05 * min_dim, 0.35 * min_dim),
            "inner_ratio": random.uniform(0.55, 0.80),
            "tooth_width_frac": random.uniform(0.35, 0.65),
            "angle": random.uniform(0, 2 * math.pi),
            "phase": random.uniform(0, 1),
            "amplitude": random.uniform(0.08, 0.30),
            "rot_vel": random.uniform(-0.08, 0.08),
            "phase_vel": random.uniform(0.02, 0.06),
            "drift_x": random.uniform(-0.015 * sc, 0.015 * sc),
            "drift_y": random.uniform(-0.015 * sc, 0.015 * sc),
            "filled": random.random() < 0.6,
            "thickness": max(1, int(random.uniform(0.005 * sc, 0.04 * sc))),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        h["angle"] = (h["angle"] + h["rot_vel"]) % (2 * math.pi)
        h["phase"] = (h["phase"] + h["phase_vel"]) % 1.0
        h["center_x"] += h["drift_x"]; h["center_y"] += h["drift_y"]
        if h["center_x"] < 0 or h["center_x"] > frame_width:
            h["drift_x"] *= -1
            h["center_x"] = max(0.0, min(float(frame_width), h["center_x"]))
        if h["center_y"] < 0 or h["center_y"] > frame_height:
            h["drift_y"] *= -1
            h["center_y"] = max(0.0, min(float(frame_height), h["center_y"]))

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        factor = 1.0 + h["amplitude"] * math.sin(2 * math.pi * h["phase"])
        outer = h["outer"] * factor
        inner = outer * h["inner_ratio"]
        verts = _gear_vertices(h["n_teeth"], outer, inner, h["tooth_width_frac"], h["angle"])
        pts = _to_cv2_points(verts, h["center_x"], h["center_y"])
        if h["filled"]:
            cv2.fillPoly(frame, [pts], 255)
        else:
            cv2.polylines(frame, [pts], isClosed=True, color=255, thickness=h["thickness"])
        return frame


# =============================================================================
# NEW PLUGIN: Voronoi cells (agent 6)
# =============================================================================

class _VoronoiPlugin(_ShapePlugin):
    """
    Command. Voronoi cellular mask. fill_mode = each cell takes its seed's binary color.
    edge_mode = only Voronoi boundaries are white (wireframe). Mutates frame.
    """

    _grid_cache = {}

    def __init__(self):
        super().__init__("voronoi")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        area_sqrt = (frame_width * frame_height) ** 0.5
        min_n = max(5, int(area_sqrt * 0.05))
        max_n = max(min_n + 1, int(area_sqrt * 0.20))
        N = random.randint(min_n, max_n)

        xs = np.random.uniform(0, frame_width, N).astype(np.float32)
        ys = np.random.uniform(0, frame_height, N).astype(np.float32)
        speed_factors = np.random.uniform(0.005, 0.03, N).astype(np.float32)
        angles = np.random.uniform(0, 2 * np.pi, N).astype(np.float32)

        colors = np.zeros(N, dtype=np.uint8)
        colors[np.random.choice(N, max(1, N // 2), replace=False)] = 255

        self.hyperparameters = {
            "xy": np.stack([xs, ys], axis=1),
            "vxy": np.stack([np.cos(angles) * speed_factors * sc,
                              np.sin(angles) * speed_factors * sc], axis=1).astype(np.float32),
            "colors": colors,
            "edge_mode": random.random() < 0.5,
        }

    @classmethod
    def _get_grid(cls, H, W):
        key = (H, W)
        if key not in cls._grid_cache:
            ys, xs = np.mgrid[0:H, 0:W]
            cls._grid_cache[key] = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        return cls._grid_cache[key]

    def draw(self, frame, frame_width, frame_height):
        H, W = frame_height, frame_width
        h = self.hyperparameters
        grid = self._get_grid(H, W)
        diff = grid[:, np.newaxis, :] - h["xy"][np.newaxis, :, :]
        labels = (diff ** 2).sum(axis=2).argmin(axis=1).reshape(H, W)

        if h["edge_mode"]:
            edge = np.zeros((H, W), dtype=bool)
            rd = labels[:, :-1] != labels[:, 1:]
            bd = labels[:-1, :] != labels[1:, :]
            edge[:, :-1] |= rd; edge[:, 1:] |= rd
            edge[:-1, :] |= bd; edge[1:, :] |= bd
            frame[:] = np.where(edge, np.uint8(255), np.uint8(0))
        else:
            frame[:] = h["colors"][labels]
        return frame

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        h["xy"] += h["vxy"]
        hit_l = h["xy"][:, 0] < 0
        hit_r = h["xy"][:, 0] > frame_width
        h["vxy"][hit_l | hit_r, 0] *= -1
        h["xy"][:, 0] = np.clip(h["xy"][:, 0], 0, frame_width)
        hit_t = h["xy"][:, 1] < 0
        hit_b = h["xy"][:, 1] > frame_height
        h["vxy"][hit_t | hit_b, 1] *= -1
        h["xy"][:, 1] = np.clip(h["xy"][:, 1], 0, frame_height)


# =============================================================================
# NEW PLUGIN: Metaballs (agent 7)
# =============================================================================

class _MetaballPlugin(_ShapePlugin):
    """
    Command. N metaballs bouncing. Field = sum r_i^2 / (dist^2 + eps); thresholded to binary.
    Liquid-like merging/splitting. Mutates frame.
    """

    _EPS = 1e-4

    def __init__(self):
        super().__init__("metaball")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        n = random.randint(3, min(8, 3 + int(sc / 80)))
        balls = []
        for _ in range(n):
            r = random.uniform(0.08, 0.18) * sc
            cx = random.uniform(r, frame_width - r)
            cy = random.uniform(r, frame_height - r)
            speed = random.uniform(0.01, 0.06) * sc
            a = random.uniform(0, 2 * np.pi)
            balls.append({"cx": cx, "cy": cy, "r": r,
                          "vx": speed * math.cos(a), "vy": speed * math.sin(a)})
        self.hyperparameters = {"balls": balls, "threshold": random.uniform(0.5, 1.5)}

    def update_hyperparameters(self, frame_width, frame_height):
        for b in self.hyperparameters["balls"]:
            b["cx"] += b["vx"]; b["cy"] += b["vy"]
            r = b["r"]
            if b["cx"] < r:
                b["cx"] = r; b["vx"] = abs(b["vx"])
            elif b["cx"] > frame_width - r:
                b["cx"] = frame_width - r; b["vx"] = -abs(b["vx"])
            if b["cy"] < r:
                b["cy"] = r; b["vy"] = abs(b["vy"])
            elif b["cy"] > frame_height - r:
                b["cy"] = frame_height - r; b["vy"] = -abs(b["vy"])

    def draw(self, frame, frame_width, frame_height):
        ys, xs = np.mgrid[0:frame_height, 0:frame_width].astype(np.float32)
        field = np.zeros((frame_height, frame_width), dtype=np.float32)
        for b in self.hyperparameters["balls"]:
            d2 = (xs - b["cx"]) ** 2 + (ys - b["cy"]) ** 2
            field += (b["r"] ** 2) / (d2 + self._EPS)
        frame[:] = np.where(field > self.hyperparameters["threshold"], np.uint8(255), np.uint8(0))
        return frame


# =============================================================================
# NEW PLUGIN: Flow field particles (agent 8)
# =============================================================================

def _evaluate_vortex_field(positions, vortices):
    """
    Pure function. Returns (P, 2) unit vectors from sum of vortex contributions.
    Each vortex contributes perpendicular-to-radial direction with Gaussian falloff.
    """
    P = positions.shape[0]
    field = np.zeros((P, 2), dtype=np.float64)
    eps = 1e-6
    for cx, cy, strength, sigma in vortices:
        dx = positions[:, 0] - cx
        dy = positions[:, 1] - cy
        d2 = dx * dx + dy * dy
        w = np.exp(-d2 / (sigma * sigma))
        d = np.sqrt(d2) + eps
        field[:, 0] += strength * w * (-dy) / d
        field[:, 1] += strength * w * dx / d
    norms = np.sqrt(field[:, 0] ** 2 + field[:, 1] ** 2) + eps
    field[:, 0] /= norms
    field[:, 1] /= norms
    return field


def _spawn_particles_random_edge(count, frame_width, frame_height):
    """Pure function. (count, 2) float positions along frame edges (slight inward offset)."""
    pos = np.zeros((count, 2), dtype=np.float64)
    for i in range(count):
        edge = random.randint(0, 3)
        if edge == 0:
            pos[i] = [random.uniform(0, frame_width), 1.0]
        elif edge == 1:
            pos[i] = [random.uniform(0, frame_width), float(frame_height - 2)]
        elif edge == 2:
            pos[i] = [1.0, random.uniform(0, frame_height)]
        else:
            pos[i] = [float(frame_width - 2), random.uniform(0, frame_height)]
    return pos


def _out_of_bounds_mask(positions, frame_width, frame_height, margin=5):
    """Pure function. (P,) bool mask — True where particle is outside frame+margin."""
    x, y = positions[:, 0], positions[:, 1]
    return ((x < -margin) | (x >= frame_width + margin)
            | (y < -margin) | (y >= frame_height + margin))


class _FlowFieldPlugin(_ShapePlugin):
    """
    Command. Particles swirl through a K-vortex field. Each frame paints
    line from prev pos to current pos + dot at tip. Mutates frame.
    """

    _REF_AREA = 5400
    _REF_PARTICLES = 20

    def __init__(self):
        super().__init__("flow_field")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        K = random.randint(3, 6)
        sigma = random.uniform(0.20, 0.35) * sc
        vortices = [(random.uniform(0, frame_width), random.uniform(0, frame_height),
                     random.uniform(0.5, 1.5) * random.choice([-1, 1]), sigma)
                    for _ in range(K)]
        area = frame_width * frame_height
        P = max(20, min(100, int(self._REF_PARTICLES * area / self._REF_AREA)))
        positions = _spawn_particles_random_edge(P, frame_width, frame_height)
        self.hyperparameters = {
            "vortices": vortices,
            "positions": positions,
            "prev_positions": positions.copy(),
            "velocity_scale": random.uniform(0.02, 0.05) * sc,
            "dot_radius": max(1, int(0.015 * sc)),
            "stroke_thickness": max(1, int(0.01 * sc)),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        h["prev_positions"] = h["positions"].copy()
        field = _evaluate_vortex_field(h["positions"], h["vortices"])
        new_pos = h["positions"] + field * h["velocity_scale"]
        oob = _out_of_bounds_mask(new_pos, frame_width, frame_height, margin=5)
        if np.any(oob):
            re = _spawn_particles_random_edge(int(np.sum(oob)), frame_width, frame_height)
            new_pos[oob] = re
            h["prev_positions"][oob] = re
        h["positions"] = new_pos

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        pos, prev = h["positions"], h["prev_positions"]
        for i in range(len(pos)):
            px, py = int(pos[i, 0]), int(pos[i, 1])
            ox, oy = int(prev[i, 0]), int(prev[i, 1])
            cv2.line(frame, (ox, oy), (px, py), 255, h["stroke_thickness"])
            if 0 <= px < frame_width and 0 <= py < frame_height:
                cv2.circle(frame, (px, py), h["dot_radius"], 255, -1)
        return frame


# =============================================================================
# NEW PLUGIN: Ribbon / worm (agent 9)
# =============================================================================

class _RibbonPlugin(_ShapePlugin):
    """
    Command. Slithering ribbon: head wanders by random walk + sinusoidal heading
    oscillation; body is a deque of past head positions, drawn as a thick polyline.
    Variants: thick, thin, tapered. Mutates frame.
    """

    _VARIANTS = ("thick", "thin", "taper")

    def __init__(self):
        super().__init__("ribbon")

    def start_drawing(self, frame_width, frame_height, T=25):
        """Override: ribbon needs long duration so its tail is visible."""
        self.randomize_hyperparameters(frame_width, frame_height)
        self.duration = random.randint(max(8, T // 4), max(20, T))
        self.remaining_duration = self.duration
        self.is_animated = True

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = _scale(frame_width, frame_height)
        body_length = max(20, int(sc * 0.5))
        head_x = random.uniform(frame_width * 0.1, frame_width * 0.9)
        head_y = random.uniform(frame_height * 0.1, frame_height * 0.9)
        self.hyperparameters = {
            "head_x": head_x, "head_y": head_y,
            "heading": random.uniform(0, 2 * math.pi),
            "osc_phase": 0.0,
            "body": deque([(head_x, head_y)] * body_length, maxlen=body_length),
            "step_size": sc * random.uniform(0.02, 0.05),
            "rw_sigma": random.uniform(0.03, 0.12),
            "osc_amplitude": random.uniform(0.05, 0.25),
            "osc_freq": random.uniform(0.2, 0.6),
            "base_thickness": max(1, int(sc * random.uniform(0.01, 0.04))),
            "variant": random.choice(self._VARIANTS),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        h["osc_phase"] += h["osc_freq"]
        h["heading"] += random.gauss(0, h["rw_sigma"]) + h["osc_amplitude"] * math.sin(h["osc_phase"])
        h["head_x"] = (h["head_x"] + h["step_size"] * math.cos(h["heading"])) % frame_width
        h["head_y"] = (h["head_y"] + h["step_size"] * math.sin(h["heading"])) % frame_height
        h["body"].appendleft((h["head_x"], h["head_y"]))

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        body = list(h["body"])
        n = len(body)
        if n < 2:
            return frame
        variant = h["variant"]
        thick = h["base_thickness"]
        if variant == "thick":
            pts = np.array([(int(x), int(y)) for x, y in body], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=255, thickness=thick, lineType=cv2.LINE_AA)
        elif variant == "thin":
            thin = max(1, thick // 3)
            pts = np.array([(int(x), int(y)) for x, y in body], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=255, thickness=thin, lineType=cv2.LINE_AA)
        else:  # taper
            for i in range(n - 1):
                frac = i / max(n - 1, 1)
                t = max(1, int(thick * (1.0 - frac * 0.85)))
                x1, y1 = int(body[i][0]), int(body[i][1])
                x2, y2 = int(body[i + 1][0]), int(body[i + 1][1])
                cv2.line(frame, (x1, y1), (x2, y2), 255, thickness=t, lineType=cv2.LINE_AA)
        return frame


# =============================================================================
# NEW PLUGINS: Radar sweep + pie wedge (agent 10)
# =============================================================================

class _RadarSweepPlugin(_ShapePlugin):
    """Command. Rotating filled sector — radar sweep. Mutates frame."""

    def __init__(self):
        super().__init__("radar_sweep")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = min(frame_width, frame_height)
        self.hyperparameters = {
            "center_x": random.randint(int(0.1 * frame_width), int(0.9 * frame_width)),
            "center_y": random.randint(int(0.1 * frame_height), int(0.9 * frame_height)),
            "radius": int(random.uniform(0.15, 0.60) * sc),
            "sweep_angle": random.uniform(15, 90),
            "current_angle": random.uniform(0, 360),
            "delta_angle": random.choice([-1, 1]) * random.uniform(5, 25),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        h = self.hyperparameters
        h["current_angle"] = (h["current_angle"] + h["delta_angle"]) % 360

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        cv2.ellipse(frame, (int(h["center_x"]), int(h["center_y"])),
                    (int(h["radius"]), int(h["radius"])), 0,
                    h["current_angle"] - h["sweep_angle"], h["current_angle"], 255, -1)
        return frame


class _PieWedgePlugin(_ShapePlugin):
    """Command. Static filled pie slice. Mutates frame."""

    def __init__(self):
        super().__init__("pie_wedge")

    def randomize_hyperparameters(self, frame_width, frame_height):
        sc = min(frame_width, frame_height)
        start = random.uniform(0, 360)
        self.hyperparameters = {
            "center_x": random.randint(0, frame_width - 1),
            "center_y": random.randint(0, frame_height - 1),
            "radius": int(random.uniform(0.10, 0.70) * sc),
            "start_angle": start,
            "end_angle": start + random.uniform(20, 300),
        }

    def update_hyperparameters(self, frame_width, frame_height):
        pass  # static

    def draw(self, frame, frame_width, frame_height):
        h = self.hyperparameters
        cv2.ellipse(frame, (int(h["center_x"]), int(h["center_y"])),
                    (int(h["radius"]), int(h["radius"])), 0,
                    h["start_angle"], h["end_angle"], 255, -1)
        return frame


# =============================================================================
# Post-processing actions
# =============================================================================

class _ActionBase:
    """Mixin: dummy interface methods so actions can live alongside plugins."""
    is_animated = False
    def start_drawing(self, frame_width, frame_height, T=25): pass
    def is_active(self): return False
    def decrement_duration(self): pass
    def draw(self, frame, frame_width, frame_height): return frame
    def update_hyperparameters(self, frame_width, frame_height): pass


class _FramePersistenceAction(_ActionBase):
    """Command. Persistence duration scales with T: (1, max(2, T//8)). Mutates video."""

    def __init__(self):
        self.name = "FramePersistence"

    def apply(self, video):
        T = video.shape[0]
        persistence = random.randint(1, max(2, T // 8))
        for t in sorted(random.sample(range(T), random.randint(1, T))):
            for i in range(1, persistence + 1):
                if t + i < T:
                    video[t + i] = video[t]
        return video


class _FrameShiftAction(_ActionBase):
    """
    Command. Per-frame translate+rotate warp on random subset of frames.
    motion_proportion and max_rotation_angle stay dimensionless.
    Duration scales with T: (1, max(2, T//5)). Mutates video.
    """

    def __init__(self, motion_proportion=0.2, max_rotation_angle=20):
        self.motion_proportion = motion_proportion
        self.max_rotation_angle = max_rotation_angle
        self.name = "FrameShift"

    def apply(self, video):
        T = video.shape[0]
        duration = random.randint(1, max(2, T // 5))
        frames_to_shift = sorted(random.sample(range(T), random.randint(0, T // 2)))

        for t in frames_to_shift:
            rows, cols = video[0].shape
            sx = int(cols * self.motion_proportion)
            sy = int(rows * self.motion_proportion)
            sx0 = random.randint(-sx, sx); sx1 = random.randint(-sx, sx)
            sy0 = random.randint(-sy, sy); sy1 = random.randint(-sy, sy)
            a0 = random.randint(-self.max_rotation_angle, self.max_rotation_angle)
            a1 = random.randint(-self.max_rotation_angle, self.max_rotation_angle)

            for i in range(duration):
                fi = t + i
                if fi >= T:
                    break
                frame = video[fi]
                rows, cols = frame.shape
                ease = _ease_in_out_cubic(i / (duration - 1) if duration > 1 else 0)
                cx = sx0 + ease * (sx1 - sx0)
                cy = sy0 + ease * (sy1 - sy0)
                ca = a0 + ease * (a1 - a0)
                Mt = np.float32([[1, 0, cx], [0, 1, cy]])
                Mr = cv2.getRotationMatrix2D((cols / 2, rows / 2), ca, 1)
                shifted = cv2.warpAffine(frame, Mt, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                video[fi] = cv2.warpAffine(shifted, Mr, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return video


class _ZoomAction(_ActionBase):
    """Command. Smooth slow zoom (in or out) on random frame runs. Mutates video."""

    def __init__(self, zoom_range=(0.75, 1.35), duration_range=(4, 12), num_runs_range=(1, 3)):
        self.zoom_range = zoom_range
        self.duration_range = duration_range
        self.num_runs_range = num_runs_range
        self.name = "Zoom"

    def apply(self, video):
        T, H, W = video.shape
        num_runs = random.randint(*self.num_runs_range)
        # Scale duration range with T but keep originals as minima.
        d_min = self.duration_range[0]
        d_max = max(d_min + 1, min(self.duration_range[1], max(d_min + 1, T // 3)))

        for _ in range(num_runs):
            duration = random.randint(d_min, d_max)
            start_t = random.randint(0, max(0, T - duration))
            end_t = min(T, start_t + duration)
            actual = end_t - start_t
            cx = random.uniform(0.2 * W, 0.8 * W)
            cy = random.uniform(0.2 * H, 0.8 * H)
            end_scale = random.uniform(*self.zoom_range)

            for i, t in enumerate(range(start_t, end_t)):
                ease = _ease_in_out_cubic(i / max(1, actual - 1))
                s = 1.0 + ease * (end_scale - 1.0)
                M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
                video[t] = cv2.warpAffine(video[t], M, (W, H),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return video


class _PulseAction(_ActionBase):
    """Command. Morphological pulse (dilate↔erode) over a random window. Mutates video."""

    def __init__(self, max_kernel_fraction=0.06, num_cycles_range=(1, 3), duration_fraction=0.6):
        self.max_kernel_fraction = max_kernel_fraction
        self.num_cycles_range = num_cycles_range
        self.duration_fraction = duration_fraction
        self.name = "Pulse"

    def apply(self, video):
        T, H, W = video.shape
        max_radius = max(1, int(self.max_kernel_fraction * min(H, W)))
        num_cycles = random.randint(*self.num_cycles_range)
        window = max(4, int(T * self.duration_fraction))
        start_t = random.randint(0, max(0, T - window))
        end_t = min(T, start_t + window)
        actual = end_t - start_t

        for i, t in enumerate(range(start_t, end_t)):
            phase = (i / max(1, actual - 1)) * num_cycles * 2 * np.pi
            pulse = np.sin(phase)
            radius = int(abs(pulse) * max_radius)
            if radius < 1:
                continue
            size = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            if pulse > 0:
                video[t] = cv2.dilate(video[t], kernel, iterations=1)
            else:
                video[t] = cv2.erode(video[t], kernel, iterations=1)
        return video


# =============================================================================
# Video generator
# =============================================================================

class _VideoGenerator:
    """
    Command. Renders frames by activating plugins and applies post-processing actions.
    """

    def __init__(self, T=13, H=60, W=90, N=2, plugins=None, actions=None):
        self.T = T
        self.H = H
        self.W = W
        self.N = N
        self.plugins = plugins or []
        self.actions = actions or []
        self.active_plugins = []

    def generate_video(self):
        video = np.zeros((self.T, self.H, self.W), dtype=np.uint8)
        for t in range(self.T):
            if random.random() < 0.4 and self.plugins:
                p = random.choice(self.plugins)
                p.start_drawing(self.W, self.H, T=self.T)
                self.active_plugins.append(p)
            for p in self.active_plugins:
                if p.is_active():
                    video[t] = p.draw(video[t], self.W, self.H)
                    p.update_hyperparameters(self.W, self.H)
                    p.decrement_duration()
            self.active_plugins = [p for p in self.active_plugins if p.is_active()]
        for a in self.actions:
            video = a.apply(video)
        return video


# =============================================================================
# Plugin pools
# =============================================================================

def _make_classic_plugins():
    """Query. Fresh instances of the original-style plugins."""
    return [
        _RectanglePlugin(),
        _EllipsePlugin(),
        _ScribblePlugin(), _ScribblePlugin(), _ScribblePlugin(),
        _BoopySaltPlugin(),
        _SaltPlugin(),
        _TrianglePlugin(),
    ]


def _make_new_plugins():
    """Query. Fresh instances of the new creative plugins."""
    return [
        _3DPlanePlugin(),
        _ParametricCurvePlugin(),
        _BezierWormPlugin(),
        _BezierCalligraphyPlugin(),
        _BezierLoopPlugin(),
        _PolygonPlugin(),
        _StarPlugin(),
        _GearPlugin(),
        _VoronoiPlugin(),
        _MetaballPlugin(),
        _FlowFieldPlugin(),
        _RibbonPlugin(),
        _RadarSweepPlugin(),
        _PieWedgePlugin(),
    ]


def _make_all_plugins():
    """Query. Combined classic + new."""
    return _make_classic_plugins() + _make_new_plugins()


def _random_shapes_video(plugins, *, T, H, W, N):
    """
    Command. Generates one random mask video using a sample of plugins
    and random subset of post-processing actions.
    """
    actions = [
        _FramePersistenceAction(),
        _FrameShiftAction(),
        _ZoomAction(),
        _PulseAction(),
    ]
    plugins_used = random.sample(plugins, random.randint(1, len(plugins)))
    actions_used = random.sample(actions, random.randint(0, len(actions)))
    return _VideoGenerator(T, H, W, N, plugins_used, actions_used).generate_video()


# =============================================================================
# Public API
# =============================================================================

def get_random_video_mask(T=25, H=60, W=90):
    """
    Generates a random video mask with given num frames, height, width.

    Returns a bool numpy video in (T, H, W) form (True = masked).

    50% of the time draws only from the classic plugin pool; the other 50%
    draws from the full classic + new-creative pool.

    The video is built by OR'ing/AND'ing several random shape videos to
    create varied coverage patterns, then a few frames are randomly forced
    to be fully masked (a common inpainting-training augmentation).

    Args:
        T (int): Number of frames.
        H (int): Frame height.
        W (int): Frame width.

    Returns:
        np.ndarray: (T, H, W) bool.
    """
    attempts = 0
    while True:
        attempts += 1
        assert attempts < 100, 'get_random_video_mask reached 100 attempts. Is it broken?'
        try:
            N = 4
            plugins = _make_classic_plugins() if random.random() < 0.5 else _make_all_plugins()

            def randvid():
                """Returns a (T, H, W) bool mask from a random subset of `plugins`."""
                return _random_shapes_video(plugins=plugins, T=T, H=H, W=W, N=N) > 0

            video = randvid()
            thresh = rp.random_float() ** 4
            for _ in range(rp.random_int(2, 10)):
                try:
                    if rp.random_chance(0.1):
                        video |= randvid()
                    newvideo = randvid()
                    if rp.random_chance():
                        newvideo = ~newvideo
                    newvideo = randvid() & newvideo
                    video |= newvideo
                    if video.mean() > thresh:
                        break
                except Exception:
                    pass

            full_frame_indices = (
                [0] * rp.random_chance(0.1)
                + [T - 1] * rp.random_chance(0.1)
                + rp.random_batch(range(T), rp.random_int(0, 5) * rp.random_chance(0.1))
            )
            if rp.random_chance(1 / 10):
                full_frame_indices += list(range(rp.random_int(T)))
            for x in full_frame_indices:
                video[x] = True

            if rp.random_chance(1 / 10):
                video = ~video

            return video

        except Exception:
            rp.print_stack_trace()


def demo(output_folder="random_video_masks_demo"):
    """Demonstrates the get_random_video_mask() function."""
    print("PWD: " + rp.fansi_highlight_path(rp.get_current_directory()))
    for i in range(15):
        video = get_random_video_mask()
        rp.display_video(video)
        image = rp.tiled_images(video, border_color="red")
        path = rp.save_image(
            image, rp.get_unique_copy_path(rp.path_join(output_folder, "random_masks.jpg"))
        )
        print("    " + rp.fansi_highlight_path(path))


if __name__ == '__main__':
    demo()
