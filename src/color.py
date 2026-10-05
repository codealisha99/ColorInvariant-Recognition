"""Palette changes that keep the motif in place.

A saree colorway is a new hue on the same weave. Rotating the Lab chroma
axes (a, b) changes hue and leaves lightness, and therefore the motif
edges, where they were. Matching one image's mean chroma to another's
builds the same-palette impostor test.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ).astype(np.float32)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_DELTA = 6.0 / 29.0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA**3, np.cbrt(t), t / (3 * _DELTA**2) + 4 / 29)


def _f_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA, t**3, 3 * _DELTA**2 * (t - 4 / 29))


def rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rgb is float HxWx3 in 0..1. Returns L, a, b."""
    linear = _srgb_to_linear(rgb)
    xyz = linear @ _RGB_TO_XYZ.T
    xyz = xyz / _D65
    fxyz = _f(xyz)
    L = 116.0 * fxyz[..., 1] - 16.0
    a = 500.0 * (fxyz[..., 0] - fxyz[..., 1])
    b = 200.0 * (fxyz[..., 1] - fxyz[..., 2])
    return L, a, b


def lab_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=-1) * _D65
    linear = xyz @ _XYZ_TO_RGB.T
    return np.clip(_linear_to_srgb(linear), 0.0, 1.0)


def _to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(rgb: np.ndarray) -> Image.Image:
    arr = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _weighted_chroma(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean chroma angle (degrees) and magnitude, ignoring near-gray pixels."""
    magnitude = np.sqrt(a * a + b * b)
    weight = np.clip(magnitude - 8.0, 0.0, None)
    if float(weight.sum()) < 1.0:
        weight = np.ones_like(magnitude)
    mean_a = float((a * weight).sum() / weight.sum())
    mean_b = float((b * weight).sum() / weight.sum())
    angle = float(np.degrees(np.arctan2(mean_b, mean_a)))
    mag = float(np.hypot(mean_a, mean_b))
    return angle, mag


def chroma_rotate(image: Image.Image, degrees: float, sat_scale: float = 1.0) -> Image.Image:
    """Rotate hue in Lab and scale saturation. Lightness is unchanged."""
    if abs(degrees) < 1e-6 and abs(sat_scale - 1.0) < 1e-6:
        return image.convert("RGB")
    L, a, b = rgb_to_lab(_to_float(image))
    theta = np.deg2rad(degrees).astype(np.float32)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a_rot = sat_scale * (a * cos_t - b * sin_t)
    b_rot = sat_scale * (a * sin_t + b * cos_t)
    return _to_image(lab_to_rgb(L, a_rot, b_rot))


def palette_match(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Recolor `image` so its dominant hue and saturation match `reference`."""
    src = _to_float(image)
    ref = _to_float(reference)
    L, a, b = rgb_to_lab(src)
    _, ra, rb = rgb_to_lab(ref)
    src_angle, src_mag = _weighted_chroma(a, b)
    ref_angle, ref_mag = _weighted_chroma(ra, rb)
    sat_scale = 1.0 if src_mag < 1e-3 else ref_mag / src_mag
    sat_scale = float(np.clip(sat_scale, 0.5, 2.0))
    return chroma_rotate(image, ref_angle - src_angle, sat_scale)
