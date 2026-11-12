"""Card geometry shared by the synthetic generator and the dataset exporter.

A card is described by its `quad`: the 4 corners of the card face in image pixels, ordered
top-left, top-right, bottom-right, bottom-left *as printed on the card* (so the quad also
encodes the card's orientation). Everything else (corner region, boxes) is derived from it.
"""

from __future__ import annotations

import cv2
import numpy as np

# Face template size in pixels. Ratio ≈ 57 × 90 mm — measure your deck and adjust.
FACE_W, FACE_H = 360, 568

# Region in the top-left corner that shows rank + suit, as fractions (x0, y0, x1, y1) of the
# face. Tune this to your deck (look at a few cards and check the box in the generated images).
CORNER_REGION = (0.0, 0.0, 0.24, 0.24)


def face_rect() -> np.ndarray:
    return np.float32([[0, 0], [FACE_W, 0], [FACE_W, FACE_H], [0, FACE_H]])


def corner_rect() -> np.ndarray:
    x0, y0, x1, y1 = CORNER_REGION
    return np.float32(
        [
            [x0 * FACE_W, y0 * FACE_H],
            [x1 * FACE_W, y0 * FACE_H],
            [x1 * FACE_W, y1 * FACE_H],
            [x0 * FACE_W, y1 * FACE_H],
        ]
    )


def transform(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply a 3×3 homography to an (N, 2) array of points."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(h, dtype=np.float64)).reshape(-1, 2)


def face_to_quad(quad) -> np.ndarray:
    """Homography mapping face-template pixels onto the card quad in the image."""
    return cv2.getPerspectiveTransform(face_rect(), np.float32(quad))


def corner_quad_from_card_quad(quad) -> np.ndarray:
    return transform(corner_rect(), face_to_quad(quad))


def aabb(points, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Axis-aligned box of `points`, clipped to the image; None if nothing is left."""
    pts = np.asarray(points, dtype=np.float64)
    x1, y1 = np.clip(pts.min(axis=0), 0, [width, height])
    x2, y2 = np.clip(pts.max(axis=0), 0, [width, height])
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return float(x1), float(y1), float(x2), float(y2)
