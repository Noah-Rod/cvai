"""Detector abstraction.

Everything that consumes predictions (API, live tool, frontend) talks to `Detector`, so the
model implementation can be swapped (YOLO .pt, ONNX, a two-stage detector+classifier, ...)
without touching the apps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from PIL import Image

from jass_cv.cards import CARDS, card_by_id, card_by_label
from jass_cv.config import Settings
from jass_cv.schemas import BBox, Detection

log = logging.getLogger(__name__)


class Detector(Protocol):
    name: str

    def predict(self, image: Image.Image, conf: float, iou: float) -> list[Detection]: ...


def make_detection(class_id: int, confidence: float, box: tuple[float, ...]) -> Detection:
    card = card_by_id(class_id)
    x1, y1, x2, y2 = (float(v) for v in box)
    return Detection(
        class_id=card.class_id,
        label=card.label,
        display_name=card.display_name,
        suit=card.suit,
        rank=card.rank,
        confidence=round(float(confidence), 4),
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def dedupe_by_card(detections: list[Detection]) -> list[Detection]:
    """A Jass deck contains every card exactly once, so keep only the best box per card."""
    best: dict[int, Detection] = {}
    for d in detections:
        if d.class_id not in best or d.confidence > best[d.class_id].confidence:
            best[d.class_id] = d
    return sorted(best.values(), key=lambda d: -d.confidence)


class DummyDetector:
    """Returns one fixed, fake detection. Lets the API/frontend run before a model exists."""

    name = "dummy"

    def predict(self, image: Image.Image, conf: float, iou: float) -> list[Detection]:
        w, h = image.size
        box = (w * 0.3, h * 0.2, w * 0.7, h * 0.8)
        return [make_detection(card_by_label("rosen_ass").class_id, 0.99, box)]


class YoloDetector:
    """Ultralytics YOLO model (.pt or exported .onnx)."""

    def __init__(self, weights: Path, device: str = "cpu", img_size: int = 640) -> None:
        from ultralytics import YOLO  # heavy import, keep it lazy

        self.model = YOLO(str(weights))
        self.device = device
        self.img_size = img_size
        self.name = weights.name
        self._check_class_names()
        self.predict(Image.new("RGB", (img_size, img_size)), conf=0.5, iou=0.5)  # warm-up

    def _check_class_names(self) -> None:
        names = dict(getattr(self.model, "names", None) or {})
        if len(names) != len(CARDS):
            raise ValueError(
                f"{self.name} has {len(names)} classes, expected {len(CARDS)} Jass cards"
            )
        if names != {c.class_id: c.label for c in CARDS}:
            log.warning("Model class names differ from jass_cv.cards — labels may be wrong")

    def predict(self, image: Image.Image, conf: float, iou: float) -> list[Detection]:
        result = self.model.predict(
            image, conf=conf, iou=iou, imgsz=self.img_size, device=self.device, verbose=False
        )[0]
        b = result.boxes
        rows = zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist(), strict=True)
        return [make_detection(int(cls), conf, tuple(box)) for cls, conf, box in rows]


def load_detector(settings: Settings) -> Detector:
    backend = settings.model_backend
    if backend == "dummy":
        return DummyDetector()
    if settings.model_path.exists():
        return YoloDetector(settings.model_path, settings.device, settings.img_size)
    if backend == "yolo":
        raise FileNotFoundError(f"Model weights not found: {settings.model_path}")
    log.warning("No weights at %s — falling back to DummyDetector", settings.model_path)
    return DummyDetector()
