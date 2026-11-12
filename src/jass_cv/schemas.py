from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned box in absolute pixel coordinates of the input image."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    label: str = Field(examples=["rosen_ass"])
    display_name: str = Field(examples=["Rosen Ass"])
    suit: str
    rank: str
    confidence: float = Field(ge=0, le=1)
    bbox: BBox


class PredictionResponse(BaseModel):
    model: str = Field(description="Name of the model that produced the result ('dummy' = fake).")
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[Detection]
    cards: list[str] = Field(description="Unique card labels found, sorted by class id.")


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str
    version: str
