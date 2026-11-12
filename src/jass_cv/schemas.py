from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from jass_cv.cards import CLASS_NAMES


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
    polygon: list[tuple[float, float]] | None = Field(
        default=None, description="4 corners of the rotated box (OBB models only)"
    )


class PredictionResponse(BaseModel):
    model: str = Field(description="Name of the model that produced the result ('dummy' = fake).")
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[Detection]
    cards: list[str] = Field(description="Card labels in the hand, sorted by suit and rank.")


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str
    version: str


class HandDiffRequest(BaseModel):
    before: list[str] = Field(examples=[["rosen_ass", "eicheln_6"]])
    after: list[str] = Field(examples=[["eicheln_6"]])

    @field_validator("before", "after")
    @classmethod
    def _known_cards(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(CLASS_NAMES))
        if unknown:
            raise ValueError(f"unknown card labels: {unknown}")
        return v
