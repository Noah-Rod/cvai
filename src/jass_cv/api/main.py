"""FastAPI prediction service.

Run locally:  uvicorn jass_cv.api.main:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError

from jass_cv import __version__
from jass_cv.cards import CARDS
from jass_cv.config import get_settings
from jass_cv.detector import Detector, dedupe_by_card, load_detector
from jass_cv.drawing import draw_detections
from jass_cv.schemas import Detection, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.detector = load_detector(settings)
    logging.getLogger(__name__).info("Loaded detector: %s", app.state.detector.name)
    yield


app = FastAPI(title="Jass Card Detection API", version=__version__, lifespan=lifespan)

Conf = Annotated[float | None, Query(ge=0, le=1, description="Confidence threshold")]
Iou = Annotated[float | None, Query(ge=0, le=1, description="NMS IoU threshold")]
Dedupe = Annotated[bool, Query(description="Keep only the best box per card (deck has no dupes)")]


async def _read_image(file: UploadFile, max_mb: int) -> Image.Image:
    data = await file.read()
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {max_mb} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)  # phone photos carry rotation in EXIF
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, "Could not decode image") from e


async def _predict(
    request: Request, file: UploadFile, conf: float | None, iou: float | None, dedupe: bool
) -> tuple[Image.Image, list[Detection], float]:
    settings = request.app.state.settings
    detector: Detector = request.app.state.detector
    image = await _read_image(file, settings.max_upload_mb)
    start = time.perf_counter()
    detections = await run_in_threadpool(
        detector.predict,
        image,
        conf if conf is not None else settings.conf_threshold,
        iou if iou is not None else settings.iou_threshold,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    if dedupe:
        detections = dedupe_by_card(detections)
    return image, detections, elapsed_ms


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    detector = request.app.state.detector
    return HealthResponse(
        status="ok",
        model=detector.name,
        backend=type(detector).__name__,
        version=__version__,
    )


@app.get("/classes")
def classes() -> list[dict]:
    return [
        {
            "class_id": c.class_id,
            "label": c.label,
            "display_name": c.display_name,
            "suit": c.suit,
            "rank": c.rank,
        }
        for c in CARDS
    ]


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Image (jpg/png/...)")],
    conf: Conf = None,
    iou: Iou = None,
    dedupe: Dedupe = True,
) -> PredictionResponse:
    image, detections, ms = await _predict(request, file, conf, iou, dedupe)
    return PredictionResponse(
        model=request.app.state.detector.name,
        image_width=image.width,
        image_height=image.height,
        inference_ms=round(ms, 1),
        detections=detections,
        cards=[d.label for d in sorted(detections, key=lambda d: d.class_id)],
    )


@app.post(
    "/predict/annotated", response_class=Response, responses={200: {"content": {"image/jpeg": {}}}}
)
async def predict_annotated(
    request: Request,
    file: Annotated[UploadFile, File()],
    conf: Conf = None,
    iou: Iou = None,
    dedupe: Dedupe = True,
) -> Response:
    image, detections, _ = await _predict(request, file, conf, iou, dedupe)
    buf = io.BytesIO()
    draw_detections(image, detections).save(buf, format="JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")
