# Prediction API (FastAPI + Ultralytics, CPU-only PyTorch to keep the image small-ish).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1 YOLO_CONFIG_DIR=/tmp/ultralytics
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU wheels first so ultralytics doesn't pull the multi-GB CUDA build.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml --extra api

COPY src ./src
RUN useradd --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "jass_cv.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
