# jass-cv — Swiss German Jass card detection

Detect the 36 Swiss German Jass cards (Eicheln, Schilten, Rosen, Schellen × 6–Ass) in photos
and live video.

| Component | What | Where |
|---|---|---|
| **Prediction API** | FastAPI, `POST /predict` → JSON boxes, `POST /predict/annotated` → JPEG | `src/jass_cv/api/` |
| **Frontend** | Streamlit: upload a photo or take a webcam snapshot, see the boxes | `src/jass_cv/frontend/` |
| **Live-feed dev tool** | OpenCV window on webcam/video/RTSP, local model or via the API; `s` saves frames for the dataset | `tools/live_detect.py` |
| **Training** | Ultralytics YOLO fine-tuning wrapper | `tools/train.py`, `data/jass.yaml` |
| **Core** | card classes, detector interface, schemas, drawing | `src/jass_cv/` |

> **Status:** scaffold. No trained model yet — the API falls back to a *dummy* detector that
> always returns one fake "Rosen Ass" box, so the whole stack runs end to end today.
> Architecture and tooling are still open: see [docs/architecture.md](docs/architecture.md).

## Quickstart (Docker)

```bash
cp .env.example .env
docker compose up -d --build
```

- Frontend: <http://localhost:8501>
- API docs (Swagger): <http://localhost:8000/docs>

Once you have weights, drop them in `models/best.pt` and `docker compose restart api`.

Hot reload while developing (syncs `src/` into the containers):

```bash
docker compose watch
```

## Local development (uv)

```bash
make test        # pytest (dummy model, no torch needed)
make lint
uv run --extra api uvicorn jass_cv.api.main:app --reload
uv run --extra frontend streamlit run src/jass_cv/frontend/app.py
```

### Live-feed dev tool

Runs on the host, not in Docker (needs the camera and a GUI window):

```bash
make live        # model in-process, models/best.pt
make live-api    # send frames to the running API instead
uv run --extra live python tools/live_detect.py --source path/to/video.mp4
```

Keys: `q` quit · `s` save frame to `captures/` · `p` pause · `+`/`-` confidence.

### Training

```bash
uv run --extra inference python tools/train.py --model yolo11n.pt --epochs 100 --device 0
```

See [docs/dataset.md](docs/dataset.md) for dataset layout and labeling rules.

## Repository layout

```
src/jass_cv/
  cards.py        # the 36 classes — single source of truth for class ids
  detector.py     # Detector protocol, YoloDetector, DummyDetector, dedupe
  schemas.py      # pydantic API models
  drawing.py      # box rendering (PIL)
  config.py       # JASS_* settings
  api/main.py     # FastAPI app
  frontend/app.py # Streamlit app
tools/            # live_detect.py, train.py
docker/           # api.Dockerfile, frontend.Dockerfile
data/jass.yaml    # YOLO dataset config (data itself is not in git)
models/           # weights (not in git)
docs/             # architecture, dataset, API, project plan
```

## Docs

- [Architecture & open decisions](docs/architecture.md)
- [Dataset & training](docs/dataset.md)
- [API reference](docs/api.md)
- [Project plan (2 × 15 h)](docs/project-plan.md)
