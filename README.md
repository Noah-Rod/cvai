# jass-cv: recognise a fanned Jass hand

Recognise the cards of a **fanned Swiss German Jass hand photographed from the player's own
point of view**. Take a photo after each card is played to see which card it was.

Course project (CV/AI, 2 people × ~15 h) built as a small study:

> *Can a detector trained only on synthetic images recognise the cards of a real fanned
> hand, and which kind of box is best for heavily overlapping cards?*

| Part | What | Where |
|---|---|---|
| **Synthetic data generator** | fanned hands with overlap, perspective, thumb, lighting, blur; exact labels for free | `src/jass_cv/synth.py` |
| **Dataset export** | 5 box variants from the same images (whole card / visible part / corner, upright / rotated) | `src/jass_cv/dataset.py` |
| **Training & evaluation** | YOLO11 (upright + rotated boxes); task-level evaluation on real photos | `tools/train.py`, `tools/evaluate.py` |
| **Prediction API** | FastAPI: `/predict`, `/predict/annotated`, `/hand/diff` | `src/jass_cv/api/` |
| **Frontend** | Streamlit: photo/snapshot → hand, "played since last photo" | `src/jass_cv/frontend/` |
| **Live-feed dev tool** | OpenCV window on webcam/video, local model or via the API | `tools/live_detect.py` |
| **Labeling** | Label Studio (self-hosted) for the real photos | `labeling/`, compose profile `label` |

**Status:** pipeline complete and tested end to end with placeholder cards. Next: real card
photos and real test photos (see the [project plan](docs/project-plan.md)). Without trained
weights the API uses a *dummy* detector that returns one fake card.

## Quickstart: app (Docker Desktop)

```bash
cp .env.example .env
docker compose up -d --build
```

Frontend <http://localhost:8501> · API docs <http://localhost:8000/docs>

To use a trained model, set `JASS_MODEL_PATH=models/<name>.pt` in `.env` and run
`docker compose restart api`.

## Quickstart: data & training (Windows / Linux / macOS, with uv)

```bash
# 1. synthetic hands (placeholder faces until you have real card photos)
uv run --extra synth python -m jass_cv.synth --out data/synth/smoke --n 200 --placeholder-cards
# 2. YOLO dataset for one box variant
uv run --extra synth python -m jass_cv.dataset export --variant corner_obb --train data/synth/smoke/train --val data/synth/smoke/val --out data/exports/smoke-corner_obb
# 3. train (CPU smoke test; real runs on Kaggle, see docs/training.md)
uv run --extra inference python tools/train.py --data data/exports/smoke-corner_obb/data.yaml --model yolo11n-obb.pt --name smoke --epochs 3
# 4. evaluate on real photos
uv run --extra inference python tools/evaluate.py --weights models/smoke.pt --images data/real/test/images --hands data/real/test/hands.csv
```

Live webcam (runs on the host, not in Docker):

```bash
uv run --extra live python tools/live_detect.py --weights models/smoke.pt
uv run --extra live python tools/live_detect.py --api http://localhost:8000
```

Keys: `q` quit · `s` save frame to `captures/` · `p` pause · `+`/`-` confidence.

## Development

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check src tools tests
uv run --extra dev ruff format src tools tests
docker compose watch          # hot reload of src/ into the containers
```

## Docs

- [Architecture, decisions & experiments](docs/architecture.md)
- [Data: card faces, synthetic hands, real photos, labeling](docs/dataset.md)
- [Training on Windows or Kaggle](docs/training.md)
- [API reference](docs/api.md)
- [Project plan, 2 × 15 h](docs/project-plan.md)
