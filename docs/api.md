# API reference

Base URL: `http://localhost:8000`. Interactive docs: `/docs` (Swagger) and `/redoc`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status and the loaded model (`model: "dummy"` means no real weights) |
| GET | `/classes` | The 36 classes with ids, labels, suit and rank |
| POST | `/predict` | Multipart `file` → detections as JSON |
| POST | `/predict/annotated` | Multipart `file` → JPEG with boxes drawn |
| POST | `/hand/diff` | `{"before": [...], "after": [...]}` → `{"played": [...], "appeared": [...]}` |

Query parameters for both `/predict*` endpoints:

| Param | Default | Meaning |
|---|---|---|
| `conf` | `JASS_CONF_THRESHOLD` (0.25) | Minimum confidence |
| `iou` | `JASS_IOU_THRESHOLD` (0.45) | NMS IoU threshold |
| `dedupe` | `true` | Keep only the best box per card (a deck contains each card once) |
| `max_cards` | – | Hand size, if known: keep only the N most confident cards |

## Example

```bash
curl -F file=@hand.jpg "http://localhost:8000/predict?conf=0.4"
```

```json
{
  "model": "best.pt",
  "image_width": 1280,
  "image_height": 960,
  "inference_ms": 41.3,
  "detections": [
    {
      "class_id": 26,
      "label": "rosen_ass",
      "display_name": "Rosen Ass",
      "suit": "rosen",
      "rank": "ass",
      "confidence": 0.93,
      "bbox": {"x1": 384.0, "y1": 192.0, "x2": 896.0, "y2": 768.0},
      "polygon": [[384.0, 230.5], [880.2, 192.0], [896.0, 731.1], [399.4, 768.0]]
    }
  ],
  "cards": ["rosen_ass"]
}
```

Boxes are absolute pixel coordinates in the uploaded image after EXIF rotation. `polygon` is
set for rotated-box (OBB) models; `bbox` is always the upright bounding box. `cards` is
sorted by suit, then rank.

### Which card was played?

Photograph the hand, play a card, photograph again, then compare the two `cards` lists:

```bash
curl -X POST localhost:8000/hand/diff -H "Content-Type: application/json" \
  -d '{"before": ["rosen_ass", "eicheln_6"], "after": ["eicheln_6"]}'
# {"played": ["rosen_ass"], "appeared": []}
```

`appeared` non-empty means a recognition error (a card can't appear in a hand mid-trick).

Errors: `400` undecodable image, `413` file larger than `JASS_MAX_UPLOAD_MB`, `422` bad params.

## Configuration

All settings are `JASS_*` environment variables (see `.env.example`):

| Variable | Default | |
|---|---|---|
| `JASS_MODEL_BACKEND` | `auto` | `auto` (weights if present, else dummy), `yolo`, `dummy` |
| `JASS_MODEL_PATH` | `models/best.pt` | `.pt` or `.onnx` |
| `JASS_DEVICE` | `cpu` | `cpu`, `0` (CUDA), `mps` |
| `JASS_IMG_SIZE` | `640` | inference size |
| `JASS_MAX_UPLOAD_MB` | `10` | |
