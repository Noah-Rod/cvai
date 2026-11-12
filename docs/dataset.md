# Dataset & training

## Classes

36 classes = 4 suits × 9 ranks. Ids are fixed in `src/jass_cv/cards.py`, and
`data/jass.yaml` must use the same order (a test checks this).

| id | label | | id | label | | id | label | | id | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | eicheln_6 | | 9 | schilten_6 | | 18 | rosen_6 | | 27 | schellen_6 |
| 1 | eicheln_7 | | 10 | schilten_7 | | 19 | rosen_7 | | 28 | schellen_7 |
| 2 | eicheln_8 | | 11 | schilten_8 | | 20 | rosen_8 | | 29 | schellen_8 |
| 3 | eicheln_9 | | 12 | schilten_9 | | 21 | rosen_9 | | 30 | schellen_9 |
| 4 | eicheln_banner | | 13 | schilten_banner | | 22 | rosen_banner | | 31 | schellen_banner |
| 5 | eicheln_under | | 14 | schilten_under | | 23 | rosen_under | | 32 | schellen_under |
| 6 | eicheln_ober | | 15 | schilten_ober | | 24 | rosen_ober | | 33 | schellen_ober |
| 7 | eicheln_koenig | | 16 | schilten_koenig | | 25 | rosen_koenig | | 34 | schellen_koenig |
| 8 | eicheln_ass | | 17 | schilten_ass | | 26 | rosen_ass | | 35 | schellen_ass |

Pairs we expect the model to confuse: **Under vs Ober** (same figure type; the suit symbol
sits in a different place), **6 / 7 / 8 / 9** (same pip, different count), and suits under
poor or coloured lighting.

## Layout (YOLO format)

```
data/
  jass.yaml
  raw/                         # original photos, never edited
  jass/
    images/{train,val,test}/IMG_0001.jpg
    labels/{train,val,test}/IMG_0001.txt   # one line per box:
                                           # <class_id> <cx> <cy> <w> <h>   (normalised 0..1)
```

`data/` and `models/` are git-ignored. Where they are stored is an open decision
(see [architecture.md](architecture.md#d6-data--model-storage)).

## Labeling guidelines (draft — agree before labeling!)

1. **Box = the visible part of the card face**, tight, axis-aligned. *(Alternative: box only
   the corner region. That's decision D1.)*
2. Label a card only if you yourself can tell which card it is; otherwise leave it unlabeled.
3. Card backs are not labeled (or get their own class later, if we need that).
4. Every card in the image is labeled. A missing box teaches the model "this is background".

## Collecting data

- Use `tools/live_detect.py` and press `s` to save frames to `captures/` (works with the
  dummy model, so you can start before any model exists).
- Vary everything: tables or tablecloths, lighting (daylight, lamp, phone flash), angle,
  distance, rotation, partially covered cards, several decks if available, phone and webcam.
- **Split by session, not by frame.** Frames from the same video are near-duplicates. If they
  end up in both train and test, the test score will look far better than it really is.
  Put whole capture sessions into train, val or test.
- **The test set is real photos only**, even if training uses synthetic data.

Rough target for a first model: ≥ 40 labeled instances per class in train
(≈ 1,500 boxes total, many cards per photo), plus ~150 real photos for val and test.

## Training

```bash
# smoke test (CPU, 1 epoch, 10 % of the data)
docker compose --profile train run --rm train
# real run on a GPU machine / Colab / Kaggle
uv run --extra inference python tools/train.py --model yolo11n.pt --epochs 100 --device 0
```

`tools/train.py` fine-tunes a pretrained checkpoint. It turns off horizontal flips, because
the card figures are not mirror-symmetric. It copies the best weights to `models/best.pt`,
reports mAP on the test split, and with `--export onnx` writes an ONNX file. Runs, plots and
the confusion matrix go to `runs/jass/`.

Metrics to report: mAP@50 and mAP@50-95 on **test**, the confusion matrix, per-class
recall, and a handful of failure-case images.
