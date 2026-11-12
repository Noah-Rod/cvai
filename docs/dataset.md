# Data

Three kinds of data. None of it is in git (see `data/README.md`).

| Data | Used for | Effort |
|---|---|---|
| 36 card faces (`data/cards/`) | input of the generator | 36 photos + 1 command |
| Synthetic hands (`data/synth/`) | training + validation | generated |
| Real POV photos (`data/real/`) | **test** (all experiments) + a few for training (experiment 1) | photos + labeling |

## Classes

36 classes = 4 suits × 9 ranks, ids fixed in `src/jass_cv/cards.py`:

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

## 1. Card faces

1. Photograph each card **upright**, from straight above, on a dark, plain background, with
   no glare. Name each JPEG `image_<suit><rank>.jpg`. Suit codes: `E` = Eicheln, `S` =
   Schilten, `R` = Rosen, `G` = Schellen. Rank codes: `6`, `7`, `8`, `9`, `B` = Banner,
   `U` = Under, `O` = Ober, `K` = König, `A` = Ass. For example, `image_SU.jpg` is
   Schellen Under and `image_EA.jpg` is Eicheln Ass. The extractor writes standard label
   names such as `data/cards/schellen_under.png`.
2. `uv run --extra synth python tools/extract_cards.py --input photos/cards --out data/cards`
3. Open `data/cards/_overview.jpg`. Every card should be straight and upright. The red box
   should cover rank + suit symbol in the corner; if it doesn't, adjust `CORNER_REGION` (and
   the card ratio `FACE_W/FACE_H`) in `src/jass_cv/geometry.py`.

Optional but valuable for realism:

- `data/backgrounds/`: 50–200 photos of what's behind a hand in POV (tables, floors, laps,
  rooms). Your own phone photos are best.
- `data/hands/`: a few RGBA PNG cut-outs of thumbs/hands (e.g. cut out with any
  background-removal tool). Used as occluders instead of the default skin-coloured ellipse.

## 2. Synthetic hands

```bash
uv run --extra synth python -m jass_cv.synth --out data/synth/v1 --n 3000 --aug full
# smoke test without card photos:
uv run --extra synth python -m jass_cv.synth --out data/synth/smoke --n 50 --placeholder-cards
```

Each image has 3–12 cards (`--min-cards/--max-cards`) fanned around a pivot, a random
rotation, perspective, a thumb, and photometric augmentation (`--aug none|basic|full`).
`--seed` makes it reproducible; `--workers` parallelises it (default: all cores but one).

Every card in `annotations.jsonl` has:

| Field | Meaning |
|---|---|
| `quad` | 4 corners of the full card (TL, TR, BR, BL as printed), also where hidden |
| `corner_quad` | the rank/suit corner region |
| `visible` | fraction of the card that is visible |
| `corner_visible` | fraction of the corner that is visible |
| `visible_bbox` | upright box of the visible pixels |
| `layer` | 0 = bottom of the fan |

**Look at the images before training** (open a few in `data/synth/v1/train/images`). Do they
look like your real photos? That's the main lever for experiment 1.

## 3. Export to YOLO (one dataset per box variant)

```bash
uv run --extra synth python -m jass_cv.dataset export --variant corner_obb \
    --train data/synth/v1/train --val data/synth/v1/val --out data/exports/v1-corner_obb
```

Variants: `full_aabb`, `full_obb`, `visible_aabb`, `corner_aabb`, `corner_obb` (see
[architecture.md](architecture.md#experiment-2-what-should-the-box-cover)). Cards whose
corner is less than 60 % visible get no corner box (`--min-corner-visible`). Cards that are
almost fully hidden (< 5 % visible) get no box at all (`--min-visible`).
Images are hard-linked, not copied, so the five exports cost almost no disk space.

For experiment 1, add real photos to training by repeating `--train`:
`--train data/synth/v1/train --train data/real/train`.

## 4. Real photos

### Capture protocol

- **Own POV**: hold the fan as you would in a game; phone at eye position.
- Vary: number of cards (3–12), which cards, fan width, lighting (daylight, lamp, evening),
  background, both of your hands, phone and webcam.
- **Capture sessions**: one session = one place/light/person. **Split by session, not by
  photo**: photos of the same session are near-duplicates, so a session goes completely into
  train *or* test.
- Target: **test ≈ 80–100 photos** (≥ 5 sessions), **real train ≈ 50 photos** (other sessions).

### Test set: just write down the cards

The evaluation only needs to know *which cards are in the hand*, not boxes. That's fast:

```
data/real/test/images/IMG_0001.jpg ...
data/real/test/hands.csv
  file,cards
  IMG_0001.jpg,rosen_ass rosen_6 eicheln_under schellen_koenig schilten_9
```

Tip: lay the hand on the table, take a top-down photo for reference, then fan it and take
the POV photo.

### Real training photos (experiment 1): Label Studio

```bash
docker compose --profile label up -d labelstudio      # http://localhost:8080
```

1. Create a project and paste `labeling/label_studio_config.xml` under
   *Settings → Labeling Interface → Code*.
2. Import the photos. Rotate phone photos to their final orientation first (EXIF rotation),
   so the coordinates match.
3. For each card, click the 4 corners **top-left, top-right, bottom-right, bottom-left as
   printed on the card**, including estimated hidden corners. Choose the card label.
4. Export as **JSON**, then:

```bash
uv run --extra synth python tools/labelstudio_to_annotations.py \
    --export export.json --images photos/real/train --out data/real/train
```

The corner box is computed from the outline, so these photos work for every box variant
except `visible_aabb`. *The converter hasn't been tested against a real Label Studio export
yet: check the first conversion by drawing it (see `tests/test_synth_dataset.py` for the
format).*
