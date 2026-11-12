"""Evaluate a model on real hand photos at the task level: were the right cards recognised?

Ground truth is a CSV with one row per photo (cards separated by spaces):

  file,cards
  IMG_0001.jpg,rosen_ass rosen_6 eicheln_under schellen_koenig ...

  python tools/evaluate.py --weights models/v1-corner_obb.pt --images data/real/test/images \\
      --hands data/real/test/hands.csv --known-count

Reports card precision/recall/F1, the share of hands where *every* card is right, per-card
recall and the most likely confusions; writes everything to runs/eval/<weights>.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jass_cv.cards import CLASS_NAMES  # noqa: E402
from jass_cv.detector import YoloDetector  # noqa: E402
from jass_cv.hand import score_hand, select_hand  # noqa: E402


def read_hands(path: Path) -> dict[str, set[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        hands = {row["file"]: set(row["cards"].split()) for row in csv.DictReader(fh)}
    unknown = {c for cards in hands.values() for c in cards} - set(CLASS_NAMES)
    if unknown:
        raise SystemExit(f"Unknown card labels in {path}: {sorted(unknown)}")
    return hands


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--hands", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--known-count",
        action="store_true",
        help="tell the model how many cards the hand has (keep the N best)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "eval")
    args = ap.parse_args()

    detector = YoloDetector(args.weights, args.device, args.imgsz)
    hands = read_hands(args.hands)
    tp = fp = fn = exact = 0
    per_card_hit, per_card_total, confusions = Counter(), Counter(), Counter()
    per_image = []
    for file, truth in hands.items():
        image = ImageOps.exif_transpose(Image.open(args.images / file)).convert("RGB")
        dets = select_hand(
            detector.predict(image, args.conf, args.iou), len(truth) if args.known_count else None
        )
        pred = {d.label for d in dets}
        s = score_hand(pred, truth)
        tp, fp, fn, exact = tp + s.tp, fp + s.fp, fn + s.fn, exact + s.exact
        for card in truth:
            per_card_total[card] += 1
            per_card_hit[card] += card in pred
        missed, extra = truth - pred, pred - truth
        if len(missed) == 1 and len(extra) == 1:  # one swap → very likely a confusion
            confusions[f"{next(iter(missed))} → {next(iter(extra))}"] += 1
        per_image.append({"file": file, "missed": sorted(missed), "extra": sorted(extra)})

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "weights": str(args.weights),
        "known_count": args.known_count,
        "images": len(hands),
        "card_precision": round(precision, 4),
        "card_recall": round(recall, 4),
        "card_f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
        "exact_hand_accuracy": round(exact / max(len(hands), 1), 4),
        "per_card_recall": {
            c: round(per_card_hit[c] / per_card_total[c], 3) for c in sorted(per_card_total)
        },
        "likely_confusions": dict(confusions.most_common(10)),
        "per_image": per_image,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.weights.stem}{'-known' if args.known_count else ''}.json"
    out.write_text(json.dumps(result, indent=2))
    for key in ("card_precision", "card_recall", "card_f1", "exact_hand_accuracy"):
        print(f"{key:22s} {result[key]:.3f}")
    print("likely confusions:", result["likely_confusions"] or "–")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
