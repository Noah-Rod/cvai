"""Convert a Label Studio JSON export (polygon per card) into our annotations.jsonl.

  python tools/labelstudio_to_annotations.py --export export.json \\
      --images photos/real/train --out data/real/train

Expects the labeling config in labeling/label_studio_config.xml: one 4-point polygon per card,
clicked top-left → top-right → bottom-right → bottom-left as printed on the card. The corner
box is derived from the outline, so real photos can be exported in every box variant
(except visible_aabb). Images are matched by file name; Label Studio prefixes uploaded files
with "<hash>-", which is stripped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jass_cv.cards import card_by_label  # noqa: E402
from jass_cv.geometry import corner_quad_from_card_quad  # noqa: E402


def original_name(ls_path: str) -> str:
    name = Path(ls_path).name
    head, sep, rest = name.partition("-")
    return rest if sep and len(head) == 8 else name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--export", type=Path, required=True, help="Label Studio JSON export")
    ap.add_argument("--images", type=Path, required=True, help="folder with the original photos")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="validate image matches and annotations without writing converted data",
    )
    args = ap.parse_args()

    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    records, n_card = [], 0
    skipped = []
    for task in tasks:
        name = original_name(task["data"]["image"])
        src = args.images / name
        if not src.exists() or not task.get("annotations"):
            skipped.append(f"{name}: {'image missing' if not src.exists() else 'not labeled'}")
            continue
        result = task["annotations"][0].get("result", [])
        polygons = [r for r in result if r.get("type") == "polygonlabels"]
        if not polygons:
            skipped.append(f"{name}: no card polygons")
            continue
        cards, size = [], None
        with Image.open(src) as image:
            actual_size = image.size
        for index, polygon in enumerate(polygons, start=1):
            try:
                w, h = int(polygon["original_width"]), int(polygon["original_height"])
                points = polygon["value"]["points"]
                labels = polygon["value"]["polygonlabels"]
                if len(points) != 4:
                    raise ValueError(f"expected four corners, got {len(points)}")
                if len(labels) != 1:
                    raise ValueError("expected exactly one card label")
                if w <= 0 or h <= 0:
                    raise ValueError(f"invalid image dimensions {w}x{h}")
                if size is not None and size != (w, h):
                    raise ValueError("polygon dimensions differ within the same image")
                if actual_size != (w, h):
                    raise ValueError(
                        f"Label Studio image size {w}x{h} differs from source "
                        f"{actual_size[0]}x{actual_size[1]}"
                    )
                size = (w, h)
                label = labels[0]
                card = card_by_label(label)
                pts = [[float(x) / 100 * w, float(y) / 100 * h] for x, y in points]
                if any(not 0 <= v <= 100 for point in points for v in point):
                    raise ValueError("corner coordinates must be percentages from 0 to 100")
                cards.append(
                    {
                        "class_id": card.class_id,
                        "label": label,
                        "quad": [[round(x, 1), round(y, 1)] for x, y in pts],
                        "corner_quad": [
                            [round(float(x), 1), round(float(y), 1)]
                            for x, y in corner_quad_from_card_quad(pts)
                        ],
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid polygon #{index} in {name}: {exc}") from exc
        record = {"file": f"images/{name}", "width": size[0], "height": size[1], "cards": cards}
        records.append((src, name, record))
        n_card += len(cards)

    for message in skipped:
        print(f"skip {message}")
    if args.check_only:
        print(f"Valid: {len(records)} images, {n_card} cards; skipped {len(skipped)} tasks")
        return

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.out / "annotations.jsonl", "w", encoding="utf-8") as fh:
        for src, name, record in records:
            shutil.copy2(src, args.out / "images" / name)
            fh.write(json.dumps(record) + "\n")
    print(f"{len(records)} images, {n_card} cards → {args.out / 'annotations.jsonl'}")


if __name__ == "__main__":
    main()
