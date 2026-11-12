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
    args = ap.parse_args()

    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    n_img = n_card = 0
    with open(args.out / "annotations.jsonl", "w", encoding="utf-8") as fh:
        for task in tasks:
            name = original_name(task["data"]["image"])
            src = args.images / name
            if not src.exists() or not task.get("annotations"):
                print(f"skip {name}: {'no image' if not src.exists() else 'not labeled'}")
                continue
            cards, size = [], None
            for r in task["annotations"][0]["result"]:
                if r.get("type") != "polygonlabels":
                    continue
                w, h = r["original_width"], r["original_height"]
                size = (w, h)
                pts = [[x / 100 * w, y / 100 * h] for x, y in r["value"]["points"]]
                label = r["value"]["polygonlabels"][0]
                if len(pts) != 4:
                    print(f"skip {label} in {name}: polygon has {len(pts)} points, need 4")
                    continue
                cards.append(
                    {
                        "class_id": card_by_label(label).class_id,
                        "label": label,
                        "quad": [[round(x, 1), round(y, 1)] for x, y in pts],
                        "corner_quad": [
                            [round(float(x), 1), round(float(y), 1)]
                            for x, y in corner_quad_from_card_quad(pts)
                        ],
                    }
                )
            if size is None:
                continue
            shutil.copy2(src, args.out / "images" / name)
            record = {"file": f"images/{name}", "width": size[0], "height": size[1], "cards": cards}
            fh.write(json.dumps(record) + "\n")
            n_img, n_card = n_img + 1, n_card + len(cards)
    print(f"{n_img} images, {n_card} cards → {args.out / 'annotations.jsonl'}")


if __name__ == "__main__":
    main()
