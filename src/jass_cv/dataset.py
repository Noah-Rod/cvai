"""Annotation format and export to YOLO datasets in different box variants.

Annotations (synthetic and real) are JSONL, one image per line:

  {"file": "images/x.jpg", "width": 1024, "height": 768,
   "cards": [{"class_id": 26, "label": "rosen_ass",
              "quad": [[x, y] × 4],          # card outline, TL TR BR BL as printed
              "corner_quad": [[x, y] × 4],   # rank/suit corner (derived from quad if missing)
              "visible": 0.31,               # fraction of the card that is visible (optional)
              "corner_visible": 0.95,        # same for the corner (optional)
              "visible_bbox": [x1, y1, x2, y2]}]}  # optional, synthetic only

Variants (experiment 2):
  full_aabb     upright box around the whole card (incl. hidden parts)
  full_obb      rotated box around the whole card
  visible_aabb  upright box around the visible part (synthetic only)
  corner_aabb   upright box around the rank/suit corner
  corner_obb    rotated box around the rank/suit corner

  python -m jass_cv.dataset export --variant corner_obb \\
      --train data/synth/v1/train --val data/synth/v1/val --out data/exports/v1-corner_obb
  # experiment 1: add real photos to training by repeating --train
      --train data/synth/v1/train --train data/real/train
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from jass_cv.cards import CLASS_NAMES
from jass_cv.geometry import aabb, corner_quad_from_card_quad

VARIANTS = ("full_aabb", "full_obb", "visible_aabb", "corner_aabb", "corner_obb")


def read_annotations(split_dir: Path) -> Iterator[dict]:
    with open(split_dir / "annotations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _fmt(values) -> str:
    return " ".join(f"{v:.6f}" for v in values)


def label_lines(record: dict, variant: str, min_visible=0.05, min_corner_visible=0.6) -> list[str]:
    """YOLO label lines for one image (detect: `c cx cy w h`, OBB: `c x1 y1 … x4 y4`)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    w, h = record["width"], record["height"]
    lines = []
    for card in record["cards"]:
        cls = card["class_id"]
        if variant.startswith("corner"):
            if card.get("corner_visible", 1.0) < min_corner_visible:
                continue
            pts = card.get("corner_quad") or corner_quad_from_card_quad(card["quad"])
        elif variant == "visible_aabb":
            if card.get("visible", 1.0) < min_visible or not card.get("visible_bbox"):
                continue
            x1, y1, x2, y2 = card["visible_bbox"]
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            if card.get("visible", 1.0) < min_visible:
                continue
            pts = card["quad"]

        pts = np.asarray(pts, dtype=np.float64)
        if variant.endswith("obb"):
            pts = np.clip(pts, 0, [w, h]) / [w, h]
            lines.append(f"{cls} {_fmt(pts.ravel())}")
        else:
            box = aabb(pts, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            values = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
            lines.append(f"{cls} {_fmt(values)}")
    return lines


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard link (no extra disk, no admin rights needed on NTFS); copy across drives."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export(
    out: Path, variant: str, splits: dict[str, list[Path]], min_visible=0.05, min_corner_visible=0.6
) -> Path:
    if "train" not in splits or "val" not in splits:
        raise ValueError("need at least one --train and one --val source")
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for split, sources in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        n_img = n_box = 0
        for si, src in enumerate(sources):
            for record in read_annotations(src):
                img = src / record["file"]
                name = f"s{si}_{Path(record['file']).stem}"
                _link_or_copy(img, out / "images" / split / f"{name}{img.suffix}")
                lines = label_lines(record, variant, min_visible, min_corner_visible)
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                n_img, n_box = n_img + 1, n_box + len(lines)
        stats[split] = {"images": n_img, "boxes": n_box}

    yaml_lines = [f"# variant: {variant}", f"path: {out.resolve().as_posix()}"]
    yaml_lines += [f"{s}: images/{s}" for s in splits]
    yaml_lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    (out / "export.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "sources": {k: [str(p) for p in v] for k, v in splits.items()},
                "min_visible": min_visible,
                "min_corner_visible": min_corner_visible,
                "stats": stats,
            },
            indent=2,
        )
    )
    return out / "data.yaml"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="build a YOLO dataset for one box variant")
    ex.add_argument("--variant", choices=VARIANTS, required=True)
    ex.add_argument("--train", type=Path, action="append", required=True)
    ex.add_argument("--val", type=Path, action="append", required=True)
    ex.add_argument("--test", type=Path, action="append")
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--min-visible", type=float, default=0.05)
    ex.add_argument("--min-corner-visible", type=float, default=0.6)
    args = ap.parse_args(argv)

    splits = {"train": args.train, "val": args.val}
    if args.test:
        splits["test"] = args.test
    yaml_path = export(args.out, args.variant, splits, args.min_visible, args.min_corner_visible)
    print(json.loads((args.out / "export.json").read_text())["stats"])
    print(f"→ {yaml_path}")


if __name__ == "__main__":
    main()
