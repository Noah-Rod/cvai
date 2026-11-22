"""Convert a Label Studio JSON export (polygon per card) into our annotations.jsonl.

  python tools/labelstudio_to_annotations.py --export export.json \\
      --images photos/real/train --out data/real/train --preview data/real/train-preview

Expects the labeling config in labeling/label_studio_config.xml: one 4-point polygon per card,
clicked top-left → top-right → bottom-right → bottom-left as printed on the card. The corner
box is derived from the outline, so real photos can be exported in every box variant
(except visible_aabb).

- Export format: Label Studio's **JSON** export (not JSON-MIN or CSV).
- Images are matched by file name; the "<8 hex>-" prefix Label Studio adds to uploaded files
  is stripped.
- Phone photos stored sideways with an EXIF orientation tag are fine: Label Studio shows them
  upright, so they are written to --out upright (orientation applied), matching the labels.
- `--preview DIR` draws every converted card outline (and its corner box) onto the photo:
  look at a few before training on them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jass_cv.cards import card_by_label  # noqa: E402
from jass_cv.geometry import corner_quad_from_card_quad  # noqa: E402

UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-(.+)$")


class ConversionError(ValueError):
    pass


def resolve_image(ls_path: str, images: Path) -> Path:
    """The original photo for a Label Studio image path ("/data/upload/1/0a1b2c3d-IMG_1.jpg")."""
    url = urlparse(ls_path)
    # local storage: /data/local-files/?d=<path>; uploads and URLs: the name is in the path
    name = Path(unquote(parse_qs(url.query).get("d", [url.path])[0])).name
    match = UPLOAD_PREFIX.match(name)
    if match and not (images / name).exists():  # a real name may also look like "<8 hex>-…"
        return images / match.group(1)
    return images / name


def upright(path: Path) -> Image.Image:
    """The photo as Label Studio (and every browser) shows it: EXIF orientation applied."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image)


def needs_rotation(path: Path) -> bool:
    with Image.open(path) as image:
        return image.getexif().get(0x0112, 1) not in (0, 1)


def card_from_polygon(polygon: dict, size: tuple[int, int]) -> dict:
    w, h = int(polygon["original_width"]), int(polygon["original_height"])
    points = polygon["value"]["points"]
    labels = polygon["value"]["polygonlabels"]
    if len(points) != 4:
        raise ConversionError(f"expected four corners, got {len(points)}")
    if len(labels) != 1:
        raise ConversionError("expected exactly one card label")
    if (w, h) != size:
        raise ConversionError(
            f"Label Studio image size {w}x{h} differs from the photo ({size[0]}x{size[1]} "
            "after EXIF rotation): was the photo edited after the import?"
        )
    if any(not 0 <= float(v) <= 100 for point in points for v in point):
        raise ConversionError("corner coordinates must be percentages from 0 to 100")
    card = card_by_label(labels[0])
    quad = [[float(x) / 100 * w, float(y) / 100 * h] for x, y in points]
    return {
        "class_id": card.class_id,
        "label": card.label,
        "quad": [[round(x, 1), round(y, 1)] for x, y in quad],
        "corner_quad": [
            [round(float(x), 1), round(float(y), 1)] for x, y in corner_quad_from_card_quad(quad)
        ],
    }


def convert(tasks: list, images: Path) -> tuple[list[tuple[Path, dict]], list[str]]:
    """(photo, annotation record) per labeled task, plus the reasons tasks were skipped."""
    if not isinstance(tasks, list) or (tasks and "data" not in tasks[0]):
        raise ConversionError(
            "this doesn't look like a Label Studio JSON export (a list of tasks with 'data' and "
            "'annotations'); export as 'JSON', not 'JSON-MIN' or 'CSV'"
        )
    records, skipped = [], []
    for task in tasks:
        src = resolve_image(task["data"]["image"], images)
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not src.exists():
            skipped.append(f"{src.name}: image missing in {images}")
            continue
        if not annotations:
            skipped.append(f"{src.name}: not labeled")
            continue
        result = annotations[-1].get("result", [])  # the latest submitted annotation
        polygons = [r for r in result if r.get("type") == "polygonlabels"]
        if not polygons:
            skipped.append(f"{src.name}: no card polygons")
            continue
        size = upright(src).size
        cards = []
        for index, polygon in enumerate(polygons, start=1):
            try:
                cards.append(card_from_polygon(polygon, size))
            except (KeyError, TypeError, ValueError) as exc:
                raise ConversionError(f"invalid polygon #{index} in {src.name}: {exc}") from exc
        labels = [c["label"] for c in cards]
        if len(set(labels)) != len(labels):
            dupes = sorted({x for x in labels if labels.count(x) > 1})
            raise ConversionError(f"{src.name}: card labeled twice: {', '.join(dupes)}")
        record = {"file": f"images/{src.name}", "width": size[0], "height": size[1]}
        records.append((src, {**record, "cards": cards}))
    return records, skipped


def write(records: list[tuple[Path, dict]], out: Path) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "annotations.jsonl", "w", encoding="utf-8") as fh:
        for src, record in records:
            dst = out / record["file"]
            if needs_rotation(src):  # store upright so pixels and labels agree everywhere
                upright(src).convert("RGB").save(dst, quality=95)
            else:
                shutil.copy2(src, dst)
            fh.write(json.dumps(record) + "\n")


def draw_preview(src: Path, record: dict) -> Image.Image:
    from jass_cv.drawing import SUIT_COLORS

    image = upright(src).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 300))
    try:
        font = ImageFont.load_default(size=max(12, round(min(image.size) / 40)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for card in record["cards"]:
        color = SUIT_COLORS[card_by_label(card["label"]).suit]
        quad = [tuple(p) for p in card["quad"]]
        draw.line([*quad, quad[0]], fill=color, width=width)
        corner = [tuple(p) for p in card["corner_quad"]]
        draw.line([*corner, corner[0]], fill=(255, 255, 255), width=width)
        draw.ellipse(  # the top-left corner as printed: shows if the click order was wrong
            [quad[0][0] - 3 * width, quad[0][1] - 3 * width, quad[0][0] + 3 * width,
             quad[0][1] + 3 * width], fill=color)  # fmt: skip
        draw.text((quad[0][0] + 4 * width, quad[0][1]), card["label"], fill=color, font=font)
    return image


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--export", type=Path, required=True, help="Label Studio JSON export")
    ap.add_argument("--images", type=Path, required=True, help="folder with the original photos")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--preview", type=Path, help="also draw the converted outlines into this folder"
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="validate image matches and annotations without writing converted data",
    )
    args = ap.parse_args(argv)

    try:
        records, skipped = convert(json.loads(args.export.read_text(encoding="utf-8")), args.images)
    except ConversionError as exc:
        raise SystemExit(f"error: {exc}") from exc
    for message in skipped:
        print(f"skip {message}")
    n_card = sum(len(record["cards"]) for _, record in records)
    if args.check_only:
        print(f"Valid: {len(records)} images, {n_card} cards; skipped {len(skipped)} tasks")
        return
    write(records, args.out)
    print(f"{len(records)} images, {n_card} cards → {args.out / 'annotations.jsonl'}")
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
        for src, record in records:
            draw_preview(src, record).save(args.preview / f"{src.stem}.jpg", quality=85)
        print(f"previews → {args.preview}")


if __name__ == "__main__":
    main()
