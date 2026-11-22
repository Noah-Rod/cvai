"""Put synthetic hands next to real photos and say where they differ.

  uv run --extra synth python tools/compare_synth_real.py \\
      --real data/real/test/images --synth data/synth/v1-aug_full --out runs/compare

Writes runs/compare/sheet.jpg (real photos on top, synthetic hands below) and
runs/compare/report.md: per image statistic, the real median against the spread of the
synthetic images, plus a hint where the real photos fall outside the middle 80 % of the
synthetic ones. Every image is first scaled to the same size, so phone resolution doesn't
count. The photometric augmentation lives in `HandGenerator.photometric` (src/jass_cv/synth.py).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SIZE = 1024  # long side for every statistic


@dataclass(frozen=True)
class Stat:
    key: str
    name: str
    low_hint: str  # real photos below the synthetic range
    high_hint: str  # real photos above it


STATS = (
    Stat("brightness", "brightness (L*, 0–100)",
         "real photos are darker: widen the brightness shift towards dark",
         "real photos are brighter: widen the brightness shift towards bright"),
    Stat("contrast", "contrast (std of L*)",
         "real photos are flatter: allow lower contrast factors",
         "real photos have more contrast: allow higher contrast factors"),
    Stat("saturation", "saturation (HSV S, 0–1)",
         "real photos are less colourful: add desaturation",
         "real photos are more colourful: check the card faces and backgrounds"),
    Stat("sharpness", "sharpness (log10 Laplacian var.)",
         "real photos are blurrier: blur more often or more strongly",
         "real photos are sharper: blur less often or less strongly"),
    Stat("warmth", "colour cast (b*, + = warm)",
         "real photos are cooler (bluer): widen the colour temperature towards blue",
         "real photos are warmer (lamp light): widen the colour temperature towards warm"),
)  # fmt: skip


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load(path: Path) -> np.ndarray:
    """BGR image, EXIF orientation applied, long side scaled to SIZE."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    img = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    scale = SIZE / max(img.shape[:2])
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def image_stats(img: np.ndarray) -> dict[str, float]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[..., 0] * 100 / 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(lightness.mean()),
        "contrast": float(lightness.std()),
        "saturation": float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255),
        "sharpness": float(np.log10(cv2.Laplacian(gray, cv2.CV_64F).var() + 1e-6)),
        "warmth": float(lab[..., 2].mean() - 128),
    }


def compare(real: list[dict], synth: list[dict]) -> list[tuple[Stat, float, float, float, str]]:
    """(stat, real median, synthetic 10th and 90th percentile, hint or '') per statistic."""
    rows = []
    for stat in STATS:
        r = float(np.median([s[stat.key] for s in real]))
        lo, hi = np.percentile([s[stat.key] for s in synth], [10, 90])
        hint = stat.low_hint if r < lo else stat.high_hint if r > hi else ""
        rows.append((stat, r, float(lo), float(hi), hint))
    return rows


def report(rows, n_real: int, n_synth: int) -> str:
    lines = [
        f"# Synthetic vs real ({n_real} real photos, {n_synth} synthetic hands)",
        "",
        "| statistic | real (median) | synthetic (10th–90th pct.) | |",
        "|---|---|---|---|",
    ]
    for stat, r, lo, hi, hint in rows:
        lines.append(
            f"| {stat.name} | {r:.2f} | {lo:.2f} – {hi:.2f} | {'⚠ ' + hint if hint else 'ok'} |"
        )
    if any(hint for *_, hint in rows):
        lines += ["", "To change: `HandGenerator.photometric` in src/jass_cv/synth.py."]
    else:
        lines += ["", "The real photos sit inside the synthetic range on every statistic."]
    lines.append(
        "Not measured here: backgrounds, thumbs, card size and fan shape; "
        "compare those by eye in sheet.jpg."
    )
    return "\n".join(lines) + "\n"


def contact_sheet(real: list[np.ndarray], synth: list[np.ndarray], cols: int = 6) -> np.ndarray:
    tile_h, gap = 260, 8

    def tile(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        out = cv2.resize(img, (round(w * tile_h / h), tile_h), interpolation=cv2.INTER_AREA)
        return cv2.copyMakeBorder(out, 0, 0, 0, gap, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    def block(images: list[np.ndarray], title: str) -> np.ndarray:
        rows = []
        for i in range(0, len(images), cols):
            row = np.hstack([tile(img) for img in images[i : i + cols]])
            rows.append(cv2.copyMakeBorder(row, 0, gap, 0, 0, cv2.BORDER_CONSTANT,
                                           value=(255, 255, 255)))  # fmt: skip
        width = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255)) for r in rows]  # fmt: skip
        head = np.full((48, width, 3), 255, np.uint8)
        cv2.putText(head, title, (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        return np.vstack([head, *rows])

    a, b = block(real, "real photos"), block(synth, "synthetic hands")
    width = max(a.shape[1], b.shape[1])
    pad = [cv2.copyMakeBorder(x, 0, 0, 0, width - x.shape[1], cv2.BORDER_CONSTANT,
                              value=(255, 255, 255)) for x in (a, b)]  # fmt: skip
    return np.vstack(pad)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--real", type=Path, action="append", required=True, help="repeatable")
    ap.add_argument("--synth", type=Path, required=True, help="jass_cv.synth output folder")
    ap.add_argument("--out", type=Path, default=Path("runs/compare"))
    ap.add_argument("--n", type=int, default=12, help="images per group on the contact sheet")
    ap.add_argument("--max", type=int, default=300, help="images per group for the statistics")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    groups = {}
    for name, paths in (
        ("real", [p for folder in args.real for p in list_images(folder)]),
        ("synth", list_images(args.synth)),
    ):
        if not paths:
            raise SystemExit(f"no {name} images found")
        groups[name] = rng.sample(paths, min(args.max, len(paths)))

    stats = {name: [image_stats(load(p)) for p in paths] for name, paths in groups.items()}
    rows = compare(stats["real"], stats["synth"])
    text = report(rows, len(stats["real"]), len(stats["synth"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    sheet = contact_sheet(*[[load(p) for p in groups[g][: args.n]] for g in ("real", "synth")])
    cv2.imwrite(str(args.out / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(text)
    print(f"→ {args.out / 'sheet.jpg'}, {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
