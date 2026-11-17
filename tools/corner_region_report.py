"""Compare candidate corner regions: how distinct are the 36 corners, how often are they visible?

A bigger corner box shows more of each card (easier to tell cards apart) but is more often
covered by the next card in the fan (fewer usable labels). For every candidate region this
prints two numbers:

- distinctiveness: for each card, the RMS pixel distance between its corner crop and the
  closest other card's corner crop. Higher is better; the closest pairs are listed.
- visibility: share of synthetic cards whose corner is at least `--min-corner-visible`
  visible (the threshold `jass_cv.dataset export` uses to keep a corner label).

  python tools/corner_region_report.py --cards-dir data/cards --n 300
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jass_cv import geometry, synth  # noqa: E402
from jass_cv.cards import CLASS_NAMES  # noqa: E402

DEFAULT_WIDTHS = (0.20, 0.24, 0.30, 0.36)
DEFAULT_HEIGHTS = (0.24, 0.30, 0.36, 0.42)


def corner_crops(faces: dict[int, np.ndarray], region) -> np.ndarray:
    """Corner crop of every card, resized to 48 × 48 and slightly blurred, in [0, 1]."""
    x0, y0, x1, y1 = region
    crops = []
    for cid in range(len(CLASS_NAMES)):
        f = faces[cid]
        h, w = f.shape[:2]
        crop = f[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
        crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
        crops.append(cv2.GaussianBlur(crop, (0, 0), 1).astype(np.float32).ravel() / 255)
    return np.stack(crops)


def distinctiveness(faces, region) -> tuple[np.ndarray, list[tuple[float, str, str]]]:
    """RMS pixel distance from every card's corner to the closest other card's corner.

    Plain distance on purpose: a contrast-normalised score (NCC) blows up the noise in
    near-empty corners and makes two blank corners look different.
    """
    crops = corner_crops(faces, region)
    dist = np.sqrt(((crops[:, None] - crops[None]) ** 2).mean(-1))
    np.fill_diagonal(dist, np.inf)
    pairs = sorted(
        (float(dist[i, j]), CLASS_NAMES[i], CLASS_NAMES[j])
        for i, j in itertools.combinations(range(len(crops)), 2)
    )
    return dist.min(axis=1), pairs


def sample_layouts(faces, n: int, seed: int) -> list[tuple[tuple[int, int], list[np.ndarray]]]:
    """Card quads (bottom card first) of n synthetic hands, rendered once."""
    gen = synth.HandGenerator(faces, synth.SynthConfig(aug="none", thumb_prob=0.0))
    layouts = []
    for i in range(n):
        gen.rng = np.random.default_rng([seed, i])
        img, cards = gen.sample()
        layouts.append((img.shape[:2], [np.float32(c["quad"]) for c in cards]))
    return layouts


def visibility(layouts, region, min_corner_visible: float) -> float:
    """Share of cards whose corner is visible enough, from the card quads alone.

    Paints the cards in pile order (rectangles, so rounded corners and the thumb are
    ignored) and measures how much of each corner the cards above it leave visible.
    """
    geometry.CORNER_REGION = region  # corner_rect() reads it at call time
    kept = total = 0
    for (h, w), quads in layouts:
        idmap = np.zeros((h, w), np.int16)
        for k, quad in enumerate(quads, start=1):
            cv2.fillConvexPoly(idmap, np.round(quad).astype(np.int32), k)
        for k, quad in enumerate(quads, start=1):
            corner = geometry.corner_quad_from_card_quad(quad)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, np.round(corner).astype(np.int32), 1)
            area = int(mask.sum())
            seen = int(((idmap == k) & (mask > 0)).sum()) / area if area else 0.0
            kept += seen >= min_corner_visible
            total += 1
    return kept / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cards-dir", type=Path, default=Path("data/cards"))
    ap.add_argument("--n", type=int, default=300, help="synthetic hands for the visibility check")
    ap.add_argument("--widths", type=float, nargs="+", default=DEFAULT_WIDTHS)
    ap.add_argument("--heights", type=float, nargs="+", default=DEFAULT_HEIGHTS)
    ap.add_argument("--min-corner-visible", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=8, help="most similar pairs to list per region")
    args = ap.parse_args()

    faces = synth.load_faces(args.cards_dir, placeholder=False)
    layouts = sample_layouts(faces, args.n, args.seed)
    print(f"{'region (x0,y0,x1,y1)':<26}{'min NN':>8}{'mean NN':>9}{'visible':>9}")
    for w, h in itertools.product(args.widths, args.heights):
        region = (0.0, 0.0, w, h)
        nn, pairs = distinctiveness(faces, region)
        vis = visibility(layouts, region, args.min_corner_visible)
        print(f"{str(region):<26}{nn.min():8.3f}{nn.mean():9.3f}{vis:9.1%}")
        for score, a, b in pairs[: args.pairs]:
            print(f"    {score:.3f}  {a} ~ {b}")


if __name__ == "__main__":
    main()
