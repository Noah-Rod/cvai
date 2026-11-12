"""Cut the 36 card faces out of photos, for the synthetic generator.

Take one photo per card: card upright (top of the card towards the top of the photo), on a
plain background that contrasts with the card (dark table, black cloth), viewed from above,
no flash glare. Name each photo after the card label, e.g. `rosen_ass.jpg` (see
`docs/dataset.md` for all 36 labels). Then:

  python tools/extract_cards.py --input photos/cards --out data/cards

The tool finds the card outline (largest 4-cornered contour), straightens it with a
perspective warp and saves `data/cards/<label>.png`. It also writes `_overview.jpg` with the
corner region drawn on every card: check it and tune CORNER_REGION in
src/jass_cv/geometry.py so the box covers rank + suit symbol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jass_cv.cards import CLASS_NAMES  # noqa: E402
from jass_cv.geometry import FACE_H, FACE_W, corner_rect, face_rect  # noqa: E402


def order_corners(pts: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL by image position."""
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def find_card(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:  # card should be the white part
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(c) < 0.05 * img.shape[0] * img.shape[1]:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2).astype(np.float32))
        # rounded corners can produce >4 points: fall back to the min-area rectangle
        return order_corners(cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/cards"))
    ap.add_argument("--inset", type=float, default=0.0, help="crop this fraction off each edge")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    done, failed = [], []
    for label in CLASS_NAMES:
        path = next((p for p in args.input.glob(f"{label}.*")), None)
        img = cv2.imread(str(path)) if path else None
        quad = find_card(img) if img is not None else None
        if quad is None:
            failed.append(label)
            continue
        w, h = np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0])
        if w > h:  # photographed sideways: rotate so the card is portrait
            quad = np.roll(quad, -1, axis=0)
        face = cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(quad, face_rect()), (FACE_W, FACE_H)
        )
        if args.inset:
            dx, dy = int(FACE_W * args.inset), int(FACE_H * args.inset)
            face = cv2.resize(face[dy : FACE_H - dy, dx : FACE_W - dx], (FACE_W, FACE_H))
        cv2.imwrite(str(args.out / f"{label}.png"), face)
        done.append(face)

    if done:
        thumbs = []
        for face in done:
            t = face.copy()
            cv2.polylines(t, [np.int32(corner_rect())], True, (0, 0, 255), 3)
            thumbs.append(cv2.resize(t, (FACE_W // 2, FACE_H // 2)))
        thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % 9)
        rows = [np.hstack(thumbs[i : i + 9]) for i in range(0, len(thumbs), 9)]
        cv2.imwrite(str(args.out / "_overview.jpg"), np.vstack(rows))
    print(f"extracted {len(done)}/36 → {args.out}")
    if failed:
        print("missing or not found:", ", ".join(failed))
        print("(sideways photos are rotated to portrait; check upside-down cards in _overview.jpg)")


if __name__ == "__main__":
    main()
