"""Cut the 36 card faces out of photos, for the synthetic generator.

Take one photo per card: card upright (top of the card towards the top of the photo), on a
plain background that contrasts with the card (dark table, black cloth), viewed from above,
no flash glare. Name each photo `image_<suit><rank>.jpg`: suits E=Eicheln, S=Schilten,
R=Rosen, G=Schellen; ranks 6, 7, 8, 9, B=Banner, U=Under, O=Ober, K=König, A=Ass.
For example, `image_SU.jpg` is Schellen Under and `image_EA.jpg` is Eicheln Ass. Then:

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

SUIT_CODES = {"eicheln": "E", "schilten": "S", "rosen": "R", "schellen": "G"}
RANK_CODES = {
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "banner": "B",
    "under": "U",
    "ober": "O",
    "koenig": "K",
    "ass": "A",
}


def input_filename(label: str) -> str:
    """Return the user's two-letter JPEG filename for a standard card label."""
    suit, rank = label.rsplit("_", 1)
    return f"image_{SUIT_CODES[suit]}{RANK_CODES[rank]}.jpg"


def order_corners(pts: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL by image position."""
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def find_card(img: np.ndarray) -> np.ndarray | None:
    expected_ratio = max(FACE_W, FACE_H) / min(FACE_W, FACE_H)
    min_area = 0.2 * img.shape[0] * img.shape[1]

    def best_quad(contours: list[np.ndarray]) -> np.ndarray | None:
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            points = approx.reshape(4, 2).astype(np.float32)
            edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            ratio = max(edges) / max(min(edges), 1.0)
            # Real cards should be close to the template's long-to-short edge ratio.
            if abs(ratio - expected_ratio) > 0.35:
                continue
            candidates.append((area, points))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:  # card should be the white part
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = best_quad(contours)
    if quad is None:
        # Otsu thresholding can join a dark clip or miss the card boundary entirely.
        # Canny edges preserve the card's outer border in those photos.
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quad = best_quad(contours)
    return order_corners(quad) if quad is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/cards"))
    ap.add_argument("--inset", type=float, default=0.0, help="crop this fraction off each edge")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    done, failed = [], []
    for label in CLASS_NAMES:
        path = args.input / input_filename(label)
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
        print("missing or unreadable input photos:")
        for label in failed:
            print(f"  {input_filename(label)} -> {label}")
        print("(sideways photos are rotated to portrait; check upside-down cards in _overview.jpg)")


if __name__ == "__main__":
    main()
