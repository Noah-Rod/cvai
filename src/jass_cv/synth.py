"""Synthetic generator: fanned Jass hands seen from the player's own point of view.

Every image gets exact labels for free: the full card outline (`quad`), the rank/suit
corner (`corner_quad`), the visible part (`visible_bbox`) and how much of the card and of
its corner is visible. `python -m jass_cv.dataset export` turns these into YOLO datasets
for the different box variants we compare.

  # pipeline smoke test without real card photos
  python -m jass_cv.synth --out data/synth/smoke --n 50 --placeholder-cards
  # real run: needs data/cards/<label>.png (see tools/extract_cards.py)
  python -m jass_cv.synth --out data/synth/v1 --n 3000 --aug full

Output:  <out>/{train,val}/images/*.jpg  and  <out>/{train,val}/annotations.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from jass_cv.cards import CARDS, Card
from jass_cv.geometry import FACE_H, FACE_W, corner_rect, face_rect, transform

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class SynthConfig:
    size: tuple[int, int] = (1024, 768)  # long side, short side; orientation is random
    min_cards: int = 3
    max_cards: int = 12
    card_height: tuple[float, float] = (0.45, 0.8)  # card height / image short side
    spread_deg: tuple[float, float] = (5.0, 13.0)  # angle between neighbouring cards
    pivot_depth: tuple[float, float] = (0.8, 1.3)  # pivot below card top, in card heights
    rotation_deg: tuple[float, float] = (-25.0, 25.0)
    perspective: float = 0.10  # corner jitter of the global homography, fraction of size
    thumb_prob: float = 0.85
    aug: str = "full"  # none | basic | full   (ablation knob for experiment 1)


# ---------------------------------------------------------------------------- assets


def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in [
        (radius, radius),
        (w - radius, radius),
        (radius, h - radius),
        (w - radius, h - radius),
    ]:
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


FACE_MASK = _rounded_mask(FACE_W, FACE_H, radius=FACE_W // 22)

_SUIT_BGR = {
    "eicheln": (40, 110, 40),
    "schilten": (180, 110, 20),
    "rosen": (40, 30, 200),
    "schellen": (20, 150, 220),
}
_RANK_SHORT = {
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "banner": "10",
    "under": "U",
    "ober": "O",
    "koenig": "K",
    "ass": "A",
}


def _suit_symbol(img: np.ndarray, suit: str, center: tuple[int, int], r: int) -> None:
    color, (x, y) = _SUIT_BGR[suit], center
    if suit == "eicheln":
        cv2.circle(img, (x, y), r, color, -1)
    elif suit == "schilten":
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), color, -1)
    elif suit == "rosen":
        pts = np.int32([[x, y - r], [x + r, y], [x, y + r], [x - r, y]])
        cv2.fillConvexPoly(img, pts, color)
    else:
        pts = np.int32([[x, y - r], [x + r, y + r], [x - r, y + r]])
        cv2.fillConvexPoly(img, pts, color)


def placeholder_face(card: Card) -> np.ndarray:
    """Fake card face (rank + suit symbol in the corner) so the pipeline runs without photos."""
    img = np.full((FACE_H, FACE_W, 3), 238, np.uint8)
    color = _SUIT_BGR[card.suit]
    cv2.rectangle(img, (8, 8), (FACE_W - 9, FACE_H - 9), color, 3)
    cv2.putText(img, _RANK_SHORT[card.rank], (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    _suit_symbol(img, card.suit, (38, 98), 18)
    _suit_symbol(img, card.suit, (FACE_W // 2, FACE_H // 2 - 40), 60)
    cv2.putText(
        img, card.display_name, (40, FACE_H // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
    )
    return img


def load_faces(cards_dir: Path | None, placeholder: bool) -> dict[int, np.ndarray]:
    if placeholder:
        return {c.class_id: placeholder_face(c) for c in CARDS}
    faces, missing = {}, []
    for c in CARDS:
        path = next(
            (p for p in sorted(cards_dir.glob(f"{c.label}.*")) if p.suffix.lower() in IMAGE_EXTS),
            None,
        )
        if path is None:
            missing.append(c.label)
            continue
        faces[c.class_id] = cv2.resize(
            cv2.imread(str(path)), (FACE_W, FACE_H), interpolation=cv2.INTER_AREA
        )
    if missing:
        raise SystemExit(
            f"Missing card images in {cards_dir}: {', '.join(missing)}\n"
            "Create them with tools/extract_cards.py or use --placeholder-cards."
        )
    return faces


def _list_images(folder: Path | None) -> list[Path]:
    if folder is None or not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------- matrices


def _t(x: float, y: float) -> np.ndarray:
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)


def _r(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _s(f: float) -> np.ndarray:
    return np.diag([f, f, 1.0])


def _roi(pts: np.ndarray, w: int, h: int, pad: int) -> tuple[int, int, int, int]:
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    return int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))


# ---------------------------------------------------------------------------- generator


class HandGenerator:
    def __init__(
        self,
        faces: dict[int, np.ndarray],
        cfg: SynthConfig,
        backgrounds: list[Path] = (),
        hands: list[Path] = (),
        seed: int = 0,
    ) -> None:
        self.faces, self.cfg = faces, cfg
        self.backgrounds = list(backgrounds)
        self.hands = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in hands]
        self.hands = [h for h in self.hands if h is not None and h.ndim == 3 and h.shape[2] == 4]
        self.rng = np.random.default_rng(seed)

    def u(self, lo_hi: tuple[float, float]) -> float:
        return float(self.rng.uniform(*lo_hi))

    # -- background -------------------------------------------------------------
    def background(self, w: int, h: int) -> np.ndarray:
        if self.backgrounds:
            img = cv2.imread(str(self.backgrounds[self.rng.integers(len(self.backgrounds))]))
            if img is not None:
                s = max(w / img.shape[1], h / img.shape[0]) * self.u((1.0, 1.5))
                img = cv2.resize(img, (math.ceil(img.shape[1] * s), math.ceil(img.shape[0] * s)))
                x = self.rng.integers(img.shape[1] - w + 1)
                y = self.rng.integers(img.shape[0] - h + 1)
                return img[y : y + h, x : x + w].copy()
        # procedural fallback: gradient + blobs + noise
        c0, c1 = self.rng.integers(0, 256, 3), self.rng.integers(0, 256, 3)
        t = np.linspace(0, 1, w if self.rng.random() < 0.5 else h, dtype=np.float32)
        grad = (c0[None] * (1 - t[:, None]) + c1[None] * t[:, None]).astype(np.float32)
        img = np.broadcast_to(grad[None] if len(t) == w else grad[:, None], (h, w, 3)).copy()
        for _ in range(self.rng.integers(0, 8)):
            center = (int(self.rng.integers(w)), int(self.rng.integers(h)))
            axes = (int(self.rng.integers(10, w // 3)), int(self.rng.integers(10, h // 3)))
            color = self.rng.integers(0, 256, 3).tolist()
            cv2.ellipse(img, center, axes, float(self.rng.uniform(0, 180)), 0, 360, color, -1)
        img += self.rng.normal(0, 6, img.shape).astype(np.float32)
        return np.clip(cv2.GaussianBlur(img, (0, 0), 2), 0, 255).astype(np.uint8)

    # -- layout -------------------------------------------------------------------
    def layout(self, n: int, w: int, h: int) -> tuple[list[np.ndarray], np.ndarray, float]:
        """Homographies face→image for n fanned cards (index 0 = left/bottom of the pile)."""
        cfg = self.cfg
        pivot = np.array([FACE_W * self.u((0.3, 0.6)), FACE_H * self.u(cfg.pivot_depth)])
        spread = self.u(cfg.spread_deg)
        fan = []
        for i in range(n):
            angle = (i - (n - 1) / 2) * spread + self.rng.normal(0, 0.8)
            slide = _t(self.rng.normal(0, 0.01 * FACE_W), self.rng.normal(0, 0.02 * FACE_H))
            fan.append(_r(angle) @ _t(*-pivot) @ slide)  # pivot ends up at the origin

        pts = np.concatenate([transform(face_rect(), m) for m in fan])
        center = (pts.min(0) + pts.max(0)) / 2
        card_h = self.u(cfg.card_height) * min(w, h)
        sim = _t(w / 2, h / 2) @ _r(self.u(cfg.rotation_deg)) @ _s(card_h / FACE_H) @ _t(*-center)

        jitter = self.rng.uniform(-cfg.perspective, cfg.perspective, (4, 2)) * [w, h]
        img_rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        persp = cv2.getPerspectiveTransform(img_rect, np.float32(img_rect + jitter))
        hs = [persp @ sim @ m for m in fan]

        # fit the whole fan inside the image at a random position
        pts = np.concatenate([transform(face_rect(), m) for m in hs])
        lo, hi = pts.min(0), pts.max(0)
        f = min(1.0, 0.96 * w / (hi[0] - lo[0]), 0.96 * h / (hi[1] - lo[1]))
        lo, hi = lo * f, hi * f
        tx = self.rng.uniform(-lo[0], w - hi[0]) if w - hi[0] > -lo[0] else -lo[0]
        ty = self.rng.uniform(-lo[1], h - hi[1]) if h - hi[1] > -lo[1] else -lo[1]
        fit = _t(tx, ty) @ _s(f)
        hs = [fit @ m for m in hs]
        pivot_img = transform(np.zeros((1, 2)), fit @ persp @ sim)[0]
        return hs, pivot_img, card_h * f

    # -- occluder (thumb / hand) ------------------------------------------------
    def occluder(self, canvas, idmap, pivot, quads, card_h) -> None:
        towards = np.concatenate(quads).mean(0) - pivot
        towards /= np.linalg.norm(towards) + 1e-6
        side = np.array([-towards[1], towards[0]])
        center = (
            pivot + towards * card_h * self.u((0.15, 0.4)) + side * card_h * self.u((-0.25, 0.1))
        )
        h, w = idmap.shape
        mask = np.zeros((h, w), np.float32)
        layer = np.zeros_like(canvas)
        if self.hands and self.rng.random() < 0.7:
            hand = self.hands[self.rng.integers(len(self.hands))]
            s = card_h * self.u((0.5, 0.9)) / max(hand.shape[:2])
            m = cv2.getRotationMatrix2D((hand.shape[1] / 2, hand.shape[0] / 2), self.u((0, 360)), s)
            m[:, 2] += center - np.array([hand.shape[1] / 2, hand.shape[0] / 2])
            warped = cv2.warpAffine(hand, m, (w, h))
            layer, mask = warped[..., :3], warped[..., 3].astype(np.float32) / 255
        else:  # skin-coloured thumb ellipse
            hsv = np.uint8(
                [
                    [
                        [
                            self.rng.integers(5, 20),
                            self.rng.integers(60, 150),
                            self.rng.integers(110, 230),
                        ]
                    ]
                ]
            )
            layer[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            axes = (int(card_h * self.u((0.22, 0.32))), int(card_h * self.u((0.1, 0.15))))
            angle = math.degrees(math.atan2(towards[1], towards[0])) + self.rng.normal(0, 20)
            cv2.ellipse(mask, tuple(int(v) for v in center), axes, angle, 0, 360, 1.0, -1)
            shade = cv2.GaussianBlur(mask, (0, 0), axes[1] / 2)
            layer = (layer * (0.75 + 0.25 * shade[..., None])).astype(np.uint8)
            mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
        canvas[:] = (canvas * (1 - mask[..., None]) + layer * mask[..., None]).astype(np.uint8)
        idmap[mask > 0.5] = 0

    # -- photometric augmentation --------------------------------------------------
    def photometric(self, img: np.ndarray) -> np.ndarray:
        level = self.cfg.aug
        if level == "none":
            return img
        f = img.astype(np.float32)
        f = f * self.u((0.7, 1.25)) + self.u((-30, 30))  # contrast / brightness
        f *= self.rng.uniform(0.9, 1.1, 3)  # white balance
        if level == "full":
            h, w = f.shape[:2]
            temp = self.u((-0.12, 0.12))  # colour temperature
            f[..., 0] *= 1 - temp
            f[..., 2] *= 1 + temp
            a = self.u((0, 2 * math.pi))  # uneven lighting
            ramp = (
                np.cos(a) * np.linspace(-1, 1, w)[None] + np.sin(a) * np.linspace(-1, 1, h)[:, None]
            )
            f *= (1 + self.u((0, 0.35)) * ramp)[..., None]
            if self.rng.random() < 0.5:  # defocus or motion blur
                if self.rng.random() < 0.5:
                    f = cv2.GaussianBlur(f, (0, 0), self.u((0.5, 2.0)))
                else:
                    k = int(self.rng.integers(5, 13))
                    kernel = np.zeros((k, k), np.float32)
                    kernel[k // 2] = 1 / k
                    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), self.u((0, 180)), 1)
                    f = cv2.filter2D(f, -1, cv2.warpAffine(kernel, rot, (k, k)))
            f += self.rng.normal(0, self.u((0, 8)), f.shape)  # sensor noise
        return np.clip(f, 0, 255).astype(np.uint8)

    # -- one sample -------------------------------------------------------------
    def sample(self) -> tuple[np.ndarray, list[dict]]:
        cfg = self.cfg
        long, short = cfg.size
        w, h = (long, short) if self.rng.random() < 0.5 else (short, long)
        n = int(self.rng.integers(cfg.min_cards, cfg.max_cards + 1))
        ids = self.rng.choice(len(CARDS), n, replace=False)

        canvas = self.background(w, h)
        idmap = np.zeros((h, w), np.int16)
        hs, pivot, card_h = self.layout(n, w, h)
        full_area, rois = [], []
        pad = int(card_h * 0.05) + 2
        for k, (cid, hm) in enumerate(zip(ids, hs, strict=True), start=1):
            # work only inside the card's bounding box (+ shadow margin): ~10x faster
            x0, y0, x1, y1 = _roi(transform(face_rect(), hm), w, h, pad)
            local = _t(-x0, -y0) @ hm
            size = (x1 - x0, y1 - y0)
            mask = cv2.warpPerspective(FACE_MASK, local, size).astype(np.float32) / 255
            roi = canvas[y0:y1, x0:x1]
            if cfg.aug == "full":  # soft drop shadow of this card onto what's below
                shadow = cv2.GaussianBlur(mask, (0, 0), card_h * 0.02)
                shadow = np.roll(shadow, (int(card_h * 0.01), int(card_h * 0.01)), (0, 1))
                roi[:] = (roi * (1 - 0.35 * shadow[..., None])).astype(np.uint8)
            face = cv2.warpPerspective(self.faces[int(cid)], local, size, flags=cv2.INTER_LINEAR)
            roi[:] = (roi * (1 - mask[..., None]) + face * mask[..., None]).astype(np.uint8)
            idmap[y0:y1, x0:x1][mask > 0.5] = k
            full_area.append(max(1, int((mask > 0.5).sum())))
            rois.append((x0, y0, x1, y1))

        quads = [transform(face_rect(), hm) for hm in hs]
        if self.rng.random() < cfg.thumb_prob:
            self.occluder(canvas, idmap, pivot, quads, card_h)

        cards = []
        for k, (cid, hm, quad) in enumerate(zip(ids, hs, quads, strict=True), start=1):
            x0, y0, x1, y1 = rois[k - 1]
            corner = transform(corner_rect(), hm)
            visible = idmap[y0:y1, x0:x1] == k
            corner_mask = np.zeros(visible.shape, np.uint8)
            cv2.fillConvexPoly(corner_mask, np.round(corner - [x0, y0]).astype(np.int32), 1)
            corner_area = max(1, int(corner_mask.sum()))
            ys, xs = np.nonzero(visible)
            card = CARDS[int(cid)]
            cards.append(
                {
                    "class_id": card.class_id,
                    "label": card.label,
                    "layer": k - 1,
                    "quad": np.round(quad, 1).tolist(),
                    "corner_quad": np.round(corner, 1).tolist(),
                    "visible": round(len(xs) / full_area[k - 1], 4),
                    "corner_visible": round(
                        int((visible & (corner_mask > 0)).sum()) / corner_area, 4
                    ),
                    "visible_bbox": (
                        [
                            int(xs.min()) + x0,
                            int(ys.min()) + y0,
                            int(xs.max()) + x0 + 1,
                            int(ys.max()) + y0 + 1,
                        ]
                        if len(xs)
                        else None
                    ),
                }
            )
        return self.photometric(canvas), cards


# ---------------------------------------------------------------------------- CLI


_WORKER: HandGenerator | None = None


def _init_worker(args: tuple, out: Path) -> None:
    # runs in every worker process (Windows "spawn" re-imports this module, so pass everything)
    global _WORKER
    _WORKER = HandGenerator(*args)
    _WORKER.out = out


def _render(job: tuple[int, int, str]) -> tuple[str, dict]:
    """Render one image. Seeded per image, so the output doesn't depend on --workers."""
    seed, index, split = job
    gen = _WORKER
    gen.rng = np.random.default_rng([seed, index])
    img, cards = gen.sample()
    name = f"{split}_{index:06d}.jpg"
    q = int(gen.rng.integers(70, 95)) if gen.cfg.aug == "full" else 95
    cv2.imwrite(str(gen.out / split / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    record = {
        "file": f"images/{name}",
        "width": img.shape[1],
        "height": img.shape[0],
        "cards": cards,
    }
    return split, record


def generate(
    out: Path, n: int, val_fraction: float, gen_args: tuple, seed: int = 0, workers: int = 1
) -> None:
    """gen_args = (faces, cfg, backgrounds, hands) for HandGenerator."""
    n_val = round(n * val_fraction)
    jobs = [(seed, i, "train") for i in range(n - n_val)]
    jobs += [(seed, i, "val") for i in range(n - n_val, n)]
    files = {}
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        files[split] = open(out / split / "annotations.jsonl", "w", encoding="utf-8")
    try:
        if workers > 1:
            # "spawn" everywhere: what Windows does anyway, and fork + OpenCV threads can deadlock
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, _init_worker, (gen_args, out)) as pool:
                results = pool.imap(_render, jobs, chunksize=8)
                _write(results, files, len(jobs))
        else:
            _init_worker(gen_args, out)
            _write(map(_render, jobs), files, len(jobs))
    finally:
        for fh in files.values():
            fh.close()


def _write(results, files: dict, total: int) -> None:
    for done, (split, record) in enumerate(results, start=1):
        files[split].write(json.dumps(record) + "\n")
        if done % 250 == 0 or done == total:
            print(f"{done}/{total}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000, help="number of images (train + val)")
    ap.add_argument("--val", type=float, default=0.1, help="validation fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cards-dir", type=Path, default=Path("data/cards"))
    ap.add_argument("--backgrounds-dir", type=Path, default=Path("data/backgrounds"))
    ap.add_argument(
        "--hands-dir",
        type=Path,
        default=Path("data/hands"),
        help="RGBA PNG cut-outs of thumbs/hands (optional)",
    )
    ap.add_argument("--placeholder-cards", action="store_true", help="use fake card faces")
    ap.add_argument("--aug", choices=["none", "basic", "full"], default="full")
    ap.add_argument("--min-cards", type=int, default=3)
    ap.add_argument("--max-cards", type=int, default=12)
    ap.add_argument("--no-thumb", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)

    cfg = SynthConfig(
        aug=args.aug,
        min_cards=args.min_cards,
        max_cards=args.max_cards,
        thumb_prob=0.0 if args.no_thumb else SynthConfig.thumb_prob,
    )
    backgrounds, hands = _list_images(args.backgrounds_dir), _list_images(args.hands_dir)
    print(f"backgrounds: {len(backgrounds) or 'procedural'}  hand cut-outs: {len(hands)}")
    gen_args = (load_faces(args.cards_dir, args.placeholder_cards), cfg, backgrounds, hands)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"config": asdict(cfg), "args": {k: str(v) for k, v in vars(args).items()}}
    (args.out / "config.json").write_text(json.dumps(meta, indent=2))
    generate(args.out, args.n, args.val, gen_args, seed=args.seed, workers=args.workers)
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
