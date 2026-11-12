"""Train a YOLO model on the Jass dataset and copy the best weights to models/.

  python tools/train.py --model yolo11n.pt --epochs 100
  python tools/train.py --model yolo11s.pt --device 0 --export onnx

For real training use a GPU (local, Colab or Kaggle); CPU is fine only for a smoke test
(`--epochs 1 --fraction 0.1`).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/jass.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the train set")
    ap.add_argument("--name", default="jass")
    ap.add_argument("--export", choices=["onnx"], help="also export the best model")
    args = ap.parse_args()

    from ultralytics import YOLO

    os.chdir(ROOT)  # data/jass.yaml uses repo-relative paths
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        # Card faces are not mirror-symmetric: no horizontal flips, small rotations instead.
        fliplr=0.0,
        degrees=15.0,
    )
    best = Path(model.trainer.best)
    dest = ROOT / "models/best.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print(f"Best weights copied to {dest}")

    metrics = YOLO(dest).val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs"),
        name=f"{args.name}-test",
        exist_ok=True,
    )
    print(f"test mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")

    if args.export:
        out = YOLO(dest).export(format=args.export, imgsz=args.imgsz)
        print(f"Exported {out}")


if __name__ == "__main__":
    main()
