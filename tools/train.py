"""Train a YOLO model on an exported dataset and copy the best weights to models/<name>.pt.

  # upright boxes (full_aabb / visible_aabb / corner_aabb exports)
  python tools/train.py --data data/exports/v1-corner_aabb/data.yaml \
      --model yolo11n.pt --name v1-corner_aabb
  # rotated boxes (full_obb / corner_obb exports) need an OBB checkpoint
  python tools/train.py --data data/exports/v1-corner_obb/data.yaml \
      --model yolo11n-obb.pt --name v1-corner_obb

Same command on Windows, Linux, macOS and Kaggle; only --device changes (cpu, 0 for CUDA, mps).
Quick CPU smoke test: --epochs 1 --fraction 0.1 --imgsz 320
Real-photo evaluation is done separately with tools/evaluate.py.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--data", type=Path, required=True, help="data.yaml from jass_cv.dataset export"
    )
    ap.add_argument("--model", default="yolo11n.pt", help="checkpoint to fine-tune (or a .yaml)")
    ap.add_argument("--name", required=True, help="run name, also the weights file name")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the train set")
    ap.add_argument("--export", choices=["onnx"], help="also export the best model")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        # Card faces are not mirror-symmetric; the generator already varies rotation/perspective.
        fliplr=0.0,
        degrees=5.0,
    )
    dest = ROOT / "models" / f"{args.name}.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, dest)
    print(f"Best weights → {dest}   (plots and metrics: runs/{args.name}/)")

    if args.export:
        print(f"Exported {YOLO(dest).export(format=args.export, imgsz=args.imgsz)}")


if __name__ == "__main__":
    main()
