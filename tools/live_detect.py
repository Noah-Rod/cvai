"""Live-feed dev tool: run detection on a webcam / video / RTSP stream and show the result.

Runs on the host (not in Docker) because it needs the camera and a GUI window.

  # model in-process (fastest, needs weights locally)
  python tools/live_detect.py --source 0 --weights models/best.pt
  # send frames to the running API (tests the deployed model)
  python tools/live_detect.py --source 0 --api http://localhost:8000

Keys:  q = quit   s = save current frame to captures/ (handy for collecting training data)
       p = pause  +/- = change confidence threshold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jass_cv.drawing import SUIT_COLORS  # noqa: E402
from jass_cv.schemas import Detection, PredictionResponse  # noqa: E402


class LocalPredictor:
    def __init__(self, weights: Path, device: str, imgsz: int) -> None:
        from jass_cv.detector import YoloDetector

        self.detector = YoloDetector(weights, device, imgsz)
        self.name = f"local:{self.detector.name}"

    def __call__(self, frame_bgr: np.ndarray, conf: float, iou: float) -> list[Detection]:
        from jass_cv.detector import dedupe_by_card

        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return dedupe_by_card(self.detector.predict(img, conf, iou))


class ApiPredictor:
    def __init__(self, url: str) -> None:
        import requests

        self.session = requests.Session()
        self.url = url.rstrip("/")
        self.name = f"api:{self.session.get(f'{self.url}/health', timeout=5).json()['model']}"

    def __call__(self, frame_bgr: np.ndarray, conf: float, iou: float) -> list[Detection]:
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        r = self.session.post(
            f"{self.url}/predict",
            files={"file": ("frame.jpg", jpg.tobytes())},
            params={"conf": conf, "iou": iou},
            timeout=10,
        )
        r.raise_for_status()
        return PredictionResponse.model_validate(r.json()).detections


def draw(frame: np.ndarray, detections: list[Detection]) -> None:
    for d in detections:
        r, g, b = SUIT_COLORS.get(d.suit, (255, 255, 255))
        color = (b, g, r)
        p1 = (int(d.bbox.x1), int(d.bbox.y1))
        p2 = (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(
            frame,
            f"{d.display_name} {d.confidence:.2f}",
            (p1[0], max(15, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def hud(frame: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, video file or stream URL")
    ap.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--api", help="API base URL; if set, frames are sent to the API instead")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="cpu, 0 (cuda), mps")
    ap.add_argument("--captures", type=Path, default=Path("captures"))
    args = ap.parse_args()

    predictor = (
        ApiPredictor(args.api)
        if args.api
        else LocalPredictor(args.weights, args.device, args.imgsz)
    )
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open source {args.source!r}")

    conf, paused, fps, frame, detections = args.conf, False, 0.0, None, []
    while True:
        if not paused or frame is None:
            ok, raw = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            detections = predictor(raw, conf, args.iou)
            fps = 0.9 * fps + 0.1 * (1 / max(time.perf_counter() - t0, 1e-6))
            frame = raw
        shown = frame.copy()
        draw(shown, detections)
        hud(
            shown,
            [
                f"{predictor.name}  {fps:.1f} FPS  conf={conf:.2f}",
                f"{len(detections)} cards" + ("  [PAUSED]" if paused else ""),
            ],
        )
        cv2.imshow("Jass live detection", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
        elif key == ord("-"):
            conf = max(0.05, conf - 0.05)
        elif key == ord("s"):
            args.captures.mkdir(parents=True, exist_ok=True)
            path = args.captures / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**6}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
