import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "compare_synth_real", Path(__file__).resolve().parents[1] / "tools" / "compare_synth_real.py"
)
cmp = importlib.util.module_from_spec(_spec)
sys.modules["compare_synth_real"] = cmp
_spec.loader.exec_module(cmp)


def scene(rng: np.random.Generator, brightness: float, blur: float) -> np.ndarray:
    img = rng.integers(0, 256, (240, 320, 3)).astype(np.float32)
    img = cv2.resize(cv2.resize(img, (32, 24)), (320, 240), interpolation=cv2.INTER_NEAREST)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return np.clip(img * brightness, 0, 255).astype(np.uint8)


def test_dark_blurry_real_photos_are_flagged(tmp_path, capsys):
    rng = np.random.default_rng(0)
    (tmp_path / "real").mkdir()
    (tmp_path / "synth" / "train" / "images").mkdir(parents=True)
    for i in range(8):
        cv2.imwrite(str(tmp_path / "real" / f"r{i}.jpg"), scene(rng, 0.4, 4))
    for i in range(30):
        img = scene(rng, rng.uniform(0.8, 1.2), 0)
        cv2.imwrite(str(tmp_path / "synth" / "train" / "images" / f"s{i}.jpg"), img)

    cmp.main(["--real", str(tmp_path / "real"), "--synth", str(tmp_path / "synth"),
              "--out", str(tmp_path / "out")])  # fmt: skip
    text = (tmp_path / "out" / "report.md").read_text()
    assert "real photos are darker" in text and "real photos are blurrier" in text
    assert "8 real photos, 30 synthetic hands" in text
    assert cv2.imread(str(tmp_path / "out" / "sheet.jpg")) is not None


def test_matching_images_raise_no_hint():
    rng = np.random.default_rng(1)
    stats = [cmp.image_stats(scene(rng, rng.uniform(0.8, 1.2), 0)) for _ in range(40)]
    rows = cmp.compare(stats[:10], stats[10:])
    assert [hint for *_, hint in rows] == [""] * len(cmp.STATS)
