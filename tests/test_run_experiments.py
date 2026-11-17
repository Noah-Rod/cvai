import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_experiments", Path(__file__).resolve().parents[1] / "tools" / "run_experiments.py"
)
rx = importlib.util.module_from_spec(_spec)
sys.modules["run_experiments"] = rx  # dataclasses look their module up there
_spec.loader.exec_module(rx)


def test_exp2_covers_the_four_box_variants():
    assert [r.variant for r in rx.exp2_runs()] == list(rx.BOX_VARIANTS)
    assert {r.model for r in rx.exp2_runs()} == {"yolo11n.pt", "yolo11n-obb.pt"}


def test_exp1_reuses_the_exp2_run_for_full_augmentation():
    runs = rx.select(["exp2", "exp1"], "corner_aabb")
    names = [r.name for r in runs]
    assert len(names) == len(set(names)) == 4 + 4  # exp2-corner_aabb is shared
    assert rx.Run("exp2-corner_aabb", "corner_aabb") in rx.exp1_runs("corner_aabb")
    assert [r.real for r in rx.exp1_runs("corner_aabb")] == [0, 0, 0, 20, 50]


def test_unknown_run_is_rejected():
    with pytest.raises(SystemExit):
        rx.select(["exp3"], "corner_obb")


def test_real_subsets_are_nested_and_point_at_the_photos(tmp_path, monkeypatch):
    real = tmp_path / "real" / "train"
    real.mkdir(parents=True)
    records = [
        {"file": f"images/{i:03d}.jpg", "width": 1, "height": 1, "cards": []} for i in range(60)
    ]
    (real / "annotations.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    monkeypatch.setattr(rx, "REAL_TRAIN", real)
    monkeypatch.setattr(rx, "ROOT", tmp_path)

    def files(n):
        lines = (rx.real_subset(n) / "annotations.jsonl").read_text().splitlines()
        return [json.loads(line)["file"] for line in lines]

    small, large = files(20), files(50)
    assert len(small) == 20 and small == large[:20]
    assert all(
        Path(f).is_absolute() and Path(f).parent == (real / "images").resolve() for f in large
    )
    with pytest.raises(SystemExit):
        rx.real_subset(61)


def test_summary_reads_eval_and_validation_results(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "ROOT", tmp_path)
    run_dir, eval_dir = tmp_path / "runs" / "exp2-full_obb", tmp_path / "runs" / "eval"
    run_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch, metrics/mAP50(B), metrics/mAP50-95(B)\n1, 0.40, 0.2\n2, 0.65, 0.3\n3, 0.60, 0.3\n"
    )
    (eval_dir / "exp2-full_obb.json").write_text('{"card_f1": 0.8, "exact_hand_accuracy": 0.5}')
    (eval_dir / "exp2-full_obb-known.json").write_text('{"exact_hand_accuracy": 0.7}')

    assert rx.val_map50("exp2-full_obb") == pytest.approx(0.65)
    table = rx.summary_table(rx.exp2_runs())
    assert "| exp2-full_obb | 0.800 | 0.500 | 0.700 | 0.650 |" in table
    assert "| exp2-full_aabb |  |  |  |  |" in table


def test_model_size_changes_checkpoint_and_name_but_not_data():
    run = rx.Run("exp2-corner_obb", "corner_obb")
    small = rx.sized(run, "s")
    assert small.name == "exp2-corner_obb-yolo11s" and small.model == "yolo11s-obb.pt"
    assert small.data_key == run.data_key  # n and s train on the same exported dataset
    assert rx.sized(small, "n") == run
    assert rx.sized(rx.Run("exp2-full_aabb", "full_aabb"), "m").model == "yolo11m.pt"
    assert [r.name for r in rx.select(["exp2-full_obb"], "corner_obb", "s")] == [
        "exp2-full_obb-yolo11s"
    ]


def test_exp1_full_augmentation_shares_the_exp2_dataset():
    keys = {r.name: r.data_key for r in rx.exp1_runs("corner_obb")}
    assert keys["exp2-corner_obb"] == "v1-corner_obb-aug_full"
    assert keys["exp1-corner_obb-real20"] == "v1-corner_obb-aug_full-real20"
    assert len(set(keys.values())) == len(keys)


def test_summary_adds_other_sizes_once_they_have_results(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    table = rx.summary_table(rx.exp2_runs())
    assert "yolo11s" not in table
    (tmp_path / "models" / "exp2-corner_obb-yolo11s.pt").write_bytes(b"")
    rows = [line.split(" | ")[0] for line in rx.summary_table(rx.exp2_runs()).splitlines()]
    assert rows.index("| exp2-corner_obb-yolo11s") == rows.index("| exp2-corner_obb") + 1
