import json

import numpy as np
import pytest

from jass_cv.dataset import VARIANTS, export, label_lines, read_annotations
from jass_cv.geometry import corner_quad_from_card_quad, corner_rect
from jass_cv.synth import SynthConfig, generate, load_faces


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    gen_args = (load_faces(None, placeholder=True), SynthConfig(size=(320, 240)), [], [])
    generate(out, n=6, val_fraction=0.34, gen_args=gen_args, seed=1)
    return out


def test_output_independent_of_workers(tmp_path):
    gen_args = (load_faces(None, placeholder=True), SynthConfig(size=(160, 120)), [], [])
    generate(tmp_path / "a", n=4, val_fraction=0.5, gen_args=gen_args, seed=3, workers=1)
    generate(tmp_path / "b", n=4, val_fraction=0.5, gen_args=gen_args, seed=3, workers=2)
    for split in ("train", "val"):
        a = (tmp_path / "a" / split / "annotations.jsonl").read_text()
        assert a == (tmp_path / "b" / split / "annotations.jsonl").read_text()


def test_generator_writes_consistent_annotations(synth_dir):
    records = list(read_annotations(synth_dir / "train")) + list(
        read_annotations(synth_dir / "val")
    )
    assert len(records) == 6
    for r in records:
        assert (synth_dir / "train" / r["file"]).exists() or (
            synth_dir / "val" / r["file"]
        ).exists()
        assert 3 <= len(r["cards"]) <= 12
        assert len({c["class_id"] for c in r["cards"]}) == len(r["cards"])  # no duplicate cards
        for c in r["cards"]:
            assert 0 <= c["visible"] <= 1 and 0 <= c["corner_visible"] <= 1
            quad = np.array(c["quad"])
            assert quad.shape == (4, 2)
            assert (quad >= -1).all() and (quad[:, 0] <= r["width"] + 1).all()
        # the top card of the fan is drawn last, so most of it is visible
        assert max(r["cards"], key=lambda c: c["layer"])["visible"] > 0.5


def test_corner_quad_derivable_from_card_quad(synth_dir):
    card = next(read_annotations(synth_dir / "train"))["cards"][0]
    np.testing.assert_allclose(
        corner_quad_from_card_quad(card["quad"]), card["corner_quad"], atol=0.5
    )
    assert corner_rect().shape == (4, 2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_export_variants(synth_dir, tmp_path, variant):
    yaml_path = export(
        tmp_path / variant, variant, {"train": [synth_dir / "train"], "val": [synth_dir / "val"]}
    )
    assert "names:" in yaml_path.read_text()
    labels = list((tmp_path / variant / "labels" / "train").glob("*.txt"))
    assert len(labels) == 4
    n_values = 9 if variant.endswith("obb") else 5
    for f in labels:
        for line in f.read_text().splitlines():
            values = line.split()
            assert len(values) == n_values
            assert 0 <= int(values[0]) < 36
            assert all(0 <= float(v) <= 1 for v in values[1:])
    stats = json.loads((tmp_path / variant / "export.json").read_text())["stats"]
    assert stats["train"]["boxes"] > 0


def test_label_lines_filters_hidden_corners():
    record = {
        "width": 100,
        "height": 100,
        "cards": [
            {
                "class_id": 0,
                "quad": [[10, 10], [40, 10], [40, 60], [10, 60]],
                "corner_visible": 0.1,
            },
            {
                "class_id": 1,
                "quad": [[50, 10], [80, 10], [80, 60], [50, 60]],
            },  # real label: no ratios
        ],
    }
    assert [ln.split()[0] for ln in label_lines(record, "corner_aabb")] == ["1"]
    assert len(label_lines(record, "full_obb")) == 2
