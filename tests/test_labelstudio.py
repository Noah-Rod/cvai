"""The Label Studio converter against an export shaped like the real thing.

The export is built from synthetic hands whose card outlines are known, so the converted
quads can be compared with the truth. One photo is stored sideways with an EXIF orientation
tag, as phones do; Label Studio shows it upright and its percentages refer to that view.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from jass_cv.dataset import export, label_lines, read_annotations
from jass_cv.synth import SynthConfig, generate, load_faces

_spec = importlib.util.spec_from_file_location(
    "labelstudio_to_annotations",
    Path(__file__).resolve().parents[1] / "tools" / "labelstudio_to_annotations.py",
)
ls = importlib.util.module_from_spec(_spec)
sys.modules["labelstudio_to_annotations"] = ls
_spec.loader.exec_module(ls)

DATE_NAME = "20261117-153012.jpg"  # 8 digits + "-", like the upload prefix
SIDEWAYS_NAME = "IMG_0002.jpg"  # stored landscape, EXIF says "rotate 90° clockwise"


def ls_task(task_id: int, name: str, record: dict, cancelled_first: bool = False) -> dict:
    w, h = record["width"], record["height"]
    result = [
        {
            "id": f"r{task_id}_{i}",
            "type": "polygonlabels",
            "from_name": "card",
            "to_name": "image",
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "origin": "manual",
            "value": {
                "points": [[x / w * 100, y / h * 100] for x, y in card["quad"]],
                "closed": True,
                "polygonlabels": [card["label"]],
            },
        }
        for i, card in enumerate(record["cards"])
    ]
    annotations = [{"id": task_id * 10, "result": result, "was_cancelled": False}]
    if cancelled_first:
        annotations.insert(0, {"id": task_id * 10 - 1, "result": [], "was_cancelled": True})
    return {
        "id": task_id,
        "data": {"image": f"/data/upload/3/{task_id:08x}-{name}"},
        "annotations": annotations,
    }


@pytest.fixture(scope="module")
def labeled(tmp_path_factory):
    """Photos folder + Label Studio export + the true records, keyed by photo name."""
    root = tmp_path_factory.mktemp("ls")
    synth = root / "synth"
    gen_args = (load_faces(None, placeholder=True), SynthConfig(size=(320, 240)), [], [])
    generate(synth, n=3, val_fraction=0.0, gen_args=gen_args, seed=5, workers=1)
    truth = list(read_annotations(synth / "train"))
    photos = root / "photos"
    photos.mkdir()

    Image.open(synth / "train" / truth[0]["file"]).save(photos / DATE_NAME, quality=98)
    shown = Image.open(synth / "train" / truth[1]["file"])
    exif = Image.Exif()
    exif[0x0112] = 6
    shown.transpose(Image.Transpose.ROTATE_90).save(photos / SIDEWAYS_NAME, quality=98, exif=exif)
    assert ImageOps.exif_transpose(Image.open(photos / SIDEWAYS_NAME)).size == shown.size
    Image.open(synth / "train" / truth[2]["file"]).save(photos / "IMG_0003.jpg")

    tasks = [
        ls_task(1, DATE_NAME, truth[0]),
        ls_task(2, SIDEWAYS_NAME, truth[1], cancelled_first=True),
        {"id": 3, "data": {"image": "/data/upload/3/00000003-IMG_0003.jpg"}, "annotations": []},
        ls_task(4, "IMG_9999.jpg", truth[2]),  # photo not in the folder
    ]
    export_path = root / "export.json"
    export_path.write_text(json.dumps(tasks))
    return root, export_path, photos, {DATE_NAME: truth[0], SIDEWAYS_NAME: truth[1]}


def test_converted_quads_match_the_true_card_outlines(labeled, capsys):
    root, export_path, photos, truth = labeled
    out = root / "real-train"
    ls.main(["--export", str(export_path), "--images", str(photos), "--out", str(out),
             "--preview", str(root / "preview")])  # fmt: skip
    printed = capsys.readouterr().out
    assert "skip IMG_0003.jpg: not labeled" in printed
    assert "skip IMG_9999.jpg: image missing" in printed

    records = {Path(r["file"]).name: r for r in read_annotations(out)}
    assert set(records) == set(truth)
    for name, record in records.items():
        expected = truth[name]
        assert (record["width"], record["height"]) == (expected["width"], expected["height"])
        assert [c["label"] for c in record["cards"]] == [c["label"] for c in expected["cards"]]
        for got, want in zip(record["cards"], expected["cards"], strict=True):
            np.testing.assert_allclose(got["quad"], want["quad"], atol=0.1)
            np.testing.assert_allclose(got["corner_quad"], want["corner_quad"], atol=0.2)
        with Image.open(out / record["file"]) as image:  # stored upright, no EXIF rotation left
            assert image.size == (record["width"], record["height"])
            assert image.getexif().get(0x0112, 1) == 1
    assert len(list((root / "preview").glob("*.jpg"))) == 2


def test_converted_photos_export_in_every_real_variant(labeled, tmp_path):
    root, export_path, photos, truth = labeled
    out = tmp_path / "real"
    ls.main(["--export", str(export_path), "--images", str(photos), "--out", str(out)])
    for variant in ("full_aabb", "full_obb", "corner_aabb", "corner_obb"):
        data_yaml = export(tmp_path / variant, variant, {"train": [out], "val": [out]})
        assert data_yaml.exists()
    for record in read_annotations(out):
        assert len(label_lines(record, "corner_obb")) == len(record["cards"])


def test_upload_prefix_is_stripped_only_when_the_photo_exists(tmp_path):
    (tmp_path / "IMG_1.jpg").touch()
    (tmp_path / "abcdef12-cards.jpg").touch()
    assert ls.resolve_image("/data/upload/1/0a1b2c3d-IMG_1.jpg", tmp_path).name == "IMG_1.jpg"
    assert ls.resolve_image("/data/local-files/?d=x/abcdef12-cards.jpg", tmp_path).name == (
        "abcdef12-cards.jpg"
    )


def test_json_min_export_is_rejected_with_a_hint(tmp_path):
    with pytest.raises(ls.ConversionError, match="JSON-MIN"):
        ls.convert([{"image": "/data/upload/1/x.jpg", "card": []}], tmp_path)


def test_bad_polygons_are_reported(labeled):
    _, export_path, photos, truth = labeled
    task = json.loads(export_path.read_text())[0]
    polygon = task["annotations"][0]["result"][0]
    twice = {**task, "annotations": [{"result": [polygon, polygon]}]}
    with pytest.raises(ls.ConversionError, match="labeled twice"):
        ls.convert([twice], photos)
    three_points = {
        **polygon,
        "value": {**polygon["value"], "points": polygon["value"]["points"][:3]},
    }
    with pytest.raises(ls.ConversionError, match="four corners"):
        ls.convert([{**task, "annotations": [{"result": [three_points]}]}], photos)
    wrong_size = {**polygon, "original_width": polygon["original_width"] + 10}
    with pytest.raises(ls.ConversionError, match="differs from the photo"):
        ls.convert([{**task, "annotations": [{"result": [wrong_size]}]}], photos)
