import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "label_hands", Path(__file__).resolve().parents[1] / "tools" / "label_hands.py"
)
lh = importlib.util.module_from_spec(_spec)
sys.modules["label_hands"] = lh
_spec.loader.exec_module(lh)  # streamlit is only imported inside app()

ROOT = Path(__file__).resolve().parents[1]


def test_hands_csv_round_trip_is_what_evaluate_reads(tmp_path):
    path = tmp_path / "hands.csv"
    hands = {"b.jpg": ["schellen_ass", "eicheln_6"], "a.jpg": ["rosen_under"], "c.jpg": []}
    lh.write_hands(path, hands)
    assert path.read_text() == "file,cards\na.jpg,rosen_under\nb.jpg,eicheln_6 schellen_ass\n"
    assert lh.read_hands(path) == {"a.jpg": ["rosen_under"], "b.jpg": ["eicheln_6", "schellen_ass"]}

    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "tools" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    assert evaluate.read_hands(path) == {
        "a.jpg": {"rosen_under"},
        "b.jpg": {"eicheln_6", "schellen_ass"},
    }


def test_unknown_labels_are_rejected(tmp_path):
    path = tmp_path / "hands.csv"
    path.write_text("file,cards\na.jpg,rosen_ace\n")
    with pytest.raises(ValueError, match="rosen_ace"):
        lh.read_hands(path)


def test_played_card_is_the_one_that_disappeared():
    before = ["rosen_6", "rosen_ass", "eicheln_under"]
    assert lh.played_since(before, ["rosen_ass", "eicheln_under"]) == (["rosen_6"], [])
    assert lh.played_since(before, ["rosen_ass", "eicheln_under", "schellen_9"]) == (
        ["rosen_6"],
        ["schellen_9"],
    )
    assert lh.played_since(None, before) == ([], [])


def test_photos_are_listed_by_name_and_hands_csv_defaults_next_to_them(tmp_path):
    images = tmp_path / "test" / "images"
    images.mkdir(parents=True)
    for name in ("IMG_2.JPG", "IMG_1.jpg", "notes.txt", "IMG_3.png"):
        (images / name).touch()
    assert lh.list_photos(images) == ["IMG_1.jpg", "IMG_2.JPG", "IMG_3.png"]
    assert lh.parse_args(["--images", str(images)]).hands == tmp_path / "test" / "hands.csv"
