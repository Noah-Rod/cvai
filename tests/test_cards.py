from pathlib import Path

import yaml

from jass_cv.cards import CARDS, CLASS_NAMES, RANKS, SUITS, card_by_label

ROOT = Path(__file__).resolve().parents[1]


def test_36_unique_cards():
    assert len(CARDS) == len(SUITS) * len(RANKS) == 36
    assert len(set(CLASS_NAMES)) == 36
    assert [c.class_id for c in CARDS] == list(range(36))


def test_display_name():
    assert card_by_label("rosen_ass").display_name == "Rosen Ass"
    assert card_by_label("eicheln_koenig").display_name == "Eicheln König"


def test_dataset_yaml_matches_classes():
    cfg = yaml.safe_load((ROOT / "data/jass.yaml").read_text())
    assert cfg["names"] == dict(enumerate(CLASS_NAMES))
