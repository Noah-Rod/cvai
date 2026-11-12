import re
from pathlib import Path

from jass_cv.cards import CARDS, CLASS_NAMES, RANKS, SUITS, card_by_label

ROOT = Path(__file__).resolve().parents[1]


def test_36_unique_cards():
    assert len(CARDS) == len(SUITS) * len(RANKS) == 36
    assert len(set(CLASS_NAMES)) == 36
    assert [c.class_id for c in CARDS] == list(range(36))


def test_display_name():
    assert card_by_label("rosen_ass").display_name == "Rosen Ass"
    assert card_by_label("eicheln_koenig").display_name == "Eicheln König"


def test_label_studio_config_matches_classes():
    xml = (ROOT / "labeling/label_studio_config.xml").read_text()
    assert re.findall(r'Label value="([a-z0-9_]+)"', xml) == list(CLASS_NAMES)
