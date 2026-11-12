from jass_cv.detector import make_detection
from jass_cv.hand import diff_hands, score_hand, select_hand, sort_cards


def test_select_hand_dedupes_and_limits():
    dets = [
        make_detection(0, 0.6, (0, 0, 10, 10)),
        make_detection(0, 0.9, (5, 5, 15, 15)),
        make_detection(1, 0.7, (0, 0, 10, 10)),
        make_detection(2, 0.3, (0, 0, 10, 10)),
    ]
    assert [(d.class_id, d.confidence) for d in select_hand(dets)] == [(0, 0.9), (1, 0.7), (2, 0.3)]
    assert [d.class_id for d in select_hand(dets, max_cards=2)] == [0, 1]


def test_sort_cards_by_suit_then_rank():
    assert sort_cards(["rosen_6", "eicheln_ass", "eicheln_6"]) == [
        "eicheln_6",
        "eicheln_ass",
        "rosen_6",
    ]


def test_diff_hands_finds_played_card():
    before = ["rosen_ass", "eicheln_6", "schellen_under"]
    diff = diff_hands(before, ["eicheln_6", "schellen_under"])
    assert diff.played == ["rosen_ass"] and diff.appeared == []
    assert diff_hands(before, ["eicheln_6", "schellen_ober"]).appeared == ["schellen_ober"]


def test_score_hand():
    s = score_hand({"a_1", "a_2", "x"}, {"a_1", "a_2", "a_3"})
    assert (s.tp, s.fp, s.fn, s.exact) == (2, 1, 1, False)
    assert score_hand({"a"}, {"a"}).exact
