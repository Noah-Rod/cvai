"""The 36 Swiss German Jass cards and their class ids.

The class order here is the single source of truth: `data/jass.yaml` and every trained
model must use the same order (enforced by tests/test_cards.py).
"""

from __future__ import annotations

from dataclasses import dataclass

SUITS: tuple[str, ...] = ("eicheln", "schilten", "rosen", "schellen")
RANKS: tuple[str, ...] = ("6", "7", "8", "9", "banner", "under", "ober", "koenig", "ass")

SUIT_DISPLAY = {s: s.capitalize() for s in SUITS}
RANK_DISPLAY = {
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "banner": "Banner",
    "under": "Under",
    "ober": "Ober",
    "koenig": "König",
    "ass": "Ass",
}


@dataclass(frozen=True)
class Card:
    class_id: int
    suit: str
    rank: str

    @property
    def label(self) -> str:
        """Machine label, e.g. `rosen_ass`."""
        return f"{self.suit}_{self.rank}"

    @property
    def display_name(self) -> str:
        """Human label, e.g. `Rosen Ass`."""
        return f"{SUIT_DISPLAY[self.suit]} {RANK_DISPLAY[self.rank]}"


CARDS: tuple[Card, ...] = tuple(
    Card(class_id=i, suit=suit, rank=rank)
    for i, (suit, rank) in enumerate((s, r) for s in SUITS for r in RANKS)
)
CLASS_NAMES: tuple[str, ...] = tuple(c.label for c in CARDS)
_BY_LABEL = {c.label: c for c in CARDS}


def card_by_id(class_id: int) -> Card:
    return CARDS[class_id]


def card_by_label(label: str) -> Card:
    return _BY_LABEL[label]
