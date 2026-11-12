"""Hand-level logic on top of raw detections: pick the hand, compare hands, score hands."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from jass_cv.cards import RANKS, SUITS, card_by_label
from jass_cv.detector import dedupe_by_card
from jass_cv.schemas import Detection


def select_hand(detections: list[Detection], max_cards: int | None = None) -> list[Detection]:
    """One box per card (a deck has no duplicates); if the hand size is known, keep the top N."""
    hand = dedupe_by_card(detections)
    return hand[:max_cards] if max_cards else hand


def sort_cards(labels: Iterable[str]) -> list[str]:
    """Sort like a player would: by suit, then by rank."""

    def key(label: str) -> tuple[int, int]:
        card = card_by_label(label)
        return SUITS.index(card.suit), RANKS.index(card.rank)

    return sorted(set(labels), key=key)


@dataclass
class HandDiff:
    played: list[str]  # in the previous hand, gone now
    appeared: list[str]  # new now — a recognition error (or a card was picked up)


def diff_hands(before: Iterable[str], after: Iterable[str]) -> HandDiff:
    before, after = set(before), set(after)
    return HandDiff(played=sort_cards(before - after), appeared=sort_cards(after - before))


@dataclass
class HandScore:
    tp: int
    fp: int
    fn: int

    @property
    def exact(self) -> bool:
        return self.fp == 0 and self.fn == 0


def score_hand(predicted: Iterable[str], truth: Iterable[str]) -> HandScore:
    p, t = set(predicted), set(truth)
    return HandScore(tp=len(p & t), fp=len(p - t), fn=len(t - p))
