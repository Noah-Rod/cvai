"""Write hands.csv for the real test photos by ticking cards instead of typing labels.

  uv run --extra frontend streamlit run tools/label_hands.py -- \\
      --images data/real/test/images --hands data/real/test/hands.csv

Shows one photo at a time with a 4 × 9 grid of the cards; every change is saved to hands.csv
straight away (one row per photo, the format tools/evaluate.py reads). "Copy previous photo"
starts from the previous photo's cards, so in a "photo → play a card → photo" sequence you
only untick the played card. Rows for photos that are not in the folder are kept.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jass_cv.cards import CARDS, CLASS_NAMES, RANKS, SUITS, card_by_label  # noqa: E402
from jass_cv.hand import sort_cards  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
RANK_SHORT = {"banner": "10", "under": "U", "ober": "O", "koenig": "K", "ass": "A"}
MAX_HAND = 9  # 36 cards, 4 players


def list_photos(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_hands(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = {row["file"]: row["cards"].split() for row in csv.DictReader(fh)}
    unknown = {c for cards in rows.values() for c in cards} - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"unknown card labels in {path}: {sorted(unknown)}")
    return rows


def write_hands(path: Path, hands: dict[str, list[str]]) -> None:
    """Rewrite the CSV atomically, sorted by file, cards sorted by suit and rank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["file", "cards"])
        for file in sorted(hands):
            if hands[file]:
                writer.writerow([file, " ".join(sort_cards(hands[file]))])
    tmp.replace(path)


def played_since(previous: list[str] | None, current: list[str]) -> tuple[list[str], list[str]]:
    """(cards gone since the previous photo, cards that appeared) — normally one and none."""
    if not previous:
        return [], []
    return sort_cards(set(previous) - set(current)), sort_cards(set(current) - set(previous))


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=ROOT / "data" / "real" / "test" / "images")
    ap.add_argument("--hands", type=Path, default=None, help="default: <images>/../hands.csv")
    args = ap.parse_args(argv)
    args.hands = args.hands or args.images.parent / "hands.csv"
    return args


# ---------------------------------------------------------------------------- app


def app() -> None:
    import streamlit as st
    from PIL import Image, ImageOps

    args = parse_args(sys.argv[1:])
    st.set_page_config(page_title="Hand labels", page_icon="🃏", layout="wide")
    if not args.images.is_dir():
        st.error(f"Photo folder not found: {args.images}")
        st.stop()
    photos = list_photos(args.images)
    if not photos:
        st.error(f"No photos ({', '.join(sorted(IMAGE_EXTS))}) in {args.images}")
        st.stop()
    try:
        hands = read_hands(args.hands)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    state = st.session_state
    state.setdefault("index", next((i for i, p in enumerate(photos) if p not in hands), 0))
    state.index = min(state.index, len(photos) - 1)
    photo = photos[state.index]
    prev = photos[state.index - 1] if state.index > 0 else None

    def key(file: str, label: str) -> str:
        return f"{file}|{label}"

    def go(index: int) -> None:
        state.index = state.jump = max(0, min(index, len(photos) - 1))

    def copy_previous() -> None:
        for label in CLASS_NAMES:
            state[key(photo, label)] = label in hands.get(prev, [])

    def clear() -> None:
        for label in CLASS_NAMES:
            state[key(photo, label)] = False

    for label in CLASS_NAMES:  # initialise this photo's boxes from the CSV once
        state.setdefault(key(photo, label), label in hands.get(photo, []))

    # -- sidebar: progress and navigation
    done = sum(p in hands for p in photos)
    st.sidebar.markdown(f"**{done} / {len(photos)}** photos labelled")
    st.sidebar.progress(done / len(photos))
    st.sidebar.caption(f"Saving to `{args.hands}`")
    first_open = next((i for i, p in enumerate(photos) if p not in hands), None)
    if first_open is not None:
        st.sidebar.button("Next unlabelled photo", on_click=go, args=(first_open,))
    state.setdefault("jump", state.index)
    st.sidebar.selectbox(
        "Jump to",
        range(len(photos)),
        format_func=lambda i: f"{i + 1}. {photos[i]}",  # stable labels keep the widget's state
        key="jump",
        on_change=lambda: go(state.jump),
    )

    # -- main: photo left, cards right
    left, right = st.columns([4, 3], gap="medium")
    with left:
        st.markdown(f"#### {photo}  ·  photo {state.index + 1} of {len(photos)}")
        with Image.open(args.images / photo) as image:
            shown = ImageOps.exif_transpose(image).convert("RGB")
        shown.thumbnail((1400, 1400))
        st.image(shown, width="stretch")

    with right:
        st.markdown("#### Cards in this photo")
        header = st.columns([3] + [1] * len(RANKS))
        for col, rank in zip(header[1:], RANKS, strict=True):
            col.markdown(f"**{RANK_SHORT.get(rank, rank)}**")
        for suit in SUITS:
            cols = st.columns([3] + [1] * len(RANKS))
            cols[0].markdown(suit.capitalize())
            for col, rank in zip(cols[1:], RANKS, strict=True):
                col.checkbox(f"{suit} {rank}", key=key(photo, f"{suit}_{rank}"),
                             label_visibility="collapsed")  # fmt: skip

        selected = [c.label for c in CARDS if state[key(photo, c.label)]]
        if selected != sort_cards(hands.get(photo, [])) and (selected or photo in hands):
            hands[photo] = selected
            if not selected:
                hands.pop(photo)
            write_hands(args.hands, hands)

        names = ", ".join(card_by_label(c).display_name for c in sort_cards(selected))
        st.markdown(f"**{len(selected)} cards**" + (f": {names}" if names else ""))
        if len(selected) > MAX_HAND:
            st.warning(f"A Jass hand has at most {MAX_HAND} cards.")
        gone, new = played_since(hands.get(prev) if prev else None, selected)
        if prev and prev in hands and selected:
            if len(gone) == 1 and not new:
                st.info(f"Played since {prev}: **{card_by_label(gone[0]).display_name}**")
            elif gone or new:
                st.caption(f"vs {prev}: gone {', '.join(gone) or '–'}; new {', '.join(new) or '–'}")

        b1, b2 = st.columns(2)
        b1.button("Copy previous photo", on_click=copy_previous, disabled=prev not in hands,
                  width="stretch")  # fmt: skip
        b2.button("Clear", on_click=clear, width="stretch")
        n1, n2 = st.columns(2)
        n1.button("◀ Back", on_click=go, args=(state.index - 1,), disabled=state.index == 0,
                  width="stretch")  # fmt: skip
        n2.button("Next ▶", on_click=go, args=(state.index + 1,), type="primary",
                  disabled=state.index == len(photos) - 1, width="stretch")  # fmt: skip
        st.caption("Changes are saved immediately.")


if __name__ == "__main__":
    app()
