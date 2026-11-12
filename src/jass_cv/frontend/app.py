"""Streamlit frontend. Talks to the API only (no model in this container).

Run locally:  streamlit run src/jass_cv/frontend/app.py
"""

from __future__ import annotations

import io
import json
import os

import requests
import streamlit as st
from PIL import Image, ImageOps

from jass_cv.cards import card_by_label
from jass_cv.drawing import draw_detections
from jass_cv.hand import diff_hands
from jass_cv.schemas import PredictionResponse

API_URL = os.getenv("JASS_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Jass Card Detector", page_icon="🃏", layout="wide")
st.title("🃏 Jass Card Detector")


@st.cache_data(ttl=10, show_spinner=False)
def api_health(url: str) -> dict | None:
    try:
        return requests.get(f"{url}/health", timeout=3).json()
    except requests.RequestException:
        return None


def call_predict(image_bytes: bytes, filename: str, conf: float, iou: float, max_cards: int):
    params = {"conf": conf, "iou": iou, "dedupe": True}
    if max_cards:
        params["max_cards"] = max_cards
    r = requests.post(
        f"{API_URL}/predict", files={"file": (filename, image_bytes)}, params=params, timeout=60
    )
    r.raise_for_status()
    return PredictionResponse.model_validate(r.json())


with st.sidebar:
    st.header("Settings")
    st.caption(f"API: `{API_URL}`")
    health = api_health(API_URL)
    if health is None:
        st.error("API not reachable")
    else:
        st.success(f"Model: `{health['model']}`")
        if health["model"] == "dummy":
            st.warning("Dummy model active — results are fake. Put weights in `models/best.pt`.")
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.05)
    max_cards = st.number_input(
        "Cards in hand (0 = unknown)",
        min_value=0,
        max_value=12,
        value=0,
        help="If you know how many cards you hold, only the N most confident cards are kept.",
    )

upload_tab, camera_tab = st.tabs(["📁 Upload image", "📷 Camera snapshot"])
with upload_tab:
    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "bmp"])
with camera_tab:
    snapshot = st.camera_input("Take a picture")

source = uploaded or snapshot
if source is None:
    st.info("Upload an image or take a snapshot to detect cards.")
    st.stop()


def names(labels: list[str]) -> str:
    return ", ".join(card_by_label(label).display_name for label in labels) or "–"


raw = source.getvalue()
image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")

try:
    with st.spinner("Detecting…"):
        result = call_predict(raw, source.name or "image.jpg", conf, iou, int(max_cards))
except requests.RequestException as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

left, right = st.columns([3, 2])
with left:
    st.image(draw_detections(image, result.detections), width="stretch")
with right:
    c1, c2 = st.columns(2)
    c1.metric("Cards found", len(result.cards))
    c2.metric("Inference", f"{result.inference_ms:.0f} ms")
    st.markdown(f"**Hand:** {names(result.cards)}")

    # Game tracking: photo before and after playing → which card was played?
    previous = st.session_state.get("previous_hand")
    if previous is not None:
        diff = diff_hands(previous, result.cards)
        st.markdown(f"**Played since last photo:** {names(diff.played)}")
        if diff.appeared:
            st.warning(f"New since last photo (recognition error?): {names(diff.appeared)}")
    b1, b2 = st.columns(2)
    if b1.button("Remember this hand", help="Take the next photo after playing a card"):
        st.session_state["previous_hand"] = result.cards
        st.rerun()
    if previous is not None and b2.button("Forget"):
        del st.session_state["previous_hand"]
        st.rerun()

    if result.detections:
        st.dataframe(
            [{"card": d.display_name, "confidence": d.confidence} for d in result.detections],
            hide_index=True,
            width="stretch",
        )
    st.download_button(
        "Download JSON",
        json.dumps(result.model_dump(), indent=2),
        file_name="detections.json",
        mime="application/json",
    )
