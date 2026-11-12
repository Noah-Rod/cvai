import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from jass_cv.config import get_settings
from jass_cv.detector import dedupe_by_card, make_detection


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JASS_MODEL_BACKEND", "dummy")
    get_settings.cache_clear()
    from jass_cv.api.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _jpeg(w=320, h=240) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "green").save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model"] == "dummy"


def test_classes(client):
    assert len(client.get("/classes").json()) == 36


def test_predict(client):
    r = client.post("/predict", files={"file": ("x.jpg", _jpeg())}, params={"conf": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert (body["image_width"], body["image_height"]) == (320, 240)
    assert body["cards"] == ["rosen_ass"]
    assert body["detections"][0]["bbox"]["x2"] <= 320


def test_predict_annotated(client):
    r = client.post("/predict/annotated", files={"file": ("x.jpg", _jpeg())})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    Image.open(io.BytesIO(r.content)).verify()


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image")})
    assert r.status_code == 400


def test_dedupe_keeps_best_box_per_card():
    a = make_detection(0, 0.6, (0, 0, 10, 10))
    b = make_detection(0, 0.9, (5, 5, 15, 15))
    c = make_detection(1, 0.7, (0, 0, 10, 10))
    out = dedupe_by_card([a, b, c])
    assert [(d.class_id, d.confidence) for d in out] == [(0, 0.9), (1, 0.7)]
