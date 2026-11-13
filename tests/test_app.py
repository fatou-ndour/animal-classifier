"""Tests de l'API Flask, avec un faux modèle (pas besoin de TensorFlow)."""
import io

import numpy as np
import pytest
from PIL import Image

import app as app_module


class FakeModel:
    """Renvoie toujours : 80 % chien, 15 % chat, 5 % oiseau (ordre de class_names)."""

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[0.05, 0.15, 0.80]])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "model", FakeModel())
    monkeypatch.setattr(app_module, "class_names", ["bird", "cat", "dog"])
    return app_module.app.test_client()


def make_image(fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), "green").save(buffer, fmt)
    buffer.seek(0)
    return buffer


def test_home_page(client):
    assert client.get("/").status_code == 200


def test_prediction_ok(client):
    response = client.post("/predict", data={"image": (make_image(), "photo.png")})
    data = response.get_json()
    assert response.status_code == 200
    assert data["best"]["name"] == "Chien"
    assert data["best"]["probability"] == 80.0
    assert data["is_confident"] is True
    assert [r["name"] for r in data["results"]] == ["Chien", "Chat", "Oiseau"]


def test_low_confidence(client, monkeypatch):
    class Unsure(FakeModel):
        def predict(self, batch, verbose=0):
            return np.array([[0.30, 0.35, 0.35]])

    monkeypatch.setattr(app_module, "model", Unsure())
    data = client.post("/predict", data={"image": (make_image(), "a.png")}).get_json()
    assert data["is_confident"] is False


def test_missing_file(client):
    assert client.post("/predict", data={}).status_code == 400


def test_wrong_extension(client):
    response = client.post("/predict", data={"image": (io.BytesIO(b"hello"), "notes.txt")})
    assert response.status_code == 400


def test_corrupted_image(client):
    response = client.post("/predict", data={"image": (io.BytesIO(b"pas une image"), "x.png")})
    assert response.status_code == 400


def test_model_not_loaded(monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    response = app_module.app.test_client().post(
        "/predict", data={"image": (make_image(), "photo.png")}
    )
    assert response.status_code == 503
