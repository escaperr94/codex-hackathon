"""Unit and regression tests for thud.models."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from thud import audio, models


def write_test_wav(path, rate=48000, frequency=1000.0, amplitude=0.4, onset=0.1):
    t = np.arange(round(0.6 * rate)) / rate
    elapsed = np.maximum(t - onset, 0)
    x = amplitude * np.sin(2 * np.pi * frequency * elapsed) * np.exp(-elapsed / 0.04)
    x[t < onset] = 0
    wavfile.write(path, rate, x.astype(np.float32))
    return path


def test_fit_and_predict_parity_tile():
    rng = np.random.default_rng(42)
    n = 20
    X = rng.normal(size=(n, len(audio.FEATURE_NAMES)))
    y = np.array(["bonded"] * 10 + ["debonded"] * 10)
    X[:10, 0] -= 4.0
    X[10:, 0] += 4.0

    model = models.fit_model(X, y, task="tile")
    models.validate_model(model)
    pred = models.predict_model(model, X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (n,)
    assert set(pred).issubset({"bonded", "debonded"})
    assert np.mean(pred == y) >= 0.95


def test_fit_and_predict_parity_fill():
    rng = np.random.default_rng(123)
    n = 24
    X = rng.normal(size=(n, len(audio.FEATURE_NAMES)))
    y = np.array([10.0, 50.0, 90.0] * 8)
    X += (y[:, None] / 100.0) * 3.0

    model = models.fit_model(X, y, task="fill", family="can-330ml")
    models.validate_model(model)
    pred = models.predict_model(model, X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()

    manual_scaled = (X - np.array(model["mean"])) / np.array(model["scale"])
    manual_pred = manual_scaled @ np.array(model["coefficients"]) + model["intercept"]
    np.testing.assert_allclose(pred, manual_pred, rtol=1e-12, atol=1e-12)


def test_fit_guards():
    X = np.ones((4, len(audio.FEATURE_NAMES)))
    with pytest.raises(ValueError, match="both bonded and debonded"):
        models.fit_model(X, ["bonded"] * 4, task="tile")

    with pytest.raises(ValueError, match="family"):
        models.fit_model(X, ["bonded", "debonded", "bonded", "debonded"], task="tile", family="can")

    y_two_levels = np.array([0.0, 100.0, 0.0, 100.0])
    with pytest.raises(ValueError, match="three distinct calibration levels"):
        models.fit_model(X, y_two_levels, task="fill", family="can")


def test_validate_model_rejections():
    valid = {
        "schema": "thud-linear",
        "version": 1,
        "task": "tile",
        "feature_version": audio.FEATURE_VERSION,
        "feature_names": list(audio.FEATURE_NAMES),
        "mean": [0.0] * len(audio.FEATURE_NAMES),
        "scale": [1.0] * len(audio.FEATURE_NAMES),
        "coefficients": [0.1] * len(audio.FEATURE_NAMES),
        "intercept": 0.0,
        "classes": ["bonded", "debonded"],
    }
    models.validate_model(valid)

    bad_schema = dict(valid, schema="invalid")
    with pytest.raises(ValueError, match="schema"):
        models.validate_model(bad_schema)

    bad_scale = dict(valid, scale=[-1.0] * len(audio.FEATURE_NAMES))
    with pytest.raises(ValueError, match="scale"):
        models.validate_model(bad_scale)

    bad_classes = dict(valid, classes=["debonded", "bonded"])
    with pytest.raises(ValueError, match="ordered"):
        models.validate_model(bad_classes)

    nonfinite_coef = dict(valid, coefficients=[np.nan] * len(audio.FEATURE_NAMES))
    with pytest.raises(ValueError, match="finite"):
        models.validate_model(nonfinite_coef)


def test_manifest_duplicate_and_identity_guards(tmp_path):
    wav1 = write_test_wav(tmp_path / "tap1.wav", frequency=900)
    wav2 = write_test_wav(tmp_path / "tap2.wav", frequency=1200)

    # Identical recording bytes guard
    wav_dup_bytes = tmp_path / "tap_copy.wav"
    wav_dup_bytes.write_bytes(wav1.read_bytes())

    manifest_dup_bytes = tmp_path / "dup_bytes.csv"
    manifest_dup_bytes.write_text(
        "path,label,object_id,session_id,recording_id\n"
        f"tap1.wav,bonded,t1,s1,rec1\n"
        f"tap_copy.wav,debonded,t2,s1,rec2\n"
    )
    with pytest.raises(ValueError, match="duplicate recording bytes"):
        models.load_dataset(manifest_dup_bytes, task="tile")

    # Inconsistent recording_id guard
    manifest_id_mismatch = tmp_path / "id_mismatch.csv"
    manifest_id_mismatch.write_text(
        "path,label,object_id,session_id,recording_id\n"
        f"tap1.wav,bonded,t1,s1,rec1\n"
        f"tap2.wav,debonded,t2,s1,rec1\n"
    )
    with pytest.raises(ValueError, match="inconsistent object/session"):
        models.load_dataset(manifest_id_mismatch, task="tile")


def test_evaluate_isolates_held_out_groups(tmp_path):
    # 4 distinct objects, 2 sessions
    rows = ["path,label,object_id,session_id,recording_id"]
    for i in range(4):
        label = "bonded" if i < 2 else "debonded"
        freq = 800 if label == "bonded" else 1500
        path = write_test_wav(tmp_path / f"t_{i}.wav", frequency=freq + i * 5)
        rows.append(f"{path.name},{label},tile_{i},session_{i % 2},rec_{i}")
    manifest = tmp_path / "eval_manifest.csv"
    manifest.write_text("\n".join(rows) + "\n")

    records, X, y = models.load_dataset(manifest, task="tile")
    result = models.evaluate(records, X, y, task="tile", group_by="object_id")
    assert result["task"] == "tile"
    assert len(result["folds"]) == 4
    for fold in result["folds"]:
        assert fold["held_out_group"] not in fold["training_groups"]
        assert len(fold["training_indices"]) == 3
        assert len(fold["held_out_indices"]) == 1
    assert "accuracy" in result["metrics"]
    assert "confusion_matrix" in result["metrics"]
