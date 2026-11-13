import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from thud.transfer import (
    ENCODERS, evaluate_transfer_head, fit_transfer_head, load_transfer_head,
    predict_transfer_head, save_transfer_head, validate_transfer_head,
)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_fill_head_matches_reference_after_json_roundtrip(tmp_path, encoder):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, ENCODERS[encoder]))
    y = np.array([10., 50., 90.] * 4)
    scaler = StandardScaler().fit(X)
    reference = Ridge(alpha=1.0).fit(scaler.transform(X), y)
    head = fit_transfer_head(X, y, "fill", family="jar", encoder=encoder)
    path = tmp_path / "head.json"
    save_transfer_head(head, path)
    prediction = predict_transfer_head(load_transfer_head(path), X, encoder=encoder)
    np.testing.assert_allclose(prediction, reference.predict(scaler.transform(X)), atol=1e-10)
    with pytest.raises(FileExistsError):
        save_transfer_head(head, path)


def test_holdout_predictions_equal_train_only_reference():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 1024))
    X[6:] += 10
    y = np.array([0., 50., 100.] * 4)
    records = [{"path": f"{i}.wav", "label": float(y[i]), "object_id": f"obj-{i // 6}",
                "session_id": f"session-{i // 6}", "recording_id": f"r-{i}", "family": "jar"}
               for i in range(12)]
    report = evaluate_transfer_head(records, X, y, "fill", "object_id")
    for fold in report["folds"]:
        train, test = fold["training_indices"], fold["held_out_indices"]
        scaler = StandardScaler().fit(X[train])
        reference = Ridge(alpha=1.0).fit(scaler.transform(X[train]), y[train])
        observed = [report["predictions"][i]["prediction"] for i in test]
        np.testing.assert_allclose(observed, reference.predict(scaler.transform(X[test])), atol=1e-10)


@pytest.mark.parametrize("field,value", [("scale", 0), ("coefficients", float("nan")), ("mean", True)])
def test_malformed_numeric_head_rejected(field, value):
    head = fit_transfer_head(np.zeros((4, 1024)), ["bonded", "debonded"] * 2, "tile")
    head[field][0] = value
    with pytest.raises(ValueError):
        validate_transfer_head(head)


def test_encoder_mismatch_and_invalid_fill_rejected():
    head = fit_transfer_head(np.zeros((4, 1024)), ["bonded", "debonded"] * 2, "tile")
    with pytest.raises(ValueError, match="match"):
        predict_transfer_head(head, np.zeros((1, 384)), encoder="efficientat-mn04-as")
    with pytest.raises(ValueError, match="1024"):
        fit_transfer_head(np.zeros((3, 521)), ["bonded", "debonded", "bonded"], "tile")
    with pytest.raises(ValueError, match="\[0, 100\]"):
        fit_transfer_head(np.zeros((3, 1024)), [-1., 50., 100.], "fill", family="jar")
