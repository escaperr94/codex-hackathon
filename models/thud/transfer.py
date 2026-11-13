"""Portable linear task heads over frozen pretrained audio embeddings."""

import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .models import _metrics, _records, _targets, _task

ENCODERS = {"yamnet": 1024, "efficientat-mn04-as": 384}
SCHEMA = "thud-pretrained-head"
VERSION = 1


def _matrix(values, encoder):
    if encoder not in ENCODERS:
        raise ValueError("encoder must be yamnet or efficientat-mn04-as")
    values = np.asarray(values)
    dimension = ENCODERS[encoder]
    if values.dtype.kind not in "iuf" or values.ndim != 2 or values.shape[1] != dimension or not len(values):
        raise ValueError(f"embeddings must be numeric shape (n, {dimension}), n > 0")
    values = values.astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    return values


def fit_transfer_head(embeddings, labels, task, family=None, *, encoder="yamnet"):
    _task(task)
    X = _matrix(embeddings, encoder)
    y = _targets(labels, task, len(X))
    if task == "tile":
        if family is not None or set(y.tolist()) != {"bonded", "debonded"}:
            raise ValueError("tile transfer requires both bonded and debonded labels and no family")
        estimator = LogisticRegression(C=1.0, max_iter=1000)
        target = (y == "debonded").astype(int)
    else:
        if not isinstance(family, str) or not family.strip() or family != family.strip() or len(np.unique(y)) < 3:
            raise ValueError("fill transfer requires one family and three levels")
        estimator = Ridge(alpha=1.0)
        target = y
    scaler = StandardScaler()
    try:
        with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise", divide="raise"):
            warnings.simplefilter("error", ConvergenceWarning)
            transformed = scaler.fit_transform(X)
            estimator.fit(transformed, target)
    except (FloatingPointError, ConvergenceWarning) as exc:
        raise ValueError(f"cannot fit finite converged pretrained head: {exc}") from exc
    head = {
        "schema": SCHEMA, "version": VERSION, "task": task,
        "encoder": encoder, "pooling": "frame_mean" if encoder == "yamnet" else "global_average",
        "embedding_dim": ENCODERS[encoder],
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficients": np.asarray(estimator.coef_).reshape(-1).tolist(),
        "intercept": float(np.asarray(estimator.intercept_).item()),
        "classes": ["bonded", "debonded"] if task == "tile" else None,
        "family": family,
    }
    validate_transfer_head(head)
    return head


def validate_transfer_head(head):
    required = {"schema", "version", "task", "encoder", "pooling", "embedding_dim", "mean", "scale",
                "coefficients", "intercept", "classes", "family"}
    if not isinstance(head, dict) or set(head) != required:
        raise ValueError("invalid pretrained head fields")
    if head["schema"] != SCHEMA or type(head["version"]) is not int or head["version"] != VERSION:
        raise ValueError("invalid pretrained head schema; retrain legacy heads made with the recreated encoder")
    _task(head["task"])
    encoder = head["encoder"]
    if not isinstance(encoder, str) or encoder not in ENCODERS:
        raise ValueError("invalid pretrained encoder")
    dimension = ENCODERS[encoder]
    pooling = "frame_mean" if encoder == "yamnet" else "global_average"
    if head["pooling"] != pooling or type(head["embedding_dim"]) is not int or head["embedding_dim"] != dimension:
        raise ValueError("invalid pretrained encoder contract")
    for field in ("mean", "scale", "coefficients"):
        values = head[field]
        if not isinstance(values, list) or len(values) != dimension or any(type(value) not in (int, float) for value in values):
            raise ValueError(f"head {field} must be a numeric vector of length {dimension}")
        try:
            finite = np.isfinite(np.asarray(values, dtype=float)).all()
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"head {field} must be finite") from exc
        if not finite:
            raise ValueError(f"head {field} must be finite")
    if any(value <= 0 for value in head["scale"]):
        raise ValueError("head scale must be positive")
    try:
        finite = type(head["intercept"]) in (int, float) and np.isfinite(float(head["intercept"]))
    except OverflowError as exc:
        raise ValueError("head intercept must be finite") from exc
    if not finite:
        raise ValueError("head intercept must be finite")
    if head["task"] == "tile":
        if head["classes"] != ["bonded", "debonded"] or head["family"] is not None:
            raise ValueError("invalid tile head labels")
    elif head["classes"] is not None or not isinstance(head["family"], str) or not head["family"].strip() or head["family"] != head["family"].strip():
        raise ValueError("invalid fill head family")


def predict_transfer_head(head, embeddings, *, encoder=None):
    validate_transfer_head(head)
    if encoder is not None and encoder != head["encoder"]:
        raise ValueError("head encoder does not match the supplied embeddings")
    X = _matrix(embeddings, head["encoder"])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            score = ((X - head["mean"]) / head["scale"]) @ head["coefficients"] + head["intercept"]
    except FloatingPointError as exc:
        raise ValueError("pretrained head prediction overflowed") from exc
    if not np.isfinite(score).all():
        raise ValueError("pretrained head prediction must be finite")
    return np.where(score > 0, "debonded", "bonded") if head["task"] == "tile" else score


def evaluate_transfer_head(records, embeddings, labels, task, group_by, *, encoder="yamnet"):
    """Reuse frozen embeddings, but fit task/scaling statistics on training groups only."""
    _task(task)
    X = _matrix(embeddings, encoder)
    y = _targets(labels, task, len(X))
    family = _records(records, y, task)
    if group_by not in ("object_id", "session_id"):
        raise ValueError("group_by must be object_id or session_id")
    groups = np.asarray([record[group_by] for record in records])
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("evaluation requires at least two distinct held-out groups")
    predictions = [None] * len(records)
    folds = []
    for group in unique:
        train = np.flatnonzero(groups != group)
        test = np.flatnonzero(groups == group)
        try:
            head = fit_transfer_head(X[train], y[train], task, family, encoder=encoder)
        except ValueError as exc:
            raise ValueError(f"held-out {group_by}={group}: {exc}") from exc
        predicted = predict_transfer_head(head, X[test], encoder=encoder).tolist()
        for index, value in zip(test, predicted):
            predictions[index] = value
        folds.append({"held_out_group": str(group), "training_groups": sorted(set(groups[train])),
                      "training_indices": train.tolist(), "held_out_indices": test.tolist(),
                      "metrics": _metrics(y[test].tolist(), predicted, task)})
    return {"task": task, "encoder": encoder, "group_by": group_by, "folds": folds,
            "predictions": [{"path": row["path"], "object_id": row["object_id"],
                             "session_id": row["session_id"], "actual": y[i].item(),
                             "prediction": predictions[i]} for i, row in enumerate(records)],
            "metrics": _metrics(y.tolist(), predictions, task)}


def save_transfer_head(head, path):
    validate_transfer_head(head)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(head, handle, allow_nan=False, indent=2)
        handle.write("\n")


def load_transfer_head(path):
    with Path(path).open(encoding="utf-8") as handle:
        head = json.load(handle)
    validate_transfer_head(head)
    return head
