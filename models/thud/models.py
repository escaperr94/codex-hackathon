"""Offline linear baselines and strictly validated, non-executable JSON inference."""

import csv
import hashlib
from pathlib import Path
import warnings

import numpy as np

from .audio import FEATURE_NAMES, FEATURE_VERSION

_CLASSES = ["bonded", "debonded"]
_METADATA = ("path", "object_id", "session_id", "recording_id")


def _task(task):
    if task not in ("tile", "fill"):
        raise ValueError("task must be 'tile' or 'fill'")


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, trimmed string")
    return value


def _matrix(X):
    try:
        X = np.asarray(X)
        if X.dtype.kind not in "iuf":
            raise ValueError("features must be numeric")
        X = X.astype(float, copy=False)
    except (TypeError, OverflowError) as exc:
        raise ValueError("features must be numeric") from exc
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or X.shape[0] == 0:
        raise ValueError(f"features must have shape (n, {len(FEATURE_NAMES)}) with n > 0")
    if not np.isfinite(X).all():
        raise ValueError("features must be finite")
    return X


def _targets(y, task, count):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != count:
        raise ValueError("targets must be one dimensional and match feature rows")
    if task == "tile":
        if any(not isinstance(v, (str, np.str_)) or v not in _CLASSES for v in y):
            raise ValueError("tile labels must be 'bonded' or 'debonded'")
        return y.astype(str)
    if y.dtype.kind not in "iuf":
        raise ValueError("fill labels must be numeric percentage points")
    y = y.astype(float, copy=False)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise ValueError("fill labels must be finite percentage points in [0, 100]")
    return y


def _records(records, y, task):
    if len(records) != len(y):
        raise ValueError("records and targets must have the same number of rows")
    paths = set()
    hashes = set()
    identities = {}
    families = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("each record must be a metadata dictionary")
        for field in _METADATA:
            _text(record.get(field), field)
        path = str(Path(record["path"]).resolve())
        if path in paths:
            raise ValueError("duplicate WAV path")
        paths.add(path)
        digest = record.get("content_sha256")
        if digest is not None:
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("invalid content_sha256")
            if digest in hashes:
                raise ValueError("duplicate recording bytes")
            hashes.add(digest)
        identity = (record["object_id"], record["session_id"])
        previous = identities.setdefault(record["recording_id"], identity)
        if previous != identity:
            raise ValueError("recording_id has inconsistent object/session identity")
        label = record.get("label")
        if task == "tile":
            if label not in _CLASSES or label != y[index]:
                raise ValueError("record label does not match tile target")
        else:
            if (isinstance(label, bool) or not isinstance(label, (int, float, np.number))
                    or not np.isfinite(label) or label != y[index]):
                raise ValueError("record label does not match fill target")
            families.add(_text(record.get("family"), "family"))
    if task == "fill" and len(families) != 1:
        raise ValueError("fill dataset must contain exactly one container family")
    return next(iter(families)) if families else None


def load_dataset(manifest, task):
    """Read a CSV of original-recording identities and analyze each unique WAV.

    Fill labels are volume-labelled percentage points, not inferred liquid height.
    Crops must retain their original recording_id; undeclared crops cannot be detected.
    """
    from .audio import analyze_wav

    _task(task)
    manifest = Path(manifest)
    required = {"path", "label", "object_id", "session_id", "recording_id"}
    if task == "fill":
        required.add("family")
    records, rows, targets = [], [], []
    paths, hashes = set(), set()
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = reader.fieldnames
            if (headers is None or len(headers) != len(set(headers))
                    or set(headers) != required):
                raise ValueError(f"manifest columns must be exactly {', '.join(sorted(required))}")
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None or not value.strip() for value in row.values()):
                    raise ValueError(f"manifest row {line} has missing or extra values")
                row = {key: value.strip() for key, value in row.items()}
                if task == "tile":
                    if row["label"] not in _CLASSES:
                        raise ValueError(f"manifest row {line}: tile label must be bonded or debonded")
                    target = row["label"]
                else:
                    try:
                        target = float(row["label"])
                    except ValueError as exc:
                        raise ValueError(f"manifest row {line}: invalid fill label") from exc
                    if not np.isfinite(target) or not 0 <= target <= 100:
                        raise ValueError(f"manifest row {line}: fill label must be in [0, 100]")
                path = (manifest.parent / row["path"]).resolve()
                if path.suffix.lower() != ".wav" or not path.is_file():
                    raise ValueError(f"manifest row {line}: expected an existing WAV file: {path}")
                if path in paths:
                    raise ValueError("duplicate WAV path")
                paths.add(path)
                digest = hashlib.sha256()
                with path.open("rb") as audio:
                    for chunk in iter(lambda: audio.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest = digest.hexdigest()
                if digest in hashes:
                    raise ValueError("duplicate recording bytes")
                hashes.add(digest)
                result = analyze_wav(path)
                features = result["features"]
                if set(features) != set(FEATURE_NAMES):
                    raise ValueError("audio feature names do not match the current feature contract")
                rows.append([features[name] for name in FEATURE_NAMES])
                targets.append(target)
                row.update(path=str(path), label=target, content_sha256=digest)
                records.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot read manifest/audio: {exc}") from exc
    X = _matrix(rows)
    y = _targets(targets, task, len(X))
    _records(records, y, task)
    return records, X, y


def validate_model(model):
    """Reject incompatible schemas, nonfinite numbers, and ambiguous array shapes."""
    if not isinstance(model, dict):
        raise ValueError("model must be a JSON object")
    _task(model.get("task"))
    fields = {"schema", "version", "task", "feature_version", "feature_names",
              "mean", "scale", "coefficients", "intercept"}
    fields.add("classes" if model["task"] == "tile" else "family")
    if set(model) != fields:
        raise ValueError("model fields do not match the schema")
    if model["schema"] != "thud-linear" or type(model["version"]) is not int or model["version"] != 1:
        raise ValueError("unsupported model schema/version")
    if type(model["feature_version"]) is not int or model["feature_version"] != FEATURE_VERSION:
        raise ValueError("incompatible feature version")
    if not isinstance(model["feature_names"], list) or model["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("incompatible feature names/order")
    for field in ("mean", "scale", "coefficients"):
        values = model[field]
        if (not isinstance(values, list) or len(values) != len(FEATURE_NAMES)
                or any(type(v) not in (int, float) for v in values)):
            raise ValueError(f"{field} must be a numeric vector of feature length")
        try:
            finite = np.isfinite(np.asarray(values, dtype=float)).all()
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be finite") from exc
        if not finite:
            raise ValueError(f"{field} must be finite")
    if any(value <= 0 for value in model["scale"]):
        raise ValueError("scale must be strictly positive")
    value = model["intercept"]
    try:
        finite = type(value) in (int, float) and np.isfinite(float(value))
    except OverflowError as exc:
        raise ValueError("intercept must be a finite scalar") from exc
    if not finite:
        raise ValueError("intercept must be a finite scalar")
    if model["task"] == "tile":
        if not isinstance(model["classes"], list) or model["classes"] != _CLASSES:
            raise ValueError("tile classes must be ordered bonded, debonded")
    else:
        _text(model["family"], "family")


def fit_model(X, y, task, family=None):
    """Fit training-only scaling and L2 logistic/ridge weights; export JSON data."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    _task(task)
    X = _matrix(X)
    y = _targets(y, task, len(X))
    if task == "tile":
        if family is not None:
            raise ValueError("tile model does not accept a container family")
        if set(y) != set(_CLASSES):
            raise ValueError("tile training requires both bonded and debonded classes")
        estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        target = (y == "debonded").astype(int)
    else:
        _text(family, "family")
        if len(np.unique(y)) < 3:
            raise ValueError("fill training requires at least three distinct calibration levels")
        estimator = Ridge(alpha=1.0)
        target = y
    scaler = StandardScaler()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standardized = scaler.fit_transform(X)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(standardized, target)
    except (FloatingPointError, ConvergenceWarning) as exc:
        raise ValueError(f"cannot fit a finite converged linear model: {exc}") from exc
    model = {
        "schema": "thud-linear", "version": 1, "task": task,
        "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficients": np.asarray(estimator.coef_).reshape(-1).tolist(),
        "intercept": float(np.asarray(estimator.intercept_).item()),
    }
    if task == "tile":
        model["classes"] = list(_CLASSES)
    else:
        model["family"] = family
    validate_model(model)
    return model


def predict_model(model, X):
    """Return class labels or unclamped percentage points, using NumPy only.

    Tile decisions are screening labels, not calibrated physical defect probabilities.
    """
    validate_model(model)
    X = _matrix(X)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scores = ((X - model["mean"]) / model["scale"]) @ model["coefficients"] + model["intercept"]
    except FloatingPointError as exc:
        raise ValueError("prediction overflowed; input/model cannot produce finite inference") from exc
    if not np.isfinite(scores).all():
        raise ValueError("prediction must be finite")
    if model["task"] == "tile":
        return np.where(scores > 0, "debonded", "bonded")
    return scores


def _metrics(actual, predicted, task):
    if task == "fill":
        with np.errstate(over="raise", invalid="raise"):
            try:
                mae = float(np.mean(np.abs(np.asarray(actual) - predicted)))
            except FloatingPointError as exc:
                raise ValueError("fill metric overflowed") from exc
        return {"mae_percentage_points": mae, "count": len(actual)}
    matrix = [[sum(a == truth and p == guess for a, p in zip(actual, predicted))
               for guess in _CLASSES] for truth in _CLASSES]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {
        "classes": list(_CLASSES),
        "confusion_matrix": matrix,
        "defect_recall": tp / (tp + fn) if (tp + fn) else None,
        "false_positives": fp,
        "false_positive_rate": fp / (tn + fp) if (tn + fp) else None,
        "accuracy": (tn + tp) / len(actual),
        "coverage": 1.0,
        "count": len(actual),
    }


def evaluate(records, X, y, task, group_by):
    """Leave each whole object/session out, fitting all statistics on training rows."""
    _task(task)
    if group_by not in ("object_id", "session_id"):
        raise ValueError("group_by must be object_id or session_id")
    X = _matrix(X)
    y = _targets(y, task, len(X))
    family = _records(records, y, task)
    groups = np.asarray([record[group_by] for record in records])
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("evaluation requires at least two distinct held-out groups")
    predictions = [None] * len(records)
    folds = []
    for group in unique:
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        try:
            model = fit_model(X[train], y[train], task, family=family)
        except ValueError as exc:
            raise ValueError(f"held-out {group_by}={group}: {exc}") from exc
        predicted = predict_model(model, X[test])
        held_out = []
        for index, prediction in zip(test, predicted):
            prediction = str(prediction) if task == "tile" else float(prediction)
            record = {field: records[index][field] for field in _METADATA}
            if task == "fill":
                record["family"] = family
            record.update(
                row_index=int(index),
                actual=str(y[index]) if task == "tile" else float(y[index]),
                prediction=prediction,
            )
            held_out.append(record)
            predictions[index] = record
        folds.append({
            "held_out_group": str(group),
            "training_groups": sorted(set(groups[train].tolist())),
            "training_indices": train.tolist(),
            "held_out_indices": test.tolist(),
            "held_out_records": held_out,
            "metrics": _metrics(y[test].tolist(), predicted.tolist(), task),
        })
    result = {
        "task": task,
        "group_by": group_by,
        "folds": folds,
        "predictions": predictions,
        "metrics": _metrics(y.tolist(), [r["prediction"] for r in predictions], task),
    }
    if task == "fill":
        result["family"] = family
    return result
