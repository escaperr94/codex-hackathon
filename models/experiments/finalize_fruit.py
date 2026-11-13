"""Finite nested fruit-head ablations and real-input TFLite exports; run from models/."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hashlib
import json
import platform
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments import coconut_benchmark as coconut
from experiments import durian_benchmark as durian

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/final/fruit"
CS = (.01, .1, 1.)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, prediction):
    return {"n": len(y), "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist()}


def feature_names(task):
    if task == "durian":
        per_impact = ([f"mfcc_mean_{i}" for i in range(20)] + [f"mfcc_std_{i}" for i in range(20)]
            + [f"{stat}_{name}" for stat in ("mean", "std") for name in ("centroid", "bandwidth", "rolloff", "flatness")]
            + ["zcr_mean", "zcr_std"] + [f"band_ratio_{a}_{b}_hz" for a, b in ((50, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7600))])
        return [f"impact_{stat}__{name}" for stat in ("mean", "std") for name in per_impact]
    return ([f"mfcc_{stat}_{i}" for stat in ("mean", "std") for i in range(13)]
        + [f"logmel_{stat}_{i}" for stat in ("mean", "std") for i in range(32)]
        + ["log_rms", "log_peak", "crest", "spectral_centroid_hz", "spectral_bandwidth_hz", "dominant_frequency_hz", "zero_crossing_ratio", "under_1000hz_energy_ratio"])


def load_task(task):
    if task == "durian":
        baseline_path = ROOT / "artifacts/durian-results.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["experiments"]["actual_iphone_primary"]
        all_rows = json.loads((ROOT / "artifacts/durian-recordings.json").read_text(encoding="utf-8"))
        indices = [i for i, row in enumerate(all_rows) if row["device"] == "iphone" and row["label"] is not None]
        rows = [{**all_rows[i], "cache_row": i} for i in indices]
        with np.load(ROOT / "artifacts/durian-features.npz") as cache:
            X = cache["mfcc_spectral"][indices]
        groups = np.array([r["fruit"] for r in rows])
        y = np.array([r["label"] for r in rows])
        folds = [(np.array(f["train_rows"]), np.array(f["test_rows"])) for f in baseline["folds"]]
        blocks = {"all": list(range(112)), "mfcc": list(range(40)) + list(range(56, 96)),
                  "spectral": list(range(40, 56)) + list(range(96, 112))}
        old_metrics = {k: v["fruit"] for k, v in baseline["metrics"].items()}
        labels = ["Unripe", "Ripe", "Overripe"]
    else:
        baseline_path = ROOT / "artifacts/coconut-benchmark.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        rows = baseline["results"]["balanced_logistic"]["ridge_predictions"]
        rows = [{k: v for k, v in row.items() if k != "prediction"} for row in rows]
        labels = ["im", "m", "om"]
        with np.load(ROOT / "artifacts/coconut-features.npz") as cache:
            X = cache["X"]
            assert np.array_equal(cache["source_indices"], [r["source_index"] for r in rows])
        groups = np.array([r["fruit_id"] for r in rows])
        y = np.array([labels.index(r["label"]) for r in rows])
        folds = [(np.flatnonzero(np.isin(groups, f["train_fruit_ids"])), np.flatnonzero(np.isin(groups, f["test_fruit_ids"])))
                 for f in baseline["results"]["balanced_logistic"]["folds"]]
        blocks = {"all": list(range(98)), "mfcc": list(range(26)), "no_amplitude": list(range(90)) + list(range(93, 98))}
        old_metrics = {k: v["fruit_metrics"] for k, v in baseline["results"].items()}
    assert X.shape == (len(rows), len(feature_names(task))) and X.dtype == np.float32 and np.isfinite(X).all()
    assert all(len(set(y[groups == g])) == 1 for g in set(groups))
    coverage = np.zeros(len(rows), int)
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert set(train) | set(test) == set(range(len(rows)))
        coverage[test] += 1
    assert (coverage == 1).all()
    return X, y, groups, rows, folds, blocks, labels, old_metrics, baseline_path


def fit(X, y, groups, candidate, blocks, task):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=candidate["C"], max_iter=3000,
        random_state=durian.SEED if task == "durian" else 42))
    counts = Counter(groups)
    weight = np.array([1 / counts[g] for g in groups]) if task == "durian" else np.ones(len(y))
    if candidate["class_weight"] == "balanced":
        totals = np.bincount(y, weights=weight, minlength=3)
        weight *= weight.sum() / (3 * totals[y])
    # Coconut's original scaler is unweighted; keep its fitting semantics unchanged.
    kwargs = {"logisticregression__sample_weight": weight}
    if task == "durian":
        kwargs["standardscaler__sample_weight"] = weight
    model.fit(X[:, blocks[candidate["block"]]], y, **kwargs)
    assert model.classes_.tolist() == [0, 1, 2]
    return model


def values(model, X, candidate, blocks, task):
    data = X[:, blocks[candidate["block"]]]
    return model.predict_proba(data) if task == "durian" else model.decision_function(data)


def pooled(y, groups, scores):
    ids = np.unique(groups)
    return (np.array([y[np.flatnonzero(groups == g)[0]] for g in ids]),
            np.stack([scores[groups == g].mean(0) for g in ids]), ids)


def select(X, y, groups, rows, space, blocks, task, seed):
    fruits = np.unique(groups)
    fruit_y = np.array([y[np.flatnonzero(groups == g)[0]] for g in fruits])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(fruits, fruit_y))
    contracts = [{"train_fruits": fruits[a].tolist(), "validation_fruits": fruits[b].tolist(),
                  "train_cache_rows": [r.get("cache_row", r.get("source_index")) for r, g in zip(rows, groups) if g in fruits[a]],
                  "validation_cache_rows": [r.get("cache_row", r.get("source_index")) for r, g in zip(rows, groups) if g in fruits[b]]}
                 for a, b in folds]
    results = []
    for candidate in space:
        oof = np.zeros((len(y), 3))
        fold_scores = []
        for a, b in folds:
            train, valid = np.flatnonzero(np.isin(groups, fruits[a])), np.flatnonzero(np.isin(groups, fruits[b]))
            assert not set(groups[train]) & set(groups[valid])
            model = fit(X[train], y[train], groups[train], candidate, blocks, task)
            oof[valid] = values(model, X[valid], candidate, blocks, task)
            truth, predictions, _ = pooled(y[valid], groups[valid], oof[valid])
            fold_scores.append(metrics(truth, predictions.argmax(1))["balanced_accuracy"])
        truth, predictions, _ = pooled(y, groups, oof)
        # Preserve each source benchmark's original selection aggregation.
        score = metrics(truth, predictions.argmax(1))["balanced_accuracy"] if task == "durian" else float(np.mean(fold_scores))
        results.append({**candidate, "inner_fruit_balanced_accuracy": score, "inner_fold_scores": fold_scores})
    return results, contracts


def choose(results, block=None, weight=None):
    eligible = [r for r in results if (block is None or r["block"] == block) and (weight is None or r["class_weight"] == weight)]
    result = max(eligible, key=lambda r: r["inner_fruit_balanced_accuracy"])
    return {k: result[k] for k in ("block", "class_weight", "C")}


def checked_input(matrix, width):
    matrix = np.asarray(matrix)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != width or not np.isfinite(matrix).all():
        raise ValueError("Expected finite float32 [N, full_frontend_dimensions] features")
    return matrix


def exercise(path, X):
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema
    content = path.read_bytes()
    flat = schema.Model.GetRootAsModel(content, 0)
    builtin_codes = [flat.OperatorCodes(i).BuiltinCode() for i in range(flat.OperatorCodesLength())]
    assert schema.BuiltinOperator.CUSTOM not in builtin_codes, "Custom/Flex op in export"
    interpreter = tf.lite.Interpreter(model_content=content, num_threads=2)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details(), interpreter.get_output_details()
    assert len(inp) == len(out) == 1
    assert inp[0]["dtype"] == out[0]["dtype"] == np.float32
    assert inp[0]["shape"].tolist() == [1, X.shape[1]] and out[0]["shape"].tolist() == [1, 3]
    checked_input(X, X.shape[1])
    output = []
    for row in X:
        interpreter.set_tensor(inp[0]["index"], row[None])
        interpreter.invoke()
        output.append(interpreter.get_tensor(out[0]["index"])[0])
    output = np.stack(output)
    assert np.isfinite(output).all()
    return output, {"input_shape": inp[0]["shape"].tolist(), "output_shape": out[0]["shape"].tolist(),
        "dtype": "float32", "builtin_operator_codes": builtin_codes,
        "operators": sorted({op["op_name"] for op in interpreter._get_ops_details()}),
        "no_custom_or_flex_ops": True, "finite_outputs": True, "real_feature_rows_exercised": len(X)}


def export(model, candidate, blocks, task, name, X, row_indices):
    import tensorflow as tf
    selected = blocks[candidate["block"]]
    width = X.shape[1]

    class Head(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, width], tf.float32, name="reference_frontend_features")])
        def __call__(self, features):
            features = tf.gather(features, selected, axis=1)
            normalized = (features - tf.constant(model[0].mean_, tf.float32)) / tf.constant(model[0].scale_, tf.float32)
            logits = tf.matmul(normalized, tf.constant(model[-1].coef_.T, tf.float32)) + tf.constant(model[-1].intercept_, tf.float32)
            return {"class_scores": tf.nn.softmax(logits) if task == "durian" else logits}

    module = Head()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.__call__.get_concrete_function()], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    path = OUT / task / f"{name}.tflite"
    path.write_bytes(converter.convert())
    prediction, contract = exercise(path, X)
    reference = values(model, X, candidate, blocks, task)
    error = float(np.max(np.abs(reference - prediction)))
    changes = int(np.sum(reference.argmax(1) != prediction.argmax(1)))
    assert changes == 0 and error < 1e-4, (name, error, changes)
    np.savez_compressed(path.with_suffix(".npz"), mean=model[0].mean_, scale=model[0].scale_,
                        coef=model[-1].coef_, intercept=model[-1].intercept_, classes=model.classes_, feature_indices=selected)
    record = {"file": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size,
        "parameters_sha256": sha(path.with_suffix(".npz")), "candidate": candidate, **contract,
        "max_absolute_error": error, "argmax_changes": changes, "evaluated_row_indices": list(map(int, row_indices))}
    save(path.with_suffix(".json"), record)
    return record, prediction


def boundary_selfcheck(model, candidate, blocks, task, width):
    # No synthetic task training: a deterministic normalized-origin probe checks class-column semantics.
    assert model.classes_.tolist() == [0, 1, 2]
    probe = np.zeros((1, width), np.float32)
    probe[:, blocks[candidate["block"]]] = model[0].mean_.astype(np.float32)
    checked_input(probe, width)
    for bad in (probe.astype(np.float64), probe[:, :-1], np.full_like(probe, np.nan)):
        try:
            checked_input(bad, width)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid export-boundary input accepted")
    logits = ((probe[:, blocks[candidate["block"]]] - model[0].mean_) / model[0].scale_) @ model[-1].coef_.T + model[-1].intercept_
    expected = np.exp(logits - logits.max(1, keepdims=True)) if task == "durian" else logits
    if task == "durian":
        expected /= expected.sum(1, keepdims=True)
    assert np.allclose(values(model, probe, candidate, blocks, task), expected, atol=1e-10)
    return {"class_column_ids": [0, 1, 2], "normalized_origin_semantics": "passed", "wrong_dtype_shape_nonfinite_rejected": True}


def audio_smoke(task, X, y, rows, model, candidate, blocks, deploy_path):
    selected = [int(np.flatnonzero(y == label)[0]) for label in range(3)]
    if task == "coconut":
        fruit_ids = {rows[i]["fruit_id"] for i in selected}
        selected = [i for i, row in enumerate(rows) if row["fruit_id"] in fruit_ids]
        signals = np.load(coconut.DATA / "signals.npy", mmap_mode="r")
        frontend = lambda i: coconut.frontend(coconut.waveform(signals, rows[i]["source_index"]))
        features = np.stack([frontend(i) for i in selected])
    else:
        features = []
        with zipfile.ZipFile(durian.DATA / "dataset_clean_sound.zip") as archive:
            for i in selected:
                content = archive.read(rows[i]["member"])
                assert hashlib.sha256(content).hexdigest() == rows[i]["source_audio_sha256"]
                # M4A atoms may require seeking; pipe input can silently decode no samples.
                with tempfile.TemporaryDirectory(dir=OUT) as temporary:
                    source = Path(temporary) / ("input" + Path(rows[i]["member"]).suffix)
                    source.write_bytes(content)
                    decoded = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(source), "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"], capture_output=True, check=True, timeout=60)
                    audio = np.frombuffer(decoded.stdout, dtype="<f4").copy()
                assert np.isfinite(audio).all() and len(audio) >= 1600
                windows, centers = durian.impact_windows(audio)
                assert centers == rows[i]["impact_centers_seconds"]
                features.append(durian.acoustic_features(windows))
        features = np.stack(features)
    differences = np.max(np.abs(features - X[selected]), axis=1)
    assert np.allclose(features, X[selected], atol=1e-5, rtol=1e-6), differences
    prediction, contract = exercise(deploy_path, features)
    reference = values(model, features, candidate, blocks, task)
    assert np.array_equal(prediction.argmax(1), reference.argmax(1))
    assert np.max(np.abs(prediction - reference)) < 1e-4
    return {"reference_frontend": f"experiments/{task}_benchmark.py", "rows": [
        {"row_index": i, "source": rows[i], "feature_cache_max_absolute_difference": float(error),
         "tflite_scores": score.tolist(), "reference_scores": ref.tolist()}
        for i, error, score, ref in zip(selected, differences, prediction, reference)],
        "contract": contract, "max_score_error": float(np.max(np.abs(prediction - reference))),
        "source_kind": "decoded compressed audio through ffmpeg" if task == "durian" else "real numeric audio waveforms decoded from source workbook cache; 44100Hz"}


def run_task(task):
    X, y, groups, rows, outer, blocks, labels, old, baseline_path = load_task(task)
    directory = OUT / task
    directory.mkdir(parents=True, exist_ok=True)
    space = [{"block": block, "class_weight": weight, "C": C} for block in blocks for weight in ("balanced", "none") for C in CS]
    policies = {f"{block}__{weight}": (block, weight) for block in blocks for weight in ("balanced", "none")}
    policies["nested_selected"] = (None, None)
    predictions = {name: np.zeros((len(y), 3)) for name in policies}
    majority = np.zeros((len(y), 3))
    fold_ids = np.full(len(y), -1)
    records, exports = [], []
    tflite_oof = np.zeros((len(y), 3))
    baseline_space = [{"block": "all", "class_weight": "balanced", "C": C} for C in (.01, .1, 1., 10.)]
    baseline_oof = np.zeros((len(y), 3))
    baseline_exports, baseline_fold_records = [], []
    if task == "coconut":
        predictions["release_baseline"] = np.zeros((len(y), 3))
    for fold, (train, test) in enumerate(outer):
        seed = durian.SEED + fold if task == "durian" else 100 + fold
        tuning, inner_folds = select(X[train], y[train], groups[train], [rows[i] for i in train], space, blocks, task, seed)
        chosen = {}
        for name, (block, weight) in policies.items():
            candidate = choose(tuning, block, weight)
            model = fit(X[train], y[train], groups[train], candidate, blocks, task)
            predictions[name][test] = values(model, X[test], candidate, blocks, task)
            chosen[name] = candidate
            if name == "nested_selected":
                record, tflite_oof[test] = export(model, candidate, blocks, task, f"oof-fold-{fold}", X[test], test)
                exports.append(record)
        if task == "coconut":
            baseline_tuning, _ = select(X[train], y[train], groups[train], [rows[i] for i in train], baseline_space, blocks, task, seed)
            baseline_candidate = choose(baseline_tuning)
            baseline_model = fit(X[train], y[train], groups[train], baseline_candidate, blocks, task)
            predictions["release_baseline"][test] = values(baseline_model, X[test], baseline_candidate, blocks, task)
            baseline_record, baseline_oof[test] = export(baseline_model, baseline_candidate, blocks, task, f"baseline-oof-fold-{fold}", X[test], test)
            baseline_exports.append(baseline_record)
            baseline_fold_records.append({"fold": fold, "candidate_scores": baseline_tuning, "selected": baseline_candidate})
        truth, _, _ = pooled(y[train], groups[train], np.zeros((len(train), 3)))
        majority[test, np.bincount(truth, minlength=3).argmax()] = 1
        fold_ids[test] = fold
        records.append({"fold": fold, "train_row_indices": train.tolist(), "test_row_indices": test.tolist(),
            "train_fruits": np.unique(groups[train]).tolist(), "test_fruits": np.unique(groups[test]).tolist(),
            "inner_seed": seed, "inner_folds": inner_folds, "candidate_scores": tuning, "selected": chosen})
        print(task, "completed nested/export fold", fold, flush=True)
    assert (fold_ids >= 0).all()
    predictions["majority"] = majority
    comparison = {}
    for name, scores in predictions.items():
        truth, pooled_scores, _ = pooled(y, groups, scores)
        comparison[name] = {"fruit": metrics(truth, pooled_scores.argmax(1)), "recording_secondary": metrics(y, scores.argmax(1))}
    truth, pooled_reference, fruits = pooled(y, groups, predictions["nested_selected"])
    _, pooled_lite, _ = pooled(y, groups, tflite_oof)
    assert np.array_equal(pooled_reference.argmax(1), pooled_lite.argmax(1))
    full_seed = durian.SEED if task == "durian" else 100
    deployment_space = baseline_space if task == "coconut" else space
    full_tuning, full_folds = select(X, y, groups, rows, deployment_space, blocks, task, full_seed)
    deployment_candidate = choose(full_tuning)
    deployment_model = fit(X, y, groups, deployment_candidate, blocks, task)
    selfcheck = boundary_selfcheck(deployment_model, deployment_candidate, blocks, task, X.shape[1])
    deployment, _ = export(deployment_model, deployment_candidate, blocks, task, "deployment-float32", X, range(len(X)))
    smoke = audio_smoke(task, X, y, rows, deployment_model, deployment_candidate, blocks, directory / "deployment-float32.tflite")
    save(directory / "real-audio-smoke.json", smoke)
    baseline_parity = None
    if task == "coconut":
        baseline_truth, baseline_pooled, _ = pooled(y, groups, baseline_oof)
        _, baseline_reference, _ = pooled(y, groups, predictions["release_baseline"])
        assert np.array_equal(baseline_pooled.argmax(1), baseline_reference.argmax(1))
        baseline_parity = {"argmax_changes_recording": int(np.sum(baseline_oof.argmax(1) != predictions["release_baseline"].argmax(1))),
            "argmax_changes_fruit": 0, "max_absolute_error": float(np.max(np.abs(baseline_oof - predictions["release_baseline"]))),
            "fruit_metrics": metrics(baseline_truth, baseline_pooled.argmax(1))}
        assert baseline_parity["argmax_changes_recording"] == 0
    fruit_predictions = []
    for i, fruit in enumerate(fruits):
        mask = groups == fruit
        fruit_predictions.append({"fruit": int(fruit), "target": int(truth[i]), "outer_fold": int(fold_ids[mask][0]),
            "recording_rows": np.flatnonzero(mask).tolist(), "predictions": {name: {"class": int(score[mask].mean(0).argmax()), "scores": score[mask].mean(0).tolist()} for name, score in predictions.items()},
            "tflite_nested_selected_scores": pooled_lite[i].tolist()})
    report = {"task": task, "labels_in_output_column_order": labels, "status": "research deployment head; frontend not included; not physical-phone-tested",
        "finite_candidates": space, "family": "StandardScaler + multinomial L2 logistic regression",
        "release_policy": "conservative original all-feature balanced-logistic C search; ablation selector remains exploratory" if task == "coconut" else "exploratory nested logistic selector",
        "release_baseline": {"fold_selection": baseline_fold_records, "oof_exports": baseline_exports, "tflite_parity": baseline_parity},
        "selection": {"outer": "Original whole-fruit folds reused verbatim; no outer result selects deployment",
            "inner": "3-fold stratified unique-fruit; fit normalization and weights only within each training split",
            "objective": "pooled inner OOF fruit balanced accuracy" if task == "durian" else "mean inner-fold fruit balanced accuracy",
            "tie_rule": "first candidate in fixed block, weight, C order", "exploratory": "Existing outer scores were previously inspected; nested procedure is not a new untouched test set"},
        "full_data_deployment_selection": {"seed": full_seed, "inner_folds": full_folds, "candidate_scores": full_tuning, "selected": deployment_candidate,
            "candidate_space": deployment_space, "training_row_indices": list(range(len(y))), "training_fruits": fruits.tolist(), "not_outer_winner_selection": True},
        "baseline_source": baseline_path.relative_to(ROOT).as_posix(), "baseline_original_fruit_metrics": old,
        "new_nested_metrics": comparison, "folds": records,
        "perexample_predictions": [{"row_index": i, "provenance": row, "target": int(y[i]), "target_name": labels[y[i]], "outer_fold": int(fold_ids[i]),
            "predictions": {name: {"class": int(scores[i].argmax()), "scores": scores[i].tolist()} for name, scores in predictions.items()},
            "tflite_nested_selected_scores": tflite_oof[i].tolist()} for i, row in enumerate(rows)],
        "fruit_predictions": fruit_predictions, "oof_exports": exports, "deployment": deployment,
        "oof_tflite_parity": {"argmax_changes_recording": int(np.sum(tflite_oof.argmax(1) != predictions["nested_selected"].argmax(1))),
            "argmax_changes_fruit": int(np.sum(pooled_lite.argmax(1) != pooled_reference.argmax(1))),
            "max_absolute_error": float(np.max(np.abs(tflite_oof - predictions["nested_selected"]))), "fruit_metrics": metrics(truth, pooled_lite.argmax(1))},
        "selfcheck": selfcheck, "real_audio_smoke": "real-audio-smoke.json",
        "score_units": "per-recording softmax probabilities; mean probabilities per fruit; not calibrated physical ripeness probabilities" if task == "durian" else "multinomial logistic decision logits; mean logits over all 3 ridges then argmax; NOT probabilities",
        "feature_contract": {"input": [1, X.shape[1]], "dtype": "float32", "feature_order": feature_names(task), "ablation_feature_indices": blocks,
            "includes": "feature-index selection + fitted scaler + linear logits + (durian only) softmax",
            "excludes": "waveform decoding, resampling and feature frontend; no raw-audio input",
            "reference_frontend": "experiments/durian_benchmark.py:impact_windows + acoustic_features" if task == "durian" else "experiments/coconut_benchmark.py:waveform + frontend",
            "audio_contract": "ffmpeg mono float32 16000Hz; 100Hz highpass impact detector; <=18 one-second windows (100ms pre,900ms post); whole-recording impact feature mean/std" if task == "durian" else "contiguous 44100Hz source numeric waveform, trailing blank trim; DC remove; resample_poly160/441 to16000Hz; full 0.25-3second waveform",
            "frontend_source_sha256": sha(ROOT / f"experiments/{task}_benchmark.py")},
        "limits": (["133 iPhone recordings/fruits after original ambiguity, contradictory-duplicate, terminal-stage and missing/disease-label exclusions; no reinstatement", "Actual_Ripening_Status source labels; collection date, maturity stage and device may confound target; no new session/orchard validation", "Not a universal fruit-ripeness or food-safety classifier"] if task == "durian" else ["127 fruits,381 ridge recordings; only8 immature fruits; source expert labels not destructive chemistry/firmness", "Mechanized tap/condenser microphone/acoustic foam, NOT phone audio", "No farm/session/operator holdout metadata; amplitude/geometry and acquisition confounds remain"])
            + ["No physical iPhone test, mobile frontend conversion, latency, energy or memory measurement", "Float32 builtin-only export numerically exercised on desktop; no FP16/int8 claim"],
        "license": "Dataset CC BY 4.0; preserve source attribution. Exact model redistribution/commercial product suitability requires review; no added license grant."
            + (" Attribute Mesa-Satina et al. (2026), Zenodo18603796." if task == "durian" else " Mendeley hxh8kd3snj v1; article separately CC BY-NC-ND."),
        "source": "https://doi.org/10.5281/zenodo.18603796" if task == "durian" else "https://data.mendeley.com/datasets/hxh8kd3snj/1"}
    save(directory / "results.json", report)
    return {"task": task, "new_nested_fruit_metrics": comparison["nested_selected"]["fruit"], "original_fruit_metrics": old,
        "release_fruit_metrics": comparison["release_baseline" if task == "coconut" else "nested_selected"]["fruit"],
        "release_baseline_parity": baseline_parity,
        "deployment": deployment, "oof_parity": report["oof_tflite_parity"], "real_audio_records": len(smoke["rows"]), "selfcheck": selfcheck}


def add_fixtures():
    """Create small real-cache release probes without re-running model selection."""
    for task in ("durian", "coconut"):
        X, y, groups, rows, _, _, labels, _, _ = load_task(task)
        directory = OUT / task
        indices = np.concatenate([np.flatnonzero(y == label)[:2] for label in range(3)])
        with np.load(directory / "deployment-float32.npz") as params:
            normalized = (X[indices][:, params["feature_indices"]] - params["mean"]) / params["scale"]
            reference = normalized @ params["coef"].T + params["intercept"]
        if task == "durian":
            reference = np.exp(reference - reference.max(1, keepdims=True))
            reference /= reference.sum(1, keepdims=True)
        fixture = directory / "verification-fixture.npz"
        np.savez_compressed(fixture, features=X[indices], reference_scores=reference, labels=y[indices],
                            groups=groups[indices], row_indices=indices, class_names=np.asarray(labels))
        actual, contract = exercise(directory / "deployment-float32.tflite", X[indices])
        assert np.max(np.abs(actual - reference)) < 1e-4
        assert np.array_equal(actual.argmax(1), reference.argmax(1))
        save(directory / "verification-fixture.json", {"file": fixture.relative_to(ROOT).as_posix(),
            "sha256": sha(fixture), "model": "deployment-float32.tflite",
            "model_sha256": sha(directory / "deployment-float32.tflite"),
            "schema": {"features": f"float32[6,{X.shape[1]}]", "reference_scores": "float64[6,3]",
                "labels": "int64[6] source truth; not in-sample performance evidence", "groups": "int64[6] source fruit IDs",
                "row_indices": "int64[6] results.json perexample row indices", "class_names": "unicode[3] output-column labels"},
            "score_units": "softmax probabilities" if task == "durian" else "decision logits; NOT probabilities",
            "row_provenance": [rows[i] for i in indices], "contract": contract,
            "max_absolute_error": float(np.max(np.abs(actual - reference))), "argmax_changes": 0})


def main():
    import scipy
    import sklearn
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    preserved = [ROOT / "artifacts" / name for name in ("durian-results.json", "durian-recordings.json", "durian-features.npz", "coconut-benchmark.json", "coconut-features.npz")]
    preserved += [ROOT / "docs/durian-report.md"]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in preserved}
    summaries = [run_task(task) for task in ("durian", "coconut")]
    add_fixtures()
    assert before == {p.relative_to(ROOT).as_posix(): sha(p) for p in preserved}, "Existing evidence changed"
    save(OUT / "summary.json", {"tasks": summaries, "runtime": {"platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "tensorflow": tf.__version__, "threads_max": 2,
        "physical_phone_tested": False}, "preserved_input_sha256": before, "script_sha256": sha(Path(__file__))})
    manifest = {p.relative_to(OUT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"}
    save(OUT / "manifest.json", manifest)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
