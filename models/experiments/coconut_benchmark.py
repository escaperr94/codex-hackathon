"""Coconut workbook benchmark: fruit-grouped evaluation, never ridge-random splitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "datasets" / "coconut"
API = "https://data.mendeley.com/public-api/datasets/hxh8kd3snj/files?folder_id=root&version=1"
CAP = 300_000_000
EXPECTED_SIZE = 278_424_725
EXPECTED_SHA = "629fbc02a8ffac1a40222a2c6a536a1e4a51e812fe9defaaed93417e6c151971"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire():
    DATA.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    (DATA / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert len(metadata) == 1
    item = metadata[0]
    details = item["content_details"]
    assert item["filename"] == "coconut_acoustic signals.xlsx"
    assert details["size"] == EXPECTED_SIZE < CAP
    assert details["sha256_hash"] == EXPECTED_SHA
    path = DATA / item["filename"]
    if not path.exists():
        partial = path.with_suffix(".partial")
        req = urllib.request.Request(details["download_url"], headers={"User-Agent": "Mozilla/5.0"})
        count = 0
        try:
            with urllib.request.urlopen(req, timeout=180) as response, partial.open("wb") as target:
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) > CAP:
                    raise ValueError("Download exceeds cap")
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > CAP:
                        raise ValueError("Download exceeds cap")
                    target.write(block)
            assert count == EXPECTED_SIZE
            assert digest(partial) == EXPECTED_SHA
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    assert path.stat().st_size == EXPECTED_SIZE
    assert digest(path) == EXPECTED_SHA
    return path


def inspect(path):
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    output = {}
    for sheet in workbook:
        rows = []
        for index, row in enumerate(sheet.iter_rows(values_only=True)):
            rows.append(list(row[:12]))
            if index >= 5:
                break
        output[sheet.title] = {"max_row": sheet.max_row, "max_column": sheet.max_column, "first_rows_first_12_columns": rows}
    workbook.close()
    (DATA / "inspection.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def load_signals(path):
    import re
    import numpy as np
    import openpyxl

    cache = DATA / "signals.npy"
    manifest_path = DATA / "signals-manifest.json"
    if cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["workbook_sha256"] == EXPECTED_SHA
        assert digest(cache) == manifest["signals_sha256"]
        return np.load(cache, mmap_mode="r"), manifest
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    assert workbook.sheetnames == ["Ridge A", "Ridge B", "Ridge C"]
    first_headers = None
    manifest = {"workbook_sha256": EXPECTED_SHA, "sheets": [], "records": []}
    signals = np.lib.format.open_memmap(cache, mode="w+", dtype="float32", shape=(3, 132300, 127))
    for ridge, sheet in enumerate(workbook):
        assert (sheet.max_row, sheet.max_column) == (132301, 127)
        rows = sheet.iter_rows(values_only=True)
        headers = list(next(rows))
        assert len(set(headers)) == 127
        if first_headers is None:
            first_headers = headers
        assert [header.removeprefix("c") for header in headers] == [header.removeprefix("c") for header in first_headers], "Ridge fruit identities/order differ"
        for column, header in enumerate(headers):
            match = re.fullmatch(r"c?(\d+)_(im|m|om)", header)
            assert match, header
            manifest["records"].append({"ridge": sheet.title, "column": column + 1,
                "header": header, "fruit_id": int(match[1]), "label": match[2]})
        count = 0
        for count, row in enumerate(rows, 1):
            assert count <= 132300 and len(row) == 127
            assert all(value is None or isinstance(value, (int, float)) for value in row), (sheet.title, count)
            signals[ridge, count - 1] = [np.nan if value is None else value for value in row]
        assert count == 132300
        assert not np.isinf(signals[ridge]).any()
        manifest["sheets"].append({"name": sheet.title, "waveform_row_slots": count,
            "signal_columns": 127, "minimum": float(np.nanmin(signals[ridge])), "maximum": float(np.nanmax(signals[ridge])),
            "missing_cells_per_column": {header: int(np.isnan(signals[ridge, :, column]).sum()) for column, header in enumerate(headers) if np.isnan(signals[ridge, :, column]).any()}})
        signals.flush()
        print(f"Parsed {sheet.title}: {count} rows x 127 columns", flush=True)
    workbook.close()
    ids = {row["fruit_id"] for row in manifest["records"]}
    manifest["missing_fruit_ids_vs_1_to_129"] = sorted(set(range(1, 130)) - ids)
    manifest["headers_missing_c_prefix"] = [header for header in first_headers if not header.startswith("c")]
    manifest["actual_fruit_counts"] = {label: sum(row["label"] == label for row in manifest["records"][:127]) for label in ("im", "m", "om")}
    assert manifest["actual_fruit_counts"] == {"im": 8, "m": 36, "om": 83}
    manifest["signals_sha256"] = digest(cache)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return signals, manifest


def waveform(signals, source_index):
    import numpy as np
    values = signals[source_index // 127, :, source_index % 127]
    valid = np.flatnonzero(np.isfinite(values))
    if not valid.size or valid[0] != 0 or not np.all(np.diff(valid) == 1):
        raise ValueError("Missing start or internal waveform samples")
    if len(valid) < 11025:
        raise ValueError("Less than0.25seconds of contiguous waveform")
    return values[:len(valid)]


def frontend(signal):
    import numpy as np
    from scipy.fft import dct
    from scipy.signal import resample_poly

    x = np.asarray(signal, dtype=np.float64)
    assert x.ndim == 1 and len(x) >= 11025 and np.isfinite(x).all()
    x = x - x.mean()
    x = resample_poly(x, 160, 441)
    rate, frame, hop = 16000, 400, 160
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(frame), n=512)) ** 2
    hz = np.fft.rfftfreq(512, 1 / rate)
    edges = 700 * (10 ** (np.linspace(2595 * np.log10(1 + 80 / 700), 2595 * np.log10(1 + 7600 / 700), 34) / 2595) - 1)
    filters = np.maximum(0, np.minimum((hz[None] - edges[:-2, None]) / np.diff(edges)[:-1, None],
        (edges[2:, None] - hz[None]) / np.diff(edges)[1:, None]))
    assert np.all(filters.sum(1) > 0)
    logmel = np.log(np.maximum(power @ filters.T, 1e-10))
    cep = dct(logmel, type=2, axis=1, norm="ortho")[:, :13]
    spectrum = power.mean(0)
    total = max(spectrum.sum(), 1e-20)
    centroid = float(spectrum @ hz / total)
    rms = np.sqrt(np.mean(x * x))
    peak = np.max(np.abs(x))
    extra = [np.log(max(rms, 1e-10)), np.log(max(peak, 1e-10)), peak / max(rms, 1e-10),
        centroid, np.sqrt(spectrum @ ((hz - centroid) ** 2) / total), hz[spectrum.argmax()],
        np.mean(x[1:] * x[:-1] < 0), np.sum(spectrum[hz < 1000]) / total]
    result = np.r_[cep.mean(0), cep.std(0), logmel.mean(0), logmel.std(0), extra].astype(np.float32)
    assert result.shape == (98,) and np.isfinite(result).all()
    return result


def metrics(truth, prediction):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "confusion_matrix_im_m_om": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist()}


def build_model(candidate):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    name, c = candidate
    if name == "rbf_svm":
        head = SVC(C=c, gamma="scale", class_weight="balanced", decision_function_shape="ovr")
    else:
        head = LogisticRegression(C=c, class_weight=None if name == "unweighted_logistic" else "balanced", max_iter=3000, random_state=42)
    return make_pipeline(StandardScaler(), head)


def fruit_scores(model, X, groups, fruit_ids):
    import numpy as np
    scores = model.decision_function(X)
    assert scores.shape == (len(X), 3)
    pooled = np.stack([scores[groups == fruit].mean(0) for fruit in fruit_ids])
    return model.classes_[pooled.argmax(1)], pooled, scores


def benchmark(signals, manifest):
    import platform
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from sklearn.model_selection import StratifiedKFold

    all_rows = manifest["records"]
    records, exclusions = [], []
    for index, record in enumerate(all_rows):
        row = {**record, "source_index": index}
        try:
            values = waveform(signals, index)
            row.update(samples=len(values), seconds=len(values) / 44100,
                waveform_float32_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest())
            records.append(row)
        except ValueError as error:
            exclusions.append({**row, "reason": str(error)})
    excluded_fruits = {row["fruit_id"] for row in exclusions}
    rows = [row for row in records if row["fruit_id"] not in excluded_fruits]
    assert rows and len(rows) % 3 == 0
    X = np.stack([frontend(waveform(signals, row["source_index"])) for row in rows])
    feature_path = ROOT / "artifacts" / "coconut-features.npz"
    np.savez_compressed(feature_path, X=X, source_sha256=EXPECTED_SHA,
        source_indices=np.array([row["source_index"] for row in rows]))
    assert X.shape == (len(rows), 98) and np.isfinite(X).all()
    labels = ["im", "m", "om"]
    y = np.array([labels.index(row["label"]) for row in rows])
    groups = np.array([row["fruit_id"] for row in rows])
    fruit_ids = np.array([row["fruit_id"] for row in rows if row["ridge"] == "Ridge A"])
    fruit_y = np.array([labels.index(row["label"]) for row in rows if row["ridge"] == "Ridge A"])
    assert np.bincount(fruit_y, minlength=3).min() >= 4
    assert all(np.sum(groups == fruit) == 3 and len(set(y[groups == fruit])) == 1 for fruit in fruit_ids)
    folds = list(StratifiedKFold(4, shuffle=True, random_state=42).split(fruit_ids, fruit_y))
    candidates = [(family, c) for family in ("balanced_logistic", "rbf_svm") for c in (.01, .1, 1., 10.)]
    families = {"balanced_logistic": candidates[:4], "rbf_svm": candidates[4:], "nested_selected": candidates,
        "unweighted_logistic": [("unweighted_logistic", 1.)]}
    outputs = {name: {"fruit_prediction": np.full(len(fruit_ids), -1), "ridge_prediction": np.full(len(rows), -1), "folds": []} for name in [*families, "majority"]}
    for fold, (train_fruits, test_fruits) in enumerate(folds):
        train = np.flatnonzero(np.isin(groups, fruit_ids[train_fruits]))
        test = np.flatnonzero(np.isin(groups, fruit_ids[test_fruits]))
        assert not set(groups[train]) & set(groups[test])
        inner = list(StratifiedKFold(3, shuffle=True, random_state=100 + fold).split(train_fruits, fruit_y[train_fruits]))
        scores = {}
        for candidate in candidates:
            values = []
            for inner_train, inner_valid in inner:
                a, b = train_fruits[inner_train], train_fruits[inner_valid]
                fit_rows = np.flatnonzero(np.isin(groups, fruit_ids[a]))
                val_rows = np.flatnonzero(np.isin(groups, fruit_ids[b]))
                assert not set(groups[fit_rows]) & set(groups[val_rows])
                model = build_model(candidate).fit(X[fit_rows], y[fit_rows])
                predicted, _, _ = fruit_scores(model, X[val_rows], groups[val_rows], fruit_ids[b])
                values.append(metrics(fruit_y[b], predicted)["balanced_accuracy"])
            scores[candidate] = float(np.mean(values))
        contract = {"outer_fold": fold, "train_fruit_ids": fruit_ids[train_fruits].tolist(), "test_fruit_ids": fruit_ids[test_fruits].tolist(),
            "inner_folds": [{"train_fruit_ids": fruit_ids[train_fruits[a]].tolist(), "validation_fruit_ids": fruit_ids[train_fruits[b]].tolist()} for a, b in inner],
            "candidate_inner_fruit_balanced_accuracy": [{"model": candidate[0], "C": candidate[1], "score": value} for candidate, value in scores.items()]}
        for name, space in families.items():
            selected = max(space, key=lambda candidate: scores.get(candidate, -1))
            model = build_model(selected).fit(X[train], y[train])
            predicted, pooled, ridge_scores = fruit_scores(model, X[test], groups[test], fruit_ids[test_fruits])
            result = outputs[name]
            result["fruit_prediction"][test_fruits] = predicted
            result["ridge_prediction"][test] = model.classes_[ridge_scores.argmax(1)]
            result["folds"].append({**contract, "selected_model": selected[0], "C": selected[1], "test_fruit_scores": pooled.tolist(), "score_units": "mean of three ridge decision_function scores; not calibrated probabilities"})
        majority = int(np.bincount(y[train], minlength=3).argmax())
        outputs["majority"]["fruit_prediction"][test_fruits] = majority
        outputs["majority"]["ridge_prediction"][test] = majority
        outputs["majority"]["folds"].append({"outer_fold": fold, "train_fruit_ids": contract["train_fruit_ids"], "test_fruit_ids": contract["test_fruit_ids"], "train_majority_label": labels[majority]})
        print(f"Finished whole-fruit outer fold {fold + 1}/4", flush=True)
    for result in outputs.values():
        fruit_prediction = result.pop("fruit_prediction")
        ridge_prediction = result.pop("ridge_prediction")
        assert np.all(fruit_prediction >= 0) and np.all(ridge_prediction >= 0)
        result["fruit_metrics"] = metrics(fruit_y, fruit_prediction)
        result["ridge_metrics_secondary_not_independent"] = metrics(y, ridge_prediction)
        result["fruit_predictions"] = [{"fruit_id": int(fruit), "truth": labels[int(truth)], "prediction": labels[int(prediction)]} for fruit, truth, prediction in zip(fruit_ids, fruit_y, fruit_prediction)]
        result["ridge_predictions"] = [{**row, "prediction": labels[int(prediction)]} for row, prediction in zip(rows, ridge_prediction)]
    # Deployment fit uses all data only after the out-of-fold predictions above.
    chosen = max(candidates, key=lambda candidate: np.mean([next(value["score"] for value in fold["candidate_inner_fruit_balanced_accuracy"] if (value["model"], value["C"]) == candidate) for fold in outputs["nested_selected"]["folds"]]))
    deploy = build_model(chosen).fit(X, y)
    model_path = ROOT / "artifacts" / "coconut-classifier.joblib"
    joblib.dump(deploy, model_path, compress=3)
    timings, front, heads = [], [], []
    for iteration in range(35):
        start = time.perf_counter()
        feature = frontend(waveform(signals, rows[0]["source_index"]))
        middle = time.perf_counter()
        deploy.predict(feature[None])
        end = time.perf_counter()
        if iteration >= 5:
            timings.append((end - start) * 1000)
            front.append((middle - start) * 1000)
            heads.append((end - middle) * 1000)
    fruit_timings = []
    probe_fruit = rows[0]["fruit_id"]
    probe_rows = [row for row in rows if row["fruit_id"] == probe_fruit]
    for iteration in range(35):
        start = time.perf_counter()
        probe_features = np.stack([frontend(waveform(signals, row["source_index"])) for row in probe_rows])
        fruit_scores(deploy, probe_features, np.full(3, probe_fruit), [probe_fruit])
        elapsed = (time.perf_counter() - start) * 1000
        if iteration >= 5:
            fruit_timings.append(elapsed)
    report = {"status": "complete", "dataset": "Tall Philippine coconut maturity", "source": "https://data.mendeley.com/datasets/hxh8kd3snj/1",
        "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9926183/", "dataset_license": "CC BY 4.0 (repository); article is separately CC BY-NC-ND", "api": API,
        "workbook_bytes": EXPECTED_SIZE, "workbook_sha256": EXPECTED_SHA, "download_cap_bytes": CAP, "script_sha256": digest(Path(__file__)),
        "data_audit": {"actual": {"fruit": 127, "ridge_signals": 381, "sample_slots_per_column": 132300, "classes": manifest["actual_fruit_counts"]},
            "included": {"fruit": len(fruit_ids), "signals": len(rows), "classes": {labels[i]: int((fruit_y == i).sum()) for i in range(3)}, "samples_min": min(row["samples"] for row in rows), "samples_max": max(row["samples"] for row in rows)},
            "excluded_invalid_ridges": exclusions, "excluded_whole_fruit_ids": sorted(excluded_fruits),
            "sheets": manifest["sheets"], "length_policy": "Trailing empty cells trimmed, no sample imputation; internal gaps or<0.25second exclude whole fruit so3ridges retained",
            "repository_claim": {"fruit": 129, "ridge_signals": 387, "classes": {"im": 8, "m": 36, "om": 85}},
            "missing_fruit_ids": manifest["missing_fruit_ids_vs_1_to_129"], "all_three_ridge_headers_identical": False,
            "all_three_ridge_canonical_fruit_ids_and_labels_align": True, "literal_header_difference": "RidgeA/B106to110 lack c prefix; RidgeC includes it",
            "headers_missing_c_prefix_ridge_a_b": manifest["headers_missing_c_prefix"],
            "numeric_content": "raw time-series waveform, not extracted feature table", "sample_rate_hz": 44100, "sample_rate_evidence": "Data in Brief specifications and section3.2;132300 row slots correspond to3seconds but trailing blanks shorten some recordings",
            "amplitude_units": "source numeric amplitude units; not calibrated Pa or dB SPL", "paper_count_typo": "paper says35 mature whereas workbook/repository36; workbook also lacks two claimed overmature fruits"},
        "frontend": {"dimensions": 98, "input_seconds": "all available contiguous samples,0.25to3seconds", "resampling": "44100 to16000 Hz via scipy resample_poly160/441", "features": "13MFCC mean/std +32logmel mean/std + log RMS/log peak/crest/centroid/bandwidth/dominant frequency/zero-crossing/under1kHz energy ratio; whole available waveform; no label-based feature selection", "amplitude_normalization": "DC removal only; recording-level amplitude and duration may confound maturity"},
        "fruit_cpu_latency": {"unit": "milliseconds per fruit including3waveform frontends,3head evaluations,and decision pooling", "warmup": 5, "repetitions": 30, "p50": float(np.median(fruit_timings)), "p95": float(np.quantile(fruit_timings, .95)), "ridge_input_seconds": [row["seconds"] for row in probe_rows], "excludes": "capture wait,workbook decoding,file I/O", "iphone": "not measured"},
        "evaluation": {"unit": "physical fruit; three ridges pooled by mean decision score", "outer": "4fold StratifiedKFold on unique fruit IDs seed42; all three ridges stay together", "inner": "3fold stratified unique-fruit CV seeded100+outer_fold; choose C/model by mean fruit balanced accuracy; scaling fit inside training pipeline", "finite_candidates": candidates, "tie_rule": "first candidate in fixed order", "no_iphone_or_session_holdout": True},
        "results": outputs, "deployment_fit": {"model": chosen[0], "C": chosen[1], "path": str(model_path.relative_to(ROOT)), "bytes": model_path.stat().st_size, "sha256": digest(model_path), "not_used_for_reported_predictions": True},
        "cpu_latency": {"unit": "milliseconds per3second single-ridge input already decoded as numeric waveform", "warmup": 5, "repetitions": 30, "threads_max": 2, "full_frontend_plus_head_p50": float(np.median(timings)), "full_frontend_plus_head_p95": float(np.quantile(timings, .95)), "frontend_p50": float(np.median(front)), "head_p50": float(np.median(heads)), "excludes": "capture wait, workbook decoding, file I/O; pooling3ridges requires3frontends", "platform": platform.platform(), "processor": platform.processor(), "iphone": "not measured"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "artifacts": {"feature_sha256": digest(feature_path), "manifest": str((DATA / "signals-manifest.json").relative_to(ROOT))},
        "limits": ["Only8premature fruits; imbalanced small single-source study; no universal fruit-ripeness claim", "Mechanized rubber tap and condenser microphone in acoustic foam, not phone recording", "Farmer/expert labels, not destructive firmness/chemical reference", "No held-out farm/session/operator metadata, and fruit IDs are class-ordered", "CPU candidate only; no Core ML export/iPhone latency/energy validation", "No global best-method claim: finite feature-classifier comparison only"]}
    out = ROOT / "artifacts" / "coconut-benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "metrics": {name: result["fruit_metrics"] for name, result in outputs.items()}, "cpu_latency": report["cpu_latency"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    arguments = parser.parse_args()
    workbook_path = acquire()
    if arguments.inspect:
        inspect(workbook_path)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            signals, manifest = load_signals(workbook_path)
            benchmark(signals, manifest)
