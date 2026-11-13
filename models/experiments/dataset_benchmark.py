"""Bounded public-dataset benchmarks with CPU edge proxies.

Run from repository root. This intentionally benchmarks only archives already
selected in DATASETS.md; it never downloads data or uses test labels for fitting.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "datasets"
OUT = ROOT / "artifacts" / "dataset-benchmarks.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def feature(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if rate != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // g, rate // g).astype(np.float32)
        rate = 16000
    if len(audio) < rate // 10:
        raise ValueError(f"Audio too short: {path}")
    audio -= audio.mean()
    n = min(len(audio), rate * 4)
    x = audio[:n]
    frame = 400
    hop = 160
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(frame), n=512)) ** 2
    hz = np.fft.rfftfreq(512, 1 / rate)
    mel_edges = np.linspace(2595 * np.log10(1 + 80 / 700), 2595 * np.log10(1 + 7600 / 700), 34)
    edges = 700 * (10 ** (mel_edges / 2595) - 1)
    rising = (hz[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    falling = (edges[2:, None] - hz[None, :]) / np.diff(edges)[1:, None]
    filters = np.maximum(0, np.minimum(rising, falling))
    assert np.all(filters.sum(axis=1) > 0)
    logmel = np.log(np.maximum(power @ filters.T, 1e-10))
    cep = dct(logmel, type=2, axis=1, norm="ortho")[:, :13]
    # Retain spectral level; recording gain remains a potential confound.
    values = np.r_[cep.mean(0), cep.std(0), logmel.mean(0), logmel.std(0)]
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite acoustic features: {path}")
    return values.astype(np.float32)


def edge_probe(path: Path) -> dict:
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        value = feature(path)
        timings.append((time.perf_counter() - started) * 1000)
    return {"feature_dimensions": int(value.size), "feature_ms_p50": float(np.median(timings)),
            "feature_ms_p95": float(np.quantile(timings, .95))}


def grouped_cv(rows: list[dict], X: np.ndarray, y: np.ndarray, groups: np.ndarray, method="logistic") -> dict:
    predictions = np.full(y.shape, -1, dtype=int)
    folds = []
    probabilities = np.full(len(y), np.nan)
    majority = np.full(len(y), -1, dtype=int)
    identities = np.array([sha256(row["path"]) for row in rows])
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        # Identical recordings under different names must not become references.
        train = train[~np.isin(identities[train], identities[test])]
        if len(np.unique(y[train])) < 2:
            continue
        estimator = (LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
                     if method == "logistic" else SVC(C=1.0, gamma="scale", class_weight="balanced"))
        model = make_pipeline(StandardScaler(), estimator)
        started = time.perf_counter()
        model.fit(X[train], y[train])
        predictions[test] = model.predict(X[test])
        majority[test] = DummyClassifier(strategy="most_frequent").fit(X[train], y[train]).predict(X[test])
        if set(np.unique(y)) == {0, 1}:
            probabilities[test] = (model.predict_proba(X[test])[:, list(model.classes_).index(1)]
                                   if method == "logistic" else model.decision_function(X[test]))
        folds.append({"group": str(groups[test[0]]), "n_train": int(len(train)), "n_test": int(len(test)),
                      "fit_predict_ms": (time.perf_counter() - started) * 1000})
    mask = predictions >= 0
    if not mask.any():
        return {"status": "not_scored", "reason": "Every training fold lacked two classes"}
    result = {"status": "complete", "n": int(mask.sum()),
              "accuracy": float(accuracy_score(y[mask], predictions[mask])),
              "balanced_accuracy": float(balanced_accuracy_score(y[mask], predictions[mask])),
              "macro_f1": float(f1_score(y[mask], predictions[mask], average="macro", zero_division=0)),
              "folds": folds,
              "majority_accuracy": float(accuracy_score(y[mask], majority[mask])),
              "method": method,
              "predictions": [{"path": str(row["path"].relative_to(ROOT)), "sha256": str(identities[i]),
                               "group": str(groups[i]), "truth": int(y[i]), "prediction": int(predictions[i])}
                              for i, row in enumerate(rows)]}
    if np.isfinite(probabilities).all():
        from sklearn.metrics import roc_auc_score, average_precision_score
        result.update(roc_auc=float(roc_auc_score(y, probabilities)),
                      average_precision=float(average_precision_score(y, probabilities)),
                      positive_scores=probabilities.tolist())
    warm_input = X[:1]
    for _ in range(20):
        model.predict(warm_input)
    timings = []
    for _ in range(100):
        started = time.perf_counter()
        model.predict(warm_input)
        timings.append(1000 * (time.perf_counter() - started))
    import pickle
    result["last_fold_head_cpu"] = {"warmups": 20, "timed_runs": 100,
                                    "p50_ms": float(np.median(timings)), "p95_ms": float(np.quantile(timings, .95)),
                                    "pickle_bytes": len(pickle.dumps(model, protocol=5))}
    return result


def run_acm() -> dict:
    base = DATA / "acm_s2"
    archive = base / "ACM-S2_dataset.zip"
    annotations = json.loads((base / "acm_s2_annotations.json").read_text(encoding="utf-8-sig"))
    if hashlib.md5(archive.read_bytes()).hexdigest() != "473b48b907ce4e23106e3a5a3726b786":
        raise ValueError("ACM-S2 archive checksum mismatch")
    audio_rows = {int(x["audio_id"]): x for x in annotations["audios"]}
    samples = {int(row["id"]): row for row in annotations["annotations"]}
    # ACM-S2 annotations use the same id for audio and semantic sample records.
    rows = []
    for audio_id, meta in audio_rows.items():
        path = base / "acm_s2_audio" / meta["filename"]
        if not path.exists():
            continue
        semantic = samples[audio_id]
        rows.append({"path": path, "label": int(semantic["filling_level"]), "group": str(semantic["containerID"]),
                     "filling_type": int(semantic["filling_type"]), "sample_id": audio_id})
    if len(rows) != 21:
        raise ValueError(f"Expected 21 ACM-S2 files, got {len(rows)}")
    # Source level codes 0/1/2 represent empty/50%/90%.
    X = np.stack([feature(r["path"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    return {"dataset": "ACM-S2", "archive": str(archive.relative_to(ROOT)),
            "archive_sha256": sha256(archive), "task": "filling_level_classification",
            "protocol": "container-held-out LOGO; audio from pouring/shaking, not static taps",
            "metrics": grouped_cv(rows, X, y, groups),
            "rbf_comparator": grouped_cv(rows, X, y, groups, "rbf"), "edge": edge_probe(rows[0]["path"]),
            "counts": {"audio": len(rows), "containers": int(len(np.unique(groups))),
                       "levels": {str(k): int((y == k).sum()) for k in np.unique(y)}}}


def extract_rar(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if not any(output.rglob("*.wav")):
        subprocess.run(["7z", "x", "-y", f"-o{output}", str(archive)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def run_leak() -> dict:
    base = DATA / "leak"
    extracted = base / "extracted"
    archives = [base / "leak-acoustic-data.rar", base / "no-leak-acoustic-data.rar", base / "environmental-noise.rar"]
    expected = ("2ec7cef54b0cbc09adab0047761fe9b2", "d6b9fcd8a730f33fa6c155191b32f2a2", "6080966b540eeeb1b5e61a2cb1cc2639")
    for archive, digest in zip(archives, expected):
        if hashlib.md5(archive.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archive checksum mismatch: {archive}")
    for archive in archives:
        extract_rar(archive, extracted / archive.stem)
    rows = []
    for path in extracted.rglob("*.wav"):
        label = int("leak-acoustic-data" in path.parts)
        # Keep adjacent time segments and duplicate takes of one condition together.
        stem = re.sub(r"[-_]\d+-\d+(?:_\d+)?$", "", path.stem)
        rows.append({"path": path, "label": label, "group": stem})
    if len(rows) != 1000 or sum(r["label"] for r in rows) != 500:
        raise ValueError("Expected all 1000 released clips with 500 leak examples")
    X = np.stack([feature(r["path"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    devices = np.array(["hydrophone" if "hydrophone" in r["path"].name.lower()
                        else "noise_logger" if "noise logger" in r["path"].name.lower()
                        else "other" for r in rows])
    known = devices != "other"
    return {"dataset": "water-distribution-leak-audio", "archives": [str(x.relative_to(ROOT)) for x in archives],
            "archive_sha256": {str(x.relative_to(ROOT)): sha256(x) for x in archives},
            "task": "leak_vs_no_leak_classification", "protocol": "filename-group-held-out LOGO; audible pipe leak proxy, not seal-tap inference",
            "metrics": grouped_cv(rows, X, y, groups),
            "rbf_comparator": grouped_cv(rows, X, y, groups, "rbf"), "edge": edge_probe(rows[0]["path"]),
            "device_transfer_logistic": grouped_cv([r for r, use in zip(rows, known) if use],
                                                   X[known], y[known], devices[known]),
            "device_transfer_scope": "Leave hydrophone/noise-logger device type out; unknown-device noise excluded; filename groups alone do not prove session independence.",
            "noise_policy": "All 114 supplied environmental-noise clips are negative examples; original file-condition groups remain held out.",
            "counts": {"audio": len(rows), "positive": int(y.sum()), "negative": int((y == 0).sum()),
                       "groups": int(len(np.unique(groups)))}}


def main() -> None:
    report = {"status": "complete", "script_sha256": sha256(Path(__file__)),
              "feature": "32 triangular mel filters + 13-DCT mean/std; first four seconds; no pretrained encoder",
              "edge_proxy": "Intel i5-11400H Windows CPU decode/resample/feature and final-fold-head timing; adb devices returned no attached hardware",
              "results": [run_acm(), run_leak()],
              "limitations": ["Small or protocol-specific public subsets; no THUD accuracy claim", "No iPhone/Core ML runtime connected", "ACM-S2 LOGO is internal cross-validation on 21 external-setup samples, not the pretrained ACC external test", "Leak recording/session IDs are not supplied; filename grouping and exact-duplicate exclusion cannot guarantee independent acquisition", "Leak audio may use hydrophones/contact loggers; it is not verified phone-airborne seal integrity sensing", "Out-of-fold binary scores come from separately fitted heads; pooled AUC is descriptive, not a calibrated deployable threshold"]}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(OUT), "datasets": [r["dataset"] for r in report["results"]]}, indent=2))


if __name__ == "__main__":
    main()
