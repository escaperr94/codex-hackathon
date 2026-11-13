"""Contact impact-echo experiments, not airborne microphone audio.

Run: python experiments/impact_echo_benchmark.py
Uses only preserved bounded downloads in artifacts/datasets/impact_echo.
CCNY is descriptive-only: no defensible per-signal defect label registration.
SDNET author-preprocessed original-rate mirror: leave entire bridge deck out.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.fft import dct
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/datasets/impact_echo"
OUT = ROOT / "artifacts"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def spectral_features(signals, rate):
    """Fixed first-20ms transient spectrum; no resampling or audio mel frontend."""
    n = min(signals.shape[1], round(rate * .020))
    x = np.asarray(signals[:, :n], dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(x * np.hanning(n), axis=1)) ** 2
    hz = np.fft.rfftfreq(n, 1 / rate)
    edges = np.linspace(500, 25000, 65)
    bands = np.stack([power[:, (hz >= lo) & (hz < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
    relative = bands / np.maximum(bands.sum(1, keepdims=True), 1e-30)
    logbands = np.log(np.maximum(relative, 1e-12))
    centers = (edges[:-1] + edges[1:]) / 2
    centroid = relative @ centers
    width = np.sqrt((relative * (centers[None, :] - centroid[:, None]) ** 2).sum(1))
    peak = centers[relative.argmax(1)]
    flatness = np.exp(np.log(np.maximum(relative, 1e-12)).mean(1)) / np.maximum(relative.mean(1), 1e-12)
    features = np.column_stack([logbands, dct(logbands, norm="ortho", axis=1)[:, 1:17], centroid, width, peak, flatness])
    assert features.shape == (len(signals), 84) and np.isfinite(features).all()
    return features.astype(np.float32)


def metrics(y, pred):
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "recall_by_class_1_2_3": recall_score(y, pred, labels=[1, 2, 3], average=None, zero_division=0).tolist(),
            "confusion_matrix_labels": [1, 2, 3], "confusion_matrix": confusion_matrix(y, pred, labels=[1, 2, 3]).tolist()}


def ccny_descriptive():
    x = np.load(BASE / "time_amplitude_all.npy", allow_pickle=False)
    t = np.load(BASE / "time_all.npy", allow_pickle=False)
    f = np.load(BASE / "freq.npy", allow_pickle=False)
    a = np.load(BASE / "freq_amplitude.npy", allow_pickle=False)
    assert x.shape == t.shape == (1824, 1024)
    assert a.shape == f.shape == (1824, 124)
    assert np.all(np.diff(t, axis=1) == 2)
    assert np.isfinite(x).all() and np.isfinite(a).all()
    started = time.perf_counter()
    features = spectral_features(x, 500000)
    elapsed = time.perf_counter() - started
    scale = StandardScaler().fit(features[:1178])
    z = scale.transform(features)
    pca = PCA(n_components=2, random_state=42).fit(z[:1178])
    coords = pca.transform(z)
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(z[:1178])
    clusters = km.predict(z)
    rows = []
    for i in range(len(x)):
        part = 1 if i < 1178 else 2
        rows.append({"signal_index": i, "part": part, "role": "fit_descriptive_projection" if part == 1 else "project_only",
                     "defect_label": "unavailable", "cluster_id_not_defect_class": int(clusters[i]),
                     "pc1": float(coords[i, 0]), "pc2": float(coords[i, 1]),
                     "peak_above_3_9_khz": float(f[i, 8 + a[i, 8:].argmax()]),
                     "rms_native_amplitude": float(np.sqrt(np.mean(x[i] ** 2)))})
    write_csv(OUT / "impact-echo-ccny-descriptive.csv", rows)
    np.savez_compressed(OUT / "impact-echo-ccny-features.npz", features=features, projection=coords, cluster_id=clusters)
    joblib.dump({"scale": scale, "pca": pca, "kmeans": km}, OUT / "impact-echo-ccny-descriptive.joblib")
    return {"status": "descriptive_only", "n_signals": len(x), "sample_rate_hz": 500000,
            "signal_samples": 1024, "duration_ms": 2.048, "time_axis_native_unit": "microseconds (2-unit step consistent with documented 500 kHz)",
            "frequency_axis_native_unit": "kHz (repository documentation; rounded 0.49 kHz steps)",
            "amplitude_unit": "not specified in released arrays/loader; not assumed pressure or PCM",
            "classification_metrics": None, "classification_blocker": "No per-signal labels or explicit calibrated index-to-schematic coordinate registration. Author frequency-map script reshapes/flips arrays, but schematic endpoints conflict with nonsquare grid counts and does not define physical interpolation/boundaries. No labels inferred from signal appearance or clusters.",
            "fit_indices": [0, 1178], "project_only_indices": [1178, 1824],
            "feature_extraction_ms_all_signals": elapsed * 1000,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cluster_counts_by_part": {str(part): np.bincount(clusters[:1178] if part == 1 else clusters[1178:], minlength=4).tolist() for part in [1, 2]},
            "peak_khz_percentiles": np.quantile([r["peak_above_3_9_khz"] for r in rows], [0, .25, .5, .75, 1]).tolist()}


def sdnet():
    pieces, labels, groups, identities = [], [], [], []
    source = BASE / "repository/IE_data_sdnet2021_original"
    for path in sorted(source.glob("*/*_origin_*.npy")):
        deck = path.parent.name
        prefix, section = path.stem.split("_origin_")
        values = np.load(path, allow_pickle=False)
        y = np.load(path.parent / f"labels_{prefix}_{section}.npy", allow_pickle=False)
        assert values.shape == (121, 10000) and y.shape == (121,)
        assert np.isfinite(values).all() and set(np.unique(y)) <= {1, 2, 3}
        pieces.append(values); labels.append(y); groups.extend([deck] * len(y))
        identities.extend([{"deck": deck, "section": section, "index_in_section": i, "file": str(path.relative_to(ROOT))} for i in range(len(y))])
    raw = np.concatenate(pieces)
    y, groups = np.concatenate(labels), np.array(groups)
    assert len(y) == 1936 and len(set(groups)) == 5
    started = time.perf_counter()
    X = spectral_features(raw, 102400)
    feature_ms = (time.perf_counter() - started) * 1000
    np.savez_compressed(OUT / "impact-echo-sdnet-features.npz", features=X, labels=y, deck=groups)
    predictions, folds, split_rows = [], [], []
    all_pred = {"spectral_svm": np.zeros_like(y), "small_spectral_mlp": np.zeros_like(y)}
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        held = str(groups[test[0]])
        assert set(groups[train]).isdisjoint(groups[test])
        for i in range(len(y)):
            split_rows.append({"held_out_deck": held, "signal_index": i, "deck": groups[i], "role": "test" if groups[i] == held else "train"})
        search = GridSearchCV(make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
                              {"svc__C": [.1, 1., 10.], "svc__kernel": ["linear", "rbf"]},
                              scoring="balanced_accuracy", cv=LeaveOneGroupOut(), n_jobs=1, refit=True)
        started = time.perf_counter()
        search.fit(X[train], y[train], groups=groups[train])
        svm_seconds = time.perf_counter() - started
        models = {"spectral_svm": search.best_estimator_,
                  "small_spectral_mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", solver="lbfgs", alpha=1., max_iter=300, random_state=42))}
        mlp = models["small_spectral_mlp"]
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mlp.fit(X[train], y[train])
        mlp_seconds = time.perf_counter() - started
        for name, model in models.items():
            pred = model.predict(X[test]); all_pred[name][test] = pred
            timings = []
            for _ in range(20): model.predict(X[test[:1]])
            for _ in range(100):
                started = time.perf_counter(); model.predict(X[test[:1]]); timings.append((time.perf_counter() - started) * 1000)
            target = OUT / f"impact-echo-sdnet-{name}-{held}.joblib"
            joblib.dump(model, target)
            fold = {"model": name, "held_out_deck": held, "n_train": len(train), "metrics": metrics(y[test], pred),
                    "fit_seconds": svm_seconds if name == "spectral_svm" else mlp_seconds,
                    "head_only_cpu_ms_p50": float(np.median(timings)), "head_only_cpu_ms_p95": float(np.quantile(timings, .95)),
                    "model_file": str(target.relative_to(ROOT)), "serialized_bytes": target.stat().st_size,
                    "training_warnings": [] if name == "spectral_svm" else [str(w.message) for w in caught]}
            if name == "spectral_svm":
                fold.update(best_params=search.best_params_, inner_training_deck_balanced_accuracy=float(search.best_score_))
            else:
                fold["parameter_count"] = sum(a.size for a in mlp[-1].coefs_) + sum(a.size for a in mlp[-1].intercepts_)
            folds.append(fold)
            for i, prediction in zip(test, pred):
                predictions.append(dict(identities[i], signal_index=int(i), model=name, true_class=int(y[i]), predicted_class=int(prediction), held_out_deck=held))
    write_csv(OUT / "impact-echo-sdnet-predictions.csv", predictions)
    write_csv(OUT / "impact-echo-sdnet-splits.csv", split_rows)
    return {"status": "complete_author_preprocessed_mirror_not_official_raw_archive", "n_signals": len(y),
            "native_rate_hz": 102400, "preserved_signal_samples": 10000, "preserved_duration_ms": 10000 / 102400 * 1000,
            "analysis_samples": 2048, "analysis_duration_ms": 20., "feature_dimensions": X.shape[1],
            "feature_extraction_ms_all_signals": feature_ms,
            "amplitude_unit": "not declared by mirror; preserved floating-point sensor amplitude, not assumed PCM/Pa",
            "label_meanings": {"1": "sound concrete", "2": "delamination above top rebar mat", "3": "delamination below top rebar mat"},
            "class_counts": {str(v): int((y == v).sum()) for v in [1, 2, 3]},
            "split": "5 outer leave-whole-deck-out folds; all sections from held deck excluded; SVM C/kernel selected only by inner leave-training-deck-out CV; fixed MLP, no test tuning",
            "aggregated": {name: metrics(y, p) for name, p in all_pred.items()}, "folds": folds}


def main():
    with threadpool_limits(limits=1):
        report = {"environment": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
                                  "device": "desktop CPU, one BLAS thread; no physical iPhone", "iphone_latency_memory_energy": "not measured"},
                  "method": "84D relative 0.5–25 kHz transient spectrum (64 log bands, 16 DCT coefficients, centroid/width/peak/flatness); fixed native-rate prefix up to 20ms; no resampling; training-only scaling",
                  "ccny": ccny_descriptive(), "sdnet2021": sdnet()}
    inputs = [p for p in BASE.rglob("*") if p.is_file()]
    save_json(OUT / "impact-echo-input-hashes.json", [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(inputs)])
    save_json(OUT / "impact-echo-results.json", report)
    print(json.dumps({"ccny": report["ccny"]["status"], "sdnet2021": report["sdnet2021"]["aggregated"]}, indent=2))


if __name__ == "__main__":
    main()
