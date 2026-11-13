"""Finite exploratory valve ablations; run: python -m experiments.dcase_ablation.

Previously inspected test data are not independent confirmation. No test labels
enter feature fitting, domain adaptation, or scorer construction.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FEATURES = ART / "dcase-ablation-features.npz"
REPORT = ART / "dcase-ablation.json"
YAMNET = ART / "tfhub/9616fd04ec2360621642ef9455b84f4b668e219e"
# Fixed before test scoring. All normal target clips are allowed training data.
CONFIGS = [
    ("control", "mean", "source", "mahal"),
    ("mean_adapt", "mean", "all", "mahal"),
    ("temporal_source", "mean_std", "source", "mahal"),
    ("temporal_adapt", "mean_std", "all", "mahal"),
    ("mean_domain_centers", "mean", "all", "domain_centers"),
    ("temporal_domain_centers", "mean_std", "all", "domain_centers"),
    ("mean_knn_source", "mean", "source", "cosine"),
    ("mean_knn_adapt", "mean", "all", "cosine"),
    ("temporal_knn_adapt", "mean_std", "all", "cosine"),
    ("mel_pca_source", "mel", "source", "pca"),
    ("mel_pca_adapt", "mel", "all", "pca"),
    ("normalized_mel_pca", "mel_normalized", "all", "pca"),
    ("mel_mahal", "mel", "all", "mahal"),
    ("normalized_mel_mahal", "mel_normalized", "all", "mahal"),
    ("hybrid_pca", "hybrid", "all", "pca"),
    ("efficientat_mahal_source", "efficientat", "source", "mahal"),
    ("efficientat_mahal_adapt", "efficientat", "all", "mahal"),
    ("efficientat_knn_source", "efficientat", "source", "cosine"),
    ("efficientat_knn_adapt", "efficientat", "all", "cosine"),
]


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def extract(rows):
    paths = [ART / "dcase/dev_valve_extracted/valve" / r["split"] / r["filename"] for r in rows]
    hashes = [sha(p) for p in paths]
    assert hashes == [r["wav_sha256"] for r in rows]
    identity = hashlib.sha256(json.dumps([(r["filename"], h) for r, h in zip(rows, hashes)]).encode()).hexdigest()
    weights = [(str(p.relative_to(YAMNET)), sha(p)) for p in sorted(YAMNET.rglob("*")) if p.is_file()]
    signature = hashlib.sha256(json.dumps([identity, weights, sha(ART / "mn04_as_mAP_432.pt"), "mean_std_mel64_globalmedian_v1"]).encode()).hexdigest()
    if FEATURES.exists():
        with np.load(FEATURES, allow_pickle=False) as saved:
            assert str(saved["signature"]) == signature, "Cache identity mismatch"
            return {k: saved[k] for k in ("mean", "mean_std", "mel", "mel_normalized", "efficientat")}, signature
    import tensorflow as tf
    import torch
    from thud.efficientat import load_efficientat, efficientat_wav_embedding
    torch.set_num_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = tf.saved_model.load(str(YAMNET))
    bundle = load_efficientat(ART / "mn04_as_mAP_432.pt", ART / "efficientat-source")
    values = {k: [] for k in ("mean", "mean_std", "mel", "mel_normalized", "efficientat")}
    for i, path in enumerate(paths):
        x, sr = sf.read(path, dtype="float32")
        assert sr == 16000 and x.shape == (160000,) and np.isfinite(x).all() and np.max(np.abs(x)) <= 1
        _, frames, mel = model(x)
        frames, mel = np.asarray(frames), np.asarray(mel)
        mean, std = frames.mean(0), frames.std(0)
        values["mean"].append(mean)
        values["mean_std"].append(np.r_[mean, std])
        values["mel"].append(np.r_[mel.mean(0), mel.std(0)])
        # Global median subtraction removes clip-wide log-level, not spectral shape.
        normalized = mel - np.median(mel)
        values["mel_normalized"].append(np.r_[normalized.mean(0), normalized.std(0)])
        values["efficientat"].append(efficientat_wav_embedding(bundle, path))
        if (i + 1) % 100 == 0:
            print(f"Extracted {i + 1}/{len(paths)}", flush=True)
    values = {k: np.stack(v) for k, v in values.items()}
    assert all(np.isfinite(v).all() for v in values.values())
    np.savez_compressed(FEATURES, signature=signature, **values)
    return values, signature


def score_normal(train, query, method, domains):
    """Fit normal-only inputs. Test labels/domain are intentionally absent."""
    if method == "mahal":
        return LedoitWolf().fit(train).mahalanobis(query)
    if method == "cosine":
        model = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(train)
        return model.kneighbors(query)[0][:, 0]
    if method == "pca":
        scaler = StandardScaler().fit(train)
        normal, test = scaler.transform(train), scaler.transform(query)
        model = PCA(n_components=16, svd_solver="full").fit(normal)
        return np.mean((test - model.inverse_transform(model.transform(test))) ** 2, axis=1)
    if method == "domain_centers":
        centers = {d: train[domains == d].mean(0) for d in np.unique(domains)}
        residuals = np.stack([row - centers[d] for row, d in zip(train, domains)])
        model = LedoitWolf(assume_centered=True).fit(residuals)
        # Both centers are tried for every clip; never route by its test domain.
        return np.min([model.mahalanobis(query - center) for center in centers.values()], axis=0)
    raise ValueError(method)


def metrics(labels, scores, domains):
    result = {"pooled_auc": float(roc_auc_score(labels, scores)),
              "pauc_0_1": float(roc_auc_score(labels, scores, max_fpr=.1))}
    for domain in ("source", "target"):
        mask = (labels == 1) | (domains == domain)
        result[domain + "_auc"] = float(roc_auc_score(labels[mask], scores[mask]))
    return result


def self_check():
    rng = np.random.default_rng(42)
    normal = rng.normal(size=(80, 24))
    queries = np.stack([normal[0], np.full(24, 20.)])
    for method in ("mahal", "pca", "domain_centers"):
        scores = score_normal(normal, queries, method, np.array(["source"] * 40 + ["target"] * 40))
        assert scores.shape == (2,) and scores[1] > scores[0]
    assert score_normal(normal, normal[:1], "cosine", None)[0] < 1e-10


def main():
    self_check()
    start = time.perf_counter()
    baseline = json.loads((ART / "dcase-benchmark.json").read_text())
    rows = sorted(baseline["split_records"], key=lambda r: (r["split"], r["filename"]))
    names = [r["filename"] for r in rows]
    assert len(set(names)) == 1200
    train = np.array([r["split"] == "train" for r in rows])
    domains = np.array([r["domain"] for r in rows])
    assert all(r["label"] == "normal" for r in rows if r["split"] == "train")
    assert train.sum() == 1000 and (~train).sum() == 200
    plan = {"variants": CONFIGS, "test_status": "Previously inspected; exploratory only", "tuning": "None; fixed variants, PCA16, cosine1NN, LedoitWolf shrinkage", "bootstrap": "1000 paired stratified clip resamples, seed42; descriptive not selection-adjusted"}
    (ART / "dcase-ablation-plan.json").write_text(json.dumps(plan, indent=2))
    features, signature = extract(rows)
    features["hybrid"] = np.c_[features["mean_std"], features["mel_normalized"]]
    results = {}
    # Construct all scores before reading test labels for evaluation.
    for name, representation, population, method in CONFIGS:
        fit = train & ((domains == "source") if population == "source" else True)
        # Preserve the float32 control: covariance inversion is precision-sensitive.
        x = features[representation]
        started = time.perf_counter()
        score = score_normal(x[fit], x[~train], method, domains[fit])
        assert np.isfinite(score).all() and score.shape == (200,)
        results[name] = {"features": representation, "scorer": method, "normal_training_count": int(fit.sum()), "dimensions": x.shape[1], "scores": score.tolist(), "fit_score_seconds": time.perf_counter() - started}
        print("Scored", name, flush=True)
    labels = np.array([int(r["label"] == "anomaly") for r in rows if r["split"] == "test"])
    test_domains = domains[~train]
    rng = np.random.default_rng(42)
    resamples = [np.concatenate([rng.choice(np.flatnonzero(labels == y), (labels == y).sum(), replace=True) for y in (0, 1)]) for _ in range(1000)]
    control = np.asarray(results["control"]["scores"])
    base_auc = roc_auc_score(labels, control)
    assert abs(base_auc - baseline["metrics"]["pooled_all_test"]["roc_auc"]) < 1e-3
    for name, result in results.items():
        score = np.asarray(result["scores"])
        result.update(metrics(labels, score, test_domains))
        delta = [roc_auc_score(labels[ix], score[ix]) - roc_auc_score(labels[ix], control[ix]) for ix in resamples]
        result["auc_delta_vs_control"] = result["pooled_auc"] - base_auc
        result["paired_bootstrap_delta_95_interval"] = np.quantile(delta, [.025, .975]).tolist()
        print(name, {k: result[k] for k in ("pooled_auc", "pauc_0_1", "source_auc", "target_auc")}, flush=True)
    report = {**plan, "signature": signature, "script_sha256": sha(Path(__file__)), "dataset": baseline["dataset"], "test_records": [r for r in rows if r["split"] == "test"], "results": results, "wall_seconds": time.perf_counter() - start,
              "limitations": ["One valve section; test reused and 19 variants, no independent generalization claim", "Clip bootstrap assumes independence; overlapping sessions may make intervals optimistic", "No threshold calibration or fault diagnosis", "No edge timing or deployment measured", "CC BY-NC-SA 4.0 data; no commercial-use conclusion"]}
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
