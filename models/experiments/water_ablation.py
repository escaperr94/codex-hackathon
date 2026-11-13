"""Exploratory fixed-matrix shape ablation; run from the repository root.

Reads the original 125-file pilot and immutable caches, writes only water-ablation*
artifacts. No downloads, label-dependent feature extraction, or backbone updates.
"""
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import collections
import csv
import hashlib
import json
import math
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path("artifacts/sound-of-water")
SOURCE_REPORT = Path("artifacts/sound-water-benchmark.json")
REPORT = Path("artifacts/water-ablation.json")
DECLARATION = Path("artifacts/water-ablation-protocol.json")
TEMPORAL_CACHE = Path("artifacts/water-ablation-temporal.npz")
HEADS = Path("artifacts/water-ablation-heads.npz")
SEED = 42
CS = [0.01, 0.1, 1.0, 10.0]
# Fixed before extraction or scoring; the nonlinear comparator adds only eight fits per fold.
CONFIGURATIONS = [
    {"name": "yamnet_mean_control", "features": "yamnet_mean", "head": "logistic", "dimensions": 1024},
    {"name": "yamnet_mean_std", "features": "yamnet_mean_std", "head": "logistic", "dimensions": 2048},
    {"name": "yamnet_three_bins", "features": "yamnet_three_bins", "head": "logistic", "dimensions": 3072},
    {"name": "yamnet_mfcc_fusion", "features": "yamnet_mfcc", "head": "logistic", "dimensions": 1050},
    {"name": "yamnet_efficientat_fusion", "features": "yamnet_efficientat", "head": "logistic", "dimensions": 1408},
    {"name": "yamnet_mean_rbf", "features": "yamnet_mean", "head": "rbf", "dimensions": 1024},
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(truth, predictions, classes):
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
            "macro_f1": float(f1_score(truth, predictions, labels=classes, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=classes).tolist(),
            "class_order": classes}


def aggregate(indices, scores, head_classes, rows, score_type):
    grouped = collections.defaultdict(list)
    truth = {}
    for i, score in zip(indices, scores):
        group = rows[i]["container_id"]
        grouped[group].append(score)
        if group in truth:
            assert truth[group] == rows[i]["shape"]
        truth[group] = rows[i]["shape"]
    result = []
    for group in sorted(grouped):
        mean = np.mean(grouped[group], axis=0)
        result.append({"container_id": group, "truth": truth[group],
                       "prediction": str(head_classes[np.argmax(mean)]),
                       "mean_" + score_type: {str(c): float(v) for c, v in zip(head_classes, mean)},
                       "recordings": len(grouped[group])})
    return result


def make_head(config, parameters):
    if config["head"] == "logistic":
        estimator = LogisticRegression(**parameters, class_weight="balanced", max_iter=3000, random_state=SEED)
    else:
        # No hidden recording-level probability-calibration CV: aggregate OVR scores instead.
        estimator = SVC(**parameters, kernel="rbf", class_weight="balanced", probability=False,
                        decision_function_shape="ovr", break_ties=True, cache_size=64, random_state=SEED)
    return make_pipeline(StandardScaler(), estimator)


def scores_for(head, x, config):
    return head.predict_proba(x) if config["head"] == "logistic" else head.decision_function(x)


def head_arrays(head):
    scaler, estimator = head.steps[0][1], head.steps[1][1]
    arrays = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "classes": estimator.classes_}
    if isinstance(estimator, LogisticRegression):
        arrays.update(coefficients=estimator.coef_, intercept=estimator.intercept_)
    else:
        arrays.update(support_vectors=estimator.support_vectors_, dual_coefficients=estimator.dual_coef_,
                      intercept=estimator.intercept_, n_support=estimator.n_support_, gamma=np.asarray(estimator._gamma))
    return arrays


def main():
    started = time.perf_counter()
    original = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    classes = original["target"]["classes"]
    items = [row["item_id"] for row in original["audio_file_metadata"]]
    assert len(items) == len(set(items)) == 125
    metadata = {}
    for filename in ("train.csv", "test_I.csv", "test_II.csv", "test_III.csv"):
        with (ROOT / "splits" / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["item_id"] in metadata:
                    assert all(metadata[row["item_id"]][key] == row[key] for key in ("container_id", "shape"))
                metadata[row["item_id"]] = row
    rows = [metadata[item] for item in items]
    index = {item: i for i, item in enumerate(items)}
    y = np.array([row["shape"] for row in rows])

    def indices(split):
        result = np.array([index[item] for item in split["item_ids"]])
        assert len(result) == len(set(result))
        assert {rows[i]["container_id"] for i in result} == set(split["container_ids"])
        return result

    train, test = (indices(original["split"][part]) for part in ("train", "test"))
    train_groups = {rows[i]["container_id"] for i in train}
    test_groups = {rows[i]["container_id"] for i in test}
    assert len(train) == 97 and len(test) == 28
    assert len(train_groups) == 36 and len(test_groups) == 12
    assert not train_groups & test_groups
    assert not set(train) & set(test) and set(train) | set(test) == set(range(125))
    folds = []
    for fold in original["split"]["inner_validation"]:
        tr, va = indices(fold["train"]), indices(fold["validation"])
        tr_groups, va_groups = ({rows[i]["container_id"] for i in subset} for subset in (tr, va))
        assert not tr_groups & va_groups and tr_groups | va_groups == train_groups
        assert not (tr_groups | va_groups) & test_groups
        assert not set(tr) & set(va) and set(tr) | set(va) == set(train)
        assert set(y[tr]) == set(classes) and set(y[va]) == set(classes)
        folds.append((tr, va))
    assert len(folds) == 3
    assert sorted(i for _, va in folds for i in va) == sorted(train)
    for group in train_groups | test_groups:
        assert len({row["shape"] for row in rows if row["container_id"] == group}) == 1

    grids = {"logistic": [{"C": c} for c in CS],
             "rbf": [{"C": c, "gamma": gamma} for c in CS for gamma in ("scale", 0.0001)]}
    protocol = {"status": "declared_before_current_run_predictions", "exploratory": True,
                "configurations": CONFIGURATIONS, "parameter_grids": grids,
                "selection": "Maximum mean inner-fold container macro-F1; ties use first declared candidate. No test-dependent changes.",
                "standardization": "Per-feature StandardScaler fit within each training fold; fusion concatenates original mean YAMNet with the indicated cached representation, with no block reweighting.",
                "aggregation": {"logistic": "Mean recording probabilities per container, then argmax.",
                                "rbf": "Mean uncalibrated OVR decision scores per container, then argmax; no internal probability-calibration CV."},
                "temporal_pooling": "Mean+population std of frozen frame embeddings; three-bin pooling uses numpy.array_split on ordered frames and concatenates the three means, with no crop, augmentation or duration selection.",
                "split": original["split"], "item_ids_in_cache_order": items,
                "source_report_sha256": sha256(SOURCE_REPORT), "script_sha256": sha256(Path(__file__)),
                "test_exposure": "All original test results were previously inspected. This is exploratory comparison, not untouched confirmation."}
    DECLARATION.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    manifest = [(item, sha256(ROOT / "audios" / (item + ".wav"))) for item in items]
    fingerprint = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    assert fingerprint == original["reproducibility"]["manifest_fingerprint"]
    provenance = original["reproducibility"]
    base_cache = Path(provenance["feature_cache_path"])
    efficient_cache = Path(provenance["efficientat"]["cache_path"])
    assert sha256(base_cache) == provenance["feature_cache_sha256"]
    assert sha256(efficient_cache) == provenance["efficientat"]["cache_sha256"]
    with np.load(base_cache, allow_pickle=False) as saved:
        assert str(saved["manifest_fingerprint"]) == fingerprint
        assert [r["item_id"] for r in json.loads(str(saved["audio_info_json"]))] == items
        mean, mfcc = saved["yamnet"], saved["mfcc"]
    with np.load(efficient_cache, allow_pickle=False) as saved:
        assert str(saved["manifest_fingerprint"]) == fingerprint and saved["item_ids"].tolist() == items
        assert str(saved["checkpoint_sha256"]) == original["models"]["efficientat"]["checkpoint_sha256"]
        efficient = saved["embeddings"]
    assert mean.shape == (125, 1024) and mfcc.shape == (125, 26) and efficient.shape == (125, 384)
    checkpoint_files = original["models"]["yamnet"]["checkpoint_files"]
    for record in checkpoint_files:
        assert sha256(Path(record["path"])) == record["sha256"]
    checkpoint_fingerprint = hashlib.sha256(json.dumps(checkpoint_files, sort_keys=True).encode()).hexdigest()
    temporal_reused = TEMPORAL_CACHE.exists()
    tensorflow_version = None
    if temporal_reused:
        with np.load(TEMPORAL_CACHE, allow_pickle=False) as saved:
            assert saved["item_ids"].tolist() == items
            assert str(saved["manifest_fingerprint"]) == fingerprint
            assert str(saved["checkpoint_fingerprint"]) == checkpoint_fingerprint
            std, thirds, recomputed_mean = saved["std"], saved["three_bins"], saved["mean"]
            frame_counts = saved["frame_counts"].tolist()
            tensorflow_version = str(saved["tensorflow_version"])
    else:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tensorflow_version = tf.__version__
        model = tf.saved_model.load(original["models"]["yamnet"]["saved_model_path"])
        means, stds, bins, frame_counts = [], [], [], []
        for position, item in enumerate(items):
            waveform, rate = sf.read(ROOT / "audios" / (item + ".wav"), dtype="float32", always_2d=True)
            mono = waveform.mean(axis=1)
            assert mono.size and np.isfinite(mono).all() and np.abs(mono).max() <= 1.000001
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(mono, 16000 // divisor, rate // divisor).astype(np.float32) if rate != 16000 else mono
            _, embeddings, _ = model(audio)
            embeddings = np.asarray(embeddings)
            assert embeddings.ndim == 2 and embeddings.shape[1] == 1024 and len(embeddings) >= 3
            assert np.isfinite(embeddings).all()
            means.append(embeddings.mean(axis=0))
            stds.append(embeddings.std(axis=0))
            bins.append(np.concatenate([part.mean(axis=0) for part in np.array_split(embeddings, 3)]))
            frame_counts.append(len(embeddings))
            if position % 20 == 0:
                print(json.dumps({"phase": "frozen_temporal_features", "completed": position + 1, "total": 125}), flush=True)
        recomputed_mean, std, thirds = map(np.asarray, (means, stds, bins))
        np.savez_compressed(TEMPORAL_CACHE, item_ids=items, mean=recomputed_mean, std=std, three_bins=thirds,
                            frame_counts=frame_counts, manifest_fingerprint=fingerprint,
                            checkpoint_fingerprint=checkpoint_fingerprint, tensorflow_version=tensorflow_version)
    assert std.shape == (125, 1024) and thirds.shape == (125, 3072)
    assert frame_counts == [row["yamnet_frames"] for row in original["audio_file_metadata"]]
    assert np.allclose(recomputed_mean, mean, rtol=1e-5, atol=1e-6), "Frozen extraction differs from original mean cache"
    features = {"yamnet_mean": mean, "yamnet_mean_std": np.concatenate([mean, std], axis=1),
                "yamnet_three_bins": thirds, "yamnet_mfcc": np.concatenate([mean, mfcc], axis=1),
                "yamnet_efficientat": np.concatenate([mean, efficient], axis=1)}
    assert all(np.isfinite(x).all() for x in features.values())

    # Select every head before any current-run test prediction is computed.
    fitted, selections = {}, {}
    for config in CONFIGURATIONS:
        x = features[config["features"]]
        assert x.shape == (125, config["dimensions"])
        score_type = "probabilities" if config["head"] == "logistic" else "decision_scores"
        search = []
        for parameters in grids[config["head"]]:
            fold_metrics = []
            for tr, va in folds:
                head = make_head(config, parameters).fit(x[tr], y[tr])
                records = aggregate(va, scores_for(head, x[va], config), head.classes_, rows, score_type)
                fold_metrics.append(metrics([r["truth"] for r in records], [r["prediction"] for r in records], classes))
            search.append({"parameters": parameters, "fold_container_metrics": fold_metrics,
                           "mean_container_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics]))})
        best = max(search, key=lambda entry: entry["mean_container_macro_f1"])
        selections[config["name"]] = {"selected_parameters": best["parameters"],
                                       "selected_validation_macro_f1": best["mean_container_macro_f1"],
                                       "hyperparameter_search_train_only": search}
        fitted[config["name"]] = make_head(config, best["parameters"]).fit(x[train], y[train])
    selected_configuration = max(selections, key=lambda name: selections[name]["selected_validation_macro_f1"])
    REPORT.write_text(json.dumps({"status": "training_selection_complete_before_test_prediction", "protocol": protocol,
                                 "selections": selections, "selected_configuration_train_only": selected_configuration}, indent=2), encoding="utf-8")

    results, saved_heads = {}, {}
    for config in CONFIGURATIONS:
        name, head = config["name"], fitted[config["name"]]
        x = features[config["features"]]
        score_type = "probabilities" if config["head"] == "logistic" else "decision_scores"
        scores = scores_for(head, x[test], config)
        predictions = head.classes_[scores.argmax(axis=1)]
        records = aggregate(test, scores, head.classes_, rows, score_type)
        arrays = head_arrays(head)
        saved_heads.update({name + "__" + key: value for key, value in arrays.items()})
        numeric = [value for key, value in arrays.items() if key != "classes"]
        results[name] = {**selections[name], "feature_dimensions": config["dimensions"],
                         "held_out_recording_metrics": metrics(y[test], predictions, classes),
                         "held_out_container_metrics": metrics([r["truth"] for r in records], [r["prediction"] for r in records], classes),
                         "container_predictions": records,
                         "recording_predictions": [{"item_id": items[i], "container_id": rows[i]["container_id"],
                                                    "truth": rows[i]["shape"], "prediction": str(prediction)} for i, prediction in zip(test, predictions)],
                         "head_size": {"pickle_protocol_5_bytes": len(pickle.dumps(head, protocol=5)),
                                       "numeric_inference_array_bytes": sum(value.nbytes for value in numeric),
                                       "numeric_inference_array_elements": sum(value.size for value in numeric),
                                       "support_vectors": int(len(head.steps[-1][1].support_vectors_)) if config["head"] == "rbf" else 0,
                                       "scope": "Head including scaler only; excludes encoder weights, runtime RAM, input features and Python object overhead for array-byte count."}}
        print(json.dumps({"configuration": name, "validation_macro_f1": selections[name]["selected_validation_macro_f1"],
                          "test_container_metrics": results[name]["held_out_container_metrics"]}), flush=True)
    np.savez_compressed(HEADS, **saved_heads)
    control = results["yamnet_mean_control"]
    old_control = original["results"]["yamnet_mean_1024"]
    control_matches = (control["selected_parameters"]["C"] == old_control["selected_C"] and
                       control["held_out_container_metrics"] == old_control["held_out_container_metrics"] and
                       control["recording_predictions"] == old_control["recording_predictions"])
    assert control_matches, "Original mean-control result was not reproduced"
    report = {"status": "completed", "experiment": "Fixed finite exploratory Sound-of-Water shape ablation",
              "command": "python experiments/water_ablation.py", "protocol": protocol,
              "selected_configuration_train_only": selected_configuration, "results": results,
              "original_control_reproduced": control_matches, "source": original["source"],
              "reproducibility": {"manifest_fingerprint": fingerprint, "audio_manifest": manifest,
                                  "original_cache_sha256": sha256(base_cache), "efficientat_cache_sha256": sha256(efficient_cache),
                                  "temporal_cache": str(TEMPORAL_CACHE), "temporal_cache_sha256": sha256(TEMPORAL_CACHE),
                                  "temporal_cache_reused": temporal_reused, "checkpoint_files": checkpoint_files,
                                  "mean_reextraction_max_absolute_difference": float(np.max(np.abs(recomputed_mean - mean))),
                                  "head_arrays_path": str(HEADS), "head_arrays_sha256": sha256(HEADS),
                                  "head_arrays_compressed_file_bytes": HEADS.stat().st_size,
                                  "assertions": ["exact 125 unique cached items and audio hashes", "exact original outer/inner item and container splits",
                                                 "complete disjoint fold coverage", "cache/checkpoint identity", "finite feature dimensions",
                                                 "temporal extraction matches original frozen mean", "original control reproduced"]},
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "sklearn": sklearn.__version__, "tensorflow": tensorflow_version,
                              "cpu_threads": 2, "tensorflow_interop_threads": 1, "device": "desktop CPU"},
              "wall_seconds": time.perf_counter() - started,
              "caveats": ["Previously inspected test set: all ablations are exploratory, not independent confirmation; no test scores selected hyperparameters or configurations.",
                          "Only 12 test containers / 28 correlated recordings; balanced accuracy and macro-F1 use the declared three classes. No confidence interval or significance claim.",
                          "Shape from full pouring recordings is not water level, volume, tile debonding, or tap-fill inference; material, geometry, speaker and recording conditions may confound shape.",
                          "Original custom pilot split and all 125 clips are preserved; this is not the official paper split or entire dataset.",
                          "Fusion standardizes coordinates, not feature blocks; wider blocks have more coordinates. No fusion weights were tuned.",
                          "RBF aggregation uses uncalibrated OVR decision scores, unlike logistic probabilities; its head includes stored support vectors, not just coefficients.",
                          "Frozen official YAMNet and existing EfficientAT features only; no backbone finetuning. Temporal bins preserve relative frame order but not absolute duration.",
                          "Head sizes are desktop serialization/numeric storage, not measured iPhone RAM, latency, energy or mobile conversion.",
                          "Dataset and exact checkpoint permissions need separate review; no additional downloads occurred."]}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "selected_configuration_train_only": selected_configuration,
                      "status": "completed", "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
