"""Released CORSMAL ACC, frozen external ACM-S2 evaluation; run from repo root.

Assets live in artifacts/datasets/acc. No training or test-set tuning occurs.
Original CORSMAL preprocessing/cascade source is pinned and MIT licensed there.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/datasets/acc"
SOURCE = BASE / "source-33c32e1"
DATA = ROOT / "artifacts/datasets/acm_s2"
OUT = ROOT / "artifacts/acc-benchmark.json"
sys.path.insert(0, str(BASE / "python-deps"))
sys.path.insert(0, str(SOURCE))

import numpy as np

# Historical librosa 0.8.0 references removed NumPy scalar aliases.
np.complex = complex
np.float = float
np.int = int
import h5py
import librosa
import pandas as pd
import psutil
import scipy
from scipy import ndimage
import skimage
from skimage.transform import AffineTransform, warp
import tensorflow as tf
from keras.src.legacy.saving import saving_utils
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
LABELS = ["empty", "pasta_50", "pasta_90", "rice_50", "rice_90", "water_50", "water_90"]
ACTIONS = ["empty", "pouring", "shaking"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_resize():
    # 0.18 used an affine warp; modern resize uses ndimage.zoom instead.
    tree = ast.parse((BASE / "skimage-0.18.1-warps.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resize")
    namespace = {"np": np, "ndi": ndimage, "AffineTransform": AffineTransform, "warp": warp}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "skimage-0.18.1-resize", "exec"), namespace)
    return namespace["resize"]


PROCESSOR = source_module("acc_audio_processing", SOURCE / "methods/processing/audio_processing.py")
PROCESSOR.resize = legacy_resize()
CASCADE = source_module("acc_components", SOURCE / "methods/acc/src/acc_components.py")


def prepare(paths):
    waves = [librosa.load(str(p), sr=22050, mono=True, res_type="kaiser_fast")[0] for p in paths]
    df = pd.DataFrame({"waveform": waves, "spectrogram": [None] * len(waves)})
    df = PROCESSOR.audio_processing(df)
    x = np.stack(df.spectrogram).astype(np.float32)[..., None]
    assert x.shape == (len(paths), 96, 96, 1) and np.isfinite(x).all()
    return x


def load_models():
    result, info = [], []
    for name in ("action", "pouring", "shaking"):
        architecture = BASE / f"acc_{name}.json"
        weights = BASE / f"acc_{name}.h5"
        cfg = json.loads(architecture.read_text())
        started = time.perf_counter()
        model = saving_utils.model_from_config(cfg)
        model.load_weights(weights)  # Strict topology loading: no skipped/mismatched weights.
        elapsed = (time.perf_counter() - started) * 1000
        # Prove every loaded variable equals its released HDF5 tensor exactly.
        with h5py.File(weights) as f:
            group = f["model_weights"] if "model_weights" in f else f
            arrays = []
            for layer_name in group.attrs["layer_names"]:
                layer = group[layer_name]
                arrays.extend(layer[n][()] for n in layer.attrs["weight_names"])
        assert len(arrays) == len(model.weights)
        for loaded, released in zip(model.weights, arrays):
            np.testing.assert_array_equal(loaded.numpy(), released)
        assert model.input_shape == (None, 96, 96, 1)
        result.append(model)
        info.append({"name": name, "parameters": model.count_params(), "input_shape": model.input_shape,
                     "output_shape": model.output_shape, "load_ms": elapsed,
                     "architecture_sha256": digest(architecture), "weights_sha256": digest(weights),
                     "h5_bytes": weights.stat().st_size, "exact_hdf5_tensor_equality": True,
                     "released_keras_version": cfg.get("keras_version")})
    return result, info


def predict(models, x):
    probabilities = models[0](x, training=False).numpy()
    actions = probabilities.argmax(1)
    joints, contents = [], []
    for i, action in enumerate(actions):
        if action == 0:
            joints.append(0)
            contents.append(None)
        else:
            scores = models[int(action)](x[i:i + 1], training=False).numpy()[0]
            joints.append(int(scores.argmax()) + 1)
            contents.append(scores.tolist())
    return np.array(joints), actions, probabilities, contents


def split_joint(joint):
    return np.where(joint == 0, 0, (joint + 1) // 2), np.where(joint == 0, 0, (joint - 1) % 2 + 1)


def metrics(truth, pred, labels):
    return {"accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=list(range(len(labels))), average="macro", zero_division=0)),
            "labels": labels, "confusion_true_rows_pred_columns": confusion_matrix(truth, pred, labels=list(range(len(labels)))).tolist()}


def timing(values):
    return {"n": len(values), "p50_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)),
            "min_ms": float(min(values)), "max_ms": float(max(values))}


def run_ccm(root, split):
    """Score complete published filename splits; never infer test labels."""
    started = time.perf_counter()
    split_files = {
        name: SOURCE / "CCM_samples" / f"{name}_sam_filename.txt"
        for name in ("train", "validation")
    }
    lists = {name: path.read_text().splitlines() for name, path in split_files.items()}
    assert len(lists["train"]) == 547 and len(lists["validation"]) == 137
    assert not set(lists["train"]) & set(lists["validation"])
    names = lists[split] if split != "all-labelled" else lists["train"] + lists["validation"]
    assert len(names) == len(set(names))
    root = Path(root).resolve()
    paths = [(root / name).resolve() for name in names]
    assert all(p.is_relative_to(root) for p in paths)
    missing = [name for name, path in zip(names, paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CCM {split}: missing {len(missing)}/{len(paths)} files; first: {missing[:3]}. Root must contain train/<container>/audio/<filename>.")
    models, model_info = load_models()
    examples, tensors, timings = [], [], []
    for name, path in zip(names, paths):
        before = time.perf_counter()
        x = prepare([path])
        frontend_ms = (time.perf_counter() - before) * 1000
        before = time.perf_counter()
        joint, actions, probabilities, content = predict(models, x)
        inference_ms = (time.perf_counter() - before) * 1000
        stem = path.name
        true_type = int(stem[5])
        true_level = int(stem[9])
        assert true_type in range(4) and true_level in range(3)
        truth = 0 if true_type == 0 else (true_type - 1) * 2 + true_level
        pred_type, pred_level = split_joint(joint)
        tensors.append(x[0])
        examples.append({"path": name, "audio_sha256": digest(path), "container_id": int(Path(name).parts[1]),
                         "true_type": true_type, "true_level": true_level, "true_joint_id": truth,
                         "true_joint": LABELS[truth], "pred_joint_id": int(joint[0]), "pred_joint": LABELS[int(joint[0])],
                         "pred_type": int(pred_type[0]), "pred_level": int(pred_level[0]),
                         "pred_action": ACTIONS[int(actions[0])], "action_probabilities": probabilities[0].tolist(),
                         "routed_content_probabilities": content[0], "correct": bool(truth == joint[0]),
                         "frontend_ms": frontend_ms, "initial_inference_ms": inference_ms})
    x = np.stack(tensors)
    for i in range(20):
        j = i % len(x)
        predict(models, x[j:j + 1])
    for i in range(100):
        j = i % len(x)
        before = time.perf_counter()
        predict(models, x[j:j + 1])
        timings.append((time.perf_counter() - before) * 1000)
    scores = {
        "filling_type": metrics([e["true_type"] for e in examples], [e["pred_type"] for e in examples], ["empty", "pasta", "rice", "water"]),
        "filling_level": metrics([e["true_level"] for e in examples], [e["pred_level"] for e in examples], ["empty_0", "half_50", "full_90"]),
        "joint": metrics([e["true_joint_id"] for e in examples], [e["pred_joint_id"] for e in examples], LABELS),
    }
    report = {"status": "complete", "dataset": "CORSMAL Containers Manipulation", "split": split, "n": len(examples),
              "protocol": "Frozen released ACC; published filename split; no fitting. Validation shares containers with training and is not container-held-out/external generalization.",
              "split_source": {str(p.relative_to(BASE)): digest(p) for p in split_files.values()},
              "split_caveat": "Filename lists used exactly. Published evaluator train_ids/val_ids do not select the same file membership when interpreted against annotations.csv: 112 validation names differ. Do not equate these split definitions.",
              "ground_truth": "Semantic fi/fu labels in official labelled-training filenames; empty0, pasta1, rice2, water3; fu0/1/2 denotes0/50/90%.",
              "public_private_status": "Unscored: official annotations contain -1 filling labels for containers10-15.",
              "models": model_info, "metrics": scores, "examples": examples,
              "misclassifications": [e for e in examples if not e["correct"]],
              "desktop_cpu": {"platform": platform.platform(), "tensorflow": tf.__version__, "keras": tf.keras.__version__,
                              "intra_threads": 2, "inter_threads": 2, "gpu_disabled": True,
                              "warmups": 20, "preprocessed_routed_inference": timing(timings),
                              "per_clip_frontend": timing([e["frontend_ms"] for e in examples]),
                              "process_rss_bytes": psutil.Process().memory_info().rss},
              "script_sha256": digest(__file__), "source_commit": "33c32e147d207b6f66bcbb9e1fa1b24a77fd7552",
              "weights_sha256": digest(BASE / "acc_pretrained_models.zip"),
              "licenses": {"code": "MIT", "weights": "CC-BY-4.0", "CCM_audio": "Consult exact CCM distribution license; do not infer from code or weights"},
              "total_runtime_seconds": time.perf_counter() - started,
              "limitations": ["No iPhone or mobile conversion measurement", "Released filename validation is in-domain; do not call it independent benchmark", "Train/all-labelled results include training-source recordings and are resubstitution evidence only", "No action truth is inferred from container IDs or materials"]}
    output = ROOT / f"artifacts/acc-ccm-{split}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / f"artifacts/acc-ccm-{split}-inputs.npz", spectrograms=x)
    print(json.dumps({"report": str(output), "metrics": scores, "inference": timing(timings)}, indent=2))


def main():
    started = time.perf_counter()
    process = psutil.Process()
    before_rss = process.memory_info().rss
    archive = BASE / "acc_pretrained_models.zip"
    assert archive.stat().st_size == 60659599
    assert hashlib.md5(archive.read_bytes()).hexdigest() == "05574ea3430c970e07d3af8e4beabc23"
    models, model_info = load_models()
    loaded_rss = process.memory_info().rss
    ann = json.loads((DATA / "acm_s2_annotations.json").read_text(encoding="utf-8-sig"))
    semantic = {a["id"]: a for a in ann["annotations"]}
    audios = sorted(ann["audios"], key=lambda a: a["audio_id"])
    assert len(audios) == 21 and len(semantic) == 21
    paths = [DATA / "acm_s2_audio" / a["filename"] for a in audios]
    pre_start = time.perf_counter()
    x = prepare(paths)
    preprocess_ms = (time.perf_counter() - pre_start) * 1000
    pred_start = time.perf_counter()
    joint, actions, action_probs, content_probs = predict(models, x)
    predict_ms = (time.perf_counter() - pred_start) * 1000
    # Compare the direct inference path against the unmodified published cascade.
    original_joint, original_actions = CASCADE.acc_cascade(models, x)
    np.testing.assert_array_equal(joint, original_joint)
    np.testing.assert_array_equal(actions, original_actions)
    truth_type = np.array([semantic[a["audio_id"]]["filling_type"] for a in audios])
    truth_level = np.array([semantic[a["audio_id"]]["filling_level"] for a in audios])
    truth_joint = np.where(truth_type == 0, 0, (truth_type - 1) * 2 + truth_level)
    pred_type, pred_level = split_joint(joint)
    np.testing.assert_array_equal(split_joint(np.arange(7))[0], [0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(split_joint(np.arange(7))[1], [0, 1, 2, 1, 2, 1, 2])
    examples = []
    for i, (a, p) in enumerate(zip(audios, paths)):
        s = semantic[a["audio_id"]]
        examples.append({"sample_id": a["audio_id"], "filename": a["filename"], "container_id": s["containerID"],
                         "audio_sha256": digest(p), "duration_seconds": a["num_frames"] / a["sampling_freq"],
                         "true_type": int(truth_type[i]), "pred_type": int(pred_type[i]),
                         "true_level": int(truth_level[i]), "pred_level": int(pred_level[i]),
                         "true_joint": LABELS[int(truth_joint[i])], "pred_joint": LABELS[int(joint[i])],
                         "pred_action": ACTIONS[int(actions[i])], "action_probabilities": action_probs[i].tolist(),
                         "routed_content_probabilities": content_probs[i], "correct": bool(joint[i] == truth_joint[i])})
    # Fixed round-robin clips: 20 warmups, then 100 single-recording routed inferences.
    for i in range(20):
        predict(models, x[i % len(x):i % len(x) + 1])
    latencies, routed_latency = [], {a: [] for a in ACTIONS}
    for i in range(100):
        j = i % len(x)
        t = time.perf_counter()
        _, a, _, _ = predict(models, x[j:j + 1])
        elapsed = (time.perf_counter() - t) * 1000
        latencies.append(elapsed)
        routed_latency[ACTIONS[int(a[0])]].append(elapsed)
    # Frontend has separate cost: do not conflate captured-audio wait with compute.
    frontend_times = []
    for i in range(5):
        t = time.perf_counter()
        prepare([paths[i]])
        frontend_times.append((time.perf_counter() - t) * 1000)
    smoke_paths = sorted((BASE / "demo").glob("*.wav"))
    assert len(smoke_paths) == 7
    smoke_x = prepare(smoke_paths)
    smoke_joint, smoke_actions, _, _ = predict(models, smoke_x)
    smoke_truth = np.array([0 if int(p.name[5]) == 0 else (int(p.name[5]) - 1) * 2 + int(p.name[9]) for p in smoke_paths])
    smoke = [{"filename": p.name, "audio_sha256": digest(p), "true_joint": LABELS[int(y)],
              "pred_joint": LABELS[int(yp)], "pred_action": ACTIONS[int(a)]}
             for p, y, yp, a in zip(smoke_paths, smoke_truth, smoke_joint, smoke_actions)]
    report = {
        "status": "complete", "script_sha256": digest(__file__),
        "protocol": "Frozen released ACC ensemble; all 21 ACM-S2 recordings external test; no fitting, tuning, random windows, or calibration on ACM-S2",
        "training_source": "Zenodo metadata: weights trained on CCM training split; ACM-S2 is the authors' separate setup validation set, not a new independent replication dataset",
        "source_commit": "33c32e147d207b6f66bcbb9e1fa1b24a77fd7552",
        "source_url": "https://github.com/CORSMAL/ACC/tree/33c32e147d207b6f66bcbb9e1fa1b24a77fd7552",
        "weights_url": "https://zenodo.org/records/4770061", "weights_archive_bytes": archive.stat().st_size,
        "weights_archive_sha256": digest(archive), "weights_archive_md5": "05574ea3430c970e07d3af8e4beabc23",
        "dataset_archive_sha256": digest(DATA / "ACM-S2_dataset.zip"),
        "annotations_sha256": digest(DATA / "acm_s2_annotations.json"),
        "licenses": {"code": "MIT; pinned LICENSE preserved", "weights": "CC-BY-4.0 in Zenodo 4770061 metadata", "dataset": "CC-BY-4.0 in ACM-S2 annotations"},
        "preprocessing": "Original audio_processing.py: librosa 0.8.0 mono 22050Hz kaiser_fast; peak normalize; crop at second detected onset if available; truncate/pad 10 seconds; default STFT n_fft2048/hop512/Hann/center reflect; amplitude_to_db ref1 top_db80; historical skimage 0.18.1 affine resize 96x96 with Gaussian anti-aliasing; float32 NHWC",
        "compatibility": ["NumPy removed scalar aliases restored only inside benchmark process for librosa 0.8.0", "Keras legacy saving_utils.model_from_config with strict load_weights; every loaded tensor exactly equals released HDF5 tensor", "Historical resize function extracted from pinned scikit-image 0.18.1 source; current scikit-image affine warp compiled primitive and SciPy Gaussian primitive used", "Direct model(training=False) cascade matches original published acc_cascade on all 21 test recordings", "Full historical TF2.3 environment equivalence not measured; released architecture metadata itself varies across Keras2.4 and2.2.4-tf"],
        "routing": "action 0 => joint empty; action1 => pouring argmax+1; action2 => shaking argmax+1. Joint0 empty,1 pasta50,2 pasta90,3 rice50,4 rice90,5 water50,6 water90",
        "models": model_info,
        "metrics": {"filling_type": metrics(truth_type, pred_type, ["empty", "pasta", "rice", "water"]), "filling_level": metrics(truth_level, pred_level, ["empty_0", "half_50", "full_90"]), "joint": metrics(truth_joint, joint, LABELS)},
        "examples": examples,
        "misclassifications": [e for e in examples if not e["correct"]],
        "action_ground_truth": "Not present in ACM-S2 annotations; action accuracy not scored",
        "desktop_cpu": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count(), "tensorflow_threads": {"intra": 2, "inter": 2}, "gpu_disabled": True,
                        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": tf.keras.__version__, "numpy": np.__version__, "librosa": librosa.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__},
                        "warmups": 20, "inference": timing(latencies), "inference_per_route": {k: timing(v) for k, v in routed_latency.items() if v}, "timing_protocol": "100 single-clip direct calls incl action and routed content network, round robin original21; preprocessed input, synchronized via numpy; no capture wait",
                        "frontend_five_clips": timing(frontend_times), "initial_21_frontend_ms_including_first_use": preprocess_ms, "initial_21_predict_ms": predict_ms,
                        "rss_before_models_bytes": before_rss, "rss_after_models_bytes": loaded_rss, "rss_final_bytes": process.memory_info().rss,
                        "windows_peak_working_set_bytes": process.memory_info().peak_wset if hasattr(process.memory_info(), "peak_wset") else None},
        "demo_smoke": {"role": "Seven official CCM validation demo clips, same in-domain source; NOT independent external benchmark", "accuracy": float(accuracy_score(smoke_truth, smoke_joint)), "examples": smoke},
        "source_file_sha256": {str(p.relative_to(BASE)): digest(p) for p in SOURCE.rglob("*") if p.is_file() and "__pycache__" not in p.parts},
        "historical_resize_source_sha256": digest(BASE / "skimage-0.18.1-warps.py"),
        "limitations": ["21 clips only, four containers, one scenario; no confidence intervals implying broad generalization", "Pouring/shaking protocol, not taps or universal fill percentage regression", "Model timings are Windows desktop CPU only; no iPhone, CoreML, energy or mobile conversion measurements", "Resident process memory includes Python/TensorFlow/dependencies, not solely model tensors", "No ground-truth actions in external annotation; route errors cannot be definitively labeled from filling labels"],
        "total_runtime_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / "artifacts/acc-predictions.npz", input_spectrograms=x, joint_truth=truth_joint, joint_prediction=joint, action_prediction=actions, action_probabilities=action_probs)
    print(json.dumps({"report": str(OUT), "metrics": report["metrics"], "inference": report["desktop_cpu"]["inference"], "smoke_accuracy": report["demo_smoke"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccm-root", type=Path, help="Existing CCM root containing train/<container>/audio; no downloads")
    parser.add_argument("--ccm-split", choices=["validation", "train", "all-labelled"], default="validation")
    args = parser.parse_args()
    try:
        if args.ccm_root is None:
            main()
        else:
            run_ccm(args.ccm_root, args.ccm_split)
    except Exception:
        (ROOT / "artifacts/acc-failure.json").write_text(json.dumps({"status": "failed", "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise
