"""Fruit-held-out durian audio benchmark. Run from repository root with --prepare or --run."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/datasets/durian"
PREFIX = ROOT / "artifacts/durian"
API = "https://zenodo.org/api/records/18603796"
MODEL = ROOT / "artifacts/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e"
SEED = 20261114


def filehash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_get(url, target, expected_size, md5):
    if target.exists() and target.stat().st_size == expected_size and filehash(target, "md5") == md5:
        return
    cap = 1_700_000_000 if expected_size > 1_000_000 else 1_000_000
    assert expected_size <= cap
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            if length and (int(length) > cap or int(length) != expected_size):
                raise ValueError(f"Unexpected Content-Length: {length}")
            digest = hashlib.md5()
            count = 0
            with partial.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    count += len(block)
                    if count > expected_size or count > cap:
                        raise ValueError("Download exceeded declared size/cap")
                    handle.write(block)
                    digest.update(block)
            if count != expected_size or digest.hexdigest() != md5:
                raise ValueError(f"Size/checksum mismatch: {count}, {digest.hexdigest()}")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    # Bound small API metadata too; no unbounded response buffering.
    with requests.get(API, stream=True, timeout=(30, 60)) as response:
        response.raise_for_status()
        body = bytearray()
        for block in response.iter_content(65536):
            body.extend(block)
            if len(body) > 1_000_000:
                raise ValueError("API metadata exceeds cap")
    record = json.loads(body)
    (DATA / "record.json").write_bytes(body)
    chosen = {"dataset_clean_sound.zip", "durian_characteristics_cleaned.csv", "README.md"}
    manifest = {"api": API, "doi": record["doi"], "license": record["metadata"]["license"], "files": []}
    for entry in record["files"]:
        if entry["key"] not in chosen:
            continue
        target = DATA / entry["key"]
        print("Downloading/verifying", entry["key"], entry["size"], flush=True)
        bounded_get(entry["links"]["self"], target, entry["size"], entry["checksum"].split(":", 1)[1])
        manifest["files"].append({**entry, "sha256": filehash(target), "md5_verified": True})
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(DATA / "dataset_clean_sound.zip") as archive:
        listing = [{"name": x.filename, "size": x.file_size, "compressed": x.compress_size} for x in archive.infolist()]
        (DATA / "archive-listing.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
        for member in archive.namelist():
            if member.lower().endswith("metadata.csv"):
                content = archive.read(member)
                if len(content) > 1_000_000:
                    raise ValueError("Metadata exceeds cap")
                (DATA / "sound-metadata.csv").write_bytes(content)
                print(content[:5000].decode("utf-8-sig"), flush=True)
    print("Prepared", len(listing), "archive entries", flush=True)


def decode(archive, member):
    source = DATA / ("working" + Path(member).suffix)
    try:
        if archive.getinfo(member).file_size > 50_000_000:
            raise ValueError("Unexpected audio member size")
        content = archive.read(member)
        source.write_bytes(content)
        result = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(source),
                                 "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                                capture_output=True, check=True, timeout=60)
        waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
        if not 1600 <= len(waveform) <= 16000 * 180 or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid decoded audio: {member}")
        return waveform
    finally:
        source.unlink(missing_ok=True)


def impact_windows(audio):
    from scipy.signal import butter, find_peaks, sosfilt
    filtered = sosfilt(butter(3, 100, "highpass", fs=16000, output="sos"), audio)
    frame, hop = 160, 80
    padded = np.pad(filtered, (0, max(0, frame - len(filtered))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    energy = np.sqrt(np.mean(frames * frames, axis=1))
    threshold = max(float(np.quantile(energy, .25)) * 3, float(energy.max()) * .12)
    peaks, _ = find_peaks(energy, height=threshold, prominence=threshold * .5, distance=100)
    if len(peaks) == 0:
        peaks = np.array([int(energy.argmax())])
    peaks = np.sort(peaks[np.argsort(energy[peaks])[-18:]])
    windows = []
    for peak in peaks:
        center = int(peak * hop + frame // 2)
        start, end = center - 1600, center + 14400
        segment = audio[max(0, start):min(len(audio), end)]
        windows.append(np.pad(segment, (max(0, -start), max(0, end - len(audio)))))
    result = np.asarray(windows, dtype=np.float32)
    assert result.shape == (len(peaks), 16000)
    return result, (peaks * hop / 16000).tolist()


def acoustic_features(windows):
    import librosa
    values = []
    for window in windows:
        window = window - window.mean()
        power = np.abs(librosa.stft(window, n_fft=512, hop_length=160, win_length=400)) ** 2
        mel = librosa.feature.melspectrogram(S=power, sr=16000, n_fft=512, n_mels=40, fmin=50, fmax=7600)
        cepstra = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max), n_mfcc=20)
        centroid = librosa.feature.spectral_centroid(S=np.sqrt(power), sr=16000)
        bandwidth = librosa.feature.spectral_bandwidth(S=np.sqrt(power), sr=16000)
        rolloff = librosa.feature.spectral_rolloff(S=np.sqrt(power), sr=16000)
        flatness = librosa.feature.spectral_flatness(S=np.sqrt(power))
        zcr = librosa.feature.zero_crossing_rate(window, frame_length=400, hop_length=160)
        stats = np.vstack([centroid, bandwidth, rolloff, flatness])
        hz = librosa.fft_frequencies(sr=16000, n_fft=512)
        spectrum = power.sum(axis=1)
        bands = np.array([spectrum[(hz >= lo) & (hz < hi)].sum() for lo, hi in
                          [(50, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7600)]])
        bands /= max(bands.sum(), 1e-12)
        values.append(np.r_[cepstra.mean(1), cepstra.std(1), stats.mean(1), stats.std(1),
                            zcr.mean(), zcr.std(), bands])
    matrix = np.asarray(values)
    result = np.r_[matrix.mean(0), matrix.std(0)].astype(np.float32)
    assert result.shape == (112,) and np.isfinite(result).all()
    return result


def yamnet_features(model, windows):
    embeddings = []
    for window in windows:
        scores, features, _ = model(window)
        features = np.asarray(features)
        assert features.ndim == 2 and features.shape[1] == 1024
        embeddings.append(features.mean(axis=0))
    value = np.mean(embeddings, axis=0).astype(np.float32)
    assert value.shape == (1024,) and np.isfinite(value).all()
    return value


def scores(y, probability):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    prediction = np.asarray(probability).argmax(axis=1)
    return {"n": len(y), "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist()}


def fruit_scores(rows, y, probability):
    targets, probabilities = [], []
    for fruit, target in sorted({(r["fruit"], int(label)) for r, label in zip(rows, y)}):
        selected = np.array([r["fruit"] == fruit and label == target for r, label in zip(rows, y)])
        targets.append(target)
        probabilities.append(probability[selected].mean(axis=0))
    return scores(np.asarray(targets), np.asarray(probabilities))


def fitted_head(X, y, groups, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # Each fruit has equal total fitting weight, even when it has multiple retries/devices.
    counts = Counter(groups)
    weights = np.array([1 / counts[g] for g in groups])
    class_totals = np.array([weights[y == label].sum() for label in range(3)])
    weights *= weights.sum() / (3 * class_totals[y])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2500, random_state=SEED))
    model.fit(X, y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    return model


def nested_comparison(rows, features, target_key="label"):
    from sklearn.model_selection import StratifiedKFold
    selected_rows = [i for i, r in enumerate(rows) if r.get(target_key) in (0, 1, 2)]
    rows = [rows[i] for i in selected_rows]
    features = {name: matrix[selected_rows] for name, matrix in features.items()}
    groups = np.array([r["fruit"] for r in rows])
    y = np.array([r[target_key] for r in rows])
    fruits = np.array(sorted(set(groups)))
    stratify = y if target_key == "label" else np.array([r["stratification_label"] for r in rows])
    fruit_y = np.array([stratify[np.flatnonzero(groups == g)[0]] for g in fruits])
    assert all(len(set(stratify[groups == g])) == 1 for g in fruits)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities = {name: np.zeros((len(y), 3)) for name in ["majority", *features]}
    scored = np.zeros(len(rows), dtype=bool)
    fold_records, fitted = [], {}
    for fold, (train_fruit, test_fruit) in enumerate(outer.split(fruits, fruit_y)):
        training = np.flatnonzero(np.isin(groups, fruits[train_fruit]))
        testing = np.flatnonzero(np.isin(groups, fruits[test_fruit]))
        assert not set(groups[training]) & set(groups[testing])
        scored[testing] = True
        fruit_weights = np.array([1 / Counter(groups[training])[g] for g in groups[training]])
        major = int(np.bincount(y[training], weights=fruit_weights, minlength=3).argmax())
        probabilities["majority"][testing, major] = 1
        fold_record = {"fold": fold, "train_fruits": fruits[train_fruit].tolist(),
                       "test_fruits": fruits[test_fruit].tolist(), "train_rows": training.tolist(),
                       "test_rows": testing.tolist(), "models": {}}
        for name, matrix in features.items():
            inner_fruits = np.array(sorted(set(groups[training])))
            inner_y = np.array([stratify[np.flatnonzero(groups == g)[0]] for g in inner_fruits])
            tuning = []
            for C in (.01, .1, 1):
                inner_probability = np.zeros((len(rows), 3))
                inner_scored = np.zeros(len(rows), dtype=bool)
                for inner_train, inner_test in StratifiedKFold(3, shuffle=True, random_state=SEED + fold).split(inner_fruits, inner_y):
                    fit = training[np.isin(groups[training], inner_fruits[inner_train])]
                    validate = training[np.isin(groups[training], inner_fruits[inner_test])]
                    assert not set(groups[fit]) & set(groups[validate])
                    model = fitted_head(matrix[fit], y[fit], groups[fit], C)
                    inner_probability[validate] = model.predict_proba(matrix[validate])
                    inner_scored[validate] = True
                validation_rows = [r for i, r in enumerate(rows) if inner_scored[i]]
                metric = fruit_scores(validation_rows, y[inner_scored], inner_probability[inner_scored])["balanced_accuracy"]
                tuning.append({"C": C, "inner_fruit_balanced_accuracy": metric})
            best = max(tuning, key=lambda item: item["inner_fruit_balanced_accuracy"])
            model = fitted_head(matrix[training], y[training], groups[training], best["C"])
            probabilities[name][testing] = model.predict_proba(matrix[testing])
            fitted[name] = model
            fold_record["models"][name] = {"selected_C": best["C"], "tuning": tuning}
        fold_records.append(fold_record)
        print("Scored", target_key, "fold", fold, flush=True)
    output_rows = [r for i, r in enumerate(rows) if scored[i]]
    metrics = {}
    for name, probability in probabilities.items():
        probability = probability[scored]
        metrics[name] = {"recording": scores(y[scored], probability),
                         "fruit": fruit_scores(output_rows, y[scored], probability),
                         "by_device": {}}
        for device in sorted({r["device"] for r in output_rows}):
            mask = np.array([r["device"] == device for r in output_rows])
            metrics[name]["by_device"][device] = scores(y[scored][mask], probability[mask])
            device_rows = [row for row, keep in zip(output_rows, mask) if keep]
            metrics[name]["by_device"][device]["fruit"] = fruit_scores(device_rows, y[scored][mask], probability[mask])
    predictions = []
    for i in np.flatnonzero(scored):
        predictions.append({**rows[i], "target": int(y[i]), "predictions": {
            name: {"class": int(prob[i].argmax()), "probability": prob[i].tolist()}
            for name, prob in probabilities.items()}})
    fruit_predictions = []
    for fruit, target in sorted({(r["fruit"], int(label)) for r, label in zip(rows, y)}):
        mask = np.array([r["fruit"] == fruit and label == target for r, label in zip(rows, y)]) & scored
        if not mask.any():
            continue
        fruit_predictions.append({"fruit": fruit, "target": target, "recordings": int(mask.sum()),
            "predictions": {name: {"class": int(prob[mask].mean(axis=0).argmax()),
                "probability": prob[mask].mean(axis=0).tolist()} for name, prob in probabilities.items()}})
    return {"metrics": metrics, "folds": fold_records, "predictions": predictions,
            "fruit_predictions": fruit_predictions}, fitted


def benchmark_cpu(model, heads, audio):
    result = {}
    for context, waveform in (("single_impact_1s", impact_windows(audio)[0][0]), ("whole_recording", audio)):
        for name in ("mfcc_spectral", "yamnet", "hybrid"):
            durations = []
            for repeat in range(12):
                started = time.perf_counter()
                windows = waveform[None, :] if context == "single_impact_1s" else impact_windows(waveform)[0]
                if name == "mfcc_spectral":
                    feature = acoustic_features(windows)
                elif name == "yamnet":
                    feature = yamnet_features(model, windows)
                else:
                    feature = np.r_[acoustic_features(windows), yamnet_features(model, windows)]
                heads[name].predict_proba(feature[None, :])
                elapsed = (time.perf_counter() - started) * 1000
                if repeat >= 2:
                    durations.append(elapsed)
            result[f"{context}/{name}"] = {"p50_ms": float(np.median(durations)),
                "p95_ms": float(np.quantile(durations, .95)), "repetitions": len(durations),
                "warmups": 2, "audio_seconds": len(waveform) / 16000,
                "impact_windows": len(windows), "head_parameters": int(heads[name][-1].coef_.size + heads[name][-1].intercept_.size)}
    return result


def write_report(result):
    lines = ["# Durian knock-audio experiment", "", "## Evidence and sources", "",
        "Executed on desktop CPU, not an iPhone. Acoustic input only; no image/thermal/chemical features.",
        "Dataset: [Zenodo 18603796](https://doi.org/10.5281/zenodo.18603796), "
        "[primary API](https://zenodo.org/api/records/18603796), "
        "[README](https://zenodo.org/api/records/18603796/files/README.md/content). CC BY 4.0; "
        "attribute Mesa-Satina et al. (2026). Downloaded only audio, characteristics CSV, README and API metadata.", "",
        "Reproduce: `python experiments/durian_benchmark.py --prepare --run`. Verified source checksums and SHA-256 "
        "are in `artifacts/datasets/durian/manifest.json`. Model hashes, package versions, features, splits, tuning and "
        "predictions are retained in `artifacts/durian-*`.", "", "## Population and protocol", "",
        "```json", json.dumps({key: value for key, value in result["counts"].items() if key not in
            ("not_terminal_stage_members", "date_by_actual_status")}, indent=2), "```", "",
        "Full recording-level exclusions and date/label cross-tabs remain in results JSON.", "",
        "The physical fruit identifier is the numeric replicate suffix in characteristics Code / sound sample_num, "
        "not the recording filename, phone, timestamp, retry or ripeness code. Every observed recording of a fruit "
        "stays together. Five stratified fruit folds; three inner fruit folds tune C ∈ {0.01, 0.1, 1} "
        "using training-only per-fruit balanced accuracy. The same outer/inner seed and fruit protocol are used for "
        "each representation. Scaling and regularized multinomial logistic regression are fitted only to each "
        "training fold; class/fruit-balanced fit weights prevent repeats dominating.", "",
        "Primary experiment is iPhone-only: target is the separate `Actual_Ripening_Status` field, restricted to recording-stage labels "
        "matching the characteristics table's nominal `Ripeness`. This operational terminal-stage association "
        "is not a verified per-recording destructive-reference timestamp; biological accuracy remains conditional "
        "on that association. Earlier unripe audio is never assigned the fruit's terminal ripe/overripe status. "
        "Disease/missing statuses, ambiguous source-member associations and contradictory duplicate labels are excluded. "
        "A second iPhone-only task uses the sound CSV's per-recording `ripeness` stages across usable audio, including earlier sessions, "
        "as a nominal-label sensitivity analysis. Characteristic nominal and actual disagreements are audited "
        "separately (19 fruits); these are not corrected using model predictions.", "",
        "The sound CSV cross-joins multiple originals onto repeated output filenames. Each actual ZIP member "
        "is counted once; archive uncompressed bytes must match one unique source original/timestamp. Four "
        "Android members with equal-sized but different possible originals are excluded from both supervised "
        "tasks. Full candidate rows, matching origins, exclusions and audio hashes are retained in recordings JSON. "
        "This does not recover retries that the cleaned archive did not preserve.", "",
        "A byte-level SHA-256 audit found 105 contradictory Android groups (211 archive members): exactly identical "
        "audio bytes under two or three different stage filenames. All such members are excluded; no model is "
        "asked to distinguish identical recordings with contradictory labels. No exact duplicates cross fruit IDs. "
        "The cleaned files are therefore not 562 independent trustworthy labelled examples. See integrity-audit JSON.", "",
        "All audio is decoded by FFmpeg to 16 kHz mono PCM. A fixed 100 Hz high-pass RMS peak detector selects up to "
        "18 peaks at least 0.5 seconds apart; 1-second windows include 0.1 seconds before each selected peak. "
        "No supervised tap selection or test-dependent tuning. Some selected peaks may be handling/noise/speech: "
        "the dataset has no tap boundary annotations, so this is a limitation, not verified perfect segmentation.", "",
        "MFCC branch: 20 MFCC means/SD, centroid/bandwidth/rolloff/flatness mean/SD, zero-crossing mean/SD, "
        "six normalized spectral bands; mean/SD across windows = 112 floats. YAMNet: official local Google "
        "SavedModel, frozen 1,024D frame embeddings mean-pooled across windows, not its 521 sound labels. "
        "Hybrid concatenates the two (1,136D). No neural backbone fine-tuning. A fresh tiny CNN was not trained: "
        "the small, noisy-label fruit population does not justify adding a new high-variance learner to this "
        "bounded comparison; this does not establish that CNNs cannot improve.", "", "## Held-out results", "",
        "Class order in all confusion matrices: Unripe, Ripe, Overripe. Majority is selected on training fruit only. "
        "Per-fruit scores average recording probability vectors, not taps counted as independent fruits. "
        "For the longitudinal nominal-stage task, aggregation is fruit × true recording-stage (a fruit can have "
        "multiple targets at different times); these groups are never averaged across incompatible labels. "
        "All stages still share a single held-out fruit fold. Primary and sensitivity task populations differ, "
        "so their scores are not a head-to-head biological-label comparison.", "",
        "| Target / protocol | Model | Recording balanced accuracy | Fruit(/stage) balanced accuracy | Fruit(/stage) macro F1 |",
        "|---|---|---:|---:|---:|"]
    for protocol, entry in result["experiments"].items():
        for name, metric in entry["metrics"].items():
            lines.append(f"| {protocol} | {name} | {metric['recording']['balanced_accuracy']:.3f} | "
                         f"{metric['fruit']['balanced_accuracy']:.3f} | {metric['fruit']['macro_f1']:.3f} |")
    primary = result["experiments"]["actual_iphone_primary"]["metrics"]
    lines += ["", f"On the fixed 133-fruit primary task, MFCC/spectral balanced accuracy was "
        f"{primary['mfcc_spectral']['fruit']['balanced_accuracy']:.1%}, YAMNet "
        f"{primary['yamnet']['fruit']['balanced_accuracy']:.1%}, and hybrid "
        f"{primary['hybrid']['fruit']['balanced_accuracy']:.1%}, versus "
        f"{primary['majority']['fruit']['balanced_accuracy']:.1%} for majority. This small observed gap "
        "does not establish statistical superiority; the hybrid did not improve the primary comparison. "
        "The low-dimensional head is therefore a reasonable conservative edge starting point, conditional "
        "on new independent acquisition and reliable time-matched ground truth.", ""]
    lines += ["", "Cross-device three-class evaluation is **not estimable** after integrity exclusions: "
        "the Android primary actual-status subset contains only Unripe. Android nominal stage data have 68 "
        "Unripe, one Ripe and one Overripe; this does not support a reliable stratified training comparison. "
        "The pooled actual-status experiment is secondary and explicitly device/class-confounded. Its per-device "
        "scores below are descriptive, not evidence of Android three-class ripeness generalization. "
        "Different phone codecs/formats remain confounded with device identity.", "",
        "### Secondary pooled per-device recording scores", "", "| Model | Device | N | Balanced accuracy | Macro F1 |",
        "|---|---|---:|---:|---:|"]
    for name, metric in result["experiments"]["actual_pooled_secondary_device_confounded"]["metrics"].items():
        for device, value in metric["by_device"].items():
            lines.append(f"| {name} | {device} | {value['n']} | {value['balanced_accuracy']:.3f} | {value['macro_f1']:.3f} |")
    lines += ["", "## CPU inference proxy", "", f"Host: `{result['runtime']['platform']}` / `{result['runtime']['processor']}`. "
        "Two math/TF threads requested. Timings include feature extraction and a fitted head, but exclude decode, "
        "capture wait and model cold-load; whole-recording timings include peak detection. Single-impact timings "
        "start from an already isolated 1-second segment. Twelve calls, first two discarded. Not a phone latency, "
        "memory, thermal or energy benchmark.", "", "| Input / representation | Windows | p50 ms | p95 ms |", "|---|---:|---:|---:|"]
    for name, timing in result["cpu"].items():
        lines.append(f"| {name} | {timing['impact_windows']} | {timing['p50_ms']:.2f} | {timing['p95_ms']:.2f} |")
    lines += ["", "## Interpretation and limits", "",
        "This is one farm, one harvest season, controlled rubber-stick tapping at approximately five inches, "
        "iPhone 13 and Redmi Note 14; it is not iPhone 11 validation or arbitrary-fruit ripeness. "
        "Maturity stage, harvest batch, collection date/time, storage treatment, background and nominal classes "
        "can covary with ripeness. Date/class/maturity counts are preserved; group holding prevents same-fruit "
        "leakage but does not remove these acquisition confounds. A new farm/season/user/session and known "
        "reference measurement validation are required before causal or commercial ripeness claims.", "",
        "The README describes temporal tracking yet the characteristics table provides one Code/Replicate per "
        "fruit, not a validated longitudinal identity map across distinct numeric IDs. Numeric fruit identity "
        "is therefore the strongest released grouping evidence; undisclosed identity reuse cannot be excluded. "
        "Class probabilities are uncalibrated. Five-fold estimates are not a product acceptance threshold "
        "or evidence that the best observed row is universally best. Cross-device plus fruit holdout is stricter "
        "than testing a second phone on the same training fruit.", "",
        "No physical iPhone inference, Core ML conversion, energy, battery, peak RAM or unseen-farm "
        "generalization was measured. Model sizes are not runtime RAM. Raw data retain the source CC BY license; "
        "no other modality archives were downloaded.", ""]
    export_path = PREFIX.with_name("durian-tflite-heads.json")
    if export_path.exists():
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        lines += ["## Portable TensorFlow Lite numerical heads", "",
            "Reproduce: `python experiments/durian_benchmark.py --export-heads`. Five fixed-shape "
            "float32 builtin-only `.tflite` files preserve each primary outer fold's training-only chosen C. "
            "They include 112D feature scaling, linear logits and softmax (input `[1,112]`, output `[1,3]`). "
            "These are evaluation-fold heads, not a full-data production model. No runtime dependencies installed.", "",
            f"Across all 133 original held-out predictions: maximum probability absolute discrepancy "
            f"{exported['max_probability_absolute_difference']:.3g}; class-decision changes "
            f"{exported['decision_changes']}. Desktop TFLite head calls include input/output tensor copies, "
            "20 warm-ups plus 100 timed calls per fold, two CPU threads requested.", "",
            "| Fold | Bytes | Head CPU p50 ms | p95 ms | Max probability difference |",
            "|---|---:|---:|---:|---:|"]
        for fold in exported["folds"]:
            lines.append(f"| {fold['fold']} | {fold['size_bytes']} | {fold['desktop_cpu_tflite_head_ms_p50']:.5f} | "
                         f"{fold['desktop_cpu_tflite_head_ms_p95']:.5f} | {fold['float32_tflite_probability_max_absolute_difference']:.3g} |")
        lines += ["", "Only the numerical head is converted. Audio decoding, impact selection and MFCC/spectral "
            "frontend still run in the desktop reference implementation; frontend parity and full-pipeline "
            "mobile conversion are unmeasured. These microsecond head times exclude features/capture and are "
            "not end-to-end iPhone latency. No physical iPhone deployment was exercised.", ""]
    (ROOT / "docs" / "durian-report.md").write_text("\n".join(lines), encoding="utf-8")


def export_heads():
    """Export the five MFCC primary evaluation heads, never a newly test-tuned model."""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    result = json.loads(PREFIX.with_name("durian-results.json").read_text(encoding="utf-8"))
    recordings = json.loads(PREFIX.with_name("durian-recordings.json").read_text(encoding="utf-8"))
    primary_rows = [row for row in recordings if row["device"] == "iphone" and row["label"] is not None]
    indices = [i for i, row in enumerate(recordings) if row["device"] == "iphone" and row["label"] is not None]
    with np.load(PREFIX.with_name("durian-features.npz")) as features:
        matrix = features["mfcc_spectral"][indices]
    assert matrix.shape == (133, 112)
    y = np.array([row["label"] for row in primary_rows])
    groups = np.array([row["fruit"] for row in primary_rows])
    previous = result["experiments"]["actual_iphone_primary"]
    previous_probability = np.asarray([row["predictions"]["mfcc_spectral"]["probability"] for row in previous["predictions"]])
    exported_probability = np.zeros((len(matrix), 3))
    fold_results = []

    class NumericalHead(tf.Module):
        def __init__(self, head):
            super().__init__()
            self.mean = tf.constant(head[0].mean_.astype(np.float32))
            self.scale = tf.constant(head[0].scale_.astype(np.float32))
            self.weights = tf.constant(head[-1].coef_.T.astype(np.float32))
            self.bias = tf.constant(head[-1].intercept_.astype(np.float32))

        @tf.function(input_signature=[tf.TensorSpec([1, 112], tf.float32, name="mfcc_spectral_features")])
        def __call__(self, features):
            return {"probabilities": tf.nn.softmax(tf.matmul((features - self.mean) / self.scale, self.weights) + self.bias)}

    for fold in previous["folds"]:
        train, test = np.asarray(fold["train_rows"]), np.asarray(fold["test_rows"])
        assert not set(groups[train]) & set(groups[test])
        C = fold["models"]["mfcc_spectral"]["selected_C"]
        head = fitted_head(matrix[train], y[train], groups[train], C)
        sklearn_probability = head.predict_proba(matrix[test])
        replay_difference = float(np.max(np.abs(sklearn_probability - previous_probability[test])))
        assert replay_difference < 1e-10, "Reconstructed head differs from original OOF results"
        module = NumericalHead(head)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([module.__call__.get_concrete_function()], module)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        content = converter.convert()
        path = PREFIX.with_name(f"durian-mfcc-fold-{fold['fold']}.tflite")
        path.write_bytes(content)
        interpreter = tf.lite.Interpreter(model_content=content, num_threads=2)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
        assert input_info["shape"].tolist() == [1, 112] and output_info["shape"].tolist() == [1, 3]
        assert input_info["dtype"] == output_info["dtype"] == np.float32
        for i in test:
            interpreter.set_tensor(input_info["index"], matrix[i:i + 1])
            interpreter.invoke()
            exported_probability[i] = interpreter.get_tensor(output_info["index"])[0]
        timings = []
        for iteration in range(120):
            sample = matrix[test[iteration % len(test)]:test[iteration % len(test)] + 1]
            started = time.perf_counter()
            interpreter.set_tensor(input_info["index"], sample)
            interpreter.invoke()
            interpreter.get_tensor(output_info["index"])
            if iteration >= 20:
                timings.append((time.perf_counter() - started) * 1000)
        fold_results.append({"fold": fold["fold"], "selected_C": C, "model": str(path.relative_to(ROOT)),
            "size_bytes": len(content), "sha256": filehash(path), "heldout_count": len(test),
            "reconstructed_sklearn_oof_max_difference": replay_difference,
            "float32_tflite_probability_max_absolute_difference": float(np.max(np.abs(exported_probability[test] - sklearn_probability))),
            "decision_changes": int(np.sum(exported_probability[test].argmax(1) != sklearn_probability.argmax(1))),
            "desktop_cpu_tflite_head_ms_p50": float(np.median(timings)),
            "desktop_cpu_tflite_head_ms_p95": float(np.quantile(timings, .95)),
            "warmups": 20, "timed_inferences": 100,
            "heldout_predictions": [{"fruit": primary_rows[i]["fruit"], "target": int(y[i]),
                "probabilities": exported_probability[i].tolist()} for i in test]})
        print("Exported and exercised fold", fold["fold"], flush=True)
    export_result = {"format": "TensorFlow Lite float32 builtin-only", "feature_dimensions": 112,
        "input_shape": [1, 112], "output_shape": [1, 3], "classes": ["Unripe", "Ripe", "Overripe"],
        "includes": "StandardScaler normalization, multinomial linear head and softmax",
        "excludes": "Audio decoding, impact selection and MFCC/spectral frontend; no full mobile pipeline conversion",
        "hardware": "Desktop Windows CPU; not physical iPhone", "threads_requested": 2,
        "tensorflow": tf.__version__, "folds": fold_results,
        "sklearn_original_metrics": previous["metrics"]["mfcc_spectral"]["recording"],
        "tflite_oof_metrics": scores(y, exported_probability),
        "max_probability_absolute_difference": float(np.max(np.abs(exported_probability - previous_probability))),
        "decision_changes": int(np.sum(exported_probability.argmax(1) != previous_probability.argmax(1)))}
    PREFIX.with_name("durian-tflite-heads.json").write_text(json.dumps(export_result, indent=2), encoding="utf-8")
    write_report(result)
    print(json.dumps({key: value for key, value in export_result.items() if key != "folds"}, indent=2), flush=True)


def run():
    import librosa
    import sklearn
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    labels = {"Unripe": 0, "Ripe": 1, "Overripe": 2}
    characteristics = list(csv.DictReader((DATA / "durian_characteristics_cleaned.csv").open(encoding="utf-8-sig", newline="")))
    import re
    by_fruit = {int(re.fullmatch(r"R(\d+)", row["Replicate"]).group(1)): row for row in characteristics}
    assert len(by_fruit) == len(characteristics) == 189
    assert len({row["Code"] for row in characteristics}) == 189
    sound_metadata = list(csv.DictReader((DATA / "sound-metadata.csv").open(encoding="utf-8-sig", newline="")))
    rows = []
    with zipfile.ZipFile(DATA / "dataset_clean_sound.zip") as archive:
        by_basename = {Path(member).name: member for member in archive.namelist() if not member.endswith("/")}
        candidates = {}
        for meta in sound_metadata:
            for device in ("iphone", "android"):
                filename = meta[f"{device}_output"].strip()
                if filename:
                    candidates.setdefault((filename, device), []).append(meta)
        seen = set()
        for (filename, device), alternatives in sorted(candidates.items()):
            member = by_basename[filename]
            seen.add(member)
            size = archive.getinfo(member).file_size
            matched = [meta for meta in alternatives if int(meta[f"{device}_size_bytes"]) == size]
            origins = {(meta[f"{device}_original"], meta[f"{device}_date"]) for meta in matched}
            fruit = int(alternatives[0]["sample_num"])
            char = by_fruit[fruit]
            assert len({meta["sample_num"] for meta in alternatives}) == 1
            assert len({meta["ripeness"] for meta in alternatives}) == 1
            ambiguous = len(origins) != 1
            meta = matched[0] if matched else alternatives[0]
            terminal_match = meta["ripeness"] == char["Ripeness"]
            rows.append({"member": member, "fruit": fruit, "device": device,
                "timestamp": meta[f"{device}_date"] if not ambiguous else "",
                "source_sample_num": meta["sample_num"], "source_code": char["Code"], "source_replicate": char["Replicate"],
                "maturity": char["Maturity_Stage"], "source_class": char["Class"],
                "stratification_label": labels[char["Ripeness"]],
                "nominal": labels[meta["ripeness"]] if not ambiguous else None,
                "label": labels.get(char["Actual_Ripening_Status"]) if terminal_match and not ambiguous else None,
                "terminal_stage_match": terminal_match, "ambiguous_origin": ambiguous,
                "actual_status": char["Actual_Ripening_Status"], "metadata_ripeness": meta["ripeness"],
                "source_metadata_candidates": alternatives, "size_matching_origins": sorted(origins)})
        audio_members = {member for member in by_basename.values() if Path(member).suffix.lower() in (".m4a", ".wav")}
        if seen != audio_members:
            raise ValueError(f"Unmapped audio members: {audio_members - seen}")
        print("Mapped recordings", len(rows), flush=True)
        model_started = time.perf_counter()
        model = tf.saved_model.load(str(MODEL))
        model_load_seconds = time.perf_counter() - model_started
        cache = PREFIX.with_name("durian-features.npz")
        row_cache = PREFIX.with_name("durian-recordings.json")
        if cache.exists() and row_cache.exists():
            old_rows = json.loads(row_cache.read_text(encoding="utf-8"))
            assert len(rows) == len(old_rows)
            for row, old in zip(rows, old_rows):
                assert row["member"] == old["member"] and row["fruit"] == old["fruit"]
                row.update({key: old[key] for key in ("duration_seconds", "impact_centers_seconds", "mfcc_extraction_ms", "yamnet_extraction_ms", "source_audio_sha256")})
            with np.load(cache) as saved:
                classical, embeddings = saved["mfcc_spectral"], saved["yamnet"]
        else:
            classical, embeddings = [], []
            for i, row in enumerate(rows):
                audio = decode(archive, row["member"])
                windows, centers = impact_windows(audio)
                started = time.perf_counter()
                classical.append(acoustic_features(windows))
                acoustic_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                embeddings.append(yamnet_features(model, windows))
                row.update(duration_seconds=len(audio) / 16000, impact_centers_seconds=centers,
                           mfcc_extraction_ms=acoustic_ms, yamnet_extraction_ms=(time.perf_counter() - started) * 1000,
                           source_audio_sha256=hashlib.sha256(archive.read(row["member"])).hexdigest())
                if (i + 1) % 10 == 0:
                    print("Extracted", i + 1, "/", len(rows), flush=True)
            classical, embeddings = np.stack(classical), np.stack(embeddings)
            np.savez_compressed(cache, mfcc_spectral=classical, yamnet=embeddings)
            row_cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        probe_audio = decode(archive, rows[0]["member"])
    hashes = {}
    for row in rows:
        prior = hashes.setdefault(row["source_audio_sha256"], row["fruit"])
        if prior != row["fruit"]:
            raise ValueError("Identical source audio assigned to different fruits")
    same_hash = {}
    for row in rows:
        same_hash.setdefault(row["source_audio_sha256"], []).append(row)
    conflicts = [group for group in same_hash.values() if len({r["metadata_ripeness"] for r in group}) > 1]
    conflicting_members = {r["member"] for group in conflicts for r in group}
    audit = {"contradictory_duplicate_groups": [[{"member": r["member"], "fruit": r["fruit"],
        "stage": r["metadata_ripeness"], "sha256": r["source_audio_sha256"]} for r in group] for group in conflicts],
        "ambiguous_origin_members": [r["member"] for r in rows if r["ambiguous_origin"]]}
    PREFIX.with_name("durian-integrity-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for row in rows:
        row["contradictory_duplicate"] = row["member"] in conflicting_members
        if row["contradictory_duplicate"]:
            row["nominal"] = row["label"] = None
    row_cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    features = {"mfcc_spectral": classical, "yamnet": embeddings,
                "hybrid": np.concatenate([classical, embeddings], axis=1)}
    disagreements = [{"fruit": fruit, "Code": char["Code"], "Ripeness": char["Ripeness"],
                      "Actual_Ripening_Status": char["Actual_Ripening_Status"]}
                     for fruit, char in by_fruit.items() if char["Ripeness"] != char["Actual_Ripening_Status"]]
    counts = {"source_characteristic_fruits": len(by_fruit), "recordings": len(rows),
              "recorded_fruits": len({r["fruit"] for r in rows}),
              "unrecorded_fruit_ids": sorted(set(by_fruit) - {r["fruit"] for r in rows}),
              "nominal_fruit_counts": dict(Counter(char["Ripeness"] for char in characteristics)),
              "actual_fruit_counts": dict(Counter(char["Actual_Ripening_Status"] for char in characteristics)),
              "excluded_actual_ids": sorted(fruit for fruit, char in by_fruit.items() if char["Actual_Ripening_Status"] not in labels),
              "primary_recordings": sum(r["label"] is not None and r["device"] == "iphone" for r in rows),
              "primary_fruits": len({r["fruit"] for r in rows if r["label"] is not None and r["device"] == "iphone"}),
              "primary_excluded_fruit_ids": sorted(set(by_fruit) - {r["fruit"] for r in rows if r["label"] is not None and r["device"] == "iphone"}),
              "primary_date_by_class": dict(Counter(f"{r['timestamp'][:10]} / {r['actual_status']}" for r in rows if r["label"] is not None and r["device"] == "iphone")),
              "contradictory_duplicate_groups": len(conflicts), "contradictory_duplicate_members": len(conflicting_members),
              "usable_actual_by_device_class": dict(Counter(f"{r['device']} / {r['actual_status']}" for r in rows if r["label"] is not None)),
              "usable_nominal_by_device_class": dict(Counter(f"{r['device']} / {r['metadata_ripeness']}" for r in rows if r["nominal"] is not None)),
              "ambiguous_origin_members": [r["member"] for r in rows if r["ambiguous_origin"]],
              "not_terminal_stage_members": [r["member"] for r in rows if not r["terminal_stage_match"]],
              "nominal_by_actual_status": dict(Counter(f"{char['Ripeness']} / {char['Actual_Ripening_Status']}" for char in characteristics)),
              "class_by_actual_status": dict(Counter(f"{char['Class']} / {char['Actual_Ripening_Status']}" for char in characteristics)),
              "actual_recording_counts": dict(Counter(r["actual_status"] for r in rows)),
              "device_counts": dict(Counter(r["device"] for r in rows)),
              "date_by_actual_status": dict(Counter(f"{r['timestamp'][:10]} / {r['actual_status']}" for r in rows)),
              "maturity_by_actual_status": dict(Counter(f"{char['Maturity_Stage']} / {char['Actual_Ripening_Status']}" for char in characteristics)),
              "label_disagreements": len(disagreements),
              "retries_per_fruit_device": dict(Counter(str(n) for n in Counter((r["fruit"], r["device"]) for r in rows).values())),
              "audio_duration_seconds": {"min": min(r["duration_seconds"] for r in rows),
                  "median": float(np.median([r["duration_seconds"] for r in rows])), "max": max(r["duration_seconds"] for r in rows)}}
    iphone_indices = [i for i, r in enumerate(rows) if r["device"] == "iphone"]
    iphone_rows = [rows[i] for i in iphone_indices]
    iphone_features = {name: value[iphone_indices] for name, value in features.items()}
    experiments = {}
    experiments["actual_iphone_primary"], heads = nested_comparison(iphone_rows, iphone_features)
    import joblib
    joblib.dump({"description": "Last primary outer-fold heads used only for CPU timing; not full-data production models",
                 "seed": SEED, "heads": heads}, PREFIX.with_name("durian-timing-heads.joblib"))
    experiments["actual_pooled_secondary_device_confounded"], _ = nested_comparison(rows, features)
    experiments["nominal_iphone_stage_sensitivity"], _ = nested_comparison(iphone_rows, iphone_features, target_key="nominal")
    result = {"dataset": "Zenodo 18603796 durian knock audio", "seed": SEED, "classes": labels,
        "primary_target_column": "durian_characteristics_cleaned.csv:Actual_Ripening_Status",
        "primary_stage_matching_column": "durian_characteristics_cleaned.csv:Ripeness == sound-metadata.csv:ripeness",
        "sensitivity_target_column": "sound-metadata.csv:ripeness",
        "identity_columns": ["Replicate", "Code", "sample_num"],
        "counts": counts, "label_audit": disagreements, "experiments": experiments,
        "cross_device": {"status": "not_estimable", "reason": "After contradictory-duplicate exclusion, Android primary actual-status subset contains only Unripe; no meaningful three-class cross-device comparison or source training is possible.",
            "usable_actual_by_device_class": counts["usable_actual_by_device_class"]},
        "runtime": {"platform": platform.platform(), "processor": platform.processor(), "threads_requested": 2,
            "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
            "librosa": librosa.__version__, "tensorflow": tf.__version__, "model_load_seconds": model_load_seconds},
        "yamnet": {"path": str(MODEL.relative_to(ROOT)), "official_source": "https://tfhub.dev/google/yamnet/1",
            "files_sha256": {str(p.relative_to(MODEL)): filehash(p) for p in MODEL.rglob("*") if p.is_file()}},
        "cpu": benchmark_cpu(model, heads, probe_audio)}
    PREFIX.with_name("durian-results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_report(result)
    print(json.dumps({name: value["metrics"] for name, value in experiments.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--export-heads", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.export_heads:
        export_heads()
    if args.run:
        run()
