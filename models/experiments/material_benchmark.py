"""Bounded YCB robot-impact material experiment; run from repository root.

Uses only local verified audio; downloading is deliberately a separate operation.
All windows from one physical object stay together. No iPhone claims are made.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly, stft
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/datasets/material"
ART = ROOT / "artifacts"
MODEL = ART / "tfhub/9616fd04ec2360621642ef9455b84f4b668e219e"
CS = (0.01, 0.1, 1.0, 10.0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def waveform(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    assert len(audio) and np.isfinite(audio).all() and rate > 0
    mono = audio.mean(axis=1)
    divisor = np.gcd(rate, 16000)
    mono = resample_poly(mono, 16000 // divisor, rate // divisor).astype(np.float32)
    # One deterministic transient context per source recording, not random windows.
    frame = 320
    energy = np.array([np.dot(x, x) for x in np.array_split(mono[:len(mono) // frame * frame], len(mono) // frame)])
    center = int(np.argmax(energy)) * frame + frame // 2
    start = max(0, min(center - 8000, len(mono) - 16000))
    x = np.pad(mono[start:start + 16000], (0, max(0, start + 16000 - len(mono))))
    assert x.shape == (16000,) and np.max(np.abs(x)) <= 1.0
    return x, start / 16000


def acoustic(x):
    _, _, z = stft(x, fs=16000, nperseg=400, noverlap=240, nfft=512, boundary=None, padded=False)
    hz = np.fft.rfftfreq(512, 1 / 16000)
    edges = 700 * (10 ** (np.linspace(2595 * np.log10(1 + 80 / 700), 2595 * np.log10(1 + 7600 / 700), 66) / 2595) - 1)
    filters = np.maximum(0, np.minimum((hz[None] - edges[:-2, None]) / np.diff(edges)[:-1, None], (edges[2:, None] - hz[None]) / np.diff(edges)[1:, None]))
    assert np.all(filters.sum(1) > 0)
    mel = np.log(np.maximum(np.abs(z.T) ** 2 @ filters.T, 1e-10))
    cep = dct(mel, type=2, axis=1, norm="ortho")[:, :13]
    v = np.r_[mel.mean(0), mel.std(0), cep.mean(0), cep.std(0)].astype(np.float32)
    assert v.shape == (154,) and np.isfinite(v).all()
    return v


def metric(y, pred, labels):
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}


def fit(x, y, c):
    return make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight="balanced", dual="auto", max_iter=20000, random_state=42)).fit(x, y)


def object_predictions(model, x, rows, indices):
    scores = model.decision_function(x[indices])
    groups = sorted({rows[i]["object"] for i in indices})
    truth, pred = [], []
    for group in groups:
        local = [k for k, i in enumerate(indices) if rows[i]["object"] == group]
        truth.append(rows[indices[local[0]]]["label"])
        pred.append(str(model.classes_[np.argmax(scores[local].mean(0))]))
    return groups, truth, pred


def main():
    begin = time.perf_counter()
    rows = json.loads((DATA / "manifest.json").read_text())
    assert len(rows) == 131 and sum(r["bytes"] for r in rows) < 1_000_000_000
    labels = sorted({r["label"] for r in rows})
    objects = {label: sorted({r["object"] for r in rows if r["label"] == label}) for label in labels}
    assert len(labels) == 9 and all(len(v) == 3 for v in objects.values())
    assert len({r["sha256"] for r in rows}) == len(rows), "Exact audio duplicate"
    for row in rows:
        p = DATA / row["local_path"]
        assert p.stat().st_size == row["bytes"] and digest(p) == row["sha256"]
    for label in labels:
        for rank, group in enumerate(objects[label]):
            for row in rows:
                if row["object"] == group:
                    row["object_rank"] = rank
    weights = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(MODEL.rglob("*")) if p.is_file()]
    assert any(p["path"].endswith("saved_model.pb") for p in weights)
    key = hashlib.sha256(json.dumps([rows, weights, "peak20ms-centered-1s-v1-154features"], sort_keys=True).encode()).hexdigest()
    cache = ART / "material-features.npz"
    extract_times = []
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    loaded = time.perf_counter()
    encoder = tf.saved_model.load(str(MODEL))
    load_seconds = time.perf_counter() - loaded
    if cache.exists():
        with np.load(cache, allow_pickle=False) as values:
            assert str(values["fingerprint"]) == key
            features = {"mfcc_logmel": values["mfcc_logmel"], "yamnet": values["yamnet"]}
            starts = values["starts"].tolist()
    else:
        conventional, embeddings, starts = [], [], []
        for row in rows:
            started = time.perf_counter()
            x, offset = waveform(DATA / row["local_path"])
            conventional.append(acoustic(x))
            score, embedding, mel = encoder(x)
            embedding = embedding.numpy()
            assert embedding.ndim == 2 and embedding.shape[1] == 1024 and np.isfinite(embedding).all()
            embeddings.append(embedding.mean(0))
            starts.append(offset)
            extract_times.append(time.perf_counter() - started)
        features = {"mfcc_logmel": np.stack(conventional), "yamnet": np.stack(embeddings)}
        np.savez_compressed(cache, fingerprint=key, starts=starts, **features)
    y = np.array([r["label"] for r in rows])
    ranks = np.array([r["object_rank"] for r in rows])
    results, heads = {}, {}
    with threadpool_limits(limits=2):
        for name, matrix in features.items():
            predictions = np.full(len(rows), "", dtype=object)
            object_rows, folds = [], []
            for fold in range(3):
                train = np.flatnonzero(ranks != fold)
                test = np.flatnonzero(ranks == fold)
                assert not ({rows[i]["object"] for i in train} & {rows[i]["object"] for i in test})
                assert set(y[train]) == set(y[test]) == set(labels)
                inner_scores = []
                for c in CS:
                    scores = []
                    for inner in sorted(set(ranks[train])):
                        tr = train[ranks[train] != inner]
                        va = train[ranks[train] == inner]
                        model = fit(matrix[tr], y[tr], c)
                        _, truth, pred = object_predictions(model, matrix, rows, va)
                        scores.append(f1_score(truth, pred, labels=labels, average="macro", zero_division=0))
                    inner_scores.append(float(np.mean(scores)))
                selected = CS[int(np.argmax(inner_scores))]
                started = time.perf_counter()
                model = fit(matrix[train], y[train], selected)
                predictions[test] = model.predict(matrix[test])
                groups, truth, pred = object_predictions(model, matrix, rows, test)
                object_rows.extend({"object":g,"truth":t,"prediction":p,"outer_fold":fold} for g,t,p in zip(groups,truth,pred))
                folds.append({"outer_fold":fold,"train_objects":sorted({rows[i]["object"] for i in train}),"test_objects":groups,
                              "train_recordings":len(train),"test_recordings":len(test),"selected_C":selected,
                              "inner_C":list(CS),"inner_object_macro_f1":inner_scores,"fit_predict_seconds":time.perf_counter()-started})
                heads[f"{name}_fold{fold}"] = model
            assert all(predictions != "") and len(object_rows) == 27
            results[name] = {"dimensions":matrix.shape[1], "recording_metrics":metric(y,predictions,labels),
                             "object_metrics":metric([r["truth"] for r in object_rows],[r["prediction"] for r in object_rows],labels),
                             "folds":folds,"object_predictions":object_rows,
                             "recording_predictions":[{"path":r["local_path"],"object":r["object"],"truth":r["label"],"prediction":p,"outer_fold":r["object_rank"],"crop_start_seconds":o} for r,p,o in zip(rows,predictions,starts)]}
    head_path = ART / "material-heads.joblib"
    joblib.dump(heads, head_path)
    x, _ = waveform(DATA / rows[0]["local_path"])
    latency = {}
    for name in features:
        times = []
        model = heads[f"{name}_fold0"]
        for rep in range(40):
            start = time.perf_counter()
            f = acoustic(x) if name == "mfcc_logmel" else encoder(x)[1].numpy().mean(0)
            model.predict(f[None])
            elapsed = (time.perf_counter()-start)*1000
            if rep >= 10:
                times.append(elapsed)
        latency[name] = {"p50_ms":float(np.median(times)),"p95_ms":float(np.quantile(times,.95)),"warmups":10,"timed":30,
                         "input_seconds":1,"includes":"feature or frozen encoder + standardization + linear head; excludes file decode/resample/crop and capture wait"}
    peak = None
    try:
        import psutil
        mem = psutil.Process().memory_info()
        peak = {"peak_process_working_set_bytes":getattr(mem,"peak_wset",None),"current_rss_bytes":mem.rss,"scope":"whole Python process, both models, TensorFlow loaded; not per-model or iPhone RAM"}
    except ImportError:
        pass
    report = {"dataset":"YCB robot impact subset (bj5w8)","status":"complete","script_sha256":digest(__file__),
              "manifest_sha256":digest(DATA/"manifest.json"),"feature_sha256":digest(cache),"feature_fingerprint":key,
              "head_sha256":digest(head_path),"head_artifact_bytes":head_path.stat().st_size,"model_files":weights,
              "counts":{"recordings":len(rows),"objects":27,"classes":len(labels),"audio_bytes_selected":sum(r["bytes"] for r in rows)},
              "labels":labels,"objects_by_label":objects,
              "protocol":{"outer":"Three fixed object-rank folds; one unseen object of every class per fold. All retained complete recordings per object. No recording/window split.",
                          "inner":"Two object-rank folds within outer training only; choose C by mean object macro-F1; ties choose first C. Separate choices for each representation.",
                          "input":"Stereo averaged, polyphase resampled 44.1kHz->16kHz, one 1-second window centered on loudest 20ms energy block per recording; no amplitude normalization. Mean frozen YAMNet frames.",
                          "aggregation":"Mean uncalibrated linear SVM decision scores per object, then argmax; no calibrated probability claim.",
                          "subset":"Nine classes with three available distinct object IDs each; vertical pokes only, five clips/object initially. Four files excluded because two exact audio hashes were assigned to both small-marker and large-clamp object folders; 131 recordings retained. Alphabetical first three eligible IDs except steel uses skillet/padlock/wrench to avoid mixed-surface utensils, and pitcher lid omitted to avoid same-vessel base/lid leakage.",
                          "ground_truth":"Released material-suffixed object folders; suffix verified against surface labels and ycb_audio.xlsx. Masterchef can takes recorded aluminium-surface suffix despite secondary paper in table. No invented independent composition test."},
              "results":results,"desktop_cpu_latency":latency,"memory":peak,
              "runtime":{"platform":platform.platform(),"tensorflow":tf.__version__,"threads_max":2,"model_load_seconds":load_seconds,
                         "feature_extraction_seconds":sum(extract_times),"total_seconds":time.perf_counter()-begin},
              "limitations":["Small controlled robot/gripper airborne recordings, not phone material identification or tile bonding.",
                             "Each source clip contains multiple pokes; one peak-centered excerpt is scored. No claim to reproduce paper segmentation or full-dataset results.",
                             "Only three objects per class, nested tuning has one training object per class. Split uncertainty is large; objects differ in shape/content as well as material.",
                             "Soft-plastic fruit are YCB manufactured replicas, not real fruit ripeness.",
                             "Source node_license null: local research only; no redistribution or commercial license inferred.",
                             "No iPhone/Core ML conversion, latency, energy, or on-device memory measured."]}
    (ART/"material-benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"results":{k:{"recording":v["recording_metrics"],"object":v["object_metrics"]} for k,v in results.items()},"latency":latency,"runtime":report["runtime"]},indent=2))


def realimpact():
    """Score predeclared first30 complete deconvolved rows per physical object."""
    begin = time.perf_counter()
    source = DATA / "realimpact-prefix"
    manifest = json.loads((source / "manifest.json").read_text())
    evidence = json.loads((source / "material-label-evidence.json").read_text())
    assert len(manifest) == 9
    labels = sorted({m["label"] for m in manifest})
    groups = {label: sorted(m["object"] for m in manifest if m["label"] == label) for label in labels}
    assert len(labels) == 3 and all(len(v) == 3 for v in groups.values())
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    started = time.perf_counter()
    encoder = tf.saved_model.load(str(MODEL))
    load_seconds = time.perf_counter() - started
    rows, classic, embeddings = [], [], []
    started = time.perf_counter()
    for meta in manifest:
        name, label = meta["object"], meta["label"]
        assert evidence["mapping"][name]["material"] == label
        path = source / name / "audio-first30.npy"
        assert digest(path) == meta["audio_npy_sha256"]
        sounds = np.load(path, allow_pickle=False)
        assert sounds.shape == (30, meta["full_shape"][1]) and np.isfinite(sounds).all()
        for index, sound in enumerate(sounds):
            signal = resample_poly(sound, 1, 3).astype(np.float32)
            energy = np.array([np.dot(x, x) for x in np.array_split(signal[:len(signal)//320*320], len(signal)//320)])
            center = int(np.argmax(energy))*320+160
            start = max(0, min(center-8000, len(signal)-16000))
            x = signal[start:start+16000].copy()
            # Deconvolution is not PCM amplitude: fixed per-example peak scaling before both encoders.
            scale = float(np.max(np.abs(x)))
            assert scale > 0 and x.shape == (16000,)
            x *= 0.95 / scale
            classic.append(acoustic(x))
            emb = encoder(x)[1].numpy()
            assert emb.shape[1] == 1024 and np.isfinite(emb).all()
            embeddings.append(emb.mean(0))
            rows.append({"object":name,"label":label,"object_rank":groups[label].index(name),"source_row":index,
                         "crop_start_seconds":start/16000,"pre_normalization_peak":scale,
                         "vertexID":meta["annotations"]["vertexID"]["first30"][index],
                         "micID":meta["annotations"]["micID"]["first30"][index],
                         "angle":meta["annotations"]["angle"]["first30"][index],
                         "distance":meta["annotations"]["distance"]["first30"][index]})
    extraction_seconds = time.perf_counter()-started
    features = {"mfcc_logmel":np.stack(classic),"yamnet":np.stack(embeddings)}
    feature_path = ART / "material-realimpact-features.npz"
    np.savez_compressed(feature_path, **features)
    y = np.array([r["label"] for r in rows])
    ranks = np.array([r["object_rank"] for r in rows])
    results, heads = {}, {}
    with threadpool_limits(limits=2):
        for name, matrix in features.items():
            predictions = np.full(len(rows), "", dtype=object)
            object_rows, folds = [], []
            for fold in range(3):
                train, test = np.flatnonzero(ranks != fold), np.flatnonzero(ranks == fold)
                assert not ({rows[i]["object"] for i in train} & {rows[i]["object"] for i in test})
                scores = []
                for c in CS:
                    local = []
                    for inner in sorted(set(ranks[train])):
                        tr, va = train[ranks[train] != inner], train[ranks[train] == inner]
                        model = fit(matrix[tr], y[tr], c)
                        _, truth, pred = object_predictions(model, matrix, rows, va)
                        local.append(f1_score(truth,pred,labels=labels,average="macro",zero_division=0))
                    scores.append(float(np.mean(local)))
                c = CS[int(np.argmax(scores))]
                model = fit(matrix[train],y[train],c)
                predictions[test] = model.predict(matrix[test])
                names, truth, pred = object_predictions(model,matrix,rows,test)
                object_rows.extend({"object":g,"truth":t,"prediction":p,"outer_fold":fold} for g,t,p in zip(names,truth,pred))
                folds.append({"outer_fold":fold,"selected_C":c,"inner_C":list(CS),"inner_object_macro_f1":scores,
                              "train_objects":sorted({rows[i]["object"] for i in train}),"test_objects":names})
                heads[f"{name}_fold{fold}"] = model
            assert len(object_rows) == 9 and all(predictions != "")
            results[name] = {"recording_metrics":metric(y,predictions,labels),
                             "object_metrics":metric([r["truth"] for r in object_rows],[r["prediction"] for r in object_rows],labels),
                             "object_predictions":object_rows,"folds":folds,
                             "recording_predictions":[dict(r,prediction=p) for r,p in zip(rows,predictions)]}
    head_path = ART / "material-realimpact-heads.joblib"
    joblib.dump(heads,head_path)
    latency = {}
    for name in features:
        elapsed = []
        for repeat in range(40):
            started = time.perf_counter()
            vector = acoustic(x) if name == "mfcc_logmel" else encoder(x)[1].numpy().mean(0)
            heads[f"{name}_fold0"].predict(vector[None])
            if repeat >= 10:
                elapsed.append((time.perf_counter()-started)*1000)
        latency[name] = {"p50_ms":float(np.median(elapsed)),"p95_ms":float(np.quantile(elapsed,.95)),"warmups":10,"timed":30,"scope":"preprocessed 1s waveform -> feature/encoder + linear head, excludes decode/resample/crop and capture"}
    import psutil
    memory = psutil.Process().memory_info()
    report = {"dataset":"RealImpact bounded first30 rows/object","status":"complete","classes":labels,
              "counts":{"objects":9,"recordings":len(rows),"compressed_audio_prefix_bytes":sum(m["compressed_bytes_consumed"] for m in manifest)},
              "material_evidence":evidence,"source_manifest_sha256":digest(source/"manifest.json"),
              "script_sha256":digest(__file__),"feature_sha256":digest(feature_path),"heads_sha256":digest(head_path),
              "protocol":"Three fixed object-rank outer folds, one unseen object/class. Two inner object-rank folds select C from .01/.1/1/10 using object macro-F1, ties first. Scaling fitted within each training fold. First30 full rows/object fixed before models; no random recording split.",
              "preprocessing":"48kHz force-deconvolved float32 -> polyphase16kHz -> 1s loudest20ms-centered crop -> 0.95 peak normalization identically for both encoders. MFCC/logmel154D or frozen official YAMNet mean1024D.",
              "results":results,"desktop_cpu_latency":latency,
              "runtime":{"threads_max":2,"model_load_seconds":load_seconds,"feature_extraction_seconds":extraction_seconds,"total_seconds":time.perf_counter()-begin,"peak_process_working_set_bytes":getattr(memory,"peak_wset",None)},
              "limitations":["Deterministic first30 rows represent nearby microphones/positions and mostly one source vertex, not independent diverse impacts. Whole physical objects are held out.","Plastic bowls are same product family at different sizes; material and shape remain confounded.","Three objects/class and nine scored objects give high uncertainty; no universal or phone accuracy claim.","Force-deconvolved preprocessed impulse responses are not raw phone tapping audio; gain calibration unavailable, so per-clip normalization applied.","Partial DEFLATE decompression verified shape/finite complete rows and hashed prefixes; full-member/archive CRC or complete-file SHA256 NOT verified. Small annotation members have CRC verified.","No iPhone or mobile conversion measured; source dataset redistribution permission requires review."]}
    (ART/"material-realimpact-benchmark.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"metrics":{k:{"recording":v["recording_metrics"],"object":v["object_metrics"]} for k,v in results.items()},"latency":latency,"runtime":report["runtime"]},indent=2))


if __name__ == "__main__":
    import sys
    realimpact() if "--realimpact" in sys.argv else main()
