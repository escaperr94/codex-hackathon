"""DCASE2020 pretrained valve encoder -> DCASE2024 transfer (CPU only).

Run: python -m experiments.machine_task_benchmark
Uses pinned, digest-validated upstream source/assets in datasets/machine-task.
No 2024 test labels enter preprocessing, covariance fitting, or threshold choice.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ASSETS = ART / "datasets/machine-task"
REPORT = ART / "machine-task-benchmark.json"
CACHE = ART / "machine-task-embeddings.npz"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def metrics(y, scores, domains):
    result = {"pooled_auc": float(roc_auc_score(y, scores)),
              "pauc_max_fpr_0_1_standardized_sklearn": float(roc_auc_score(y, scores, max_fpr=.1))}
    for domain in ("source", "target"):
        # DCASE2024 compares domain-specific normals against ALL anomalies.
        mask = (y == 1) | (domains == domain)
        result[domain + "_auc_official_mask"] = float(roc_auc_score(y[mask], scores[mask]))
        within = domains == domain
        result[domain + "_auc_within_domain"] = float(roc_auc_score(y[within], scores[within]))
    return result


def split_audit(rows):
    train = {r["wav_sha256"] for r in rows if r["split"] == "train"}
    test = {r["wav_sha256"] for r in rows if r["split"] == "test"}
    assert not train & test
    return {"train_unique_wav_sha256": len(train), "test_unique_wav_sha256": len(test),
        "train_test_sha256_overlap": 0,
        "grouping": "official section00 split; source990+target10 normal train; no subject/session holdout",
        "pretraining_overlap": "not independently audited against original2020 waveform hashes"}

def main():
    started = time.perf_counter()
    manifest = json.loads((ASSETS / "download-manifest.json").read_text())
    for asset in manifest["assets"]:
        assert sha(ROOT / asset["path"]) == asset["sha256"], asset["path"]
    sys.path.insert(0, str(ASSETS / "upstream/src"))
    import tensorflow as tf
    import keras
    import iaad.models  # registers original custom Keras objects
    from iaad.features import Normalizer, build_mel_filterbank, load_wav, log_mel_fast, extract_patches, maybe_cmvn
    from iaad.inference import AnomalyDetector
    from iaad.scoring import clip_agg_vec, maha_score_from_params
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.mixed_precision.set_global_policy("float32")
    baseline = json.loads((ART / "dcase-benchmark.json").read_text())
    rows = sorted(baseline["split_records"], key=lambda r: (r["split"], r["filename"]))
    paths = [ART / "dcase/dev_valve_extracted/valve" / r["split"] / r["filename"] for r in rows]
    assert len(rows) == len(set(r["filename"] for r in rows)) == 1200
    assert all(sha(p) == r["wav_sha256"] for p, r in zip(paths, rows))
    train = np.array([r["split"] == "train" for r in rows])
    domains = np.array([r["domain"] for r in rows])
    assert train.sum() == 1000 and (~train).sum() == 200
    assert all(r["label"] == "normal" for r in rows if r["split"] == "train")
    modeldir = ASSETS / "model"
    model = keras.models.load_model(modeldir / "valve_emb_s0.keras", compile=False)
    norm = Normalizer.load(modeldir / "valve_norm_s0.npz")
    fb = build_mel_filterbank()
    calib = dict(np.load(modeldir / "valve_calib_s0.npz", allow_pickle=False))

    def embedding(path):
        from scipy.io import wavfile
        sr, wav_header = wavfile.read(path)
        assert sr == 16000 and wav_header.shape == (160000,)
        wav = load_wav(path)
        # Official inference drops incomplete trailing patches; no replacement DSP.
        patches = extract_patches(log_mel_fast(wav, filterbank=fb), 64)
        assert patches.shape == (9, 64, 128)
        x = np.clip(norm.transform(maybe_cmvn(patches)), -5., 5.)
        raw = model.encode(x, training=False).numpy().astype(np.float32)
        vec, _ = clip_agg_vec(raw, np.zeros(len(raw), dtype=np.int32))
        assert vec.shape == (1, 128) and np.isfinite(vec).all()
        return vec[0]

    signature = hashlib.sha256(json.dumps({"rows": rows, "manifest": manifest,
        "frontend": "official_inference_cmvn_then_saved_norm_clip5_mean_patch_embeddings_v1"}, sort_keys=True).encode()).hexdigest()
    with threadpool_limits(limits=2):
        if CACHE.exists():
            saved = np.load(CACHE, allow_pickle=False)
            assert str(saved["signature"]) == signature
            vectors = saved["embeddings"]
        else:
            vectors = []
            for i, path in enumerate(paths):
                vectors.append(embedding(path))
                if (i + 1) % 100 == 0:
                    print(f"Embedded {i + 1}/1200", flush=True)
            vectors = np.stack(vectors).astype(np.float32)
            np.savez_compressed(CACHE, signature=signature, embeddings=vectors,
                filenames=np.array([r["filename"] for r in rows]))
        assert vectors.shape == (1200, 128) and np.isfinite(vectors).all()
        testvec = vectors[~train]
        zero = maha_score_from_params(calib["emb_loc_pooled"], calib["emb_prec_pooled"], testvec)
        # Float64 covariance fitting avoids the observed precision sensitivity of float32.
        fit_started = time.perf_counter()
        refit = LedoitWolf().fit(vectors[train].astype(np.float64))
        adapted = refit.mahalanobis(testvec.astype(np.float64))
        fit_seconds = time.perf_counter() - fit_started
        assert np.isfinite(zero).all() and np.isfinite(adapted).all()
        assert np.std(zero) > 0 and np.std(adapted) > 0
        np.savez_compressed(ART / "machine-task-refit.npz", location=refit.location_, precision=refit.precision_,
                            covariance=refit.covariance_, shrinkage=refit.shrinkage_)
        # Compare real full inference against upstream API on two disjoint clips.
        det = AnomalyDetector.from_local(modeldir, machine_type="valve", seed=0)
        parity = []
        for ix in (0, int(np.flatnonzero(train)[0])):
            official = det.score_file(paths[ix], score="mahalanobis").clip_score
            ours = float(maha_score_from_params(calib["emb_loc_pooled"], calib["emb_prec_pooled"], vectors[ix:ix+1])[0])
            assert np.isclose(ours, official, rtol=1e-5, atol=1e-5), (ours, official)
            parity.append({"filename": rows[ix]["filename"], "official": official, "ours": ours, "absolute_error": abs(ours-official)})
        # Actual one-clip end-to-end CPU timings, not batched throughput or phone timing.
        timing = []
        for i in range(33):
            tick = time.perf_counter()
            e = embedding(paths[i % 200])
            s = refit.mahalanobis(e[None].astype(np.float64))
            assert np.isfinite(s).all()
            if i >= 3:
                timing.append((time.perf_counter() - tick) * 1000)
    # Only now use test labels for evaluation. This test was inspected in earlier experiments.
    y = np.array([int(r["label"] == "anomaly") for r in rows if r["split"] == "test"])
    test_domains = domains[~train]
    results = {}
    for name, scores in (("2020_zero_shot_pooled_calibration", zero), ("2024_normal_only_mahalanobis_refit", adapted)):
        results[name] = {**metrics(y, scores, test_domains), "scores": scores.tolist()}
    existing = json.loads((ART / "dcase-ablation.json").read_text())
    comparisons = {name: {k: r[k] for k in ("pooled_auc", "source_auc", "target_auc", "pauc_0_1")}
                   for name, r in existing["results"].items()}
    report = {"status": "completed_actual_cpu_inference", "model": "Jvachier/dcase2020-task2-anomaly-detection seed0",
        "model_card": "https://huggingface.co/Jvachier/dcase2020-task2-anomaly-detection",
        "upstream_commit": manifest["github_commit"], "hf_revision": manifest["hf_revision"],
        "download_total_bytes": sum(a["bytes"] for a in manifest["assets"]), "model_bytes": (modeldir / "valve_emb_s0.keras").stat().st_size,
        "parameter_count": model.count_params(), "dataset": baseline["dataset"],
        "split_counts": {"normal_train": int(train.sum()), "normal_train_source": int((train & (domains == "source")).sum()),
                         "normal_train_target": int((train & (domains == "target")).sum()), "test": 200},
        "frontend": {"implementation": "unaltered pinned iaad.features and iaad.models", "sample_rate": 16000,
            "fft": 512, "hop": 256, "mel_bins": 128, "fmin": 50, "fmax": 8000,
            "patch_frames": 64, "patches_per_clip": 9, "discarded_trailing_frames": 48,
            "normalization": "per-mel clip CMVN over retained patches, saved2020 per-bin zscore, clip+-5 per official inference",
            "pooling": "arithmetic mean of 9 L2-normalized 128D patch embeddings, no post-mean L2 normalization"},
        "precision": {"network_and_embedding": "float32", "zero_shot": "upstream float32 squared Mahalanobis output; stored calibration dtype preserved",
            "refit": "float32 embeddings cast float64 before LedoitWolf fit and squared Mahalanobis score", "shrinkage": float(refit.shrinkage_)},
        "normal_refit_fit_score_seconds": fit_seconds, "threshold": "None applied; original2020 percentile thresholds not treated as validated2024 thresholds",
        "test_status": "Previously inspected exploratory test; this is cross-year transfer, not reproduction of original2020 benchmark",
        "metrics_definition": "Pooled AUC over200; official source/target masks use normals from that domain and all anomalies; within-domain AUC also provided; pAUC sklearn standardized max_fpr0.1",
        "results": results, "existing_same_test_comparisons": comparisons, "official_api_parity": parity,
        "cpu_timing": {"host": platform.platform(), "processor": platform.processor(), "threads_max": 2, "warmup_clips": 3,
            "timed_clips": 30, "includes": "WAV read+official frontend+encoder+refit Mahalanobis; no10s capture waiting",
            "milliseconds": timing, "p50_ms": float(np.median(timing)), "p95_ms": float(np.percentile(timing,95))},
        "versions": {"python": sys.version, "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__},
        "script_sha256": sha(Path(__file__)), "embedding_sha256": sha(CACHE), "signature": signature,
        "split_records": rows, "wall_seconds": time.perf_counter()-started,
        "limitations": ["Only2024 valve section00; no other machine, section, site, or iPhone generalization measurement",
            "Frozen2020 encoder; normal-only2024 refit is adaptation, not unadapted zero-shot",
            "Test set previously inspected; no independent confirmation or tuning claim",
            "No threshold validation, calibrated anomaly probability, or specific fault diagnosis",
            "Model card Apache2.0;2024 dataset CC BY-NC-SA4.0, no commercial data-rights conclusion",
            "No CoreML conversion, device memory, battery, or iPhone latency measured"]}
    report["split_audit"] = split_audit(rows)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"results": {k: {a:b for a,b in v.items() if a != "scores"} for k,v in results.items()},
        "timing": report["cpu_timing"]["p50_ms"], "parameters": model.count_params()}, indent=2), flush=True)

def stgram_main():
    """Use released2020 feature extractor, never invent a2024 machine-ID label."""
    import importlib.util
    import torch
    import torch.nn.functional as F
    import torchaudio
    import soundfile as sf
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    manifest = json.loads((ASSETS / "stgram-download-manifest.json").read_text())
    for asset in manifest["assets"]:
        assert sha(ROOT / asset["path"]) == asset["sha256"]
    spec = importlib.util.spec_from_file_location("machine_stgram_original", ASSETS / "stgram-original/model.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    spec = importlib.util.spec_from_file_location("machine_stgram_dataset", ASSETS / "stgram-original/dataset.py")
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)
    # First release fixed in advance, not selected using2024 anomaly labels.
    checkpoint = ASSETS / "stgram-model/stgrammfn_checkpoint_best1.pth.tar"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["clf_state_dict"]
    arc = original.ArcMarginProduct(in_features=128, out_features=41, m=.7, s=30)
    model = original.STgramMFN(num_class=41, arcface=arc).eval()
    model.load_state_dict(state, strict=True)
    frontend = dataset.Generator(sr=16000, n_fft=1024, n_mels=128, win_length=1024, hop_length=512, power=2.)
    baseline = json.loads((ART / "dcase-benchmark.json").read_text())
    rows = sorted(baseline["split_records"], key=lambda r: (r["split"], r["filename"]))
    paths = [ART / "dcase/dev_valve_extracted/valve" / r["split"] / r["filename"] for r in rows]
    assert len(rows) == 1200 and all(sha(p) == r["wav_sha256"] for p,r in zip(paths,rows))
    train = np.array([r["split"] == "train" for r in rows])
    domains = np.array([r["domain"] for r in rows])
    assert train.sum() == 1000 and all(r["label"] == "normal" for r in rows if r["split"] == "train")

    def embedding(path):
        wav, sr = sf.read(path, dtype="float32")
        assert sr == 16000 and wav.shape == (160000,)
        x = torch.from_numpy(wav)
        mel = frontend(x)
        assert mel.shape == (128, 313)
        # Unmodified original encoder; bypass only the2020 ID-classification heads.
        with torch.inference_mode():
            temporal = model.tgramnet(x[None, None]).unsqueeze(1)
            h = torch.cat((mel[None, None], temporal), dim=1)
            mobile = model.mobilefacenet
            for layer in (mobile.conv1, mobile.dw_conv1, mobile.blocks, mobile.conv2, mobile.linear7, mobile.linear1):
                h = layer(h)
            raw = h.reshape(1, 128)
            vec = F.normalize(raw, dim=1).numpy()[0]
        assert np.isfinite(vec).all()
        return vec

    cache = ART / "machine-task-stgram-embeddings.npz"
    signature = hashlib.sha256(json.dumps({"rows":rows,"manifest":manifest,"frontend":"official_Generator_sf_pcm_identical_noresample_l2feature_v1"},sort_keys=True).encode()).hexdigest()
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        if cache.exists():
            saved = np.load(cache, allow_pickle=False)
            assert str(saved["signature"]) == signature
            vectors = saved["embeddings"]
        else:
            vectors = []
            for i,path in enumerate(paths):
                vectors.append(embedding(path))
                if (i+1) % 100 == 0:
                    print(f"STgram embedded {i+1}/1200", flush=True)
            vectors = np.stack(vectors)
            np.savez_compressed(cache, signature=signature, embeddings=vectors, filenames=np.array([r["filename"] for r in rows]))
        assert vectors.shape == (1200,128) and np.isfinite(vectors).all()
        refit = LedoitWolf().fit(vectors[train].astype(np.float64))
        scores = refit.mahalanobis(vectors[~train].astype(np.float64))
        assert np.isfinite(scores).all() and np.std(scores)>0
        np.savez_compressed(ART / "machine-task-stgram-refit.npz", location=refit.location_, precision=refit.precision_, shrinkage=refit.shrinkage_)
        # Self-check equivalence to original forward; dummy ID affects logits only.
        wav, sr = sf.read(paths[0], dtype="float32")
        x = torch.from_numpy(wav)
        with torch.inference_mode():
            _, raw = model(x[None,None], frontend(x)[None,None], torch.tensor([0]))
            official_vec = F.normalize(raw,dim=1).numpy()[0]
        error = float(np.max(np.abs(vectors[0]-official_vec)))
        assert error < 1e-6
        timings = []
        for i in range(33):
            tick = time.perf_counter()
            e = embedding(paths[i % 200])
            assert np.isfinite(refit.mahalanobis(e[None].astype(np.float64))).all()
            if i>=3:
                timings.append((time.perf_counter()-tick)*1000)
    y = np.array([int(r["label"]=="anomaly") for r in rows if r["split"]=="test"])
    result = metrics(y,scores,domains[~train])
    report = {"status":"completed_actual_cpu_inference","model":"STgram-MFN checkpoint_best1; frozen2020 41-ID encoder, new2024 normal-only scorer",
        "model_source":"https://github.com/liuyoude/STgram-MFN", "checkpoint_source":"https://zenodo.org/records/7194640",
        "checkpoint_sha256":sha(checkpoint),"checkpoint_bytes":checkpoint.stat().st_size,"compatible_source_commit":manifest["checkpoint_compatible_commit"],
        "current_main_incompatible":"Current main has6 bottlenecks versus15 in released checkpoint; used original historical model with strict all-key loading",
        "parameter_count":sum(p.numel() for p in model.parameters()),"dataset":baseline["dataset"],"split_records":rows,
        "train_count":1000,"test_count":200,"signature":signature,"embedding_sha256":sha(cache),"script_sha256":sha(Path(__file__)),
        "frontend":{"sample_rate":16000,"samples":160000,"fft":1024,"hop":512,"mel_bins":128,"mel_frames":313,
            "implementation":"unaltered original dataset.Generator torchaudio MelSpectrogram+AmplitudeToDB(power), center=True defaults; soundfile float32 PCM matches librosa load at native16k mono",
            "normalization":"No waveform/CMVN zscore; L2-normalize128D pre-classifier feature for normal-only covariance"},
        "precision":"float32 network/features; float64 LedoitWolf fit+squared Mahalanobis scoring","shrinkage":float(refit.shrinkage_),
        "official_forward_embedding_max_abs_error":error,"metrics":result,"scores":scores.tolist(),
        "cpu_timing":{"host":platform.platform(),"threads_max":2,"warmups":3,"timed_clips":30,"p50_ms":float(np.median(timings)),"p95_ms":float(np.percentile(timings,95)),"milliseconds":timings,
            "includes":"WAV read+official frontend+encoder+Mahalanobis; excludes10s capture wait"},
        "versions":{"torch":torch.__version__,"torchaudio":torchaudio.__version__,"numpy":np.__version__},"wall_seconds":time.perf_counter()-started,
        "limitations":["Transfer experiment, not original2020 per-ID cross-entropy anomaly benchmark;2020 IDs do not map to2024",
            "No meaningful zero-shot2024 class-score threshold; only normal-only2024 scorer evaluated","Only valve section00; previously inspected200-test exploratory set",
            "Zenodo checkpoint record metadata CC BY4.0; repository code license unresolved: no LICENSE found in inspected tree",
            "No iPhone/CoreML export, memory, energy, or latency measured; desktop CPU only"]}
    report["split_audit"] = split_audit(rows)
    (ART / "machine-task-stgram.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"STgram_metrics":result,"cpu_p50_ms":report["cpu_timing"]["p50_ms"],"parameter_count":report["parameter_count"]},indent=2),flush=True)

if __name__ == "__main__":
    if "--stgram" in sys.argv:
        stgram_main()
    else:
        main()
