"""Reproducible DCASE 2024 Task 2 valve/YAMNet benchmark.

Downloads are intentionally not performed here. Run from repository root after
placing the official archive at artifacts/dcase/dev_valve.zip. The detector is
fit only on source-domain normal training clips; test labels are used only for
reported metrics. CPU thread limits prevent TensorFlow oversubscription.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import soundfile as sf
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "dcase"
ARCHIVE = ART / "dev_valve.zip"
EXTRACT = ART / "dev_valve_extracted"
REPORT = ROOT / "artifacts" / "dcase-benchmark.json"
EMBEDDINGS = ART / "valve_yamnet_embeddings.npz"
URL = "https://zenodo.org/api/records/10902294/files/dev_valve.zip/content"
RECORD = "https://zenodo.org/records/10902294"
EXPECTED_MD5 = "d03786b9502275226c6879f37a6cfe31"


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_clip(path: Path) -> dict[str, str | int]:
    # section_00_source_train_normal_0001_noAttributes.wav
    name = path.stem
    fields = name.split("_")
    if len(fields) < 6 or fields[0] != "section":
        raise ValueError(f"Unexpected DCASE filename: {path.name}")
    return {
        "path": str(path),
        "machine": path.parent.parent.name if path.parent.name in {"train", "test"} else "valve",
        "section": fields[1],
        "domain": fields[2],
        "split": fields[3],
        "label": fields[4],
        "index": int(fields[5]),
        "filename": path.name,
    }


def manifest_fingerprint(rows: list[dict]) -> str:
    payload = "\n".join(f"{row['filename']}\t{row['wav_sha256']}" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def discover() -> list[dict]:
    files = sorted(EXTRACT.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No wav files below {EXTRACT}; extract {ARCHIVE} first")
    rows = []
    for path in files:
        row = parse_clip(path)
        info = sf.info(path)
        row.update(sample_rate=int(info.samplerate), frames=int(info.frames),
                   seconds=float(info.frames / info.samplerate), wav_sha256=checksum(path))
        rows.append(row)
    return rows


def extract_archive() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Missing {ARCHIVE}")
    if checksum(ARCHIVE, "md5") != EXPECTED_MD5:
        raise ValueError("archive MD5 differs from Zenodo record 10902294")
    if not EXTRACT.exists():
        EXTRACT.mkdir(parents=True)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(EXTRACT)


def embed(rows: list[dict]) -> np.ndarray:
    import tensorflow as tf
    import tensorflow_hub as hub
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    checkpoint = ART.parent / "tfhub" / "9616fd04ec2360621642ef9455b84f4b668e219e"
    model = hub.load(str(checkpoint))
    from scipy.signal import resample_poly
    out = []
    for i, row in enumerate(rows, 1):
        audio, rate = sf.read(row["path"], dtype="float32", always_2d=False)
        if audio.ndim != 1:
            audio = np.mean(audio, axis=1)
        if rate != 16000:
            gcd = np.gcd(int(rate), 16000)
            audio = resample_poly(audio, 16000 // gcd, int(rate) // gcd).astype(np.float32)
        _, frames, _ = model(audio)
        value = np.asarray(frames, dtype=np.float32).mean(axis=0)
        if value.shape != (1024,) or not np.isfinite(value).all():
            raise ValueError(f"unexpected embedding for {row['filename']}: {value.shape}")
        out.append(value)
        if i % 100 == 0:
            print(f"embedded {i}/{len(rows)}", flush=True)
    matrix = np.stack(out)
    np.savez_compressed(EMBEDDINGS, embeddings=matrix, encoder="yamnet", manifest_fingerprint=manifest_fingerprint(rows))
    return matrix


def statistical_features(rows: list[dict]) -> np.ndarray:
    """Dependency-light fallback: mean/std log spectral bands per clip."""
    from scipy.signal import resample_poly, stft
    output = []
    for row in rows:
        audio, rate = sf.read(row["path"], dtype="float32", always_2d=False)
        if audio.ndim != 1:
            audio = np.mean(audio, axis=1)
        if rate != 16000:
            gcd = np.gcd(int(rate), 16000)
            audio = resample_poly(audio, 16000 // gcd, int(rate) // gcd).astype(np.float32)
        _, _, values = stft(audio, fs=16000, nperseg=512, noverlap=256,
                            boundary=None, padded=False)
        power = np.log(np.maximum(np.abs(values) ** 2, 1e-10))
        bands = np.array_split(power[:257], 64, axis=0)
        feature = np.concatenate([
            np.array([band.mean(axis=0).mean() for band in bands]),
            np.array([band.mean(axis=0).std() for band in bands]),
        ]).astype(np.float32)
        output.append(feature)
    matrix = np.stack(output)
    np.savez_compressed(EMBEDDINGS, embeddings=matrix, encoder="statistical_log_spectrum", manifest_fingerprint=manifest_fingerprint(rows))
    return matrix


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "n": int(y.size),
        "normal": int((y == 0).sum()),
        "anomaly": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pauc_max_fpr_0_1_standardized": float(roc_auc_score(y, score, max_fpr=0.1)),
    }


def main() -> int:
    extract_archive()
    rows = discover()
    encoder_name = "yamnet"
    cache_valid = False
    if EMBEDDINGS.is_file():
        saved = np.load(EMBEDDINGS)
        matrix = np.asarray(saved["embeddings"], dtype=np.float32)
        encoder_name = str(saved.get("encoder", "yamnet"))
        cached_fingerprint = str(saved.get("manifest_fingerprint", ""))
        cache_valid = matrix.shape[0] == len(rows) and cached_fingerprint == manifest_fingerprint(rows)
    else:
        matrix = None
    if not cache_valid:
        matrix = None
    if matrix is None:
        try:
            matrix = embed(rows)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"YAMNet unavailable ({type(exc).__name__}: {exc}); using statistical fallback", file=sys.stderr)
            matrix = statistical_features(rows)
            encoder_name = "statistical_log_spectrum"
    train = np.array([r["split"] == "train" and r["domain"] == "source" for r in rows])
    train_normal = train & np.array([r["label"] == "normal" for r in rows])
    test = np.array([r["split"] == "test" for r in rows])
    if train_normal.sum() != 990:
        raise ValueError(f"expected 990 source normal training clips, got {train_normal.sum()}")
    # Fit only source-normal training embeddings. Ledoit-Wolf is unsupervised
    # shrinkage; no test labels enter fitting or threshold/parameter selection.
    estimator = LedoitWolf().fit(matrix[train_normal])
    score = estimator.mahalanobis(matrix)
    labels = np.array([int(r["label"] == "anomaly") for r in rows])
    test_y, test_score = labels[test], score[test]
    source_normal = test & np.array([r["domain"] == "source" for r in rows]) & (labels == 0)
    target_normal = test & np.array([r["domain"] == "target" for r in rows]) & (labels == 0)
    all_anomaly = test & (labels == 1)
    # Official-style domain score: each domain's NORMAL test clips versus all
    # anomaly test clips; also retain same-domain and pooled views explicitly.
    source_official = source_normal | all_anomaly
    target_official = target_normal | all_anomaly
    source_same = source_normal | (test & np.array([r["domain"] == "source" for r in rows]) & (labels == 1))
    target_same = target_normal | (test & np.array([r["domain"] == "target" for r in rows]) & (labels == 1))
    split_records = [{k: v for k, v in row.items() if k != "path"} for row in rows]
    report = {
        "status": "complete",
        "dataset": {
            "name": "DCASE 2024 Challenge Task 2 Development Dataset",
            "record": RECORD,
            "record_api": "https://zenodo.org/api/records/10902294",
            "license": "CC BY-NC-SA 4.0",
            "machine_type": "valve",
            "archive": ARCHIVE.name,
            "download_url": URL,
            "archive_bytes": ARCHIVE.stat().st_size,
            "archive_md5": checksum(ARCHIVE, "md5"),
            "archive_sha256": checksum(ARCHIVE),
            "expected_record_md5": EXPECTED_MD5,
            "access": "open Zenodo content URL",
        },
        "method": {
            "encoder": encoder_name,
            "checkpoint": "artifacts/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e" if encoder_name == "yamnet" else None,
            "input": "mono float waveform, resampled to 16 kHz by benchmark copy",
            "embedding": "mean over per-frame 1024-D YAMNet embeddings" if encoder_name == "yamnet" else "mean/std of 64 uniformly divided log-power spectrum bands (128 dimensions)",
            "anomaly_model": "Ledoit-Wolf-shrunk covariance Mahalanobis distance",
            "fit_clips": "source-domain normal training clips only",
            "test_labels_used_for_fit": False,
            "threads": 2,
        },
        "counts": {
            "all_clips": len(rows),
            "train_source_normal": int(train_normal.sum()),
            "train_target_normal": int((np.array([r["split"] == "train" and r["domain"] == "target" and r["label"] == "normal" for r in rows])).sum()),
            "test_source_normal": int(source_normal.sum()),
            "test_source_anomaly": int((test & (np.array([r["domain"] == "source" for r in rows])) & (labels == 1)).sum()),
            "test_target_normal": int(target_normal.sum()),
            "test_target_anomaly": int((test & (np.array([r["domain"] == "target" for r in rows])) & (labels == 1)).sum()),
        },
        "metrics": {
            "pooled_all_test": metrics(test_y, test_score),
            "official_style_source_normals_vs_all_test_anomalies": metrics(labels[source_official], score[source_official]),
            "official_style_target_normals_vs_all_test_anomalies": metrics(labels[target_official], score[target_official]),
            "same_domain_source_only": metrics(labels[source_same], score[source_same]),
            "same_domain_target_only": metrics(labels[target_same], score[target_same]),
        },
        "split_records": split_records,
        "caveats": [
            "This is one valve section, not the official aggregate Task 2 score across all sections/machines.",
            "The official-style domain metrics use that domain's normal test clips versus all anomalous test clips; same-domain metrics are reported separately.",
            "YAMNet is a pretrained AudioSet model, not a valve-fault model; these metrics are DCASE machine-monitoring evidence only.",
            "No iPhone runtime, conversion, memory, energy, or THUD task accuracy was measured.",
            "Dataset license is CC BY-NC-SA 4.0; review compatibility before commercial use or redistribution.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
