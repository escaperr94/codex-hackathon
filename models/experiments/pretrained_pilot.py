"""Controlled WAV pilot for actual pretrained encoders; not physical task evidence.

Run from repo root: python -m experiments.pretrained_pilot --output results.json
Generated WAVs live in a TemporaryDirectory and are removed after extraction.
"""

import argparse
import csv
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.io import wavfile

from thud.models import evaluate as evaluate_classical, load_dataset
from thud.pretrained import YAMNET_HANDLE, build_yamnet_transfer_dataset
from thud.transfer import evaluate_transfer_head, fit_transfer_head, predict_transfer_head


def create_pilot(root, task, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    rate = 48000
    time_axis = np.arange(round(0.8 * rate)) / rate
    for specimen in range(8 if task == "tile" else 2):
        levels = ["bonded", "debonded"] if task == "tile" else [0, 25, 50, 75, 100]
        for session in range(2):
            for label in levels:
                index = len(rows)
                frequency = (750 if label == "bonded" else 2200) if task == "tile" else 650 + 12 * label
                frequency += specimen * 4 + session * 6
                elapsed = np.maximum(time_axis - 0.12, 0)
                signal = (0.25 + 0.08 * rng.random()) * np.sin(2 * np.pi * frequency * elapsed)
                signal *= np.exp(-elapsed / (0.035 + 0.005 * session))
                signal[time_axis < 0.12] = 0
                signal += rng.normal(0, 1e-5, len(signal))
                path = root / f"{task}-{index}.wav"
                wavfile.write(path, rate, signal.astype(np.float32))
                row = {"path": path.name, "label": label, "object_id": f"specimen-{specimen}",
                       "session_id": f"session-{session}", "recording_id": f"{task}-{index}"}
                if task == "fill":
                    row["family"] = "synthetic-frequency-vessel"
                rows.append(row)
    manifest = root / f"{task}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def run(checkpoint=YAMNET_HANDLE, efficientat_bundle=None):
    result = {"evidence": "synthetic software experiment; NOT real tile/fill accuracy",
              "seed": 42, "encoder": YAMNET_HANDLE, "tasks": {}}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for task in ("tile", "fill"):
            manifest = create_pilot(root, task)
            started = time.perf_counter()
            _, records, embeddings, labels = build_yamnet_transfer_dataset(checkpoint, manifest, task)
            embedding_seconds = time.perf_counter() - started
            family = records[0].get("family")
            head = fit_transfer_head(embeddings, labels, task, family)
            prediction = predict_transfer_head(head, embeddings)
            classical_records, features, targets = load_dataset(manifest, task)
            groups = {}
            for group_by in ("object_id", "session_id"):
                groups[group_by] = {
                    "yamnet": evaluate_transfer_head(records, embeddings, labels, task, group_by)["metrics"],
                    "classical": evaluate_classical(classical_records, features, targets, task, group_by)["metrics"],
                }
            entry = {"samples": len(records), "embedding_shape": list(embeddings.shape),
                     "embedding_seconds_desktop_including_load": embedding_seconds,
                     "training_metric": float(np.mean(prediction == labels)) if task == "tile" else float(np.mean(np.abs(prediction - labels))),
                     "held_out": groups}
            if efficientat_bundle is not None:
                from thud.efficientat import efficientat_wav_embedding
                matrix = np.stack([efficientat_wav_embedding(efficientat_bundle, row["path"]) for row in records])
                entry["efficientat_embedding_shape"] = list(matrix.shape)
                for group_by in ("object_id", "session_id"):
                    groups[group_by]["efficientat"] = evaluate_transfer_head(
                        records, matrix, labels, task, group_by, encoder="efficientat-mn04-as",
                    )["metrics"]
            result["tasks"][task] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=YAMNET_HANDLE)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        parser.error("output exists; choose a new filename")
    result = run(args.checkpoint)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
