"""Controlled synthetic CNN diagnostic, NOT evidence of real tile/fill accuracy.

Run from the repository root with the deep extra installed:
    python -m experiments.cnn_pilot --output cnn-pilot.json --epochs 20

WAVs live only in a TemporaryDirectory. No training, downloads, or output writes
occur on import. The requested JSON destination must not already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import wavfile
import torch

from thud.deep import load_deep_dataset, predict_deep, train_deep


def generate_dataset(directory, task, seed=0):
    """Write a tiny labelled damped-tone dataset with crossed object/session IDs."""
    if task not in ("tile", "fill"):
        raise ValueError("task must be tile or fill")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fields = ["path", "label", "object_id", "session_id", "recording_id"]
    if task == "fill":
        fields.append("family")
    manifest = directory / "manifest.csv"
    with manifest.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for object_index in range(3):
            for session in range(2):
                # Later session deliberately places every impact beyond one second.
                rate = 16000 if session == 0 else 48000
                onset = 0.10 if session == 0 else 1.20
                t = np.arange(round(1.8 * rate)) / rate
                u = np.maximum(t - onset, 0)
                for label_index, label in enumerate(("bonded", "debonded") if task == "tile" else (0, 50, 100)):
                    frequency = (650 + 1000 * label_index if task == "tile" else 650 + 10 * label)
                    frequency += (object_index - 1) * 18 + session * 9
                    decay = 0.035 + 0.025 * label_index + 0.002 * object_index
                    signal = np.sin(2 * np.pi * frequency * u)
                    signal += 0.15 * np.sin(2 * np.pi * 1.8 * frequency * u)
                    signal *= rng.uniform(0.25, 0.45) * np.exp(-u / decay)
                    signal[t < onset] = 0
                    signal += rng.normal(0, 0.00002, len(t))
                    name = f"o{object_index}_s{session}_l{label_index}.wav"
                    wavfile.write(directory / name, rate, signal.astype(np.float32))
                    record = {"path": name, "label": label, "object_id": f"object{object_index}",
                              "session_id": f"session{session}", "recording_id": Path(name).stem}
                    if task == "fill":
                        record["family"] = "synthetic-vessel"
                    writer.writerow(record)
    return manifest


def _metrics(actual, predicted, task):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if task == "fill":
        return {"count": len(actual), "mae_percentage_points": float(np.mean(np.abs(actual - predicted)))}
    matrix = [[int(np.sum((actual == truth) & (predicted == guess)))
               for guess in (0, 1)] for truth in (0, 1)]
    return {"count": len(actual), "accuracy": float(np.mean(actual == predicted)),
            "confusion_matrix": matrix, "classes": ["bonded", "debonded"]}


def run_pilot(*, epochs=20, seed=0):
    """Fit each fold independently; compare resubstitution with genuine group holdouts."""
    results = {}
    with tempfile.TemporaryDirectory(prefix="thud-cnn-pilot-") as temporary:
        for task in ("tile", "fill"):
            manifest = generate_dataset(Path(temporary) / task, task, seed)
            records, waveforms, targets, family = load_deep_dataset(manifest, task)
            actual = targets.numpy()
            evaluations = {}
            for group_by in ("object_id", "session_id"):
                groups = np.array([record[group_by] for record in records])
                held_out_predictions = np.empty(len(records), dtype=int if task == "tile" else float)
                folds = []
                for group in sorted(set(groups)):
                    train = np.flatnonzero(groups != group)
                    held_out = np.flatnonzero(groups == group)
                    assert not set(train) & set(held_out)
                    checkpoint = train_deep(waveforms[train], targets[train], task,
                                            family=family, epochs=epochs, seed=seed)
                    training_predictions = predict_deep(checkpoint, waveforms[train])
                    predictions = predict_deep(checkpoint, waveforms[held_out])
                    held_out_predictions[held_out] = predictions
                    folds.append({"held_out_group": str(group),
                                  "training_groups": sorted(set(groups[train].tolist())),
                                  "training_indices": train.tolist(), "held_out_indices": held_out.tolist(),
                                  "training_metrics": _metrics(actual[train], training_predictions, task),
                                  "held_out_metrics": _metrics(actual[held_out], predictions, task),
                                  "training_predictions": training_predictions.tolist(),
                                  "held_out_predictions": predictions.tolist()})
                evaluations[group_by] = {"folds": folds,
                                         "held_out_metrics": _metrics(actual, held_out_predictions, task),
                                         "held_out_predictions": held_out_predictions.tolist()}
            results[task] = {"count": len(records), "actual": actual.tolist(),
                             "records": [{key: value for key, value in record.items()
                                          if key in ("object_id", "session_id", "recording_id", "label", "family")}
                                         for record in records],
                             "evaluations": evaluations}
    return {"experiment": "synthetic-damped-tone-cnn-pilot", "seed": seed, "epochs": epochs,
            "device": "cpu", "torch_version": str(torch.__version__),
            "limitations": ["Synthetic diagnostic only; no real tile, vessel, or iPhone accuracy evidence.",
                            "Object/session IDs denote controlled simulated perturbations, not physical specimens.",
                            "Generator assigns frequency/decay cues; success does not establish physical observability.",
                            "Training metrics are resubstitution, not independent validation."],
            "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New JSON report path (never overwritten)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 10000:
        parser.error("--epochs must be between 1 and 10000")
    if args.output.exists():
        parser.error("--output already exists")
    # Tiny CPU workloads are slower with large thread pools; keep this pilot bounded.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report = run_pilot(epochs=args.epochs, seed=args.seed)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote synthetic-only CNN diagnostic: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
