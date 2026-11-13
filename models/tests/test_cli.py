"""CLI behavioral and regression tests."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from thud.__main__ import main


def write_test_tap(path, frequency=900.0, amplitude=0.4, rate=48000):
    t = np.arange(round(0.7 * rate)) / rate
    elapsed = np.maximum(t - 0.1, 0)
    x = amplitude * np.sin(2 * np.pi * frequency * elapsed) * np.exp(-elapsed / 0.04)
    x[t < 0.1] = 0
    wavfile.write(path, rate, x.astype(np.float32))
    return path


def write_pilot_manifest(directory, task):
    rows = []
    if task == "tile":
        # 4 distinct objects, 2 classes each
        for obj_idx in range(4):
            for cls_idx, label in enumerate(("bonded", "debonded")):
                idx = len(rows)
                freq = 700.0 + 800.0 * cls_idx + 10.0 * idx
                wav = write_test_tap(directory / f"tap_{idx}.wav", frequency=freq)
                rows.append({
                    "path": wav.name,
                    "label": label,
                    "object_id": f"tile_{obj_idx}",
                    "session_id": f"session_{obj_idx % 2}",
                    "recording_id": f"rec_{idx}",
                })
    else:
        # Fill: 2 vessels, 3 distinct fill levels each
        for vessel_idx in range(2):
            for level in (10, 50, 90):
                idx = len(rows)
                freq = 600.0 + level * 5.0 + vessel_idx * 20.0
                wav = write_test_tap(directory / f"fill_{idx}.wav", frequency=freq)
                rows.append({
                    "path": wav.name,
                    "label": str(float(level)),
                    "object_id": f"vessel_{vessel_idx}",
                    "session_id": f"session_{vessel_idx}",
                    "recording_id": f"rec_{idx}",
                    "family": "steel-bottle",
                })
    manifest = directory / "dataset.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, directory / rows[0]["path"], len(rows)


def run_cli(capsys, *args):
    code = main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


def test_analyze_command(tmp_path, capsys):
    wav = write_test_tap(tmp_path / "tap.wav")
    code, out, err = run_cli(capsys, "analyze", wav)
    assert code == 0, err
    assert err == ""
    data = json.loads(out)
    assert data["sample_rate"] == 48000
    assert "features" in data
    assert "peak_hz" in data["features"]


def test_module_execution_subprocess(tmp_path):
    wav = write_test_tap(tmp_path / "tap.wav")
    res = subprocess.run(
        [sys.executable, "-m", "thud", "analyze", str(wav)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["sample_rate"] == 48000


@pytest.mark.parametrize("task", ["tile", "fill"])
def test_train_predict_evaluate_workflow(tmp_path, capsys, task):
    manifest, sample_wav, sample_count = write_pilot_manifest(tmp_path, task)
    model_path = tmp_path / f"{task}_model.json"

    # 1. Train
    code, out, err = run_cli(capsys, "train", manifest, "--task", task, "--output", model_path)
    assert code == 0, err
    train_res = json.loads(out)
    assert train_res["task"] == task
    assert train_res["samples"] == sample_count

    # 2. Refuse overwrite
    code_dup, out_dup, err_dup = run_cli(capsys, "train", manifest, "--task", task, "--output", model_path)
    assert code_dup != 0
    assert "File exists" in err_dup or "thud:" in err_dup

    # 3. Predict
    predict_args = ["predict", model_path, sample_wav]
    if task == "fill":
        predict_args.extend(["--family", "steel-bottle"])
    code, out, err = run_cli(capsys, *predict_args)
    assert code == 0, err
    pred_res = json.loads(out)
    assert pred_res["task"] == task
    assert "prediction" in pred_res
    if task == "tile":
        assert pred_res["prediction"] in ("bonded", "debonded")
    else:
        assert isinstance(pred_res["prediction"], (int, float))
        assert pred_res["family"] == "steel-bottle"

    # 4. Evaluate
    code, out, err = run_cli(capsys, "evaluate", manifest, "--task", task, "--group-by", "object_id")
    assert code == 0, err
    eval_res = json.loads(out)
    assert eval_res["task"] == task
    assert "metrics" in eval_res
    assert len(eval_res["folds"]) >= 2


def test_predict_guards_family(tmp_path, capsys):
    manifest, sample_wav, _ = write_pilot_manifest(tmp_path, "fill")
    model_path = tmp_path / "fill_model.json"
    run_cli(capsys, "train", manifest, "--task", "fill", "--output", model_path)

    # Missing family
    code, out, err = run_cli(capsys, "predict", model_path, sample_wav)
    assert code != 0
    assert "requires --family" in err

    # Mismatched family
    code, out, err = run_cli(capsys, "predict", model_path, sample_wav, "--family", "glass-jar")
    assert code != 0
    assert "matching the model family" in err

def test_deep_cli_train_predict_evaluate_export(tmp_path, capsys):
    manifest, sample_wav, sample_count = write_pilot_manifest(tmp_path, "tile")
    model_path = tmp_path / "deep_model.pt"
    script_path = tmp_path / "deep_model.ts"

    code, out, err = run_cli(
        capsys, "deep-train", manifest, "--task", "tile", "--output", model_path,
        "--epochs", "1",
    )
    assert code == 0, err
    trained = json.loads(out)
    assert trained["task"] == "tile"
    assert trained["samples"] == sample_count
    assert trained["parameter_count"] < 200_000

    code, out, err = run_cli(capsys, "deep-predict", model_path, sample_wav)
    assert code == 0, err
    prediction = json.loads(out)
    assert prediction["task"] == "tile"
    assert prediction["prediction"] in ("bonded", "debonded")

    code, out, err = run_cli(
        capsys, "deep-evaluate", manifest, "--task", "tile", "--group-by", "object_id",
        "--epochs", "1",
    )
    assert code == 0, err
    evaluation = json.loads(out)
    assert len(evaluation["folds"]) == 4
    assert "accuracy" in evaluation["metrics"]

    code, out, err = run_cli(capsys, "deep-export", model_path, "--output", script_path)
    assert code == 0, err
    exported = json.loads(out)
    assert exported["format"] == "torchscript"
    assert script_path.is_file()


def test_deep_fill_cli_requires_matching_family(tmp_path, capsys):
    manifest, sample_wav, _ = write_pilot_manifest(tmp_path, "fill")
    model_path = tmp_path / "deep_fill.pt"
    code, out, err = run_cli(
        capsys, "deep-train", manifest, "--task", "fill", "--output", model_path,
        "--epochs", "1",
    )
    assert code == 0, err

    code, out, err = run_cli(capsys, "deep-predict", model_path, sample_wav)
    assert code != 0
    assert "requires --family" in err

    code, out, err = run_cli(
        capsys, "deep-predict", model_path, sample_wav, "--family", "steel-bottle",
    )
    assert code == 0, err
    assert json.loads(out)["family"] == "steel-bottle"
