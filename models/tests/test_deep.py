import numpy as np
import pytest
import torch
from scipy.io import wavfile

from thud.deep import (
    INPUT_SECONDS,
    SAMPLE_RATE,
    ThudCNN,
    evaluate_deep,
    load_checkpoint,
    load_deep_dataset,
    load_deep_waveform,
    predict_deep,
    save_checkpoint,
    train_deep,
)


def waves(count=6):
    t = torch.arange(round(SAMPLE_RATE * INPUT_SECONDS), dtype=torch.float32) / SAMPLE_RATE
    values = []
    for i in range(count):
        frequency = 700 + 450 * (i % 2) + i * 3
        u = torch.clamp(t - 0.1, min=0)
        x = 0.5 * torch.sin(2 * torch.pi * frequency * u) * torch.exp(-u / 0.04)
        x[t < 0.1] = 0
        values.append(x)
    return torch.stack(values)


def test_model_is_small_and_has_expected_frontend():
    model = ThudCNN("tile")
    assert sum(p.numel() for p in model.parameters()) < 200_000
    assert model(waves(2)).shape == (2, 2)
    assert ThudCNN("fill")(waves(2)).shape == (2, 1)


def test_train_save_reload_predict_parity(tmp_path):
    waveform = waves()
    tile_targets = torch.tensor([0, 1, 0, 1, 0, 1])
    checkpoint = train_deep(waveform, tile_targets, "tile", epochs=2, seed=4)
    model_path = tmp_path / "tile.pt"
    save_checkpoint(checkpoint, model_path)
    loaded = load_checkpoint(model_path)
    np.testing.assert_array_equal(predict_deep(checkpoint, waveform), predict_deep(loaded, waveform))


def test_fill_training_and_grouped_evaluation():
    waveform = waves(6)
    targets = torch.tensor([0.0, 25.0, 50.0, 0.0, 25.0, 50.0])
    records = [
        {"path": f"{i}.wav", "object_id": f"v{i // 3}", "session_id": f"s{i // 3}",
         "recording_id": f"r{i}", "family": "jar", "label": float(targets[i])}
        for i in range(6)
    ]
    result = evaluate_deep(records, waveform, targets, "fill", "object_id", epochs=1)
    assert len(result["folds"]) == 2
    assert "mae_percentage_points" in result["metrics"]


def test_save_refuses_overwrite(tmp_path):
    waveform = waves()
    checkpoint = train_deep(waveform, torch.tensor([0, 1, 0, 1, 0, 1]), "tile", epochs=1)
    path = tmp_path / "model.pt"
    save_checkpoint(checkpoint, path)
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, path)


def test_invalid_waveform_shape_rejected():
    checkpoint = train_deep(waves(), torch.tensor([0, 1, 0, 1, 0, 1]), "tile", epochs=1)
    with pytest.raises(ValueError, match="fixed input shape"):
        predict_deep(checkpoint, torch.zeros(2, 100))


def test_waveform_aligns_late_impact_and_resamples(tmp_path):
    outputs = []
    for index, (rate, onset) in enumerate(((16000, 0.1), (48000, 1.3))):
        t = np.arange(round(rate * 1.8)) / rate
        u = np.maximum(t - onset, 0)
        x = 0.4 * np.sin(2 * np.pi * 900 * u) * np.exp(-u / 0.04)
        x[t < onset] = 0
        path = tmp_path / f"{index}.wav"
        wavfile.write(path, rate, x.astype(np.float32))
        outputs.append(load_deep_waveform(path))
    for output in outputs:
        assert output.shape == (SAMPLE_RATE,)
        assert torch.isfinite(output).all()
        assert 0.04 < int(output.abs().argmax()) / SAMPLE_RATE < 0.07
        assert output.square().sum() > 1
    assert torch.nn.functional.cosine_similarity(*outputs, dim=0) > 0.99


@pytest.mark.parametrize("task, targets, kwargs, message", [
    ("other", torch.tensor([0, 1]), {}, "task"),
    ("tile", torch.tensor([[0], [1]]), {}, "one dimensional"),
    ("tile", torch.tensor([0, 2]), {}, "tile targets"),
    ("tile", torch.tensor([0.0, float("nan")]), {}, "finite"),
    ("fill", torch.tensor([-1.0, 50.0]), {"family": "jar"}, "percentage points"),
    ("tile", torch.tensor([0, 1]), {"epochs": 1.5}, "epochs"),
    ("tile", torch.tensor([0, 1]), {"learning_rate": float("nan")}, "learning_rate"),
    ("tile", torch.tensor([0, 1]), {"learning_rate": 0}, "learning_rate"),
])
def test_training_rejects_invalid_inputs(task, targets, kwargs, message):
    with pytest.raises(ValueError, match=message):
        train_deep(waves(2), targets, task, **{"epochs": 1, **kwargs})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_training_rejects_nonfinite_waveforms(value):
    waveform = waves(2)
    waveform[0, 10] = value
    with pytest.raises(ValueError, match="finite"):
        train_deep(waveform, torch.tensor([0, 1]), "tile", epochs=1)


def test_training_accepts_numeric_tensor_dtypes():
    checkpoint = train_deep(waves(2).double(), torch.tensor([0.0, 1.0]), "tile", epochs=1)
    assert predict_deep(checkpoint, waves(2).double()).shape == (2,)
    for bad in (torch.full((2, SAMPLE_RATE), float("nan")), torch.empty(0, SAMPLE_RATE)):
        with pytest.raises(ValueError, match="finite|no rows"):
            predict_deep(checkpoint, bad)
    checkpoint["state_dict"]["head.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        predict_deep(checkpoint, waves(2))


def test_evaluation_rejects_mismatched_records():
    with pytest.raises(ValueError, match="same number"):
        evaluate_deep([], waves(2), torch.tensor([0, 1]), "tile", "object_id", epochs=1)


@pytest.mark.parametrize("task, count", [("tile", 12), ("fill", 18)])
def test_pilot_generates_grouped_temporary_wavs(tmp_path, task, count):
    from experiments.cnn_pilot import generate_dataset

    manifest = generate_dataset(tmp_path, task, seed=4)
    records, waveform, targets, family = load_deep_dataset(manifest, task)
    assert len(records) == count
    assert waveform.shape == (count, SAMPLE_RATE)
    assert torch.isfinite(waveform).all()
    assert set(record["object_id"] for record in records) == {"object0", "object1", "object2"}
    assert set(record["session_id"] for record in records) == {"session0", "session1"}
    assert len(targets.unique()) == (2 if task == "tile" else 3)
    assert family == ("synthetic-vessel" if task == "fill" else None)


def test_pilot_reports_train_and_disjoint_holdouts(monkeypatch):
    from experiments import cnn_pilot

    def fit(waveform, targets, task, **kwargs):
        return task

    def predict(task, waveform):
        return np.zeros(len(waveform), dtype=int if task == "tile" else float)

    monkeypatch.setattr(cnn_pilot, "train_deep", fit)
    monkeypatch.setattr(cnn_pilot, "predict_deep", predict)
    result = cnn_pilot.run_pilot(epochs=1, seed=4)
    assert "Synthetic diagnostic only" in result["limitations"][0]
    for task, summary in result["results"].items():
        for group_by, evaluation in summary["evaluations"].items():
            all_test_indices = []
            for fold in evaluation["folds"]:
                train, held_out = fold["training_indices"], fold["held_out_indices"]
                assert not set(train) & set(held_out)
                assert len(train) + len(held_out) == summary["count"]
                assert fold["held_out_group"] not in fold["training_groups"]
                assert fold["training_metrics"]["count"] == len(train)
                assert fold["held_out_metrics"]["count"] == len(held_out)
                all_test_indices.extend(held_out)
            assert sorted(all_test_indices) == list(range(summary["count"]))
