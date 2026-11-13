"""Task-trained log-mel depthwise-separable CNN for THUD research.

This module trains only on labelled local WAVs. It contains no pretrained weights
and makes no claim that a physical property is observable from every tap protocol.
The checkpoint is a Torch state-dict artifact; do not load untrusted checkpoints.
"""

from __future__ import annotations

import math
from numbers import Real
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.signal import resample_poly

from .audio import _read_pcm, analyze_wav
from .models import _records, _task, load_dataset

MODEL_SCHEMA = "thud-logmel-cnn"
MODEL_VERSION = 1
SAMPLE_RATE = 16_000
INPUT_SECONDS = 1.0
N_MELS = 64
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160


def _mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                    f_min=50.0, f_max=7500.0):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    points = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index:index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        left_slice = np.arange(left, min(center, filters.shape[1]))
        right_slice = np.arange(center, min(right, filters.shape[1]))
        if len(left_slice):
            filters[index, left_slice] = (left_slice - left) / (center - left)
        if len(right_slice):
            filters[index, right_slice] = (right - right_slice) / (right - center)
    return torch.from_numpy(filters)


class LogMelFrontend(nn.Module):
    """Fixed-shape waveform-to-log-mel frontend for 16 kHz mono audio."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN_LENGTH))
        self.register_buffer("mel_filterbank", _mel_filterbank())

    def forward(self, waveform):
        if waveform.ndim != 2 or waveform.shape[1] != round(SAMPLE_RATE * INPUT_SECONDS):
            raise ValueError(f"waveform must have shape (batch, {round(SAMPLE_RATE * INPUT_SECONDS)})")
        spectrum = torch.stft(
            waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            window=self.window, center=False, return_complex=True,
        )
        power = spectrum.abs().square()
        mel = torch.matmul(self.mel_filterbank, power)
        return torch.log(torch.clamp(mel, min=1e-8)).unsqueeze(1)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, stride, 1,
                      groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ThudCNN(nn.Module):
    """Small task head; default parameter count is below 200,000."""

    def __init__(self, task):
        super().__init__()
        if task not in ("tile", "fill"):
            raise ValueError("task must be tile or fill")
        self.task = task
        self.frontend = LogMelFrontend()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12), nn.ReLU6(inplace=True),
            DepthwiseSeparableBlock(12, 24, 2),
            DepthwiseSeparableBlock(24, 40, 2),
            DepthwiseSeparableBlock(40, 64, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, 2 if task == "tile" else 1)

    def embedding(self, waveform):
        x = self.encoder(self.frontend(waveform))
        return x.flatten(1)

    def forward(self, waveform):
        return self.head(self.embedding(waveform))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _waveform_from_path(path):
    onset_s = analyze_wav(path)["onset_s"]
    rate, samples = _read_pcm(path)
    start = max(0, round((onset_s - 0.05) * rate))
    samples = samples[start:start + round(rate * INPUT_SECONDS)]
    divisor = np.gcd(rate, SAMPLE_RATE)
    if rate != SAMPLE_RATE:
        samples = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor)
    samples = np.asarray(samples, dtype=np.float32)
    samples -= samples.mean()
    peak = np.max(np.abs(samples))
    if peak > 1e-8:
        samples /= peak
    target = round(SAMPLE_RATE * INPUT_SECONDS)
    if len(samples) < target:
        samples = np.pad(samples, (0, target - len(samples)))
    else:
        samples = samples[:target]
    return torch.from_numpy(samples.copy())


def _load_records(manifest, task):
    records, _, y = load_dataset(manifest, task)
    waveforms = torch.stack([_waveform_from_path(record["path"]) for record in records])
    if task == "tile":
        targets = torch.tensor([int(value == "debonded") for value in y], dtype=torch.long)
        family = None
    else:
        targets = torch.tensor(y, dtype=torch.float32)
        family = records[0]["family"]
    return records, waveforms, targets, family


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _waveforms(waveforms):
    if (not isinstance(waveforms, torch.Tensor) or waveforms.ndim != 2
            or waveforms.shape[1] != round(SAMPLE_RATE * INPUT_SECONDS)
            or len(waveforms) == 0):
        raise ValueError("waveforms have the wrong fixed input shape or no rows")
    if waveforms.is_complex() or waveforms.dtype == torch.bool:
        raise ValueError("waveforms must be real numeric tensors")
    waveforms = waveforms.detach().float()
    if not torch.isfinite(waveforms).all():
        raise ValueError("waveforms must be finite float32 values")
    return waveforms


def _target_tensor(targets, task, count):
    _task(task)
    if (not isinstance(targets, torch.Tensor) or targets.ndim != 1
            or len(targets) != count):
        raise ValueError("targets must be one dimensional and match waveform rows")
    if targets.is_complex() or targets.dtype == torch.bool or not torch.isfinite(targets).all():
        raise ValueError("targets must be finite real numeric tensors")
    if task == "tile":
        if not ((targets == 0) | (targets == 1)).all():
            raise ValueError("tile targets must be 0 (bonded) or 1 (debonded)")
        return targets.detach().long()
    if not ((targets >= 0) & (targets <= 100)).all():
        raise ValueError("fill targets must be percentage points in [0, 100]")
    return targets.detach().float()


def train_deep(waveforms, targets, task, *, family=None, epochs=40, learning_rate=1e-3, seed=0,
               device="cpu"):
    """Fit the task CNN on provided rows and return a portable checkpoint dictionary."""
    _task(task)
    waveforms = _waveforms(waveforms)
    targets = _target_tensor(targets, task, len(waveforms))
    if task == "tile" and family is not None:
        raise ValueError("tile training does not accept a container family")
    if task == "fill" and (not isinstance(family, str) or not family.strip()):
        raise ValueError("fill training requires a nonempty container family")
    if task == "tile" and set(targets.tolist()) != {0, 1}:
        raise ValueError("tile training requires both bonded and debonded classes")
    if task == "fill" and len(torch.unique(targets)) < 3:
        raise ValueError("fill training requires at least three distinct calibration levels")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 10000:
        raise ValueError("epochs must be an integer between 1 and 10000")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, Real)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    _seed(seed)
    model = ThudCNN(task).to(device)
    inputs = waveforms.to(device)
    values = targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() if task == "tile" else nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = criterion(output, values if task == "tile" else values.unsqueeze(1))
        if not torch.isfinite(loss):
            raise ValueError("deep model training produced a nonfinite loss")
        loss.backward()
        optimizer.step()
    model.eval()
    return {
        "schema": MODEL_SCHEMA,
        "version": MODEL_VERSION,
        "task": task,
        "sample_rate": SAMPLE_RATE,
        "input_seconds": INPUT_SECONDS,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "classes": ["bonded", "debonded"] if task == "tile" else None,
        "family": family,
        "parameter_count": parameter_count(model),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("deep checkpoint must be a dictionary")
    required = {"schema", "version", "task", "sample_rate", "input_seconds", "n_mels",
                "n_fft", "win_length", "hop_length", "classes", "family", "parameter_count",
                "state_dict"}
    if set(checkpoint) != required or checkpoint["schema"] != MODEL_SCHEMA or checkpoint["version"] != MODEL_VERSION:
        raise ValueError("unsupported deep checkpoint schema")
    if checkpoint["task"] not in ("tile", "fill"):
        raise ValueError("deep checkpoint task must be tile or fill")
    if checkpoint["sample_rate"] != SAMPLE_RATE or checkpoint["input_seconds"] != INPUT_SECONDS:
        raise ValueError("deep checkpoint frontend contract does not match this runtime")
    if checkpoint["n_mels"] != N_MELS or checkpoint["n_fft"] != N_FFT:
        raise ValueError("deep checkpoint mel contract does not match this runtime")
    if checkpoint["win_length"] != WIN_LENGTH or checkpoint["hop_length"] != HOP_LENGTH:
        raise ValueError("deep checkpoint STFT contract does not match this runtime")
    if checkpoint["task"] == "tile":
        if checkpoint["classes"] != ["bonded", "debonded"] or checkpoint["family"] is not None:
            raise ValueError("invalid tile checkpoint labels")
    elif checkpoint["classes"] is not None or not isinstance(checkpoint["family"], str) or not checkpoint["family"].strip():
        raise ValueError("invalid fill checkpoint family")
    if not isinstance(checkpoint["state_dict"], dict):
        raise ValueError("deep checkpoint state_dict must be a dictionary")
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
           for value in checkpoint["state_dict"].values()):
        raise ValueError("deep checkpoint weights must be finite tensors")


def checkpoint_model(checkpoint):
    validate_checkpoint(checkpoint)
    model = ThudCNN(checkpoint["task"])
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"deep checkpoint weights do not match model: {exc}") from exc
    if parameter_count(model) != checkpoint["parameter_count"]:
        raise ValueError("deep checkpoint parameter count is incorrect")
    model.eval()
    return model


def save_checkpoint(checkpoint, path):
    validate_checkpoint(checkpoint)
    with Path(path).open("xb") as handle:
        torch.save(checkpoint, handle)


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError(f"cannot load deep checkpoint: {exc}") from exc
    validate_checkpoint(checkpoint)
    return checkpoint


def predict_deep(checkpoint, waveforms):
    model = checkpoint_model(checkpoint)
    if isinstance(waveforms, torch.Tensor) and waveforms.ndim == 1:
        waveforms = waveforms.unsqueeze(0)
    waveforms = _waveforms(waveforms)
    with torch.inference_mode():
        output = model(waveforms.cpu())
    if not torch.isfinite(output).all():
        raise ValueError("deep model prediction produced nonfinite values")
    if checkpoint["task"] == "tile":
        return np.asarray(output.argmax(dim=1).cpu().tolist(), dtype=int)
    return np.asarray(output[:, 0].cpu().tolist(), dtype=float)


def load_deep_dataset(manifest, task):
    """Load manifest WAVs into fixed-shape tensors and task targets."""
    return _load_records(manifest, task)


def load_deep_waveform(path):
    """Read a WAV, aligning its impact at 50 ms before padding to one second."""
    return _waveform_from_path(path)


def evaluate_deep(records, waveforms, targets, task, group_by, *, epochs=40, seed=0):
    _task(task)
    waveforms = _waveforms(waveforms)
    targets = _target_tensor(targets, task, len(waveforms))
    labels = targets.cpu().numpy()
    if task == "tile":
        labels = np.where(labels == 0, "bonded", "debonded")
    family = _records(records, labels, task)
    if group_by not in ("object_id", "session_id"):
        raise ValueError("group_by must be object_id or session_id")
    groups = np.asarray([record[group_by] for record in records])
    unique = sorted(set(groups.tolist()))
    if len(unique) < 2:
        raise ValueError("evaluation requires at least two distinct held-out groups")
    predictions = [None] * len(records)
    folds = []
    for group in unique:
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        checkpoint = train_deep(
            waveforms[train], targets[train], task, family=family, epochs=epochs, seed=seed,
        )
        predicted = predict_deep(checkpoint, waveforms[test])
        if task == "tile":
            actual = targets[test].cpu().numpy()
            matrix = [[int(np.sum((actual == truth) & (predicted == guess))) for guess in (0, 1)] for truth in (0, 1)]
            metrics = {"accuracy": float(np.mean(actual == predicted)), "confusion_matrix": matrix,
                       "defect_recall": matrix[1][1] / sum(matrix[1]) if sum(matrix[1]) else None}
        else:
            actual = targets[test].cpu().numpy()
            metrics = {"mae_percentage_points": float(np.mean(np.abs(actual - predicted)))}
        for index, value in zip(test, predicted):
            predictions[index] = int(value) if task == "tile" else float(value)
        folds.append({"held_out_group": str(group), "training_groups": sorted(set(groups[train].tolist())),
                      "held_out_indices": test.tolist(), "metrics": metrics})
    actual = targets.cpu().numpy()
    if task == "tile":
        aggregate = {"accuracy": float(np.mean(actual == predictions))}
    else:
        aggregate = {"mae_percentage_points": float(np.mean(np.abs(actual - np.asarray(predictions))))}
    return {"task": task, "group_by": group_by, "folds": folds, "predictions": predictions, "metrics": aggregate}

def export_torchscript(checkpoint, path):
    """Export a validated checkpoint to fixed-shape TorchScript inference."""
    model = checkpoint_model(checkpoint)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    example = torch.zeros(1, round(SAMPLE_RATE * INPUT_SECONDS))
    traced = torch.jit.trace(model, example, strict=True)
    traced.save(str(destination))
