"""Frozen official YAMNet SavedModel; no architecture recreation.

The official handle downloads/caches Google's release on first use. A local
SavedModel directory works offline. Transfer heads live in thud.transfer.
"""

from pathlib import Path

import numpy as np

from .audio import _read_pcm
from .encoders import YAMNET_CONTRACT, prepare_waveform, validate_encoder_output
from .models import load_dataset

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_EMBEDDING_DIM = 1024


def load_yamnet(checkpoint=YAMNET_HANDLE):
    """Load Google's release handle or a local SavedModel directory (not .h5)."""
    import tensorflow_hub as hub

    source = str(checkpoint)
    if source != YAMNET_HANDLE:
        directory = Path(source)
        if not directory.is_dir() or not (directory / "saved_model.pb").is_file():
            raise ValueError("Use the official YAMNet URL or a local SavedModel directory, not weights-only .h5")
    return hub.load(source)


def yamnet_infer(model, waveform, sample_rate=16000):
    """Return official frame scores, 1024D frame embeddings and log-mel features."""
    waveform = np.asarray(waveform)
    if waveform.size == 0 or np.iscomplexobj(waveform):
        raise ValueError("YAMNet input must be nonempty real mono audio")
    audio = prepare_waveform(waveform, sample_rate, YAMNET_CONTRACT)
    if np.max(np.abs(audio)) > 1:
        raise ValueError("YAMNet float waveform must be in [-1, 1]")
    scores, embeddings, spectrogram = model(audio)
    values = [np.asarray(value) for value in (scores, embeddings, spectrogram)]
    validate_encoder_output(values[0], YAMNET_CONTRACT, "scores")
    validate_encoder_output(values[1], YAMNET_CONTRACT, "embeddings")
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("YAMNet output must be finite")
    return tuple(values)


def yamnet_wav_embedding(model, path):
    """Mean-pool official frame embeddings; keep each recording in one split."""
    rate, waveform = _read_pcm(path)
    _, embeddings, _ = yamnet_infer(model, waveform, rate)
    return embeddings.mean(axis=0).astype(np.float32)


def build_yamnet_transfer_dataset(checkpoint, manifest, task):
    """Reuse baseline capture/manifest/duplicate/identity validation before encoding."""
    rows, _, labels = load_dataset(manifest, task)
    model = load_yamnet(checkpoint)
    embeddings = np.stack([yamnet_wav_embedding(model, row["path"]) for row in rows])
    return model, rows, embeddings, labels
