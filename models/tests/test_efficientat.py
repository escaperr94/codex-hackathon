import hashlib
import io
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from scipy.io import wavfile

import thud.efficientat as efficientat
from thud.efficientat import (
    EFFICIENTAT_CHECKPOINT_SHA256,
    EFFICIENTAT_EMBEDDING_DIM,
    EFFICIENTAT_SOURCE_COMMIT,
    efficientat_wav_embedding,
    fetch_efficientat_source,
    load_efficientat,
)


def test_loader_rejects_missing_and_unofficial_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Cannot read EfficientAT checkpoint"):
        load_efficientat(tmp_path / "missing.pt", tmp_path)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"not pretrained weights")
    with pytest.raises(ValueError, match="not the official"):
        load_efficientat(checkpoint, tmp_path)


def test_fetch_pins_content_and_preserves_existing_files(tmp_path, monkeypatch):
    content = b"# pinned source\n"
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    monkeypatch.setattr(efficientat, "_SOURCE_BLOBS", {"models/test.py": blob})
    urls = []

    def download(url, timeout):
        urls.append(url)
        assert timeout == 60
        return io.BytesIO(content)

    monkeypatch.setattr(efficientat, "urlopen", download)
    assert fetch_efficientat_source(tmp_path) == tmp_path.resolve()
    target = tmp_path / "models/test.py"
    assert target.read_bytes() == content
    assert urls == [efficientat.EFFICIENTAT_SOURCE_URL + "models/test.py"]
    fetch_efficientat_source(tmp_path)
    assert len(urls) == 1
    target.write_bytes(b"local changes")
    with pytest.raises(ValueError, match="differs from pinned commit"):
        fetch_efficientat_source(tmp_path)
    assert target.read_bytes() == b"local changes"
    assert len(urls) == 1


def test_fetch_rejects_wrong_download_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(efficientat, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="differs from pinned commit"):
        fetch_efficientat_source(tmp_path)
    assert not (tmp_path / "LICENSE").exists()


@pytest.fixture(scope="module")
def real_bundle():
    root = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("EFFICIENTAT_SOURCE_DIR", root / "artifacts/efficientat-source"))
    checkpoint = root / "artifacts/mn04_as_mAP_432.pt"
    if not checkpoint.is_file() or not (source / "models/mn/model.py").is_file():
        pytest.skip("Requires official checkpoint and explicitly fetched pinned source in artifacts")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchaudio")
    pytest.importorskip("librosa")
    return load_efficientat(checkpoint, source)


def _write_tone(path, rate, frequency, stereo=False):
    t = np.arange(rate // 2) / rate
    waveform = (0.15 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    if stereo:
        waveform = np.column_stack((waveform, waveform * 0.5))
    wavfile.write(path, rate, waveform)


def test_real_official_embeddings_and_upstream_frontend(real_bundle, tmp_path):
    import librosa
    import torch

    bundle = real_bundle
    assert bundle.source_commit == EFFICIENTAT_SOURCE_COMMIT
    assert bundle.checkpoint_sha256 == EFFICIENTAT_CHECKPOINT_SHA256
    assert not bundle.model.training and not bundle.frontend.training
    assert all(not parameter.requires_grad for parameter in bundle.model.parameters())
    assert (bundle.frontend.sr, bundle.frontend.n_mels,
            bundle.frontend.win_length, bundle.frontend.hopsize) == (32000, 128, 800, 320)
    assert bundle.frontend.fmax == 15000
    assert bundle.frontend.n_fft == 1024

    path = tmp_path / "stereo48k.wav"
    other = tmp_path / "mono16k.wav"
    _write_tone(path, 48000, 440, stereo=True)
    _write_tone(other, 16000, 1800)
    actual = efficientat_wav_embedding(bundle, path)
    repeat = efficientat_wav_embedding(bundle, path)
    different = efficientat_wav_embedding(bundle, other)
    assert actual.shape == (EFFICIENTAT_EMBEDDING_DIM,)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, repeat)
    assert not np.allclose(actual, different)

    waveform, _ = librosa.load(path, sr=32000, mono=True)
    with torch.inference_mode():
        spectrogram = bundle.frontend(torch.from_numpy(waveform[None, :]))
        logits, features = bundle.model(spectrogram.unsqueeze(1))
    assert logits.shape == (1, 527)
    assert features.shape == (1, 384)
    assert not features.requires_grad
    np.testing.assert_allclose(actual, features[0].numpy(), rtol=0, atol=0)


def test_real_waveform_failures_are_value_errors(real_bundle, tmp_path):
    with pytest.raises(ValueError, match="Cannot decode EfficientAT WAV"):
        efficientat_wav_embedding(real_bundle, tmp_path / "missing.wav")
    short = tmp_path / "short.wav"
    wavfile.write(short, 32000, np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError, match="at least 514 samples"):
        efficientat_wav_embedding(real_bundle, short)
    nonfinite = tmp_path / "nonfinite.wav"
    wavfile.write(nonfinite, 32000, np.full(16000, np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        efficientat_wav_embedding(real_bundle, nonfinite)


def test_real_loader_strictly_rejects_missing_weights(real_bundle, monkeypatch):
    import torch

    root = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("EFFICIENTAT_SOURCE_DIR", root / "artifacts/efficientat-source"))
    checkpoint = root / "artifacts/mn04_as_mAP_432.pt"
    weights = dict(real_bundle.model.state_dict())
    weights.pop("classifier.5.bias")

    def damaged_load(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        return weights

    monkeypatch.setattr(torch, "load", damaged_load)
    with pytest.raises(ValueError, match="Missing key"):
        load_efficientat(checkpoint, source)


def test_real_source_imports_do_not_change_global_import_state(real_bundle, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("EFFICIENTAT_SOURCE_DIR", root / "artifacts/efficientat-source"))
    sentinel = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", sentinel)
    original_path = list(sys.path)
    original_cwd = Path.cwd()
    efficientat._upstream_modules(source)
    assert sys.modules["models"] is sentinel
    assert sys.path == original_path
    assert Path.cwd() == original_cwd
