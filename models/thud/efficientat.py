"""Frozen official EfficientAT mn04_as, with its unchanged upstream frontend.

Requires matching torch/torchvision/torchaudio builds and librosa>=0.10. Fetch
source explicitly into ignored artifacts; loading and embedding never download.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from types import ModuleType
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

EFFICIENTAT_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
EFFICIENTAT_SOURCE_URL = (
    "https://raw.githubusercontent.com/fschmid56/EfficientAT/"
    f"{EFFICIENTAT_SOURCE_COMMIT}/"
)
EFFICIENTAT_CHECKPOINT_SHA256 = "899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c"
EFFICIENTAT_SAMPLE_RATE = 32_000
EFFICIENTAT_EMBEDDING_DIM = 384
# Git blob IDs from the pinned upstream tree; downloaded Python executes unchanged.
_SOURCE_BLOBS = {
    "LICENSE": "d1c76a5b494a1765209dc9004efe92920391316d",
    "metadata/class_labels_indices.csv": "3a2767e81114adecde59992cf6607f31c1862f4c",
    "helpers/utils.py": "972808feb3a7bb8dc62e808d261a5b204cb54c93",
    "models/mn/utils.py": "440b7025c1dfba37928ba9e20231aefa821e23e3",
    "models/mn/block_types.py": "d11ff3830956cf51ea7310d56591e2145adbf07d",
    "models/mn/attention_pooling.py": "03a2b41504092ef01fc0ddd75b162747281a5d96",
    "models/mn/model.py": "cb47b2ca6dc575b6011cabbc04619cbc98706955",
    "models/preprocess.py": "6c2f2995636e56a67d875e93f1d57925e045082f",
}


@dataclass(frozen=True)
class EfficientATBundle:
    model: object
    frontend: object
    source_commit: str
    checkpoint_sha256: str


def _validate_blob(name, content):
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if digest != _SOURCE_BLOBS[name]:
        raise ValueError(f"EfficientAT source differs from pinned commit: {name}")


def fetch_efficientat_source(source_dir):
    """Download only the pinned inference sources, labels and MIT license.

    Existing files are verified, never overwritten. No checkpoint is downloaded.
    A complete checkout of the same commit can also be passed to the loader.
    """
    root = Path(source_dir).resolve()
    try:
        for name in _SOURCE_BLOBS:
            path = root / name
            if path.exists():
                _validate_blob(name, path.read_bytes())
        for name in _SOURCE_BLOBS:
            path = root / name
            if path.exists():
                continue
            with urlopen(EFFICIENTAT_SOURCE_URL + name, timeout=60) as response:
                content = response.read()
            _validate_blob(name, content)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as target:
                target.write(content)
    except (OSError, URLError) as exc:
        raise ValueError(f"Cannot fetch pinned EfficientAT source: {exc}") from exc
    return root


def _upstream_modules(source_dir):
    root = Path(source_dir).resolve()
    sources = {}
    try:
        for name in _SOURCE_BLOBS:
            content = (root / name).read_bytes()
            _validate_blob(name, content)
            if name.endswith(".py"):
                sources[name[:-3].replace("/", ".")] = content
    except OSError as exc:
        raise ValueError(
            f"Cannot read pinned EfficientAT source in {root}; "
            "call fetch_efficientat_source(source_dir) first"
        ) from exc

    modules = {}

    def upstream_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in sources:
            return load_module(name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def source_open(file, *args, **kwargs):
        # Upstream helpers/utils.py reads labels relative to its working directory.
        if file == "metadata/class_labels_indices.csv":
            file = root / file
        return builtins.open(file, *args, **kwargs)

    def load_module(name):
        if name not in modules:
            module = ModuleType(name)
            module.__file__ = str(root / (name.replace(".", "/") + ".py"))
            module.__package__ = name.rpartition(".")[0]
            module.__dict__["__builtins__"] = {
                **vars(builtins), "__import__": upstream_import, "open": source_open,
                "print": logging.getLogger(__name__).debug,
            }
            modules[name] = module
            exec(compile(sources[name], module.__file__, "exec"), module.__dict__)
        return modules[name]

    # Local import/open bindings avoid mutating cwd, sys.path or global modules.
    return load_module("models.mn.model"), load_module("models.preprocess")


def load_efficientat(checkpoint, source_dir):
    """Strict-load the official mn04_as checkpoint on CPU; freeze both modules."""
    path = Path(checkpoint)
    try:
        with path.open("rb") as source:
            checkpoint_hash = hashlib.sha256(source.read()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Cannot read EfficientAT checkpoint {path}: {exc}") from exc
    if checkpoint_hash != EFFICIENTAT_CHECKPOINT_SHA256:
        raise ValueError("EfficientAT checkpoint is not the official mn04_as_mAP_432.pt")
    try:
        import torch

        architecture, preprocessing = _upstream_modules(source_dir)
        # No upstream pretrained_name: its download/partial-load path is bypassed.
        model = architecture.get_model(
            num_classes=527, pretrained_name=None, width_mult=0.4,
            strides=(2, 2, 2, 2), head_type="mlp",
        )
        weights = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        frontend = preprocessing.AugmentMelSTFT(
            n_mels=128, sr=EFFICIENTAT_SAMPLE_RATE, win_length=800, hopsize=320,
        )
        model.eval().requires_grad_(False)
        frontend.eval().requires_grad_(False)
    except (ImportError, OSError) as exc:
        raise ValueError(
            "EfficientAT needs matching torch, torchvision and torchaudio builds; "
            f"upstream import failed: {exc}"
        ) from exc
    except RuntimeError as exc:
        raise ValueError(f"Cannot load official EfficientAT weights/runtime: {exc}") from exc
    return EfficientATBundle(model, frontend, EFFICIENTAT_SOURCE_COMMIT, checkpoint_hash)


def efficientat_wav_embedding(model_bundle, path):
    """Return the 384D pooled feature vector, not the 527 AudioSet logits.

    Use the full clip at its original amplitude, with librosa's mono conversion
    and 32 kHz resampling as in upstream inference.py. No DC removal, onset crop,
    peak normalization, synthetic padding or approximate mel frontend is applied.
    """
    try:
        import librosa
        import torch

        waveform, _ = librosa.load(Path(path), sr=EFFICIENTAT_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError(f"Cannot decode EfficientAT WAV {path}: {exc}") from exc
    if waveform.ndim != 1 or len(waveform) < 514 or not np.isfinite(waveform).all():
        raise ValueError("EfficientAT requires finite mono audio with at least 514 samples at 32 kHz")
    model_bundle.model.eval().requires_grad_(False)
    model_bundle.frontend.eval().requires_grad_(False)
    try:
        with torch.inference_mode():
            audio = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))[None, :]
            spectrogram = model_bundle.frontend(audio)
            _, features = model_bundle.model(spectrogram.unsqueeze(1))
            embedding = features[0].cpu().numpy().copy()
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"EfficientAT embedding failed for {path}: {exc}") from exc
    if embedding.shape != (EFFICIENTAT_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError("EfficientAT returned an invalid 384D embedding")
    return embedding
