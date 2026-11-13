"""Contracts and adapters for already-loaded audio encoders.

No model package, checkpoint, download, or conversion is performed here.
``YAMNET_CONTRACT`` records 16 kHz mono audio and its 521-score/1024-D
outputs. ``EFFICIENTAT_CONTRACT`` records 32 kHz audio with 128 mel bins,
800-sample windows, and 320-sample hops. ``prepare_waveform`` resamples a
waveform for a contract, while ``wrap_loaded_encoder`` validates the output of
an already-loaded callable or framework module. Shape validation is not a claim
of pretrained weights, conversion success, or task accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Callable, Mapping

import numpy as np

__all__ = [
    "CONTRACTS",
    "EFFICIENTAT_CONTRACT",
    "EncoderContract",
    "LoadedEncoder",
    "OutputSpec",
    "YAMNET_CONTRACT",
    "prepare_waveform",
    "validate_encoder_output",
    "wrap_loaded_encoder",
]


@dataclass(frozen=True)
class OutputSpec:
    """One named encoder output with a fixed final dimension."""

    dimension: int
    description: str


@dataclass(frozen=True)
class EncoderContract:
    """Input frontend and output-shape metadata for a pretrained encoder."""

    name: str
    sample_rate_hz: int
    frontend: Mapping[str, int | float]
    outputs: Mapping[str, OutputSpec]


YAMNET_CONTRACT = EncoderContract(
    name="yamnet",
    sample_rate_hz=16_000,
    frontend={
        "patch_seconds": 0.96,
        "patch_hop_seconds": 0.48,
        "window_samples": 400,
        "hop_samples": 160,
        "mel_bins": 64,
        "mel_min_hz": 125,
        "mel_max_hz": 7500,
    },
    outputs={
        "scores": OutputSpec(521, "per-frame AudioSet class scores"),
        "embeddings": OutputSpec(1024, "per-frame YAMNet embeddings"),
    },
)

EFFICIENTAT_CONTRACT = EncoderContract(
    name="efficientat",
    sample_rate_hz=32_000,
    frontend={
        "mel_bins": 128,
        "window_samples": 800,
        "hop_samples": 320,
    },
    outputs={
        "logits": OutputSpec(527, "per-example AudioSet class logits"),
    },
)

CONTRACTS = {c.name: c for c in (YAMNET_CONTRACT, EFFICIENTAT_CONTRACT)}


def prepare_waveform(
    waveform: Any,
    sample_rate_hz: int,
    contract: EncoderContract,
) -> np.ndarray:
    """Return finite mono float32 audio at ``contract.sample_rate_hz``.

    Resampling is performed on a copy. This function does not pad, crop,
    normalize, or run an encoder frontend.
    """

    if not isinstance(sample_rate_hz, (int, np.integer)) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive integer")
    x = np.asarray(waveform)
    if x.ndim != 1:
        raise ValueError("encoder input must be a one-dimensional mono waveform")
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("encoder input must be numeric")
    x = x.astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("encoder input contains nonfinite samples")
    if sample_rate_hz == contract.sample_rate_hz:
        return x.copy()

    from scipy.signal import resample_poly

    divisor = gcd(int(sample_rate_hz), contract.sample_rate_hz)
    return np.asarray(
        resample_poly(
            x,
            contract.sample_rate_hz // divisor,
            int(sample_rate_hz) // divisor,
        ),
        dtype=np.float32,
    )


def validate_encoder_output(
    output: Any,
    contract: EncoderContract,
    output_kind: str,
) -> Any:
    """Validate and return an already-produced encoder output unchanged.

    Outputs must have rank two: ``(frames_or_batch, dimension)``. This
    deliberately distinguishes YAMNet's 521-score output from its 1024-D
    embedding output instead of guessing from a tuple or framework type.
    """

    try:
        spec = contract.outputs[output_kind]
    except KeyError as exc:
        known = ", ".join(sorted(contract.outputs))
        raise ValueError(
            f"Unknown {contract.name} output kind {output_kind!r}; choose {known}"
        ) from exc

    shape = getattr(output, "shape", None)
    if shape is None:
        raise ValueError("encoder output must expose a tensor-like shape")
    try:
        actual = tuple(int(d) for d in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("encoder output shape is not concrete") from exc
    if len(actual) != 2 or actual[0] <= 0 or actual[1] != spec.dimension:
        raise ValueError(
            f"{contract.name} {output_kind} output must have shape "
            f"(N, {spec.dimension}), got {actual}"
        )
    return output


class LoadedEncoder:
    """Shape-validating wrapper around one already-loaded callable/module."""

    def __init__(
        self,
        encoder: Callable[[Any], Any],
        contract: EncoderContract,
        output_kind: str,
    ) -> None:
        if not callable(encoder):
            raise TypeError("encoder must be callable")
        if output_kind not in contract.outputs:
            known = ", ".join(sorted(contract.outputs))
            raise ValueError(
                f"Unknown {contract.name} output kind {output_kind!r}; choose {known}"
            )
        self.encoder = encoder
        self.contract = contract
        self.output_kind = output_kind

    def __call__(self, waveform: Any, *, sample_rate_hz: int | None = None) -> Any:
        if sample_rate_hz is not None:
            waveform = prepare_waveform(waveform, sample_rate_hz, self.contract)
        return validate_encoder_output(
            self.encoder(waveform), self.contract, self.output_kind
        )


def wrap_loaded_encoder(
    encoder: Callable[[Any], Any],
    contract: EncoderContract,
    output_kind: str,
) -> LoadedEncoder:
    """Wrap an already-loaded callable/module; never downloads or converts it."""

    return LoadedEncoder(encoder, contract, output_kind)
