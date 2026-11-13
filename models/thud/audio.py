"""Deterministic isolated-impact features, not calibrated physical measurements."""

from math import gcd
from pathlib import Path
import warnings

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly

FEATURE_VERSION = 1
FEATURE_NAMES = (
    "peak_hz", "centroid_hz", "bandwidth_hz", "flatness",
    "band_100_500", "band_500_1500", "band_1500_4000", "band_4000_7500",
    "energy_t50_s", "energy_t90_s", "tail_energy_ratio",
    *(f"mfcc_mean_{i}" for i in range(1, 14)),
    *(f"mfcc_std_{i}" for i in range(1, 14)),
)


def _read_pcm(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", wavfile.WavFileWarning)
            rate, pcm = wavfile.read(Path(path))
    except (OSError, ValueError, EOFError, wavfile.WavFileWarning) as exc:
        raise ValueError(f"Cannot read PCM/float WAV {path}: {exc}") from exc
    if rate < 16000 or rate > 192000:
        raise ValueError("Capture sample rate must be between 16000 and 192000 Hz")
    if pcm.ndim not in (1, 2) or (pcm.ndim == 2 and pcm.shape[1] not in (1, 2)):
        raise ValueError("Use mono or stereo WAV audio")
    if not 0.4 <= len(pcm) / rate <= 10:
        raise ValueError("Use a 0.4–10 second isolated-tap recording")
    if pcm.dtype == np.uint8:
        x = (pcm.astype(np.float64) - 128) / 128
    elif pcm.dtype.kind == "i":
        # SciPy returns 24-bit PCM left-justified in int32 as well.
        x = pcm.astype(np.float64) / (2 ** (8 * pcm.dtype.itemsize - 1))
    elif pcm.dtype.kind == "f":
        x = pcm.astype(np.float64)
    else:
        raise ValueError(f"Unsupported WAV sample type: {pcm.dtype}")
    if not np.isfinite(x).all():
        raise ValueError("Audio contains nonfinite samples")
    if np.max(np.abs(x)) >= 0.999:
        raise ValueError("Audio reaches full scale; record again with lower gain/distance")
    if x.ndim == 2:
        x = x.mean(axis=1)
    x -= x.mean()
    return rate, x


def _mfcc(segment, rate):
    divisor = gcd(rate, 16000)
    x = resample_poly(segment, 16000 // divisor, rate // divisor)
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    power = np.abs(np.fft.rfft(frames * np.hanning(400), n=512)) ** 2
    hz = np.fft.rfftfreq(512, 1 / 16000)
    mel_edges = np.linspace(2595 * np.log10(1 + 100 / 700),
                            2595 * np.log10(1 + 7500 / 700), 42)
    edges = 700 * (10 ** (mel_edges / 2595) - 1)
    rising = (hz[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    falling = (edges[2:, None] - hz[None, :]) / np.diff(edges)[1:, None]
    filters = np.maximum(0, np.minimum(rising, falling))
    mel = power @ filters.T
    # Relative floor and omission of c0 remove global gain, not impactor variation.
    cepstra = dct(np.log(np.maximum(mel, power.max() * 1e-12)),
                   type=2, axis=1, norm="ortho")[:, 1:14]
    return np.concatenate((cepstra.mean(axis=0), cepstra.std(axis=0)))


def analyze_wav(path):
    """Analyze one tap with >=50 ms quiet lead-in and >=250 ms after onset.

    Thresholds/windows are exploratory capture heuristics, not validated defect
    criteria. Energy times describe this fixed window, not T20/EDT or RT60.
    """
    rate, x = _read_pcm(path)
    block_size = max(1, round(rate * 0.002))
    block_power = np.mean(x[:len(x) // block_size * block_size].reshape(-1, block_size) ** 2,
                          axis=1)
    peak_power = float(block_power.max())
    if peak_power < 1e-10:
        raise ValueError("No usable impact: recording is silent or too quiet")
    baseline = float(np.mean(block_power[:round(0.05 * rate / block_size)]))
    if peak_power < max(baseline * 100, 1e-10):
        raise ValueError("No clear impact above the first 50 ms noise baseline")
    onset_block = int(np.flatnonzero(block_power >= max(peak_power * 0.05, baseline * 100))[0])
    onset = onset_block * block_size
    if onset < round(rate * 0.05):
        raise ValueError("Impact starts too early: include at least 50 ms quiet lead-in")
    size = round(rate * 0.25)
    if onset + size > len(x):
        raise ValueError("Impact ends too late: include at least 250 ms after onset")
    segment = x[onset:onset + size].copy()
    segment -= segment.mean()
    energy = segment ** 2
    total = float(energy.sum())
    cumulative = np.cumsum(energy) / total
    t50, t90 = (np.searchsorted(cumulative, q) / rate for q in (0.5, 0.9))
    tail_ratio = float(energy[-round(rate * 0.025):].sum() / total)
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(size))) ** 2
    frequencies = np.fft.rfftfreq(size, 1 / rate)
    mask = (frequencies >= 100) & (frequencies < 7500)
    power = spectrum[mask]
    hz = frequencies[mask]
    power_sum = float(power.sum())
    if power_sum <= np.finfo(float).tiny:
        raise ValueError("No usable spectral energy between 100 and 7500 Hz")
    centroid = float(np.dot(power, hz) / power_sum)
    bandwidth = float(np.sqrt(np.dot(power, (hz - centroid) ** 2) / power_sum))
    peaks, _ = find_peaks(power)
    peak_index = int(peaks[np.argmax(power[peaks])]) if len(peaks) else int(np.argmax(power))
    flatness = float(np.exp(np.mean(np.log(np.maximum(power, power.max() * 1e-12))))
                     / power.mean())
    bands = [(100, 500), (500, 1500), (1500, 4000), (4000, 7500)]
    ratios = [float(power[(hz >= lo) & (hz < hi)].sum() / power_sum) for lo, hi in bands]
    normalized = segment / np.sqrt(total / size)
    values = [float(hz[peak_index]), centroid, bandwidth, flatness, *ratios,
              float(t50), float(t90), tail_ratio, *_mfcc(normalized, rate).tolist()]
    if not np.isfinite(values).all():
        raise ValueError("Feature extraction produced nonfinite values")
    return {
        "sample_rate": int(rate),
        "duration_s": len(x) / rate,
        "onset_s": onset / rate,
        "feature_version": FEATURE_VERSION,
        "features": dict(zip(FEATURE_NAMES, values, strict=True)),
        "quality": {
            "impact_to_baseline_db": float(10 * np.log10(peak_power / max(baseline, 1e-20))),
            "analysis_window_s": 0.25,
            "tail_energy_ratio": tail_ratio,
            "warnings": ["Ring-down may exceed the 250 ms analysis window"] if tail_ratio > 0.02 else [],
        },
    }
