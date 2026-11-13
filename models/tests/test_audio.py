import numpy as np
import pytest
from scipy.io import wavfile

from thud.audio import FEATURE_NAMES, analyze_wav


def write_tap(path, rate=48000, frequency=1200, amplitude=0.5, onset=0.12):
    t = np.arange(round(0.7 * rate)) / rate
    elapsed = np.maximum(t - onset, 0)
    x = amplitude * np.sin(2 * np.pi * frequency * elapsed) * np.exp(-elapsed / 0.045)
    x[t < onset] = 0
    wavfile.write(path, rate, x.astype(np.float32))
    return x


def test_frequency_and_gain_invariance_across_capture_rates(tmp_path):
    results = []
    for rate, amplitude in [(16000, 0.1), (44100, 0.7), (48000, 0.4)]:
        path = tmp_path / f"{rate}.wav"
        write_tap(path, rate=rate, amplitude=amplitude)
        analysis = analyze_wav(path)
        assert analysis["sample_rate"] == rate
        assert abs(analysis["onset_s"] - 0.12) <= 0.004
        assert analysis["features"]["peak_hz"] == pytest.approx(1200, abs=8)
        assert analysis["features"]["energy_t50_s"] < analysis["features"]["energy_t90_s"]
        results.append(analysis["features"])
    assert results[0]["band_500_1500"] > 0.99
    np.testing.assert_allclose(
        [results[0][name] for name in FEATURE_NAMES],
        [results[2][name] for name in FEATURE_NAMES], atol=0.1, rtol=0.02,
    )


@pytest.mark.parametrize("kind,match", [
    ("silent", "silent"), ("nonfinite", "nonfinite"), ("clipped", "full scale"),
    ("early", "early|baseline"), ("late", "late"), ("short", "0.4"),
    ("low_rate", "sample rate"), ("noise", "baseline"),
])
def test_unusable_recordings_are_rejected(tmp_path, kind, match):
    path = tmp_path / "bad.wav"
    x = write_tap(path, onset=0.01 if kind == "early" else 0.6 if kind == "late" else 0.12)
    rate = 48000
    if kind == "silent":
        x[:] = 0
    elif kind == "nonfinite":
        x[100] = np.nan
    elif kind == "clipped":
        x[10000] = 1
    elif kind == "short":
        x = x[:1000]
    elif kind == "low_rate":
        rate = 8000
    elif kind == "noise":
        x = np.random.default_rng(0).normal(0, 0.05, len(x))
    wavfile.write(path, rate, x.astype(np.float32))
    with pytest.raises(ValueError, match=match):
        analyze_wav(path)


def test_stereo_and_integer_pcm_match_mono(tmp_path):
    path = tmp_path / "mono.wav"
    x = write_tap(path)
    mono = analyze_wav(path)
    stereo = np.stack([x, x], axis=1)
    wavfile.write(path, 48000, (stereo * 32768).astype(np.int16))
    integer = analyze_wav(path)
    assert integer["features"]["peak_hz"] == mono["features"]["peak_hz"]
    assert integer["features"]["energy_t90_s"] == pytest.approx(mono["features"]["energy_t90_s"], abs=0.001)
