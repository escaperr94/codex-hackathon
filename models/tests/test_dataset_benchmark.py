import numpy as np
import soundfile as sf

from experiments.dataset_benchmark import feature


def test_spectral_features_are_finite_and_separate_frequency(tmp_path):
    rate = 16000
    t = np.arange(rate) / rate
    values = []
    for hz in (150, 3000):
        path = tmp_path / f"tone-{hz}.wav"
        sf.write(path, 0.4 * np.sin(2 * np.pi * hz * t), rate, subtype="FLOAT")
        values.append(feature(path))
    assert all(x.shape == (90,) and np.isfinite(x).all() for x in values)
    # Low/high-frequency inputs must not collapse after cepstral summarization.
    assert np.argmax(values[0][26:58]) < np.argmax(values[1][26:58])


def test_stereo_audio_matches_mono_features(tmp_path):
    rate = 8000
    t = np.arange(rate) / rate
    x = (0.3 * np.sin(2 * np.pi * 900 * t)).astype(np.float32)
    mono, stereo = tmp_path / "mono.wav", tmp_path / "stereo.wav"
    sf.write(mono, x, rate, subtype="FLOAT")
    sf.write(stereo, np.c_[x, x], rate, subtype="FLOAT")
    np.testing.assert_allclose(feature(mono), feature(stereo), atol=1e-5)
