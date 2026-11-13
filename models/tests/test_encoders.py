import numpy as np
import pytest

from thud.encoders import (
    EFFICIENTAT_CONTRACT,
    YAMNET_CONTRACT,
    prepare_waveform,
    validate_encoder_output,
    wrap_loaded_encoder,
)


def test_yamnet_resamples_48khz_capture_to_16khz_without_mutating_input():
    capture = np.zeros(48_000, dtype=np.float64)
    capture[100] = 1

    prepared = prepare_waveform(capture, 48_000, YAMNET_CONTRACT)

    assert prepared.dtype == np.float32
    assert prepared.shape == (16_000,)
    assert capture.dtype == np.float64
    assert capture[100] == 1
    assert prepared is not capture
    assert YAMNET_CONTRACT.sample_rate_hz == 16_000


def test_efficientat_frontend_contract_is_explicit():
    assert EFFICIENTAT_CONTRACT.sample_rate_hz == 32_000
    assert EFFICIENTAT_CONTRACT.frontend == {
        "mel_bins": 128,
        "window_samples": 800,
        "hop_samples": 320,
    }
    assert EFFICIENTAT_CONTRACT.outputs["logits"].dimension == 527


def test_yamnet_scores_and_embeddings_are_distinct_output_kinds():
    scores = np.zeros((3, 521), dtype=np.float32)
    embeddings = np.zeros((3, 1024), dtype=np.float32)

    assert validate_encoder_output(scores, YAMNET_CONTRACT, "scores") is scores
    assert (
        validate_encoder_output(embeddings, YAMNET_CONTRACT, "embeddings")
        is embeddings
    )
    with pytest.raises(ValueError, match="scores.*521"):
        validate_encoder_output(embeddings, YAMNET_CONTRACT, "scores")
    with pytest.raises(ValueError, match="embeddings.*1024"):
        validate_encoder_output(scores, YAMNET_CONTRACT, "embeddings")


@pytest.mark.parametrize(
    "contract,kind,shape",
    [
        (YAMNET_CONTRACT, "scores", (1, 520)),
        (YAMNET_CONTRACT, "embeddings", (1, 1025)),
        (EFFICIENTAT_CONTRACT, "logits", (1, 526)),
        (EFFICIENTAT_CONTRACT, "logits", (527,)),
        (EFFICIENTAT_CONTRACT, "logits", (0, 527)),
    ],
)
def test_wrong_pretrained_output_shapes_are_rejected(contract, kind, shape):
    with pytest.raises(ValueError, match="output must have shape"):
        validate_encoder_output(np.zeros(shape, dtype=np.float32), contract, kind)


def test_loaded_callable_is_wrapped_and_input_is_resampled():
    seen = []

    def model(waveform):
        seen.append(waveform)
        return np.zeros((1, 1024), dtype=np.float32)

    wrapped = wrap_loaded_encoder(model, YAMNET_CONTRACT, "embeddings")
    result = wrapped(np.zeros(48_000), sample_rate_hz=48_000)

    assert result.shape == (1, 1024)
    assert seen[0].shape == (16_000,)
    assert seen[0].dtype == np.float32


def test_unknown_output_kind_is_rejected_before_calling_model():
    called = False

    def model(_):
        nonlocal called
        called = True
        return np.zeros((1, 521), dtype=np.float32)

    with pytest.raises(ValueError, match="Unknown yamnet output kind"):
        wrap_loaded_encoder(model, YAMNET_CONTRACT, "features")
    assert not called
