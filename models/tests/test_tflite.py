from pathlib import Path

import numpy as np
import pytest

from thud.transfer import fit_transfer_head, predict_transfer_head


@pytest.fixture(scope="module")
def saved_model():
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    models = list((Path(__file__).resolve().parents[1] / "artifacts/tfhub").glob("*/saved_model.pb"))
    if not models:
        pytest.skip("Requires cached official YAMNet SavedModel")
    return models[0].parent

@pytest.mark.parametrize("task", ["tile", "fill"])
def test_export_preserves_waveform_predictions(saved_model, tmp_path, task):
    from thud.pretrained import load_yamnet, yamnet_infer
    from thud.tflite import INPUT_SAMPLES, export_yamnet_tflite, run_tflite

    t = np.arange(INPUT_SAMPLES) / 16000
    waves = [(0.2 * np.sin(2 * np.pi * f * t) * np.exp(-4 * t)).astype(np.float32)
             for f in (300, 600, 1200, 1800)]
    model = load_yamnet(saved_model)
    embeddings = np.stack([yamnet_infer(model, wave)[1].mean(axis=0) for wave in waves])
    labels = ["bonded", "bonded", "debonded", "debonded"] if task == "tile" else [0., 25., 75., 100.]
    head = fit_transfer_head(embeddings, labels, task, family="jar" if task == "fill" else None)
    path = tmp_path / "model.tflite"
    export_yamnet_tflite(saved_model, path, head=head)
    scores = np.array([run_tflite(path, wave).item() for wave in waves])
    reference = ((embeddings - head["mean"]) / head["scale"]) @ head["coefficients"] + head["intercept"]
    np.testing.assert_allclose(scores, reference, atol=0.002, rtol=0.005)
    if task == "tile":
        assert np.where(scores > 0, "debonded", "bonded").tolist() == predict_transfer_head(head, embeddings).tolist()
    with pytest.raises(FileExistsError):
        export_yamnet_tflite(saved_model, path, head=head)
    for invalid in (waves[0][:-1], np.full(INPUT_SAMPLES, np.nan), waves[0] + 2):
        with pytest.raises(ValueError, match="waveform"):
            run_tflite(path, invalid)
