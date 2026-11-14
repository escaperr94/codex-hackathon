"""Real Sound-of-Water waveform fixture parity for bundled joint YAMNet.

Run: python -m pytest models/tests/test_audio_events.py
Fixture provenance: experiments/finalize_research.py:waveform_encoder, first,
middle, and final 975ms windows of an actual Sound-of-Water recording.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_joint_encoder_preserves_embeddings_and_official_events():
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema

    path = ROOT / 'assets/models/yamnet-waveform-embedding-events-f32.tflite'
    old = ROOT / 'models/artifacts/final/research/yamnet-waveform-embedding-f32.tflite'
    saved = ROOT / 'models/artifacts/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e'
    fixture = ROOT / 'models/artifacts/final/research/yamnet-waveform-embedding-f32.parity.npz'
    model = tf.saved_model.load(str(saved))
    current = tf.lite.Interpreter(model_path=str(path))
    previous = tf.lite.Interpreter(model_path=str(old))
    for interpreter in (current, previous):
        interpreter.allocate_tensors()
    assert tuple(current.get_input_details()[0]['shape']) == (15600,)
    assert tuple(current.get_output_details()[0]['shape']) == (1545,)
    flat = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    assert all(flat.OperatorCodes(i).BuiltinCode() != schema.BuiltinOperator.CUSTOM
               for i in range(flat.OperatorCodesLength()))
    with np.load(fixture) as data:
        assert data['inputs'].shape == (3, 15600)
        for waveform, reference in zip(data['inputs'], data['reference']):
            outputs = []
            for interpreter in (current, previous):
                interpreter.set_tensor(interpreter.get_input_details()[0]['index'], waveform)
                interpreter.invoke()
                outputs.append(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))
            np.testing.assert_allclose(outputs[0][:1024], outputs[1], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(outputs[0][:1024], reference, rtol=3e-3, atol=3e-4)
            scores = np.asarray(model(waveform)[0]).max(axis=0)
            np.testing.assert_allclose(outputs[0][1024:], scores, rtol=3e-3, atol=3e-4)
            assert np.isfinite(outputs[0]).all()
            assert ((outputs[0][1024:] >= 0) & (outputs[0][1024:] <= 1)).all()
