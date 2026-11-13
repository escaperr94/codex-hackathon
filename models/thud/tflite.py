"""Builtin-only TFLite export of official YAMNet and an optional frozen task head."""

from pathlib import Path

import numpy as np

from .pretrained import load_yamnet
from .transfer import validate_transfer_head

INPUT_SAMPLES = 15600  # 975 ms: one full 96-frame patch including the last STFT window.


def export_yamnet_tflite(checkpoint, output, *, head=None):
    """Export fixed 16 kHz mono float waveform -> embedding or task score.

    With a tile head, output is a raw decision margin (positive = debonded), not
    a calibrated probability. Fill output is unclamped volume percentage points.
    The 975 ms window must be applied consistently during training and inference.
    """
    import tensorflow as tf

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    if head is not None:
        validate_transfer_head(head)
        if head["encoder"] != "yamnet":
            raise ValueError("TFLite YAMNet export requires a YAMNet head")
    encoder = load_yamnet(checkpoint)

    class EdgeModel(tf.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            if head is not None:
                self.mean = tf.constant(head["mean"], dtype=tf.float32)
                self.scale = tf.constant(head["scale"], dtype=tf.float32)
                self.coefficients = tf.constant(head["coefficients"], dtype=tf.float32)
                self.intercept = tf.constant(head["intercept"], dtype=tf.float32)

        @tf.function(input_signature=[tf.TensorSpec([INPUT_SAMPLES], tf.float32, name="waveform")])
        def infer(self, waveform):
            _, embeddings, _ = self.encoder(waveform)
            embedding = tf.reduce_mean(embeddings, axis=0)
            if head is None:
                return {"embedding": embedding}
            margin = tf.reduce_sum(((embedding - self.mean) / self.scale) * self.coefficients) + self.intercept
            return {"score": tf.reshape(margin, [1])}

    module = EdgeModel()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.infer.get_concrete_function()], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    data = converter.convert()
    # Allocation exercises the exported tensor contract before publishing bytes.
    interpreter = tf.lite.Interpreter(model_content=data)
    interpreter.allocate_tensors()
    with destination.open("xb") as handle:
        handle.write(data)
    return {"output": str(destination), "bytes": len(data), "precision": "float32",
            "input_samples": INPUT_SAMPLES, "sample_rate": 16000, "operators": "TFLITE_BUILTINS",
            "output_kind": "embedding" if head is None else "tile_margin" if head["task"] == "tile" else "fill_percentage_points"}


def run_tflite(model_path, waveform):
    """Desktop interpreter check; not an iPhone performance measurement."""
    import tensorflow as tf

    audio = np.asarray(waveform)
    if audio.shape != (INPUT_SAMPLES,) or audio.dtype.kind not in "iuf" or not np.isfinite(audio).all():
        raise ValueError(f"TFLite waveform must be finite shape ({INPUT_SAMPLES},)")
    if np.max(np.abs(audio)) > 1:
        raise ValueError("TFLite waveform must be in [-1, 1]")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or tuple(inputs[0]["shape"]) != (INPUT_SAMPLES,) or inputs[0]["dtype"] != np.float32:
        raise ValueError("model does not implement the fixed YAMNet waveform contract")
    interpreter.set_tensor(inputs[0]["index"], audio.astype(np.float32))
    interpreter.invoke()
    result = interpreter.get_tensor(outputs[0]["index"])
    if not np.isfinite(result).all():
        raise ValueError("TFLite output must be finite")
    return result
