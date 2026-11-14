"""Export the cached official YAMNet; no training or network downloads.

The first 1024 values are unchanged mean embeddings; the final 521 values are
AudioSet event scores, for capture contamination only, not physical inference.
Run from repository root: python models/experiments/export_audio_events.py
"""
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'models/artifacts/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e'
DESTINATION = ROOT / 'assets/models/yamnet-waveform-embedding-events-f32.tflite'


class Model(tf.Module):
    def __init__(self):
        super().__init__()
        self.encoder = tf.saved_model.load(str(CHECKPOINT))

    @tf.function(input_signature=[tf.TensorSpec([15600], tf.float32, name='waveform')])
    def infer(self, waveform):
        scores, embeddings, _ = self.encoder(waveform)
        return {'embedding_events': tf.concat([
            tf.reduce_mean(embeddings, axis=0), tf.reduce_max(scores, axis=0)], axis=0)}


def export():
    model = Model()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([
        model.infer.get_concrete_function()], model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    DESTINATION.write_bytes(converter.convert())
    print(DESTINATION, DESTINATION.stat().st_size)


if __name__ == '__main__':
    export()
