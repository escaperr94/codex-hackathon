import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import 'audio_encoder_preprocessing.dart';

class AudioEncoder {
  Future<tfl.Interpreter>? _interpreter;
  Future<void>? _disposal;
  bool _disposed = false;

  Future<tfl.Interpreter> _load() async {
    final interpreter = await tfl.Interpreter.fromAsset(audioEncoderAsset);
    try {
      final inputs = interpreter.getInputTensors();
      final outputs = interpreter.getOutputTensors();
      if (inputs.length != 1 || outputs.length != 1 ||
          inputs.single.type != tfl.TensorType.float32 ||
          outputs.single.type != tfl.TensorType.float32 ||
          inputs.single.shape.length != 1 ||
          inputs.single.shape.single != audioEncoderWindowLength ||
          outputs.single.shape.length != 1 ||
          outputs.single.shape.single != audioEmbeddingLength) {
        throw StateError('YAMNet requires float32[15600] → float32[1024]');
      }
      return interpreter;
    } catch (_) {
      interpreter.close();
      rethrow;
    }
  }

  Future<List<double>> encode(List<double> pcm, int sampleRate) async {
    if (_disposed) throw StateError('AudioEncoder has been disposed');
    final window = audioEncoderWindow(pcm, sampleRate);
    final interpreter = await (_interpreter ??= _load());
    if (_disposed) throw StateError('AudioEncoder has been disposed');
    interpreter.getInputTensor(0).data = window.buffer.asUint8List();
    interpreter.invoke();
    final bytes = interpreter.getOutputTensor(0).data;
    final embedding = Float32List.fromList(Float32List.view(
      bytes.buffer, bytes.offsetInBytes, audioEmbeddingLength,
    ));
    return checkedAudioEmbedding(embedding);
  }

  Future<void> dispose() => _disposal ??= _dispose();

  Future<void> _dispose() async {
    _disposed = true;
    final loading = _interpreter;
    if (loading != null) (await loading).close();
  }
}
