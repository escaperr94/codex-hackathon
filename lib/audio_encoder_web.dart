import 'dart:async';
import 'dart:js_interop';
import 'package:web/web.dart' as web;
import 'audio_encoder_preprocessing.dart';

class AudioEncoder {
  web.Worker? _worker;
  final _pending = <int, Completer<List<double>>>{};
  var _nextId = 0;
  var _disposed = false;

  void _start() {
    final worker = web.Worker(Uri.base.resolve('audio_encoder_worker.js').toString().toJS);
    _worker = worker;
    worker.onmessage = ((web.MessageEvent event) {
      final message = event.data.dartify() as List;
      final completer = _pending.remove(message[0]);
      if (completer == null) return;
      if (message[2] != null) {
        completer.completeError(StateError('YAMNet: ${message[2]}'));
      } else {
        try {
          final embedding = (message[1] as List).map((v) => (v as num).toDouble()).toList();
          completer.complete(checkedAudioEmbedding(embedding));
        } catch (error, stack) {
          completer.completeError(error, stack);
        }
      }
    }).toJS;
    worker.onerror = ((web.Event event) {
      final detail = (event as web.ErrorEvent).message;
      _fail(StateError('Could not load the bundled YAMNet WebAssembly runtime: $detail'));
      worker.terminate();
      _worker = null;
    }).toJS;
  }

  Future<List<double>> encode(List<double> pcm, int sampleRate) {
    if (_disposed) throw StateError('AudioEncoder has been disposed');
    final window = audioEncoderWindow(pcm, sampleRate);
    if (_worker == null) _start();
    final id = _nextId++;
    final completer = Completer<List<double>>();
    _pending[id] = completer;
    _worker!.postMessage([
      id,
      Uri.base.resolve('assets/$audioEncoderAsset').toString(),
      window,
      Uri.base.resolve('audio_encoder_runtime/').toString(),
    ].jsify());
    return completer.future;
  }

  void _fail(Object error) {
    for (final completer in _pending.values) {
      completer.completeError(error);
    }
    _pending.clear();
  }

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _worker?.terminate();
    _worker = null;
    _fail(StateError('AudioEncoder has been disposed'));
  }
}
