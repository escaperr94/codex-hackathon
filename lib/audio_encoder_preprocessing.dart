import 'dart:math' as math;
import 'dart:typed_data';
import 'acoustic_engine.dart';

const embeddingVersion = 'yamnet-waveform-embedding-events-v2';
const audioEmbeddingLength = 1024;
const audioEncoderSampleRate = 16000;
const audioEncoderWindowLength = 15600;
const audioEncoderAsset = 'assets/models/yamnet-waveform-embedding-events-f32.tflite';
const audioEventCount = 521;
const audioEncoderOutputLength = audioEmbeddingLength + audioEventCount;

/// A 975 ms window centered on the strongest sample in the loudest 20 ms block.
/// PCM is normalized mono, not PCM16 integers. Missing edge samples are silence.
Float32List audioEncoderWindow(List<double> pcm, int sampleRate, {int? centerSample}) {
  if (sampleRate < 16000 || sampleRate > 192000) {
    throw ArgumentError.value(sampleRate, 'sampleRate', 'Expected 16–192 kHz');
  }
  if (pcm.isEmpty || pcm.length > sampleRate * 30) {
    throw ArgumentError('Expected nonempty PCM lasting at most 30 seconds');
  }
  for (final sample in pcm) {
    if (!sample.isFinite || sample < -1 || sample > 1) {
      throw ArgumentError('PCM must contain finite samples in [-1, 1]');
    }
  }
  return _audioEncoderWindow(pcm, sampleRate, centerSample: centerSample);
}

Float32List _audioEncoderWindow(List<double> pcm, int sampleRate, {int? centerSample}) {

  var impact = centerSample ?? 0;
  if (centerSample != null) {
    if (centerSample < 0 || centerSample >= pcm.length) {
      throw ArgumentError.value(centerSample, 'centerSample');
    }
  } else {
    final blockLength = sampleRate ~/ 50;
    var loudestEnergy = -1.0;
    for (var start = 0; start < pcm.length; start += blockLength) {
      final end = math.min(start + blockLength, pcm.length);
      var energy = 0.0;
      var peak = start;
      for (var i = start; i < end; i++) {
        energy += pcm[i] * pcm[i];
        if (pcm[i].abs() > pcm[peak].abs()) peak = i;
      }
      if (energy > loudestEnergy) {
        loudestEnergy = energy;
        impact = peak;
      }
    }
  }

  final output = Float32List(audioEncoderWindowLength);
  const middle = audioEncoderWindowLength ~/ 2;
  if (sampleRate == audioEncoderSampleRate) {
    for (var i = 0; i < output.length; i++) {
      final source = impact + i - middle;
      if (source >= 0 && source < pcm.length) output[i] = pcm[source];
    }
    return output;
  }

  // Blackman-windowed sinc polyphase low-pass; unlike linear interpolation,
  // frequencies above the new Nyquist limit are rejected before decimation.
  final divisor = sampleRate.gcd(audioEncoderSampleRate);
  final step = sampleRate ~/ divisor;
  final phases = audioEncoderSampleRate ~/ divisor;
  final radius = (32 * sampleRate / audioEncoderSampleRate).ceil();
  final cutoff = 0.45 * audioEncoderSampleRate / sampleRate;
  final kernels = List<Float64List?>.filled(phases, null);
  for (var i = 0; i < output.length; i++) {
    final numerator = (i - middle) * step;
    final sourceOffset = (numerator / phases).floor();
    final phase = numerator - sourceOffset * phases;
    var kernel = kernels[phase];
    if (kernel == null) {
      kernel = Float64List(radius * 2 + 1);
      final fraction = phase / phases;
      var total = 0.0;
      for (var tap = -radius; tap <= radius; tap++) {
        final distance = tap - fraction;
        final relative = distance / radius;
        if (relative.abs() > 1) continue;
        final angle = 2 * math.pi * cutoff * distance;
        final sinc = angle.abs() < 1e-12 ? 1.0 : math.sin(angle) / angle;
        final window = 0.42 +
            0.5 * math.cos(math.pi * relative) +
            0.08 * math.cos(2 * math.pi * relative);
        final weight = 2 * cutoff * sinc * window;
        kernel[tap + radius] = weight;
        total += weight;
      }
      for (var tap = 0; tap < kernel.length; tap++) {
        kernel[tap] /= total;
      }
      kernels[phase] = kernel;
    }
    final center = impact + sourceOffset;
    var value = 0.0;
    final first = math.max(-radius, -center);
    final last = math.min(radius, pcm.length - 1 - center);
    for (var tap = first; tap <= last; tap++) {
      value += pcm[center + tap] * kernel[tap + radius];
    }
    output[i] = value.clamp(-1.0, 1.0).toDouble();
  }
  return output;
}

List<double> checkedAudioEmbedding(List<double> embedding) {
  if (embedding.length != audioEmbeddingLength ||
      embedding.any((value) => !value.isFinite)) {
    throw StateError('YAMNet returned an invalid 1024-value embedding');
  }
  return embedding;
}

class AudioEncoding {
  final List<double> embedding;
  final SpeechEvidence speechEvidence;
  const AudioEncoding(this.embedding, this.speechEvidence);
}

/// Median pooling protects repeated-tap features from one unusual strike.
Future<AudioEncoding> encodeAcousticRecording(List<double> pcm, int sampleRate,
    Future<List<double>> Function(Float32List) infer,
    {List<int>? impactSamples}) async {
  // Validate the complete input once even when explicit impact windows are used.
  final primary = audioEncoderWindow(pcm, sampleRate);
  final centers = impactSamples == null || impactSamples.isEmpty
      ? null : impactSamples;
  if (centers != null && centers.length > 32) {
    throw ArgumentError('At most 32 impact embeddings per recording');
  }
  final embeddings = <List<double>>[];
  var maximumSpeech = 0.0;
  var windows = 0;
  void observe(List<double> output) {
    if (output.length != audioEncoderOutputLength ||
        output.any((x) => !x.isFinite) ||
        output.skip(audioEmbeddingLength).any((x) => x < 0 || x > 1)) {
      throw StateError('Invalid YAMNet embedding/event output');
    }
    // Official class map: Speech, Child speech, Conversation, Narration,
    // Babbling, Speech synthesizer, Shout..Screaming, Whispering (0..12).
    for (var event = 0; event <= 12; event++) {
      maximumSpeech = math.max(maximumSpeech, output[audioEmbeddingLength + event]);
    }
    windows++;
  }
  if (centers == null) {
    final output = await infer(primary);
    observe(output);
    embeddings.add(output.sublist(0, audioEmbeddingLength));
  } else {
    for (final center in centers) {
      final output = await infer(_audioEncoderWindow(pcm, sampleRate, centerSample: center));
      observe(output);
      embeddings.add(output.sublist(0, audioEmbeddingLength));
    }
  }
  // The impact-centered patch alone can miss a person talking between taps.
  final stride = (sampleRate * audioEncoderWindowLength / audioEncoderSampleRate).round();
  for (var start = 0; start < pcm.length; start += stride) {
    final center = math.min(pcm.length - 1, start + stride ~/ 2);
    observe(await infer(_audioEncoderWindow(pcm, sampleRate, centerSample: center)));
  }
  final pooled = List<double>.generate(audioEmbeddingLength, (i) {
    final values = embeddings.map((e) => e[i]).toList()..sort();
    final middle = values.length ~/ 2;
    return values.length.isOdd ? values[middle] : (values[middle - 1] + values[middle]) / 2;
  }, growable: false);
  return AudioEncoding(checkedAudioEmbedding(pooled), SpeechEvidence(maximumSpeech, windows));
}
