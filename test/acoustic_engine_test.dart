import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:thud/acoustic_engine.dart';
import 'package:thud/audio_encoder_preprocessing.dart';

const rate = 16000;
List<double> taps(List<double> seconds, {double amplitude = .6, double hz = 1000}) {
  final pcm = List<double>.generate(rate * 3, (i) => .00001 * math.sin(i * 1.234));
  for (final second in seconds) {
    final start = (second * rate).round();
    for (var i = 0; i < rate ~/ 2 && start + i < pcm.length; i++) {
      pcm[start + i] += amplitude * math.exp(-i / (rate * .04)) * math.cos(2 * math.pi * hz * i / rate);
    }
  }
  return pcm;
}

void main() {
  test('rejects malformed signals at the input boundary', () {
    expect(() => analyzeAcoustics([]), throwsArgumentError);
    expect(() => analyzeAcoustics(List.filled(5000, double.nan)), throwsArgumentError);
    expect(() => analyzeAcoustics(List.filled(5000, 1.1)), throwsArgumentError);
    expect(() => analyzeAcoustics(taps([.2]), sampleRate: 8000), throwsArgumentError);
    expect(analyzeAcoustics(List.filled(rate, 0), sampleRate: rate).signalUsable, isFalse);
  });
  test('onsets retain attack/ring windows and calibrated tone features', () {
    final result = analyzeAcoustics(taps([.2]), sampleRate: rate,
        speechEvidence: SpeechEvidence(.01, 4));
    expect(result.impacts, hasLength(1));
    final impact = result.impacts.single;
    expect(impact.onset / rate, closeTo(.2, .01));
    expect(impact.attackStart, lessThanOrEqualTo(impact.onset));
    expect(impact.ringEnd, greaterThan(impact.ringStart));
    expect(result.modalPeaksHz.first, closeTo(1000, 15));
    expect(result.decayMs, closeTo(92, 30));
    expect(result.damping, greaterThan(0));
    expect(result.mfcc, hasLength(13));
    expect(result.mfcc.every((x) => x.isFinite), isTrue);
    expect(result.flatness, lessThan(.1));
    expect(result.usable, isTrue);
    expect(result.stft.length, lessThanOrEqualTo(64));
    expect(result.stft.every((r) => r.length == 64 && r.every((v) => v >= 0 && v <= 1)), isTrue);
    expect(() => jsonEncode(result.toJson()), returnsNormally);
  });
  test('multiple isolated taps aggregate and near duplicates are excluded', () {
    final multi = analyzeAcoustics(taps([.2, 1.1, 2]), sampleRate: rate,
        speechEvidence: SpeechEvidence(.01, 4));
    expect(multi.impacts, hasLength(3));
    expect(multi.impacts.every((i) => !i.duplicate), isTrue);
    expect(multi.usable, isTrue);
    final doubleHit = analyzeAcoustics(taps([.2, .28]), sampleRate: rate);
    expect(doubleHit.impacts.where((i) => i.duplicate), hasLength(2));
    expect(doubleHit.signalUsable, isFalse);
  });
  test('clipping/noise reject and classifier evidence is mandatory', () {
    final clipped = taps([.2])..[3200] = 1;
    expect(analyzeAcoustics(clipped, sampleRate: rate).signalUsable, isFalse);
    final noise = List<double>.generate(rate, (i) => .01 * math.sin(i * 2.123));
    expect(analyzeAcoustics(noise, sampleRate: rate).signalUsable, isFalse);
    final clean = analyzeAcoustics(taps([.2]), sampleRate: rate);
    expect(clean.usable, isFalse);
    expect(clean.withSpeechEvidence(SpeechEvidence(.8, 4)).usable, isFalse);
    expect(clean.withSpeechEvidence(SpeechEvidence(.01, 4)).usable, isTrue);
  });
  test('operating mode accepts steady machine tone without fabricated decay', () {
    final tone = List<double>.generate(rate * 2, (i) => .1 * math.sin(2 * math.pi * 450 * i / rate));
    final result = analyzeAcoustics(tone, sampleRate: rate,
        mode: AcousticMode.operating, speechEvidence: SpeechEvidence(.01, 3));
    expect(result.usable, isTrue);
    expect(result.decayMs, 0);
    expect(result.damping, 0);
    expect(result.impacts, isEmpty);
    expect(result.qualityIssues.any((s) => s.contains('separate background')), isTrue);
  });
  test('speech scan covers between taps and median pools actual outputs', () async {
    var calls = 0;
    final result = await encodeAcousticRecording(taps([.2, 1.1, 2]), rate, (window) async {
      final call = calls++;
      final output = List<double>.filled(audioEncoderOutputLength, 0);
      for (var i = 0; i < audioEmbeddingLength; i++) { output[i] = call == 2 ? 100 : call.toDouble(); }
      if (call == 5) output[audioEmbeddingLength] = .9;
      return output;
    }, impactSamples: [3200, 17600, 32000]);
    expect(result.embedding.first, 1);
    expect(result.speechEvidence.contaminated, isTrue);
    expect(calls, 7); // Three impact windows plus four full-recording patches.
    expect(result.speechEvidence.windows, calls);
  });
  test('joint encoder rejects malformed classifier scores', () async {
    await expectLater(encodeAcousticRecording(taps([.2]), rate, (_) async =>
        List<double>.filled(audioEncoderOutputLength, double.nan)), throwsStateError);
  });
}
