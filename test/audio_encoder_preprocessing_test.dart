import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:thud/audio_encoder_preprocessing.dart';

void main() {
  test('rejects malformed PCM and invalid rates before model loading', () {
    expect(() => audioEncoderWindow([], 48000), throwsArgumentError);
    expect(() => audioEncoderWindow([0], 0), throwsArgumentError);
    expect(() => audioEncoderWindow([0], 8000), throwsArgumentError);
    expect(() => audioEncoderWindow([double.nan], 48000), throwsArgumentError);
    expect(() => audioEncoderWindow([double.infinity], 48000), throwsArgumentError);
    expect(() => audioEncoderWindow([1.01], 48000), throwsArgumentError);
    expect(() => audioEncoderWindow(List.filled(480001, 0.0), 16000),
        throwsArgumentError);
  });

  test('centers first and final impacts and pads both boundaries', () {
    final first = audioEncoderWindow([1, 0, 0], 16000);
    expect(first.length, 15600);
    expect(first[7800], 1);
    expect(first.where((value) => value != 0).length, 1);
    final pcm = List<double>.filled(16001, 0)..last = -1;
    final last = audioEncoderWindow(pcm, 16000);
    expect(last[7800], -1);
    expect(last.where((value) => value != 0).length, 1);
    final short48k = audioEncoderWindow([1], 48000);
    expect(short48k.length, 15600);
    expect(short48k.first, 0);
    expect(short48k.last, 0);
    expect(short48k[7800], greaterThan(0));
    expect(short48k.every((value) => value.isFinite && value.abs() <= 1), isTrue);
  });

  test('selects the louder impact rather than the first one', () {
    final pcm = List<double>.filled(48000, 0);
    pcm[2000] = 0.2;
    pcm[40000] = -0.9;
    final result = audioEncoderWindow(pcm, 16000);
    expect(result[7800], closeTo(-0.9, 1e-6));
  });

  test('48k decimation preserves the passband and rejects aliasing', () {
    double rmsAt(double frequency, int rate) {
      final pcm = List<double>.generate(rate,
          (i) => 0.5 * math.sin(2 * math.pi * frequency * i / rate));
      pcm[rate ~/ 5] = 1.0;
      final result = audioEncoderWindow(pcm, rate);
      var power = 0.0;
      for (var i = 8300; i < 13800; i++) {
        power += result[i] * result[i];
      }
      return math.sqrt(power / 5500);
    }

    expect(rmsAt(1000, 48000), closeTo(math.sqrt(0.125), 0.01));
    expect(rmsAt(12000, 48000), lessThan(0.002));
    // Fractional-rate paths must retain the same sample-time convention.
    expect(rmsAt(1000, 44100), closeTo(math.sqrt(0.125), 0.01));
  });
}
