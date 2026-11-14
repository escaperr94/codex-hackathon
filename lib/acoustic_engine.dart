import 'dart:math' as math;
import 'dart:typed_data';

enum AcousticMode { impact, operating }

/// AudioSet event-classifier evidence, not a voice heuristic or a physical diagnosis.
class SpeechEvidence {
  final double score;
  final int windows;
  final String classifier;
  SpeechEvidence(this.score, this.windows,
      {this.classifier = 'Official YAMNet AudioSet speech-family maximum'}) {
    if (!score.isFinite || score < 0 || score > 1 || windows < 1) {
      throw ArgumentError('Invalid speech classifier evidence');
    }
  }
  // Conservative capture-quality policy, not a calibrated speech probability.
  bool get contaminated => score >= .30;
  Map<String, Object> toJson() => {
    'score': score, 'windows': windows, 'classifier': classifier,
    'threshold': .30,
  };
}

class ImpactWindow {
  final int onset, attackStart, attackEnd, ringStart, ringEnd;
  final bool duplicate;
  final Map<String, double> features;
  ImpactWindow(this.onset, this.attackStart, this.attackEnd, this.ringStart,
      this.ringEnd, this.duplicate, this.features);
  Map<String, Object> toJson() => {
    'onset': onset, 'attackStart': attackStart, 'attackEnd': attackEnd,
    'ringStart': ringStart, 'ringEnd': ringEnd, 'duplicate': duplicate,
    'features': features,
  };
}

class AcousticAnalysis {
  final AcousticMode mode;
  final List<ImpactWindow> impacts;
  final Map<String, double> aggregate, dispersion;
  final List<List<double>> stft;
  final List<double> mfcc, modalPeaksHz;
  final List<String> _signalIssues;
  final SpeechEvidence? speechEvidence;
  final double snrDb, clippedFraction, _uncertainty;
  const AcousticAnalysis({required this.mode, required this.impacts,
    required this.aggregate, required this.dispersion, required this.stft,
    required this.mfcc, required this.modalPeaksHz,
    required List<String> signalIssues, required this.snrDb,
    required this.clippedFraction, required double uncertainty,
    this.speechEvidence}) : _signalIssues = signalIssues, _uncertainty = uncertainty;
  double get centroidHz => aggregate['centroidHz'] ?? 0;
  double get flatness => aggregate['flatness'] ?? 0;
  double get decayMs => aggregate['decayMs'] ?? 0;
  /// Dimensionless damping ratio from an exponential amplitude fit and dominant mode.
  double get damping => aggregate['damping'] ?? 0;
  bool get signalUsable => !_signalIssues.any((s) => s.startsWith('Reject:'));
  bool get usable => signalUsable && speechEvidence != null && !speechEvidence!.contaminated;
  List<String> get qualityIssues => List.unmodifiable([
    ..._signalIssues,
    if (speechEvidence == null) 'Reject: speech classifier evidence unavailable',
    if (speechEvidence?.contaminated ?? false) 'Reject: speech contamination (YAMNet event classifier)',
  ]);
  List<String> get issues => qualityIssues;
  double get uncertainty => usable ? _uncertainty : 1;
  AcousticAnalysis withSpeechEvidence(SpeechEvidence evidence) => AcousticAnalysis(
    mode: mode, impacts: impacts, aggregate: aggregate, dispersion: dispersion,
    stft: stft, mfcc: mfcc, modalPeaksHz: modalPeaksHz,
    signalIssues: _signalIssues, snrDb: snrDb, clippedFraction: clippedFraction,
    uncertainty: _uncertainty, speechEvidence: evidence);
  Map<String, Object?> toJson() => {
    'mode': mode.name, 'stft': stft, 'centroidHz': centroidHz,
    'flatness': flatness, 'mfcc': mfcc, 'modalPeaksHz': modalPeaksHz,
    'damping': damping, 'decayMs': decayMs,
    'impacts': impacts.map((i) => i.toJson()).toList(),
    'aggregate': aggregate, 'dispersion': dispersion, 'snrDb': snrDb,
    'clippedFraction': clippedFraction, 'qualityIssues': qualityIssues,
    'usable': usable, 'uncertainty': uncertainty,
    'speechEvidence': speechEvidence?.toJson(),
    'uncertaintyMeaning': 'capture repeatability/quality index, not calibrated diagnostic probability',
    'stftMeaning': 'time rows; 64 linear frequency bands, relative power -80..0 dB',
  };
}

/// Bounded to 30 seconds, 32 detected impacts, 64 STFT rows and radix-2 FFTs.
/// Invalid PCM throws; inadequate evidence returns an explicit rejected result.
AcousticAnalysis analyzeAcoustics(List<double> pcm, {
  int sampleRate = 48000, AcousticMode mode = AcousticMode.impact,
  SpeechEvidence? speechEvidence,
}) {
  if (sampleRate < 16000 || sampleRate > 192000 ||
      pcm.length < sampleRate ~/ 10 || pcm.length > sampleRate * 30) {
    throw ArgumentError('Expected 0.1–30 seconds of 16–192 kHz mono PCM');
  }
  var mean = 0.0, clipped = 0;
  for (final x in pcm) {
    if (!x.isFinite || x.abs() > 1) throw ArgumentError('PCM must be finite in [-1,1]');
    mean += x;
    if (x.abs() >= .999) clipped++;
  }
  mean /= pcm.length;
  final hop = math.max(1, sampleRate ~/ 200).toInt();
  final envelope = <double>[];
  var totalEnergy = 0.0;
  for (var start = 0; start + hop <= pcm.length; start += hop) {
    var e = 0.0;
    for (var i = start; i < start + hop; i++) {
      final x = pcm[i] - mean;
      e += x * x;
    }
    envelope.add(e / hop);
    totalEnergy += e;
  }
  final ordered = List<double>.of(envelope)..sort();
  final quietCount = math.max(1, ordered.length ~/ 5).toInt();
  var noise = 0.0;
  for (var i = 0; i < quietCount; i++) { noise += ordered[i]; }
  noise = math.max(1e-12, noise / quietCount);
  final peakPower = ordered.last;
  final snr = _db(math.max(1e-12, peakPower - noise) / noise).clamp(-100.0, 120.0).toDouble();
  final issues = <String>[];
  if (clipped > 0) issues.add('Reject: clipping detected');
  if (totalEnergy / pcm.length < 1e-10) issues.add('Reject: no usable audio');
  if (mode == AcousticMode.impact && snr < 15) issues.add('Reject: low impact signal-to-noise ratio');
  if (mode == AcousticMode.operating) {
    // A steady machine is not background noise; a single take cannot separate the two.
    issues.add('Operating sound: SNR requires a separate background recording');
    if (_db(totalEnergy / pcm.length) < -65) issues.add('Reject: operating sound too quiet');
    if (pcm.length < sampleRate) issues.add('Reject: capture at least one second of operating sound');
  }
  final onsets = <int>[];
  for (var i = 0; i < envelope.length; i++) {
    var baseline = noise;
    if (i > 0) {
      final past = envelope.sublist(math.max(0, i - 12), i)..sort();
      baseline = math.max(noise, past[past.length ~/ 2]);
    }
    final previous = i == 0 ? noise : envelope[i - 1];
    if (envelope[i] > math.max(noise * 10, baseline * 4) &&
        envelope[i] > previous * 2 &&
        (onsets.isEmpty || i - onsets.last >= 8)) {
      onsets.add(i);
      if (onsets.length == 32) { issues.add('Reject: too many impacts; shorten the recording'); break; }
    }
  }
  final impacts = <ImpactWindow>[];
  final features = <_Features>[];
  if (mode == AcousticMode.impact) {
    for (var i = 0; i < onsets.length; i++) {
      final frame = onsets[i];
      final duplicate = (i > 0 && frame - onsets[i - 1] < 24) ||
          (i + 1 < onsets.length && onsets[i + 1] - frame < 24);
      final attackStart = math.max(0, (frame - 1) * hop).toInt();
      final attackEnd = math.min(pcm.length, (frame + 4) * hop).toInt();
      final ringEnd = math.min(pcm.length, math.min((frame + 240) * hop,
          i + 1 < onsets.length ? (onsets[i + 1] - 1) * hop : pcm.length)).toInt();
      final f = _extract(pcm, sampleRate, mean, frame * hop,
          math.max(attackEnd, ringEnd).toInt(), envelope, frame, noise);
      impacts.add(ImpactWindow(frame * hop, attackStart, attackEnd,
          attackEnd, math.max(attackEnd, ringEnd).toInt(), duplicate, f.values));
      if (!duplicate) features.add(f);
    }
    if (impacts.any((i) => i.duplicate)) issues.add('Closely spaced duplicate impacts excluded; separate taps by at least 120 ms');
    if (features.isEmpty) issues.add('Reject: no isolated impact detected');
    if (features.isNotEmpty && features.every((f) => f.values['decayMs'] == 0)) {
      issues.add('20 dB decay not resolved above background or before the next impact');
    }
  } else {
    // Analyze distributed operating windows rather than inventing an impact/decay.
    final count = math.min(8, math.max(1, pcm.length ~/ sampleRate)).toInt();
    for (var i = 0; i < count; i++) {
      final start = count == 1 ? 0 : i * (pcm.length - math.min(pcm.length, sampleRate)) ~/ (count - 1);
      features.add(_extract(pcm, sampleRate, mean, start,
          math.min(pcm.length, start + sampleRate), const [], 0, noise));
    }
  }
  final aggregate = <String, double>{}, dispersion = <String, double>{};
  if (features.isNotEmpty) {
    for (final key in features.first.values.keys) {
      final values = features.map((f) => f.values[key]!).toList();
      final median = _median(values);
      aggregate[key] = median;
      dispersion[key] = _median(values.map((x) => (x - median).abs()).toList());
    }
  }
  final mfcc = List<double>.generate(13, (i) => features.isEmpty ? 0 : _median(features.map((f) => f.mfcc[i]).toList()));
  final representative = features.isEmpty ? null : features.reduce((a, b) =>
      (a.values['centroidHz']! - (aggregate['centroidHz'] ?? 0)).abs() <
      (b.values['centroidHz']! - (aggregate['centroidHz'] ?? 0)).abs() ? a : b);
  final relativeSpread = (dispersion['centroidHz'] ?? 0) / math.max(1, aggregate['centroidHz'] ?? 1);
  if (relativeSpread > .25) issues.add('Reject: inconsistent repeated acoustic signatures');
  final uncertainty = (.15 + .35 / math.sqrt(math.max(1, features.length)) +
      relativeSpread + (mode == AcousticMode.impact && snr < 25 ? .2 : 0)).clamp(0.0, 1.0).toDouble();
  return AcousticAnalysis(mode: mode, impacts: List.unmodifiable(impacts),
    aggregate: Map.unmodifiable(aggregate), dispersion: Map.unmodifiable(dispersion),
    stft: _spectrogram(pcm, mean), mfcc: List.unmodifiable(mfcc),
    modalPeaksHz: representative?.peaks ?? const [], signalIssues: List.unmodifiable(issues),
    snrDb: snr, clippedFraction: clipped / pcm.length, uncertainty: uncertainty,
    speechEvidence: speechEvidence);
}

class _Features {
  final Map<String, double> values;
  final List<double> mfcc, peaks;
  _Features(this.values, this.mfcc, this.peaks);
}

_Features _extract(List<double> pcm, int rate, double mean, int start, int end,
    List<double> envelope, int frame, double noise) {
  const n = 4096;
  final powers = _power(pcm, start, math.min(end, start + n), mean, n);
  var energy = 0.0, weighted = 0.0, logarithm = 0.0, low = 0.0, high = 0.0;
  final first = math.max(1, (40 * n / rate).ceil()).toInt();
  final last = math.min(powers.length - 2, (10000 * n / rate).floor()).toInt();
  for (var i = first; i <= last; i++) {
    final p = math.max(1e-20, powers[i]);
    final hz = i * rate / n;
    energy += p; weighted += p * hz; logarithm += math.log(p);
    if (hz >= 450 && hz < 2200) low += p;
    if (hz >= 2200 && hz <= 6500) high += p;
  }
  final peakBins = <int>[];
  for (var i = first; i <= last; i++) {
    if (powers[i] > powers[i - 1] && powers[i] >= powers[i + 1]) peakBins.add(i);
  }
  peakBins.sort((a, b) => powers[b].compareTo(powers[a]));
  final peaks = <double>[];
  for (final bin in peakBins) {
    final hz = bin * rate / n;
    if (peaks.every((p) => (p - hz).abs() >= 60) && powers[bin] > energy * .005) peaks.add(hz);
    if (peaks.length == 5) break;
  }
  var decayMs = 0.0, damping = 0.0;
  if (envelope.isNotEmpty) {
    final limit = math.min(envelope.length, end ~/ (rate ~/ 200)).toInt();
    var peak = frame;
    for (var i = frame; i < math.min(limit, frame + 5).toInt(); i++) {
      if (envelope[i] > envelope[peak]) peak = i;
    }
    final signal = math.max(1e-12, envelope[peak] - noise);
    var sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, count = 0;
    for (var i = peak + 1; i < limit; i++) {
      final p = envelope[i] - noise;
      final level = _db(math.max(1e-20, p) / signal);
      if (decayMs == 0 && level <= -20 && i + 2 < limit &&
          envelope[i + 1] - noise <= signal * .01 && envelope[i + 2] - noise <= signal * .01 && signal / noise >= 200) {
        decayMs = (i - peak) * 5.0;
      }
      if (level <= -5 && level >= -25 && p > noise * 3) {
        final t = (i - peak) * .005;
        sx += t; sy += level; sxx += t * t; sxy += t * level; count++;
      }
    }
    final denominator = count * sxx - sx * sx;
    if (count >= 5 && denominator > 1e-12 && peaks.isNotEmpty) {
      final slope = (count * sxy - sx * sy) / denominator;
      if (slope < 0) damping = -slope * math.ln10 / (20 * 2 * math.pi * peaks.first);
    }
  }
  final mel = List<double>.filled(26, 0);
  final maxMel = 2595 * math.log(1 + math.min(10000, rate / 2) / 700) / math.ln10;
  final edges = List<double>.generate(28, (i) =>
      (700 * (math.pow(10, maxMel * i / 27 / 2595) - 1)).toDouble());
  for (var m = 0; m < 26; m++) {
    for (var bin = math.max(1, (edges[m] * n / rate).floor()).toInt();
        bin < powers.length && bin * rate / n <= edges[m + 2]; bin++) {
      final hz = bin * rate / n;
      final weight = hz <= edges[m + 1]
          ? (hz - edges[m]) / (edges[m + 1] - edges[m])
          : (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
      mel[m] += powers[bin] * math.max(0.0, weight);
    }
    // Normalize amplitude for shape/MFCC; retain digital amplitude separately.
    mel[m] = math.log(math.max(1e-12, mel[m] / math.max(1e-20, energy)));
  }
  final mfcc = List<double>.generate(13, (k) {
    var sum = 0.0;
    for (var m = 0; m < mel.length; m++) {
      sum += mel[m] * math.cos(math.pi * k * (m + .5) / mel.length);
    }
    return sum * math.sqrt((k == 0 ? 1 : 2) / mel.length);
  });
  return _Features({'centroidHz': weighted / math.max(1e-20, energy),
    'flatness': (math.exp(logarithm / (last - first + 1)) /
        math.max(1e-20, energy / (last - first + 1))).clamp(0.0, 1.0).toDouble(),
    'peakHz': peaks.isEmpty ? 0 : peaks.first,
    'hiiDb': _db(math.max(1e-20, low) / math.max(1e-20, high)),
    'decayMs': decayMs, 'damping': damping}, mfcc, peaks);
}

List<List<double>> _spectrogram(List<double> pcm, double mean) {
  const n = 1024;
  final rows = math.min(64, math.max(1, (pcm.length - n) ~/ 512 + 1)).toInt();
  final result = <List<double>>[];
  var maximum = 1e-20;
  for (var row = 0; row < rows; row++) {
    final start = rows == 1 ? 0 : row * math.max(0, pcm.length - n) ~/ (rows - 1);
    final powers = _power(pcm, start, math.min(pcm.length, start + n), mean, n);
    final bands = List<double>.filled(64, 0);
    for (var bin = 1; bin < powers.length; bin++) { bands[math.min(63, (bin - 1) ~/ 8)] += powers[bin]; }
    for (final p in bands) { maximum = math.max(maximum, p); }
    result.add(bands);
  }
  return result.map((r) => r.map((p) => ((_db(p / maximum) + 80) / 80).clamp(0.0, 1.0).toDouble()).toList(growable: false)).toList(growable: false);
}

Float64List _power(List<double> pcm, int start, int end, double mean, int n) {
  final real = Float64List(n), imag = Float64List(n);
  for (var i = 0; i < n && start + i < end; i++) {
    real[i] = (pcm[start + i] - mean) * (.5 - .5 * math.cos(2 * math.pi * i / (n - 1)));
  }
  _fft(real, imag);
  final power = Float64List(n ~/ 2 + 1);
  for (var i = 0; i < power.length; i++) { power[i] = (real[i] * real[i] + imag[i] * imag[i]) / (n * n); }
  return power;
}

double _median(List<double> values) {
  values.sort();
  final middle = values.length ~/ 2;
  return values.length.isOdd ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
double _db(double p) => 10 * math.log(math.max(1e-20, p)) / math.ln10;

// Same in-place radix-2 transform as the capture analyzer, with bounded FFT sizes.
void _fft(Float64List real, Float64List imag) {
  final n = real.length;
  for (var i = 1, j = 0; i < n; i++) {
    var bit = n >> 1;
    while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      final r = real[i]; real[i] = real[j]; real[j] = r;
      final im = imag[i]; imag[i] = imag[j]; imag[j] = im;
    }
  }
  for (var size = 2; size <= n; size <<= 1) {
    final angle = -2 * math.pi / size;
    final stepR = math.cos(angle), stepI = math.sin(angle);
    for (var start = 0; start < n; start += size) {
      var wr = 1.0, wi = 0.0;
      for (var j = 0; j < size ~/ 2; j++) {
        final a = start + j, b = a + size ~/ 2;
        final tr = wr * real[b] - wi * imag[b], ti = wr * imag[b] + wi * real[b];
        real[b] = real[a] - tr; imag[b] = imag[a] - ti;
        real[a] += tr; imag[a] += ti;
        final next = wr * stepR - wi * stepI;
        wi = wr * stepI + wi * stepR; wr = next;
      }
    }
  }
}
