import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum Module { slab, vessel, scope }

class Measurement {
  const Measurement(
      {required this.id,
      required this.label,
      required this.source,
      required this.created,
      required this.module,
      required this.peakHz,
      required this.hiiDb,
      required this.decayMs,
      required this.rmsDb,
      required this.snrDb,
      required this.waveform,
      required this.spectrum,
      required this.flags,
      this.projectId,
      this.cell});

  final String id, label, source;
  final DateTime created;
  final Module module;
  final double peakHz, hiiDb, decayMs, rmsDb, snrDb;
  final List<double> waveform, spectrum;
  final List<String> flags;
  final String? projectId;
  final int? cell;

  Map<String, dynamic> toJson() => {
        'id': id,
        'label': label,
        'source': source,
        'created': created.toIso8601String(),
        'module': module.name,
        'peakHz': peakHz,
        'hiiDb': hiiDb,
        'decayMs': decayMs,
        'rmsDb': rmsDb,
        'snrDb': snrDb,
        'waveform': waveform,
        'spectrum': spectrum,
        'flags': flags,
        'projectId': projectId,
        'cell': cell,
      };

  factory Measurement.fromJson(Map<String, dynamic> value) => Measurement(
        id: value['id'] as String,
        label: value['label'] as String,
        source: value['source'] as String,
        created: DateTime.parse(value['created'] as String),
        module: Module.values.byName(value['module'] as String),
        peakHz: (value['peakHz'] as num).toDouble(),
        hiiDb: (value['hiiDb'] as num).toDouble(),
        decayMs: (value['decayMs'] as num).toDouble(),
        rmsDb: (value['rmsDb'] as num).toDouble(),
        snrDb: (value['snrDb'] as num).toDouble(),
        waveform: List<double>.unmodifiable(
            (value['waveform'] as List).map((e) => (e as num).toDouble())),
        spectrum: List<double>.unmodifiable(
            (value['spectrum'] as List).map((e) => (e as num).toDouble())),
        flags:
            List<String>.unmodifiable((value['flags'] as List).cast<String>()),
        projectId: value['projectId'] as String?,
        cell: value['cell'] as int?,
      );

  Measurement _assigned(String? projectId, int? cell) => Measurement(
        id: id,
        label: label,
        source: source,
        created: created,
        module: module,
        peakHz: peakHz,
        hiiDb: hiiDb,
        decayMs: decayMs,
        rmsDb: rmsDb,
        snrDb: snrDb,
        waveform: waveform,
        spectrum: spectrum,
        flags: flags,
        projectId: projectId,
        cell: cell,
      );
}

class Project {
  Project(
      {required this.id,
      required this.name,
      required this.room,
      required this.rows,
      required this.columns,
      this.notes = ''});
  final String id, name, room;
  final int rows, columns;
  String notes;

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'room': room,
        'rows': rows,
        'columns': columns,
        'notes': notes
      };
  factory Project.fromJson(Map<String, dynamic> value) => Project(
        id: value['id'] as String,
        name: value['name'] as String,
        room: value['room'] as String,
        rows: value['rows'] as int,
        columns: value['columns'] as int,
        notes: value['notes'] as String? ?? '',
      );
}

class ThudStore extends ChangeNotifier {
  static const _key = 'thud.store.v1';
  List<Project> projects = [];
  List<Measurement> measurements = [];
  bool haptics = true, sounds = false;
  Measurement? reference;
  SharedPreferences? _prefs;
  Future<void>? _loading;
  Future<void> _pending = Future<void>.value();
  bool _disposed = false;
  static int _sequence = 0;
  static String _id() =>
      '${DateTime.now().microsecondsSinceEpoch}-${_sequence++}';

  Future<void> load() => _loading ??= _load();
  Future<void> _load() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_key);
      if (raw != null) {
        final data = jsonDecode(raw) as Map<String, dynamic>;
        final savedProjects = (data['projects'] as List)
            .map((e) => Project.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
        final savedMeasurements = (data['measurements'] as List)
            .map((e) =>
                Measurement.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
        final savedReference = data['reference'] == null
            ? null
            : Measurement.fromJson(
                Map<String, dynamic>.from(data['reference'] as Map));
        projects = savedProjects;
        measurements = savedMeasurements;
        reference = savedReference;
        haptics = data['haptics'] as bool? ?? true;
        sounds = data['sounds'] as bool? ?? false;
      }
      _prefs = prefs;
      if (!_disposed) notifyListeners();
    } catch (error) {
      _loading = null;
      throw Exception('Could not load saved THUD data: $error');
    }
  }

  Future<T> _change<T>(T Function() action) {
    final result = _pending.then((_) async {
      await load();
      if (_disposed) throw Exception('The project store is closed.');
      final oldProjects = List<Project>.of(projects);
      final oldMeasurements = List<Measurement>.of(measurements);
      final oldNotes = {for (final p in projects) p.id: p.notes};
      final oldReference = reference;
      final oldHaptics = haptics, oldSounds = sounds;
      try {
        final value = action();
        final saved = await _prefs!.setString(
            _key,
            jsonEncode({
              'projects': projects.map((p) => p.toJson()).toList(),
              'measurements': measurements.map((m) => m.toJson()).toList(),
              'reference': reference?.toJson(),
              'haptics': haptics,
              'sounds': sounds,
            }));
        if (!saved) throw Exception('Device storage rejected the write.');
        if (!_disposed) notifyListeners();
        return value;
      } catch (error) {
        projects = oldProjects;
        measurements = oldMeasurements;
        for (final p in projects) {
          p.notes = oldNotes[p.id]!;
        }
        reference = oldReference;
        haptics = oldHaptics;
        sounds = oldSounds;
        throw Exception('Could not save THUD data: $error');
      }
    });
    _pending =
        result.then<void>((_) {}, onError: (Object error, StackTrace stack) {});
    return result;
  }

  Future<void> setFeedback({bool? haptics, bool? sounds}) => _change(() {
        if (haptics != null) this.haptics = haptics;
        if (sounds != null) this.sounds = sounds;
      });

  Future<Project> addProject(String name, String room, int rows, int columns) =>
      _change(() {
        if (name.trim().isEmpty) throw Exception('Enter a project name.');
        if (rows < 1 || columns < 1 || rows > 100 || columns > 100) {
          throw Exception('Choose between 1 and 100 rows and columns.');
        }
        final value = Project(
            id: _id(),
            name: name.trim(),
            room: room.trim(),
            rows: rows,
            columns: columns);
        projects.insert(0, value);
        return value;
      });

  Future<void> updateNotes(Project project, String notes) => _change(() {
        final index = projects.indexWhere((p) => p.id == project.id);
        if (index < 0) throw Exception('This project is no longer available.');
        projects[index].notes = notes;
      });

  Future<void> saveMeasurement(Measurement value,
          {String? projectId, int? cell}) =>
      _change(() {
        final target = projectId ?? value.projectId;
        final targetCell = cell ?? value.cell;
        if (target != null) {
          final index = projects.indexWhere((p) => p.id == target);
          if (index < 0) throw Exception('Select an existing project.');
          final p = projects[index];
          if (targetCell != null &&
              (targetCell < 0 || targetCell >= p.rows * p.columns)) {
            throw Exception('The selected grid cell is outside this project.');
          }
        } else if (targetCell != null) {
          throw Exception('A grid cell requires a project.');
        }
        measurements.removeWhere((m) => m.id == value.id);
        measurements.insert(0, value._assigned(target, targetCell));
      });

  Future<void> deleteMeasurement(String id) => _change(() {
        measurements.removeWhere((m) => m.id == id);
        if (reference?.id == id) reference = null;
      });

  List<Measurement> forProject(String id) =>
      measurements.where((m) => m.projectId == id).toList(growable: false);

  Future<void> setReference(Measurement m) => _change(() {
        if (m.source.toLowerCase().contains('synthetic') ||
            m.flags.isNotEmpty) {
          throw Exception(
              'Use an unflagged microphone recording as the reference.');
        }
        reference = m;
      });

  String reportText(Project p) {
    final out = StringBuffer(
        'THUD — Acoustic inspection record\n${p.name}\nRoom: ${p.room}\nGrid: ${p.rows} × ${p.columns}\n\nNotes\n${p.notes.isEmpty ? "No notes." : p.notes}\n\n');
    final records = forProject(p.id);
    if (records.isEmpty) out.writeln('No saved measurements.');
    for (final m in records) {
      out.writeln('${m.label} · ${m.created.toIso8601String()}');
      out.writeln('Source: ${m.source}');
      out.writeln(
          'Module: ${m.module.name} · Cell: ${m.cell == null ? "unassigned" : m.cell! + 1}');
      out.writeln(
          'Peak: ${m.peakHz.toStringAsFixed(1)} Hz · HII: ${m.hiiDb.toStringAsFixed(1)} dB');
      out.writeln(
          '20 dB decay estimate: ${m.decayMs > 0 ? "${m.decayMs.toStringAsFixed(0)} ms" : "not measurable"}');
      out.writeln(
          'RMS: ${m.rmsDb.toStringAsFixed(1)} dBFS · estimated SNR: ${m.snrDb.toStringAsFixed(1)} dB');
      out.writeln(
          'Quality: ${m.flags.isEmpty ? "No automatic flags" : m.flags.join(", ")}\n');
    }
    out.writeln(
        'Method: mono PCM at 48 kHz; 4096-point Hann-window FFT around the strongest impulse. HII = 10 log10(E450–2200 Hz / E2200–6500 Hz). Decay is the observed 20 dB amplitude reduction, not an RT60 extrapolation. SNR uses the quietest fifth of 10 ms windows as estimated background. Levels are digital dBFS, not calibrated sound pressure.');
    out.writeln(
        'These are acoustic measurements, not a material, structural, defect, or safety diagnosis. Synthetic samples are explicitly identified and are not field observations.');
    if (reference != null) {
      out.writeln(
          'Reference: ${reference!.label} (${reference!.id}); ${reference!.source}.');
    }
    return out.toString();
  }

  @override
  void dispose() {
    _disposed = true;
    super.dispose();
  }
}

/// Pure analysis of normalized mono PCM, with no microphone/platform access.
/// Waveform: <=256 signed peak-preserving bins in [-1,1]. Spectrum: 256
/// logarithmic bins, f(i)=20*(sampleRate/40)^(i/255), with [0,1] representing
/// [-100,0] dBFS. All scalar levels are digital, not calibrated SPL.
Measurement analyzePcm(
  List<double> pcm,
  Module module, {
  int sampleRate = 48000,
  String source = 'Microphone · PCM16 / 48 kHz',
  String? id,
  DateTime? created,
}) {
  const fftSize = 4096;
  if (sampleRate < 16000 || sampleRate > 192000) {
    throw Exception('Use a PCM sample rate between 16 and 192 kHz.');
  }
  if (pcm.length < fftSize) {
    throw Exception(
        'Recording too short. Capture at least 0.1 seconds of audio.');
  }
  var sum = 0.0, energy = 0.0, maximum = 0.0;
  var peakIndex = 0, clipped = 0;
  for (var i = 0; i < pcm.length; i++) {
    final x = pcm[i];
    if (!x.isFinite || x.abs() > 1.0001) {
      throw Exception(
          'PCM must contain finite normalized samples between -1 and 1.');
    }
    sum += x;
    energy += x * x;
    if (x.abs() > maximum) {
      maximum = x.abs();
      peakIndex = i;
    }
    if (x.abs() >= .999) clipped++;
  }
  final mean = sum / pcm.length;
  final acEnergy = math.max(0.0, energy / pcm.length - mean * mean);
  if (maximum < 1e-7 || acEnergy < 1e-14) {
    throw Exception(
        'No usable audio detected. Check the microphone and try another tap.');
  }
  final flags = <String>[];
  if (clipped > 0) flags.add('Clipping detected');
  final window = math.max(1, (sampleRate * .01).round());
  final envelope = <double>[];
  for (var start = 0; start + window <= pcm.length; start += window) {
    var e = 0.0;
    for (var i = start; i < start + window; i++) {
      final x = pcm[i] - mean;
      e += x * x;
    }
    envelope.add(e / window);
  }
  final sorted = List<double>.of(envelope)..sort();
  final quietCount = math.max(1, sorted.length ~/ 5);
  var noise = 0.0;
  for (var i = 0; i < quietCount; i++) {
    noise += sorted[i];
  }
  noise = math.max(1e-12, noise / quietCount);
  var envelopePeak = 0;
  for (var i = 1; i < envelope.length; i++) {
    if (envelope[i] > envelope[envelopePeak]) envelopePeak = i;
  }
  final signalPower = envelope[envelopePeak];
  final snr = _db(math.max(1e-12, signalPower - noise) / noise)
      .clamp(-100.0, 120.0)
      .toDouble();
  if (snr < 15) flags.add('Low SNR: repeat in a quieter environment');
  var decayMs = 0.0;
  final threshold = (signalPower - noise) / 100 + noise;
  if (snr >= 23) {
    for (var i = envelopePeak + 1; i + 2 < envelope.length; i++) {
      if (envelope[i] <= threshold &&
          envelope[i + 1] <= threshold &&
          envelope[i + 2] <= threshold) {
        decayMs = (i - envelopePeak) * window * 1000 / sampleRate;
        break;
      }
    }
  }
  if (decayMs == 0) flags.add('20 dB decay not measurable');
  if (pcm.length < sampleRate ~/ 4) {
    flags.add('Short recording: decay estimate may be limited');
  }

  final real = Float64List(fftSize), imag = Float64List(fftSize);
  final start = (peakIndex - fftSize ~/ 8).clamp(0, pcm.length - fftSize);
  var windowSum = 0.0;
  for (var i = 0; i < fftSize; i++) {
    final w = .5 - .5 * math.cos(2 * math.pi * i / (fftSize - 1));
    real[i] = (pcm[start + i] - mean) * w;
    windowSum += w;
  }
  _fft(real, imag);
  final powers = Float64List(fftSize ~/ 2 + 1);
  var lowBand = 0.0, highBand = 0.0;
  var strongest = 1;
  final minPeakBin = math.max(1, (40 * fftSize / sampleRate).ceil());
  final maxPeakBin =
      math.min(powers.length - 2, (10000 * fftSize / sampleRate).floor());
  strongest = minPeakBin;
  for (var i = 1; i < powers.length; i++) {
    final p =
        (real[i] * real[i] + imag[i] * imag[i]) * 4 / (windowSum * windowSum);
    powers[i] = p;
    final hz = i * sampleRate / fftSize;
    if (hz >= 450 && hz < 2200) lowBand += p;
    if (hz >= 2200 && hz <= 6500) highBand += p;
    if (i >= minPeakBin && i <= maxPeakBin && p > powers[strongest]) {
      strongest = i;
    }
  }
  final left = math.log(math.max(1e-20, powers[strongest - 1]));
  final center = math.log(math.max(1e-20, powers[strongest]));
  final right = math.log(math.max(1e-20, powers[strongest + 1]));
  final divisor = left - 2 * center + right;
  final shift = divisor.abs() < 1e-12
      ? 0.0
      : (.5 * (left - right) / divisor).clamp(-.5, .5).toDouble();
  final spectrum = List<double>.generate(256, (i) {
    final hz = 20 * math.pow(sampleRate / 40, i / 255);
    final bin = (hz * fftSize / sampleRate).round().clamp(1, powers.length - 1);
    return ((_db(powers[bin]) + 100) / 100).clamp(0.0, 1.0).toDouble();
  }, growable: false);
  final time = created ?? DateTime.now();
  return Measurement(
      id: id ?? ThudStore._id(),
      label:
          '${module.name[0].toUpperCase()}${module.name.substring(1)} capture',
      source: source,
      created: time,
      module: module,
      peakHz: (strongest + shift) * sampleRate / fftSize,
      hiiDb: _db(math.max(1e-12, lowBand) / math.max(1e-12, highBand))
          .clamp(-120.0, 120.0)
          .toDouble(),
      decayMs: decayMs,
      rmsDb: _db(acEnergy).clamp(-140.0, 0.0).toDouble(),
      snrDb: snr,
      waveform: List<double>.unmodifiable(_waveform(pcm, 256)),
      spectrum: List<double>.unmodifiable(spectrum),
      flags: List<String>.unmodifiable(flags));
}

double _db(double power) => 10 * math.log(math.max(1e-20, power)) / math.ln10;

List<double> _waveform(List<double> pcm, int count) {
  if (pcm.isEmpty) return const [];
  final length = math.min(count, pcm.length);
  return List<double>.generate(length, (bin) {
    final start = bin * pcm.length ~/ length;
    final end = (bin + 1) * pcm.length ~/ length;
    var peak = 0.0;
    for (var i = start; i < end; i++) {
      if (pcm[i].abs() > peak.abs()) peak = pcm[i];
    }
    return peak.clamp(-1.0, 1.0).toDouble();
  }, growable: false);
}

void _fft(Float64List real, Float64List imag) {
  final n = real.length;
  for (var i = 1, j = 0; i < n; i++) {
    var bit = n >> 1;
    while ((j & bit) != 0) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      final r = real[i];
      real[i] = real[j];
      real[j] = r;
      final im = imag[i];
      imag[i] = imag[j];
      imag[j] = im;
    }
  }
  for (var size = 2; size <= n; size <<= 1) {
    final angle = -2 * math.pi / size;
    final stepR = math.cos(angle), stepI = math.sin(angle);
    for (var start = 0; start < n; start += size) {
      var wr = 1.0, wi = 0.0;
      for (var j = 0; j < size ~/ 2; j++) {
        final a = start + j, b = a + size ~/ 2;
        final tr = wr * real[b] - wi * imag[b];
        final ti = wr * imag[b] + wi * real[b];
        real[b] = real[a] - tr;
        imag[b] = imag[a] - ti;
        real[a] += tr;
        imag[a] += ti;
        final nextR = wr * stepR - wi * stepI;
        wi = wr * stepI + wi * stepR;
        wr = nextR;
      }
    }
  }
}

class CaptureService {
  static const _sampleRate = 48000;
  static const _maxSamples = _sampleRate * 30;
  AudioRecorder? _recorder;
  final ValueNotifier<List<double>> live = ValueNotifier(const []);
  bool recording = false;
  bool _closing = false, _hasTake = false, _overflow = false;
  int _epoch = 0, _length = 0;
  int? _oddByte;
  Float64List? _pcm;
  Object? _streamError;
  StreamSubscription<Uint8List>? _subscription;
  Completer<void>? _streamDone;
  final Stopwatch _clock = Stopwatch();
  int _lastLiveMs = 0;
  Future<void> _pending = Future<void>.value();
  Future<void>? _disposing;

  Future<T> _serial<T>(Future<T> Function() action) {
    final result = _pending.then((_) => action());
    _pending =
        result.then<void>((_) {}, onError: (Object error, StackTrace stack) {});
    return result;
  }

  Future<void> start() {
    final epoch = _epoch;
    return _serial(() async {
      if (_closing) throw Exception('The microphone service is closed.');
      if (epoch != _epoch) throw Exception('Recording was cancelled.');
      if (_hasTake) {
        throw Exception('Stop or cancel the current recording first.');
      }
      final recorder = _recorder ??= AudioRecorder();
      try {
        if (!await recorder.hasPermission()) {
          throw Exception(
              'Microphone permission is required. Enable it in device settings.');
        }
        if (_closing || epoch != _epoch) {
          throw Exception('Recording was cancelled.');
        }
        if (!await recorder.isEncoderSupported(AudioEncoder.pcm16bits)) {
          throw Exception(
              'This device does not support PCM16 microphone streaming.');
        }
        await recorder.setOnConfigChanged((config) {
          if (config.sampleRate != _sampleRate ||
              config.numChannels != 1 ||
              config.encoder != AudioEncoder.pcm16bits) {
            _streamError = Exception(
                'The microphone changed format. Reconnect a 48 kHz mono input and retry.');
          }
        });
        _pcm = Float64List(_maxSamples);
        _length = 0;
        _oddByte = null;
        _streamError = null;
        _overflow = false;
        _streamDone = Completer<void>();
        _lastLiveMs = 0;
        _clock
          ..reset()
          ..start();
        final stream = await recorder.startStream(const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: _sampleRate,
          numChannels: 1,
          autoGain: false,
          echoCancel: false,
          noiseSuppress: false,
        ));
        _hasTake = true;
        _subscription = stream.listen(_receive, onError: (Object error) {
          _streamError = error;
          recording = false;
        }, onDone: () {
          recording = false;
          if (!(_streamDone?.isCompleted ?? true)) _streamDone!.complete();
        });
        if (_closing || epoch != _epoch) {
          await recorder.cancel();
          await _clear();
          throw Exception('Recording was cancelled.');
        }
        recording = true;
      } catch (error) {
        try {
          await recorder.cancel();
        } catch (_) {/* Preserve the original microphone error. */}
        await _clear();
        throw Exception('Could not start microphone: $error');
      }
    });
  }

  void _receive(Uint8List bytes) {
    final pcm = _pcm;
    if (pcm == null || _closing) return;
    var i = 0;
    if (_oddByte != null && bytes.isNotEmpty) {
      _append(_oddByte! | (bytes[0] << 8));
      _oddByte = null;
      i = 1;
    }
    for (; i + 1 < bytes.length; i += 2) {
      _append(bytes[i] | (bytes[i + 1] << 8));
    }
    if (i < bytes.length) _oddByte = bytes[i];
    final ms = _clock.elapsedMilliseconds;
    if (ms - _lastLiveMs >= 100 && _length > 0) {
      _lastLiveMs = ms;
      final start = math.max(0, _length - 4800);
      live.value = _waveform(Float64List.sublistView(pcm, start, _length), 128);
    }
  }

  void _append(int raw) {
    if (_length >= _maxSamples) {
      _overflow = true;
      return;
    }
    _pcm![_length++] = (raw >= 32768 ? raw - 65536 : raw) / 32768;
  }

  Future<Measurement> stop(Module module) {
    final epoch = _epoch;
    return _serial(() async {
      if (_closing) throw Exception('The microphone service is closed.');
      if (epoch != _epoch) throw Exception('Recording was cancelled.');
      if (!_hasTake) throw Exception('Start a recording before stopping.');
      try {
        await _recorder!.stop();
        await _streamDone!.future.timeout(const Duration(seconds: 3));
        if (_closing || epoch != _epoch) {
          throw Exception('Recording was cancelled.');
        }
        if (_streamError != null) {
          throw Exception('Microphone stream failed: $_streamError');
        }
        if (_oddByte != null) {
          throw Exception(
              'The microphone returned an incomplete PCM sample. Please record again.');
        }
        if (_overflow) {
          throw Exception(
              'Recording exceeded 30 seconds. Record a shorter tap.');
        }
        return analyzePcm(Float64List.sublistView(_pcm!, 0, _length), module);
      } catch (error) {
        throw Exception('Could not analyze microphone recording: $error');
      } finally {
        await _clear();
      }
    });
  }

  /// Explicit demo source, never an inferred material or defect prediction.
  Future<Measurement> sample(Module module, {bool resonant = false}) =>
      _serial(() async {
        if (_closing) throw Exception('The microphone service is closed.');
        if (_hasTake) {
          throw Exception(
              'Stop or cancel microphone capture before opening a sample.');
        }
        final pcm = Float64List(_sampleRate * 2);
        final random =
            math.Random(715 + module.index * 31 + (resonant ? 1 : 0));
        final base = module == Module.vessel
            ? (resonant ? 690.0 : 1030.0)
            : (resonant ? 330.0 : 840.0);
        final tau = resonant ? .19 : .055;
        for (var i = 0; i < pcm.length; i++) {
          final t = i / _sampleRate;
          var value = (random.nextDouble() * 2 - 1) * .0008;
          // A quiet lead-in followed by a physically-inspired damped modal impulse.
          final dt = t - .18;
          if (dt >= 0) {
            final onset = 1 - math.exp(-dt * 2200);
            value += onset *
                (.50 * math.exp(-dt / tau) * math.sin(2 * math.pi * base * dt) +
                    .21 *
                        math.exp(-dt / (tau * .57)) *
                        math.sin(2 * math.pi * base * 2.73 * dt) +
                    .08 *
                        math.exp(-dt / .018) *
                        math.sin(2 * math.pi * 4200 * dt));
          }
          // Quantize exactly as signed PCM16, including the quiet noise floor.
          pcm[i] = (value.clamp(-1.0, .999969) * 32768).round() / 32768;
        }
        return analyzePcm(pcm, module,
            source:
                'Synthetic · deterministic ${module.name} ${resonant ? "long" : "short"}-decay PCM16 sample');
      });

  Future<void> cancel() {
    _epoch++;
    return _serial(() async {
      if (_closing) return;
      try {
        if (_hasTake) await _recorder?.cancel();
      } catch (error) {
        throw Exception('Could not cancel microphone capture: $error');
      } finally {
        await _clear();
      }
    });
  }

  Future<void> _clear() async {
    recording = false;
    _hasTake = false;
    _clock.stop();
    await _subscription?.cancel();
    _subscription = null;
    _streamDone = null;
    _pcm = null;
    _length = 0;
    _oddByte = null;
    _streamError = null;
  }

  Future<void> dispose() {
    if (_disposing != null) return _disposing!;
    _closing = true;
    _epoch++;
    return _disposing = _serial(() async {
      try {
        if (_hasTake) await _recorder?.cancel();
      } finally {
        await _clear();
        try {
          await _recorder?.dispose();
        } finally {
          live.dispose();
        }
      }
    });
  }
}
