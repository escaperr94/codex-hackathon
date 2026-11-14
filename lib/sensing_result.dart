import 'dart:convert';
import 'package:flutter/widgets.dart';
import 'package:material_ui/material_ui.dart';
import 'design.dart';
import 'store.dart';
import 'sensing_flow.dart';
import 'learned_calibration.dart';

/// Consumer summary shared by the live result, history and inspection report.
class InspectionVerdict {
  const InspectionVerdict(this.title, this.advice, {this.caution = true});
  final String title, advice;
  final bool caution;

  static InspectionVerdict from(Measurement m, SensingIntent intent) {
    if (!ReferenceAnchor.clean(m)) {
      return const InspectionVerdict('Try another recording', 'Move somewhere quiet. Avoid clipping and repeat the measurement.');
    }
    if (m.context['resultVersion'] == '4') {
      return InspectionVerdict(m.context['interpretation'] ?? 'Insufficient evidence',
          m.context['nextAction'] ?? 'Collect clean references for this object.',
          caution: m.context['resultCaution'] != 'false');
    }
    if (intent == SensingIntent.scope) {
      return const InspectionVerdict('Sound captured', 'Explore the waveform, spectrum and decay.', caution: false);
    }
    return const InspectionVerdict('Insufficient evidence',
        'This older result has no compatible neural calibration. Record again; acoustic similarity is not a physical diagnosis.');
  }
}

class SensingInterpretation {
  static List<double> _features(Measurement m) {
    final raw = m.context['acousticAnalysis'];
    final analysis = raw == null ? const <String, dynamic>{}
        : Map<String, dynamic>.from(jsonDecode(raw) as Map);
    double? read(String key) => (analysis[key] as num?)?.toDouble();
    return LearnedCalibration.features(m.embedding, peakHz: m.peakHz,
      decayMs: m.decayMs, centroidHz: read('centroidHz'),
      flatness: read('flatness'), damping: read('damping'));
  }
  static Map<String, String> evaluate(SensingIntent intent,
      List<Measurement> queries, List<ReferenceAnchor> references) {
    final result = <String, String>{
      'resultVersion': '4', 'interpretation': 'Insufficient evidence',
      'nextAction': 'Choose clean neural references for the same object and setup. Signal-only references must be recorded again.',
      'resultCaution': 'true', 'modelStatus': 'calibration required',
      'classifierStatus': 'insufficient-evidence',
      'methodNote': 'YAMNet unit-normalized neural1024 + bounded resonance/decay/centroid/flatness/damping features (secondary block norm ≤0.1). Local task calibration; no validated physical diagnosis.',
    };
    if (queries.isEmpty || queries.any((m) => !ReferenceAnchor.clean(m) || !m.hasEmbedding)) {
      return {...result, 'modelStatus': 'invalid capture', 'interpretation': 'Record again',
        'nextAction': 'Capture clean audio with the neural encoder available.'};
    }
    final first = queries.first;
    final demo = first.source.toLowerCase().contains('synthetic');
    final operating = first.context['captureMode'] == 'operating';
    if (queries.any((m) => m.encoderVersion != first.encoderVersion ||
        m.source.toLowerCase().contains('synthetic') != demo ||
        (m.context['captureMode'] == 'operating') != operating)) {
      return {...result, 'modelStatus': 'invalid capture', 'nextAction': 'Do not mix encoders, demo audio or capture modes.'};
    }
    final refs = references.where((a) => a.intent == intent && a.usable && a.synthetic == demo &&
        a.taps.every((m) => m.encoderVersion == first.encoderVersion &&
          (m.context['captureMode'] == 'operating') == operating)).toList();
    if (refs.map((a) => a.object.trim().toLowerCase()).toSet().length > 1) {
      return {...result, 'nextAction': 'Choose one object profile; do not mix fruit types, containers or devices.'};
    }
    try {
    final query = LearnedCalibration.aggregate(queries.map(_features).toList());
    result.addAll({'queryTapIds': queries.map((m) => m.id).join(','),
      'queryCount': '${queries.length}', 'aggregation': 'coordinate median',
      'encoderVersion': first.encoderVersion!,
      'captureMode': operating ? 'operating' : 'impact',
      'referenceProfiles': jsonEncode(refs.map((a) => {
        'object': a.object, 'label': a.label, 'metadata': a.metadata,
        'tapIds': a.taps.map((m) => m.id).toList(),
      }).toList()),
      'referenceTapIds': refs.expand((a) => a.taps.map((m) => m.id)).join(',')});
    if (refs.isNotEmpty) {
      final taps = refs.expand((a) => a.taps).toList();
      double mean(double Function(Measurement) read) => taps.fold<double>(0, (sum, m) => sum + read(m)) / taps.length;
      result['peakShiftPercent'] = '${100 * (first.peakHz / mean((m) => m.peakHz) - 1)}';
      result['decayShiftMs'] = '${first.decayMs - mean((m) => m.decayMs)}';
      result['referencePeakHz'] = '${mean((m) => m.peakHz)}';
      result['referenceDecayMs'] = '${mean((m) => m.decayMs)}';
    }
    if (intent == SensingIntent.scope) {
      result.addAll({'interpretation': 'Sound captured', 'nextAction': 'Explore the measured signal. Save a reference to compare another recording.',
        'resultCaution': 'false', 'modelStatus': 'signal available'});
      if (refs.isNotEmpty) {
        final prediction = LearnedCalibration.fitReference(refs.expand((a) => a.taps.map(_features)).toList()).predict(query);
        result['embeddingDistance'] = '${prediction.distance}';
      }
      return result;
    }
    if (refs.isEmpty) return result;
      final embeddings = refs.expand((a) => a.taps.map(_features)).toList();
      if (intent == SensingIntent.vessel) {
        if (refs.any((a) => a.level == null) || refs.map((a) => a.level).toSet().length < 3) {
          return {...result, 'nextAction': 'Teach at least three measured fill levels for this exact container.'};
        }
        final model = LearnedCalibration.fitFill(embeddings, refs.expand((a) => a.taps.map((_) => a.level!)).toList());
        final p = model.predict(query);
        result.addAll({'fittedCalibration': jsonEncode(model.toJson()),
          'fillCalibrationStatus': p.calibrationStatus, 'fillLowerPercent': '${p.lower}',
          'fillUpperPercent': '${p.upper}', 'intervalMethod': 'empirical calibration-resolution and embedding-support envelope; not validated coverage'});
        if (!p.supported) return {...result, 'interpretation': 'Fill inconclusive', 'nextAction': 'Outside calibrated range. Repeat the setup or add measured fill references.'};
        result.addAll({'estimatedFillPercent': '${p.value}', 'interpretation': 'About ${p.value.round()}% full',
          'nextAction': '${p.lower.round()}–${p.upper.round()}% empirical range. Experimental calibration for this container, not a validated confidence interval.',
          'modelStatus': 'prediction available', 'resultCaution': 'false'});
        final capacities = refs.map((a) => a.metadata['capacityMl']).toSet();
        final capacity = capacities.length == 1 ? double.tryParse(capacities.single ?? '') : null;
        if (capacity != null && capacity.isFinite && capacity > 0) result['estimatedVolumeMl'] = '${capacity * p.value / 100}';
        return result;
      }
      final labelled = refs.where((a) => a.label != null).toList();
      final labels = labelled.map((a) => ReferenceAnchor.normalizedLabel(a.label)!).toSet();
      final canClassify = intent != SensingIntent.devices && labels.length >= 2 &&
          (intent != SensingIntent.slab || labels.containsAll(['Supported', 'Suspected Unsupported']));
      if (intent == SensingIntent.fruit && !canClassify) {
        return {...result, 'nextAction': 'Teach labelled firmer and softer examples of one fruit type/cultivar. This cannot determine exact ripeness, sweetness or food safety.'};
      }
      if (canClassify) {
        final model = LearnedCalibration.fitClasses(labelled.expand((a) => a.taps.map(_features)).toList(),
            labelled.expand((a) => a.taps.map((_) => ReferenceAnchor.normalizedLabel(a.label)!)).toList());
        final p = model.predict(query);
        result.addAll({'classifierMethod': 'yamnet-nearest-prototype', 'classifierStatus': p.supported ? 'matched' : 'outside-reference-range',
          'embeddingDistance': '${p.distance}', 'fittedCalibration': jsonEncode(model.toJson()), 'modelStatus': 'prediction available'});
        if (intent == SensingIntent.slab && labelled.every((a) => (a.metadata['specimenId'] ?? '').isNotEmpty)) {
          result['tileHoldoutEvaluation'] = jsonEncode(LearnedCalibration.evaluateTiles(
            labelled.expand((a) => a.taps.map(_features)).toList(),
            labelled.expand((a) => a.taps.map((_) => ReferenceAnchor.normalizedLabel(a.label)!)).toList(),
            labelled.expand((a) => a.taps.map((_) => a.metadata['specimenId']!)).toList()));
        }
        if (!p.supported) {
          return {...result,
            'interpretation': intent == SensingIntent.materials ? 'Unknown material signature' : 'Inconclusive',
            'nextAction': 'Outside your labelled examples or ambiguous. Repeat measurements and collect more independent objects.'};
        }
        result['classifierLabel'] = p.label;
        result['interpretation'] = switch (intent) {
          SensingIntent.slab => p.label,
          SensingIntent.fruit => '${p.label} reference similarity',
          _ => 'Similar to ${p.label}',
        };
        result['nextAction'] = switch (intent) {
          SensingIntent.slab => 'Experimental support classification only. Not certified bond or construction-defect detection; verify manually.',
          SensingIntent.fruit => 'Relative acoustic similarity for this fruit type only. Not exact maturity, sweetness, internal defects or food safety.',
          _ => 'Acoustic similarity to your enrolled examples, not chemical composition or authenticity identification.',
        };
        if (intent == SensingIntent.slab) result['supportVerdict'] = p.label;
        return result;
      }
      final normal = intent == SensingIntent.slab
          ? refs.where((a) => a.label == 'Supported' || a.label == 'Solid').toList()
          : intent == SensingIntent.devices
              ? refs.where((a) => a.label == null || a.label == 'Normal' || a.label == 'Reference').toList()
              : refs;
      if (normal.isEmpty) return result;
      final model = LearnedCalibration.fitReference(normal.expand((a) => a.taps.map(_features)).toList());
      final p = model.predict(query);
      result.addAll({'fittedCalibration': jsonEncode(model.toJson()), 'embeddingDistance': '${p.distance}',
        'learnedThreshold': '${p.threshold}', 'classifierMethod': 'yamnet-normal-centroid',
        'classifierStatus': p.different ? 'outside-reference-range' : 'matched',
        'modelStatus': 'prediction available', 'suspicion': p.different ? 'Measured difference' : 'Reference match'});
      result['interpretation'] = switch (intent) {
        SensingIntent.slab => 'Inconclusive',
        SensingIntent.materials => p.different ? 'Unknown material signature' : 'Similar to your reference',
        SensingIntent.devices => p.different ? 'Acoustic change detected' : 'Similar to normal recordings',
        _ => p.different ? 'Different from reference' : 'Reference similarity',
      };
      result['nextAction'] = intent == SensingIntent.slab
          ? '${p.different ? 'Different from' : 'Similar to'} supported references. Normal-only calibration cannot assign support state; teach unsupported tiles too.'
          : intent == SensingIntent.devices
              ? 'Compare the same device, load and microphone position over time. An acoustic change does not identify a mechanical failure.'
              : 'Acoustic similarity is not definitive material identification. Enroll labelled examples for few-shot comparison.';
      if (intent == SensingIntent.slab) result['supportVerdict'] = 'Inconclusive';
      return result;
    } catch (e) {
      return {...result, 'modelStatus': 'calibration invalid',
        'nextAction': 'Calibration needs compatible, distinct examples: $e'};
    }
  }
}

class SensingResult extends StatefulWidget {
  const SensingResult({required this.measurement, required this.intent,
    required this.store, required this.onSave, required this.onMap,
    required this.onTrace, required this.onAgain, required this.saved, super.key});
  final Measurement measurement;
  final SensingIntent intent;
  final ThudStore store;
  final VoidCallback onAgain;
  final ValueChanged<Measurement> onSave, onMap, onTrace;
  final bool saved;
  @override
  State<SensingResult> createState() => _SensingResultState();
}

class _SensingResultState extends State<SensingResult> {
  List<ReferenceAnchor> anchors = [];
  final extraQueries = <Measurement>[];
  final capture = CaptureService();
  bool recording = false, capturing = false;
  String? object, error;
  bool loading = true;
  bool get _demo => widget.measurement.source.toLowerCase().contains('synthetic');
  bool get _usesReferences => true;
  bool get loadingReferences => loading;

  @override
  void initState() {
    super.initState();
    object = widget.measurement.context['referenceObject'];
    _load();
  }

  List<ReferenceAnchor> _eligible(List<ReferenceAnchor> all) => all
      .where((a) =>
          a.intent == widget.intent &&
          a.synthetic == _demo &&
          a.object.trim().isNotEmpty &&
          a.taps.length >= 3 &&
          a.taps.every((t) =>
              ReferenceAnchor.clean(t) &&
              t.source.toLowerCase().contains('synthetic') == _demo))
      .toList();

  Future<void> _load() async {
    try {
      final all = await AnchorRepository.load();
      final eligible = _eligible(all);
      final names = eligible.map((a) => a.object).toSet().toList()..sort();
      setState(() {
        anchors = all;
        if (!names.contains(object)) {
          object = names.length == 1 ? names.single : null;
        }
        loading = false;
        error = null;
      });
    } catch (e) {
      if (mounted) setState(() { loading = false; error = 'Calibration could not be loaded: $e'; });
    }
  }

  Future<void> _openLibrary() async {
    final selected = await Navigator.of(context).push(sensingRoute<String>(
        context,
        ReferenceLibraryScreen(intent: widget.intent, store: widget.store)));
    if (!mounted) return;
    if (selected != null) setState(() => object = selected);
    await _load();
  }

  Future<void> _addQuery() async {
    if (capturing) return;
    setState(() { capturing = true; error = null; });
    try {
      if (!recording) {
        await capture.start();
        if (!mounted) { await capture.cancel(); return; }
        setState(() => recording = true);
      } else {
        final query = await capture.stop(widget.intent.module,
          operating: widget.measurement.context['captureMode'] == 'operating');
        if (!mounted) return;
        setState(() => recording = false);
        if (!ReferenceAnchor.clean(query) || !query.hasEmbedding) {
          throw Exception('Recording rejected: ${query.flags.join(', ')}');
        }
        setState(() => extraQueries.add(query));
      }
    } catch (e) {
      if (mounted) setState(() { recording = false; error = e.toString(); });
    } finally {
      if (mounted) setState(() => capturing = false);
    }
  }

  @override
  void dispose() {
    capture.dispose();
    super.dispose();
  }

  String _qualityLabel(Measurement measurement) {
    final flags = measurement.flags.join(' ').toLowerCase();
    if (flags.contains('snr') || flags.contains('nois')) return 'Too noisy';
    if (flags.contains('clip')) return 'Clipped';
    if (flags.contains('short')) return 'Tap too short';
    return 'Tap again';
  }

  @override
  Widget build(BuildContext context) {
    final m = widget.measurement;
    final accent = widget.intent.color;
    final selected = object;
    final refs = _eligible(anchors).where((a) => a.object == selected).toList();
    final queries = [m, ...extraQueries];
    final inference = SensingInterpretation.evaluate(widget.intent, queries, refs);
    final retry = !ReferenceAnchor.clean(m) || !m.hasEmbedding;
    final quality = retry ? _qualityLabel(m) : 'Clean signal';
    final needsReference = inference['modelStatus'] == 'calibration required';
    final retryReference = inference['modelStatus'] == 'calibration invalid';
    final different = inference['suspicion'] == 'Measured difference' ||
        inference['classifierLabel'] == 'Suspected Unsupported';
    final score = double.tryParse(inference['embeddingDistance'] ?? '');
    final method = inference['methodNote']!;
    final savedContext = Map<String, String>.from(m.context)
      ..removeWhere((key, _) => const {
        'estimatedFillPercent', 'estimatedVolumeMl', 'fillLowerPercent', 'fillUpperPercent',
        'fillCalibrationStatus', 'maturityClass', 'firmnessIndex', 'regularizedDeviation',
        'classifierLabel', 'classifierStatus', 'supportVerdict', 'embeddingDistance',
        'learnedThreshold', 'fittedCalibration', 'tileHoldoutEvaluation', 'referenceObject',
        'referenceTapIds', 'referenceProfiles', 'resultCaution', 'suspicion',
      }.contains(key));
    final aggregateRecord = queries.every((q) => ReferenceAnchor.clean(q) && q.hasEmbedding)
        ? m.withEmbedding(LearnedCalibration.aggregate(queries.map((q) => q.embedding).toList())) : m;
    final measuredRecord = aggregateRecord.withIntent(widget.intent.name).withContext({
      ...savedContext, ...inference,
      'quality': quality,
      if (selected != null) 'referenceObject': selected,
    });
    final verdict = InspectionVerdict.from(measuredRecord, widget.intent);
    final record = measuredRecord.withContext({
      ...measuredRecord.context,
      'interpretation': verdict.title,
      'nextAction': verdict.advice,
    });
    final setup = needsReference || retryReference;
    final mapPrimary =
        !retry && !setup && (widget.intent == SensingIntent.slab || different);
    final tracePrimary = !retry && widget.intent == SensingIntent.scope;
    final primaryLabel = retry
        ? 'Record again'
        : setup
            ? (retryReference ? 'Refresh reference' : 'Choose reference')
            : mapPrimary
                ? widget.intent == SensingIntent.slab
                    ? 'Add to handover'
                    : 'Mark for review'
                : tracePrimary
                    ? 'View signal'
                    : (widget.saved ? 'Update result' : 'Save result');
    final primaryIcon = retry
        ? Icons.refresh_rounded
        : setup
            ? Icons.add_rounded
            : mapPrimary
                ? Icons.location_on_outlined
                : tracePrimary
                    ? Icons.graphic_eq_rounded
                    : Icons.check_rounded;
    final VoidCallback primaryAction = retry
        ? widget.onAgain
        : setup
            ? _openLibrary
            : mapPrimary
                ? () => widget.onMap(record)
                : tracePrimary
                    ? () => widget.onTrace(record)
                    : () => widget.onSave(record);
    final statusIcon = retry
        ? Icons.refresh_rounded
        : setup
            ? Icons.add_rounded
            : widget.intent == SensingIntent.scope
                ? Icons.graphic_eq_rounded
                : verdict.caution
                    ? Icons.help_outline_rounded
                    : Icons.check_rounded;
    final theme = Theme.of(context);

    return Theme(
        data: theme.copyWith(
            colorScheme: theme.colorScheme
                .copyWith(primary: accent, onPrimary: Colors.white)),
        child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(22),
            decoration: BoxDecoration(
                color: Colors.white, borderRadius: BorderRadius.circular(28)),
            child: Column(children: [
              if (_demo || (!retry && _usesReferences))
                Row(children: [
                  if (_demo)
                    const Text('Demo',
                        style: TextStyle(color: C.muted, fontSize: 12)),
                  const Spacer(),
                  if (!retry && _usesReferences)
                    IconButton(
                        tooltip: selected == null
                            ? 'Reference setup'
                            : 'Reference: $selected',
                        onPressed: loadingReferences ? null : _openLibrary,
                        icon:
                            Icon(Icons.tune_rounded, color: accent, size: 20)),
                ]),
              const SizedBox(height: 8),
              Container(
                  width: 108,
                  height: 108,
                  decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: accent.withValues(alpha: .1)),
                  child: Icon(statusIcon, color: accent, size: 56)),
              const SizedBox(height: 20),
              AnimatedSwitcher(
                  duration: Duration(
                      milliseconds:
                          MediaQuery.disableAnimationsOf(context) ? 0 : 240),
                  child: Text(verdict.title,
                      key: ValueKey(verdict.title),
                      textAlign: TextAlign.center,
                      style: TextStyle(
                          color: accent,
                          fontSize: 30,
                          height: 1.12,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -.7))),
              const SizedBox(height: 12),
              Text(verdict.advice,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      color: C.muted, fontSize: 15, height: 1.45)),
              if (retry) ...[
                const SizedBox(height: 10),
                Text(quality == 'Tap again' ? 'No clear ring' : quality,
                    style: const TextStyle(color: C.muted, fontSize: 15)),
              ] else ...[
                ExpansionTile(
                  tilePadding: EdgeInsets.zero,
                  title: const Text('Measurement details'),
                  children: [
                    Row(children: [
                      Metric('Frequency', '${m.peakHz.round()} Hz',
                          accent: accent),
                      Metric('Ring time', '${m.decayMs.round()} ms',
                          accent: accent),
                      if (score != null)
                        Metric('Difference', score.toStringAsFixed(1),
                            accent: accent),
                    ]),
                    if (selected != null) Text('Compared with: $selected'),
                    const SizedBox(height: 12),
                    Text(method,
                        style: const TextStyle(color: C.muted, fontSize: 12)),
                    Text('${queries.length} recordings aggregated with a coordinate median.'),
                    if (inference['estimatedVolumeMl'] != null)
                      Text('Estimated volume: ${double.parse(inference['estimatedVolumeMl']!).round()} ml'),
                    if (inference['fillCalibrationStatus'] != null)
                      Text('Calibration: ${inference['fillCalibrationStatus']}'),
                    if (inference['tileHoldoutEvaluation'] != null)
                      Text('Whole-tile holdout: ${inference['tileHoldoutEvaluation']}'),
                    if (inference['peakShiftPercent'] != null)
                      Text('Peak shift: ${double.parse(inference['peakShiftPercent']!).toStringAsFixed(1)}% · decay shift: ${double.parse(inference['decayShiftMs']!).toStringAsFixed(1)} ms'),
                  ],
                ),
                if (error != null)
                  Row(children: [
                    Expanded(
                        child: Text(error!,
                            style:
                                const TextStyle(color: C.red, fontSize: 12))),
                    TextButton(
                        onPressed: () => _load(), child: const Text('Reload')),
                  ]),
              ],
              const SizedBox(height: 24),
              SizedBox(
                  width: double.infinity,
                  child: ActionButton(primaryLabel,
                      icon: primaryIcon,
                      onPressed: _usesReferences && loadingReferences && !retry
                          ? null
                          : primaryAction)),
              const SizedBox(height: 6),
              if (!_demo && !retry)
                TextButton.icon(onPressed: capturing ? null : _addQuery,
                  icon: Icon(recording ? Icons.stop_rounded : Icons.mic_rounded),
                  label: Text(recording ? 'Stop & add recording' : 'Add repeat measurement')),
              Wrap(alignment: WrapAlignment.center, spacing: 8, children: [
                if (!tracePrimary)
                  TextButton(
                      onPressed: () => widget.onTrace(record),
                      child: const Text('View signal')),
                if (!retry) ...[
                  TextButton(
                      onPressed: widget.onAgain,
                      child: const Text('Record again')),
                  if (!setup && !mapPrimary && !tracePrimary)
                    TextButton(
                        onPressed: () => widget.onMap(record),
                        child: const Text('Mark location')),
                  if (tracePrimary)
                    TextButton(
                        onPressed: () => widget.onSave(record),
                        child: Text(
                            widget.saved ? 'Update result' : 'Save result')),
                ],
              ]),
            ])));
  }
}
