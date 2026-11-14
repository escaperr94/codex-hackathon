import 'package:material_ui/material_ui.dart';
import 'design.dart';
import 'store.dart';
import 'sensing_flow.dart';
import 'learned_calibration.dart';

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
  Map<String, dynamic> models = {};
  String? object, error;
  bool loading = true;
  bool get _demo => widget.measurement.source.toLowerCase().contains('synthetic');
  bool get _usesCalibration => widget.intent != SensingIntent.scope;

  @override
  void initState() {
    super.initState();
    object = widget.measurement.context['referenceObject'];
    _load();
  }

  List<ReferenceAnchor> _eligible(List<ReferenceAnchor> all) => all.where((a) =>
      a.intent == widget.intent && a.synthetic == _demo &&
      a.object.trim().isNotEmpty).toList();

  Future<void> _load({bool afterTeaching = false}) async {
    try {
      final all = await AnchorRepository.load();
      final fitted = await AnchorRepository.models();
      if (!mounted) return;
      final names = _eligible(all).map((a) => a.object).toSet().toList()..sort();
      final previous = _eligible(anchors).map((a) => a.taps.map((t) => t.id).join(',')).toSet();
      final updated = _eligible(all).where((a) =>
          !previous.contains(a.taps.map((t) => t.id).join(',')))
          .map((a) => a.object).toSet();
      setState(() {
        anchors = all;
        models = fitted;
        if (afterTeaching && updated.length == 1) {
          object = updated.single;
        } else if (!names.contains(object)) {
          object = names.length == 1 ? names.single : null;
        }
        loading = false;
        error = null;
      });
    } catch (e) {
      if (mounted) setState(() { loading = false; error = 'Calibration could not be loaded: $e'; });
    }
  }

  Future<void> _teach() async {
    await Navigator.of(context).push(sensingRoute<void>(context,
        GuidedReferenceScreen(intent: widget.intent, store: widget.store)));
    if (mounted) await _load(afterTeaching: true);
  }

  void _setup(List<String> names) {
    showModalBottomSheet<void>(context: context, showDragHandle: true,
      builder: (sheet) => SafeArea(child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          const SectionTitle('Learned calibration'),
          for (final name in names)
            ListTile(title: Text(name), selected: object == name,
              onTap: () { setState(() => object = name); Navigator.pop(sheet); }),
          TextButton.icon(icon: const Icon(Icons.add_rounded),
            label: const Text('Train with examples'), onPressed: () {
              Navigator.pop(sheet); _teach();
            }),
        ]),
      )));
  }

  @override
  Widget build(BuildContext context) {
    final m = widget.measurement;
    final accent = widget.intent.color;
    final names = _eligible(anchors).map((a) => a.object).toSet().toList()..sort();
    final group = _eligible(anchors).where((a) => a.object == object).toList();
    final raw = object == null ? null : models[AnchorRepository.modelKey(widget.intent, object!, _demo)];
    final retry = m.flags.isNotEmpty || !m.hasEmbedding;
    var title = 'Neural signature captured';
    var explanation = 'Pretrained YAMNet audio embedding. No physical-state diagnosis.';
    var method = 'Frozen YAMNet TFLite: 15,600 mono samples at 16 kHz → 1,024 learned features.';
    var needsTraining = _usesCalibration && raw == null;
    var different = false;
    double? distance, threshold, fill;
    String? maturity;
    String? predictionError;
    if (!retry && _usesCalibration && raw != null) {
      try {
        final json = Map<String, dynamic>.from(raw as Map);
        if (widget.intent == SensingIntent.vessel) {
          final prediction = FillCalibration.fromJson(json).predict(m.embedding);
          method += ' On-device ridge regression fitted to your known-fill examples.';
          if (prediction.supported) {
            fill = prediction.value;
            title = '${fill.round()}% estimated fill';
            explanation = 'Learned for this container and capture setup only.';
          } else {
            needsTraining = true;
            title = 'Outside learned range';
            explanation = 'The model cannot support this reading. Add known-fill examples or retry the same setup.';
          }
        } else if (widget.intent == SensingIntent.fruit) {
          final prediction = ClassCalibration.fromJson(json).predict(m.embedding);
          method += ' On-device labelled nearest-centroid classifier.';
          if (prediction.supported) {
            maturity = prediction.label;
            title = '${prediction.label} · learned estimate';
            explanation = 'Relative to your labelled fruit examples; not a food-safety assessment.';
          } else {
            needsTraining = true;
            title = 'Unknown fruit signature';
            explanation = 'Outside your labelled examples or ambiguous between classes. Teach more independent fruits.';
          }
        } else {
          final prediction = ReferenceCalibration.fromJson(json).predict(m.embedding);
          distance = prediction.distance;
          threshold = prediction.threshold;
          if (!threshold.isFinite) throw StateError('Invalid learned threshold');
          different = prediction.different;
          title = different ? 'Different from learned reference' : 'Learned reference match';
          explanation = 'Normal-only model trained on your taps. Difference is not a defect or material certification.';
          method += ' On-device normal centroid; threshold fitted from leave-one-out reference distances.';
        }
      } catch (e) {
        needsTraining = true;
        predictionError = 'Calibration is incompatible or invalid: $e';
      }
    }
    if (needsTraining && title == 'Neural signature captured') {
      title = loading ? 'Loading calibration' : 'Train this object';
      explanation = switch (widget.intent) {
        SensingIntent.vessel => 'Teach at least three known fill levels, three taps each. The local regressor learns from neural features—not peak-frequency interpolation.',
        SensingIntent.fruit => 'Teach at least two known maturity classes, three taps each, under the same fruit type/cultivar name.',
        _ => 'Record three clean reference taps to fit a local model on neural audio features. Older signal-only references must be re-recorded.',
      };
    }
    if (retry) {
      title = 'Record again';
      explanation = m.flags.isNotEmpty ? m.flags.first : 'Neural encoding is missing; this older trace cannot be re-encoded without its PCM.';
    }
    final savedContext = Map<String, String>.from(m.context)
      ..removeWhere((key, _) => const {
        'estimatedFillPercent', 'maturityClass', 'embeddingDistance', 'learnedThreshold',
        'referenceTapIds', 'firmnessIndex', 'regularizedDeviation', 'materialModel',
        'materialModelLabel', 'materialModelScore',
      }.contains(key));
    final record = m.withIntent(widget.intent.name).withContext({
      ...savedContext,
      'resultVersion': '3',
      'quality': retry ? 'Retry' : 'Clean signal',
      'interpretation': title,
      'suspicion': different ? 'Measured difference' : title,
      'methodNote': method,
      'modelStatus': retry ? 'invalid capture' : needsTraining ? 'calibration required' : 'prediction available',
      if (object != null) 'referenceObject': object!,
      if (group.isNotEmpty) 'referenceTapIds': group.expand((a) => a.taps.map((t) => t.id)).join(','),
      if (fill != null) 'estimatedFillPercent': '$fill',
      if (maturity != null) 'maturityClass': maturity,
      if (distance != null) 'embeddingDistance': '$distance',
      if (threshold != null) 'learnedThreshold': '$threshold',
    });
    final canSave = !retry && !needsTraining && !loading && error == null;
    return Theme(data: Theme.of(context).copyWith(colorScheme:
        Theme.of(context).colorScheme.copyWith(primary: accent, onPrimary: Colors.white)),
      child: Container(width: double.infinity, padding: const EdgeInsets.all(22),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(28)),
        child: Column(children: [
          Row(children: [
            if (_demo) const Text('Demo · not physical validation', style: TextStyle(color: C.muted, fontSize: 12)),
            const Spacer(),
            if (_usesCalibration) IconButton(tooltip: 'Learned calibration',
              onPressed: loading ? null : () => _setup(names), icon: const Icon(Icons.tune_rounded)),
          ]),
          Icon(retry ? Icons.refresh_rounded : needsTraining ? Icons.school_outlined :
            different ? Icons.location_on_outlined : Icons.check_rounded, size: 72, color: accent),
          const SizedBox(height: 20),
          Text(title, textAlign: TextAlign.center, style: TextStyle(color: accent, fontSize: 28, fontWeight: FontWeight.w700)),
          const SizedBox(height: 12),
          Text(explanation, textAlign: TextAlign.center, style: const TextStyle(color: C.muted)),
          if (!retry) ...[
            const SizedBox(height: 16),
            Row(children: [
              Metric('Frequency', '${m.peakHz.round()} Hz', accent: accent),
              Metric('Ring time', '${m.decayMs.round()} ms', accent: accent),
              if (distance != null) Metric('ML distance', distance.toStringAsFixed(3), accent: accent),
            ]),
          ],
          if (predictionError != null || error != null) Text(predictionError ?? error!, style: const TextStyle(color: C.red)),
          const SizedBox(height: 20),
          ActionButton(retry ? 'Record again' : needsTraining ? 'Train with examples' :
              widget.saved ? 'Update result' : 'Save learned result',
            onPressed: loading ? null : retry ? widget.onAgain : needsTraining ? _teach :
              canSave ? () => widget.onSave(record) : null),
          Wrap(alignment: WrapAlignment.center, children: [
            TextButton(onPressed: () => widget.onTrace(record), child: const Text('View signal')),
            if (!retry) TextButton(onPressed: widget.onAgain, child: const Text('Record again')),
            if (canSave) TextButton(onPressed: () => widget.onMap(record), child: const Text('Mark location')),
            if (error != null) TextButton(onPressed: _load, child: const Text('Reload calibration')),
          ]),
        ]),
      ));
  }
}
