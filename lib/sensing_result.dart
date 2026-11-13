import 'dart:math' as math;
import 'package:material_ui/material_ui.dart';
import 'design.dart';
import 'store.dart';
import 'sensing_flow.dart';

class SensingResult extends StatefulWidget {
  const SensingResult(
      {required this.measurement,
      required this.intent,
      required this.store,
      required this.onSave,
      required this.onMap,
      required this.onTrace,
      required this.onAgain,
      required this.saved,
      super.key});
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
  static const fruitTypes = ['Apple', 'Pear', 'Melon'];
  List<ReferenceAnchor> anchors = [];
  String? object, error;
  String fruit = 'Apple';
  bool loadingReferences = true;
  late final TextEditingController mass;

  bool get _demo =>
      widget.measurement.source.toLowerCase().contains('synthetic');
  bool get _compares =>
      widget.intent == SensingIntent.slab ||
      widget.intent == SensingIntent.devices ||
      widget.intent == SensingIntent.materials;
  bool get _usesReferences =>
      _compares || widget.intent == SensingIntent.vessel;

  @override
  void initState() {
    super.initState();
    mass = TextEditingController(text: widget.measurement.context['massGrams']);
    object = widget.measurement.context['referenceObject'];
    final savedFruit = widget.measurement.context['fruitType'];
    if (fruitTypes.contains(savedFruit)) fruit = savedFruit!;
    if (_usesReferences) {
      _load();
    } else {
      loadingReferences = false;
    }
  }

  List<ReferenceAnchor> _eligible(List<ReferenceAnchor> all) => all
      .where((a) =>
          a.intent == widget.intent &&
          a.synthetic == _demo &&
          a.object.trim().isNotEmpty &&
          a.taps.length >= 3 &&
          a.taps.every((t) =>
              t.flags.isEmpty &&
              t.peakHz.isFinite &&
              t.peakHz > 0 &&
              t.source.toLowerCase().contains('synthetic') == _demo))
      .toList();

  Future<void> _load({bool afterTeaching = false}) async {
    try {
      final all = await AnchorRepository.load();
      if (!mounted) return;
      final eligible = _eligible(all);
      final names = eligible.map((a) => a.object).toSet().toList()..sort();
      String signature(ReferenceAnchor a) =>
          '${a.object}|${a.level}|${a.taps.map((t) => t.id).join(',')}';
      final previous = _eligible(anchors).map(signature).toSet();
      final updatedNames = eligible
          .where((a) => !previous.contains(signature(a)))
          .map((a) => a.object)
          .toSet();
      setState(() {
        anchors = all;
        if (afterTeaching && updatedNames.length == 1) {
          object = updatedNames.single;
        } else if (!names.contains(object)) {
          object = names.length == 1 ? names.single : null;
        }
        loadingReferences = false;
        error = null;
      });
    } catch (_) {
      if (mounted) {
        setState(() {
          loadingReferences = false;
          error = 'References could not be loaded.';
        });
      }
    }
  }

  Future<void> _teachReference() async {
    await Navigator.of(context).push(sensingRoute<void>(context,
        GuidedReferenceScreen(intent: widget.intent, store: widget.store)));
    if (mounted) await _load(afterTeaching: true);
  }

  @override
  void dispose() {
    mass.dispose();
    super.dispose();
  }

  String _qualityLabel(Measurement measurement) {
    final flags = measurement.flags.join(' ').toLowerCase();
    if (flags.contains('snr') || flags.contains('nois')) return 'Too noisy';
    if (flags.contains('clip')) return 'Clipped';
    if (flags.contains('short')) return 'Tap too short';
    return 'Tap again';
  }

  void _referenceSetup(List<String> names, String? selected) {
    showModalBottomSheet<void>(
        context: context,
        showDragHandle: true,
        builder: (sheet) => SafeArea(
            child: Padding(
                padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  const SectionTitle('Reference'),
                  const SizedBox(height: 16),
                  if (names.length > 1) ...[
                    DropdownButtonFormField<String>(
                        initialValue: selected,
                        isExpanded: true,
                        decoration: const InputDecoration(
                            labelText: 'Reference object'),
                        items: names
                            .map((name) => DropdownMenuItem(
                                value: name,
                                child: Text(name,
                                    overflow: TextOverflow.ellipsis)))
                            .toList(),
                        onChanged: (value) {
                          setState(() => object = value);
                          Navigator.of(sheet).pop();
                        }),
                    const SizedBox(height: 12),
                  ] else if (selected != null)
                    Padding(
                        padding: const EdgeInsets.only(bottom: 12),
                        child: Text(selected)),
                  TextButton.icon(
                      onPressed: () {
                        Navigator.of(sheet).pop();
                        _teachReference();
                      },
                      icon: const Icon(Icons.add_rounded),
                      label: const Text('Record reference')),
                ]))));
  }

  @override
  Widget build(BuildContext context) {
    final m = widget.measurement;
    final accent = widget.intent.color;
    final eligible = _eligible(anchors);
    final names = eligible.map((a) => a.object).toSet().toList()..sort();
    final selected = names.contains(object) ? object : null;
    final refs = eligible.where((a) => a.object == selected).toList();
    final reference = _compares && refs.isNotEmpty ? refs.last.taps : null;
    final hasDecay = m.decayMs.isFinite && m.decayMs > 0;
    final retry = m.flags.isNotEmpty || !hasDecay;
    final quality = retry ? _qualityLabel(m) : 'Clean signal';
    var suspicion = 'Signature recorded';
    var title = 'Signature captured';
    var method = 'Peak: Hz · Decay: ms · HII: dB · SNR: dB.';
    var needsReference = false;
    var different = false;
    double? score, peakShift, decayShift, hiiShift;
    double? referencePeak, referenceDecay, referenceHii;
    double? fill, firmness;
    List<ReferenceAnchor> fillPoints = [];
    var retryReference = false;
    final grams = double.tryParse(mass.text.trim());
    final validMass = grams != null && grams.isFinite && grams > 0;

    if (_compares) {
      method = 'D = √mean((Δ / √(s² + floor²))²), peak / decay / HII. '
          'Floors: 5% reference peak, 10 ms, 3 dB. Difference: D > 3.';
      if (reference == null) {
        needsReference = !loadingReferences;
        title = loadingReferences ? 'Loading references' : 'Reference needed';
        suspicion = title;
      } else if (reference.any((t) => !t.decayMs.isFinite || t.decayMs <= 0)) {
        retryReference = true;
        title = 'Refresh reference';
        suspicion = 'Reference decay missing';
      } else if (!retry) {
        double mean(double Function(Measurement) read) =>
            reference.map(read).reduce((a, b) => a + b) / reference.length;
        referencePeak = mean((t) => t.peakHz);
        referenceDecay = mean((t) => t.decayMs);
        referenceHii = mean((t) => t.hiiDb);
        double deviation(double Function(Measurement) read, double floor) {
          final average = mean(read);
          final variance = reference
                  .map((t) => math.pow(read(t) - average, 2))
                  .reduce((a, b) => a + b) /
              (reference.length - 1);
          return (read(m) - average).abs() /
              math.sqrt(variance + floor * floor);
        }

        score = math.sqrt(
            (math.pow(deviation((t) => t.peakHz, referencePeak * .05), 2) +
                    math.pow(deviation((t) => t.decayMs, 10), 2) +
                    math.pow(deviation((t) => t.hiiDb, 3), 2)) /
                3);
        peakShift = 100 * (m.peakHz / referencePeak - 1);
        decayShift = m.decayMs - referenceDecay;
        hiiShift = m.hiiDb - referenceHii;
        different = score > 3;
        suspicion = different ? 'Measured difference' : 'Reference match';
        title = different ? 'Check this spot' : 'Reference match';
      }
    } else if (widget.intent == SensingIntent.vessel) {
      method = 'Fill = L₁ + (f − f₁) / (f₂ − f₁) × (L₂ − L₁). '
          'L: % · f: Hz. Piecewise interpolation between fill anchors.';
      fillPoints = refs
          .where((a) =>
              a.level != null &&
              a.level!.isFinite &&
              a.level! >= 0 &&
              a.level! <= 100)
          .toList()
        ..sort((a, b) => a.level!.compareTo(b.level!));
      final distinct =
          fillPoints.map((a) => a.level).toSet().length == fillPoints.length;
      final monotonic = fillPoints.length >= 3 &&
          distinct &&
          List.generate(fillPoints.length - 1,
                  (i) => fillPoints[i].peak > fillPoints[i + 1].peak + 1)
              .every((value) => value);
      if (!retry && monotonic) {
        for (var i = 0; i < fillPoints.length - 1; i++) {
          final a = fillPoints[i], b = fillPoints[i + 1];
          if (m.peakHz >= math.min(a.peak, b.peak) &&
              m.peakHz <= math.max(a.peak, b.peak)) {
            fill = a.level! +
                (m.peakHz - a.peak) / (b.peak - a.peak) * (b.level! - a.level!);
            break;
          }
        }
      }
      if (fill != null) {
        title = '${fill.round()}% fill';
        suspicion = 'Within reference range';
      } else {
        title = loadingReferences ? 'Loading references' : 'Reference needed';
        needsReference = !loadingReferences;
        suspicion = loadingReferences
            ? 'Loading references'
            : monotonic
                ? 'Outside reference range'
                : 'Reference setup needed';
        if (monotonic && !loadingReferences) title = 'Outside range';
      }
    } else if (widget.intent == SensingIntent.fruit) {
      method = 'Firmness index = f² × m^(2/3). f: Hz · m: kg. '
          'Index: Hz²·kg^(2/3).';
      if (validMass && !retry) {
        firmness = m.peakHz * m.peakHz * math.pow(grams / 1000, 2 / 3);
        title = 'Firmness index';
      } else {
        title = 'Signature captured';
      }
    }

    if (retry) {
      title = 'Tap again';
      suspicion = 'Comparison paused';
      needsReference = false;
      retryReference = false;
    }

    final savedContext = Map<String, String>.from(m.context)
      ..removeWhere((key, _) => const {
            'massGrams',
            'fruitType',
            'referenceObject',
            'referenceTapIds',
            'referencePeakHz',
            'referenceDecayMs',
            'referenceHiiDb',
            'referenceFillAnchors',
            'estimatedFillPercent',
            'firmnessIndex',
            'regularizedDeviation',
            'peakShiftPercent',
            'decayShiftMs',
            'hiiShiftDb',
            'interpretation',
            'methodNote',
            'quality',
            'suspicion',
            'resultVersion'
          }.contains(key));
    final record = m.withIntent(widget.intent.name).withContext({
      ...savedContext,
      'resultVersion': '2',
      'quality': quality,
      'suspicion': suspicion,
      'interpretation': title,
      'methodNote': method,
      if (widget.intent == SensingIntent.fruit) ...{
        'massGrams': mass.text.trim(),
        'fruitType': fruit,
      },
      if (selected != null) 'referenceObject': selected,
      if (refs.isNotEmpty)
        'referenceTapIds':
            refs.expand((a) => a.taps.map((t) => t.id)).join(','),
      if (referencePeak != null) 'referencePeakHz': '$referencePeak',
      if (referenceDecay != null) 'referenceDecayMs': '$referenceDecay',
      if (referenceHii != null) 'referenceHiiDb': '$referenceHii',
      if (fillPoints.isNotEmpty)
        'referenceFillAnchors':
            fillPoints.map((a) => '${a.level}:${a.peak}').join(','),
      if (fill != null) 'estimatedFillPercent': fill.toStringAsFixed(2),
      if (firmness != null) 'firmnessIndex': '$firmness',
      if (score != null) 'regularizedDeviation': '$score',
      if (peakShift != null) 'peakShiftPercent': '$peakShift',
      if (decayShift != null) 'decayShiftMs': '$decayShift',
      if (hiiShift != null) 'hiiShiftDb': '$hiiShift',
    });
    final setup = needsReference || retryReference;
    final mapPrimary =
        !retry && !setup && (widget.intent == SensingIntent.slab || different);
    final tracePrimary = !retry && widget.intent == SensingIntent.scope;
    final primaryLabel = retry
        ? 'Record again'
        : setup
            ? (retryReference ? 'Refresh reference' : 'Set reference')
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
            ? (selected == null && names.length > 1
                ? () => _referenceSetup(names, selected)
                : _teachReference)
            : mapPrimary
                ? () => widget.onMap(record)
                : tracePrimary
                    ? () => widget.onTrace(record)
                    : () => widget.onSave(record);
    final statusIcon = retry
        ? Icons.refresh_rounded
        : setup
            ? Icons.add_rounded
            : different
                ? Icons.location_on_outlined
                : widget.intent == SensingIntent.scope
                    ? Icons.graphic_eq_rounded
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
                        onPressed: loadingReferences
                            ? null
                            : () => _referenceSetup(names, selected),
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
                  child: Text(title,
                      key: ValueKey(title),
                      textAlign: TextAlign.center,
                      style: TextStyle(
                          color: accent,
                          fontSize: 30,
                          height: 1.12,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -.7))),
              if (retry) ...[
                const SizedBox(height: 10),
                Text(quality == 'Tap again' ? 'No clear ring' : quality,
                    style: const TextStyle(color: C.muted, fontSize: 15)),
              ] else ...[
                if (firmness != null) ...[
                  const SizedBox(height: 8),
                  Text(firmness.toStringAsFixed(0),
                      style: TextStyle(
                          color: accent,
                          fontSize: 52,
                          height: 1.1,
                          fontWeight: FontWeight.w600)),
                ],
                const SizedBox(height: 16),
                Row(children: [
                  Metric('Frequency', '${m.peakHz.round()} Hz', accent: accent),
                  Metric('Ring time', '${m.decayMs.round()} ms',
                      accent: accent),
                  if (score != null)
                    Metric('Deviation', score.toStringAsFixed(1),
                        accent: accent),
                  if (widget.intent == SensingIntent.scope)
                    Metric('SNR', '${m.snrDb.toStringAsFixed(0)} dB',
                        accent: accent),
                ]),
                if (widget.intent == SensingIntent.fruit) ...[
                  const SizedBox(height: 8),
                  Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Expanded(
                        child: DropdownButtonFormField<String>(
                            initialValue: fruit,
                            isExpanded: true,
                            decoration: const InputDecoration(
                                labelText: 'Fruit', isDense: true),
                            items: fruitTypes
                                .map((type) => DropdownMenuItem(
                                    value: type, child: Text(type)))
                                .toList(),
                            onChanged: (value) {
                              if (value != null) setState(() => fruit = value);
                            })),
                    const SizedBox(width: 12),
                    Expanded(
                        child: TextField(
                            controller: mass,
                            keyboardType: const TextInputType.numberWithOptions(
                                decimal: true),
                            decoration: InputDecoration(
                                labelText: 'Mass',
                                suffixText: 'g',
                                isDense: true,
                                errorText:
                                    mass.text.trim().isNotEmpty && !validMass
                                        ? 'Positive mass required'
                                        : null),
                            onChanged: (_) => setState(() {}))),
                  ]),
                  if (!validMass && mass.text.trim().isEmpty)
                    const Padding(
                        padding: EdgeInsets.only(top: 8),
                        child: Text('Add mass for firmness',
                            style: TextStyle(color: C.muted, fontSize: 12))),
                ],
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
