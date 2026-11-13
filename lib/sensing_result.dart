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
  List<ReferenceAnchor> anchors = [];
  String? object, error;
  late final TextEditingController mass;
  @override
  void initState() {
    super.initState();
    mass = TextEditingController(text: widget.measurement.context['massGrams']);
    object = widget.measurement.context['referenceObject'];
    _load();
  }

  Future<void> _load() async {
    try {
      final all = await AnchorRepository.load();
      if (mounted) {
        setState(() {
          anchors = all;
          error = null;
        });
      }
    } catch (e) {
      if (mounted) setState(() => error = 'Reference storage unavailable: $e');
    }
  }

  @override
  void dispose() {
    mass.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final m = widget.measurement;
    final demo = m.source.toLowerCase().contains('synthetic');
    final eligible = anchors
        .where((a) => a.intent == widget.intent && a.synthetic == demo)
        .toList();
    final names = eligible.map((a) => a.object).toSet().toList();
    final selected = names.contains(object) ? object : null;
    final refs = eligible.where((a) => a.object == selected).toList();
    var title = 'Acoustic signature captured.';
    var explanation =
        'Explore the measured signal. No material diagnosis attached.';
    double? fill;
    var summary = 'Explore the captured signal.';
    if (m.flags.isNotEmpty) {
      title = 'Reading needs a retry.';
      summary = 'Improve the signal and tap again.';
      explanation =
          '${m.flags.join(' · ')}. Improve the signal before interpreting it.';
    } else if (widget.intent == SensingIntent.slab ||
        widget.intent == SensingIntent.devices) {
      final reference = refs.isNotEmpty ? refs.last.taps : null;
      if (reference == null) {
        title = 'Reference needed.';
        summary = widget.intent == SensingIntent.devices
            ? 'Set a known-good casing reference.'
            : 'Compare with a known solid tile.';
        explanation = widget.intent == SensingIntent.devices
            ? 'Compare the same model, casing, support and tap point with a known-good device. Power off and unplug before gentle casing taps. Do not tap screens, exposed circuits, hot or swollen batteries. Sound cannot verify electrical function or safety.'
            : 'Teach a known solid tile on this floor before comparing. You can still mark a location for review.';
      } else {
        if (m.decayMs <= 0 || reference.any((t) => t.decayMs <= 0)) {
          title = 'Decay could not be measured.';
          summary = 'Retap in a quieter setting.';
          explanation =
              'A complete reference comparison needs measurable decay. Retap in a quieter setting; you can still inspect the raw trace.';
        } else {
          final mean = reference.map((t) => t.peakHz).reduce((a, b) => a + b) /
              reference.length;
          final decay =
              reference.map((t) => t.decayMs).reduce((a, b) => a + b) /
                  reference.length;
          double deviation(double Function(Measurement) read, double floor) {
            final average =
                reference.map(read).reduce((a, b) => a + b) / reference.length;
            final variance = reference
                    .map((t) => math.pow(read(t) - average, 2))
                    .reduce((a, b) => a + b) /
                (reference.length - 1);
            return (read(m) - average).abs() /
                math.sqrt(variance + floor * floor);
          }

          final score = math.sqrt(
              (math.pow(deviation((t) => t.peakHz, mean * .05), 2) +
                      math.pow(deviation((t) => t.decayMs, 10), 2) +
                      math.pow(deviation((t) => t.hiiDb, 3), 2)) /
                  3);
          title = score > 3
              ? 'Acoustic anomaly · review.'
              : 'Within reference envelope.';
          summary = score > 3
              ? 'Acoustic difference. Inspect this spot.'
              : 'Acoustically similar. Not a safety verdict.';
          explanation =
              'Peak shift ${(100 * (m.peakHz / mean - 1)).toStringAsFixed(1)}% · decay Δ ${(m.decayMs - decay).toStringAsFixed(0)} ms versus “$selected”. Regularized deviation ${score.toStringAsFixed(2)}; review threshold 3.0 (unvalidated screening rule, not defect probability). Floors: 5% peak, 10 ms decay, 3 dB HII. ${widget.intent == SensingIntent.devices ? "Casing differences need physical inspection, not a component-failure verdict." : "A hollow-tile or acceptance verdict still requires inspector verification."}';
        }
        if (widget.intent == SensingIntent.devices) {
          title = title == 'Within reference envelope.'
              ? 'Casing matches reference.'
              : title;
          summary = 'Acoustic screening · not an electrical test.';
          explanation +=
              ' This comparison concerns casing acoustics only. It cannot identify a failed component, certify electrical safety or replace functional diagnostics. Compare the same model, support and tap position with power disconnected.';
        }
      }
    } else if (widget.intent == SensingIntent.vessel) {
      final points = refs.where((a) => a.level != null).toList()
        ..sort((a, b) => a.level!.compareTo(b.level!));
      final distinct =
          points.map((a) => a.level).toSet().length == points.length;
      final monotonic = points.length >= 3 &&
          distinct &&
          List.generate(points.length - 1,
              (i) => points[i].peak > points[i + 1].peak + 1).every((v) => v);
      if (monotonic &&
          m.peakHz <= points.first.peak &&
          m.peakHz >= points.last.peak) {
        for (var i = 0; i < points.length - 1; i++) {
          final a = points[i], b = points[i + 1];
          if (m.peakHz <= a.peak && m.peakHz >= b.peak) {
            fill = a.level! +
                (a.peak - m.peakHz) / (a.peak - b.peak) * (b.level! - a.level!);
            break;
          }
        }
        title = '${fill?.round()}% · calibrated estimate';
        summary = 'Estimated fill · same container only.';
        explanation =
            'Piecewise interpolation between known fill anchors for “$selected”. Assumes the same container, liquid, support and tap point. Not a validated accuracy interval.';
      } else {
        title = '${m.peakHz.round()} Hz · shell resonance';
        summary = 'Set three known fill references.';
        explanation = points.length < 3
            ? 'Teach at least three known fill levels on the exact container. A resonance alone cannot determine a water percentage.'
            : 'Anchors are not strictly decreasing, or this reading is outside their range. No fill estimate available; recalibrate rather than extrapolate.';
      }
    } else if (widget.intent == SensingIntent.fruit) {
      final grams = double.tryParse(mass.text);
      if (grams != null && grams.isFinite && grams > 0) {
        final proxy = m.peakHz * m.peakHz * math.pow(grams / 1000, 2 / 3);
        title = '${proxy.toStringAsFixed(0)} · firmness proxy';
        summary = 'Research proxy · not a ripeness verdict.';
        explanation =
            'f² × mass^(2/3), Hz²·kg^(2/3). Compare only matched fruit variety, shape, support and tap method. The dominant peak may not be the fruit’s fundamental mode. No ripe/unripe or food-safety classification.';
      } else {
        title = 'Explore fruit firmness.';
        summary = 'Enter mass · research only.';
        explanation =
            'Enter mass in grams for a resonance-based proxy. This research mode cannot determine ripeness or quality from a tap alone.';
      }
    }
    final record = m.withIntent(widget.intent.name).withContext({
      ...(Map<String, String>.from(m.context)
        ..removeWhere((key, _) => const [
              'massGrams',
              'referenceObject',
              'referenceTapIds',
              'estimatedFillPercent',
              'interpretation',
              'methodNote'
            ].contains(key))),
      if (widget.intent == SensingIntent.fruit) 'massGrams': mass.text,
      if (selected != null) 'referenceObject': selected,
      if (refs.isNotEmpty)
        'referenceTapIds':
            refs.expand((a) => a.taps.map((t) => t.id)).join(','),
      if (fill != null) 'estimatedFillPercent': fill.toStringAsFixed(2),
      'interpretation': title,
      'methodNote': explanation,
    });
    return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(22),
        decoration: BoxDecoration(
            color: Colors.white, borderRadius: BorderRadius.circular(28)),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(
              '${widget.intent.title.toUpperCase()} / ${demo ? 'DEMO · NOT AN INSPECTION' : 'RECORDED'}',
              style: const TextStyle(
                  color: C.muted,
                  fontFamily: 'monospace',
                  fontSize: 11,
                  letterSpacing: .6)),
          const SizedBox(height: 16),
          AnimatedSwitcher(
              duration: Duration(
                  milliseconds:
                      MediaQuery.disableAnimationsOf(context) ? 0 : 420),
              child: Text(title,
                  key: ValueKey(title),
                  style: const TextStyle(
                      color: C.ink,
                      fontSize: 28,
                      height: 1.12,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -.7))),
          const SizedBox(height: 12),
          Text(summary,
              style:
                  const TextStyle(color: C.muted, height: 1.5, fontSize: 13)),
          if (fill != null)
            Padding(
                padding: const EdgeInsets.only(top: 16),
                child: TweenAnimationBuilder<double>(
                    tween: Tween(begin: 0, end: fill / 100),
                    duration: Duration(
                        milliseconds:
                            MediaQuery.disableAnimationsOf(context) ? 0 : 650),
                    curve: Curves.easeOutCubic,
                    builder: (context, value, _) => SizedBox(
                        height: 90,
                        width: double.infinity,
                        child: CustomPaint(painter: _FillPainter(value))))),
          if (widget.intent == SensingIntent.fruit)
            Padding(
                padding: const EdgeInsets.only(top: 16),
                child: TextField(
                    controller: mass,
                    keyboardType: TextInputType.number,
                    decoration:
                        const InputDecoration(labelText: 'Fruit mass (g)'),
                    onChanged: (_) => setState(() {}))),
          if (widget.intent == SensingIntent.slab ||
              widget.intent == SensingIntent.vessel ||
              widget.intent == SensingIntent.devices) ...[
            const SizedBox(height: 12),
            if (names.isNotEmpty)
              DropdownButtonFormField<String>(
                  key: ValueKey(names.join('|')),
                  initialValue: selected,
                  decoration:
                      const InputDecoration(labelText: 'Reference object'),
                  items: names
                      .map((n) => DropdownMenuItem(value: n, child: Text(n)))
                      .toList(),
                  onChanged: (v) => setState(() => object = v)),
            TextButton(
                onPressed: () async {
                  await Navigator.of(context).push(sensingRoute<void>(
                      context,
                      GuidedReferenceScreen(
                          intent: widget.intent, store: widget.store)));
                  await _load();
                },
                child: const Text('Set reference')),
          ],
          if (error != null) Text(error!, style: const TextStyle(color: C.red)),
          ExpansionTile(
              tilePadding: EdgeInsets.zero,
              title: const Text('Evidence & limits',
                  style: TextStyle(color: C.ink, fontSize: 14)),
              children: [
                Text(explanation,
                    style: const TextStyle(
                        color: C.muted, height: 1.5, fontSize: 13)),
                const SizedBox(height: 12),
                Row(children: [
                  Metric('Peak', '${m.peakHz.round()} Hz', accent: C.blue),
                  Metric(
                      'Decay',
                      m.decayMs > 0
                          ? '${m.decayMs.round()} ms'
                          : 'Unavailable'),
                ]),
                Text(
                    'HII ${m.hiiDb.toStringAsFixed(1)} dB · estimated SNR ${m.snrDb.toStringAsFixed(1)} dB\n${m.flags.isEmpty ? 'No automated quality flags. Not a quality certificate.' : m.flags.join(', ')}',
                    style: const TextStyle(color: C.muted, height: 1.5)),
                TextButton(
                    onPressed: () => widget.onTrace(record),
                    child: const Text('View signal')),
              ]),
          const SizedBox(height: 16),
          SizedBox(
              width: double.infinity,
              child: ActionButton('Mark location',
                  icon: Icons.location_on_outlined,
                  onPressed: () => widget.onMap(record))),
          const SizedBox(height: 8),
          Row(children: [
            Expanded(
                child: TextButton(
                    onPressed: () => widget.onSave(record),
                    child: Text(widget.saved ? 'Update' : 'Save'))),
            Expanded(
                child: TextButton(
                    onPressed: widget.onAgain, child: const Text('Retap')))
          ]),
        ]));
  }
}

class _FillPainter extends CustomPainter {
  const _FillPainter(this.value);
  final double value;
  @override
  void paint(Canvas c, Size s) {
    final rect = Rect.fromLTWH(0, 4, s.width, s.height - 8);
    c.save();
    c.clipRRect(RRect.fromRectAndRadius(rect, const Radius.circular(16)));
    c.drawRect(rect, Paint()..color = C.pale);
    c.drawRect(
        Rect.fromLTRB(
            0, rect.bottom - rect.height * value, s.width, rect.bottom),
        Paint()..color = C.blue.withValues(alpha: .2));
    c.restore();
    c.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(16)),
        Paint()
          ..color = C.line
          ..style = PaintingStyle.stroke);
  }

  @override
  bool shouldRepaint(covariant _FillPainter old) => old.value != value;
}
