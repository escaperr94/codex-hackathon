import 'dart:convert';
import 'dart:math' as math;
import 'package:material_ui/material_ui.dart';
import 'package:cupertino_ui/cupertino_ui.dart' show CupertinoPageRoute;
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'design.dart';
import 'store.dart';

enum SensingIntent {
  slab('Slab', 'Apartment handover', 'Listen beneath\nthe surface.',
      Module.slab),
  vessel(
      'Vessel', 'Liquid inventory', 'Hear how much\nremains.', Module.vessel),
  fruit('Fruit', 'Firmness research', 'A different kind\nof fruit check.',
      Module.scope),
  devices('Devices', 'Electronics inspection', 'Hear the\nhardware.', Module.scope),
  scope('Scope', 'Acoustic exploration', 'Every object.\nIts own signature.',
      Module.scope);

  const SensingIntent(this.title, this.useCase, this.headline, this.module);
  final String title, useCase, headline;
  final Module module;
}

Route<T> sensingRoute<T>(BuildContext context, Widget child) =>
    _SensingRoute<T>(child, MediaQuery.disableAnimationsOf(context));

class _SensingRoute<T> extends CupertinoPageRoute<T> {
  _SensingRoute(Widget child, this.reduced) : super(builder: (_) => child);
  final bool reduced;
  @override
  Duration get transitionDuration => Duration(milliseconds: reduced ? 0 : 500);
  @override
  Duration get reverseTransitionDuration =>
      Duration(milliseconds: reduced ? 0 : 350);
}

class ChooseObjectScreen extends StatefulWidget {
  const ChooseObjectScreen(
      {required this.initial, required this.haptics, super.key});
  final SensingIntent initial;
  final bool haptics;
  @override
  State<ChooseObjectScreen> createState() => _ChooseObjectScreenState();
}

class _ChooseObjectScreenState extends State<ChooseObjectScreen> {
  late final PageController pages =
      PageController(initialPage: widget.initial.index);
  late int selected = widget.initial.index;
  @override
  void dispose() {
    pages.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Choose object')),
        body: SafeArea(
            child: Column(children: [
          Expanded(
              child: PageView.builder(
                  controller: pages,
                  itemCount: SensingIntent.values.length,
                  onPageChanged: (value) {
                    setState(() => selected = value);
                    if (widget.haptics) HapticFeedback.selectionClick();
                  },
                  itemBuilder: (context, index) {
                    final intent = SensingIntent.values[index];
                    return AnimatedBuilder(
                        animation: pages,
                        builder: (context, child) {
                          final distance =
                              !MediaQuery.disableAnimationsOf(context) &&
                                      pages.hasClients &&
                                      pages.position.haveDimensions
                                  ? ((pages.page ?? selected.toDouble()) -
                                          index)
                                      .clamp(-1.0, 1.0)
                                  : 0.0;
                          return Transform.translate(
                              offset: Offset(distance * 42, 0), child: child);
                        },
                        child: SingleChildScrollView(
                            padding: const EdgeInsets.fromLTRB(28, 24, 28, 16),
                            child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(intent.useCase.toUpperCase(),
                                      style: const TextStyle(
                                          fontFamily: 'monospace',
                                          color: C.muted,
                                          fontSize: 12,
                                          letterSpacing: 1.5)),
                                  const SizedBox(height: 20),
                                  Text(intent.headline,
                                      style: const TextStyle(
                                          fontSize: 40,
                                          height: 1.08,
                                          letterSpacing: -1.4,
                                          fontWeight: FontWeight.w600)),
                                  const SizedBox(height: 20),
                                  SizedBox(
                                      height: 230,
                                      width: double.infinity,
                                      child: CustomPaint(
                                          painter: MaterialStudy(intent))),
                                  const SizedBox(height: 16),
                                  Text(
                                      switch (intent) {
                                        SensingIntent.slab =>
                                          'Compare taps. Mark areas for review.',
                                        SensingIntent.vessel =>
                                          'Estimate fill with local references.',
                                        SensingIntent.fruit =>
                                          'Firmness proxy. Research only.',
                                        SensingIntent.devices =>
                                          'Compare the casing. Power off first.',
                                        SensingIntent.scope =>
                                          'Waveform, spectrum and decay.',
                                      },
                                      style: const TextStyle(
                                          color: C.muted,
                                          fontSize: 15,
                                          height: 1.5)),
                                ])));
                  })),
          Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(children: [
                for (final intent in SensingIntent.values)
                  Expanded(
                      child: TextButton(
                          onPressed: () {
                            if (MediaQuery.disableAnimationsOf(context)) {
                              pages.jumpToPage(intent.index);
                            } else {
                              pages.animateToPage(intent.index,
                                  duration: const Duration(milliseconds: 480),
                                  curve: Curves.easeOutCubic);
                            }
                          },
                          style: TextButton.styleFrom(
                              padding: const EdgeInsets.symmetric(horizontal: 4),
                              minimumSize: const Size(0, 48),
                              foregroundColor:
                                  selected == intent.index ? C.blue : C.muted),
                          child: Text(intent.title))),
              ])),
          Padding(
              padding: const EdgeInsets.fromLTRB(24, 12, 24, 24),
              child: SizedBox(
                  width: double.infinity,
                  child: ActionButton(
                      'Use ${SensingIntent.values[selected].title}',
                      onPressed: () => Navigator.pop(
                          context, SensingIntent.values[selected])))),
        ])),
      );
}

/// Schematic subject illustration, never an inferred interior or fill level.
class MaterialStudy extends CustomPainter {
  const MaterialStudy(this.intent);
  final SensingIntent intent;
  @override
  void paint(Canvas canvas, Size size) {
    final ink = Paint()
      ..color = C.ink
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.4;
    final blue = Paint()
      ..color = C.blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6;
    final center = size.center(Offset.zero);
    if (intent == SensingIntent.slab) {
      final path = Path()
        ..moveTo(22, 88)
        ..lineTo(size.width * .62, 35)
        ..lineTo(size.width - 22, 94)
        ..lineTo(size.width * .4, 151)
        ..close();
      canvas.drawPath(path, ink);
      canvas.drawPath(
          Path()
            ..moveTo(22, 100)
            ..lineTo(size.width * .4, 164)
            ..lineTo(size.width - 22, 107),
          ink);
      for (double x = 28; x < size.width - 26; x += 5) {
        for (double y = 173; y < 207; y += 5) {
          if ((x.toInt() + y.toInt()) % 3 != 0) {
            canvas.drawCircle(Offset(x, y), .7,
                Paint()..color = C.muted.withValues(alpha: .5));
          }
        }
      }
      for (int i = 0; i < 3; i++) {
        canvas.drawOval(
            Rect.fromCenter(
                center: Offset(center.dx, 121),
                width: 30 + i * 25,
                height: 8 + i * 7),
            blue);
      }
    } else if (intent == SensingIntent.vessel) {
      final rect = Rect.fromCenter(center: center, width: 110, height: 175);
      canvas.drawRRect(
          RRect.fromRectAndRadius(rect, const Radius.circular(22)), ink);
      canvas.drawOval(
          Rect.fromCenter(
              center: Offset(center.dx, rect.top + 14), width: 110, height: 28),
          ink);
      canvas.drawOval(
          Rect.fromCenter(
              center: Offset(center.dx, center.dy + 16),
              width: 108,
              height: 22),
          blue);
      for (double y = center.dy + 24; y < rect.bottom - 12; y += 5) {
        for (double x = rect.left + 10; x < rect.right - 9; x += 5) {
          canvas.drawCircle(
              Offset(x, y), .6, Paint()..color = C.blue.withValues(alpha: .45));
        }
      }
    } else if (intent == SensingIntent.fruit) {
      canvas.drawOval(
          Rect.fromCenter(center: center, width: 154, height: 161), ink);
      canvas.drawPath(
          Path()
            ..moveTo(center.dx, center.dy - 80)
            ..quadraticBezierTo(center.dx + 2, center.dy - 115, center.dx + 24,
                center.dy - 103),
          ink);
      for (int i = 0; i < 4; i++) {
        canvas.drawArc(
            Rect.fromCenter(
                center: center, width: 64 + i * 22, height: 100 + i * 15),
            -.8,
            1.6,
            false,
            blue);
      }
    } else if (intent == SensingIntent.devices) {
      final body = Rect.fromCenter(center: center, width: 205, height: 142);
      canvas.drawRRect(RRect.fromRectAndRadius(body, const Radius.circular(14)), ink);
      final chip = Rect.fromCenter(center: center, width: 66, height: 54);
      canvas.drawRRect(RRect.fromRectAndRadius(chip, const Radius.circular(6)), blue);
      for (var i = 0; i < 5; i++) {
        final y = chip.top + 7 + i * 10;
        canvas.drawLine(Offset(chip.left - 10, y), Offset(chip.left, y), blue);
        canvas.drawLine(Offset(chip.right, y), Offset(chip.right + 10, y), blue);
      }
      for (final dx in [-1, 1]) {
        for (final dy in [-1, 1]) {
          canvas.drawCircle(center + Offset(dx * 88.0, dy * 57.0), 3, ink);
        }
      }
      canvas.drawPath(Path()..moveTo(chip.right + 10, center.dy)
          ..lineTo(body.right - 28, center.dy)..lineTo(body.right - 28, body.bottom - 20), ink);
      canvas.drawPath(Path()..moveTo(chip.left - 10, center.dy)
          ..lineTo(body.left + 28, center.dy)..lineTo(body.left + 28, body.top + 20), ink);
    } else {
      final path = Path();
      for (int i = 0; i <= 200; i++) {
        final x = i / 200 * size.width;
        final y = center.dy +
            math.sin(i * .17) * math.exp(-math.pow((i - 100) / 58, 2)) * 65;
        if (i == 0) {
          path.moveTo(x, y);
        } else {
          path.lineTo(x, y);
        }
      }
      canvas.drawPath(path, blue);
      canvas.drawLine(Offset(0, center.dy), Offset(size.width, center.dy),
          Paint()..color = C.line);
    }
  }

  @override
  bool shouldRepaint(covariant MaterialStudy old) => old.intent != intent;
}

class ReferenceAnchor {
  const ReferenceAnchor(this.intent, this.object, this.level, this.taps);
  final SensingIntent intent;
  final String object;
  final double? level;
  final List<Measurement> taps;
  double get peak =>
      taps.map((m) => m.peakHz).reduce((a, b) => a + b) / taps.length;
  bool get synthetic =>
      taps.any((m) => m.source.toLowerCase().contains('synthetic'));
  Map<String, dynamic> toJson() => {
        'intent': intent.name,
        'object': object,
        'level': level,
        'taps': taps.map((m) => m.toJson()).toList()
      };
  factory ReferenceAnchor.fromJson(Map<String, dynamic> j) => ReferenceAnchor(
      SensingIntent.values.byName(j['intent'] as String),
      j['object'] as String,
      (j['level'] as num?)?.toDouble(),
      (j['taps'] as List)
          .map((m) => Measurement.fromJson(Map<String, dynamic>.from(m as Map)))
          .toList());
}

class AnchorRepository {
  static Future<List<ReferenceAnchor>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('thud.anchors.v1');
    if (raw == null) return [];
    return (jsonDecode(raw) as List)
        .map((j) =>
            ReferenceAnchor.fromJson(Map<String, dynamic>.from(j as Map)))
        .toList();
  }

  static Future<void> save(ReferenceAnchor anchor) async {
    if (anchor.object.trim().isEmpty ||
        anchor.taps.length != 3 ||
        anchor.taps.any(
            (m) => m.flags.isNotEmpty || !m.peakHz.isFinite || m.peakHz <= 0)) {
      throw Exception('A reference needs three clean measurable taps.');
    }
    if (anchor.taps.any((m) =>
        m.source.toLowerCase().contains('synthetic') != anchor.synthetic)) {
      throw Exception('Do not mix synthetic and microphone reference taps.');
    }
    if (anchor.intent == SensingIntent.vessel &&
        (anchor.level == null ||
            !anchor.level!.isFinite ||
            anchor.level! < 0 ||
            anchor.level! > 100)) {
      throw Exception('Known fill must be between 0 and 100%.');
    }
    final all = await load();
    all.removeWhere((a) =>
        a.intent == anchor.intent &&
        a.object == anchor.object &&
        a.level == anchor.level &&
        a.synthetic == anchor.synthetic);
    all.add(anchor);
    final prefs = await SharedPreferences.getInstance();
    if (!await prefs.setString(
        'thud.anchors.v1', jsonEncode(all.map((a) => a.toJson()).toList()))) {
      throw Exception('Could not save reference.');
    }
  }
}

class GuidedReferenceScreen extends StatefulWidget {
  const GuidedReferenceScreen(
      {required this.intent, required this.store, super.key});
  final SensingIntent intent;
  final ThudStore store;
  @override
  State<GuidedReferenceScreen> createState() => _GuidedReferenceScreenState();
}

class _GuidedReferenceScreenState extends State<GuidedReferenceScreen>
    with WidgetsBindingObserver {
  final capture = CaptureService();
  final object = TextEditingController();
  final level = TextEditingController();
  final taps = <Measurement>[];
  bool busy = false, recording = false, saved = false;
  bool? synthetic;
  String? error;
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed && recording) _cancel();
  }

  Future<void> _cancel() async {
    if (busy) return;
    setState(() => busy = true);
    try {
      await capture.cancel();
    } catch (e) {
      if (mounted) setState(() => error = e.toString());
    } finally {
      if (mounted) {
        setState(() {
          recording = false;
          busy = false;
        });
      }
    }
  }

  Future<void> _take(bool demo) async {
    if (busy || taps.length == 3 || saved) return;
    setState(() {
      busy = true;
      error = null;
    });
    try {
      if (synthetic != null && synthetic != demo) {
        throw Exception(
            'Do not mix microphone and synthetic reference taps. Reset first.');
      }
      if (object.text.trim().isEmpty) {
        throw Exception('Name the reference object before recording.');
      }
      final knownFill = double.tryParse(level.text);
      if (widget.intent == SensingIntent.vessel &&
          (knownFill == null ||
              !knownFill.isFinite ||
              knownFill < 0 ||
              knownFill > 100)) {
        throw Exception('Enter the known fill from 0 to 100% first.');
      }
      if (!demo && !recording) {
        await capture.start();
        if (!mounted) {
          await capture.cancel();
          return;
        }
        setState(() => recording = true);
        return;
      }
      final m = demo
          ? await capture.sample(widget.intent.module,
              fillFraction: widget.intent == SensingIntent.vessel
                  ? knownFill! / 100
                  : null)
          : await capture.stop(widget.intent.module);
      if (!mounted) return;
      setState(() => recording = false);
      if (m.flags.isNotEmpty) {
        throw Exception('Retry this tap: ${m.flags.join(', ')}');
      }
      if (!m.peakHz.isFinite || m.peakHz <= 0) {
        throw Exception('No measurable resonance. Retry this tap.');
      }
      if (taps.isNotEmpty && (m.peakHz / taps.first.peakHz - 1).abs() > .12) {
        throw Exception(
            'Resonance changed by more than 12%. Keep the object, tap point and tool unchanged.');
      }
      for (final previous in taps) {
        final a = previous.spectrum, b = m.spectrum;
        var dot = 0.0, aa = 0.0, bb = 0.0;
        for (var i = 0; i < math.min(a.length, b.length); i++) {
          final hz = 20 * math.pow(1200, i / 255);
          if (hz < 200 || hz > 6000) continue;
          final av = a[i] * 100 - 100, bv = b[i] * 100 - 100;
          dot += av * bv;
          aa += av * av;
          bb += bv * bv;
        }
        if (aa == 0 || bb == 0 || dot / math.sqrt(aa * bb) < .82) {
          throw Exception(
              'LOW COHERENCE: keep the tap point, tool and phone distance consistent.');
        }
      }
      setState(() {
        synthetic = demo;
        taps.add(m);
      });
      if (widget.store.haptics) HapticFeedback.selectionClick();
    } catch (e) {
      if (mounted) {
        setState(() {
          recording = false;
          error = e.toString();
        });
      }
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  Future<void> _save() async {
    final fill = double.tryParse(level.text);
    if (object.text.trim().isEmpty ||
        (widget.intent == SensingIntent.vessel &&
            (fill == null || !fill.isFinite || fill < 0 || fill > 100))) {
      setState(() => error =
          'Name the exact object; vessel anchors also need a known fill from 0 to 100%.');
      return;
    }
    setState(() => busy = true);
    try {
      await AnchorRepository.save(ReferenceAnchor(
          widget.intent,
          object.text.trim(),
          widget.intent == SensingIntent.vessel ? fill : null,
          List.of(taps)));
      if (mounted) setState(() => saved = true);
    } catch (e) {
      if (mounted) setState(() => error = e.toString());
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    capture.dispose();
    object.dispose();
    level.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(title: Text('${widget.intent.title} reference')),
      body: AppPage(children: [
        PageHeading(
            saved
                ? 'Reference saved.'
                : widget.intent == SensingIntent.slab
                    ? 'First, a solid\nreference.'
                    : 'Teach this\nexact object.',
            subtitle: widget.intent == SensingIntent.devices
                ? 'Power off. Unplug. Three gentle casing taps.'
                : 'Three taps. Same spot, tool and distance.'),
        const SizedBox(height: 24),
        TextField(
            controller: object,
            enabled: taps.isEmpty && !recording,
            decoration: const InputDecoration(labelText: 'Reference name')),
        if (widget.intent == SensingIntent.vessel)
          TextField(
              controller: level,
              enabled: taps.isEmpty && !recording,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: 'Known fill (%)')),
        const SizedBox(height: 32),
        Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: List.generate(
                3,
                (i) => AnimatedContainer(
                    duration: Duration(
                        milliseconds:
                            MediaQuery.disableAnimationsOf(context) ? 0 : 400),
                    width: 72,
                    height: 72,
                    decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: i < taps.length ? C.blue : C.pale),
                    child: Center(
                        child: Text(
                            i < taps.length
                                ? '${taps[i].peakHz.round()}'
                                : '${i + 1}',
                            style: TextStyle(
                                color: i < taps.length ? Colors.white : C.muted,
                                fontSize: 18)))))),
        const SizedBox(height: 20),
        Center(
            child: Text('${taps.length} / 3 taps',
                style:
                    const TextStyle(fontFamily: 'monospace', color: C.muted))),
        const SizedBox(height: 32),
        if (synthetic == true)
          const Text('Demo reference · not a field reading',
              style: TextStyle(color: C.amber)),
        if (error != null)
          Padding(
              padding: const EdgeInsets.symmetric(vertical: 16),
              child: Text(error!, style: const TextStyle(color: C.red))),
        if (!saved && taps.length < 3) ...[
          ActionButton(recording ? 'Stop & analyze tap' : 'Record next tap',
              onPressed: busy ? null : () => _take(false),
              icon: recording ? Icons.stop_rounded : Icons.mic_rounded),
          if (recording)
            TextButton(
                onPressed: busy ? null : _cancel,
                child: const Text('Cancel recording')),
          if (!recording)
            TextButton(
                onPressed: busy ? null : () => _take(true),
                child: const Text('Use synthetic tap')),
        ],
        if (taps.length == 3 && !saved)
          ActionButton('Save reference', onPressed: busy ? null : _save),
        if (saved)
          ActionButton('Back to measurement',
              onPressed: () => Navigator.pop(context)),
        if (saved && widget.intent == SensingIntent.vessel)
          TextButton(
              onPressed: () => setState(() {
                    saved = false;
                    taps.clear();
                    synthetic = null;
                    level.clear();
                    error = null;
                  }),
              child: const Text('Teach another fill level')),
        if (taps.isNotEmpty && !recording && !saved)
          TextButton(
              onPressed: busy
                  ? null
                  : () => setState(() {
                        taps.clear();
                        synthetic = null;
                        error = null;
                      }),
              child: const Text('Reset taps')),
        const SizedBox(height: 24),
        const ExpansionTile(
          tilePadding: EdgeInsets.zero,
          title: Text('Reference limits', style: TextStyle(fontSize: 14)),
          children: [
            Text(
                'A reference describes acoustic repeatability, not safety, ripeness or a verified defect. Vessel estimates need at least three known fill levels on the same container.',
                style: TextStyle(color: C.muted, height: 1.5))
          ],
        ),
      ]));
}
