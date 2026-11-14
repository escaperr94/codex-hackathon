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
  fruit(
      'Fruit', 'Fruit firmness', 'Find the\nfruit’s signature.', Module.scope),
  devices(
      'Devices', 'Electronics inspection', 'Hear the\nhardware.', Module.scope),
  materials('Material', 'Material comparison', 'Meet your\nmaterial match.',
      Module.scope),
  scope('Scope', 'Acoustic exploration', 'Every object.\nIts own signature.',
      Module.scope);

  const SensingIntent(this.title, this.useCase, this.headline, this.module);
  final String title, useCase, headline;
  final Module module;

  Color get color => switch (this) {
        slab => C.blue,
        vessel => const Color(0xFF087F83),
        fruit => const Color(0xFFB84B20),
        devices => const Color(0xFF7444BE),
        materials => const Color(0xFFAA3860),
        scope => const Color(0xFF293A4C),
      };

  IconData get icon => switch (this) {
        slab => Icons.grid_view_rounded,
        vessel => Icons.water_drop_outlined,
        fruit => Icons.eco_rounded,
        devices => Icons.memory_rounded,
        materials => Icons.layers_rounded,
        scope => Icons.graphic_eq_rounded,
      };
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
                                          'Tap. Measure firmness. Compare fruit.',
                                        SensingIntent.devices =>
                                          'Compare the casing. Power off first.',
                                        SensingIntent.materials =>
                                          'Compare a sample with your original.',
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
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 4),
                              minimumSize: const Size(0, 48),
                              foregroundColor: selected == intent.index
                                  ? intent.color
                                  : C.muted),
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
      ..color = intent.color
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
      canvas.drawRRect(
          RRect.fromRectAndRadius(body, const Radius.circular(14)), ink);
      final chip = Rect.fromCenter(center: center, width: 66, height: 54);
      canvas.drawRRect(
          RRect.fromRectAndRadius(chip, const Radius.circular(6)), blue);
      for (var i = 0; i < 5; i++) {
        final y = chip.top + 7 + i * 10;
        canvas.drawLine(Offset(chip.left - 10, y), Offset(chip.left, y), blue);
        canvas.drawLine(
            Offset(chip.right, y), Offset(chip.right + 10, y), blue);
      }
      for (final dx in [-1, 1]) {
        for (final dy in [-1, 1]) {
          canvas.drawCircle(center + Offset(dx * 88.0, dy * 57.0), 3, ink);
        }
      }
      canvas.drawPath(
          Path()
            ..moveTo(chip.right + 10, center.dy)
            ..lineTo(body.right - 28, center.dy)
            ..lineTo(body.right - 28, body.bottom - 20),
          ink);
      canvas.drawPath(
          Path()
            ..moveTo(chip.left - 10, center.dy)
            ..lineTo(body.left + 28, center.dy)
            ..lineTo(body.left + 28, body.top + 20),
          ink);
    } else if (intent == SensingIntent.materials) {
      for (var i = 2; i >= 0; i--) {
        final offset = i * 22.0;
        canvas.drawRRect(
            RRect.fromRectAndRadius(
                Rect.fromCenter(
                    center: center + Offset(offset - 22, 22 - offset),
                    width: 150,
                    height: 100),
                const Radius.circular(12)),
            i == 1 ? blue : ink);
      }
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

/// A user-recorded profile; absent labels preserve legacy reference matching.
class ReferenceAnchor {
  const ReferenceAnchor(this.intent, this.object, this.level, this.taps,
      {this.label, this.metadata = const {}});
  final SensingIntent intent;
  final String object;
  final double? level;
  final List<Measurement> taps;
  final String? label;
  final Map<String, String> metadata;
  String get displayLabel => label ?? 'Reference';
  double get peak =>
      taps.map((m) => m.peakHz).reduce((a, b) => a + b) / taps.length;
  bool get synthetic =>
      taps.any((m) => m.source.toLowerCase().contains('synthetic'));
  static List<String> labelsFor(SensingIntent intent) => switch (intent) {
        SensingIntent.slab => const ['Solid', 'Hollow'],
        SensingIntent.materials => const ['Original', 'Suspect'],
        SensingIntent.devices => const ['Reference', 'Anomaly'],
        _ => const ['Reference'],
      };
  static bool clean(Measurement m) =>
      m.flags.isEmpty &&
      m.peakHz.isFinite &&
      m.peakHz > 0 &&
      m.decayMs.isFinite &&
      m.decayMs > 0 &&
      m.hiiDb.isFinite;
  Map<String, dynamic> toJson() => {
        'intent': intent.name,
        'object': object,
        'level': level,
        'taps': taps.map((m) => m.toJson()).toList(),
        if (label != null) 'label': label,
        if (metadata.isNotEmpty) 'metadata': metadata,
      };
  factory ReferenceAnchor.fromJson(Map<String, dynamic> j) => ReferenceAnchor(
      SensingIntent.values.byName(j['intent'] as String),
      j['object'] as String,
      (j['level'] as num?)?.toDouble(),
      (j['taps'] as List)
          .map((m) => Measurement.fromJson(Map<String, dynamic>.from(m as Map)))
          .toList(),
      label: j['label'] as String?,
      metadata: j['metadata'] == null
          ? const {}
          : Map<String, String>.from(j['metadata'] as Map));
}

class AnchorRepository {
  static const key = 'thud.anchors.v1';
  static Future<void> _pending = Future<void>.value();
  static Future<List<ReferenceAnchor>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(key);
    if (raw == null) return [];
    return (jsonDecode(raw) as List)
        .map((j) =>
            ReferenceAnchor.fromJson(Map<String, dynamic>.from(j as Map)))
        .toList();
  }

  static void validate(ReferenceAnchor anchor) {
    if (anchor.object.trim().isEmpty ||
        anchor.taps.length != 3 ||
        !anchor.taps.every(ReferenceAnchor.clean)) {
      throw const FormatException(
          'Each profile needs a name and three clean taps with peak, decay and HII.');
    }
    if (anchor.label != null &&
        !ReferenceAnchor.labelsFor(anchor.intent).contains(anchor.label)) {
      throw FormatException(
          'Invalid ${anchor.intent.title} label: ${anchor.label}.');
    }
    if (anchor.taps.any((m) =>
        m.source.toLowerCase().contains('synthetic') != anchor.synthetic)) {
      throw const FormatException('Do not mix demo and microphone taps.');
    }
    if (anchor.intent == SensingIntent.vessel &&
        (anchor.level == null ||
            !anchor.level!.isFinite ||
            anchor.level! < 0 ||
            anchor.level! > 100)) {
      throw const FormatException('Known fill must be between 0 and 100%.');
    }
    if (anchor.level != null && !anchor.level!.isFinite) {
      throw const FormatException('Reference level must be finite.');
    }
    if (anchor.intent != SensingIntent.vessel && anchor.level != null) {
      throw const FormatException('Only vessel references have a fill level.');
    }
  }

  static List<ReferenceAnchor> decodeImport(String raw) {
    final decoded = jsonDecode(raw);
    final dynamic rows = decoded is Map && decoded.containsKey('anchors')
        ? decoded['anchors']
        : decoded is Map
            ? [decoded]
            : decoded;
    if (rows is! List || rows.isEmpty) {
      throw const FormatException(
          'Paste an anchor, an anchor list, or a database with anchors.');
    }
    final batch = rows
        .map((j) =>
            ReferenceAnchor.fromJson(Map<String, dynamic>.from(j as Map)))
        .toList();
    final identities = <String>{};
    for (final anchor in batch) {
      validate(anchor);
      if (!identities.add(_identity(anchor))) {
        throw const FormatException(
            'Duplicate profile and label in this import.');
      }
    }
    return batch;
  }

  static String _identity(ReferenceAnchor a) => jsonEncode([
        a.intent.name,
        a.object.trim().toLowerCase(),
        a.level,
        a.synthetic,
        a.label,
      ]);

  static Future<void> save(ReferenceAnchor anchor) => saveBatch([anchor]);

  static Future<void> saveBatch(List<ReferenceAnchor> batch) {
    // Serialize read/merge/write so simultaneous screens cannot lose profiles.
    final operation = _pending.then((_) async {
      if (batch.isEmpty)
        throw const FormatException('No references to import.');
      final identities = <String>{};
      for (final a in batch) {
        validate(a);
        if (!identities.add(_identity(a))) {
          throw const FormatException(
              'Duplicate profile and label in this import.');
        }
      }
      final all = await load();
      for (final anchor in batch) {
        final named = all.where((a) =>
            a.intent == anchor.intent &&
            a.object.trim().toLowerCase() ==
                anchor.object.trim().toLowerCase());
        final name =
            named.isEmpty ? anchor.object.trim() : named.first.object.trim();
        final normalized = ReferenceAnchor(
            anchor.intent, name, anchor.level, anchor.taps,
            label: anchor.label, metadata: anchor.metadata);
        all.removeWhere((a) => _identity(a) == _identity(normalized));
        all.add(normalized);
      }
      final prefs = await SharedPreferences.getInstance();
      final previous = prefs.getString(key);
      try {
        if (!await prefs.setString(
            key, jsonEncode(all.map((a) => a.toJson()).toList()))) {
          throw Exception('Could not save reference database.');
        }
      } catch (_) {
        if (previous == null) {
          await prefs.remove(key);
        } else {
          await prefs.setString(key, previous);
        }
        rethrow;
      }
    });
    _pending =
        operation.then<void>((_) {}, onError: (Object _, StackTrace __) {});
    return operation;
  }
}

/// Local user recordings and imported references; selection returns the profile name.
class ReferenceLibraryScreen extends StatefulWidget {
  const ReferenceLibraryScreen(
      {required this.intent, required this.store, super.key});
  final SensingIntent intent;
  final ThudStore store;
  @override
  State<ReferenceLibraryScreen> createState() => _ReferenceLibraryScreenState();
}

class _ReferenceLibraryScreenState extends State<ReferenceLibraryScreen> {
  List<ReferenceAnchor> anchors = [];
  String query = '';
  String? error;
  bool loading = true, demo = false, writing = false;
  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final loaded = await AnchorRepository.load();
      if (mounted)
        setState(() {
          anchors = loaded;
          loading = false;
          error = null;
        });
    } catch (e) {
      if (mounted)
        setState(() {
          loading = false;
          error = e.toString();
        });
    }
  }

  Future<void> _teach() async {
    final selected = await Navigator.of(context).push(sensingRoute<String>(
        context,
        GuidedReferenceScreen(intent: widget.intent, store: widget.store)));
    if (!mounted) return;
    await _load();
    if (selected != null && mounted) {
      final saved = anchors
          .where((a) =>
              a.intent == widget.intent &&
              a.object.trim().toLowerCase() == selected.trim().toLowerCase())
          .toList();
      if (saved.isNotEmpty) setState(() => demo = saved.last.synthetic);
    }
  }

  Future<void> _import() async {
    final input = TextEditingController();
    List<ReferenceAnchor>? batch;
    try {
      batch = await showDialog<List<ReferenceAnchor>>(
          context: context,
          builder: (dialog) {
            String? issue;
            return StatefulBuilder(
                builder: (context, update) => AlertDialog(
                        title: const Text('Import reference database'),
                        content: SizedBox(
                            width: 480,
                            child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const Text(
                                      'Paste exported JSON or an existing anchor list. All profiles are checked before saving.'),
                                  const SizedBox(height: 12),
                                  TextField(
                                      controller: input,
                                      minLines: 5,
                                      maxLines: 10,
                                      decoration: const InputDecoration(
                                          labelText: 'Reference JSON')),
                                  if (issue != null)
                                    Padding(
                                        padding: const EdgeInsets.only(top: 12),
                                        child: Text(issue!,
                                            style:
                                                const TextStyle(color: C.red))),
                                ])),
                        actions: [
                          TextButton(
                              onPressed: () => Navigator.pop(dialog),
                              child: const Text('Cancel')),
                          TextButton(
                              onPressed: () {
                                try {
                                  final parsed =
                                      AnchorRepository.decodeImport(input.text);
                                  Navigator.pop(dialog, parsed);
                                } catch (e) {
                                  update(() => issue = e.toString());
                                }
                              },
                              child: const Text('Import')),
                        ]));
          });
    } finally {
      input.dispose();
    }
    if (batch == null || !mounted) return;
    setState(() => writing = true);
    try {
      await AnchorRepository.saveBatch(batch
          .map((a) => ReferenceAnchor(a.intent, a.object, a.level, a.taps,
                  label: a.label,
                  metadata: {
                    ...a.metadata,
                    'importedAt': DateTime.now().toUtc().toIso8601String()
                  }))
          .toList());
      await _load();
      if (mounted)
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('${batch.length} references imported')));
    } catch (e) {
      if (mounted) setState(() => error = e.toString());
    } finally {
      if (mounted) setState(() => writing = false);
    }
  }

  Future<void> _export() async {
    try {
      final all = await AnchorRepository.load();
      final current = all.where((a) => a.intent == widget.intent).toList();
      await Clipboard.setData(ClipboardData(
          text: jsonEncode({
        'schema': 'thud.reference-library',
        'version': 1,
        'anchors': current.map((a) => a.toJson()).toList(),
      })));
      if (mounted)
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Reference JSON copied')));
    } catch (e) {
      if (mounted) setState(() => error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    final current = anchors
        .where((a) => a.intent == widget.intent && a.synthetic == demo)
        .toList();
    final groups = <String, List<ReferenceAnchor>>{};
    for (final a in current) {
      groups.putIfAbsent(a.object, () => []).add(a);
    }
    final names = groups.keys.where((name) {
      final text = [
        name,
        ...groups[name]!
            .map((a) => '${a.displayLabel} ${a.metadata.values.join(' ')}')
      ].join(' ').toLowerCase();
      return text.contains(query.trim().toLowerCase());
    }).toList()
      ..sort();
    return Scaffold(
        appBar: AppBar(title: Text('${widget.intent.title} library'), actions: [
          IconButton(
              tooltip: 'Export JSON',
              onPressed: loading ||
                      writing ||
                      anchors.every((a) => a.intent != widget.intent)
                  ? null
                  : _export,
              icon: const Icon(Icons.ios_share_rounded)),
        ]),
        body: AppPage(children: [
          PageHeading('Choose a reference.',
              subtitle:
                  '${groups.length} profiles · ${current.length} labeled references'),
          const SizedBox(height: 20),
          TextField(
              onChanged: (value) => setState(() => query = value),
              decoration: const InputDecoration(
                  prefixIcon: Icon(Icons.search_rounded),
                  hintText: 'Search name, material or label')),
          SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Demo references'),
              subtitle: Text(demo
                  ? 'Synthetic taps · separate from recordings'
                  : 'Recorded and imported microphone taps'),
              value: demo,
              onChanged: (value) => setState(() => demo = value)),
          if (error != null) ...[
            Text(error!, style: const TextStyle(color: C.red)),
            TextButton(onPressed: _load, child: const Text('Reload')),
          ],
          if (loading)
            const Center(child: CircularProgressIndicator())
          else if (names.isEmpty)
            Padding(
                padding: const EdgeInsets.symmetric(vertical: 28),
                child: Text(
                    current.isEmpty
                        ? 'No references yet. Teach an object or import your database.'
                        : 'No matching profiles.',
                    style: const TextStyle(color: C.muted)))
          else
            ...names.map((name) {
              final refs = groups[name]!;
              final usable = refs.any((a) =>
                  a.taps.length >= 3 && a.taps.every(ReferenceAnchor.clean));
              final labels =
                  refs.map((a) => a.displayLabel).toSet().join(' / ');
              final details = refs
                  .expand((a) => a.metadata.entries
                      .where((e) => e.key == 'material' || e.key == 'geometry')
                      .map((e) => e.value))
                  .toSet()
                  .join(' · ');
              final sources = refs
                  .map((a) => a.synthetic
                      ? 'Demo'
                      : a.metadata.containsKey('importedAt')
                          ? 'Imported'
                          : 'Recorded')
                  .toSet()
                  .join(' / ');
              return ListTile(
                  contentPadding: EdgeInsets.zero,
                  title: Text(name),
                  subtitle: Text(
                      '$labels · ${refs.fold<int>(0, (n, a) => n + a.taps.length)} taps\n$sources${details.isEmpty ? '' : ' · $details'}${usable ? '' : ' · Refresh reference'}'),
                  trailing: const Icon(Icons.chevron_right_rounded),
                  onTap: usable ? () => Navigator.pop(context, name) : null);
            }),
          const SizedBox(height: 24),
          ActionButton('Teach reference',
              icon: Icons.mic_rounded, onPressed: writing ? null : _teach),
          TextButton.icon(
              onPressed: writing ? null : _import,
              icon: const Icon(Icons.file_download_outlined),
              label: const Text('Import database')),
        ]));
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
  final material = TextEditingController();
  final geometry = TextEditingController();
  late String label;
  final taps = <Measurement>[];
  bool busy = false, recording = false, saved = false;
  bool? synthetic;
  String? error;
  @override
  void initState() {
    super.initState();
    label = ReferenceAnchor.labelsFor(widget.intent).first;
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
      if (!ReferenceAnchor.clean(m)) {
        throw Exception('No measurable peak, decay or HII. Retry this tap.');
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
          List.of(taps),
          label: label,
          metadata: {
            'source': synthetic == true ? 'demo' : 'microphone',
            'recordedAt': DateTime.now().toUtc().toIso8601String(),
            if (material.text.trim().isNotEmpty)
              'material': material.text.trim(),
            if (geometry.text.trim().isNotEmpty)
              'geometry': geometry.text.trim(),
          }));
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
    material.dispose();
    geometry.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Theme(
      data: Theme.of(context).copyWith(
          colorScheme: ColorScheme.fromSeed(seedColor: widget.intent.color)),
      child: Scaffold(
          appBar: AppBar(title: Text('${widget.intent.title} reference')),
          body: AppPage(children: [
            PageHeading(
                saved ? 'Reference saved.' : 'Teach this\nexact object.',
                subtitle: widget.intent == SensingIntent.devices
                    ? 'Power off. Unplug. Three gentle casing taps.'
                    : 'Three taps. Same spot, tool and distance.'),
            const SizedBox(height: 24),
            TextField(
                controller: object,
                enabled: taps.isEmpty && !recording,
                decoration: const InputDecoration(labelText: 'Reference name')),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
                initialValue: label,
                decoration: const InputDecoration(labelText: 'Known label'),
                items: ReferenceAnchor.labelsFor(widget.intent)
                    .map((value) =>
                        DropdownMenuItem(value: value, child: Text(value)))
                    .toList(),
                onChanged: taps.isEmpty && !recording && !saved
                    ? (value) => setState(() => label = value!)
                    : null),
            TextField(
                controller: material,
                enabled: taps.isEmpty && !recording && !saved,
                decoration: const InputDecoration(
                    labelText: 'Material / type (optional)')),
            TextField(
                controller: geometry,
                enabled: taps.isEmpty && !recording && !saved,
                decoration: const InputDecoration(
                    labelText: 'Size / geometry (optional)')),
            if (widget.intent == SensingIntent.vessel)
              TextField(
                  controller: level,
                  enabled: taps.isEmpty && !recording,
                  keyboardType: TextInputType.number,
                  decoration:
                      const InputDecoration(labelText: 'Known fill (%)')),
            const SizedBox(height: 32),
            Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: List.generate(
                    3,
                    (i) => AnimatedContainer(
                        duration: Duration(
                            milliseconds:
                                MediaQuery.disableAnimationsOf(context)
                                    ? 0
                                    : 400),
                        width: 72,
                        height: 72,
                        decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color:
                                i < taps.length ? widget.intent.color : C.pale),
                        child: Center(
                            child: Text(
                                i < taps.length
                                    ? '${taps[i].peakHz.round()}'
                                    : '${i + 1}',
                                style: TextStyle(
                                    color: i < taps.length
                                        ? Colors.white
                                        : C.muted,
                                    fontSize: 18)))))),
            const SizedBox(height: 20),
            Center(
                child: Text('${taps.length} / 3 taps',
                    style: const TextStyle(
                        fontFamily: 'monospace', color: C.muted))),
            const SizedBox(height: 32),
            if (synthetic == true)
              const Text('Demo reference', style: TextStyle(color: C.amber)),
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
                    child: const Text('Use demo tap')),
            ],
            if (taps.length == 3 && !saved)
              ActionButton('Save reference', onPressed: busy ? null : _save),
            if (saved)
              ActionButton('Use this profile',
                  onPressed: () => Navigator.pop(context, object.text.trim())),
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
          ])));
}
