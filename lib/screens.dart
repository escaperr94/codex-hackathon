import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import 'design.dart';
import 'store.dart';
import 'listening_stage.dart';
import 'sensing_flow.dart';
import 'sensing_result.dart';
import 'defect_map.dart';
import 'scope_view.dart';

typedef MeasureAction = void Function(
    Module module, String? projectId, int? cell);

String _date(DateTime d) =>
    '${d.toLocal().year}-${d.toLocal().month.toString().padLeft(2, '0')}-${d.toLocal().day.toString().padLeft(2, '0')} ${d.toLocal().hour.toString().padLeft(2, '0')}:${d.toLocal().minute.toString().padLeft(2, '0')}';
String _number(double n, [int digits = 1]) =>
    n.isFinite ? n.toStringAsFixed(digits) : 'Unavailable';
String _cellName(int cell, int columns) =>
    'R${cell ~/ columns + 1} · C${cell % columns + 1}';
bool _synthetic(Measurement m) => m.source.toLowerCase().contains('synthetic');
bool _validReference(Measurement m) =>
    !_synthetic(m) &&
    m.flags.isEmpty &&
    m.peakHz.isFinite &&
    m.peakHz > 0 &&
    m.waveform.isNotEmpty;
void _push(BuildContext context, Widget page) =>
    Navigator.of(context).push(sensingRoute<void>(context, page));
void _notice(BuildContext context, Object error) =>
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(error.toString().replaceFirst('Exception: ', ''))));
Future<void> _perform(
    BuildContext context, Future<void> Function() action) async {
  try {
    await action();
  } catch (e) {
    if (context.mounted) _notice(context, e);
  }
}

const _gap = SizedBox(height: 20);
const _small = SizedBox(height: 12);
const _body = TextStyle(color: C.muted, height: 1.5, fontSize: 14);
const _title =
    TextStyle(color: C.ink, fontWeight: FontWeight.w500, fontSize: 18);

class _RoutePage extends StatelessWidget {
  const _RoutePage({required this.title, required this.children, this.actions});
  final String title;
  final List<Widget> children;
  final List<Widget>? actions;
  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(
          title: Text(title, style: _title),
          backgroundColor: C.bg,
          actions: actions),
      body: SafeArea(
          child: Align(
              alignment: Alignment.topCenter,
              child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 900),
                  child: AppPage(children: children)))));
}

class _Empty extends StatelessWidget {
  const _Empty(this.title, this.description,
      {this.icon = Icons.graphic_eq_rounded, this.action});
  final String title, description;
  final IconData icon;
  final Widget? action;
  @override
  Widget build(BuildContext context) => Panel(
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(icon, size: 20, color: C.muted),
        _gap,
        Text(title, style: _title),
        _small,
        Text(description, style: _body),
        if (action != null) ...[_gap, action!]
      ]));
}

class ScopeScreen extends StatefulWidget {
  const ScopeScreen(
      {required this.store,
      required this.initialModule,
      required this.onModule,
      this.projectId,
      this.cell,
      this.onIntent,
      this.initialIntent,
      super.key});
  final ThudStore store;
  final Module initialModule;
  final String? projectId;
  final int? cell;
  final ValueChanged<Module> onModule;
  final ValueChanged<SensingIntent>? onIntent;
  final SensingIntent? initialIntent;
  @override
  State<ScopeScreen> createState() => _ScopeScreenState();
}

class _ScopeScreenState extends State<ScopeScreen> with WidgetsBindingObserver {
  final CaptureService _capture = CaptureService();
  final Stopwatch _elapsed = Stopwatch();
  Timer? _limit, _clock;
  late Module _module;
  late SensingIntent _intent;
  late final PageController _modes;
  final Map<SensingIntent, String> _profiles = {};
  Measurement? _result;
  String? _error;
  bool _busy = false, _recording = false, _saved = false;
  @override
  void initState() {
    super.initState();
    _module = widget.initialModule;
    _intent = widget.initialIntent ??
        switch (_module) {
          Module.slab => SensingIntent.slab,
          Module.vessel => SensingIntent.vessel,
          Module.scope => SensingIntent.scope,
        };
    _modes = PageController(initialPage: _intent.index);
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void didUpdateWidget(covariant ScopeScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!_recording && _module != widget.initialModule) {
      _module = widget.initialModule;
      _intent = switch (_module) {
        Module.slab => SensingIntent.slab,
        Module.vessel => SensingIntent.vessel,
        Module.scope => SensingIntent.scope,
      };
      if (_modes.hasClients) _modes.jumpToPage(_intent.index);
    }
  }

  Future<void> _chooseProfile(SensingIntent intent) async {
    final selected = await Navigator.of(context).push<String>(
        sensingRoute<String>(context,
            ReferenceLibraryScreen(intent: intent, store: widget.store)));
    if (mounted && selected != null)
      setState(() => _profiles[intent] = selected);
  }

  Measurement _profiled(Measurement measurement) =>
      measurement.withIntent(_intent.name).withContext({
        ...measurement.context,
        if (_profiles[_intent] case final name?) 'referenceObject': name,
      });

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed && _recording) _cancel();
  }

  void _feedback() {
    if (widget.store.haptics) HapticFeedback.selectionClick();
    if (widget.store.sounds) SystemSound.play(SystemSoundType.click);
  }

  Future<void> _start() async {
    if (_busy || _recording) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    if (widget.store.haptics) HapticFeedback.selectionClick();
    try {
      await _capture.start();
      if (!mounted) {
        await _capture.cancel();
        return;
      }
      _elapsed.reset();
      _elapsed.start();
      setState(() {
        _recording = true;
        _result = null;
        _saved = false;
      });
      _limit = Timer(const Duration(seconds: 8), _stop);
      _clock = Timer.periodic(const Duration(milliseconds: 100), (_) {
        if (mounted) setState(() {});
      });
    } catch (e) {
      if (mounted) {
        setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _stopTimers() {
    _limit?.cancel();
    _clock?.cancel();
    _elapsed.stop();
  }

  String _captureError(String error) {
    final value = error.toLowerCase();
    if (value.contains('permission')) return 'Enable microphone access';
    if (value.contains('no usable audio')) return 'No sound · tap again';
    if (value.contains('too short')) return 'Tap too short';
    if (value.contains('format') || value.contains('pcm16')) {
      return 'Reconnect microphone';
    }
    return 'Recording failed · try again';
  }

  Future<void> _stop() async {
    if (_busy || !_recording) return;
    _stopTimers();
    setState(() {
      _busy = true;
      _recording = false;
    });
    try {
      final result = await _capture.stop(_module);
      if (mounted) {
        setState(() => _result = _profiled(result));
        _feedback();
      }
    } catch (e) {
      if (mounted) {
        setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _cancel() async {
    _stopTimers();
    if (!mounted) {
      await _capture.cancel();
      return;
    }
    setState(() {
      _busy = true;
      _recording = false;
    });
    try {
      await _capture.cancel();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _sample({bool resonant = false}) async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final m = await _capture.sample(_module,
          resonant: resonant || _module == Module.vessel,
          fillFraction: _module == Module.vessel ? .5 : null);
      if (mounted) {
        setState(() {
          _result = _profiled(m);
          _saved = false;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save(Measurement m) async {
    if (_result == null) return;
    try {
      final saved = await _saveMeasurement(context, widget.store, m,
          projectId: widget.projectId, cell: widget.cell);
      if (mounted && saved) {
        setState(() => _saved = true);
        _notice(context, 'Measurement saved on this device.');
      }
    } catch (e) {
      if (mounted) _notice(context, e);
    }
  }

  void _selectIntent(int index) {
    if (_busy || _recording) return;
    final intent = SensingIntent.values[index];
    setState(() {
      _intent = intent;
      _module = intent.module;
      _result = null;
      _saved = false;
      _error = null;
    });
    if (widget.store.haptics) HapticFeedback.selectionClick();
    widget.onModule(_module);
    widget.onIntent?.call(intent);
  }

  void _goMode(int index) {
    if (_busy ||
        _recording ||
        index < 0 ||
        index >= SensingIntent.values.length) {
      return;
    }
    if (MediaQuery.disableAnimationsOf(context)) {
      _modes.jumpToPage(index);
    } else {
      _modes.animateToPage(index,
          duration: const Duration(milliseconds: 460),
          curve: Curves.easeOutCubic);
    }
  }

  Future<void> _map(Measurement m) async {
    final linked = widget.store.projects.where((p) => p.id == widget.projectId);
    Project? project = linked.isEmpty ? null : linked.first;
    project ??= await showModalBottomSheet<Project>(
        context: context,
        showDragHandle: true,
        builder: (sheet) => SafeArea(
                child: SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                PageHeading(m.intent == 'slab'
                    ? 'Add to handover'
                    : 'Add to inspection'),
                _gap,
                for (final p in widget.store.projects)
                  ListTile(
                      title: Text(p.name),
                      subtitle: Text('${p.room} · ${p.rows} × ${p.columns}'),
                      onTap: () => Navigator.pop(sheet, p)),
                ActionButton('New dossier', icon: Icons.add_rounded,
                    onPressed: () async {
                  final created = await showDialog<Project>(
                      context: sheet,
                      builder: (_) => _NewProjectDialog(store: widget.store));
                  if (sheet.mounted && created != null) {
                    Navigator.pop(sheet, created);
                  }
                }),
              ]),
            )));
    if (!mounted || project == null) return;
    final destination = project;
    _push(
        context,
        DefectMapScreen(
            store: widget.store,
            project: destination,
            measurement: m,
            onReport: () => _push(context,
                ReportScreen(store: widget.store, project: destination))));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopTimers();
    _capture.dispose();
    _modes.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) =>
      LayoutBuilder(builder: (context, bounds) {
        return PageView.builder(
          controller: _modes,
          physics: _busy || _recording
              ? const NeverScrollableScrollPhysics()
              : const PageScrollPhysics(parent: BouncingScrollPhysics()),
          itemCount: SensingIntent.values.length,
          onPageChanged: _selectIntent,
          itemBuilder: (context, index) {
            final intent = SensingIntent.values[index];
            final selected = intent == _intent;
            final m = selected ? _result : null;
            final reduced = MediaQuery.disableAnimationsOf(context);
            return Theme(
              data: Theme.of(context).copyWith(
                  colorScheme: ColorScheme.fromSeed(seedColor: intent.color)
                      .copyWith(primary: intent.color)),
              child: AnimatedBuilder(
                animation: _modes,
                builder: (context, child) {
                  final distance =
                      _modes.hasClients && _modes.position.haveDimensions
                          ? ((_modes.page ?? _intent.index.toDouble()) - index)
                              .clamp(-1.0, 1.0)
                          : 0.0;
                  return Transform.scale(
                      scale: reduced ? 1 : 1 - distance.abs() * .045,
                      child: Transform.translate(
                          offset: Offset(reduced ? 0 : distance * 24, 0),
                          child: child));
                },
                child: SingleChildScrollView(
                  child: Center(
                      child: ConstrainedBox(
                    constraints: BoxConstraints(
                        maxWidth: 540, minHeight: bounds.maxHeight),
                    child: Padding(
                        padding: const EdgeInsets.fromLTRB(20, 4, 20, 24),
                        child: Column(children: [
                          Row(children: [
                            IconButton(
                                tooltip: 'Previous mode',
                                onPressed: index > 0 && !_busy && !_recording
                                    ? () => _goMode(index - 1)
                                    : null,
                                icon: Icon(Icons.chevron_left_rounded,
                                    color: index > 0
                                        ? Colors.white
                                        : Colors.white30)),
                            Expanded(
                                child: Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                  Icon(intent.icon,
                                      color: Colors.white, size: 20),
                                  const SizedBox(width: 8),
                                  Text(intent.title,
                                      style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 18,
                                          fontWeight: FontWeight.w700)),
                                ])),
                            IconButton(
                                tooltip: 'Next mode',
                                onPressed:
                                    index < SensingIntent.values.length - 1 &&
                                            !_busy &&
                                            !_recording
                                        ? () => _goMode(index + 1)
                                        : null,
                                icon: Icon(Icons.chevron_right_rounded,
                                    color:
                                        index < SensingIntent.values.length - 1
                                            ? Colors.white
                                            : Colors.white30)),
                          ]),
                          Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                for (final mode in SensingIntent.values)
                                  Semantics(
                                      label: '${mode.title} mode',
                                      selected: mode == intent,
                                      child: GestureDetector(
                                          onTap: () => _goMode(mode.index),
                                          child: Padding(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                      horizontal: 3,
                                                      vertical: 10),
                                              child: AnimatedContainer(
                                                  duration: Duration(
                                                      milliseconds:
                                                          reduced ? 0 : 250),
                                                  width:
                                                      mode == intent ? 24 : 6,
                                                  height: 6,
                                                  decoration: BoxDecoration(
                                                      color: mode == intent
                                                          ? Colors.white
                                                          : Colors.white38,
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                              4)))))),
                              ]),
                          if (m == null &&
                              (intent == SensingIntent.slab ||
                                  intent == SensingIntent.materials ||
                                  intent == SensingIntent.devices))
                            TextButton.icon(
                                onPressed: _busy || _recording
                                    ? null
                                    : () => _chooseProfile(intent),
                                style: TextButton.styleFrom(
                                    foregroundColor: Colors.white),
                                icon: const Icon(Icons.library_books_outlined,
                                    size: 18),
                                label: Text(
                                    _profiles[intent] ??
                                        (intent == SensingIntent.slab
                                            ? 'Tile library'
                                            : 'Reference library'),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis)),
                          const SizedBox(height: 12),
                          AnimatedSwitcher(
                              duration:
                                  Duration(milliseconds: reduced ? 0 : 360),
                              transitionBuilder: _sceneTransition,
                              child: m != null
                                  ? SensingResult(
                                      key: ValueKey(
                                          'result-${m.id}-${intent.name}'),
                                      measurement: m,
                                      intent: intent,
                                      store: widget.store,
                                      saved: _saved,
                                      onSave: _save,
                                      onMap: _map,
                                      onTrace: (record) => _push(
                                          context,
                                          MeasurementScreen(
                                              store: widget.store,
                                              measurement: record)),
                                      onAgain: () => setState(() {
                                            _result = null;
                                            _saved = false;
                                          }))
                                  : Column(
                                      key: ValueKey('capture-${intent.name}'),
                                      children: [
                                          Text(
                                              selected && _busy
                                                  ? 'Getting ready'
                                                  : selected && _recording
                                                      ? 'Listening…'
                                                      : intent.headline,
                                              textAlign: TextAlign.center,
                                              style: const TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 38,
                                                  fontWeight: FontWeight.w700,
                                                  height: 1.08,
                                                  letterSpacing: -1.2)),
                                          const SizedBox(height: 12),
                                          Text(intent.useCase,
                                              style: const TextStyle(
                                                  color: Colors.white70,
                                                  fontSize: 15)),
                                          SizedBox(
                                              height: (bounds.maxHeight * .4)
                                                  .clamp(200.0, 300.0),
                                              width: double.infinity,
                                              child: ListeningStage(
                                                  live: _capture.live,
                                                  active:
                                                      selected && _recording,
                                                  busy: selected && _busy,
                                                  complete: false,
                                                  subjectIcon: intent.icon,
                                                  onTap: !selected || _busy
                                                      ? null
                                                      : _recording
                                                          ? _stop
                                                          : _start)),
                                          Text(
                                              selected && _recording
                                                  ? 'Tap once. Let it ring.'
                                                  : 'Tap to listen. Tap your object.',
                                              style: const TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 16,
                                                  fontWeight: FontWeight.w600)),
                                          const SizedBox(height: 12),
                                          if (selected && _recording)
                                            Row(
                                                mainAxisAlignment:
                                                    MainAxisAlignment.center,
                                                children: [
                                                  Text(
                                                      '${(_elapsed.elapsedMilliseconds / 1000).toStringAsFixed(1)} / 8s',
                                                      style: const TextStyle(
                                                          color:
                                                              Colors.white70)),
                                                  TextButton(
                                                      onPressed: _busy
                                                          ? null
                                                          : _cancel,
                                                      style:
                                                          TextButton.styleFrom(
                                                              foregroundColor:
                                                                  Colors.white),
                                                      child:
                                                          const Text('Cancel')),
                                                ])
                                          else
                                            Wrap(
                                                alignment: WrapAlignment.center,
                                                children: [
                                                  TextButton(
                                                      onPressed: _busy
                                                          ? null
                                                          : _sample,
                                                      style:
                                                          TextButton.styleFrom(
                                                              foregroundColor:
                                                                  Colors.white),
                                                      child: const Text(
                                                          'Try demo')),
                                                  if (intent ==
                                                          SensingIntent.slab ||
                                                      intent ==
                                                          SensingIntent.devices)
                                                    TextButton(
                                                        onPressed: _busy
                                                            ? null
                                                            : () => _sample(
                                                                resonant: true),
                                                        style: TextButton
                                                            .styleFrom(
                                                                foregroundColor:
                                                                    Colors
                                                                        .white70),
                                                        child: const Text(
                                                            'Long-ring demo')),
                                                  IconButton(
                                                      tooltip: 'How to tap',
                                                      onPressed: _instructions,
                                                      icon: const Icon(
                                                          Icons
                                                              .help_outline_rounded,
                                                          color:
                                                              Colors.white70)),
                                                ]),
                                        ])),
                          if (widget.projectId != null)
                            Padding(
                                padding: const EdgeInsets.only(top: 12),
                                child: Text(
                                    'Project · point ${(widget.cell ?? 0) + 1}',
                                    style: const TextStyle(
                                        color: Colors.white70))),
                          if (selected && _error != null)
                            Padding(
                                padding: const EdgeInsets.only(top: 16),
                                child: Panel(
                                    child: Row(children: [
                                  Icon(Icons.refresh_rounded,
                                      color: intent.color),
                                  const SizedBox(width: 12),
                                  Expanded(
                                      child: Text(_captureError(_error!),
                                          style: _body)),
                                ]))),
                        ])),
                  )),
                ),
              ),
            );
          },
        );
      });

  Widget _sceneTransition(Widget child, Animation<double> animation) =>
      FadeTransition(
        opacity: animation,
        child: SlideTransition(
            position: Tween<Offset>(
                    begin: MediaQuery.disableAnimationsOf(context)
                        ? Offset.zero
                        : const Offset(0, .06),
                    end: Offset.zero)
                .animate(CurvedAnimation(
                    parent: animation, curve: Curves.easeOutCubic)),
            child: child),
      );

  void _instructions() => showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (_) => Padding(
            padding: const EdgeInsets.fromLTRB(24, 8, 24, 32),
            child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('One clean tap',
                      style:
                          TextStyle(fontSize: 24, fontWeight: FontWeight.w500)),
                  const SizedBox(height: 16),
                  Text(
                      _intent == SensingIntent.devices
                          ? 'Power off and unplug. Gently tap the casing, away from screens and batteries. Hold your phone 10–15 cm away.'
                          : 'Hold your phone 10–15 cm away. Listen, tap once, then let it ring.',
                      style: _body),
                  TextButton(
                      onPressed: () {
                        Navigator.pop(context);
                        _chooseProfile(_intent);
                      },
                      child: const Text('Reference library')),
                ]),
          ));
}

Future<bool> _saveMeasurement(
    BuildContext context, ThudStore store, Measurement m,
    {String? projectId, int? cell}) async {
  if (projectId != null) {
    await store.saveMeasurement(m, projectId: projectId, cell: cell);
    return true;
  }
  final destination = await showModalBottomSheet<_Destination>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => _DestinationSheet(store: store));
  if (destination == null) return false;
  await store.saveMeasurement(m,
      projectId: destination.projectId, cell: destination.cell);
  return true;
}

class _Destination {
  const _Destination(this.projectId, this.cell);
  final String? projectId;
  final int? cell;
}

class _DestinationSheet extends StatefulWidget {
  const _DestinationSheet({required this.store});
  final ThudStore store;
  @override
  State<_DestinationSheet> createState() => _DestinationSheetState();
}

class _DestinationSheetState extends State<_DestinationSheet> {
  String? projectId;
  int? cell;
  @override
  Widget build(BuildContext context) {
    final matches = widget.store.projects.where((p) => p.id == projectId);
    final project = matches.isEmpty ? null : matches.first;
    return SafeArea(
        child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(24, 8, 24, 28),
            child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('Save recording', style: _title),
                  _small,
                  const Text(
                      'Save to history, or attach the recording to a project cell.',
                      style: _body),
                  _gap,
                  DropdownButtonFormField<String>(
                      initialValue: projectId ?? '',
                      decoration:
                          const InputDecoration(labelText: 'Destination'),
                      items: [
                        const DropdownMenuItem(
                            value: '', child: Text('History only')),
                        ...widget.store.projects.map((p) => DropdownMenuItem(
                            value: p.id,
                            child:
                                Text(p.name, overflow: TextOverflow.ellipsis)))
                      ],
                      onChanged: (value) => setState(() {
                            projectId = value == '' ? null : value;
                            cell = null;
                          })),
                  if (project != null) ...[
                    _gap,
                    DropdownButtonFormField<int>(
                        initialValue: cell ?? -1,
                        decoration:
                            const InputDecoration(labelText: 'Grid cell'),
                        items: [
                          const DropdownMenuItem(
                              value: -1, child: Text('Unassigned')),
                          for (var i = 0;
                              i < project.rows * project.columns;
                              i++)
                            DropdownMenuItem(
                                value: i,
                                child: Text(_cellName(i, project.columns)))
                        ],
                        onChanged: (value) =>
                            setState(() => cell = value == -1 ? null : value))
                  ],
                  _gap,
                  ActionButton('Save locally',
                      icon: Icons.bookmark_add_outlined,
                      onPressed: () =>
                          Navigator.pop(context, _Destination(projectId, cell)))
                ])));
  }
}

class _Evidence extends StatelessWidget {
  const _Evidence({required this.m, this.reference});
  final Measurement m;
  final Measurement? reference;

  static SensingIntent intentOf(Measurement m) {
    for (final intent in SensingIntent.values) {
      if (intent.name == m.intent) return intent;
    }
    return switch (m.module) {
      Module.slab => SensingIntent.slab,
      Module.vessel => SensingIntent.vessel,
      Module.scope => SensingIntent.scope,
    };
  }

  static bool needsRetry(Measurement m) =>
      m.flags.isNotEmpty ||
      !m.peakHz.isFinite ||
      m.peakHz <= 0 ||
      !m.decayMs.isFinite ||
      m.decayMs <= 0;

  static String headline(Measurement m) {
    if (needsRetry(m)) return 'Tap again';
    if (m.context['resultVersion'] != '2') return 'Signature captured';
    final classifier = m.context['classifierStatus'];
    if (classifier == 'matched' &&
        const {
          'Solid-like',
          'Hollow-like',
          'Original match',
          'Suspect match',
          'Reference match',
          'Anomaly match',
        }.contains(m.context['interpretation']))
      return m.context['interpretation']!;
    if (classifier == 'outside-reference-range') return 'Outside references';
    if (classifier == 'ambiguous') return 'Mixed signature';
    final result = (m.context['interpretation'] ?? '').toLowerCase();
    final comparison = m.context['suspicion'];
    if (comparison == 'Measured difference' || result.contains('different')) {
      return 'Check this spot';
    }
    if (comparison == 'Reference match' ||
        result.contains('matches reference')) {
      return 'Reference match';
    }
    if (result.contains('refresh reference') ||
        comparison == 'Reference decay missing') {
      return 'Refresh reference';
    }
    if (comparison == 'Loading references') return 'Loading references';
    if (comparison == 'Reference needed' ||
        comparison == 'Reference setup needed') {
      return 'Reference needed';
    }
    if (comparison == 'Outside reference range') return 'Outside range';
    if (intentOf(m) == SensingIntent.vessel) {
      final fill = double.tryParse(m.context['estimatedFillPercent'] ?? '');
      if (fill != null && fill.isFinite) return '${fill.round()}% fill';
    }
    if (intentOf(m) == SensingIntent.fruit) {
      final firmness = double.tryParse(m.context['firmnessIndex'] ?? '');
      if (firmness != null && firmness.isFinite) return 'Firmness index';
    }
    return 'Signature captured';
  }

  static String displayLabel(Measurement m) {
    final label = m.label.trim();
    final isDefault = label.isEmpty ||
        Module.values
            .any((module) => label.toLowerCase() == '${module.name} capture');
    return isDefault ? intentOf(m).title : label;
  }

  String get _retryReason {
    if (m.flags.any((flag) => flag.toLowerCase().contains('clipping'))) {
      return 'Tap a little softer';
    }
    if (m.flags.any((flag) => flag.toLowerCase().contains('snr'))) {
      return 'Find a quieter spot';
    }
    if (m.flags.any((flag) => flag.toLowerCase().contains('short'))) {
      return 'Let the tap ring';
    }
    if (!m.decayMs.isFinite || m.decayMs <= 0) {
      return 'Let the tap ring';
    }
    return 'Make one clean tap';
  }

  @override
  Widget build(BuildContext context) {
    final intent = intentOf(m);
    final retry = needsRetry(m);
    final readings = <(String, String)>[];
    void add(String label, double value, String unit, [int digits = 0]) {
      if (value.isFinite && readings.length < 3) {
        readings.add((label, '${_number(value, digits)}$unit'));
      }
    }

    if (!retry && intent == SensingIntent.fruit) {
      final firmness = double.tryParse(m.context['firmnessIndex'] ?? '');
      if (firmness != null && firmness > 0) add('Firmness index', firmness, '');
    }
    if (m.peakHz > 0) add('Peak', m.peakHz, ' Hz');
    if (m.decayMs > 0) add('Ring', m.decayMs, ' ms');
    add('SNR', m.snrDb, ' dB', 1);
    add('Level', m.rmsDb, ' dBFS', 1);

    final chips = <String>[
      if (_synthetic(m)) 'Demo recording',
      if (reference?.id == m.id) 'Current reference',
      if (m.context['classifierStatus'] == 'matched') 'Learned reference',
      if (m.context['fruitType'] case final fruit? when fruit.trim().isNotEmpty)
        fruit,
      if (double.tryParse(m.context['massGrams'] ?? '') case final mass?
          when mass.isFinite && mass > 0)
        '${_number(mass, 0)} g',
      if (m.context['referenceObject'] case final name?
          when name.trim().isNotEmpty)
        'Reference · $name',
    ];
    return Panel(
      color: intent.color.withValues(alpha: .045),
      child: SizedBox(
          width: double.infinity,
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Container(
                width: 88,
                height: 88,
                decoration: BoxDecoration(
                    color: intent.color.withValues(alpha: .1),
                    shape: BoxShape.circle),
                child: Icon(retry ? Icons.refresh_rounded : intent.icon,
                    color: intent.color, size: 44)),
            _gap,
            Text(headline(m),
                style: const TextStyle(
                    color: C.ink,
                    fontSize: 32,
                    height: 1.1,
                    letterSpacing: -.8,
                    fontWeight: FontWeight.w600)),
            if (retry) ...[
              _small,
              Text(_retryReason, style: _body),
            ],
            if (!retry && readings.isNotEmpty) ...[
              _gap,
              Wrap(spacing: 24, runSpacing: 16, children: [
                for (final reading in readings)
                  Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(reading.$1,
                            style:
                                const TextStyle(color: C.muted, fontSize: 12)),
                        const SizedBox(height: 4),
                        Text(reading.$2,
                            style: TextStyle(
                                color: intent.color,
                                fontSize: 20,
                                fontWeight: FontWeight.w600,
                                fontFeatures: const [
                                  FontFeature.tabularFigures()
                                ])),
                      ]),
              ]),
            ],
            if (chips.isNotEmpty) ...[
              _gap,
              Wrap(spacing: 8, runSpacing: 8, children: [
                for (final chip in chips)
                  Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                          color: intent.color.withValues(alpha: .07),
                          borderRadius: BorderRadius.circular(10)),
                      child: Text(chip,
                          style:
                              const TextStyle(color: C.muted, fontSize: 12))),
              ]),
            ],
          ])),
    );
  }
}

class LibraryScreen extends StatelessWidget {
  const LibraryScreen(
      {required this.store, required this.onMeasure, super.key});
  final ThudStore store;
  final MeasureAction onMeasure;
  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: store,
      builder: (_, __) => AppPage(children: [
            const PageHeading('Field library',
                subtitle: 'Saved recordings, references and field guidance'),
            _gap,
            _MenuTile(
                icon: Icons.library_books_outlined,
                title: 'Tile & material database',
                subtitle: 'Choose a profile · import · teach',
                onTap: () => _push(
                    context,
                    ReferenceLibraryScreen(
                        intent: SensingIntent.slab, store: store))),
            _small,
            _MenuTile(
                icon: Icons.history_rounded,
                title: 'Measurement history',
                subtitle:
                    '${store.measurements.length} saved recordings · stored locally',
                onTap: () => _push(context, HistoryScreen(store: store))),
            _small,
            _MenuTile(
                icon: Icons.tune_rounded,
                title: 'Reference & calibration',
                subtitle: store.reference == null
                    ? 'Choose a repeatable comparison recording'
                    : 'Reference set · ${_date(store.reference!.created)}',
                onTap: () => _push(
                    context,
                    CalibrationScreen(
                        store: store,
                        onRecord: () {
                          Navigator.of(context).pop();
                          onMeasure(Module.scope, null, null);
                        }))),
            _small,
            _MenuTile(
                icon: Icons.touch_app_outlined,
                title: 'Reference libraries',
                subtitle: 'Tile · material · device · vessel',
                onTap: () async {
                  final intent = await Navigator.of(context)
                      .push<SensingIntent>(sensingRoute<SensingIntent>(
                          context,
                          ChooseObjectScreen(
                              initial: SensingIntent.slab,
                              haptics: store.haptics)));
                  if (intent != null && context.mounted) {
                    Navigator.of(context).push<String>(sensingRoute<String>(
                        context,
                        ReferenceLibraryScreen(intent: intent, store: store)));
                  }
                }),
            _small,
            _MenuTile(
                icon: Icons.settings_outlined,
                title: 'Settings',
                subtitle: 'Haptics & sounds',
                onTap: () => _push(context, SettingsScreen(store: store))),
            _gap,
            const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Label('Keep the context'),
                  _small,
                  Text(
                      'Keep the context with the signal. Project grids, field notes and traceable recordings are more useful than a single tap in isolation.',
                      style: _body)
                ])
          ]));
}

class _MenuTile extends StatelessWidget {
  const _MenuTile(
      {required this.icon,
      required this.title,
      required this.subtitle,
      required this.onTap});
  final IconData icon;
  final String title, subtitle;
  final VoidCallback onTap;
  @override
  Widget build(BuildContext context) => Material(
      color: Colors.white,
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: C.line)),
      clipBehavior: Clip.antiAlias,
      child: ListTile(
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          leading: Icon(icon, color: C.muted, size: 20),
          title: Text(title,
              style: const TextStyle(
                  fontSize: 16, fontWeight: FontWeight.w500, color: C.ink)),
          subtitle: Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(subtitle, style: _body)),
          trailing:
              const Icon(Icons.chevron_right_rounded, color: C.muted, size: 20),
          onTap: onTap));
}

class ProjectsScreen extends StatelessWidget {
  const ProjectsScreen(
      {required this.store, required this.onMeasure, super.key});
  final ThudStore store;
  final MeasureAction onMeasure;
  Future<void> _create(BuildContext context) async {
    final p = await showDialog<Project>(
        context: context, builder: (_) => _NewProjectDialog(store: store));
    if (p != null && context.mounted) {
      _push(context,
          ProjectScreen(store: store, project: p, onMeasure: onMeasure));
    }
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: store,
      builder: (_, __) => AppPage(children: [
            PageHeading('Handover dossiers',
                subtitle: 'Rooms · findings · reports',
                trailing: IconButton(
                    tooltip: 'Create project',
                    onPressed: () => _create(context),
                    icon: const Icon(Icons.add_rounded,
                        color: C.muted, size: 20))),
            _gap,
            if (store.projects.isEmpty)
              _Empty('Your first handover', 'Add a room to start.',
                  icon: Icons.grid_view_rounded,
                  action: ActionButton('Create dossier',
                      icon: Icons.add_rounded,
                      onPressed: () => _create(context)))
            else ...[
              for (final p in store.projects) ...[
                _MenuTile(
                    icon: Icons.grid_view_rounded,
                    title: p.name,
                    subtitle:
                        '${p.room} · ${p.rows} × ${p.columns} grid · ${store.forProject(p.id).where((m) => m.cell != null).map((m) => m.cell).toSet().length} cells recorded',
                    onTap: () => _push(
                        context,
                        ProjectScreen(
                            store: store, project: p, onMeasure: onMeasure))),
                _small
              ],
              _small,
              ActionButton('New dossier',
                  icon: Icons.add_rounded, onPressed: () => _create(context))
            ]
          ]));
}

class _NewProjectDialog extends StatefulWidget {
  const _NewProjectDialog({required this.store});
  final ThudStore store;
  @override
  State<_NewProjectDialog> createState() => _NewProjectDialogState();
}

class _NewProjectDialogState extends State<_NewProjectDialog> {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController(),
      _room = TextEditingController(),
      _rows = TextEditingController(text: '3'),
      _columns = TextEditingController(text: '3');
  bool _saving = false;
  String? _error;
  @override
  void dispose() {
    _name.dispose();
    _room.dispose();
    _rows.dispose();
    _columns.dispose();
    super.dispose();
  }

  String? _required(String? v) =>
      v == null || v.trim().isEmpty ? 'Enter a value' : null;
  String? _dimension(String? v) {
    final n = int.tryParse(v?.trim() ?? '');
    return n == null || n < 1 || n > 12 ? 'Use 1–12' : null;
  }

  Future<void> _save() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      final p = await widget.store.addProject(
          _name.text.trim(),
          _room.text.trim(),
          int.parse(_rows.text.trim()),
          int.parse(_columns.text.trim()));
      if (mounted) Navigator.pop(context, p);
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _saving = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) => AlertDialog(
          title: const Text('New dossier'),
          content: SizedBox(
              width: 400,
              child: SingleChildScrollView(
                  child: Form(
                      key: _form,
                      child: Column(mainAxisSize: MainAxisSize.min, children: [
                        TextFormField(
                            controller: _name,
                            decoration: const InputDecoration(
                                labelText: 'Apartment or project'),
                            validator: _required,
                            maxLength: 80,
                            textCapitalization: TextCapitalization.sentences),
                        TextFormField(
                            controller: _room,
                            decoration: const InputDecoration(
                                labelText: 'Room or location'),
                            validator: _required,
                            maxLength: 100),
                        _small,
                        Row(children: [
                          Expanded(
                              child: TextFormField(
                                  controller: _rows,
                                  decoration: const InputDecoration(
                                      labelText: 'Rows (1–12)'),
                                  keyboardType: TextInputType.number,
                                  validator: _dimension)),
                          const SizedBox(width: 16),
                          Expanded(
                              child: TextFormField(
                                  controller: _columns,
                                  decoration: const InputDecoration(
                                      labelText: 'Columns (1–12)'),
                                  keyboardType: TextInputType.number,
                                  validator: _dimension))
                        ]),
                        if (_error != null) ...[
                          _small,
                          Text(_error!, style: const TextStyle(color: C.amber))
                        ]
                      ])))),
          actions: [
            TextButton(
                onPressed: _saving ? null : () => Navigator.pop(context),
                child: const Text('Cancel')),
            FilledButton(
                onPressed: _saving ? null : _save,
                child: Text(_saving ? 'Saving…' : 'Create'))
          ]);
}

class ProjectScreen extends StatefulWidget {
  const ProjectScreen(
      {required this.store,
      required this.project,
      required this.onMeasure,
      super.key});
  final ThudStore store;
  final Project project;
  final MeasureAction onMeasure;
  @override
  State<ProjectScreen> createState() => _ProjectScreenState();
}

class _ProjectScreenState extends State<ProjectScreen> {
  int selected = 0;
  Future<void> _notes() async {
    final value = await showDialog<String>(
        context: context,
        builder: (_) => _NotesDialog(notes: widget.project.notes));
    if (value != null && mounted) {
      await _perform(
          context, () => widget.store.updateNotes(widget.project, value));
    }
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: widget.store,
      builder: (_, __) {
        final p = widget.project;
        final all = widget.store.forProject(p.id).toList()
          ..sort((a, b) => b.created.compareTo(a.created));
        final latest = <int, Measurement>{};
        for (final m in all) {
          if (m.cell != null) latest.putIfAbsent(m.cell!, () => m);
        }
        final active = latest[selected];
        return _RoutePage(title: 'Project', children: [
          PageHeading(p.name,
              subtitle: '${p.room} · ${p.rows} rows × ${p.columns} columns'),
          _gap,
          Row(children: [
            Metric('Recorded cells', '${latest.length} / ${p.rows * p.columns}',
                accent: C.blue),
            const SizedBox(width: 8),
            Metric('Saved recordings', '${all.length}')
          ]),
          _gap,
          const SectionTitle('Inspection grid'),
          _small,
          const Text(
              'Select a cell to review its latest recording or capture another tap.',
              style: _body),
          _small,
          LayoutBuilder(builder: (_, constraints) {
            return SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: SizedBox(
                width: math.max(constraints.maxWidth, p.columns * 56.0),
                child: GridView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: p.columns,
                      crossAxisSpacing: 8,
                      mainAxisSpacing: 8),
                  itemCount: p.rows * p.columns,
                  itemBuilder: (_, i) => Semantics(
                    label:
                        '${_cellName(i, p.columns)}, ${latest.containsKey(i) ? 'recorded' : 'not recorded'}',
                    selected: selected == i,
                    child: Material(
                      color: selected == i
                          ? C.blue
                          : latest.containsKey(i)
                              ? C.bg
                              : Colors.white,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                          side: BorderSide(
                              color: latest.containsKey(i) ? C.muted : C.line)),
                      child: InkWell(
                        borderRadius: BorderRadius.circular(10),
                        onTap: () => setState(() => selected = i),
                        child: Center(
                            child: Text(
                                '${i ~/ p.columns + 1}:${i % p.columns + 1}',
                                style: TextStyle(
                                    fontFamily: 'monospace',
                                    color: selected == i ? Colors.white : C.ink,
                                    fontSize: 12,
                                    fontWeight: FontWeight.w500))),
                      ),
                    ),
                  ),
                ),
              ),
            );
          }),
          _gap,
          Panel(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                Label(_cellName(selected, p.columns), color: C.blue),
                _small,
                Text(
                    active == null
                        ? 'No recording at this point'
                        : '${_number(active.peakHz, 0)} Hz · latest peak',
                    style: _title),
                _small,
                Text(
                    active == null
                        ? 'Capture a repeatable tap to begin a record for this cell.'
                        : '${active.source}\n${_date(active.created)}',
                    style: _body),
                _small,
                Wrap(spacing: 8, runSpacing: 8, children: [
                  ActionButton('Measure this cell', icon: Icons.mic_rounded,
                      onPressed: () {
                    Navigator.of(context).pop();
                    widget.onMeasure(Module.slab, p.id, selected);
                  }),
                  if (active != null)
                    TextButton(
                        onPressed: () => _push(
                            context,
                            MeasurementScreen(
                                store: widget.store, measurement: active)),
                        child: const Text('View recording'))
                ])
              ])),
          _gap,
          SectionTitle('Field notes',
              trailing: TextButton(
                  onPressed: _notes, child: const Text('Edit notes'))),
          _small,
          Panel(
              child: Text(
                  p.notes.isEmpty
                      ? 'No notes yet. Add material, tap method and site conditions to keep comparisons meaningful.'
                      : p.notes,
                  style: _body)),
          _gap,
          ActionButton('Defect map & floor plan',
              icon: Icons.location_on_outlined,
              onPressed: () => _push(
                  context,
                  DefectMapScreen(
                      store: widget.store,
                      project: p,
                      measurement: active,
                      onReport: () => _push(context,
                          ReportScreen(store: widget.store, project: p))))),
          _small,
          ActionButton('Handover report',
              icon: Icons.description_outlined,
              secondary: true,
              onPressed: () => _push(
                  context, ReportScreen(store: widget.store, project: p))),
          if (all.any((m) => m.cell == null)) ...[
            _gap,
            const SectionTitle('Unassigned recordings'),
            _small,
            for (final m in all.where((m) => m.cell == null)) ...[
              _MeasurementTile(
                  m: m,
                  onTap: () => _push(context,
                      MeasurementScreen(store: widget.store, measurement: m))),
              _small
            ]
          ],
        ]);
      });
}

class _NotesDialog extends StatefulWidget {
  const _NotesDialog({required this.notes});
  final String notes;
  @override
  State<_NotesDialog> createState() => _NotesDialogState();
}

class _NotesDialogState extends State<_NotesDialog> {
  late final TextEditingController controller =
      TextEditingController(text: widget.notes);
  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AlertDialog(
          title: const Text('Field notes'),
          content: SizedBox(
              width: 420,
              child: TextField(
                  controller: controller,
                  minLines: 5,
                  maxLines: 10,
                  maxLength: 5000,
                  decoration: const InputDecoration(
                      hintText: 'Material, location, method, conditions…'))),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Cancel')),
            FilledButton(
                onPressed: () => Navigator.pop(context, controller.text.trim()),
                child: const Text('Save notes'))
          ]);
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({required this.store, super.key});
  final ThudStore store;
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  SensingIntent? intent;
  String query = '';
  bool microphoneOnly = false;
  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: widget.store,
      builder: (_, __) {
        final entries = widget.store.measurements
            .where((m) =>
                (intent == null || intent == _Evidence.intentOf(m)) &&
                (!microphoneOnly || !_synthetic(m)) &&
                '${m.label} ${m.source} ${_Evidence.intentOf(m).title} ${_Evidence.headline(m)}'
                    .toLowerCase()
                    .contains(query.toLowerCase()))
            .toList()
          ..sort((a, b) => b.created.compareTo(a.created));
        return _RoutePage(title: 'Measurement history', children: [
          const PageHeading('Saved recordings',
              subtitle: 'Newest first, stored on this device'),
          _gap,
          TextField(
              decoration: const InputDecoration(
                  prefixIcon:
                      Icon(Icons.search_rounded, size: 20, color: C.muted),
                  hintText: 'Search recordings',
                  border: OutlineInputBorder()),
              onChanged: (v) => setState(() => query = v)),
          _small,
          Wrap(spacing: 8, runSpacing: 8, children: [
            ChoiceChip(
                label: const Text('All modes'),
                selected: intent == null,
                onSelected: (_) => setState(() => intent = null)),
            for (final mode in SensingIntent.values)
              ChoiceChip(
                  avatar: Icon(mode.icon, size: 16, color: mode.color),
                  label: Text(mode.title),
                  selectedColor: mode.color.withValues(alpha: .12),
                  selected: intent == mode,
                  onSelected: (_) => setState(() => intent = mode))
          ]),
          SwitchListTile.adaptive(
              contentPadding: EdgeInsets.zero,
              title: const Text('Microphone recordings only',
                  style: TextStyle(fontSize: 14)),
              value: microphoneOnly,
              onChanged: (v) => setState(() => microphoneOnly = v)),
          _small,
          if (entries.isEmpty)
            const _Empty('No matching recordings',
                'Save a recording, or adjust your filters.')
          else
            for (final m in entries) ...[
              _MeasurementTile(
                  m: m,
                  onTap: () => _push(context,
                      MeasurementScreen(store: widget.store, measurement: m)),
                  trailing: IconButton(
                      tooltip: 'Delete recording',
                      icon: const Icon(Icons.delete_outline_rounded,
                          size: 20, color: C.muted),
                      onPressed: () => _delete(context, widget.store, m))),
              _small
            ]
        ]);
      });
}

class _MeasurementTile extends StatelessWidget {
  const _MeasurementTile({required this.m, required this.onTap, this.trailing});
  final Measurement m;
  final VoidCallback onTap;
  final Widget? trailing;
  @override
  Widget build(BuildContext context) {
    final intent = _Evidence.intentOf(m);
    final label = _Evidence.displayLabel(m);
    return Material(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        child: ListTile(
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
                side: const BorderSide(color: C.line)),
            leading: Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                    color: intent.color.withValues(alpha: .08),
                    borderRadius: BorderRadius.circular(12)),
                child: Icon(
                    _Evidence.needsRetry(m)
                        ? Icons.refresh_rounded
                        : intent.icon,
                    color: intent.color,
                    size: 24)),
            title: Text(label,
                style:
                    const TextStyle(fontSize: 15, fontWeight: FontWeight.w600)),
            subtitle: Text(
                '${label == intent.title ? '' : '${intent.title} · '}${_Evidence.headline(m)}\n${_date(m.created)}${_synthetic(m) ? ' · Demo' : ''}',
                style:
                    const TextStyle(color: C.muted, height: 1.5, fontSize: 12)),
            isThreeLine: true,
            trailing: trailing ??
                const Icon(Icons.chevron_right_rounded,
                    size: 20, color: C.muted),
            onTap: onTap));
  }
}

Future<bool> _delete(
    BuildContext context, ThudStore store, Measurement m) async {
  final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
              title: const Text('Delete this recording?'),
              content: const Text(
                  'The saved trace and its project association will be removed from this device. This cannot be undone.'),
              actions: [
                TextButton(
                    onPressed: () => Navigator.pop(context, false),
                    child: const Text('Keep recording')),
                FilledButton(
                    onPressed: () => Navigator.pop(context, true),
                    child: const Text('Delete'))
              ]));
  if (confirmed != true) return false;
  try {
    await store.deleteMeasurement(m.id);
    return true;
  } catch (e) {
    if (context.mounted) _notice(context, e);
    return false;
  }
}

class CalibrationScreen extends StatelessWidget {
  const CalibrationScreen(
      {required this.store, required this.onRecord, super.key});
  final ThudStore store;
  final VoidCallback onRecord;
  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: store,
      builder: (_, __) {
        final candidates = store.measurements.where(_validReference).toList()
          ..sort((a, b) => b.created.compareTo(a.created));
        return _RoutePage(title: 'Reference & calibration', children: [
          const PageHeading('Reference recording',
              subtitle: 'Keep the same phone, tool and distance.'),
          _gap,
          if (store.reference case final ref?) ...[
            const SectionTitle('Current reference'),
            _small,
            _MeasurementTile(
                m: ref,
                onTap: () => _push(context,
                    MeasurementScreen(store: store, measurement: ref))),
            _gap
          ],
          ActionButton('Record a reference',
              icon: Icons.mic_rounded, onPressed: onRecord),
          _gap,
          const SectionTitle('Eligible saved recordings'),
          _small,
          const Text('Clean microphone recordings only.', style: _body),
          _gap,
          if (candidates.isEmpty)
            const _Empty('No eligible reference yet',
                'Record and save a clean tap to get started.')
          else
            for (final m in candidates) ...[
              _MeasurementTile(
                  m: m,
                  onTap: () => _push(
                      context, MeasurementScreen(store: store, measurement: m)),
                  trailing: IconButton(
                      tooltip: store.reference?.id == m.id
                          ? 'Current reference'
                          : 'Use as reference',
                      icon: Icon(
                          store.reference?.id == m.id
                              ? Icons.check_circle_rounded
                              : Icons.radio_button_unchecked,
                          color: store.reference?.id == m.id ? C.blue : C.muted,
                          size: 20),
                      onPressed: () => _perform(context, () async {
                            await store.setReference(m);
                            if (context.mounted) {
                              _notice(context, 'Reference recording updated.');
                            }
                          }))),
              _small
            ]
        ]);
      });
}

class MeasurementScreen extends StatelessWidget {
  const MeasurementScreen(
      {required this.store, required this.measurement, super.key});
  final ThudStore store;
  final Measurement measurement;
  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: store,
      builder: (_, __) {
        var m = measurement;
        var saved = false;
        for (final entry in store.measurements) {
          if (entry.id == measurement.id) {
            m = entry;
            saved = true;
            break;
          }
        }
        final intent = _Evidence.intentOf(m);
        final retry = _Evidence.needsRetry(m);
        Project? project;
        for (final entry in store.projects) {
          if (entry.id == m.projectId) {
            project = entry;
            break;
          }
        }
        final linkedProject = project;
        return Theme(
          data: Theme.of(context).copyWith(
              colorScheme: Theme.of(context)
                  .colorScheme
                  .copyWith(primary: intent.color, onPrimary: Colors.white)),
          child: _RoutePage(title: intent.title, actions: [
            if (saved)
              IconButton(
                  tooltip: 'Delete recording',
                  icon: const Icon(Icons.delete_outline_rounded),
                  onPressed: () async {
                    if (await _delete(context, store, m) && context.mounted) {
                      Navigator.pop(context);
                    }
                  })
          ], children: [
            PageHeading(_Evidence.displayLabel(m), subtitle: _date(m.created)),
            _gap,
            _Evidence(m: m, reference: store.reference),
            _small,
            SignalExplorer(measurement: m),
            _gap,
            if (!retry && m.intent == 'slab' && linkedProject != null) ...[
              ActionButton('Handover report',
                  icon: Icons.description_outlined,
                  onPressed: () => _push(context,
                      ReportScreen(store: store, project: linkedProject))),
              _small,
            ] else if (!retry && intent == SensingIntent.slab) ...[
              ActionButton('Add to handover',
                  icon: Icons.playlist_add_rounded,
                  onPressed: () => _perform(context, () async {
                        if (await _saveMeasurement(context, store, m) &&
                            context.mounted) {
                          _notice(context, 'Saved locally.');
                        }
                      })),
              _small,
            ],
            if (!retry && !saved && intent != SensingIntent.slab) ...[
              ActionButton('Save measurement',
                  icon: Icons.bookmark_add_outlined,
                  onPressed: () => _perform(context, () async {
                        if (await _saveMeasurement(context, store, m) &&
                            context.mounted) {
                          _notice(context, 'Saved locally.');
                        }
                      })),
              _small,
            ],
            if (saved && _validReference(m)) ...[
              ActionButton(
                  store.reference?.id == m.id
                      ? 'Current reference'
                      : 'Use as reference',
                  icon: Icons.tune_rounded,
                  secondary: true,
                  onPressed: store.reference?.id == m.id
                      ? null
                      : () => _perform(context, () => store.setReference(m))),
              _small,
            ],
          ]),
        );
      });
}

class ReportScreen extends StatefulWidget {
  const ReportScreen({required this.store, required this.project, super.key});
  final ThudStore store;
  final Project project;
  @override
  State<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  bool _exporting = false;
  Future<void> _export() async {
    setState(() => _exporting = true);
    try {
      final regular =
          pw.Font.ttf(await rootBundle.load('assets/fonts/Nunito-Regular.ttf'));
      final semibold = pw.Font.ttf(
          await rootBundle.load('assets/fonts/Nunito-SemiBold.ttf'));
      final document = pw.Document(
          theme: pw.ThemeData.withFont(base: regular, bold: semibold));
      final text = widget.store.reportText(widget.project);
      final marks = widget.store.annotationsForProject(widget.project.id);
      final plan = widget.project.floorPlan;
      final planImage = plan == null ? null : await floorPlanPdfImage(plan);
      final latestMarks = <int, DefectAnnotation>{};
      for (final mark in marks) {
        latestMarks.putIfAbsent(mark.cell, () => mark);
      }
      document.addPage(pw.MultiPage(
          pageFormat: PdfPageFormat.a4,
          margin: const pw.EdgeInsets.all(40),
          header: (_) => pw.Padding(
              padding: const pw.EdgeInsets.only(bottom: 18),
              child: pw.Text('THUD / HANDOVER INSPECTION',
                  style: pw.TextStyle(
                      fontSize: 10,
                      color: PdfColors.blue800,
                      fontWeight: pw.FontWeight.bold))),
          footer: (context) => pw.Align(
              alignment: pw.Alignment.centerRight,
              child: pw.Text(
                  'Page ${context.pageNumber} of ${context.pagesCount}',
                  style: const pw.TextStyle(fontSize: 9))),
          build: (_) => [
                pw.Text(widget.project.name,
                    style: pw.TextStyle(
                        fontSize: 26, fontWeight: pw.FontWeight.bold)),
                pw.SizedBox(height: 12),
                pw.Text('Room: ${widget.project.room}'),
                pw.Text(
                    'Findings: ${marks.length} · Review: ${marks.where((m) => m.status == AnnotationStatus.review).length} · Defects: ${marks.where((m) => m.status == AnnotationStatus.defect).length} · Accepted: ${marks.where((m) => m.status == AnnotationStatus.accepted).length}'),
                pw.SizedBox(height: 14),
                if (plan != null && planImage != null) ...[
                  pw.Text('Architectural plan / ${plan.name}',
                      style: pw.TextStyle(
                          fontSize: 16, fontWeight: pw.FontWeight.bold)),
                  pw.SizedBox(height: 8),
                  floorPlanPdfMap(plan, planImage, marks),
                  pw.Text(
                      'Numbered human findings: review = amber, defect = red, accepted = green.',
                      style: const pw.TextStyle(fontSize: 9)),
                  pw.Text(
                      '${marks.where((a) => a.planX == null).length} finding(s) have no plan location; see grid and finding details.',
                      style: const pw.TextStyle(fontSize: 9)),
                  pw.SizedBox(height: 14),
                ],
                ...text.split('\n').map((line) => pw.Padding(
                    padding: const pw.EdgeInsets.only(bottom: 5),
                    child: pw.Text(line,
                        style:
                            const pw.TextStyle(fontSize: 10, lineSpacing: 3)))),
                pw.Text('Inspection map & findings',
                    style: pw.TextStyle(
                        fontSize: 16, fontWeight: pw.FontWeight.bold)),
                pw.SizedBox(height: 10),
                pw.Table(children: [
                  for (var row = 0; row < widget.project.rows; row++)
                    pw.TableRow(children: [
                      for (var col = 0; col < widget.project.columns; col++)
                        pw.Container(
                            padding: const pw.EdgeInsets.all(5),
                            decoration: pw.BoxDecoration(
                                border: pw.Border.all(color: PdfColors.grey300),
                                color: switch (latestMarks[
                                        row * widget.project.columns + col]
                                    ?.status) {
                                  AnnotationStatus.defect => PdfColors.red100,
                                  AnnotationStatus.accepted =>
                                    PdfColors.green100,
                                  AnnotationStatus.review =>
                                    PdfColors.orange100,
                                  null => PdfColors.grey100,
                                }),
                            child: pw.Text(
                                'R${row + 1}:C${col + 1}\n${latestMarks[row * widget.project.columns + col]?.status.name ?? "unmarked"}',
                                style: const pw.TextStyle(fontSize: 7))),
                    ]),
                ]),
                pw.SizedBox(height: 12),
                for (var i = 0; i < marks.length; i++)
                  pw.Padding(
                      padding: const pw.EdgeInsets.only(bottom: 8),
                      child: pw.Text(
                          '#${i + 1} / ${_cellName(marks[i].cell, widget.project.columns)} / ${marks[i].status.name}'
                          '${marks[i].planX == null ? " / no plan location" : ""}\n${marks[i].note}',
                          style: const pw.TextStyle(fontSize: 10))),
                for (final (index, mark) in marks.indexed
                    .where((entry) => entry.$2.photoBase64 != null))
                  pw.Column(
                      crossAxisAlignment: pw.CrossAxisAlignment.start,
                      children: [
                        pw.SizedBox(height: 16),
                        pw.Text(
                            'Photo / #${index + 1} / ${_cellName(mark.cell, widget.project.columns)} / ${mark.status.name}',
                            style: const pw.TextStyle(fontSize: 10)),
                        pw.SizedBox(height: 8),
                        pw.Container(
                            width: 320,
                            child: pw.Stack(children: [
                              pw.Image(
                                  pw.MemoryImage(
                                      base64Decode(mark.photoBase64!)),
                                  width: 320),
                              if (mark.x != null && mark.y != null)
                                pw.Positioned.fill(
                                    child: pw.Align(
                                        alignment: pw.Alignment(
                                            mark.x! * 2 - 1, mark.y! * 2 - 1),
                                        child: pw.Container(
                                            width: 10,
                                            height: 10,
                                            decoration: const pw.BoxDecoration(
                                                color: PdfColors.red,
                                                shape: pw.BoxShape.circle)))),
                            ])),
                        pw.Text(mark.note,
                            style: const pw.TextStyle(fontSize: 10)),
                      ]),
              ]));
      final bytes = await document.save();
      await Printing.sharePdf(
          bytes: bytes, filename: 'THUD-${widget.project.id}.pdf');
    } catch (e) {
      if (mounted) _notice(context, 'PDF export failed: $e');
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: widget.store,
      builder: (_, __) {
        final p = widget.project;
        final all = widget.store.forProject(p.id).toList()
          ..sort((a, b) => b.created.compareTo(a.created));
        final latest = <int, Measurement>{};
        for (final m in all) {
          if (m.cell != null) latest.putIfAbsent(m.cell!, () => m);
        }
        final marks = widget.store.annotationsForProject(p.id);
        final review =
            marks.where((m) => m.status == AnnotationStatus.review).length;
        final defects =
            marks.where((m) => m.status == AnnotationStatus.defect).length;
        final accepted =
            marks.where((m) => m.status == AnnotationStatus.accepted).length;
        final actual = latest.values.where((m) => !_synthetic(m)).length;
        return _RoutePage(title: 'Handover report', children: [
          PageHeading(p.name, subtitle: p.room),
          _gap,
          Row(children: [
            Metric('Review', '$review', accent: C.amber),
            Metric('Defects', '$defects', accent: C.red),
            Metric('Accepted', '$accepted', accent: C.green),
          ]),
          _small,
          ActionButton(_exporting ? 'Preparing…' : 'Export handover PDF',
              icon: Icons.ios_share_rounded,
              onPressed: _exporting ? null : _export),
          _gap,
          SectionTitle('Room map',
              trailing: TextButton(
                  onPressed: () => _push(context,
                      DefectMapScreen(store: widget.store, project: p)),
                  child: const Text('Edit findings'))),
          _small,
          if (p.floorPlan case final plan?) ...[
            FloorPlanCanvas(plan: plan, marks: marks),
            _small,
            Text('${plan.name} · Review: amber · Defect: red · Accepted: green',
                style: _body),
            Text(
                '${marks.where((a) => a.planX == null).length} finding(s) not positioned on plan.',
                style: _body),
            _small,
          ],
          GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: p.columns,
                  mainAxisSpacing: 6,
                  crossAxisSpacing: 6),
              itemCount: p.rows * p.columns,
              itemBuilder: (context, i) {
                final cellMarks = marks.where((m) => m.cell == i);
                final status =
                    cellMarks.isEmpty ? null : cellMarks.first.status;
                final color = switch (status) {
                  AnnotationStatus.review => C.amber,
                  AnnotationStatus.defect => C.red,
                  AnnotationStatus.accepted => C.green,
                  null => C.muted,
                };
                final icon = switch (status) {
                  AnnotationStatus.review => Icons.search_rounded,
                  AnnotationStatus.defect => Icons.close_rounded,
                  AnnotationStatus.accepted => Icons.check_rounded,
                  null => Icons.crop_square_rounded,
                };
                return Semantics(
                    label:
                        '${_cellName(i, p.columns)} ${status?.name ?? "Unmarked"}',
                    child: Container(
                        decoration: BoxDecoration(
                            color: color.withValues(alpha: .09),
                            borderRadius: BorderRadius.circular(12)),
                        child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(icon, color: color, size: 24),
                              Text('${i + 1}', style: TextStyle(color: color)),
                            ])));
              }),
          _small,
          Text(
              '$actual / ${p.rows * p.columns} microphone points · ${all.length} recordings',
              style: _body),
          _gap,
          for (final (index, mark) in marks.indexed) ...[
            Panel(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Row(children: [
                    Icon(
                        mark.status == AnnotationStatus.accepted
                            ? Icons.check_circle_outline_rounded
                            : Icons.location_on_outlined,
                        color: floorPlanStatusColor(mark.status)),
                    const SizedBox(width: 8),
                    Expanded(
                        child: Text(
                            '#${index + 1} · ${_cellName(mark.cell, p.columns)} · ${mark.status.name}',
                            style: _title)),
                  ]),
                  if (mark.note.isNotEmpty) ...[
                    _small,
                    Text(mark.note, style: _body)
                  ],
                  if (mark.photoBase64 != null) ...[
                    _small,
                    Image.memory(base64Decode(mark.photoBase64!),
                        height: 160, fit: BoxFit.contain)
                  ],
                ])),
            _small,
          ],
          if (latest.isNotEmpty)
            ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: const Text('Measured results'),
                children: [
                  for (final entry in latest.entries) ...[
                    Align(
                        alignment: Alignment.centerLeft,
                        child: Text(_cellName(entry.key, p.columns),
                            style: _title)),
                    _Evidence(
                        m: entry.value, reference: widget.store.reference),
                    _gap,
                  ],
                ]),
          if (p.notes.isNotEmpty) ...[_gap, Text(p.notes, style: _body)],
        ]);
      });
}

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({required this.store, super.key});
  final ThudStore store;
  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: store,
      builder: (_, __) => _RoutePage(title: 'Settings', children: [
            Panel(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                child: Column(children: [
                  SwitchListTile.adaptive(
                      title: const Text('Haptic feedback'),
                      value: store.haptics,
                      onChanged: (v) => _perform(
                          context, () => store.setFeedback(haptics: v))),
                  const Divider(height: 1),
                  SwitchListTile.adaptive(
                      title: const Text('Interface sounds'),
                      value: store.sounds,
                      onChanged: (v) =>
                          _perform(context, () => store.setFeedback(sounds: v)))
                ])),
            _gap,
            const Row(children: [
              Icon(Icons.phonelink_lock_outlined, color: C.muted, size: 20),
              SizedBox(width: 12),
              Expanded(
                  child: Text('Recordings & photos stay on this device.',
                      style: _body)),
            ]),
          ]));
}
