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

String _moduleName(Module m) => switch (m) {
      Module.slab => 'Slab',
      Module.vessel => 'Vessel',
      Module.scope => 'Scope'
    };
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
          child: Center(
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
      super.key});
  final ThudStore store;
  final Module initialModule;
  final String? projectId;
  final int? cell;
  final ValueChanged<Module> onModule;
  @override
  State<ScopeScreen> createState() => _ScopeScreenState();
}

class _ScopeScreenState extends State<ScopeScreen> with WidgetsBindingObserver {
  final CaptureService _capture = CaptureService();
  final Stopwatch _elapsed = Stopwatch();
  Timer? _limit, _clock;
  late Module _module;
  late SensingIntent _intent;
  Measurement? _result;
  String? _error;
  bool _busy = false, _recording = false, _saved = false;
  @override
  void initState() {
    super.initState();
    _module = widget.initialModule;
    _intent = switch (_module) {
      Module.slab => SensingIntent.slab,
      Module.vessel => SensingIntent.vessel,
      Module.scope => SensingIntent.scope,
    };
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
    }
  }

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
        setState(() => _result = result.withIntent(_intent.name));
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
          _result = m.withIntent(_intent.name);
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

  Future<void> _chooseIntent() async {
    final intent = await Navigator.of(context).push<SensingIntent>(
        sensingRoute<SensingIntent>(
            context,
            ChooseObjectScreen(
                initial: _intent, haptics: widget.store.haptics)));
    if (intent == null || !mounted) return;
    setState(() {
      _intent = intent;
      _module = intent.module;
      _result = null;
      _saved = false;
    });
    widget.onModule(_module);
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
                const PageHeading('Place this finding.'),
                _gap,
                for (final p in widget.store.projects)
                  ListTile(
                      title: Text(p.name),
                      subtitle: Text('${p.room} · ${p.rows} × ${p.columns}'),
                      onTap: () => Navigator.pop(sheet, p)),
                ActionButton('Create inspection map', icon: Icons.add_rounded,
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
    _push(context,
        DefectMapScreen(store: widget.store, project: project, measurement: m));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopTimers();
    _capture.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) =>
      LayoutBuilder(builder: (context, bounds) {
        final m = _result;
        final reduced = MediaQuery.disableAnimationsOf(context);
        final stageHeight = (bounds.maxHeight * .45).clamp(240.0, 340.0);
        return SingleChildScrollView(
            child: Container(
          constraints: BoxConstraints(minHeight: bounds.maxHeight),
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 12),
          child: Container(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
            decoration: BoxDecoration(
                color: C.blue, borderRadius: BorderRadius.circular(36)),
            child: Column(children: [
              Row(children: [
                Expanded(
                    child: TextButton.icon(
                  onPressed: _busy || _recording ? null : _chooseIntent,
                  style: TextButton.styleFrom(foregroundColor: Colors.white),
                  icon: const Icon(Icons.swap_horiz_rounded),
                  label: Text(_intent.title),
                )),
                IconButton(
                    tooltip: 'How to tap',
                    onPressed: _instructions,
                    icon: const Icon(Icons.help_outline_rounded,
                        color: Colors.white70, size: 20)),
              ]),
              const SizedBox(height: 32),
              AnimatedSwitcher(
                duration:
                    reduced ? Duration.zero : const Duration(milliseconds: 320),
                transitionBuilder: _sceneTransition,
                child: Text(
                    _busy
                        ? 'Getting ready'
                        : _recording
                            ? 'Listening…'
                            : m == null
                                ? 'Tap. Listen.\nKnow.'
                                : 'Got it.',
                    key: ValueKey('$_busy-$_recording-${m == null}'),
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 40,
                        fontWeight: FontWeight.w600,
                        height: 1.05,
                        letterSpacing: -1)),
              ),
              const SizedBox(height: 12),
              if (m == null)
                Text(
                    _intent == SensingIntent.devices
                        ? 'Power off. Unplug. Tap casing gently.'
                        : _recording
                            ? 'Tap once. Let it ring.'
                            : 'Touch below to begin',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 16)),
              SizedBox(
                  height: m == null ? stageHeight : 96,
                  width: double.infinity,
                  child: ListeningStage(
                    live: _capture.live,
                    active: _recording,
                    busy: _busy,
                    complete: m != null,
                    onTap: _busy
                        ? null
                        : _recording
                            ? _stop
                            : _start,
                  )),
              AnimatedSwitcher(
                duration:
                    reduced ? Duration.zero : const Duration(milliseconds: 360),
                switchInCurve: Curves.easeOutCubic,
                switchOutCurve: Curves.easeInCubic,
                transitionBuilder: _sceneTransition,
                child: m != null
                    ? SensingResult(
                        key: ValueKey('result-${m.id}-${_intent.name}'),
                        measurement: m,
                        intent: _intent,
                        store: widget.store,
                        saved: _saved,
                        onSave: _save,
                        onMap: _map,
                        onTrace: (record) => _push(
                            context,
                            MeasurementScreen(
                                store: widget.store, measurement: record)),
                        onAgain: () => setState(() {
                          _result = null;
                          _saved = false;
                        }),
                      )
                    : SizedBox(
                        key: ValueKey(_recording ? 'recording' : 'ready'),
                        height: 64,
                        child: Center(
                          child: _recording
                              ? Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                      Text(
                                          '${(_elapsed.elapsedMilliseconds / 1000).toStringAsFixed(1)} / 8s',
                                          style: const TextStyle(
                                              color: Colors.white70,
                                              fontSize: 14)),
                                      const SizedBox(width: 16),
                                      TextButton(
                                          onPressed: _busy ? null : _cancel,
                                          style: TextButton.styleFrom(
                                              foregroundColor: Colors.white),
                                          child: const Text('Cancel')),
                                    ])
                              : Wrap(
                                  alignment: WrapAlignment.center,
                                  spacing: 8,
                                  children: [
                                      TextButton(
                                          onPressed:
                                              _busy ? null : () => _sample(),
                                          style: TextButton.styleFrom(
                                              foregroundColor: Colors.white),
                                          child: const Text('Try a sample')),
                                      if (_intent == SensingIntent.slab ||
                                          _intent == SensingIntent.devices)
                                        TextButton(
                                            onPressed: _busy
                                                ? null
                                                : () => _sample(resonant: true),
                                            style: TextButton.styleFrom(
                                                foregroundColor:
                                                    Colors.white70),
                                            child:
                                                const Text('Long-ring sample')),
                                    ]),
                        )),
              ),
              if (widget.projectId != null)
                Padding(
                    padding: const EdgeInsets.only(top: 12),
                    child: Text('Project · point ${(widget.cell ?? 0) + 1}',
                        style: const TextStyle(
                            color: Colors.white70, fontSize: 12))),
              if (_error != null)
                Padding(
                    padding: const EdgeInsets.only(top: 16),
                    child: Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(20)),
                      child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Icon(Icons.info_outline_rounded,
                                color: C.red, size: 20),
                            const SizedBox(width: 12),
                            Expanded(
                                child: Text(_error!,
                                    style: const TextStyle(
                                        color: C.ink, fontSize: 13)))
                          ]),
                    )),
            ]),
          ),
        ));
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
                        _push(
                            context,
                            GuidedReferenceScreen(
                                intent: _intent, store: widget.store));
                      },
                      child: const Text('Set reference')),
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
  @override
  Widget build(BuildContext context) =>
      Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(
                _synthetic(m) ? Icons.science_outlined : Icons.mic_none_rounded,
                color: C.muted,
                size: 20),
            const SizedBox(width: 8),
            Expanded(
                child: Text(
                    _synthetic(m)
                        ? 'Synthetic sample · not an inspection'
                        : 'Recorded acoustic evidence',
                    style: _title))
          ]),
          _small,
          Text(m.source, style: _body),
          if (m.flags.isNotEmpty) ...[
            _small,
            Text('Signal review: ${m.flags.join(' · ')}',
                style:
                    const TextStyle(color: C.amber, fontSize: 14, height: 1.5))
          ] else ...[
            _small,
            const Text(
                'No automated signal-quality flags. This is not a material assessment.',
                style: _body)
          ]
        ]),
        _small,
        Row(children: [
          Metric('Peak frequency', '${_number(m.peakHz, 0)} Hz',
              accent: C.blue),
          const SizedBox(width: 8),
          Metric('20 dB decay', '${_number(m.decayMs, 0)} ms')
        ]),
        _small,
        Row(children: [
          Metric('RMS level', '${_number(m.rmsDb)} dBFS'),
          const SizedBox(width: 8),
          Metric('Est. signal/noise', '${_number(m.snrDb)} dB')
        ]),
        _small,
        Row(children: [
          Metric('Band energy ratio', '${_number(m.hiiDb)} dB'),
          const SizedBox(width: 8),
          const Expanded(
              child: Text(
                  '450–2200 Hz / 2200–6500 Hz energy. Descriptive only.',
                  style: _body))
        ]),
        if (reference != null && reference!.id != m.id) ...[
          _small,
          Text(
              'Compared with your reference: peak frequency Δ ${_number(m.peakHz - reference!.peakHz, 0)} Hz; decay Δ ${_number(m.decayMs - reference!.decayMs, 0)} ms. Differences are descriptive only, not a defect threshold.',
              style: _body)
        ],
      ]);
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
                title: 'Teach a three-tap reference',
                subtitle: 'Solid tile references and known vessel fill anchors',
                onTap: () async {
                  final intent = await Navigator.of(context)
                      .push<SensingIntent>(sensingRoute<SensingIntent>(
                          context,
                          ChooseObjectScreen(
                              initial: SensingIntent.slab,
                              haptics: store.haptics)));
                  if (intent != null && context.mounted) {
                    _push(context,
                        GuidedReferenceScreen(intent: intent, store: store));
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
            PageHeading('Projects',
                subtitle: 'Organize recordings by room and grid',
                trailing: IconButton(
                    tooltip: 'Create project',
                    onPressed: () => _create(context),
                    icon: const Icon(Icons.add_rounded,
                        color: C.muted, size: 20))),
            _gap,
            if (store.projects.isEmpty)
              _Empty('No projects yet',
                  'Create a project, define its grid, then attach real recordings to each inspection point.',
                  icon: Icons.grid_view_rounded,
                  action: ActionButton('Create project',
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
              ActionButton('New project',
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
          title: const Text('Create a project'),
          content: SizedBox(
              width: 400,
              child: SingleChildScrollView(
                  child: Form(
                      key: _form,
                      child: Column(mainAxisSize: MainAxisSize.min, children: [
                        TextFormField(
                            controller: _name,
                            decoration: const InputDecoration(
                                labelText: 'Project name'),
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
              'Select a cell to review its latest recording or make a new capture. A recorded cell is not a material assessment.',
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
          ActionButton('Defect map & photos',
              icon: Icons.location_on_outlined,
              onPressed: () => _push(
                  context,
                  DefectMapScreen(
                      store: widget.store, project: p, measurement: active))),
          _small,
          ActionButton('Project report',
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
  Module? module;
  String query = '';
  bool microphoneOnly = false;
  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: widget.store,
      builder: (_, __) {
        final entries = widget.store.measurements
            .where((m) =>
                (module == null || module == m.module) &&
                (!microphoneOnly || !_synthetic(m)) &&
                '${m.label} ${m.source} ${_moduleName(m.module)}'
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
                label: const Text('All modules'),
                selected: module == null,
                onSelected: (_) => setState(() => module = null)),
            for (final m in Module.values)
              ChoiceChip(
                  label: Text(_moduleName(m)),
                  selected: module == m,
                  onSelected: (_) => setState(() => module = m))
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
                'Save a capture from the field scope, or adjust your filters.')
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
  Widget build(BuildContext context) => Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(16),
      child: ListTile(
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
              side: const BorderSide(color: C.line)),
          leading: Icon(
              _synthetic(m) ? Icons.science_outlined : Icons.graphic_eq_rounded,
              color: C.muted,
              size: 20),
          title: Text(m.label,
              style:
                  const TextStyle(fontSize: 14, fontWeight: FontWeight.w500)),
          subtitle: Text(
              '${m.intent ?? _moduleName(m.module)} · ${_number(m.peakHz, 0)} Hz\n${_date(m.created)}${_synthetic(m) ? ' · Synthetic' : ''}',
              style:
                  const TextStyle(color: C.muted, height: 1.5, fontSize: 12)),
          isThreeLine: true,
          trailing: trailing ??
              const Icon(Icons.chevron_right_rounded, size: 20, color: C.muted),
          onTap: onTap));
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
        final m = measurement;
        final saved = store.measurements.any((entry) => entry.id == m.id);
        return _RoutePage(title: 'Measurement', actions: [
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
          PageHeading(m.label,
              subtitle:
                  '${m.intent ?? _moduleName(m.module)} · ${_date(m.created)}'),
          _gap,
          if (m.context.isNotEmpty) ...[
            Text(m.context['interpretation'] ?? 'Saved inspection context',
                style: _title),
            _small,
            for (final entry in m.context.entries)
              Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text('${entry.key}: ${entry.value}', style: _body)),
            _gap,
          ],
          _Evidence(m: m, reference: store.reference),
          _gap,
          SignalExplorer(measurement: m),
          _gap,
          Panel(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                const Label('Recording provenance'),
                _small,
                SelectableText(
                    'Source: ${m.source}\nCaptured: ${_date(m.created)}\nModule: ${_moduleName(m.module)}\nSignal flags: ${m.flags.isEmpty ? 'None detected' : m.flags.join(', ')}\nRecord ID: ${m.id}',
                    style: _body),
                _small,
                const Text(
                    'Decay is an estimated 20 dB amplitude decay duration, not RT60. Signal-to-noise is an algorithmic estimate. Ambient sound, microphone processing and tap strength can change every metric.',
                    style: _body)
              ])),
          _gap,
          if (!saved)
            ActionButton('Save measurement',
                icon: Icons.bookmark_add_outlined,
                onPressed: () => _perform(context, () async {
                      if (await _saveMeasurement(context, store, m) &&
                          context.mounted) {
                        _notice(context, 'Saved locally.');
                      }
                    })),
          if (saved && _validReference(m)) ...[
            _small,
            ActionButton(
                store.reference?.id == m.id
                    ? 'Current reference'
                    : 'Use as reference',
                icon: Icons.tune_rounded,
                secondary: true,
                onPressed: store.reference?.id == m.id
                    ? null
                    : () => _perform(context, () => store.setReference(m)))
          ],
          _gap,
          const Text(
              'Acoustic screening only. Vessel interpolation requires matched reference anchors; fruit firmness is a research proxy. Human map marks are inspection notes, not automatic defect or safety certification.',
              style: _body)
        ]);
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
      final latestMarks = <int, DefectAnnotation>{};
      for (final mark in marks) {
        latestMarks.putIfAbsent(mark.cell, () => mark);
      }
      document.addPage(pw.MultiPage(
          pageFormat: PdfPageFormat.a4,
          margin: const pw.EdgeInsets.all(40),
          header: (_) => pw.Padding(
              padding: const pw.EdgeInsets.only(bottom: 18),
              child: pw.Text('THUD / ACOUSTIC FIELD RECORD',
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
                pw.Text('Human-marked inspection map',
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
                pw.SizedBox(height: 14),
                ...text.split('\n').map((line) => pw.Padding(
                    padding: const pw.EdgeInsets.only(bottom: 5),
                    child: pw.Text(line,
                        style:
                            const pw.TextStyle(fontSize: 10, lineSpacing: 3)))),
                for (final mark in marks.where((a) => a.photoBase64 != null))
                  pw.Column(
                      crossAxisAlignment: pw.CrossAxisAlignment.start,
                      children: [
                        pw.SizedBox(height: 16),
                        pw.Text(
                            'Photo / ${_cellName(mark.cell, widget.project.columns)} / human ${mark.status.name}',
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
                pw.SizedBox(height: 16),
                pw.Text(
                    'Disclosure: Recorded airborne acoustic evidence only. No material classification, confidence score, structural safety determination or calibrated dB SPL measurement is provided. Synthetic samples, when present, are demonstration data and must not be treated as inspection evidence.',
                    style: const pw.TextStyle(fontSize: 10))
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
        final actual = latest.values.where((m) => !_synthetic(m)).length;
        return _RoutePage(title: 'Project report', children: [
          PageHeading(p.name, subtitle: '${p.room} · acoustic field record'),
          _gap,
          Row(children: [
            Metric('Mic coverage', '$actual / ${p.rows * p.columns}',
                accent: C.blue),
            const SizedBox(width: 8),
            Metric('All recordings', '${all.length}')
          ]),
          _small,
          const Text(
              'Coverage counts cells whose latest recording came from the microphone. It does not indicate material condition or quality.',
              style: _body),
          _gap,
          Panel(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                const Label('Field notes'),
                _small,
                Text(p.notes.isEmpty ? 'No field notes supplied.' : p.notes,
                    style: _body)
              ])),
          _gap,
          SectionTitle('Human-marked findings',
              trailing: TextButton(
                  onPressed: () => _push(context,
                      DefectMapScreen(store: widget.store, project: p)),
                  child: const Text('Open defect map'))),
          _small,
          for (final mark in widget.store.annotationsForProject(p.id)) ...[
            Text(
                '${_cellName(mark.cell, p.columns)} · ${mark.status.name} · human annotation',
                style: _title),
            Text(
                '${mark.note}\n${mark.measurementSource ?? "No linked acoustic recording"}',
                style: _body),
            if (mark.photoBase64 != null)
              Padding(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  child: Image.memory(base64Decode(mark.photoBase64!),
                      height: 160, fit: BoxFit.contain)),
            if (mark.x != null)
              Text(
                  'Photo pin: x=${mark.x!.toStringAsFixed(3)}, y=${mark.y!.toStringAsFixed(3)} (relative image coordinates)',
                  style: _body),
            _gap,
          ],
          const SectionTitle('Latest evidence by cell'),
          _small,
          for (var i = 0; i < p.rows * p.columns; i++)
            Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Panel(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          SizedBox(
                              width: 80,
                              child: Text(_cellName(i, p.columns),
                                  style: const TextStyle(
                                      fontFamily: 'monospace',
                                      color: C.ink,
                                      fontSize: 12,
                                      fontWeight: FontWeight.w500))),
                          Expanded(
                              child: Text(
                                  latest[i] == null
                                      ? 'Not recorded'
                                      : '${_number(latest[i]!.peakHz, 0)} Hz · ${_number(latest[i]!.decayMs, 0)} ms\n${latest[i]!.source}\n${_date(latest[i]!.created)}${latest[i]!.flags.isEmpty ? '' : '\nFlags: ${latest[i]!.flags.join(', ')}'}',
                                  style: _body))
                        ]))),
          if (all.any((m) => m.cell == null)) ...[
            _small,
            Text(
                '${all.where((m) => m.cell == null).length} project recording(s) are not assigned to a grid cell.',
                style: _body)
          ],
          _gap,
          const Text(
              'Disclosure: This report documents measured airborne sound and user notes. It is not a structural inspection, material classifier or safety certificate. Synthetic recordings are explicitly identified and excluded from microphone coverage. Relative digital levels are not calibrated dB SPL.',
              style: _body),
          _gap,
          ActionButton(_exporting ? 'Preparing PDF…' : 'Export PDF',
              icon: Icons.ios_share_rounded,
              onPressed: _exporting ? null : _export)
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
