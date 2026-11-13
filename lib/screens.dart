import 'dart:async';
import 'dart:math' as math;

import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import 'design.dart';
import 'store.dart';

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
    Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => page));
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
const _small = SizedBox(height: 10);
const _body = TextStyle(color: C.muted, height: 1.5, fontSize: 13);
const _title = TextStyle(
    color: C.ink,
    fontWeight: FontWeight.w800,
    fontSize: 18,
    letterSpacing: -.4);

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
        Icon(icon, size: 32, color: C.blue),
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
  Measurement? _result;
  String? _error;
  bool _busy = false, _recording = false, _saved = false;
  @override
  void initState() {
    super.initState();
    _module = widget.initialModule;
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void didUpdateWidget(covariant ScopeScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!_recording && oldWidget.initialModule != widget.initialModule) {
      _module = widget.initialModule;
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
      _feedback();
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
        setState(() => _result = result);
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

  Future<void> _sample() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final m =
          await _capture.sample(_module, resonant: _module == Module.vessel);
      if (mounted) {
        setState(() {
          _result = m;
          _saved = false;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save() async {
    final m = _result;
    if (m == null) return;
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

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopTimers();
    _capture.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AppPage(children: [
        PageHeading('Listen to the material.',
            subtitle: 'A short tap. A measurable acoustic trace.',
            trailing: IconButton(
                tooltip: 'Settings',
                icon: const Icon(Icons.tune_rounded),
                onPressed: _busy || _recording
                    ? null
                    : () =>
                        _push(context, SettingsScreen(store: widget.store)))),
        _gap,
        SizedBox(
            width: double.infinity,
            child: SegmentedButton<Module>(
                segments: const [
                  ButtonSegment(
                      value: Module.slab,
                      label: Text('Slab'),
                      icon: Icon(Icons.layers_outlined)),
                  ButtonSegment(
                      value: Module.vessel,
                      label: Text('Vessel'),
                      icon: Icon(Icons.water_drop_outlined)),
                  ButtonSegment(
                      value: Module.scope,
                      label: Text('Scope'),
                      icon: Icon(Icons.graphic_eq_rounded))
                ],
                selected: {
                  _module
                },
                onSelectionChanged: _busy || _recording
                    ? null
                    : (values) {
                        _feedback();
                        setState(() {
                          _module = values.first;
                          _result = null;
                          _saved = false;
                        });
                        widget.onModule(_module);
                      })),
        _gap,
        Panel(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(_recording ? Icons.mic_rounded : Icons.touch_app_outlined,
                color: C.blue, size: 22),
            const SizedBox(width: 9),
            Expanded(
                child: Label(
                    _recording
                        ? 'Microphone recording'
                        : '${_moduleName(_module)} · ready to capture',
                    color: C.blue)),
            if (_recording)
              Text(
                  '${(_elapsed.elapsedMilliseconds / 1000).toStringAsFixed(1)} / 8s',
                  style: const TextStyle(
                      fontFamily: 'monospace', fontWeight: FontWeight.bold))
          ]),
          _gap,
          Text(
              _recording
                  ? 'Tap once. Let it ring.'
                  : switch (_module) {
                      Module.slab => 'Map the sound of a surface.',
                      Module.vessel => 'Hear the resonance.',
                      Module.scope => 'See the signal, clearly.'
                    },
              style: const TextStyle(
                  fontSize: 25,
                  fontWeight: FontWeight.w800,
                  color: C.ink,
                  letterSpacing: -.8)),
          _small,
          Text(
              _recording
                  ? 'Keep the phone still. Stop after the sound decays; capture ends automatically at 8 seconds.'
                  : 'Keep the microphone close without touching the material. Record a quiet baseline, then make one gentle, consistent tap.',
              style: _body),
          _gap,
          ValueListenableBuilder<List<double>>(
              valueListenable: _capture.live,
              builder: (_, values, __) => _SignalPlot(
                  values: values,
                  height: 120,
                  label: _recording
                      ? 'LIVE PCM · RELATIVE AMPLITUDE'
                      : 'MICROPHONE TRACE',
                  empty: 'Your recording will appear here')),
          _gap,
          Wrap(spacing: 10, runSpacing: 8, children: [
            ActionButton(
                _busy
                    ? 'Please wait'
                    : _recording
                        ? 'Stop capture'
                        : 'Start recording',
                icon: _recording ? Icons.stop_rounded : Icons.mic_rounded,
                onPressed: _busy
                    ? null
                    : _recording
                        ? _stop
                        : _start),
            if (_recording)
              TextButton(
                  onPressed: _busy ? null : _cancel,
                  child: const Text('Cancel'))
            else
              TextButton(
                  onPressed: _busy ? null : _sample,
                  child: const Text('Try synthetic sample'))
          ]),
          if (widget.projectId != null) ...[
            _small,
            Text(
                'Project capture${widget.cell == null ? '' : ' · cell ${widget.cell! + 1}'}',
                style: _body)
          ],
        ])),
        if (_error != null) ...[
          _gap,
          Panel(
              color: const Color(0xFFFFF4ED),
              child:
                  Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Icon(Icons.info_outline_rounded, color: C.amber),
                const SizedBox(width: 12),
                Expanded(
                    child: Text(
                        'Capture unavailable\n$_error\n\nIf permission was denied, enable microphone access in your device settings and try again.',
                        style: _body))
              ]))
        ],
        _gap,
        if (_result case final m?) ...[
          SectionTitle('Acoustic evidence',
              trailing: TextButton(
                  onPressed: () => _push(context,
                      MeasurementScreen(store: widget.store, measurement: m)),
                  child: const Text('Full trace'))),
          _small,
          _Evidence(m: m, reference: widget.store.reference),
          _small,
          Wrap(spacing: 10, children: [
            ActionButton(_saved ? 'Saved' : 'Save measurement',
                icon:
                    _saved ? Icons.check_rounded : Icons.bookmark_add_outlined,
                onPressed: _saved ? null : _save),
            TextButton(
                onPressed: _busy
                    ? null
                    : () => setState(() {
                          _result = null;
                          _saved = false;
                        }),
                child: const Text('Discard'))
          ]),
          _gap,
        ],
        Panel(
            color: C.pale,
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Label('Evidence, not a diagnosis', color: C.blue),
              _small,
              const Text(
                  'THUD measures airborne sound. It does not classify hollowness, defects, fill level or structural safety. Compare repeatable taps under the same conditions.',
                  style: _body),
              TextButton(
                  onPressed: _recording || _busy
                      ? null
                      : () => _push(
                          context,
                          CalibrationScreen(
                              store: widget.store,
                              onRecord: () {
                                Navigator.of(context).pop();
                                _start();
                              })),
                  child: Text(widget.store.reference == null
                      ? 'Set a reference recording →'
                      : 'Review reference recording →'))
            ])),
      ]);
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
                  const Text('Keep this evidence.', style: _title),
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
        Panel(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(
                _synthetic(m) ? Icons.science_outlined : Icons.mic_none_rounded,
                color: C.blue),
            const SizedBox(width: 10),
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
                    const TextStyle(color: C.amber, fontSize: 13, height: 1.4))
          ] else ...[
            _small,
            const Text(
                'No automated signal-quality flags. This is not a material assessment.',
                style: _body)
          ]
        ])),
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
          Panel(
              color: C.pale,
              child: Text(
                  'Compared with your reference: peak frequency Δ ${_number(m.peakHz - reference!.peakHz, 0)} Hz; decay Δ ${_number(m.decayMs - reference!.decayMs, 0)} ms. Differences are descriptive only, not a defect threshold.',
                  style: _body))
        ],
      ]);
}

class _SignalPlot extends StatelessWidget {
  const _SignalPlot(
      {required this.values,
      required this.label,
      this.height = 150,
      this.empty = 'No recorded samples',
      this.spectrum = false});
  final List<double> values;
  final String label, empty;
  final double height;
  final bool spectrum;
  @override
  Widget build(BuildContext context) => Container(
      height: height,
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
          color: C.bg,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: C.line)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Label(label, fontSize: 8),
        const SizedBox(height: 12),
        Expanded(
            child: values.isEmpty
                ? Center(child: Text(empty, style: _body))
                : SizedBox.expand(
                    child: CustomPaint(
                        painter: _TracePainter(values, spectrum: spectrum))))
      ]));
}

class _TracePainter extends CustomPainter {
  _TracePainter(this.values, {required this.spectrum});
  final List<double> values;
  final bool spectrum;
  @override
  void paint(Canvas canvas, Size size) {
    final grid = Paint()
      ..color = C.line
      ..strokeWidth = 1;
    for (var i = 0; i <= 4; i++) {
      final y = size.height * i / 4;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), grid);
    }
    for (var i = 0; i <= 8; i++) {
      final x = size.width * i / 8;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), grid);
    }
    if (values.isEmpty) return;
    var maxValue = 0.0;
    for (final value in values) {
      if (value.isFinite) maxValue = math.max(maxValue, value.abs());
    }
    final path = Path();
    final range = math.max(maxValue, .000001);
    for (var i = 0; i < values.length; i++) {
      final v = values[i].isFinite ? values[i] : 0.0;
      final x = values.length == 1 ? 0.0 : size.width * i / (values.length - 1);
      final y = spectrum
          ? size.height * (1 - v.clamp(0.0, 1.0))
          : size.height / 2 - v / range * size.height * .45;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(
        path,
        Paint()
          ..color = C.blue
          ..strokeWidth = 1.5
          ..style = PaintingStyle.stroke
          ..strokeJoin = StrokeJoin.round);
  }

  @override
  bool shouldRepaint(covariant _TracePainter oldDelegate) =>
      oldDelegate.values != values || oldDelegate.spectrum != spectrum;
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
            const PageHeading('Your field library.',
                subtitle: 'A clear record of what you heard.'),
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
                icon: Icons.settings_outlined,
                title: 'Settings & privacy',
                subtitle: 'Feedback, local storage and field guidance',
                onTap: () => _push(context, SettingsScreen(store: store))),
            _gap,
            const Panel(
                color: C.pale,
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Label('A considered record', color: C.blue),
                      _small,
                      Text(
                          'Keep the context with the signal. Project grids, field notes and traceable recordings are more useful than a single tap in isolation.',
                          style: _body)
                    ]))
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
          borderRadius: BorderRadius.circular(20),
          side: const BorderSide(color: C.line)),
      clipBehavior: Clip.antiAlias,
      child: ListTile(
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
          leading: Icon(icon, color: C.blue),
          title: Text(title,
              style: const TextStyle(
                  fontSize: 15, fontWeight: FontWeight.w800, color: C.ink)),
          subtitle: Padding(
              padding: const EdgeInsets.only(top: 5),
              child: Text(subtitle, style: _body)),
          trailing: const Icon(Icons.chevron_right_rounded, color: C.muted),
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
            PageHeading('Every tap, in place.',
                subtitle: 'Organize your inspection by room and grid.',
                trailing: IconButton(
                    tooltip: 'Create project',
                    onPressed: () => _create(context),
                    icon: const Icon(Icons.add_rounded, color: C.blue))),
            _gap,
            if (store.projects.isEmpty)
              _Empty('Start with a space.',
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
                          const SizedBox(width: 14),
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
              'Select a cell to review its latest recording or make a new capture. Blue means recorded, not sound or defective.',
              style: _body),
          _small,
          LayoutBuilder(builder: (_, constraints) {
            return SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: SizedBox(
                width: math.max(constraints.maxWidth, p.columns * 54.0),
                child: GridView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: p.columns,
                      crossAxisSpacing: 7,
                      mainAxisSpacing: 7),
                  itemCount: p.rows * p.columns,
                  itemBuilder: (_, i) => Semantics(
                    label:
                        '${_cellName(i, p.columns)}, ${latest.containsKey(i) ? 'recorded' : 'not recorded'}',
                    selected: selected == i,
                    child: Material(
                      color: selected == i
                          ? C.blue
                          : latest.containsKey(i)
                              ? C.pale
                              : Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      child: InkWell(
                        borderRadius: BorderRadius.circular(12),
                        onTap: () => setState(() => selected = i),
                        child: Center(
                            child: Text(
                                '${i ~/ p.columns + 1}:${i % p.columns + 1}',
                                style: TextStyle(
                                    fontFamily: 'monospace',
                                    color: selected == i ? Colors.white : C.ink,
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold))),
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
                        ? 'No recording at this point.'
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
          const PageHeading('Keep the evidence.',
              subtitle: 'Your saved recordings, newest first.'),
          _gap,
          TextField(
              decoration: const InputDecoration(
                  prefixIcon: Icon(Icons.search_rounded),
                  hintText: 'Search recordings',
                  border: OutlineInputBorder()),
              onChanged: (v) => setState(() => query = v)),
          _small,
          Wrap(spacing: 8, children: [
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
            const _Empty('No recordings here.',
                'Save a capture from the field scope, or adjust your filters.')
          else
            for (final m in entries) ...[
              _MeasurementTile(
                  m: m,
                  onTap: () => _push(context,
                      MeasurementScreen(store: widget.store, measurement: m)),
                  trailing: IconButton(
                      tooltip: 'Delete recording',
                      icon: const Icon(Icons.delete_outline_rounded),
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
      borderRadius: BorderRadius.circular(18),
      child: ListTile(
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(18),
              side: const BorderSide(color: C.line)),
          leading: Icon(
              _synthetic(m) ? Icons.science_outlined : Icons.graphic_eq_rounded,
              color: C.blue),
          title: Text(m.label,
              style:
                  const TextStyle(fontSize: 14, fontWeight: FontWeight.w800)),
          subtitle: Text(
              '${_moduleName(m.module)} · ${_number(m.peakHz, 0)} Hz\n${_date(m.created)}${_synthetic(m) ? ' · SYNTHETIC' : ''}',
              style:
                  const TextStyle(color: C.muted, height: 1.5, fontSize: 11)),
          isThreeLine: true,
          trailing: trailing ?? const Icon(Icons.chevron_right_rounded),
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
          const PageHeading('Make comparison useful.',
              subtitle: 'A repeatable reference, not a safety threshold.'),
          _gap,
          const Panel(
              color: C.pale,
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Label('Know the limits', color: C.blue),
                    _small,
                    Text(
                        'This is a reference selection, not laboratory microphone calibration. Levels are relative digital dBFS, not calibrated dB SPL. Keep phone, distance, tap tool and surroundings consistent. No reference can establish structural safety or a defect classification.',
                        style: _body)
                  ])),
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
          const Text(
              'Only microphone recordings with a finite frequency peak, a saved trace and no signal-quality flags can be selected. Synthetic samples are excluded.',
              style: _body),
          _gap,
          if (candidates.isEmpty)
            const _Empty('No eligible reference yet.',
                'Record and save a clean, repeatable tap. Review the signal-quality notes before selecting it as a reference.')
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
                          color: C.blue),
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
              subtitle: '${_moduleName(m.module)} · ${_date(m.created)}'),
          _gap,
          _Evidence(m: m, reference: store.reference),
          _gap,
          const SectionTitle('Waveform'),
          _small,
          _SignalPlot(
              values: m.waveform,
              label: 'CAPTURE TIME → · NORMALIZED PCM AMPLITUDE',
              height: 190),
          _small,
          const Text(
              'Trace shape is scaled for visibility. Use RMS dBFS for recorded digital level; the display is not calibrated sound pressure.',
              style: _body),
          _gap,
          const SectionTitle('Frequency spectrum'),
          _small,
          _SignalPlot(
              values: m.spectrum,
              label: '20 HZ → 24 KHZ · LOG FREQUENCY',
              spectrum: true,
              height: 190),
          _small,
          const Text(
              '256 logarithmic bins from 20 Hz to 24 kHz. The vertical display spans −100 to 0 dBFS. Spectrum is derived from captured PCM; a peak is not a material prediction.',
              style: _body),
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
              'Inspection record only. THUD does not determine defects, hollowness, fill percentage or fitness for use. Seek qualified assessment for safety-critical decisions.',
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
      final document = pw.Document();
      final text = widget.store.reportText(widget.project);
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
                ...text.split('\n').map((line) => pw.Padding(
                    padding: const pw.EdgeInsets.only(bottom: 5),
                    child: pw.Text(line,
                        style:
                            const pw.TextStyle(fontSize: 10, lineSpacing: 3)))),
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
          const SectionTitle('Latest evidence by cell'),
          _small,
          for (var i = 0; i < p.rows * p.columns; i++)
            Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Panel(
                    padding: const EdgeInsets.all(14),
                    child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          SizedBox(
                              width: 80,
                              child: Text(_cellName(i, p.columns),
                                  style: const TextStyle(
                                      fontFamily: 'monospace',
                                      color: C.ink,
                                      fontSize: 11,
                                      fontWeight: FontWeight.bold))),
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
          const Panel(
              color: C.pale,
              child: Text(
                  'Disclosure: This report documents measured airborne sound and user notes. It is not a structural inspection, material classifier or safety certificate. Synthetic recordings are explicitly identified and excluded from microphone coverage. Relative digital levels are not calibrated dB SPL.',
                  style: _body)),
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
      builder: (_, __) => _RoutePage(title: 'Settings & privacy', children: [
            const PageHeading('Quietly in your control.',
                subtitle: 'Field feedback and a little clarity.'),
            _gap,
            Panel(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                child: Column(children: [
                  SwitchListTile.adaptive(
                      title: const Text('Haptic feedback'),
                      subtitle:
                          const Text('Tactile feedback for capture controls'),
                      value: store.haptics,
                      onChanged: (v) => _perform(
                          context, () => store.setFeedback(haptics: v))),
                  const Divider(height: 1),
                  SwitchListTile.adaptive(
                      title: const Text('Interface sounds'),
                      subtitle:
                          const Text('A system click for capture controls'),
                      value: store.sounds,
                      onChanged: (v) =>
                          _perform(context, () => store.setFeedback(sounds: v)))
                ])),
            _gap,
            const SectionTitle('Privacy by design'),
            _small,
            const Panel(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Icon(Icons.phonelink_lock_outlined, color: C.blue, size: 28),
                  _small,
                  Text('Your recordings stay local.', style: _title),
                  _small,
                  Text(
                      'Microphone access is used only when you start a recording. Saved traces, measured features, projects and preferences are stored on this device. THUD does not upload recordings or send them to an AI service. Exporting a PDF opens your device share sheet; you decide where it goes. Delete individual recordings in Measurement history. Removing app data removes local records.',
                      style: _body)
                ])),
            _gap,
            const SectionTitle('A better field recording'),
            _small,
            const Panel(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text('01  Keep the setup consistent', style: _title),
                  _small,
                  Text(
                      'Use the same phone, tap tool, distance and orientation. Do not strike fragile, energized or unsafe surfaces.',
                      style: _body),
                  _gap,
                  Text('02  Give the sound room', style: _title),
                  _small,
                  Text(
                      'Start in quiet surroundings. Record a brief baseline, tap once and allow the sound to decay. Review flags and repeat if the trace clips or contains interference.',
                      style: _body),
                  _gap,
                  Text('03  Record the context', style: _title),
                  _small,
                  Text(
                      'Attach recordings to a project cell and add notes. A different sound can have many causes. No acoustic feature here establishes a defect or structural safety.',
                      style: _body)
                ])),
            _gap,
            Panel(
                color: C.pale,
                child: Text(
                    MediaQuery.of(context).disableAnimations
                        ? 'Reduced motion is enabled by your device. These screens use static plots and no looping decorative animation.'
                        : 'These screens use static plots with no looping decorative animation. System reduced-motion preferences are respected.',
                    style: _body))
          ]));
}
