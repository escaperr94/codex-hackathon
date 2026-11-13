import 'dart:math' as math;
import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';
import 'design.dart';
import 'screens.dart';
import 'store.dart';

void main() => runApp(const ThudApp());

class ThudApp extends StatefulWidget {
  const ThudApp({super.key});
  @override
  State<ThudApp> createState() => _ThudAppState();
}

class _ThudAppState extends State<ThudApp> {
  final store = ThudStore();
  late final Future<void> ready;
  @override
  void initState() {
    super.initState();
    ready = store.load();
  }

  @override
  void dispose() {
    store.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'THUD · Acoustic fieldwork',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          useMaterial3: true,
          brightness: Brightness.light,
          scaffoldBackgroundColor: C.bg,
          colorScheme: ColorScheme.fromSeed(seedColor: C.blue, surface: C.bg),
          textTheme:
              const TextTheme(bodyMedium: TextStyle(color: C.ink, height: 1.4)),
          appBarTheme: const AppBarTheme(
              backgroundColor: C.bg,
              foregroundColor: C.ink,
              elevation: 0,
              scrolledUnderElevation: 0),
          dividerColor: C.line,
        ),
        home: FutureBuilder<void>(
            future: ready,
            builder: (context, snapshot) {
              if (snapshot.hasError) {
                return Scaffold(
                    body: Center(
                        child: Padding(
                            padding: const EdgeInsets.all(24),
                            child: Text(
                                'Local storage could not be opened. ${snapshot.error}'))));
              }
              if (snapshot.connectionState != ConnectionState.done) {
                return const Scaffold(
                    body: Center(
                        child: Text('THUD',
                            style: TextStyle(
                                fontSize: 32,
                                fontWeight: FontWeight.w900,
                                letterSpacing: -2))));
              }
              return ThudHome(store: store);
            }),
      );
}

enum HomeTab { home, inspect, projects, library }

class ThudHome extends StatefulWidget {
  const ThudHome({required this.store, super.key});
  final ThudStore store;
  @override
  State<ThudHome> createState() => _ThudHomeState();
}

class _ThudHomeState extends State<ThudHome> {
  HomeTab tab = HomeTab.home;
  Module module = Module.slab;
  String? activeProject;
  int? activeCell;
  void feedback() {
    if (widget.store.haptics) HapticFeedback.selectionClick();
  }

  void select(HomeTab value) {
    feedback();
    setState(() => tab = value);
  }

  void measure(Module value, String? project, int? cell) {
    feedback();
    setState(() {
      module = value;
      activeProject = project;
      activeCell = cell;
      tab = HomeTab.inspect;
    });
  }

  void open(Widget page) =>
      Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => page));

  @override
  Widget build(BuildContext context) {
    final wide = MediaQuery.sizeOf(context).width > 720;
    return Scaffold(
        body: SafeArea(
            child: Center(
                child: ConstrainedBox(
      constraints: const BoxConstraints(maxWidth: 620),
      child: DecoratedBox(
        decoration: BoxDecoration(
            color: C.bg,
            border: wide
                ? Border.symmetric(vertical: BorderSide(color: C.line))
                : null),
        child: Column(children: [
          Padding(
              padding: const EdgeInsets.fromLTRB(24, 18, 18, 8),
              child: Row(children: [
                const Text('thud',
                    style: TextStyle(
                        fontSize: 30,
                        fontWeight: FontWeight.w900,
                        letterSpacing: -2,
                        height: 1)),
                const Padding(
                    padding: EdgeInsets.only(left: 8, top: 9),
                    child: Text('®',
                        style: TextStyle(color: C.blue, fontSize: 12))),
                const Spacer(),
                const Label('On your device', fontSize: 8),
                const SizedBox(width: 8),
                const StatusDot(color: C.blue, size: 5),
                IconButton(
                    tooltip: 'Settings',
                    onPressed: () => open(SettingsScreen(store: widget.store)),
                    icon: const Icon(Icons.tune_rounded, size: 20)),
              ])),
          Expanded(
              child: AnimatedSwitcher(
                  duration: MediaQuery.disableAnimationsOf(context)
                      ? Duration.zero
                      : const Duration(milliseconds: 250),
                  child: page())),
          _Navigation(selected: tab, onSelected: select),
        ]),
      ),
    ))));
  }

  Widget page() => switch (tab) {
        HomeTab.home => HomePage(
            key: const ValueKey('home'),
            store: widget.store,
            onMeasure: measure,
            onProjects: () => select(HomeTab.projects),
            onHistory: () => open(HistoryScreen(store: widget.store)),
            onCalibration: () => open(CalibrationScreen(
                  store: widget.store,
                  onRecord: () {
                    Navigator.of(context).pop();
                    measure(Module.slab, null, null);
                  },
                ))),
        HomeTab.inspect => ScopeScreen(
            key: ValueKey('scope-$module-$activeProject-$activeCell'),
            store: widget.store,
            initialModule: module,
            projectId: activeProject,
            cell: activeCell,
            onModule: (value) {
              module = value;
            }),
        HomeTab.projects => ProjectsScreen(
            key: const ValueKey('projects'),
            store: widget.store,
            onMeasure: measure),
        HomeTab.library => LibraryScreen(
            key: const ValueKey('library'),
            store: widget.store,
            onMeasure: measure),
      };
}

class _Navigation extends StatelessWidget {
  const _Navigation({required this.selected, required this.onSelected});
  final HomeTab selected;
  final ValueChanged<HomeTab> onSelected;
  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.fromLTRB(18, 8, 18, 12),
        padding: const EdgeInsets.all(5),
        decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(25),
            border: Border.all(color: C.line),
            boxShadow: const [
              BoxShadow(
                  color: Color(0x08607AAA),
                  blurRadius: 20,
                  offset: Offset(0, 6))
            ]),
        child: Row(children: [
          item(HomeTab.home, Icons.home_outlined, 'Discover'),
          item(HomeTab.inspect, Icons.graphic_eq_rounded, 'Measure'),
          item(HomeTab.projects, Icons.grid_view_outlined, 'Projects'),
          item(HomeTab.library, Icons.layers_outlined, 'Library'),
        ]),
      );
  Widget item(HomeTab value, IconData icon, String label) => Expanded(
      child: Semantics(
          selected: value == selected,
          child: TextButton(
            onPressed: () => onSelected(value),
            style: TextButton.styleFrom(
                foregroundColor: value == selected ? C.blue : C.muted,
                backgroundColor:
                    value == selected ? C.pale : Colors.transparent,
                minimumSize: const Size(48, 56),
                padding: const EdgeInsets.symmetric(vertical: 8),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20))),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              Icon(icon, size: 22),
              const SizedBox(height: 4),
              Text(label,
                  style: const TextStyle(
                      fontSize: 10, fontWeight: FontWeight.w700))
            ]),
          )));
}

class HomePage extends StatelessWidget {
  const HomePage(
      {required this.store,
      required this.onMeasure,
      required this.onProjects,
      required this.onHistory,
      required this.onCalibration,
      super.key});
  final ThudStore store;
  final void Function(Module, String?, int?) onMeasure;
  final VoidCallback onProjects, onHistory, onCalibration;
  @override
  Widget build(BuildContext context) => AnimatedBuilder(
      animation: store,
      builder: (context, _) => AppPage(children: [
            const SizedBox(height: 12),
            const Row(children: [
              Label('Acoustic field notes', fontSize: 9),
              Spacer(),
              Label('Tap. Listen. Know.', fontSize: 8)
            ]),
            const SizedBox(height: 20),
            const Text('There’s more\nbeneath the surface.',
                style: TextStyle(
                    fontSize: 37,
                    height: 1.05,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -1.9,
                    color: C.ink)),
            const SizedBox(height: 12),
            const Text('A little sound. A closer look.',
                style: TextStyle(color: C.muted, fontSize: 15)),
            const SizedBox(height: 12),
            _AcousticObject(onTap: () => onMeasure(Module.scope, null, null)),
            const SizedBox(height: 18),
            const SectionTitle('Choose your instrument'),
            const SizedBox(height: 14),
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Expanded(
                  child: _Instrument(
                      module: Module.slab,
                      title: 'Slab',
                      subtitle: 'Explore a surface',
                      icon: Icons.grid_4x4_rounded,
                      onTap: () => onMeasure(Module.slab, null, null))),
              const SizedBox(width: 12),
              Expanded(
                  child: _Instrument(
                      module: Module.vessel,
                      title: 'Vessel',
                      subtitle: 'Listen to a container',
                      icon: Icons.water_drop_outlined,
                      onTap: () => onMeasure(Module.vessel, null, null))),
            ]),
            const SizedBox(height: 25),
            InkWell(
                onTap: onCalibration,
                borderRadius: BorderRadius.circular(18),
                child: Padding(
                    padding:
                        const EdgeInsets.symmetric(vertical: 14, horizontal: 2),
                    child: Row(children: [
                      const Icon(Icons.center_focus_strong_outlined,
                          color: C.blue, size: 22),
                      const SizedBox(width: 14),
                      Expanded(
                          child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                            Text(
                                store.reference == null
                                    ? 'Start with a reference.'
                                    : 'Your reference is ready.',
                                style: const TextStyle(
                                    fontSize: 14, fontWeight: FontWeight.w700)),
                            const SizedBox(height: 4),
                            const Text(
                                'Same room. Same tool. Better comparison.',
                                style: TextStyle(fontSize: 11, color: C.muted)),
                          ])),
                      const Icon(Icons.arrow_forward_rounded,
                          size: 18, color: C.muted),
                    ]))),
            const SizedBox(height: 18),
            SectionTitle('Your fieldwork',
                trailing: TextButton(
                    onPressed: onProjects,
                    child: const Text('Projects',
                        style: TextStyle(fontSize: 12)))),
            const SizedBox(height: 8),
            Panel(
                child: Row(children: [
              const SizedBox(
                  width: 60,
                  height: 62,
                  child: CustomPaint(painter: _GridSketch())),
              const SizedBox(width: 20),
              Expanded(
                  child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                    Text(
                        store.projects.isEmpty
                            ? 'A place for every tap.'
                            : store.projects.first.name,
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w700)),
                    const SizedBox(height: 6),
                    Text(
                        store.projects.isEmpty
                            ? 'Create a room. Map the evidence.'
                            : '${store.projects.first.room} · ${store.forProject(store.projects.first.id).length} recordings',
                        style: const TextStyle(color: C.muted, fontSize: 12)),
                    TextButton(
                        onPressed: onProjects,
                        style: TextButton.styleFrom(
                            padding: EdgeInsets.zero,
                            alignment: Alignment.centerLeft),
                        child: Text(store.projects.isEmpty
                            ? 'Create a project →'
                            : 'Open project →')),
                  ])),
            ])),
            const SizedBox(height: 24),
            SectionTitle('Recent captures',
                trailing: TextButton(
                    onPressed: onHistory,
                    child:
                        const Text('See all', style: TextStyle(fontSize: 12)))),
            if (store.measurements.isEmpty)
              const Padding(
                  padding: EdgeInsets.symmetric(vertical: 22),
                  child: Text(
                      'Nothing recorded yet. Your first tap starts the story.',
                      style: TextStyle(color: C.muted, fontSize: 12)))
            else
              ...store.measurements.take(3).map((m) => ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: const Icon(Icons.graphic_eq_rounded, color: C.blue),
                  title: Text(m.label,
                      style: const TextStyle(
                          fontSize: 14, fontWeight: FontWeight.w600)),
                  subtitle:
                      Text(m.source, style: const TextStyle(fontSize: 10)),
                  trailing: Text('${m.peakHz.round()} Hz',
                      style: const TextStyle(
                          fontFamily: 'monospace', fontSize: 12)),
                  onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                          builder: (_) => MeasurementScreen(
                              store: store, measurement: m))))),
            const SizedBox(height: 18),
          ]));
}

class _AcousticObject extends StatefulWidget {
  const _AcousticObject({required this.onTap});
  final VoidCallback onTap;
  @override
  State<_AcousticObject> createState() => _AcousticObjectState();
}

class _AcousticObjectState extends State<_AcousticObject>
    with SingleTickerProviderStateMixin {
  late final AnimationController motion;
  @override
  void initState() {
    super.initState();
    motion =
        AnimationController(vsync: this, duration: const Duration(seconds: 9));
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (MediaQuery.disableAnimationsOf(context)) {
      motion.stop();
    } else {
      motion.repeat();
    }
  }

  @override
  void dispose() {
    motion.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => SizedBox(
      height: 244,
      width: double.infinity,
      child: Stack(alignment: Alignment.center, children: [
        Positioned.fill(
            child: RepaintBoundary(
                child: AnimatedBuilder(
                    animation: motion,
                    builder: (context, _) => CustomPaint(
                        painter: _ResonancePainter(motion.value))))),
        Positioned(
            top: 12,
            left: 0,
            child: Text('     +\n. : : : .\n: : + : :\n  . : .',
                style: TextStyle(
                    color: C.muted.withValues(alpha: .45),
                    fontFamily: 'monospace',
                    fontSize: 9,
                    height: 1.4))),
        Positioned(
            bottom: 24,
            right: 0,
            child: Text('ACOUSTIC\nEXPLORER',
                textAlign: TextAlign.right,
                style: TextStyle(
                    color: C.muted.withValues(alpha: .65),
                    fontFamily: 'monospace',
                    letterSpacing: 1.2,
                    fontSize: 8))),
        Semantics(
            button: true,
            label: 'Open acoustic scope',
            child: SizedBox(
                width: 108,
                height: 108,
                child: Material(
                    color: Colors.transparent,
                    shape: const CircleBorder(),
                    child: InkWell(
                        onTap: widget.onTap,
                        customBorder: const CircleBorder(),
                        child: const Icon(Icons.graphic_eq_rounded,
                            color: Colors.white, size: 36))))),
        Positioned(
            bottom: 0,
            child: TextButton(
                onPressed: widget.onTap,
                child: const Row(mainAxisSize: MainAxisSize.min, children: [
                  Text('Open live scope',
                      style:
                          TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
                  SizedBox(width: 8),
                  Icon(Icons.north_east, size: 14)
                ]))),
      ]));
}

class _ResonancePainter extends CustomPainter {
  const _ResonancePainter(this.phase);
  final double phase;
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * .44);
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = .65
      ..color = C.blue.withValues(alpha: .12);
    for (var i = 0; i < 6; i++) {
      final r = 60 + i * 14.0 + math.sin(phase * math.pi * 2) * 2;
      canvas.drawOval(
          Rect.fromCenter(center: center, width: r * 2.25, height: r * 1.48),
          ring);
    }
    const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
    final dot = Paint()..color = C.blue.withValues(alpha: .2);
    for (var y = 0; y < size.height; y += 3) {
      for (var x = 0; x < size.width; x += 3) {
        final dx = (x - center.dx) / 145, dy = (y - center.dy) / 85;
        final d = math.sqrt(dx * dx + dy * dy);
        final field = (math.sin(d * 17 - phase * math.pi * 2) * .5 + .5) *
            math.max(0, 1 - d) *
            .5;
        if (field > (bayer[((y ~/ 3) % 4) * 4 + (x ~/ 3) % 4] + .5) / 16) {
          canvas.drawRect(
              Rect.fromLTWH(x.toDouble(), y.toDouble(), 1.2, 1.2), dot);
        }
      }
    }
    canvas.drawCircle(
        center + const Offset(0, 9),
        57,
        Paint()
          ..color = C.blue.withValues(alpha: .14)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 15));
    canvas.drawCircle(
        center,
        55,
        Paint()
          ..shader = const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [Color(0xFF5A81F4), C.blue, Color(0xFF1836A5)])
              .createShader(Rect.fromCircle(center: center, radius: 55)));
    canvas.drawCircle(
        center,
        47,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = .8
          ..color = Colors.white.withValues(alpha: .25));
  }

  @override
  bool shouldRepaint(covariant _ResonancePainter old) => old.phase != phase;
}

class _Instrument extends StatelessWidget {
  const _Instrument(
      {required this.module,
      required this.title,
      required this.subtitle,
      required this.icon,
      required this.onTap});
  final Module module;
  final String title, subtitle;
  final IconData icon;
  final VoidCallback onTap;
  @override
  Widget build(BuildContext context) => Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(22),
      child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(22),
          child: Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(22),
                  border: Border.all(color: C.line)),
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Icon(icon, color: C.blue, size: 29),
                      const Spacer(),
                      const Icon(Icons.north_east_rounded,
                          size: 16, color: C.muted)
                    ]),
                    const SizedBox(height: 25),
                    Text(title,
                        style: const TextStyle(
                            fontSize: 23,
                            fontWeight: FontWeight.w700,
                            letterSpacing: -.7)),
                    const SizedBox(height: 4),
                    Text(subtitle,
                        style: const TextStyle(fontSize: 10, color: C.muted)),
                  ]))));
}

class _GridSketch extends CustomPainter {
  const _GridSketch();
  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = C.line
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    for (var y = 0; y < 3; y++) {
      for (var x = 0; x < 3; x++) {
        final r = RRect.fromRectAndRadius(
            Rect.fromLTWH(x * 20.0, y * 20.0, 15, 15),
            const Radius.circular(3));
        canvas.drawRRect(r, p);
      }
    }
    canvas.drawRRect(
        RRect.fromRectAndRadius(
            const Rect.fromLTWH(20, 20, 15, 15), const Radius.circular(3)),
        Paint()..color = C.blue);
  }

  @override
  bool shouldRepaint(covariant _GridSketch old) => false;
}
