import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';
import 'design.dart';
import 'screens.dart';
import 'store.dart';
import 'sensing_flow.dart';

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
        title: 'THUD',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
            useMaterial3: true,
            brightness: Brightness.light,
            fontFamily: 'Nunito',
            scaffoldBackgroundColor: C.bg,
            colorScheme: ColorScheme.fromSeed(seedColor: C.blue, surface: C.bg),
            appBarTheme: const AppBarTheme(
                backgroundColor: C.bg,
                foregroundColor: C.ink,
                elevation: 0,
                scrolledUnderElevation: 0),
            textTheme: const TextTheme(
                bodyMedium:
                    TextStyle(color: C.ink, fontSize: 14, height: 1.45)),
            iconTheme: const IconThemeData(size: 20, color: C.muted),
            dividerColor: C.line),
        home: FutureBuilder<void>(
            future: ready,
            builder: (context, state) {
              if (state.hasError) {
                return Scaffold(
                    body: Center(
                        child: Padding(
                            padding: const EdgeInsets.all(24),
                            child:
                                Text('Storage unavailable: ${state.error}'))));
              }
              if (state.connectionState != ConnectionState.done) {
                return const Scaffold(
                    body: Center(
                        child: Text('thud.',
                            style: TextStyle(
                                fontSize: 28, fontWeight: FontWeight.w600))));
              }
              return ThudHome(store: store);
            }),
      );
}

class ThudHome extends StatefulWidget {
  const ThudHome({required this.store, super.key});
  final ThudStore store;
  @override
  State<ThudHome> createState() => _ThudHomeState();
}

class _ThudHomeState extends State<ThudHome> {
  int tab = 1;
  Offset sceneOffset = Offset.zero;
  Module module = Module.slab;
  SensingIntent intent = SensingIntent.slab;
  String? project;
  int? cell;
  void measure(Module value, String? id, int? point) {
    setState(() {
      module = value;
      project = id;
      intent = switch (value) {
        Module.slab => SensingIntent.slab,
        Module.vessel => SensingIntent.vessel,
        Module.scope => SensingIntent.scope,
      };
      cell = point;
      sceneOffset = const Offset(0, .08);
      tab = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    final color = tab == 0 ? intent.color : C.blue;
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light
          .copyWith(statusBarColor: color, systemNavigationBarColor: color),
      child: AnimatedContainer(
        duration: MediaQuery.disableAnimationsOf(context)
            ? Duration.zero
            : const Duration(milliseconds: 420),
        curve: Curves.easeOutCubic,
        color: color,
        child: Scaffold(
          backgroundColor: Colors.transparent,
          body: SafeArea(
              child: Column(children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(28, 16, 20, 12),
              child: Row(children: [
                const Text('thud.',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 32,
                        fontWeight: FontWeight.w600,
                        letterSpacing: -1.2)),
                const Spacer(),
                utility(
                    Icons.history_rounded,
                    'History',
                    () => Navigator.of(context).push(MaterialPageRoute<void>(
                        builder: (_) => HistoryScreen(store: widget.store)))),
                const SizedBox(width: 8),
                utility(
                    Icons.tune_rounded,
                    'Settings',
                    () => Navigator.of(context).push(MaterialPageRoute<void>(
                        builder: (_) => SettingsScreen(store: widget.store)))),
              ]),
            ),
            Expanded(
                child: AnimatedSwitcher(
              duration: MediaQuery.disableAnimationsOf(context)
                  ? Duration.zero
                  : const Duration(milliseconds: 520),
              reverseDuration: MediaQuery.disableAnimationsOf(context)
                  ? Duration.zero
                  : const Duration(milliseconds: 280),
              switchInCurve: Curves.easeOutCubic,
              switchOutCurve: Curves.easeInCubic,
              layoutBuilder: (currentChild, previousChildren) => Stack(
                fit: StackFit.expand,
                alignment: Alignment.topCenter,
                children: <Widget>[
                  ...previousChildren.map((child) =>
                      IgnorePointer(child: ExcludeSemantics(child: child))),
                  if (currentChild != null) currentChild
                ],
              ),
              transitionBuilder: (child, animation) {
                final reduce = MediaQuery.disableAnimationsOf(context);
                final offset = reduce ? Offset.zero : sceneOffset;
                return FadeTransition(
                  opacity: animation,
                  child: SlideTransition(
                    position: Tween<Offset>(begin: offset, end: Offset.zero)
                        .animate(CurvedAnimation(
                            parent: animation, curve: Curves.easeOutCubic)),
                    child: ScaleTransition(
                      scale: Tween<double>(begin: reduce ? 1 : .985, end: 1)
                          .animate(CurvedAnimation(
                              parent: animation, curve: Curves.easeOutCubic)),
                      child: child,
                    ),
                  ),
                );
              },
              child: KeyedSubtree(
                key: ValueKey(tab),
                child: ClipRRect(
                  borderRadius: tab == 0
                      ? BorderRadius.zero
                      : const BorderRadius.vertical(top: Radius.circular(32)),
                  child: ColoredBox(
                    color: tab == 0 ? Colors.transparent : C.bg,
                    child: SizedBox.expand(
                      child: switch (tab) {
                        0 => ScopeScreen(
                            key: ValueKey('$project-$cell'),
                            store: widget.store,
                            initialModule: module,
                            initialIntent: intent,
                            projectId: project,
                            cell: cell,
                            onHandover: () => setState(() => tab = 1),
                            onModule: (m) {
                              module = m;
                            },
                            onIntent: (value) =>
                                setState(() => intent = value)),
                        1 => ProjectsScreen(
                            store: widget.store, onMeasure: measure),
                        _ => LibraryScreen(
                            store: widget.store, onMeasure: measure),
                      },
                    ),
                  ),
                ),
              ),
            )),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 12, 24, 16),
              child: DecoratedBox(
                decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: .12),
                    borderRadius: BorderRadius.circular(30),
                    border:
                        Border.all(color: Colors.white.withValues(alpha: .12))),
                child: SizedBox(
                  height: 68,
                  child: Stack(fit: StackFit.expand, children: [
                    AnimatedAlign(
                      alignment: tab == 0
                          ? Alignment.centerLeft
                          : tab == 1
                              ? Alignment.center
                              : Alignment.centerRight,
                      duration: MediaQuery.disableAnimationsOf(context)
                          ? Duration.zero
                          : const Duration(milliseconds: 420),
                      curve: Curves.easeOutCubic,
                      child: FractionallySizedBox(
                        widthFactor: 1 / 3,
                        heightFactor: 1,
                        child: Padding(
                          padding: const EdgeInsets.all(6),
                          child: DecoratedBox(
                            decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(25)),
                          ),
                        ),
                      ),
                    ),
                    Row(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          nav(0, Icons.graphic_eq_rounded, 'Listen'),
                          nav(1, Icons.grid_view_rounded, 'Handover'),
                          nav(2, Icons.layers_rounded, 'Tools'),
                        ]),
                  ]),
                ),
              ),
            ),
          ])),
        ),
      ),
    );
  }

  Widget utility(IconData icon, String label, VoidCallback onPressed) =>
      IconButton(
        tooltip: label,
        onPressed: onPressed,
        style: IconButton.styleFrom(
            backgroundColor: Colors.white.withValues(alpha: .12),
            foregroundColor: Colors.white,
            fixedSize: const Size(44, 44)),
        icon: Icon(icon, size: 20),
      );

  Widget nav(int index, IconData icon, String title) => Expanded(
      child: Semantics(
          selected: tab == index,
          child: Padding(
            padding: const EdgeInsets.all(6),
            child: TextButton(
              onPressed: () {
                if (tab == index) return;
                if (widget.store.haptics) HapticFeedback.selectionClick();
                setState(() {
                  sceneOffset = index > tab
                      ? const Offset(.08, 0)
                      : const Offset(-.08, 0);
                  tab = index;
                });
              },
              style: TextButton.styleFrom(
                foregroundColor: tab == index
                    ? (tab == 0 ? intent.color : C.blue)
                    : Colors.white.withValues(alpha: .75),
                backgroundColor: Colors.transparent,
                minimumSize: const Size(48, 48),
                padding: EdgeInsets.zero,
                shape: const StadiumBorder(),
              ),
              child: Column(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(icon, size: 22),
                    const SizedBox(height: 2),
                    Text(title,
                        style: const TextStyle(
                            fontSize: 12, fontWeight: FontWeight.w600)),
                  ]),
            ),
          )));
}
