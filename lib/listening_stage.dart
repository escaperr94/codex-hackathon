import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:material_ui/material_ui.dart';

/// Motion describes capture state. Idle settles; recording reacts to PCM level.
class ListeningStage extends StatefulWidget {
  const ListeningStage(
      {required this.live,
      required this.active,
      required this.busy,
      required this.complete,
      required this.onTap,
      this.subjectIcon,
      super.key});
  final ValueNotifier<List<double>> live;
  final bool active, busy, complete;
  final VoidCallback? onTap;
  final IconData? subjectIcon;
  @override
  State<ListeningStage> createState() => _ListeningStageState();
}

class _ListeningStageState extends State<ListeningStage>
    with SingleTickerProviderStateMixin {
  late final AnimationController motion;
  ui.FragmentShader? field;
  String? shaderError;
  Future<void> _loadField() async {
    try {
      final program =
          await ui.FragmentProgram.fromAsset('shaders/acoustic_field.frag');
      if (!mounted) return;
      setState(() => field = program.fragmentShader());
    } catch (error) {
      if (mounted) setState(() => shaderError = 'GPU field unavailable');
    }
  }

  @override
  void initState() {
    super.initState();
    motion = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1800));
    _loadField();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _sync();
  }

  @override
  void didUpdateWidget(covariant ListeningStage old) {
    super.didUpdateWidget(old);
    _sync();
  }

  void _sync() {
    if ((widget.active || widget.busy) &&
        !MediaQuery.disableAnimationsOf(context)) {
      if (!motion.isAnimating) motion.repeat();
    } else {
      motion.stop();
    }
  }

  @override
  void dispose() {
    motion.dispose();
    field?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => RepaintBoundary(
          child: AnimatedBuilder(
        animation: Listenable.merge([motion, widget.live]),
        builder: (context, _) {
          final values = widget.live.value;
          var level = 0.0;
          for (final v in values) {
            level = math.max(level, v.abs());
          }
          return LayoutBuilder(builder: (context, box) {
            final diameter = math.min(
                math.min(box.maxWidth * .64, box.maxHeight * .85), 224.0);
            return Stack(alignment: Alignment.center, children: [
              if (field != null)
                Positioned.fill(
                    child: IgnorePointer(
                        child: CustomPaint(
                            painter: _ShaderField(
                                field!, motion.value, level, widget.active)))),
              if (shaderError != null)
                Positioned(
                    bottom: 0,
                    child: Text(shaderError!,
                        style: const TextStyle(
                            color: Colors.white70, fontSize: 11))),
              Positioned.fill(
                  child: CustomPaint(
                      painter: _ResonanceField(
                          phase: motion.value,
                          active: widget.active,
                          busy: widget.busy,
                          level: level))),
              TweenAnimationBuilder<double>(
                tween: Tween(end: widget.active ? 1.04 + level * .04 : 1.0),
                duration: MediaQuery.disableAnimationsOf(context)
                    ? Duration.zero
                    : const Duration(milliseconds: 180),
                builder: (context, scale, child) =>
                    Transform.scale(scale: scale, child: child),
                child: SizedBox(
                    width: diameter,
                    height: diameter,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: Colors.white,
                          boxShadow: const [
                            BoxShadow(
                                color: Color(0x1820262D),
                                blurRadius: 24,
                                offset: Offset(0, 12))
                          ]),
                      child: Material(
                        color: Colors.transparent,
                        shape: const CircleBorder(),
                        clipBehavior: Clip.antiAlias,
                        child: InkWell(
                          customBorder: const CircleBorder(),
                          onTap: widget.onTap,
                          child: Semantics(
                            button: true,
                            label: widget.active
                                ? 'Stop listening'
                                : widget.complete
                                    ? 'Listen again'
                                    : 'Start listening',
                            child: Center(
                                child: AnimatedSwitcher(
                              duration: MediaQuery.disableAnimationsOf(context)
                                  ? Duration.zero
                                  : const Duration(milliseconds: 320),
                              transitionBuilder: (child, animation) =>
                                  FadeTransition(
                                      opacity: animation,
                                      child: ScaleTransition(
                                          scale: Tween<double>(
                                                  begin: .8, end: 1)
                                              .animate(CurvedAnimation(
                                                  parent: animation,
                                                  curve: Curves.easeOutCubic)),
                                          child: child)),
                              child: widget.busy
                                  ? SizedBox(
                                      key: const ValueKey('busy'),
                                      width: 30,
                                      height: 30,
                                      child: CircularProgressIndicator(
                                          color: Theme.of(context)
                                              .colorScheme
                                              .primary,
                                          strokeWidth: 2))
                                  : widget.complete
                                      ? Icon(Icons.check_rounded,
                                          key: const ValueKey('done'),
                                          color: Theme.of(context)
                                              .colorScheme
                                              .primary,
                                          size: 54)
                                      : widget.active
                                          ? Icon(Icons.stop_rounded,
                                              key: const ValueKey('stop'),
                                              color: Theme.of(context)
                                                  .colorScheme
                                                  .primary,
                                              size: 44)
                                          : widget.subjectIcon == null
                                              ? const _TapGlyph(
                                                  key: ValueKey('ready'))
                                              : Icon(widget.subjectIcon,
                                                  key: ValueKey(
                                                      widget.subjectIcon),
                                                  size: 64,
                                                  color: Theme.of(context)
                                                      .colorScheme
                                                      .primary),
                            )),
                          ),
                        ),
                      ),
                    )),
              ),
            ]);
          });
        },
      ));
}

class _TapGlyph extends StatelessWidget {
  const _TapGlyph({super.key});
  @override
  Widget build(BuildContext context) => Row(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [18.0, 34.0, 54.0, 34.0, 18.0]
          .map((height) => Container(
              width: 5,
              height: height,
              margin: const EdgeInsets.symmetric(horizontal: 4),
              decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primary,
                  borderRadius: BorderRadius.circular(3))))
          .toList());
}

class _ResonanceField extends CustomPainter {
  const _ResonanceField(
      {required this.phase,
      required this.active,
      required this.busy,
      required this.level});
  final double phase, level;
  final bool active, busy;
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final base = math.min(size.width, size.height) * .34;
    final line = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    for (var ring = 0; ring < 3; ring++) {
      final cycle = active || busy ? (phase + ring / 3) % 1 : ring / 3;
      final radius = base + cycle * base * .62;
      line.color =
          Colors.white.withValues(alpha: (active ? .38 : .12) * (1 - cycle));
      canvas.drawCircle(center, radius + (active ? level * 8 : 0), line);
    }
  }

  @override
  bool shouldRepaint(covariant _ResonanceField old) =>
      old.phase != phase ||
      old.active != active ||
      old.level != level ||
      old.busy != busy;
}

class _ShaderField extends CustomPainter {
  const _ShaderField(this.shader, this.phase, this.level, this.active);
  final ui.FragmentShader shader;
  final double phase, level;
  final bool active;
  @override
  void paint(Canvas canvas, Size size) {
    shader
      ..setFloat(0, size.width)
      ..setFloat(1, size.height)
      ..setFloat(2, phase)
      ..setFloat(3, active ? level : 0)
      ..setFloat(4, active ? 1 : 0);
    canvas.drawRect(Offset.zero & size, Paint()..shader = shader);
  }

  @override
  bool shouldRepaint(covariant _ShaderField old) =>
      old.phase != phase ||
      old.level != level ||
      old.active != active ||
      old.shader != shader;
}
