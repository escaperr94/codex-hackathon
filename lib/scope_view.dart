import 'dart:math' as math;

import 'package:material_ui/material_ui.dart';

import 'design.dart';
import 'store.dart';

/// Views of the stored analysis, never a reconstruction of the original PCM.
class SignalExplorer extends StatefulWidget {
  const SignalExplorer({required this.measurement, super.key});

  final Measurement measurement;

  @override
  State<SignalExplorer> createState() => _SignalExplorerState();
}

enum _SignalView { wave, spectrum, decay }

class _SignalExplorerState extends State<SignalExplorer> {
  _SignalView _view = _SignalView.wave;

  @override
  Widget build(BuildContext context) {
    final measurement = widget.measurement;
    final spectrum = _view == _SignalView.spectrum;
    final values = spectrum ? measurement.spectrum : measurement.waveform;
    final problem = _dataProblem(values, spectrum: spectrum);
    final reducedMotion = MediaQuery.disableAnimationsOf(context) ||
        MediaQuery.accessibleNavigationOf(context);
    final accent = Theme.of(context).colorScheme.primary;

    return Panel(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: ExpansionTile(
        initiallyExpanded: false,
        tilePadding: EdgeInsets.zero,
        childrenPadding: const EdgeInsets.only(bottom: 16),
        title: const Text('Signal',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
        leading: Icon(Icons.graphic_eq_rounded, color: accent, size: 22),
        children: [
          Container(
            padding: const EdgeInsets.all(3),
            decoration: BoxDecoration(
                color: C.pale, borderRadius: BorderRadius.circular(14)),
            child: Row(children: [
              for (final view in _SignalView.values)
                Expanded(
                  child: Semantics(
                    selected: _view == view,
                    child: TextButton(
                      onPressed: () => setState(() => _view = view),
                      style: TextButton.styleFrom(
                        backgroundColor:
                            _view == view ? Colors.white : Colors.transparent,
                        foregroundColor: _view == view ? accent : C.muted,
                        minimumSize: const Size(0, 48),
                        padding: const EdgeInsets.symmetric(horizontal: 4),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(11)),
                        textStyle: const TextStyle(
                            fontSize: 12, fontWeight: FontWeight.w600),
                      ),
                      child: Text(
                          switch (view) {
                            _SignalView.wave => 'Wave',
                            _SignalView.spectrum => 'Spectrum',
                            _SignalView.decay => 'Ring',
                          },
                          maxLines: 1),
                    ),
                  ),
                ),
            ]),
          ),
          const SizedBox(height: 16),
          Align(
            alignment: AlignmentDirectional.centerStart,
            child: Text(
              spectrum
                  ? 'Level · dBFS'
                  : _view == _SignalView.wave
                      ? 'Amplitude · FS'
                      : 'Envelope · FS',
              style: const TextStyle(color: C.muted, fontSize: 12),
            ),
          ),
          const SizedBox(height: 8),
          if (problem != null)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 32),
              decoration: BoxDecoration(
                  color: C.bg, borderRadius: BorderRadius.circular(14)),
              child: Column(children: [
                const Icon(Icons.show_chart_rounded, color: C.muted, size: 28),
                const SizedBox(height: 12),
                Text(problem,
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: C.muted, fontSize: 14)),
              ]),
            )
          else
            Semantics(
              label: _chartDescription(measurement, _view),
              image: true,
              child: ExcludeSemantics(
                child: RepaintBoundary(
                  child: TweenAnimationBuilder<double>(
                    key: ValueKey((measurement, _view)),
                    tween: Tween(begin: 0, end: 1),
                    duration: reducedMotion
                        ? Duration.zero
                        : const Duration(milliseconds: 460),
                    curve: Curves.easeOutCubic,
                    builder: (context, reveal, _) => SizedBox(
                      height: 244 +
                          24 *
                              (MediaQuery.textScalerOf(context).scale(12) / 12 -
                                  1),
                      width: double.infinity,
                      child: CustomPaint(
                        painter: _SignalPainter(
                          values: values,
                          view: _view,
                          peakHz: measurement.peakHz,
                          reveal: reveal,
                          accent: accent,
                          textScaler: MediaQuery.textScalerOf(context),
                          textDirection: Directionality.of(context),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

String? _dataProblem(List<double> values, {required bool spectrum}) {
  if (values.isEmpty) {
    return spectrum ? 'No saved spectrum' : 'No saved waveform';
  }
  if (spectrum ? values.length != 256 : values.length > 256) {
    return 'Trace unavailable';
  }
  for (final value in values) {
    if (!value.isFinite || value < (spectrum ? 0 : -1) || value > 1) {
      return 'Trace unavailable';
    }
  }
  return null;
}

String _frequency(double hz) => hz >= 1000
    ? '${(hz / 1000).toStringAsFixed(2)} kHz'
    : '${hz.toStringAsFixed(1)} Hz';

String _chartDescription(Measurement measurement, _SignalView view) {
  if (view == _SignalView.spectrum) {
    final peak = measurement.peakHz;
    return 'Spectrum of 256 logarithmic frequency bins from 20 hertz to '
        '24 kilohertz, digital level from minus 100 to 0 dBFS. '
        '${peak.isFinite && peak > 0 ? "Measured peak ${_frequency(peak)}." : "Peak unavailable."}';
  }
  return '${view == _SignalView.wave ? "Signed waveform" : "Absolute peak envelope"}, '
      '${measurement.waveform.length} peak-preserving bins. '
      '${view == _SignalView.wave ? "Minus one to plus one" : "Zero to one"} '
      'full-scale amplitude. Horizontal axis is 0 to 100 percent of the capture.';
}

class _SignalPainter extends CustomPainter {
  const _SignalPainter({
    required this.values,
    required this.view,
    required this.peakHz,
    required this.reveal,
    required this.accent,
    required this.textScaler,
    required this.textDirection,
  });

  final List<double> values;
  final _SignalView view;
  final double peakHz;
  final double reveal;
  final Color accent;
  final TextScaler textScaler;
  final TextDirection textDirection;

  static final double _logRange = math.log(24000 / 20);

  @override
  void paint(Canvas canvas, Size size) {
    final labelHeight = textScaler.scale(12) * 1.4;
    final left = textScaler.scale(12) * 3.5;
    final plot = Rect.fromLTRB(left, labelHeight + 10, size.width - 8,
        size.height - labelHeight * 2 - 20);
    if (plot.width <= 0 || plot.height <= 0) return;
    final spectrum = view == _SignalView.spectrum;
    final signed = view == _SignalView.wave;
    final grid = Paint()
      ..color = C.line
      ..strokeWidth = 1;

    canvas.drawRRect(RRect.fromRectAndRadius(plot, const Radius.circular(8)),
        Paint()..color = C.bg);
    for (var i = 0; i <= 4; i++) {
      final y = plot.top + plot.height * i / 4;
      canvas.drawLine(Offset(plot.left, y), Offset(plot.right, y), grid);
      final label = spectrum
          ? '${-25 * i}'
          : signed
              ? const ['+1', '', '0', '', '−1'][i]
              : const ['1', '', '0.5', '', '0'][i];
      if (label.isNotEmpty) {
        _label(canvas, label, Offset(plot.left - 8, y),
            horizontal: 1, vertical: .5);
      }
    }

    if (spectrum) {
      // Keep labels legible at large text scales and on compact phones.
      final ticks = plot.width < textScaler.scale(12) * 22
          ? const <double>[20, 1000, 24000]
          : const <double>[20, 100, 1000, 10000, 24000];
      for (final hz in ticks) {
        final x = plot.left + plot.width * math.log(hz / 20) / _logRange;
        canvas.drawLine(Offset(x, plot.top), Offset(x, plot.bottom), grid);
        _label(canvas, hz >= 1000 ? '${(hz / 1000).round()}k' : '${hz.round()}',
            Offset(x, plot.bottom + 9),
            horizontal: hz == 20
                ? 0
                : hz == 24000
                    ? 1
                    : .5);
      }
    } else {
      for (var i = 0; i <= 4; i++) {
        final x = plot.left + plot.width * i / 4;
        canvas.drawLine(Offset(x, plot.top), Offset(x, plot.bottom), grid);
        if (plot.width >= textScaler.scale(12) * 20 || i.isEven) {
          _label(canvas, '${i * 25}%', Offset(x, plot.bottom + 9),
              horizontal: i == 0
                  ? 0
                  : i == 4
                      ? 1
                      : .5);
        }
      }
    }
    _label(
      canvas,
      spectrum ? 'Frequency · Hz' : 'Capture · %',
      Offset(plot.center.dx, size.height),
      horizontal: .5,
      vertical: 1,
    );

    canvas.save();
    canvas.clipRect(Rect.fromLTRB(plot.left, plot.top - 1,
        plot.left + plot.width * reveal, plot.bottom + 1));
    final trace = Path();
    var firstX = plot.left;
    var lastX = plot.right;
    for (var i = 0; i < values.length; i++) {
      // Time values summarize equal-width bins, so use bin centers. Spectrum
      // values include the endpoints of the stored logarithmic frequency axis.
      final fraction =
          spectrum ? i / (values.length - 1) : (i + .5) / values.length;
      final x = plot.left + plot.width * fraction;
      final value = values[i];
      final normalized = spectrum
          ? value
          : signed
              ? (value + 1) / 2
              : value.abs();
      final y = plot.bottom - plot.height * normalized;
      if (i == 0) {
        firstX = x;
        trace.moveTo(x, y);
      } else {
        trace.lineTo(x, y);
      }
      lastX = x;
      if (values.length == 1) {
        canvas.drawCircle(Offset(x, y), 2.5, Paint()..color = accent);
      }
    }
    if (!signed) {
      final fill = Path.from(trace)
        ..lineTo(lastX, plot.bottom)
        ..lineTo(firstX, plot.bottom)
        ..close();
      canvas.drawPath(fill, Paint()..color = accent.withValues(alpha: .07));
    }
    canvas.drawPath(
        trace,
        Paint()
          ..color = accent
          ..strokeWidth = 1.8
          ..strokeJoin = StrokeJoin.round
          ..style = PaintingStyle.stroke);
    canvas.restore();

    if (spectrum && peakHz.isFinite && peakHz >= 20 && peakHz <= 24000) {
      final fraction = math.log(peakHz / 20) / _logRange;
      if (fraction <= reveal) {
        final x = plot.left + plot.width * fraction;
        canvas.drawLine(
            Offset(x, plot.top),
            Offset(x, plot.bottom),
            Paint()
              ..color = accent.withValues(alpha: .5)
              ..strokeWidth = 1);
        _label(canvas, 'Peak ${_frequency(peakHz)}',
            Offset(x.clamp(plot.left, plot.right).toDouble(), plot.top - 8),
            horizontal: fraction < .3
                ? 0
                : fraction > .7
                    ? 1
                    : .5,
            vertical: 1,
            color: accent);
      }
    }
  }

  void _label(Canvas canvas, String value, Offset position,
      {double horizontal = 0, double vertical = 0, Color color = C.muted}) {
    final text = TextPainter(
      text: TextSpan(
        text: value,
        style: TextStyle(
            color: color,
            fontSize: 12,
            fontFamily: 'Nunito',
            fontFeatures: const [FontFeature.tabularFigures()]),
      ),
      textDirection: textDirection,
      textScaler: textScaler,
      maxLines: 1,
    )..layout();
    text.paint(canvas,
        position - Offset(text.width * horizontal, text.height * vertical));
    text.dispose();
  }

  @override
  bool shouldRepaint(covariant _SignalPainter oldDelegate) =>
      oldDelegate.values != values ||
      oldDelegate.view != view ||
      oldDelegate.peakHz != peakHz ||
      oldDelegate.reveal != reveal ||
      oldDelegate.accent != accent ||
      oldDelegate.textScaler != textScaler ||
      oldDelegate.textDirection != textDirection;
}
