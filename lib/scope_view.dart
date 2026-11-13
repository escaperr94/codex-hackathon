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
    final synthetic = measurement.source.toLowerCase().contains('synthetic');
    final peak = measurement.peakHz;
    final hasPeak = peak.isFinite && peak > 0;
    final decay = measurement.decayMs;

    return Panel(
      padding: const EdgeInsets.all(16),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Expanded(
            child: Text('Signal explorer',
                style: TextStyle(
                    color: C.ink, fontSize: 18, fontWeight: FontWeight.w600)),
          ),
          if (synthetic)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
              decoration: BoxDecoration(
                  color: C.pale, borderRadius: BorderRadius.circular(8)),
              child: const Text('Synthetic sample',
                  style: TextStyle(
                      color: C.muted,
                      fontSize: 12,
                      fontWeight: FontWeight.w600)),
            ),
        ]),
        const SizedBox(height: 16),
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
                      foregroundColor: _view == view ? C.ink : C.muted,
                      minimumSize: const Size(0, 48),
                      padding: const EdgeInsets.symmetric(horizontal: 4),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(11)),
                      textStyle: const TextStyle(
                          fontSize: 12,
                          letterSpacing: .6,
                          fontWeight: FontWeight.w600),
                    ),
                    child: Text(view.name.toUpperCase(), maxLines: 1),
                  ),
                ),
              ),
          ]),
        ),
        const SizedBox(height: 20),
        Text(
          switch (_view) {
            _SignalView.wave => 'Signed peak waveform',
            _SignalView.spectrum => 'Frequency spectrum',
            _SignalView.decay => 'Peak-binned absolute envelope',
          },
          style: const TextStyle(
              color: C.ink, fontSize: 15, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 6),
        Text(
          spectrum
              ? 'Digital level · dBFS'
              : _view == _SignalView.wave
                  ? 'Signed amplitude · full scale (FS)'
                  : 'Absolute amplitude · full scale (FS)',
          style: const TextStyle(color: C.muted, fontSize: 12),
        ),
        const SizedBox(height: 10),
        if (problem != null)
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 36),
            decoration: BoxDecoration(
                color: C.bg, borderRadius: BorderRadius.circular(14)),
            child: Column(children: [
              const Icon(Icons.show_chart_rounded, color: C.muted, size: 28),
              const SizedBox(height: 12),
              Text(problem,
                  textAlign: TextAlign.center,
                  style:
                      const TextStyle(color: C.ink, fontSize: 14, height: 1.4)),
              const SizedBox(height: 6),
              const Text('Capture another tap to obtain a usable trace.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: C.muted, fontSize: 12, height: 1.4)),
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
                        peakHz: peak,
                        reveal: reveal,
                        textScaler: MediaQuery.textScalerOf(context),
                        textDirection: Directionality.of(context),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        const SizedBox(height: 14),
        if (spectrum) ...[
          _Reading(
            label: 'Measured peak',
            value: hasPeak ? _frequency(peak) : 'Not available',
          ),
          if (hasPeak && (peak < 20 || peak > 24000))
            const Padding(
              padding: EdgeInsets.only(top: 6),
              child: Text('Peak is outside the displayed 20 Hz–24 kHz range.',
                  style: TextStyle(color: C.muted, fontSize: 12, height: 1.4)),
            ),
        ],
        if (_view == _SignalView.decay) ...[
          _Reading(
            label: 'Observed 20 dB decay',
            value: decay.isFinite && decay > 0
                ? '${decay.toStringAsFixed(0)} ms'
                : 'Not measurable',
          ),
        ],
        ExpansionTile(
          tilePadding: EdgeInsets.zero,
          title:
              const Text('About this signal', style: TextStyle(fontSize: 13)),
          children: [
            if (_view == _SignalView.decay) ...[
              const SizedBox(height: 8),
              const Text(
                'This trace is the absolute value of each stored waveform peak, '
                'not a Schroeder energy-decay curve or RT60. The observed 20 dB '
                'decay above is a separate analysis result, not fitted to this plot.',
                style: TextStyle(color: C.muted, fontSize: 12, height: 1.5),
              ),
            ],
            if (!spectrum) ...[
              if (_view == _SignalView.decay) const SizedBox(height: 8),
              const Text(
                'Time is relative to the full capture. Its duration is not stored; '
                'these bins cannot resolve individual oscillations.',
                style: TextStyle(color: C.muted, fontSize: 12, height: 1.5),
              ),
            ],
            if (spectrum) ...[
              const SizedBox(height: 8),
              const Text(
                'Stored logarithmic bins: 20 Hz–24 kHz. Normalized levels are '
                'mapped to −100…0 dBFS, not calibrated sound pressure (SPL). '
                'The marker uses the measured peak frequency.',
                style: TextStyle(color: C.muted, fontSize: 12, height: 1.5),
              ),
            ],
            const SizedBox(height: 16),
            const Divider(height: 1, color: C.line),
            const SizedBox(height: 12),
            Text(
              'Source: ${measurement.source.trim().isEmpty ? "Not recorded" : measurement.source}',
              style: const TextStyle(color: C.muted, fontSize: 12, height: 1.5),
            ),
            const SizedBox(height: 3),
            Text(
              spectrum
                  ? '${values.length} stored frequency bins · no interpolation of missing data'
                  : '${values.length} stored signed peak-preserving bins · ${_view == _SignalView.decay ? "absolute values displayed" : "original sign retained"}',
              style: const TextStyle(color: C.muted, fontSize: 12, height: 1.5),
            ),
          ],
        ),
      ]),
    );
  }
}

String? _dataProblem(List<double> values, {required bool spectrum}) {
  if (values.isEmpty) {
    return spectrum ? 'No spectrum was saved.' : 'No waveform was saved.';
  }
  if (spectrum ? values.length != 256 : values.length > 256) {
    return spectrum
        ? 'The stored spectrum does not contain the expected 256 bins.'
        : 'The stored waveform has an unsupported bin count.';
  }
  for (final value in values) {
    if (!value.isFinite || value < (spectrum ? 0 : -1) || value > 1) {
      return 'The stored trace contains invalid levels and cannot be plotted.';
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
      'full-scale amplitude. Horizontal axis is 0 to 100 percent of the '
      'capture, not milliseconds.';
}

class _Reading extends StatelessWidget {
  const _Reading({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Wrap(
        spacing: 12,
        runSpacing: 4,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          Text(label, style: const TextStyle(color: C.muted, fontSize: 12)),
          Text(value,
              style: const TextStyle(
                  color: C.ink,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  fontFeatures: [FontFeature.tabularFigures()])),
        ],
      );
}

class _SignalPainter extends CustomPainter {
  const _SignalPainter({
    required this.values,
    required this.view,
    required this.peakHz,
    required this.reveal,
    required this.textScaler,
    required this.textDirection,
  });

  final List<double> values;
  final _SignalView view;
  final double peakHz;
  final double reveal;
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
      spectrum ? 'Frequency · Hz (log)' : 'Relative capture time · %',
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
        canvas.drawCircle(Offset(x, y), 2.5, Paint()..color = C.blue);
      }
    }
    if (!signed) {
      final fill = Path.from(trace)
        ..lineTo(lastX, plot.bottom)
        ..lineTo(firstX, plot.bottom)
        ..close();
      canvas.drawPath(fill, Paint()..color = C.blue.withValues(alpha: .07));
    }
    canvas.drawPath(
        trace,
        Paint()
          ..color = C.blue
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
              ..color = C.blue.withValues(alpha: .5)
              ..strokeWidth = 1);
        _label(canvas, 'Peak ${_frequency(peakHz)}',
            Offset(x.clamp(plot.left, plot.right).toDouble(), plot.top - 8),
            horizontal: fraction < .3
                ? 0
                : fraction > .7
                    ? 1
                    : .5,
            vertical: 1,
            color: C.blue);
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
      oldDelegate.textScaler != textScaler ||
      oldDelegate.textDirection != textDirection;
}
