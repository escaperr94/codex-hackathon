import 'dart:convert';
import 'dart:math' as math;
import 'package:material_ui/material_ui.dart';
import 'design.dart';
import 'store.dart';

class AcousticDetails extends StatelessWidget {
  const AcousticDetails({required this.measurement, required this.store, super.key});
  final Measurement measurement;
  final ThudStore store;

  @override
  Widget build(BuildContext context) {
    final raw = measurement.context['acousticAnalysis'];
    Map<String, dynamic> data = {};
    if (raw != null) {
      try { data = Map<String, dynamic>.from(jsonDecode(raw) as Map); }
      on FormatException { data = {}; }
    }
    final frames = (data['stft'] as List? ?? const [])
        .map((r) => (r as List).map((x) => (x as num).toDouble()).toList()).toList();
    final comparisons = store.measurements.where((m) => m.id != measurement.id &&
        m.hasEmbedding && m.encoderVersion == measurement.encoderVersion &&
        m.context['captureMode'] == measurement.context['captureMode'] &&
        m.source.toLowerCase().contains('synthetic') == measurement.source.toLowerCase().contains('synthetic')).toList();
    double distance(Measurement other) {
      var dot = 0.0, aa = 0.0, bb = 0.0;
      for (var i = 0; i < measurement.embedding.length; i++) {
        final a = measurement.embedding[i], b = other.embedding[i];
        dot += a*b; aa += a*a; bb += b*b;
      }
      return aa <= 0 || bb <= 0 ? 1 : (1-dot/math.sqrt(aa*bb)).clamp(0.0,2.0);
    }
    if (measurement.hasEmbedding) comparisons.sort((a,b) => distance(a).compareTo(distance(b)));
    return Panel(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const SectionTitle('Acoustic evidence'),
      if (data.isNotEmpty) ...[
        const SizedBox(height: 12),
        Text('Centroid: ${data['centroidHz'] ?? 'unavailable'} Hz · Flatness: ${data['flatness'] ?? 'unavailable'}', style: const TextStyle(color: C.muted)),
        Text('Modal peaks: ${data['modalPeaksHz'] ?? 'unavailable'} Hz', style: const TextStyle(color: C.muted)),
        Text('Damping: ${data['damping'] ?? 'unavailable'} · uncertainty: ${data['uncertainty'] ?? 'unavailable'}', style: const TextStyle(color: C.muted)),
      ],
      if (frames.isNotEmpty) ...[
        const SizedBox(height: 12),
        const Text('Spectrogram · relative log power'),
        Semantics(image: true, label: 'Time-frequency spectrogram of this recording',
          child: SizedBox(height: 180, width: double.infinity,
            child: CustomPaint(painter: _Spectrogram(frames, Theme.of(context).colorScheme.primary)))),
      ],
      const SizedBox(height: 12),
      const Text('Compare neural signatures', style: TextStyle(fontWeight: FontWeight.w600)),
      if (!measurement.hasEmbedding) const Text('Re-record to obtain a neural signature.'),
      if (measurement.hasEmbedding && comparisons.isEmpty) const Text('Save another recording with the same capture mode and provenance.'),
      if (measurement.hasEmbedding) for (final m in comparisons.take(8))
        ListTile(contentPadding: EdgeInsets.zero, title: Text(m.label),
          subtitle: Text('${m.created.toLocal()} · cosine distance ${distance(m).toStringAsFixed(3)}'),
          trailing: store.reference?.id == m.id ? const Icon(Icons.check_circle_outline) : null,
          onTap: () => showDialog<void>(context: context, builder: (dialog) => AlertDialog(
            title: Text('Compare with ${m.label}'),
            content: Text('Neural cosine distance: ${distance(m).toStringAsFixed(4)}\n'
              'Peak shift: ${(measurement.peakHz-m.peakHz).toStringAsFixed(1)} Hz\n'
              'Ring shift: ${(measurement.decayMs-m.decayMs).toStringAsFixed(1)} ms\n'
              'Similarity is not a physical diagnosis.'),
            actions: [TextButton(onPressed: () => Navigator.pop(dialog), child: const Text('Close'))],
          ))),
    ]));
  }
}

class _Spectrogram extends CustomPainter {
  const _Spectrogram(this.frames, this.color);
  final List<List<double>> frames;
  final Color color;
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();
    for (var t=0;t<frames.length;t++) {
      for (var f=0;f<frames[t].length;f++) {
        final value = frames[t][f];
        paint.color = color.withValues(alpha: value.isFinite ? value.clamp(0.0,1.0) : 0);
        canvas.drawRect(Rect.fromLTRB(t*size.width/frames.length,
          size.height-(f+1)*size.height/frames[t].length,
          (t+1)*size.width/frames.length,size.height-f*size.height/frames[t].length),paint);
      }
    }
  }
  @override
  bool shouldRepaint(covariant _Spectrogram old) => old.frames != frames || old.color != color;
}
