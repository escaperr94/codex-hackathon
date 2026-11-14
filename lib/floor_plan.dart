import 'dart:ui' as ui;

import 'package:material_ui/material_ui.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;

import 'design.dart';
import 'store.dart';
import 'tile_geometry.dart';

/// All plan views use a centered contain fit; normalized pins target pixel centers.
Rect floorPlanImageRect(FloorPlan plan, Size viewport) {
  final fit = applyBoxFit(BoxFit.contain,
      Size(plan.width.toDouble(), plan.height.toDouble()), viewport);
  return Alignment.center.inscribe(fit.destination, Offset.zero & viewport);
}

Color floorPlanStatusColor(AnnotationStatus status) => switch (status) {
      AnnotationStatus.review => C.amber,
      AnnotationStatus.defect => C.red,
      AnnotationStatus.accepted => C.green,
    };

class FloorPlanCanvas extends StatelessWidget {
  const FloorPlanCanvas(
      {super.key,
      required this.plan,
      required this.marks,
      this.onSelect,
      this.onPoint,
      this.draftX,
      this.draftY,
      this.selectedId,
      this.layout,
      this.onTileSelect,
      this.selectedTileId,
      this.controlPoints = const [],
      this.height = 340});
  final FloorPlan plan;
  final List<DefectAnnotation> marks;
  final ValueChanged<DefectAnnotation>? onSelect;
  final void Function(double, double)? onPoint;
  final double? draftX, draftY;
  final String? selectedId;
  final double height;
  final TileLayout? layout;
  final ValueChanged<TileRegion>? onTileSelect;
  final String? selectedTileId;
  final List<TilePoint> controlPoints;

  @override
  Widget build(BuildContext context) => SizedBox(
        height: height,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: ColoredBox(
              color: C.pale,
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final viewport =
                      Size(constraints.maxWidth, constraints.maxHeight);
                  final rect = floorPlanImageRect(plan, viewport);
                  return InteractiveViewer(
                    minScale: 1,
                    maxScale: 5,
                    child: SizedBox(
                      width: viewport.width,
                      height: viewport.height,
                      child: GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTapUp: onPoint == null && onTileSelect == null
                            ? null
                            : (details) {
                                final point = details.localPosition;
                                if (!rect.contains(point)) return;
                                final normalized = TilePoint(
                                    (point.dx - rect.left) / rect.width,
                                    (point.dy - rect.top) / rect.height);
                                final tile = (layout ?? plan.tileLayout)?.at(normalized);
                                if (tile != null && onTileSelect != null) {
                                  onTileSelect!(tile);
                                  return;
                                }
                                onPoint?.call(
                                    ((point.dx - rect.left) / rect.width)
                                        .clamp(0.0, 1.0)
                                        .toDouble(),
                                    ((point.dy - rect.top) / rect.height)
                                        .clamp(0.0, 1.0)
                                        .toDouble());
                              },
                        child: Stack(clipBehavior: Clip.none, children: [
                          Positioned.fromRect(
                              rect: rect,
                              child: Image.memory(plan.bytes,
                                  fit: BoxFit.fill,
                                  excludeFromSemantics: true,
                                  errorBuilder: (_, __, ___) => const Center(
                                      child: Text(
                                          'Could not display imported plan.')))),
                          Positioned.fromRect(
                              rect: rect,
                              child: IgnorePointer(child: CustomPaint(
                                  painter: _TilePainter(layout ?? plan.tileLayout,
                                      selectedTileId, controlPoints)))),
                          for (var i = 0; i < marks.length; i++)
                            if (marks[i].planX != null &&
                                marks[i].planY != null &&
                                !(onPoint != null && marks[i].id == selectedId))
                              Positioned(
                                  left: rect.left +
                                      rect.width * marks[i].planX! -
                                      15,
                                  top: rect.top +
                                      rect.height * marks[i].planY! -
                                      15,
                                  child: Semantics(
                                      button: onSelect != null,
                                      label:
                                          'Finding ${i + 1}, ${marks[i].status.name}, cell ${marks[i].cell + 1}',
                                      child: GestureDetector(
                                          behavior: HitTestBehavior.opaque,
                                          onTap: onSelect == null
                                              ? null
                                              : () => onSelect!(marks[i]),
                                          child: _PlanPin(
                                              label: '${i + 1}',
                                              color: floorPlanStatusColor(
                                                  marks[i].status))))),
                          if (draftX != null && draftY != null)
                            Positioned(
                                left: rect.left + rect.width * draftX! - 15,
                                top: rect.top + rect.height * draftY! - 15,
                                child: const IgnorePointer(
                                    child:
                                        _PlanPin(label: '+', color: C.blue))),
                        ]),
                      ),
                    ),
                  );
                },
              )),
        ),
      );
}

class _PlanPin extends StatelessWidget {
  const _PlanPin({required this.label, required this.color});
  final String label;
  final Color color;
  @override
  Widget build(BuildContext context) => Container(
        width: 30,
        height: 30,
        alignment: Alignment.center,
        decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white, width: 2)),
        child: Text(label,
            style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600)),
      );
}

/// Convert decoded pixels to PNG, so WebP plans export just like JPEG/PNG plans.
Future<pw.MemoryImage> floorPlanPdfImage(FloorPlan plan) async {
  final codec = await ui.instantiateImageCodec(plan.bytes);
  try {
    final frame = await codec.getNextFrame();
    try {
      final data = await frame.image.toByteData(format: ui.ImageByteFormat.png);
      if (data == null) throw Exception('Could not encode the imported plan.');
      return pw.MemoryImage(
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
    } finally {
      frame.image.dispose();
    }
  } finally {
    codec.dispose();
  }
}

pw.Widget floorPlanPdfMap(
    FloorPlan plan, pw.MemoryImage image, List<DefectAnnotation> marks) {
  const width = 515.0, height = 340.0;
  final rect = floorPlanImageRect(plan, const Size(width, height));
  return pw.SizedBox(
    width: width,
    height: height,
    child: pw.Stack(overflow: pw.Overflow.visible, children: [
      pw.Positioned(
          left: rect.left,
          top: rect.top,
          child: pw.SizedBox(
              width: rect.width,
              height: rect.height,
              child: pw.Image(image, fit: pw.BoxFit.fill))),
      if (plan.tileLayout != null)
        pw.Positioned(left: rect.left, top: rect.top,
            child: pw.SizedBox(width: rect.width, height: rect.height,
                child: pw.SvgImage(svg: tileLayoutSvg(plan.tileLayout!,
                    plan.width.toDouble(), plan.height.toDouble())))),
      for (var i = 0; i < marks.length; i++)
        if (marks[i].planX != null && marks[i].planY != null)
          pw.Positioned(
              left: rect.left + rect.width * marks[i].planX! - 7,
              top: rect.top + rect.height * marks[i].planY! - 7,
              child: pw.Container(
                  width: 14,
                  height: 14,
                  alignment: pw.Alignment.center,
                  decoration: pw.BoxDecoration(
                      shape: pw.BoxShape.circle,
                      border: pw.Border.all(color: PdfColors.white, width: 1),
                      color: switch (marks[i].status) {
                        AnnotationStatus.review => PdfColors.orange700,
                        AnnotationStatus.defect => PdfColors.red700,
                        AnnotationStatus.accepted => PdfColors.green700,
                      }),
                  child: pw.Text('${i + 1}',
                      style: const pw.TextStyle(
                          color: PdfColors.white, fontSize: 7)))),
    ]),
  );
}

class _TilePainter extends CustomPainter {
  const _TilePainter(this.layout, this.selected, this.points);
  final TileLayout? layout;
  final String? selected;
  final List<TilePoint> points;
  @override
  void paint(Canvas canvas, Size size) {
    for (final tile in layout?.tiles ?? <TileRegion>[]) {
      final path = Path()..moveTo(tile.polygon.first.x * size.width,
          tile.polygon.first.y * size.height);
      for (final p in tile.polygon.skip(1)) {
        path.lineTo(p.x * size.width, p.y * size.height);
      }
      path.close();
      if (tile.id == selected) {
        canvas.drawPath(path, Paint()..color = C.blue.withValues(alpha: .2));
      }
      canvas.drawPath(path, Paint()..color = C.blue
          ..style = PaintingStyle.stroke..strokeWidth = 1.5);
      final center = tile.center;
      final label = TextPainter(text: TextSpan(text: tile.id,
          style: const TextStyle(fontSize: 10, color: Colors.white,
              backgroundColor: C.blue)), textDirection: TextDirection.ltr)..layout();
      label.paint(canvas, Offset(center.x * size.width - label.width / 2,
          center.y * size.height - label.height / 2));
      label.dispose();
    }
    for (var i = 0; i < points.length; i++) {
      final p = points[i];
      canvas.drawCircle(Offset(p.x * size.width, p.y * size.height), 6,
          Paint()..color = C.amber);
      final text = TextPainter(text: TextSpan(text: '${i + 1}',
          style: const TextStyle(color: Colors.black, fontSize: 10)),
          textDirection: TextDirection.ltr)..layout();
      text.paint(canvas, Offset(p.x * size.width - 3, p.y * size.height - 6));
      text.dispose();
    }
  }
  @override
  bool shouldRepaint(_TilePainter old) => old.layout != layout ||
      old.selected != selected || old.points != points;
}

/// Same normalized vertices/IDs as the interactive canvas; pins render above it.
String tileLayoutSvg(TileLayout layout, double width, double height) {
  final out = StringBuffer('<svg xmlns="http://www.w3.org/2000/svg" '
      'viewBox="0 0 $width $height">');
  for (final tile in layout.tiles) {
    final vertices = tile.polygon.map((p) => '${p.x * width},${p.y * height}').join(' ');
    out.write('<polygon points="$vertices" fill="none" stroke="#3876df" '
        'stroke-width="${width / 400}"/>');
    final p = tile.center;
    // IDs imported from storage are escaped before entering XML.
    final id = tile.id.replaceAll('&', '&amp;').replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
    out.write('<text x="${p.x * width}" y="${p.y * height}" '
        'text-anchor="middle" fill="#1757ae" font-size="${width / 50}">$id</text>');
  }
  out.write('</svg>');
  return out.toString();
}
