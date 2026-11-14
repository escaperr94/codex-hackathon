import 'dart:ui' as ui;

import 'package:material_ui/material_ui.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;

import 'design.dart';
import 'store.dart';

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
      this.height = 340});
  final FloorPlan plan;
  final List<DefectAnnotation> marks;
  final ValueChanged<DefectAnnotation>? onSelect;
  final void Function(double, double)? onPoint;
  final double? draftX, draftY;
  final String? selectedId;
  final double height;

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
                        onTapUp: onPoint == null
                            ? null
                            : (details) {
                                final point = details.localPosition;
                                if (!rect.contains(point)) return;
                                onPoint!(
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
