import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:material_ui/material_ui.dart';

import 'design.dart';
import 'floor_plan.dart';
import 'store.dart';
import 'tile_geometry.dart';

/// Geometry stays a draft until Save; replacing a grid never silently erases links.
class TileInspectionEditor extends StatefulWidget {
  const TileInspectionEditor({super.key, required this.store, required this.project});
  final ThudStore store;
  final Project project;
  @override
  State<TileInspectionEditor> createState() => _TileInspectionEditorState();
}

class _TileInspectionEditorState extends State<TileInspectionEditor> {
  late TileLayout _layout;
  String? _selected, _error;
  String _message = 'Automatic detection needs visible grout in two directions. '
      'Set four corners for an angled floor; use a manual grid if grout is hidden.';
  String? _pointMode;
  List<TilePoint> _points = [];
  bool _busy = false, _changed = false;
  @override
  void initState() {
    super.initState();
    _layout = widget.project.floorPlan!.tileLayout ?? TileLayout.empty();
  }
  TileRegion? get _tile => _layout.tiles.where((t) => t.id == _selected).firstOrNull;
  void _change(TileLayout layout) => setState(() {
    _layout = layout;
    _changed = true;
    _error = null;
    if (!_layout.tiles.any((t) => t.id == _selected)) _selected = null;
  });
  Future<bool> _confirm(String title, String message) async => await showDialog<bool>(
      context: context, builder: (context) => AlertDialog(title: Text(title),
      content: Text(message), actions: [
        TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
        TextButton(onPressed: () => Navigator.pop(context, true), child: const Text('Continue')),
      ])) ?? false;
  Future<void> _detect() async {
    setState(() { _busy = true; _error = null; });
    try {
      final plan = widget.project.floorPlan!;
      final codec = await ui.instantiateImageCodec(plan.bytes,
          targetWidth: plan.width >= plan.height ? 768 : null,
          targetHeight: plan.height > plan.width ? 768 : null,
          allowUpscaling: false);
      late TileDetection result;
      try {
        final frame = await codec.getNextFrame();
        try {
          final data = await frame.image.toByteData(format: ui.ImageByteFormat.rawRgba);
          if (data == null) throw const FormatException('Could not read image pixels.');
          result = await compute(detectTileGrid, TileRaster(
              Uint8List.fromList(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes)),
              frame.image.width, frame.image.height, _layout.corners));
        } finally { frame.image.dispose(); }
      } finally { codec.dispose(); }
      if (!mounted) return;
      setState(() => _message = result.message);
      if (result.found) _change(_layout.grid(result.xs, result.ys, source: 'grout-geometry'));
    } catch (error) {
      if (mounted) setState(() => _error = '$error');
    } finally { if (mounted) setState(() => _busy = false); }
  }
  Future<void> _grid() async {
    final rows = TextEditingController(text: '${widget.project.rows}');
    final columns = TextEditingController(text: '${widget.project.columns}');
    final result = await showDialog<(int, int)>(context: context, builder: (context) {
      String? error;
      return StatefulBuilder(builder: (context, update) => AlertDialog(
        title: const Text('Manual visible tile grid'),
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          const Text('Grid fills the selected four-corner area. Include partial edge tiles; correct their boundaries afterward.'),
          TextField(controller: rows, keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: 'Rows (1–50)')),
          TextField(controller: columns, keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: 'Columns (1–50)')),
          if (error != null) Text(error!, style: const TextStyle(color: C.red)),
        ]), actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          TextButton(onPressed: () {
            final r = int.tryParse(rows.text), c = int.tryParse(columns.text);
            if (r == null || c == null || r < 1 || c < 1 || r > 50 || c > 50) {
              update(() => error = 'Enter whole numbers from 1 to 50.');
              return;
            }
            Navigator.pop(context, (r, c));
          }, child: const Text('Create grid')),
        ]));
    });
    rows.dispose(); columns.dispose();
    if (result == null || !mounted) return;
    _change(_layout.grid([for (var c = 0; c <= result.$2; c++) c / result.$2],
        [for (var r = 0; r <= result.$1; r++) r / result.$1]));
  }
  void _startPoints(String mode) => setState(() {
    _pointMode = mode; _points = []; _error = null;
  });
  void _point(double x, double y) {
    if (_pointMode == null) return;
    setState(() => _points = [..._points, TilePoint(x, y)]);
    if (_points.length < 4) return;
    try {
      TileLayout.validateQuad(_points);
      if (_pointMode == 'corners') {
        _change(TileLayout(tiles: _layout.tiles, corners: _points,
            nextId: _layout.nextId, method: _layout.method));
        setState(() => _message = 'Perspective corners set. Detect grout or create a grid within this area.');
      } else if (_pointMode == 'add') {
        final id = 'T${_layout.nextId}';
        _change(_layout.add(_points));
        _selected = id;
      } else if (_selected != null) {
        _change(_layout.correct(_selected!, _points));
      }
      setState(() { _pointMode = null; _points = []; });
    } catch (error) {
      setState(() { _error = '$error Tap four corners again.'; _points = []; });
    }
  }
  Future<void> _save() async {
    final ids = _layout.tiles.map((t) => t.id).toSet();
    final linked = widget.store.annotationsForProject(widget.project.id)
        .where((a) => a.tileId != null && !ids.contains(a.tileId)).length;
    if (linked > 0 && !await _confirm('Reset $linked tile links?',
        'Removed/replaced tiles have saved findings. Their tile links will be cleared; '
        'notes, image pins, photos and recordings stay saved.')) return;
    if (!mounted) return;
    setState(() { _busy = true; _error = null; });
    try {
      await widget.store.updateTileLayout(widget.project, _layout, resetTileLinks: linked > 0);
      if (mounted) {
        setState(() { _changed = false; _busy = false; });
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) Navigator.pop(context);
        });
      }
    } catch (error) {
      if (mounted) setState(() => _error = 'Map was not saved. $error');
    } finally { if (mounted) setState(() => _busy = false); }
  }
  Future<void> _close() async {
    if (_busy) return;
    if (_changed && !await _confirm('Discard map changes?', 'The saved map has not changed.')) return;
    if (mounted) {
      setState(() => _changed = false);
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) Navigator.pop(context);
      });
    }
  }
  @override
  Widget build(BuildContext context) => PopScope(
      canPop: !_changed && !_busy,
      onPopInvokedWithResult: (didPop, _) { if (!didPop) _close(); },
      child: Scaffold(backgroundColor: C.bg,
        appBar: AppBar(title: const Text('Tile inspection map'), backgroundColor: C.bg,
          leading: IconButton(onPressed: _close, icon: const Icon(Icons.arrow_back))),
        body: AbsorbPointer(absorbing: _busy, child: AppPage(children: [
          PageHeading('${_layout.tiles.length} visible tiles', subtitle: 'Offline geometry · review before saving'),
          const SizedBox(height: 12),
          FloorPlanCanvas(plan: widget.project.floorPlan!, layout: _layout,
              marks: const [], selectedTileId: _selected,
              controlPoints: _pointMode == null ? _layout.corners : _points,
              onTileSelect: _pointMode != null ? null : (tile) => setState(() => _selected = tile.id),
              onPoint: _pointMode != null ? _point : null, height: 380),
          const SizedBox(height: 12),
          Text(_pointMode == null ? _message :
              '${_pointMode == 'corners' ? 'Floor area' : 'Tile boundary'}: tap top-left, top-right, bottom-right, bottom-left. ${_points.length}/4',
              style: const TextStyle(color: C.muted, height: 1.5)),
          const SizedBox(height: 12),
          if (_pointMode != null) Wrap(spacing: 8, children: [
            TextButton(onPressed: () => setState(() { _pointMode = null; _points = []; }), child: const Text('Cancel corners')),
            TextButton(onPressed: () => setState(() => _points = []), child: const Text('Restart')),
          ]) else ...[
            Wrap(spacing: 8, runSpacing: 8, children: [
              OutlinedButton.icon(onPressed: _detect, icon: const Icon(Icons.auto_fix_high), label: const Text('Detect grout')),
              OutlinedButton(onPressed: () => _startPoints('corners'), child: const Text('Perspective corners')),
              OutlinedButton(onPressed: _grid, child: const Text('Manual grid')),
              OutlinedButton(onPressed: () => _startPoints('add'), child: const Text('Add tile')),
            ]),
            if (_tile case final tile?) ...[
              const SizedBox(height: 16),
              SectionTitle('${tile.id} · row ${tile.row + 1}, column ${tile.column + 1}'),
              Wrap(spacing: 8, children: [
                TextButton(onPressed: () => _startPoints('correct'), child: const Text('Correct boundary')),
                TextButton(onPressed: () => _change(_layout.split(tile.id, vertical: true)), child: const Text('Split columns')),
                TextButton(onPressed: () => _change(_layout.split(tile.id, vertical: false)), child: const Text('Split rows')),
                TextButton(onPressed: () => _change(_layout.remove(tile.id)), child: const Text('Remove tile')),
              ]),
            ],
          ],
          if (_error != null) Padding(padding: const EdgeInsets.symmetric(vertical: 12),
              child: Text(_error!, style: const TextStyle(color: C.red))),
          if (_busy) const LinearProgressIndicator(),
          const SizedBox(height: 20),
          ActionButton(_busy ? 'Processing…' : 'Save tile map', icon: Icons.check,
              onPressed: _busy || _pointMode != null ? null : _save),
          const SizedBox(height: 12),
          const Text('Count includes only the selected image area, including partial edge tiles. '
              'Geometry is not a defect diagnosis or a physical area measurement.',
              style: TextStyle(color: C.muted, fontSize: 12, height: 1.5)),
        ])),
      ));
}
