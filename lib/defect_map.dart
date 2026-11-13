import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:image_picker/image_picker.dart';
import 'package:material_ui/material_ui.dart';

import 'design.dart';
import 'store.dart';

/// Human inspection marks are deliberately separate from acoustic evidence.
class DefectMapScreen extends StatefulWidget {
  const DefectMapScreen(
      {super.key,
      required this.store,
      required this.project,
      this.measurement});
  final ThudStore store;
  final Project project;
  final Measurement? measurement;

  @override
  State<DefectMapScreen> createState() => _DefectMapScreenState();
}

class _DefectMapScreenState extends State<DefectMapScreen> {
  final _note = TextEditingController();
  final _picker = ImagePicker();
  int _cell = 0;
  String _id = '';
  String? _linkId, _sourceSnapshot, _error;
  AnnotationStatus _status = AnnotationStatus.review;
  Uint8List? _photo;
  double? _x, _y;
  bool _busy = false;
  Object? _baseline;
  Object get _draft => (_note.text, _status, _linkId, _photo, _x, _y);
  bool get _dirty => _draft != _baseline;

  @override
  void initState() {
    super.initState();
    final m = widget.measurement;
    if (m?.projectId == widget.project.id && m?.cell != null) _cell = m!.cell!;
    _linkId = m?.id;
    _sourceSnapshot = m?.source;
    _baseline = _draft;
  }

  @override
  void dispose() {
    _note.dispose();
    super.dispose();
  }

  String _cellName(int cell) =>
      'Row ${cell ~/ widget.project.columns + 1}, column ${cell % widget.project.columns + 1}';

  String _statusName(AnnotationStatus status) => switch (status) {
        AnnotationStatus.review => 'Needs review',
        AnnotationStatus.defect => 'Marked defect',
        AnnotationStatus.accepted => 'Accepted',
      };

  Color _statusColor(AnnotationStatus status) => switch (status) {
        AnnotationStatus.review => C.amber,
        AnnotationStatus.defect => C.red,
        AnnotationStatus.accepted => C.green,
      };

  Future<void> _newMark({int? cell, bool confirmed = false}) async {
    if (!confirmed && !await _canDiscard()) return;
    if (!mounted) return;
    setState(() {
      if (cell != null) _cell = cell;
      _id = '';
      _note.clear();
      _status = AnnotationStatus.review;
      _linkId = widget.measurement?.id;
      _sourceSnapshot = widget.measurement?.source;
      _photo = null;
      _x = _y = null;
      _error = null;
      _baseline = _draft;
    });
  }

  Future<void> _open(DefectAnnotation mark, {bool confirmed = false}) async {
    if (_id == mark.id) return;
    if (!confirmed && !await _canDiscard()) return;
    if (!mounted) return;
    try {
      final bytes =
          mark.photoBase64 == null ? null : base64Decode(mark.photoBase64!);
      setState(() {
        _id = mark.id;
        _cell = mark.cell;
        _status = mark.status;
        _note.text = mark.note;
        _linkId = mark.measurementId;
        _sourceSnapshot = mark.measurementSource;
        _photo = bytes;
        _x = mark.x;
        _y = mark.y;
        _error = null;
        _baseline = _draft;
      });
    } catch (error) {
      setState(() => _error = 'Could not reopen the saved photo: $error');
    }
  }

  Future<bool> _canDiscard() async {
    if (!_dirty) return true;
    return await showDialog<bool>(
            context: context,
            builder: (context) => AlertDialog(
                    title: const Text('Discard unsaved changes?'),
                    content: const Text(
                        'Your note, status, photo and pin have not been saved.'),
                    actions: [
                      TextButton(
                          onPressed: () => Navigator.pop(context, false),
                          child: const Text('Keep editing')),
                      TextButton(
                          onPressed: () => Navigator.pop(context, true),
                          child: const Text('Discard')),
                    ])) ??
        false;
  }

  Future<void> _selectCell(int cell) async {
    if (cell == _cell || !await _canDiscard()) return;
    if (!mounted) return;
    final existing = widget.store
        .annotationsForProject(widget.project.id)
        .where((a) => a.cell == cell)
        .firstOrNull;
    if (existing == null) {
      await _newMark(cell: cell, confirmed: true);
    } else {
      await _open(existing, confirmed: true);
    }
  }

  Future<void> _pick(ImageSource source) async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final file = await _picker.pickImage(
          source: source,
          maxWidth: 1400,
          maxHeight: 1400,
          imageQuality: 70,
          requestFullMetadata: false);
      if (file == null) return;
      if (await file.length() > DefectAnnotation.maxPhotoBytes) {
        throw Exception('Photo exceeds 2 MB. Choose a smaller photo.');
      }
      final bytes = await file.readAsBytes();
      if (bytes.isEmpty || bytes.length > DefectAnnotation.maxPhotoBytes) {
        throw Exception('Choose a nonempty photo below 2 MB.');
      }
      // Decode before accepting it, so corrupt or unsupported images cannot be saved.
      final codec = await ui.instantiateImageCodec(bytes);
      try {
        final frame = await codec.getNextFrame();
        frame.image.dispose();
      } finally {
        codec.dispose();
      }
      if (!mounted) return;
      setState(() {
        _photo = bytes;
        _x = _y = null;
      });
    } catch (error) {
      if (mounted) {
        setState(() => _error =
            'Could not add photo. Allow camera or photo access in device settings if requested. $error');
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final supplied = widget.measurement;
      final saved = await widget.store.upsertAnnotation(
          DefectAnnotation(
              id: _id,
              projectId: widget.project.id,
              cell: _cell,
              status: _status,
              note: _note.text.trim(),
              measurementId: _linkId,
              photoBase64: _photo == null ? null : base64Encode(_photo!),
              x: _x,
              y: _y),
          measurement: supplied?.id == _linkId ? supplied : null);
      if (!mounted) return;
      setState(() {
        _id = saved.id;
        _baseline = _draft;
      });
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Human annotation saved on this device.')));
    } catch (error) {
      if (mounted) {
        setState(() => _error =
            'Nothing was saved. Device storage may be full or its local quota exceeded. Remove the photo or free space, then retry. $error');
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => PopScope(
      canPop: !_busy && !_dirty,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop || _busy) return;
        if (!await _canDiscard() || !context.mounted) return;
        setState(() => _baseline = _draft);
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (context.mounted) Navigator.pop(context);
        });
      },
      child: Scaffold(
        backgroundColor: C.bg,
        appBar: AppBar(title: const Text('Defect map'), backgroundColor: C.bg),
        body: AnimatedBuilder(
            animation: widget.store,
            builder: (context, _) {
              final project = widget.project;
              final marks = widget.store.annotationsForProject(project.id);
              final records = widget.store.forProject(project.id);
              final here = marks.where((a) => a.cell == _cell).toList();
              final choices = <String, Measurement>{
                for (final m in records.where((m) => m.cell == _cell)) m.id: m,
                if (widget.measurement != null)
                  widget.measurement!.id: widget.measurement!,
              };
              final linked = choices[_linkId];
              return AbsorbPointer(
                  absorbing: _busy,
                  child: AppPage(children: [
                    PageHeading(project.name,
                        subtitle:
                            '${project.room.isEmpty ? 'Inspection grid' : project.room} · ${project.rows} × ${project.columns}'),
                    const SizedBox(height: 20),
                    const Text('Your marks · not an automatic diagnosis',
                        style: TextStyle(color: C.muted, fontSize: 12)),
                    SizedBox(
                        height: 280,
                        child: SingleChildScrollView(
                            scrollDirection: Axis.horizontal,
                            child: SizedBox(
                                width: (project.columns * 60.0)
                                    .clamp(280.0, 6000.0),
                                child: GridView.builder(
                                    gridDelegate:
                                        SliverGridDelegateWithFixedCrossAxisCount(
                                            crossAxisCount: project.columns,
                                            mainAxisSpacing: 6,
                                            crossAxisSpacing: 6,
                                            childAspectRatio: 1),
                                    itemCount: project.rows * project.columns,
                                    itemBuilder: (context, cell) {
                                      final cellMarks = marks
                                          .where((a) => a.cell == cell)
                                          .toList();
                                      final cellRecords = records
                                          .where((m) => m.cell == cell)
                                          .toList();
                                      final invalid = cellRecords
                                          .any((m) => m.flags.isNotEmpty);
                                      final synthetic = cellRecords.any((m) => m
                                              .source
                                              .toLowerCase()
                                              .contains('synthetic')) ||
                                          cellMarks.any((a) =>
                                              a.measurementSource
                                                  ?.toLowerCase()
                                                  .contains('synthetic') ??
                                              false);
                                      final selected = cell == _cell;
                                      return Semantics(
                                          button: true,
                                          selected: selected,
                                          label:
                                              '${_cellName(cell)}, ${cellMarks.length} human marks${invalid ? ", flagged recording" : ""}${synthetic ? ", synthetic source" : ""}',
                                          child: InkWell(
                                            onTap: () => _selectCell(cell),
                                            borderRadius:
                                                BorderRadius.circular(12),
                                            child: AnimatedContainer(
                                                duration: Duration(
                                                    milliseconds: MediaQuery
                                                            .disableAnimationsOf(
                                                                context)
                                                        ? 0
                                                        : 180),
                                                decoration:
                                                    BoxDecoration(
                                                        color: selected
                                                            ? C.blue
                                                            : Colors.white,
                                                        borderRadius:
                                                            BorderRadius
                                                                .circular(12),
                                                        border: Border.all(
                                                            color: selected
                                                                ? C.blue
                                                                : C.line,
                                                            width: selected
                                                                ? 2
                                                                : 1)),
                                                child: Stack(children: [
                                                  Center(
                                                      child: Text('${cell + 1}',
                                                          style: TextStyle(
                                                              color: selected
                                                                  ? Colors.white
                                                                  : C.ink,
                                                              fontWeight:
                                                                  FontWeight
                                                                      .w600))),
                                                  if (cellMarks.isNotEmpty)
                                                    Positioned(
                                                        top: 5,
                                                        right: 5,
                                                        child: Container(
                                                            width: 9,
                                                            height: 9,
                                                            decoration: BoxDecoration(
                                                                color: _statusColor(
                                                                    cellMarks
                                                                        .first
                                                                        .status),
                                                                shape: BoxShape
                                                                    .circle,
                                                                border: Border.all(
                                                                    color: Colors
                                                                        .white)))),
                                                  if (invalid)
                                                    Positioned(
                                                        bottom: 3,
                                                        left: 3,
                                                        child: Icon(
                                                            Icons
                                                                .warning_amber_rounded,
                                                            size: 14,
                                                            color: selected
                                                                ? Colors.white
                                                                : C.amber)),
                                                  if (synthetic)
                                                    Positioned(
                                                        bottom: 3,
                                                        right: 5,
                                                        child: Text('S',
                                                            style: TextStyle(
                                                                fontSize: 10,
                                                                fontWeight:
                                                                    FontWeight
                                                                        .w600,
                                                                color: selected
                                                                    ? Colors
                                                                        .white
                                                                    : C.muted))),
                                                ])),
                                          ));
                                    })))),
                    const SizedBox(height: 24),
                    SectionTitle(_cellName(_cell),
                        trailing: TextButton.icon(
                            onPressed: () => _newMark(),
                            icon: const Icon(Icons.add, size: 18),
                            label: const Text('New mark'))),
                    if (here.isEmpty)
                      const Padding(
                          padding: EdgeInsets.symmetric(vertical: 12),
                          child: Text('No findings yet.',
                              style: TextStyle(color: C.muted))),
                    for (final mark in here)
                      Padding(
                          padding: const EdgeInsets.only(top: 8),
                          child: ListTile(
                              onTap: () => _open(mark),
                              selected: _id == mark.id,
                              selectedTileColor: C.pale,
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16)),
                              leading: Icon(
                                  mark.photoBase64 == null
                                      ? Icons.location_on_outlined
                                      : Icons.photo_outlined,
                                  color: _statusColor(mark.status)),
                              title: Text(_statusName(mark.status)),
                              subtitle: Text(
                                  mark.note.isEmpty
                                      ? 'Human mark · tap to reopen'
                                      : mark.note,
                                  maxLines: 2,
                                  overflow: TextOverflow.ellipsis),
                              trailing:
                                  const Icon(Icons.chevron_right_rounded))),
                    const SizedBox(height: 20),
                    Panel(
                        child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                          Text(_id.isEmpty ? 'Add finding' : 'Edit finding',
                              style: const TextStyle(
                                  fontSize: 20, fontWeight: FontWeight.w600)),
                          const SizedBox(height: 16),
                          Wrap(spacing: 8, runSpacing: 8, children: [
                            for (final status in AnnotationStatus.values)
                              ChoiceChip(
                                  label: Text(_statusName(status)),
                                  selected: _status == status,
                                  onSelected: (_) =>
                                      setState(() => _status = status))
                          ]),
                          const SizedBox(height: 16),
                          TextField(
                              controller: _note,
                              minLines: 1,
                              maxLines: 6,
                              maxLength: 10000,
                              onChanged: (_) => setState(() {}),
                              decoration: const InputDecoration(
                                  labelText: 'Note',
                                  hintText: 'What needs attention?',
                                  alignLabelWithHint: true)),
                          const SizedBox(height: 12),
                          DropdownButtonFormField<String>(
                              key: ValueKey('$_id:$_cell:$_linkId'),
                              initialValue:
                                  choices.containsKey(_linkId) ? _linkId : null,
                              isExpanded: true,
                              decoration:
                                  const InputDecoration(labelText: 'Recording'),
                              hint: const Text('No linked record'),
                              items: [
                                const DropdownMenuItem<String>(
                                    value: '', child: Text('No linked record')),
                                for (final m in choices.values)
                                  DropdownMenuItem(
                                      value: m.id,
                                      child: Text(
                                          '${m.source.toLowerCase().contains('synthetic') ? 'Synthetic · ' : ''}${m.label}',
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis))
                              ],
                              onChanged: (id) => setState(() {
                                    _linkId = id == '' ? null : id;
                                    _sourceSnapshot = choices[_linkId]?.source;
                                  })),
                          if (linked != null) ...[
                            const SizedBox(height: 10),
                            if (linked.flags.isNotEmpty)
                              Padding(
                                  padding: const EdgeInsets.only(top: 8),
                                  child: Text(
                                      'Signal flagged: ${linked.flags.join('; ')}',
                                      style: const TextStyle(
                                          color: C.amber, height: 1.4))),
                            if (linked.source
                                .toLowerCase()
                                .contains('synthetic'))
                              const Padding(
                                  padding: EdgeInsets.only(top: 8),
                                  child: Text(
                                      'Demo recording · not field evidence',
                                      style: TextStyle(
                                          color: C.amber,
                                          fontWeight: FontWeight.w600))),
                          ],
                          if (linked == null && _sourceSnapshot != null)
                            Padding(
                                padding: const EdgeInsets.only(top: 10),
                                child: Text(
                                    'Recording removed.${_sourceSnapshot!.toLowerCase().contains('synthetic') ? " Demo source." : ""}',
                                    style: const TextStyle(
                                        color: C.muted,
                                        fontSize: 12,
                                        height: 1.5))),
                          const SizedBox(height: 24),
                          const SectionTitle('Photo'),
                          const SizedBox(height: 8),
                          const Text('On-device · up to 2 MB',
                              style: TextStyle(
                                  color: C.muted, fontSize: 12, height: 1.5)),
                          const SizedBox(height: 12),
                          Wrap(spacing: 8, runSpacing: 8, children: [
                            OutlinedButton.icon(
                                onPressed: () => _pick(ImageSource.camera),
                                icon: const Icon(Icons.camera_alt_outlined),
                                label: const Text('Camera')),
                            OutlinedButton.icon(
                                onPressed: () => _pick(ImageSource.gallery),
                                icon: const Icon(Icons.photo_library_outlined),
                                label: const Text('Gallery')),
                            if (_photo != null)
                              TextButton(
                                  onPressed: () => setState(() {
                                        _photo = null;
                                        _x = _y = null;
                                      }),
                                  child: const Text('Remove photo')),
                          ]),
                          if (_photo != null) ...[
                            const SizedBox(height: 12),
                            _PhotoPin(
                                bytes: _photo!,
                                x: _x,
                                y: _y,
                                onPoint: (x, y) => setState(() {
                                      _x = x;
                                      _y = y;
                                    })),
                            const SizedBox(height: 8),
                            Text(
                                _x == null
                                    ? 'Tap to pin a spot.'
                                    : 'Pin ${(_x! * 100).toStringAsFixed(0)}%, ${(_y! * 100).toStringAsFixed(0)}%',
                                style: const TextStyle(
                                    color: C.muted, fontSize: 12)),
                            if (_x != null)
                              TextButton(
                                  onPressed: () =>
                                      setState(() => _x = _y = null),
                                  child: const Text('Clear pin')),
                          ],
                          const SizedBox(height: 20),
                          if (_error != null)
                            Padding(
                                padding: const EdgeInsets.only(bottom: 16),
                                child: Semantics(
                                    liveRegion: true,
                                    child: Text(_error!,
                                        style: const TextStyle(
                                            color: C.red, height: 1.5)))),
                          if (_busy)
                            const Padding(
                                padding: EdgeInsets.only(bottom: 16),
                                child: LinearProgressIndicator()),
                          SizedBox(
                              width: double.infinity,
                              child: ActionButton(
                                  _busy ? 'Saving…' : 'Save finding',
                                  icon: Icons.check_rounded,
                                  onPressed: _busy ? null : _save)),
                          const ExpansionTile(
                            tilePadding: EdgeInsets.zero,
                            title: Text('Map guide',
                                style: TextStyle(fontSize: 13)),
                            children: [
                              Text(
                                  'Tap a cell to add a finding. Colors show your marks. Warning icons flag audio; S means demo. Add a photo, then tap it to place a pin.',
                                  style: TextStyle(
                                      fontSize: 12,
                                      color: C.muted,
                                      height: 1.5))
                            ],
                          ),
                        ])),
                  ]));
            }),
      ));
}

/// Computes the displayed image rectangle first, so letterbox taps are ignored
/// and both the pin and normalized coordinates always refer to image pixels.
class _PhotoPin extends StatefulWidget {
  const _PhotoPin(
      {required this.bytes,
      required this.x,
      required this.y,
      required this.onPoint});
  final Uint8List bytes;
  final double? x, y;
  final void Function(double, double) onPoint;
  @override
  State<_PhotoPin> createState() => _PhotoPinState();
}

class _PhotoPinState extends State<_PhotoPin> {
  ImageStream? _stream;
  ImageStreamListener? _listener;
  Size? _size;
  String? _error;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _resolve();
  }

  @override
  void didUpdateWidget(_PhotoPin oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!identical(oldWidget.bytes, widget.bytes)) _resolve();
  }

  void _resolve() {
    if (_listener != null) _stream?.removeListener(_listener!);
    _size = null;
    _error = null;
    _stream = MemoryImage(widget.bytes)
        .resolve(createLocalImageConfiguration(context));
    _listener = ImageStreamListener((info, _) {
      final size =
          Size(info.image.width.toDouble(), info.image.height.toDouble());
      info.dispose();
      if (mounted) setState(() => _size = size);
    }, onError: (Object error, StackTrace? stack) {
      if (mounted) {
        setState(() => _error =
            'This photo could not be displayed. Remove it and choose another.');
      }
    });
    _stream!.addListener(_listener!);
  }

  @override
  void dispose() {
    if (_listener != null) _stream?.removeListener(_listener!);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => SizedBox(
        height: 280,
        child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: ColoredBox(
                color: C.pale,
                child: LayoutBuilder(builder: (context, constraints) {
                  if (_error != null) {
                    return Center(
                        child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Text(_error!,
                                style: const TextStyle(color: C.red))));
                  }
                  if (_size == null) {
                    return const Center(child: Text('Loading photo…'));
                  }
                  final viewport =
                      Size(constraints.maxWidth, constraints.maxHeight);
                  final fit = applyBoxFit(BoxFit.contain, _size!, viewport);
                  final rect = Alignment.center
                      .inscribe(fit.destination, Offset.zero & viewport);
                  return Semantics(
                    label:
                        'Annotation photo. Tap the image to place a human location pin.',
                    child: GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTapDown: (details) {
                          final point = details.localPosition;
                          if (!rect.contains(point)) return;
                          widget.onPoint(
                              ((point.dx - rect.left) / rect.width)
                                  .clamp(0.0, 1.0)
                                  .toDouble(),
                              ((point.dy - rect.top) / rect.height)
                                  .clamp(0.0, 1.0)
                                  .toDouble());
                        },
                        child: Stack(children: [
                          Positioned.fromRect(
                              rect: rect,
                              child: Image.memory(widget.bytes,
                                  fit: BoxFit.fill,
                                  gaplessPlayback: true,
                                  excludeFromSemantics: true)),
                          if (widget.x != null)
                            Positioned(
                                left: rect.left + rect.width * widget.x! - 14,
                                top: rect.top + rect.height * widget.y! - 14,
                                child: IgnorePointer(
                                    child: Container(
                                        width: 28,
                                        height: 28,
                                        decoration: BoxDecoration(
                                            color: C.blue,
                                            shape: BoxShape.circle,
                                            border: Border.all(
                                                color: Colors.white, width: 3)),
                                        child: const Icon(Icons.add,
                                            size: 18, color: Colors.white)))),
                        ])),
                  );
                }))),
      );
}
