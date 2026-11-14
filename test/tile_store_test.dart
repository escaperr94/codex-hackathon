import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:thud/store.dart';
import 'package:thud/tile_geometry.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('tile removal requires explicit unlink and persists remaining finding evidence', () async {
    SharedPreferences.setMockInitialValues({});
    final store = ThudStore();
    await store.load();
    final layout = TileLayout.empty().grid([0, .5, 1], [0, 1]);
    final project = Project(id: 'p', name: 'Inspection', room: 'Kitchen', rows: 1, columns: 2,
      floorPlan: FloorPlan(imageBase64: '', name: 'floor', width: 100, height: 100, tileLayout: layout));
    store.projects.add(project);
    await store.upsertAnnotation(DefectAnnotation(id: 'a', projectId: 'p', cell: 0,
      status: AnnotationStatus.review, tileId: layout.tiles.first.id, note: 'Check support',
      planX: .25, planY: .5));
    await store.updateProjectLocation(project, building: 'A', floor: '2', unit: '203');
    final reduced = layout.remove(layout.tiles.first.id);
    await expectLater(store.updateTileLayout(project, reduced), throwsException);
    expect(project.floorPlan!.tileLayout!.tiles.map((t) => t.id), ['T1', 'T2']);
    expect(store.annotations.single.tileId, 'T1');
    await store.updateTileLayout(project, reduced, resetTileLinks: true);
    final restored = ThudStore();
    await restored.load();
    expect(restored.annotations.single.tileId, isNull);
    expect(restored.annotations.single.note, 'Check support');
    expect(restored.annotations.single.planX, .25);
    expect(restored.projects.single.floorPlan!.tileLayout!.tiles.single.id, 'T2');
    expect(restored.projects.single.unit, '203');
    expect(restored.reportText(restored.projects.single), contains('Building: A'));
    store.dispose(); restored.dispose();
  });
}
