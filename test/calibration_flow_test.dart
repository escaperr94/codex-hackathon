import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:thud/store.dart';
import 'package:thud/sensing_flow.dart';
import 'package:thud/sensing_result.dart';
import 'package:thud/learned_calibration.dart';

List<double> vector(double x) => List<double>.filled(1024, 0)..[0] = x;
Measurement tap(String id, double x, {bool demo = false}) => Measurement(
  id: id, label: 'capture', source: demo ? 'Synthetic' : 'Microphone',
  created: DateTime(2026), module: Module.scope, peakHz: 800,
  hiiDb: 1, decayMs: 100, rmsDb: -10, snrDb: 40,
  waveform: const [.1], spectrum: const [.2], flags: const [],
).withEmbedding(vector(x));
ReferenceAnchor reference(SensingIntent intent, String name, double x,
    {double? level, String? label, bool demo = false}) => ReferenceAnchor(
  intent, name, level,
  [tap('$name-a-$x', x - .01, demo: demo), tap('$name-b-$x', x, demo: demo), tap('$name-c-$x', x + .01, demo: demo)],
  classLabel: label,
);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({}));

  test('reference model persists without conflating demo or object identity', () async {
    await AnchorRepository.save(reference(SensingIntent.slab, 'tile', 1));
    await AnchorRepository.save(reference(SensingIntent.slab, 'tile', 10, demo: true));
    final raw = await AnchorRepository.model(SensingIntent.slab, 'tile', false);
    final fitted = ReferenceCalibration.fromJson(raw!);
    expect(fitted.predict(vector(1)).different, false);
    expect(fitted.predict(vector(10)).different, true);
    expect(await AnchorRepository.model(SensingIntent.slab, 'other tile', false), isNull);
    final loaded = await AnchorRepository.load();
    expect(loaded.every((a) => a.taps.every((t) => t.hasEmbedding)), true);
    final changed = loaded.first.taps.first.withContext({'referenceObject': 'tile'}).withIntent('slab');
    expect(Measurement.fromJson(jsonDecode(jsonEncode(changed.toJson()))).hasEmbedding, true);
  });

  test('known-fill training publishes only after enough levels and rejects OOD', () async {
    for (final (level, x) in [(0.0, 0.0), (50.0, 1.0)]) {
      await AnchorRepository.save(reference(SensingIntent.vessel, 'bottle', x, level: level));
    }
    expect(await AnchorRepository.model(SensingIntent.vessel, 'bottle', false), isNull);
    await AnchorRepository.save(reference(SensingIntent.vessel, 'bottle', 2, level: 100));
    final raw = await AnchorRepository.model(SensingIntent.vessel, 'bottle', false);
    final model = FillCalibration.fromJson(raw!);
    expect(model.predict(vector(.5)).value, closeTo(25, .1));
    expect(model.predict(vector(.5)).supported, true);
    expect(model.predict(vector(10)).supported, false);
  });

  test('legacy traces stay readable but cannot masquerade as neural calibration', () async {
    final legacy = reference(SensingIntent.slab, 'old', 1).toJson();
    for (final t in legacy['taps'] as List) {
      (t as Map).remove('embedding');
      t.remove('encoderVersion');
    }
    SharedPreferences.setMockInitialValues({'thud.anchors.v1': jsonEncode([legacy])});
    expect((await AnchorRepository.load()).single.taps.first.hasEmbedding, false);
    expect(await AnchorRepository.model(SensingIntent.slab, 'old', false), isNull);
    await AnchorRepository.save(reference(SensingIntent.slab, 'new', 1));
    expect((await AnchorRepository.load()).map((a) => a.object), containsAll(['old', 'new']));
  });

  testWidgets('learned embeddings, not identical acoustic metrics, determine verdict', (tester) async {
    await AnchorRepository.save(reference(SensingIntent.materials, 'sample', 1));
    final store = ThudStore();
    await tester.pumpWidget(MaterialApp(home: Scaffold(body: SingleChildScrollView(child: SensingResult(
      measurement: tap('query', 4), intent: SensingIntent.materials, store: store,
      onSave: (_) {}, onMap: (_) {}, onTrace: (_) {}, onAgain: () {}, saved: false,
    )))));
    await tester.pumpAndSettle();
    expect(find.text('Different from learned reference'), findsOneWidget);
    expect(find.text('Model hint'), findsNothing);
    store.dispose();
  });
}
