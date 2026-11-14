import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:thud/store.dart';
import 'package:thud/sensing_flow.dart';
import 'package:thud/sensing_result.dart';

List<double> vector(double x) => List<double>.filled(1024, 0)..[0] = 1..[1] = x * .001;
Measurement tap(String id, double x, {bool demo = false}) => Measurement(
  id: id, label: 'capture', source: demo ? 'Synthetic' : 'Microphone',
  created: DateTime(2026), module: Module.scope, peakHz: 800,
  hiiDb: 1, decayMs: 100, rmsDb: -10, snrDb: 40,
  waveform: const [.1], spectrum: const [.2], flags: const [],
).withEmbedding(vector(x));
Measurement directionalTap(String id, double angle) => tap(id, 1).withEmbedding(
  List<double>.filled(1024, 0)..[0] = math.cos(angle)..[1] = math.sin(angle));
ReferenceAnchor reference(SensingIntent intent, String name, double x,
    {double? level, String? label, bool demo = false, Map<String, String> metadata = const {}}) => ReferenceAnchor(
  intent, name, level,
  [tap('$name-a-$x', x - .01, demo: demo), tap('$name-b-$x', x, demo: demo), tap('$name-c-$x', x + .01, demo: demo)],
  label: label, metadata: metadata,
);
ReferenceAnchor directional(SensingIntent intent, String label, double angle,
    {String? tile}) => ReferenceAnchor(intent, 'one type', null,
  [for (var i = -1; i <= 1; i++) directionalTap('$label-$tile-$i', angle + i * .01)],
  label: label, metadata: {if (tile != null) 'specimenId': tile});

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  setUp(() => SharedPreferences.setMockInitialValues({}));

  test('reference library persists metadata, separates demo and deletes selected profiles', () async {
    await AnchorRepository.save(reference(SensingIntent.slab, 'tile', 1, label: 'Supported', metadata: {'specimenId': 'A'}));
    await AnchorRepository.save(reference(SensingIntent.slab, 'tile', 10, label: 'Supported', demo: true));
    final loaded = await AnchorRepository.load();
    expect(loaded.length, 2);
    expect(loaded.first.metadata['specimenId'], 'A');
    final changed = loaded.first.taps.first.withContext({'referenceObject': 'tile'}).withIntent('slab');
    expect(Measurement.fromJson(jsonDecode(jsonEncode(changed.toJson()))).hasEmbedding, true);
    await AnchorRepository.delete(loaded.first);
    expect((await AnchorRepository.load()).single.synthetic, true);
  });

  test('fill requires three levels, publishes uncertainty and ml only with consistent capacity', () {
    final refs = [for (final (level, x) in [(0.0, 0.0), (50.0, 1.0), (100.0, 2.0)])
      reference(SensingIntent.vessel, 'bottle', x, level: level, metadata: {'capacityMl': '600'})];
    expect(SensingInterpretation.evaluate(SensingIntent.vessel, [tap('q', .5)], refs.take(2).toList())['estimatedFillPercent'], isNull);
    final result = SensingInterpretation.evaluate(SensingIntent.vessel, [tap('q', .5)], refs);
    expect(double.parse(result['estimatedFillPercent']!), closeTo(25, .1));
    expect(double.parse(result['estimatedVolumeMl']!), closeTo(150, 1));
    expect(double.parse(result['fillLowerPercent']!), lessThan(25));
    expect(double.parse(result['fillUpperPercent']!), greaterThan(25));
    final noCapacity = refs.map((a) => ReferenceAnchor(a.intent, a.object, a.level, a.taps)).toList();
    expect(SensingInterpretation.evaluate(SensingIntent.vessel, [tap('q', .5)], noCapacity)['estimatedVolumeMl'], isNull);
    expect(SensingInterpretation.evaluate(SensingIntent.vessel, [tap('q', 10)], refs)['estimatedFillPercent'], isNull);
  });

  test('legacy signal references stay readable but import requires compatible embeddings', () async {
    final legacy = reference(SensingIntent.slab, 'old', 1).toJson();
    for (final t in legacy['taps'] as List) { (t as Map).remove('embedding'); t.remove('encoderVersion'); }
    SharedPreferences.setMockInitialValues({'thud.anchors.v1': jsonEncode([legacy])});
    expect((await AnchorRepository.load()).single.usable, false);
    expect(() => AnchorRepository.decodeImport(jsonEncode([legacy])), throwsFormatException);
    await AnchorRepository.save(reference(SensingIntent.slab, 'new', 1));
    expect((await AnchorRepository.load()).map((a) => a.object), containsAll(['old', 'new']));
  });

  test('slab labelled classes and normal-only fallback do not certify defects', () {
    final refs = [directional(SensingIntent.slab, 'Supported', 0), directional(SensingIntent.slab, 'Suspected Unsupported', 1.57)];
    final labelled = SensingInterpretation.evaluate(SensingIntent.slab, [directionalTap('q', 1.57)], refs);
    expect(labelled['interpretation'], 'Suspected Unsupported');
    expect(labelled['nextAction'], contains('Not certified'));
    final normal = SensingInterpretation.evaluate(SensingIntent.slab, [directionalTap('q', 0)], refs.take(1).toList());
    expect(normal['interpretation'], 'Inconclusive');
    final unknown = SensingInterpretation.evaluate(SensingIntent.slab, [directionalTap('q', 3)], refs);
    expect(unknown['interpretation'], 'Inconclusive');
  });

  test('fruit and materials use neural similarity with unknown rejection', () {
    final fruit = SensingInterpretation.evaluate(SensingIntent.fruit, [directionalTap('q', 0)],
      [directional(SensingIntent.fruit, 'Firmer', 0), directional(SensingIntent.fruit, 'Softer', 1.57)]);
    expect(fruit['interpretation'], 'Firmer reference similarity');
    expect(SensingInterpretation.evaluate(SensingIntent.fruit, [directionalTap('q', 0)],
      [directional(SensingIntent.fruit, 'Firmer', 0),
       ReferenceAnchor(SensingIntent.fruit, 'different cultivar', null,
         directional(SensingIntent.fruit, 'Softer', 1.57).taps, label: 'Softer')])['modelStatus'],
      'calibration required');
    expect(fruit['nextAction'], contains('Not exact maturity'));
    final material = SensingInterpretation.evaluate(SensingIntent.materials, [directionalTap('q', 3)],
      [directional(SensingIntent.materials, 'wood', 0), directional(SensingIntent.materials, 'metal', 1.57)]);
    expect(material['interpretation'], 'Unknown material signature');
  });

  test('repeat query aggregation rejects one outlier and keeps provenance', () {
    final refs = [reference(SensingIntent.devices, 'motor', 1, label: 'Normal')];
    final result = SensingInterpretation.evaluate(SensingIntent.devices,
      [tap('a', 1), tap('b', 1.001), tap('outlier', 20)], refs);
    expect(result['interpretation'], 'Similar to normal recordings');
    expect(result['queryCount'], '3');
    expect(result['queryTapIds'], contains('outlier'));
    expect(result['nextAction'], contains('does not identify a mechanical failure'));
  });

  test('operating and casing-tap references cannot be mixed', () {
    final q = tap('q', 1).withContext({'captureMode': 'operating'});
    final result = SensingInterpretation.evaluate(SensingIntent.devices, [q],
      [reference(SensingIntent.devices, 'motor', 1, label: 'Normal')]);
    expect(result['modelStatus'], 'calibration required');
  });

  test('reference rejection distinguishes model failure from acoustic rejection', () {
    final encoded = tap('encoded', 1).withContext({'captureUsable': 'false'});
    final rejected = Measurement(
      id: encoded.id, label: encoded.label, source: encoded.source,
      created: encoded.created, module: encoded.module, peakHz: encoded.peakHz,
      hiiDb: encoded.hiiDb, decayMs: encoded.decayMs, rmsDb: encoded.rmsDb,
      snrDb: encoded.snrDb, waveform: encoded.waveform, spectrum: encoded.spectrum,
      flags: const ['Reject: low impact signal-to-noise ratio'],
      embedding: encoded.embedding, encoderVersion: encoded.encoderVersion);
    expect(referenceRecordingIssue(rejected), contains('rejected for reference training'));
    final missing = Measurement(
      id: 'missing', label: 'capture', source: 'Microphone', created: DateTime(2026),
      module: Module.scope, peakHz: 800, hiiDb: 1, decayMs: 100, rmsDb: -10,
      snrDb: 40, waveform: const [.1], spectrum: const [.2], flags: const []);
    expect(referenceRecordingIssue(missing), contains('neural model did not return'));
  });

  testWidgets('neural embeddings, not identical DSP metrics, determine result', (tester) async {
    await AnchorRepository.save(reference(SensingIntent.materials, 'sample', 1));
    final store = ThudStore();
    await tester.pumpWidget(MaterialApp(home: Scaffold(body: SingleChildScrollView(child: SensingResult(
      measurement: tap('query', 4), intent: SensingIntent.materials, store: store,
      onSave: (_) {}, onMap: (_) {}, onTrace: (_) {}, onAgain: () {}, saved: false,
    )))));
    await tester.pumpAndSettle();
    expect(find.text('Unknown material signature'), findsOneWidget);
    expect(find.text('Possible defect'), findsNothing);
    store.dispose();
  });
}
