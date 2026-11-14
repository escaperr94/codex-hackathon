import 'dart:math' as math;
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';

import 'package:thud/learned_calibration.dart';

void main() {
  test('learned calibration predictions, persistence, and invalid input boundaries', () {
  final neural = List<double>.filled(1024, 0)..[0] = 1;
  final physicalA = LearnedCalibration.features(neural, peakHz: 500, decayMs: 100,
      centroidHz: 1000, flatness: .1, damping: .01);
  final physicalB = LearnedCalibration.features(neural, peakHz: 1500, decayMs: 500,
      centroidHz: 2000, flatness: .2, damping: .1);
  check(physicalA.length == 1032 && physicalA[0] == 1, 'fused contract preserves neural block');
  check(physicalA.skip(1024).toList().toString() != physicalB.skip(1024).toList().toString(),
      'physical resonance and damping contribute to model inputs');
  check(physicalA.skip(1024).fold<double>(0, (sum, v) => sum + v * v) <= .010000001,
      'secondary DSP block cannot dominate neural unit norm');
  final reference = LearnedCalibration.fitReference([
    embedding(-0.1), embedding(0), embedding(0.1),
  ]);
  check(!reference.predict(embedding(0.05)).different, 'near reference');
  check(reference.predict(embedding(2)).different, 'separated reference');
  check(reference.threshold > 0, 'positive data-derived threshold');
  final narrow = LearnedCalibration.fitReference([
    embedding(-0.01), embedding(0), embedding(0.01),
  ]);
  check(narrow.threshold < reference.threshold, 'threshold follows training spread');
  final identical = LearnedCalibration.fitReference([
    embedding(1), embedding(1), embedding(1),
  ]);
  check(!identical.predict(embedding(1)).different, 'identical reference stability');
  check(identical.threshold > 0, 'numerical floor');

  final fill = LearnedCalibration.fitFill(
    [embedding(0), embedding(1), embedding(2)], [0, 50, 100],
  );
  final middle = fill.predict(embedding(0.5));
  check(middle.supported && (middle.value - 25).abs() < 0.01,
      'supported fill interpolation');
  check(middle.lower < middle.value && middle.upper > middle.value,
      'fill uncertainty contains prediction');
  check(middle.calibrationStatus.contains('experimental'), 'uncertainty is not validated coverage');
  check(LearnedCalibration.aggregate([embedding(1), embedding(1.01), embedding(100)])[0] == 1.01,
      'median aggregation resists one outlier');
  final tileVectors = <List<double>>[];
  final tileLabels = <String>[];
  final tileIds = <String>[];
  for (var tile = 0; tile < 4; tile++) {
    for (var tap = -1; tap <= 1; tap++) {
      tileVectors.add(direction((tile < 2 ? 0 : 1.57) + tap * .01));
      tileLabels.add(tile < 2 ? 'Supported' : 'Suspected Unsupported');
      tileIds.add('tile-$tile');
    }
  }
  final holdout = LearnedCalibration.evaluateTiles(tileVectors, tileLabels, tileIds);
  check(holdout['folds'] == 4 && holdout['tested'] == 4 && holdout['correct'] == 4,
      'whole tile folds aggregate held-out taps without leakage');
  final insufficient = LearnedCalibration.evaluateTiles(
      tileVectors.take(6).toList(), tileLabels.take(6).toList(), tileIds.take(6).toList());
  check(insufficient['status'] == 'insufficient independent tiles', 'no invented tile benchmark');
  for (final x in [0.0, 1.0, 2.0]) {
    check(fill.predict(embedding(x)).supported, 'training sample supported');
  }
  final above = fill.predict(embedding(3));
  check(!above.supported && above.value == 100, 'extrapolation rejected and bounded');
  final below = fill.predict(embedding(-1));
  check(!below.supported && below.value == 0, 'negative extrapolation rejected');
  final offManifold = embedding(1)..[1] = 100;
  check(!fill.predict(offManifold).supported, 'orthogonal OOD rejected');
  final repeated = LearnedCalibration.fitFill([
    embedding(0), embedding(0), embedding(1), embedding(2),
  ], [0, 0, 50, 100]);
  check(repeated.predict(embedding(1)).supported, 'duplicate same-level references');

  final decodedReference = ReferenceCalibration.fromJson(roundtrip(reference.toJson()));
  final decodedFill = FillCalibration.fromJson(roundtrip(fill.toJson()));
  check(decodedReference.predict(embedding(2)).different, 'reference JSON roundtrip');
  final decodedMiddle = decodedFill.predict(embedding(0.5));
  check(decodedMiddle.supported && (decodedMiddle.value - middle.value).abs() < 1e-10,
      'fill JSON roundtrip');
  final mutable = [embedding(0), embedding(1), embedding(2)];
  final copied = LearnedCalibration.fitFill(mutable, [0, 50, 100]);
  mutable[0][0] = 99;
  final exported = copied.toJson();
  (exported['center'] as List<double>)[0] = 99;
  check((copied.predict(embedding(1)).value - 50).abs() < 0.01,
      'fitted parameters do not alias caller data');

  final classes = LearnedCalibration.fitClasses([
    direction(0), direction(0.01), direction(-0.01),
    direction(1.56), direction(1.57), direction(1.58),
  ], ['Unripe', 'Unripe', 'Unripe', 'Ripe', 'Ripe', 'Ripe']);
  final ripe = classes.predict(direction(1.565));
  check(ripe.label == 'Ripe' && ripe.supported, 'supervised fruit class');
  check(!classes.predict(direction(3)).supported, 'class OOD rejected');
  final restoredClasses = ClassCalibration.fromJson(roundtrip(classes.toJson()));
  check(restoredClasses.predict(direction(0.005)).label == 'Unripe' &&
      restoredClasses.predict(direction(0.005)).supported, 'class JSON roundtrip');
  rejects(() => LearnedCalibration.fitClasses([
    direction(0), direction(0.01), direction(-0.01), direction(1.57),
  ], ['Unripe', 'Unripe', 'Unripe', 'Ripe']), 'three samples per class');
  rejects(() => LearnedCalibration.fitClasses([
    direction(0), direction(0.01), direction(-0.01),
  ], ['Ripe', 'Ripe', 'Ripe']), 'two distinct classes');
  rejects(() => classes.predict(embedding(0)), 'zero normalized class input');
  rejects(() => ClassCalibration.fromJson({
    ...classes.toJson(), 'kind': 'reference',
  }), 'class provenance');

  rejects(() => LearnedCalibration.fitReference([embedding(0)]), 'too few samples');
  rejects(() => LearnedCalibration.fitReference([
    embedding(0), embedding(1), [1, 2],
  ]), 'dimension validation');
  rejects(() => reference.predict(embedding(double.nan)), 'NaN prediction');
  rejects(() => fill.predict(embedding(double.infinity)), 'infinite prediction');
  rejects(() => LearnedCalibration.fitFill([
    embedding(0), embedding(1), embedding(2),
  ], [0, 100]), 'label count mismatch');
  rejects(() => LearnedCalibration.fitFill([
    embedding(0), embedding(1), embedding(2),
  ], [0, 0, 100]), 'three distinct levels required');
  rejects(() => LearnedCalibration.fitFill([
    embedding(0), embedding(1), embedding(2),
  ], [0, 50, double.nan]), 'nonfinite labels');
  rejects(() => LearnedCalibration.fitFill([
    embedding(0), embedding(1), embedding(2),
  ], [-1, 50, 100]), 'percentage bounds');
  rejects(() => LearnedCalibration.fitFill([
    embedding(1), embedding(1), embedding(1),
  ], [0, 50, 100]), 'contradictory labels');
  rejects(() => ReferenceCalibration.fromJson({
    ...reference.toJson(), 'version': 'old',
  }), 'calibration provenance');
  rejects(() => FillCalibration.fromJson({
    ...fill.toJson(), 'embeddingVersion': 'other-encoder',
  }), 'encoder provenance');
  rejects(() => FillCalibration.fromJson({
    ...fill.toJson(), 'weights': [double.nan, 0, 0],
  }), 'nonfinite stored coefficients');
  rejects(() => FillCalibration.fromJson({
    ...fill.toJson(), 'maxLabel': 101,
  }), 'stored label bounds');
  rejects(() => ReferenceCalibration.fromJson({
    ...reference.toJson(), 'center': [1],
  }), 'stored dimensions');
  });
}

List<double> embedding(double value) => List<double>.filled(1032, 0)..[0] = value;

List<double> direction(double angle) => List<double>.filled(1032, 0)
  ..[0] = math.cos(angle)
  ..[1] = math.sin(angle);

Map<String, dynamic> roundtrip(Map<String, dynamic> value) =>
    jsonDecode(jsonEncode(value)) as Map<String, dynamic>;

void check(bool condition, String name) {
  if (!condition) throw StateError('Failed: $name');
}

void rejects(void Function() action, String name) {
  try {
    action();
  } on ArgumentError {
    return;
  } on FormatException {
    return;
  }
  throw StateError('Expected rejection: $name');
}
