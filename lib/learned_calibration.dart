import 'dart:math' as math;

const calibrationVersion = 'yamnet-calibration-v1';
const _embeddingVersion = 'yamnet-waveform-embedding-f32-v1';
const _dimensions = 1024;

class ReferencePrediction {
  final double distance;
  final double threshold;
  final bool different;

  const ReferencePrediction(this.distance, this.threshold, this.different);
}

class FillPrediction {
  final double value;
  final bool supported;

  const FillPrediction(this.value, this.supported);
}

class ClassPrediction {
  final String label;
  final bool supported;
  final double distance;

  const ClassPrediction(this.label, this.supported, this.distance);
}

/// Fits only embedding-space models, not physical diagnoses.
class LearnedCalibration {
  static ClassCalibration fitClasses(
    List<List<double>> embeddings,
    List<String> labels,
  ) {
    final samples = _samples(embeddings);
    if (labels.length != samples.length ||
        labels.any((label) => label.trim().isEmpty || label != label.trim()) ||
        labels.toSet().length < 2) {
      throw ArgumentError('Classes require matching labels and >=2 named classes.');
    }
    final groups = <String, List<List<double>>>{};
    for (var i = 0; i < labels.length; i++) {
      groups.putIfAbsent(labels[i], () => []).add(_normalized(samples[i]));
    }
    final centers = <String, List<double>>{};
    final thresholds = <String, double>{};
    for (final entry in groups.entries) {
      final reference = fitReference(entry.value);
      centers[entry.key] = reference._center;
      thresholds[entry.key] = reference.threshold;
    }
    return ClassCalibration._(centers, thresholds);
  }

  static ReferenceCalibration fitReference(List<List<double>> embeddings) {
    final samples = _samples(embeddings);
    final center = _center(samples);
    var threshold = _floor(center);
    // Distance from each sample to the centroid fitted without that sample.
    for (final sample in samples) {
      threshold = math.max(
        threshold,
        _distance(sample, center) * samples.length / (samples.length - 1),
      );
    }
    return ReferenceCalibration._(center, threshold);
  }

  static FillCalibration fitFill(
    List<List<double>> embeddings,
    List<double> labels,
  ) {
    final samples = _samples(embeddings);
    if (labels.length != samples.length ||
        labels.any((v) => !v.isFinite || v < 0 || v > 100) ||
        labels.toSet().length < 3) {
      throw ArgumentError('Fill requires matching labels and >=3 distinct percentages.');
    }
    for (var i = 0; i < samples.length; i++) {
      for (var j = 0; j < i; j++) {
        if (_distance(samples[i], samples[j]) == 0 && labels[i] != labels[j]) {
          throw ArgumentError('Identical embeddings cannot have different labels.');
        }
      }
    }
    final center = _center(samples);
    final centered = [for (final s in samples) _subtract(s, center)];
    final variance = centered.fold<double>(0, (sum, s) => sum + _dot(s, s)) /
        samples.length;
    if (!variance.isFinite || variance <= _floor(center) * _floor(center)) {
      throw ArgumentError('Fill embeddings must contain measurable variation.');
    }
    final meanLabel = labels.reduce((a, b) => a + b) / labels.length;
    // Scale-relative ridge keeps the solve conditioned without acoustic cutoffs.
    final ridge = variance * 1e-6;
    final gram = List.generate(samples.length, (i) => List.generate(
        samples.length, (j) => _dot(centered[i], centered[j]) + (i == j ? ridge : 0)));
    final weights = _solvePositiveDefinite(
      gram,
      [for (final label in labels) label - meanLabel],
    );
    var supportRadius = _floor(center);
    for (var i = 0; i < samples.length; i++) {
      var nearest = double.infinity;
      for (var j = 0; j < samples.length; j++) {
        if (i != j && labels[i] != labels[j]) {
          nearest = math.min(nearest, _distance(samples[i], samples[j]));
        }
      }
      supportRadius = math.max(supportRadius, nearest);
    }
    return FillCalibration._(
      samples,
      center,
      weights,
      meanLabel,
      labels.reduce(math.min),
      labels.reduce(math.max),
      supportRadius,
      ridge,
    );
  }
}

class ReferenceCalibration {
  final List<double> _center;
  final double threshold;

  ReferenceCalibration._(this._center, this.threshold);

  ReferencePrediction predict(List<double> embedding) {
    _validateEmbedding(embedding);
    final distance = _distance(embedding, _center);
    return ReferencePrediction(distance, threshold, distance > threshold);
  }

  Map<String, dynamic> toJson() => {
        ..._provenance('reference'),
        'center': List<double>.of(_center),
        'threshold': threshold,
      };

  factory ReferenceCalibration.fromJson(Map<String, dynamic> json) {
    _validateProvenance(json, 'reference');
    final center = _vector(json['center']);
    final threshold = _number(json['threshold']);
    if (threshold < _floor(center)) {
      throw const FormatException('Invalid reference threshold.');
    }
    return ReferenceCalibration._(center, threshold);
  }
}

class FillCalibration {
  final List<List<double>> _samples;
  final List<double> _center;
  final List<double> _weights;
  final double _meanLabel;
  final double _minLabel;
  final double _maxLabel;
  final double _supportRadius;
  final double _ridge;

  FillCalibration._(this._samples, this._center, this._weights, this._meanLabel,
      this._minLabel, this._maxLabel, this._supportRadius, this._ridge);

  FillPrediction predict(List<double> embedding) {
    _validateEmbedding(embedding);
    final centered = _subtract(embedding, _center);
    final centerProduct = _dot(centered, _center);
    var raw = _meanLabel;
    var nearest = double.infinity;
    for (var i = 0; i < _samples.length; i++) {
      raw += _weights[i] * (_dot(centered, _samples[i]) - centerProduct);
      nearest = math.min(nearest, _distance(embedding, _samples[i]));
    }
    if (!raw.isFinite) {
      throw StateError('Fill prediction overflowed.');
    }
    // A bounded display value is not evidence of support outside calibration.
    return FillPrediction(
      raw.clamp(_minLabel, _maxLabel).toDouble(),
      nearest <= _supportRadius && raw >= _minLabel && raw <= _maxLabel,
    );
  }

  Map<String, dynamic> toJson() => {
        ..._provenance('fill'),
        'samples': [for (final sample in _samples) List<double>.of(sample)],
        'center': List<double>.of(_center),
        'weights': List<double>.of(_weights),
        'meanLabel': _meanLabel,
        'minLabel': _minLabel,
        'maxLabel': _maxLabel,
        'supportRadius': _supportRadius,
        'ridge': _ridge,
      };

  factory FillCalibration.fromJson(Map<String, dynamic> json) {
    _validateProvenance(json, 'fill');
    final rawSamples = json['samples'];
    final rawWeights = json['weights'];
    if (rawSamples is! List || rawSamples.length < 3 ||
        rawWeights is! List || rawWeights.length != rawSamples.length) {
      throw const FormatException('Invalid fill sample/weight counts.');
    }
    final samples = [for (final s in rawSamples) _vector(s)];
    final center = _vector(json['center']);
    final weights = [for (final w in rawWeights) _number(w)];
    final meanLabel = _number(json['meanLabel']);
    final minLabel = _number(json['minLabel']);
    final maxLabel = _number(json['maxLabel']);
    final radius = _number(json['supportRadius']);
    final ridge = _number(json['ridge']);
    if (minLabel < 0 || maxLabel > 100 || minLabel >= maxLabel ||
        meanLabel < minLabel || meanLabel > maxLabel ||
        radius < _floor(center) || ridge <= 0) {
      throw const FormatException('Invalid fitted fill parameters.');
    }
    return FillCalibration._(samples, center, weights, meanLabel, minLabel,
        maxLabel, radius, ridge);
  }
}

class ClassCalibration {
  final Map<String, List<double>> _centers;
  final Map<String, double> _thresholds;

  ClassCalibration._(this._centers, this._thresholds);

  ClassPrediction predict(List<double> embedding) {
    final normalized = _normalized(embedding);
    var label = _centers.keys.first;
    var distance = double.infinity;
    var second = double.infinity;
    for (final entry in _centers.entries) {
      final candidate = _distance(normalized, entry.value);
      if (candidate < distance) {
        second = distance;
        label = entry.key;
        distance = candidate;
      } else {
        second = math.min(second, candidate);
      }
    }
    return ClassPrediction(label,
        distance <= _thresholds[label]! && second - distance > _floor(normalized),
        distance);
  }

  Map<String, dynamic> toJson() => {
        ..._provenance('classes'),
        'centers': {for (final e in _centers.entries) e.key: List<double>.of(e.value)},
        'thresholds': Map<String, double>.of(_thresholds),
      };

  factory ClassCalibration.fromJson(Map<String, dynamic> json) {
    _validateProvenance(json, 'classes');
    final rawCenters = json['centers'];
    final rawThresholds = json['thresholds'];
    if (rawCenters is! Map || rawThresholds is! Map || rawCenters.length < 2 ||
        rawCenters.length != rawThresholds.length) {
      throw const FormatException('Invalid class calibration.');
    }
    final centers = <String, List<double>>{};
    final thresholds = <String, double>{};
    for (final key in rawCenters.keys) {
      if (key is! String || key.trim().isEmpty || key != key.trim()) {
        throw const FormatException('Invalid class label.');
      }
      final center = _vector(rawCenters[key]);
      final threshold = _number(rawThresholds[key]);
      if (threshold < _floor(center) || _dot(center, center) > 1 / _dimensions + 1e-12) {
        throw const FormatException('Invalid class center or support threshold.');
      }
      centers[key] = center;
      thresholds[key] = threshold;
    }
    return ClassCalibration._(centers, thresholds);
  }
}

List<double> _normalized(List<double> embedding) {
  _validateEmbedding(embedding);
  final norm = math.sqrt(_dot(embedding, embedding)) * math.sqrt(_dimensions);
  if (norm <= 0 || !norm.isFinite) {
    throw ArgumentError('Class embeddings require a nonzero finite norm.');
  }
  return [for (final value in embedding) value / norm];
}

Map<String, dynamic> _provenance(String kind) => {
      'version': calibrationVersion,
      'embeddingVersion': _embeddingVersion,
      'dimensions': _dimensions,
      'kind': kind,
    };

void _validateProvenance(Map<String, dynamic> json, String kind) {
  if (json['version'] != calibrationVersion ||
      json['embeddingVersion'] != _embeddingVersion ||
      json['dimensions'] != _dimensions || json['kind'] != kind) {
    throw const FormatException('Incompatible calibration provenance.');
  }
}

List<List<double>> _samples(List<List<double>> embeddings) {
  if (embeddings.length < 3) {
    throw ArgumentError('Calibration requires at least three embeddings.');
  }
  for (final embedding in embeddings) {
    _validateEmbedding(embedding);
  }
  return [for (final embedding in embeddings) List<double>.of(embedding)];
}

void _validateEmbedding(List<double> embedding) {
  if (embedding.length != _dimensions || embedding.any((v) => !v.isFinite)) {
    throw ArgumentError('Expected 1024 finite embedding values.');
  }
  // Reject finite inputs whose squared norm would overflow subsequent arithmetic.
  if (!_dot(embedding, embedding).isFinite) {
    throw ArgumentError('Embedding magnitude exceeds numeric range.');
  }
}

List<double> _vector(dynamic value) {
  if (value is! List || value.length != _dimensions) {
    throw const FormatException('Expected a 1024-dimensional vector.');
  }
  final vector = [for (final v in value) _number(v)];
  if (!_dot(vector, vector).isFinite) {
    throw const FormatException('Embedding magnitude exceeds numeric range.');
  }
  return vector;
}

double _number(dynamic value) {
  if (value is! num || !value.toDouble().isFinite) {
    throw const FormatException('Expected a finite number.');
  }
  return value.toDouble();
}

List<double> _center(List<List<double>> samples) => List.generate(
    _dimensions,
    (i) => samples.fold<double>(0, (sum, s) => sum + s[i] / samples.length));

List<double> _subtract(List<double> a, List<double> b) =>
    List.generate(_dimensions, (i) => a[i] - b[i]);

double _dot(List<double> a, List<double> b) {
  var sum = 0.0;
  for (var i = 0; i < _dimensions; i++) {
    sum += a[i] * b[i] / _dimensions;
  }
  return sum;
}

double _distance(List<double> a, List<double> b) {
  var sum = 0.0;
  for (var i = 0; i < _dimensions; i++) {
    final delta = a[i] - b[i];
    sum += delta * delta / _dimensions;
  }
  return math.sqrt(sum);
}

double _floor(List<double> center) =>
    1e-9 * math.max(1.0, math.sqrt(_dot(center, center)));

List<double> _solvePositiveDefinite(List<List<double>> matrix, List<double> y) {
  final n = y.length;
  final lower = List.generate(n, (_) => List<double>.filled(n, 0));
  for (var i = 0; i < n; i++) {
    for (var j = 0; j <= i; j++) {
      var value = matrix[i][j];
      for (var k = 0; k < j; k++) {
        value -= lower[i][k] * lower[j][k];
      }
      if (i == j) {
        if (!value.isFinite || value <= 0) {
          throw ArgumentError('Calibration is numerically ill-conditioned.');
        }
        lower[i][j] = math.sqrt(value);
      } else {
        lower[i][j] = value / lower[j][j];
      }
    }
  }
  final result = List<double>.of(y);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < i; j++) {
      result[i] -= lower[i][j] * result[j];
    }
    result[i] /= lower[i][i];
  }
  for (var i = n - 1; i >= 0; i--) {
    for (var j = i + 1; j < n; j++) {
      result[i] -= lower[j][i] * result[j];
    }
    result[i] /= lower[i][i];
  }
  if (result.any((v) => !v.isFinite)) {
    throw ArgumentError('Calibration coefficients exceed numeric range.');
  }
  return result;
}
