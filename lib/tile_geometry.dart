import 'dart:math' as math;
import 'dart:typed_data';

/// Image-relative coordinates, never GPS or an inferred physical dimension.
class TilePoint {
  const TilePoint(this.x, this.y);
  final double x, y;
  Map<String, dynamic> toJson() => {'x': x, 'y': y};
  factory TilePoint.fromJson(Map<String, dynamic> json) =>
      TilePoint((json['x'] as num).toDouble(), (json['y'] as num).toDouble());
  TilePoint lerp(TilePoint other, double t) =>
      TilePoint(x + (other.x - x) * t, y + (other.y - y) * t);
}

class TileRegion {
  TileRegion({required this.id, required this.row, required this.column,
      required List<TilePoint> polygon}) : polygon = List.unmodifiable(polygon);
  final String id;
  final int row, column;
  /// Clockwise: top left, top right, bottom right, bottom left.
  final List<TilePoint> polygon;
  TilePoint get center => TilePoint(
      polygon.fold<double>(0, (n, p) => n + p.x) / polygon.length,
      polygon.fold<double>(0, (n, p) => n + p.y) / polygon.length);
  bool contains(TilePoint point) {
    var positive = false, negative = false;
    for (var i = 0; i < polygon.length; i++) {
      final a = polygon[i], b = polygon[(i + 1) % polygon.length];
      final cross = (b.x - a.x) * (point.y - a.y) -
          (b.y - a.y) * (point.x - a.x);
      positive |= cross > 1e-9;
      negative |= cross < -1e-9;
    }
    return !(positive && negative);
  }
  Map<String, dynamic> toJson() => {'id': id, 'row': row, 'column': column,
    'polygon': polygon.map((p) => p.toJson()).toList()};
  factory TileRegion.fromJson(Map<String, dynamic> json) => TileRegion(
      id: json['id'] as String, row: json['row'] as int,
      column: json['column'] as int,
      polygon: (json['polygon'] as List).map((p) =>
          TilePoint.fromJson(Map<String, dynamic>.from(p as Map))).toList());
}

class TileLayout {
  TileLayout({required List<TileRegion> tiles, required List<TilePoint> corners,
      this.nextId = 1, this.method = 'manual'})
      : tiles = List.unmodifiable(tiles), corners = List.unmodifiable(corners);
  final List<TileRegion> tiles;
  final List<TilePoint> corners;
  final int nextId;
  final String method;
  static const imageCorners = [TilePoint(0, 0), TilePoint(1, 0),
    TilePoint(1, 1), TilePoint(0, 1)];
  factory TileLayout.empty() => TileLayout(tiles: [], corners: imageCorners);
  void validate() {
    if (nextId < 1 || tiles.length > 2500) {
      throw const FormatException('Use at most 2500 visible tiles.');
    }
    validateQuad(corners);
    final ids = <String>{};
    for (final tile in tiles) {
      if (tile.id.isEmpty || !ids.add(tile.id) || tile.row < 0 || tile.column < 0) {
        throw const FormatException('Tile IDs must be unique with valid grid positions.');
      }
      validateQuad(tile.polygon);
      final serial = RegExp(r'^T([0-9]+)$').firstMatch(tile.id);
      if (serial != null && int.parse(serial.group(1)!) >= nextId) {
        throw const FormatException('Next tile ID must not reuse a saved ID.');
      }
    }
  }
  static void validateQuad(List<TilePoint> points) {
    if (points.length != 4 || points.any((p) => !p.x.isFinite ||
        !p.y.isFinite || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) {
      throw const FormatException('Choose four image corners in clockwise order.');
    }
    for (var i = 0; i < 4; i++) {
      final a = points[i], b = points[(i + 1) % 4], c = points[(i + 2) % 4];
      if ((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 1e-7) {
        throw const FormatException('Corners must form a non-crossing clockwise quadrilateral.');
      }
    }
  }
  TileRegion? at(TilePoint p) {
    for (final tile in tiles.reversed) {
      if (tile.contains(p)) return tile;
    }
    return null;
  }
  TileLayout remove(String id) => TileLayout(
      tiles: tiles.where((t) => t.id != id).toList(), corners: corners,
      nextId: nextId, method: 'edited');
  TileLayout correct(String id, List<TilePoint> polygon) {
    validateQuad(polygon);
    return TileLayout(tiles: tiles.map((t) => t.id == id
        ? TileRegion(id: t.id, row: t.row, column: t.column, polygon: polygon)
        : t).toList(), corners: corners, nextId: nextId, method: 'edited');
  }
  TileLayout add(List<TilePoint> polygon) {
    validateQuad(polygon);
    return TileLayout(tiles: [...tiles, TileRegion(id: 'T$nextId',
      row: tiles.isEmpty ? 0 : tiles.map((t) => t.row).reduce(math.max) + 1,
      column: 0, polygon: polygon)], corners: corners,
      nextId: nextId + 1, method: 'edited');
  }
  /// Keep the original ID on the first half; IDs are never renumbered/reused.
  TileLayout split(String id, {required bool vertical}) {
    final tile = tiles.firstWhere((t) => t.id == id);
    final p = tile.polygon;
    final a = vertical ? p[0].lerp(p[1], .5) : p[0].lerp(p[3], .5);
    final b = vertical ? p[3].lerp(p[2], .5) : p[1].lerp(p[2], .5);
    final first = vertical ? [p[0], a, b, p[3]] : [p[0], p[1], b, a];
    final second = vertical ? [a, p[1], p[2], b] : [a, b, p[2], p[3]];
    return TileLayout(tiles: [
      for (final t in tiles)
        if (t.id != id) t else TileRegion(id: t.id, row: t.row,
          column: t.column, polygon: first),
      TileRegion(id: 'T$nextId', row: tile.row + (vertical ? 0 : 1),
          column: tile.column + (vertical ? 1 : 0), polygon: second),
    ], corners: corners, nextId: nextId + 1, method: 'edited');
  }
  TileLayout grid(List<double> xs, List<double> ys,
      {List<TilePoint>? imageCorners, String source = 'manual'}) {
    final quad = imageCorners ?? corners;
    validateQuad(quad);
    for (final axis in [xs, ys]) {
      if (axis.length < 2 || axis.first != 0 || axis.last != 1 ||
          axis.any((v) => !v.isFinite || v < 0 || v > 1)) {
        throw const FormatException('A grid must span the selected image area.');
      }
      for (var i = 1; i < axis.length; i++) {
        if (axis[i] <= axis[i - 1]) throw const FormatException('Grid lines must be ordered.');
      }
    }
    if ((xs.length - 1) * (ys.length - 1) > 2500) {
      throw const FormatException('Use at most 2500 tiles.');
    }
    final transform = TilePerspective(quad);
    var serial = nextId;
    final result = <TileRegion>[];
    final used = <String>{};
    for (var r = 0; r < ys.length - 1; r++) {
      for (var c = 0; c < xs.length - 1; c++) {
        final polygon = [
          transform.project(xs[c], ys[r]), transform.project(xs[c + 1], ys[r]),
          transform.project(xs[c + 1], ys[r + 1]), transform.project(xs[c], ys[r + 1]),
        ];
        final previous = tiles.where((tile) {
          if (used.contains(tile.id)) return false;
          for (var i = 0; i < 4; i++) {
            if ((tile.polygon[i].x - polygon[i].x).abs() >= .012 ||
                (tile.polygon[i].y - polygon[i].y).abs() >= .012) return false;
          }
          return true;
        }).firstOrNull;
        final id = previous?.id ?? 'T${serial++}';
        used.add(id);
        result.add(TileRegion(id: id, row: r, column: c, polygon: polygon));
      }
    }
    return TileLayout(tiles: result, corners: quad, nextId: serial, method: source);
  }
  Map<String, dynamic> toJson() => {'tiles': tiles.map((t) => t.toJson()).toList(),
    'corners': corners.map((p) => p.toJson()).toList(),
    'nextId': nextId, 'method': method};
  factory TileLayout.fromJson(Map<String, dynamic> json) {
    final result = TileLayout(
      tiles: (json['tiles'] as List).map((t) =>
          TileRegion.fromJson(Map<String, dynamic>.from(t as Map))).toList(),
      corners: (json['corners'] as List).map((p) =>
          TilePoint.fromJson(Map<String, dynamic>.from(p as Map))).toList(),
      nextId: json['nextId'] as int, method: json['method'] as String);
    result.validate();
    return result;
  }
}

/// Projective square-to-quadrilateral mapping, not a bilinear approximation.
class TilePerspective {
  TilePerspective(List<TilePoint> quad) {
    TileLayout.validateQuad(quad);
    final p = quad;
    final dx1 = p[1].x - p[2].x, dx2 = p[3].x - p[2].x;
    final dy1 = p[1].y - p[2].y, dy2 = p[3].y - p[2].y;
    final dx3 = p[0].x - p[1].x + p[2].x - p[3].x;
    final dy3 = p[0].y - p[1].y + p[2].y - p[3].y;
    final det = dx1 * dy2 - dx2 * dy1;
    g = (dx3 * dy2 - dx2 * dy3) / det;
    h = (dx1 * dy3 - dx3 * dy1) / det;
    a = p[1].x - p[0].x + g * p[1].x;
    b = p[3].x - p[0].x + h * p[3].x;
    c = p[0].x;
    d = p[1].y - p[0].y + g * p[1].y;
    e = p[3].y - p[0].y + h * p[3].y;
    f = p[0].y;
  }
  late final double a, b, c, d, e, f, g, h;
  TilePoint project(double x, double y) {
    final w = g * x + h * y + 1;
    return TilePoint(((a * x + b * y + c) / w).clamp(0.0, 1.0).toDouble(),
        ((d * x + e * y + f) / w).clamp(0.0, 1.0).toDouble());
  }
  TilePoint unproject(TilePoint p) {
    final aa = a - p.x * g, bb = b - p.x * h;
    final dd = d - p.y * g, ee = e - p.y * h;
    final cc = p.x - c, ff = p.y - f;
    final det = aa * ee - bb * dd;
    return TilePoint((cc * ee - bb * ff) / det, (aa * ff - cc * dd) / det);
  }
}

class TileRaster {
  const TileRaster(this.rgba, this.width, this.height, this.corners);
  final Uint8List rgba;
  final int width, height;
  final List<TilePoint> corners;
}

class TileDetection {
  const TileDetection(this.xs, this.ys, this.message);
  final List<double> xs, ys;
  final String message;
  bool get found => xs.length > 2 && ys.length > 2;
}

/// Offline narrow-line contrast + edge support in a projectively rectified ROI.
/// shortcut: requires two visible grout families; use manual corners/grid for
/// occlusion, diagonal layouts, strong texture, or grout indistinguishable from tile.
TileDetection detectTileGrid(TileRaster raster) {
  if (raster.width < 2 || raster.height < 2 ||
      raster.rgba.length != raster.width * raster.height * 4) {
    throw const FormatException('A decoded RGBA image is required.');
  }
  const n = 256;
  final transform = TilePerspective(raster.corners);
  final gray = Float64List(n * n);
  double luma(int xx, int yy) {
    final i = (yy * raster.width + xx) * 4;
    return .299 * raster.rgba[i] + .587 * raster.rgba[i + 1] + .114 * raster.rgba[i + 2];
  }
  for (var y = 0; y < n; y++) {
    for (var x = 0; x < n; x++) {
      final p = transform.project(x / (n - 1), y / (n - 1));
      final sx = p.x * (raster.width - 1), sy = p.y * (raster.height - 1);
      final ix = sx.floor(), iy = sy.floor();
      final fx = sx - ix, fy = sy - iy;
      final right = math.min(ix + 1, raster.width - 1);
      final bottom = math.min(iy + 1, raster.height - 1);
      gray[y * n + x] = (luma(ix, iy) * (1 - fx) + luma(right, iy) * fx) * (1 - fy) +
          (luma(ix, bottom) * (1 - fx) + luma(right, bottom) * fx) * fy;
    }
  }
  List<double> lines(bool vertical) {
    final scores = List<double>.filled(n, 0);
    double sample(int x, int y, int delta) =>
        gray[vertical ? y * n + x + delta : (x + delta) * n + y];
    for (var x = 5; x < n - 5; x++) {
      final bands = List<double>.filled(8, 0);
      for (var y = 4; y < n - 4; y++) {
        final center = (sample(x, y, -1) + sample(x, y, 0) + sample(x, y, 1)) / 3;
        final left = (sample(x, y, -4) + sample(x, y, -3)) / 2;
        final right = (sample(x, y, 3) + sample(x, y, 4)) / 2;
        // A grout line has contrast on both sides; a single shadow edge does not.
        final contrast = ((left - center) * (right - center) > 0)
            ? math.min((left - center).abs(), (right - center).abs()) : 0.0;
        bands[math.min(7, y * 8 ~/ n)] += contrast;
      }
      // Require support across the image, not a local object edge.
      bands.sort();
      scores[x] = bands[2] / (n / 8);
    }
    final sorted = scores.sublist(5, n - 5)..sort();
    final median = sorted[sorted.length ~/ 2];
    final maxScore = sorted.last;
    final threshold = math.max(7.0, math.max(median * 3.0, maxScore * .28));
    final candidates = <int>[];
    for (var i = 6; i < n - 6; i++) {
      if (scores[i] >= threshold && scores[i] >= scores[i - 1] && scores[i] > scores[i + 1]) {
        candidates.add(i);
      }
    }
    candidates.sort((a, b) => scores[b].compareTo(scores[a]));
    final peaks = <int>[];
    for (final i in candidates) {
      if (peaks.every((p) => (p - i).abs() >= 10)) peaks.add(i);
    }
    peaks.sort();
    if (peaks.isEmpty || peaks.length > 49) return [];
    // Interior spacing must be reasonably regular; edge tiles may be partial.
    if (peaks.length >= 3) {
      final gaps = [for (var i = 1; i < peaks.length; i++) peaks[i] - peaks[i - 1]]..sort();
      final spacing = gaps[gaps.length ~/ 2];
      if (gaps.any((gap) => gap < spacing * .65 || gap > spacing * 1.4)) return [];
    }
    return [0, ...peaks.map((p) => p / (n - 1)), 1];
  }
  final xs = lines(true), ys = lines(false);
  return TileDetection(xs, ys, xs.length > 2 && ys.length > 2
      ? 'Image-derived grout grid. Edge cells are partial visible tiles; review every boundary.'
      : 'No reliable two-direction grout grid. Take a wider overhead photo, set four floor corners, or create a manual grid.');
}
