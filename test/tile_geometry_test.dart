import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:thud/tile_geometry.dart';
import 'package:thud/floor_plan.dart';

TileRaster fixture({List<TilePoint> corners = TileLayout.imageCorners,
    bool grout = true, bool bright = false}) {
  const n = 384;
  final pixels = Uint8List(n * n * 4);
  final perspective = TilePerspective(corners);
  for (var y = 0; y < n; y++) {
    for (var x = 0; x < n; x++) {
      final p = perspective.unproject(TilePoint(x / (n - 1), y / (n - 1)));
      final inside = p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
      final line = inside && grout &&
          ([(p.x - 1 / 3).abs(), (p.x - 2 / 3).abs(), (p.y - .5).abs()]
              .any((d) => d < .007));
      final value = line ? (bright ? 235 : 25) : (inside ? (bright ? 65 : 195) : 90);
      final i = (y * n + x) * 4;
      pixels[i] = pixels[i + 1] = pixels[i + 2] = value;
      pixels[i + 3] = 255;
    }
  }
  return TileRaster(pixels, n, n, corners);
}

void main() {
  test('projective correction preserves corners and round trips interior points', () {
    const quad = [TilePoint(.2, .15), TilePoint(.85, .05),
      TilePoint(.98, .95), TilePoint(.05, .85)];
    final mapping = TilePerspective(quad);
    for (var i = 0; i < 4; i++) {
      final input = TileLayout.imageCorners[i];
      final p = mapping.project(input.x, input.y);
      expect(p.x, closeTo(quad[i].x, 1e-9));
      expect(p.y, closeTo(quad[i].y, 1e-9));
    }
    for (final point in [const TilePoint(.3, .4), const TilePoint(.8, .6)]) {
      final p = mapping.unproject(mapping.project(point.x, point.y));
      expect(p.x, closeTo(point.x, 1e-9));
      expect(p.y, closeTo(point.y, 1e-9));
    }
    // The center is not the average of the four corners under perspective.
    expect(mapping.project(.5, .5).y, isNot(closeTo(.5, .001)));
  });

  for (final bright in [false, true]) {
    test('detects ${bright ? 'light' : 'dark'} grout and counts six visible regions', () {
      final result = detectTileGrid(fixture(bright: bright));
      expect(result.found, isTrue);
      expect(result.xs.length, 4);
      expect(result.ys.length, 3);
      expect(result.xs[1], closeTo(1 / 3, .015));
      expect(result.ys[1], closeTo(.5, .015));
      final layout = TileLayout.empty().grid(result.xs, result.ys);
      expect(layout.tiles.length, 6);
      layout.validate();
    });
  }
  test('detects grout after perspective correction', () {
    const quad = [TilePoint(.25, .08), TilePoint(.75, .12),
      TilePoint(.95, .95), TilePoint(.04, .88)];
    final result = detectTileGrid(fixture(corners: quad));
    expect(result.found, isTrue);
    expect(result.xs.length, 4);
    expect(result.ys.length, 3);
  });
  test('uniform image never invents a tile count', () {
    final result = detectTileGrid(fixture(grout: false));
    expect(result.found, isFalse);
    expect(result.message, contains('manual grid'));
  });
  test('split/remove/add/correct retain stable IDs and image hit positions', () {
    final original = TileLayout.empty().grid([0, .5, 1], [0, 1]);
    final split = original.split('T1', vertical: false);
    expect(split.tiles.map((t) => t.id), ['T1', 'T2', 'T3']);
    expect(split.at(const TilePoint(.25, .75))?.id, 'T3');
    final removed = split.remove('T2');
    final added = removed.add([const TilePoint(.5, 0), const TilePoint(1, 0),
      const TilePoint(1, 1), const TilePoint(.5, 1)]);
    expect(added.tiles.map((t) => t.id), ['T1', 'T3', 'T4']);
    final corrected = added.correct('T4', [const TilePoint(.6, .1),
      const TilePoint(.9, .1), const TilePoint(.9, .9), const TilePoint(.6, .9)]);
    expect(corrected.at(const TilePoint(.55, .5)), isNull);
    expect(corrected.at(const TilePoint(.7, .5))?.id, 'T4');
    corrected.validate();
    final decoded = TileLayout.fromJson(jsonDecode(jsonEncode(corrected.toJson())) as Map<String, dynamic>);
    expect(decoded.toJson(), corrected.toJson());
  });
  test('redetecting the same grid keeps IDs; removed IDs are never reused', () {
    final first = TileLayout.empty().grid([0, .5, 1], [0, 1]);
    final again = first.grid([0, .5, 1], [0, 1]);
    expect(again.tiles.map((t) => t.id), ['T1', 'T2']);
    final restored = again.remove('T1').grid([0, .5, 1], [0, 1]);
    expect(restored.tiles.map((t) => t.id), ['T3', 'T2']);
  });
  test('PDF overlay uses the same edited normalized vertices and stable labels', () {
    final layout = TileLayout.empty().grid([0, .5, 1], [0, 1]).remove('T1');
    final svg = tileLayoutSvg(layout, 400, 200);
    expect(svg, contains('200.0,0.0 400.0,0.0 400.0,200.0 200.0,200.0'));
    expect(svg, contains('>T2</text>'));
    expect(svg, isNot(contains('>T1</text>')));
  });
  test('image-edge intervals count as partial visible tiles without extrapolation', () {
    final grid = TileLayout.empty().grid([0, .12, .5, .88, 1], [0, .1, .6, 1]);
    expect(grid.tiles.length, 12);
    expect(grid.at(const TilePoint(.02, .02))?.id, 'T1');
    expect(grid.at(const TilePoint(.98, .98))?.id, 'T12');
  });
  test('malformed pixel buffer is rejected before detection', () {
    expect(() => detectTileGrid(TileRaster(Uint8List(1), 100, 100,
        TileLayout.imageCorners)), throwsFormatException);
  });
  test('rejects crossed corners, nonfinite geometry, duplicate IDs and ID reuse', () {
    expect(() => TilePerspective([const TilePoint(0, 0), const TilePoint(1, 1),
      const TilePoint(1, 0), const TilePoint(0, 1)]), throwsFormatException);
    expect(() => TileLayout.empty().add([const TilePoint(double.nan, 0),
      const TilePoint(1, 0), const TilePoint(1, 1), const TilePoint(0, 1)]), throwsFormatException);
    final tile = TileLayout.empty().grid([0, 1], [0, 1]).tiles.single;
    expect(() => TileLayout(tiles: [tile, tile], corners: TileLayout.imageCorners,
        nextId: 2).validate(), throwsFormatException);
    expect(() => TileLayout(tiles: [tile], corners: TileLayout.imageCorners,
        nextId: 1).validate(), throwsFormatException);
  });
}
