# THUD app testing guide

This guide covers the Flutter app and the bundled TFLite model. Commands use PowerShell syntax and assume the repository root is the current directory.

## 1. Prerequisites

- Flutter 3.50+ / Dart 3.14+.
- Python 3.12 or another supported Python 3.10+ environment for model checks.
- A browser for the web smoke test. Chrome is the easiest target.
- A microphone for real capture tests. Use headphones or a quiet room when appropriate.
- For iPhone tests: macOS, Xcode, a signing team, and an iPhone with microphone permission.
- For Android tests: Android Studio/SDK and a USB-debuggable device or emulator. An emulator does not reproduce physical acoustic capture.

The Flutter app lives at the repository root. The research Python package lives under `models/`.

## 2. Install dependencies

From the repository root:

```powershell
flutter pub get
```

For the Python model package:

```powershell
cd models
python -m pip install -e "."
python -m pip install pytest
cd ..
```

Do not run research scripts from a random working directory. Model documentation and relative artifact paths assume `models/` is the working directory.

## 3. Fast Flutter checks

Run the analyzer first:

```powershell
flutter analyze
```

Run the focused tests that exercise the changed sensing and inspection paths:

```powershell
flutter test `
  test/acoustic_engine_test.dart `
  test/tile_geometry_test.dart `
  test/tile_store_test.dart
```

These cover:

- malformed PCM, clipping, low SNR, duplicate impacts, speech evidence, operating audio, MFCC/STFT features, damping and uncertainty;
- perspective correction, dark/light grout detection, visible-tile counting, partial edge tiles, stable IDs, split/remove/add/correct behavior and invalid geometry;
- annotation persistence, building/floor/unit metadata, tile-link removal confirmation and report data preservation.

Run the complete Flutter suite when a long run is acceptable:

```powershell
flutter test
```

A complete run previously exceeded a 600-second verification window in this workspace. Treat a command timeout as inconclusive; use the focused command above to isolate app behavior and inspect the failure output before retrying the full suite.

## 4. Model artifact checks

### Joint app encoder

From the repository root:

```powershell
python -m pytest models/tests/test_audio_events.py
```

This checks the shipped joint model at `assets/models/yamnet-waveform-embedding-events-f32.tflite`:

- input `float32[15600]`;
- output `float32[1545]`;
- unchanged 1,024-D embedding prefix;
- 521 finite AudioSet event scores;
- built-in TFLite operators only;
- parity against the retained official YAMNet/Sound-of-Water fixtures.

### All finalized research artifacts

From `models/`:

```powershell
cd models
python -m experiments.verify_release
cd ..
```

This is an offline release smoke check. It opens each finalized TFLite artifact, validates hashes, tensor shapes and dtypes, rejects custom/Flex operators, invokes the interpreter, checks finite outputs and compares compact real-input fixtures where available. It is not a new accuracy benchmark and does not validate physical diagnosis.

The normal Python regression suite is:

```powershell
cd models
python -m pytest -q
cd ..
```

Some research tests need cached datasets or optional dependencies. Run the release check and the joint encoder test first when validating an app checkout.

### Rebuild the joint encoder

This is normally unnecessary. If the asset must be regenerated, the official YAMNet SavedModel must already be cached locally:

```powershell
python models/experiments/export_audio_events.py
```

The exporter does not download weights or train a model. Compare the resulting asset hash and rerun `python -m pytest models/tests/test_audio_events.py` afterward.

## 5. Compile and browser smoke test

Build a self-contained web release:

```powershell
flutter build web --release --no-web-resources-cdn
```

Serve the generated files from the repository root:

```powershell
python -m http.server 8091 --directory build/web
```

Open <http://localhost:8091> in Chrome. Microphone capture requires `localhost` or HTTPS. Stop the server with `Ctrl+C`.

Expected initial surface:

- the THUD shell renders at mobile-sized and desktop-sized viewports;
- the six modes are reachable by the horizontal mode control;
- no browser console errors appear during startup;
- the bundled model/runtime files are requested locally, not from a CDN.

A useful headless smoke check is also possible with the repository browser tooling: load `http://127.0.0.1:8091`, confirm the title is `THUD`, capture a screenshot at roughly `390×844`, and inspect the browser error log. The accessibility tree may initially expose only Flutter's `Enable accessibility` button; visual inspection is therefore required.

## 6. Manual app workflow

### Basic demo path

1. Start the web build or run `flutter run -d chrome`.
2. Select **Slab**.
3. Tap **Try demo**.
4. Confirm a result card appears and that the result does **not** invent a physical defect decision without compatible references. The expected first-run state is commonly `Insufficient evidence`.
5. Open **Measurement details** and **Signal**. Confirm the measurement contains the model/capture method and quality metadata.
6. Use **Record again** to return to capture.

Demo audio is deterministic and explicitly labelled synthetic. It proves wiring and persistence only; it is not field evidence.

### Reference calibration

Use the reference/library control before capture or from a result card.

General pattern:

1. Select one mode and one exact object/container/device.
2. Record three clean captures using the same phone position, tap point and setup.
3. Save them as one reference profile.
4. Record a new query with the same provenance and capture mode.
5. Confirm the result uses the learned model and shows its method, distance/status and evidence rather than a generic signal-only hint.
6. Reload the page/app and verify the reference and result remain available.

Mode-specific requirements:

| Mode | What to teach | Expected safety boundary |
|---|---|---|
| Slab | `Supported` and, when genuinely labelled, `Suspected Unsupported`; use independent specimen/tile IDs when possible | Normal-only references remain inconclusive. This is experimental acoustic support classification, not a certified construction defect test. |
| Vessel | At least three measured fill levels for the same container, three taps per level; optionally enter capacity in ml | The app reports an empirical support interval. Volume appears only when capacity metadata is consistent. |
| Fruit | One fruit type/cultivar with labelled firmer/softer examples and at least two classes | Output is relative acoustic similarity/firmness. It does not determine exact ripeness, spoilage, sweetness, internal defects or food safety. |
| Material | Named accepted/original and suspect examples for the comparison domain | Similarity is not chemical composition or authenticity identification. |
| Devices | Power-off casing taps for a normal reference, or the explicit **Operating sound** path for safe sustained audio | Never touch powered or moving machinery. A change does not identify a named mechanical failure. |
| Scope | Any clean capture, then optionally save a reference | Use waveform, spectrum, spectrogram, modal peaks, damping and neural comparison as measurable evidence. |

The current frontend version is `yamnet-waveform-embedding-events-v2`. Old incompatible embeddings are intentionally rejected for calibration and must be recorded again.

## 7. Quality-rejection checks

Exercise these with real or generated captures when testing the result policy:

- cover or disconnect the microphone: expect no usable-audio failure;
- deliberately clip the input: expect a quality rejection, not a diagnosis;
- speak during an impact capture: expect the event-classifier speech gate to reject contamination when the event score crosses the conservative threshold;
- make two taps too close together: expect duplicate-impact exclusion or an insufficient-evidence result;
- use a sustained machine recording in **Operating sound** mode: expect operating metrics, no fabricated ring/decay value, and a note that separate background audio is needed for SNR interpretation.

The classifier score is a capture-quality gate, not a calibrated probability of speech. The uncertainty index is not diagnostic confidence.

## 8. Tile inspection workflow

1. Open **Dossiers** and create/open a project.
2. Add a valid JPEG, PNG or WebP floor image/plan.
3. Open the tile inspection editor.
4. Set four clockwise perspective corners when the image is angled.
5. Try **Detect grout** on an image with visible grout in two directions.
6. If detection is unavailable or wrong, use **Manual grid** or add/correct boundaries manually.
7. Review the visible count and stable tile IDs.
8. Select a tile and exercise correct, split and remove. Save the draft.
9. Add a human finding to a tile, then remove that tile from the draft. Confirm that the app asks before clearing the tile link while preserving notes, photos, measurement links and normalized plan pins.
10. Export the handover PDF and verify that the plan, tile outlines/IDs, human findings, measurements and provenance are present.

The detector counts visible image regions; it does not infer total floor coverage. Textured, occluded or diagonal floors may need manual correction. ARKit tracking and MobileSAM segmentation are not part of the supported path.

## 9. Native device smoke tests

The native paths were not executed on a physical iPhone or Android device in the current verification session. When hardware is available:

```powershell
flutter devices
flutter run -d <device-id>
```

On iPhone, verify microphone permission, a 48 kHz mono capture, interruption/route-change rejection, and restoration of the prior audio session after stopping. On Android, verify microphone permission, PCM stream format, model load, and repeated capture/dispose cycles.

Record device model, OS, Flutter version, capture duration, inference latency, thermal/battery observations and any delegate used. Do not convert a successful build into a field-accuracy claim.

## 10. What a valid test report should include

- exact commit or working-tree state;
- Flutter/Dart/Python versions;
- command and working directory;
- pass/fail/timeout result and full error output for failures;
- target platform and device/browser;
- whether audio was synthetic, microphone, imported or demo;
- model asset filename, version and SHA-256;
- for calibration: object identity, labels, tap count, capture mode and provenance;
- for physical claims: independent specimens, measured ground truth and holdout protocol.

See [`MODEL_APP_HANDOFF.md`](../../MODEL_APP_HANDOFF.md) for the current implementation contract, evidence, model-selection rationale and known limits.
