# THUD

Flutter tap-based inspection app: capture sound, compare object signatures, review key measurements and save findings with photos and PDF reports.

## Run

Requires Flutter 3.50+ and Dart 3.14+ (the installed stable toolchain uses the extracted `material_ui` package).

```sh
flutter pub get
flutter run -d chrome
```

For a self-contained web renderer, without runtime CDN dependencies:

```sh
flutter build web --release --no-web-resources-cdn
python3 -m http.server 8091 --directory build/web
```

Open http://localhost:8091. Microphone use requires localhost or HTTPS. iOS/macOS builds require full Xcode; Android builds require Android SDK. Native microphone permission declarations are included.

## Interface

- Listen: direct horizontal swipe across Slab, Vessel, Fruit, Devices, Material and Scope. Each mode has a distinct color, subject glyph, tactile page motion and animated shell transition. Mode selection stays locked during microphone capture. The capture/result column is capped at 540 px on wide displays.
- Results: one mode-colored status symbol, a short verdict, and at most three main metrics. Retry shows one prioritized reason with Record again and View signal; rejected readings have no save/map action. Reference controls use a compact setup sheet. Demo recordings retain a short Demo label.
- Slab / Devices / Material: three-tap named calibration fits a normal-only model on 1,024D pretrained YAMNet embeddings. New captures are scored against its learned centroid and leave-one-out threshold; acoustic metrics are displayed, not used for verdicts. A difference is not a defect/material certification.
- Vessel: at least three known-fill levels for the exact container, three taps per level, fit a regularized local regressor on neural embeddings. Out-of-support or extrapolated readings are rejected instead of peak-frequency interpolation.
- Fruit: user-labelled Unripe/Ripe/Overripe examples train a local nearest-centroid classifier on neural embeddings. Teach at least two classes, three taps per class, under one fruit type/cultivar name. More independent fruits are needed for useful generalization; this is not food-safety advice.
- Dossiers: room grids, notes, review/defect/accepted marks, linked readings, camera/gallery photos and normalized pins. Slab result → dossier → cell → saved finding → Handover report. Report summary uses actual inspector marks; export includes results, metrics, references, room map and photos. Saved Slab readings link back to their report.
- Scope: WAVE / SPECTRUM / DECAY views of stored signal data. DECAY is explicitly a peak-binned absolute envelope, not a reconstructed Schroeder curve or RT60. Stored waveform time is relative because full recording duration is not retained.
- Reports: coverage, human-marked map, measured evidence, intent/input context, photos and pins embedded in a Unicode-font PDF. Synthetic provenance and human/automatic distinctions carry into reports.
- Saved results and history reuse the mode color, concise verdict and essential metrics. Signal plots expand on demand. Capture failures use short action-oriented messages; underlying measurements and flags remain stored. Settings contains feedback toggles and a local-storage note.

The shell uses mode-specific color, the white `thud.` wordmark, rounded utility buttons and a capsule navigation dock. Dock icons and labels are vertically centered within the same inset bounds as the sliding selection capsule. Projects/Tools use light content sheets within the blue shell. Nunito is bundled locally under its SIL Open Font License (`assets/fonts/OFL.txt`).

## Measurement and storage

Captures and demos both run the bundled pretrained YAMNet TFLite encoder: 975 ms, mono 16 kHz float32 → 1,024 learned features. Input is centered on the strongest impact in the loudest 20 ms block, padded at boundaries, and anti-aliased using windowed-sinc polyphase resampling. Native Android/iOS use `tflite_flutter`; web uses a dedicated worker with locally bundled official TensorFlow.js TFLite WASM files. No inference CDN or cloud upload is required.

Calibration fits lightweight models on-device; YAMNet weights remain frozen. Fitted parameters, raw neural embeddings, exact encoder provenance, example IDs, intent/object and demo/real separation persist in `thud.calibration.v2`. Legacy `thud.anchors.v1` traces remain readable but must be re-recorded because their original PCM was discarded. Records remain in `thud.store.v1`, including embeddings and inference context; clearing site/app data removes local calibration.

Capture requests 48 kHz mono PCM16 with processing disabled. FFT frequency, band-energy ratio, ring-down, SNR and dBFS remain diagnostic measurements and capture-quality gates, not handcrafted decision fallbacks. Demos exercise the same encoder but prove software behavior only. Physical task accuracy and dataset/encoder license review remain deployment gates. See [MODEL_APP_HANDOFF.md](MODEL_APP_HANDOFF.md).

The camera/gallery attaches inspection photos with normalized pins. Photos remain bounded to 2 MB each. Storage writes report errors. Full-resolution PCM is released after neural encoding; derived traces and embeddings persist locally.

## Verification

Current learned-calibration checks: `flutter analyze`; `flutter test` (preprocessing, learned fitting/OOD, persistence/migration, and UI verdict independence from acoustic scalar metrics). Encoder parity runs the actual TFLite on three real waveform fixtures in desktop TensorFlow and the bundled browser WASM runtime. Interactive app smoke evidence and platform limits are recorded in [MODEL_APP_HANDOFF.md](MODEL_APP_HANDOFF.md).

Earlier signal-only UI/PDF smoke results predate the learned-calibration cutover and are not evidence of current ML field accuracy.

## Design references

The project includes Meng To's `no-ai-design-slop`, `dither-background` and `high-contrast-skeuomorphic-clean` skills.

Interface refinement follows **Unslop by Damir Sokolovsky ([@heydamir](https://x.com/heydamir))**, [heydamir/unslop](https://github.com/heydamir/unslop), licensed CC BY 4.0. The installed skill and original attribution are preserved under `.agents/skills/unslop`. The current user-directed adaptation uses mode-specific colors, playful subject glyphs, swipe transitions and result-first typography.
