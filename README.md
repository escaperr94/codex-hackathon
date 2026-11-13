# THUD

Flutter acoustic field notebook for capturing taps, reviewing their waveform and spectrum, and associating recordings with inspection grid locations.

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

- Listen: swipeable Slab, Vessel, Fruit and Scope chooser; native-style edge-swipe routes, tactile press compression, result reveal and sliding dock. Capture uses an 8-second limit with cancellation. GPU interference and rings respond to live PCM; no idle loop. Reduce Motion is supported.
- Slab: three separate reference taps on the same named object, repeatability checks, regularized acoustic deviation with disclosed unvalidated threshold. A screening anomaly is not a defect diagnosis.
- Vessel: at least three strictly decreasing known-fill resonance anchors for the exact named container. Piecewise interpolation only inside the measured range; no universal model, extrapolation or fabricated accuracy interval. Known-fill synthetic PCM fixtures support the demo and remain separated from microphone anchors.
- Fruit: mass input computes the research proxy `peakHz² × massKg^(2/3)`. No ripe/unripe, food-safety or validated quality classification.
- Projects: room grids, notes and multiple human marks per cell (review/defect/accepted), linked acoustic records, local camera/gallery photos and normalized image-coordinate pins. Unsaved edits require discard confirmation. Marks and supplied measurements save in one storage transaction.
- Scope: WAVE / SPECTRUM / DECAY views of stored signal data. DECAY is explicitly a peak-binned absolute envelope, not a reconstructed Schroeder curve or RT60. Stored waveform time is relative because full recording duration is not retained.
- Reports: coverage, human-marked map, measured evidence, intent/input context, photos and pins embedded in a Unicode-font PDF. Synthetic provenance and human/automatic distinctions carry into reports.
- Default surfaces keep one result, one short instruction and the primary action visible. Technical metrics/methods are under `Evidence & limits`, scope provenance under `About this signal`, calibration caveats under `Reference limits`, and map instructions under `Map guide`. Demo/research and human-mark provenance remain visible without expanding details.

The shell uses THUD blue, the white `thud.` wordmark, rounded utility buttons and a capsule navigation dock. Dock icons and labels are vertically centered within the same inset bounds as the sliding selection capsule. Projects/Tools use light content sheets within the blue shell. Nunito is bundled locally under its SIL Open Font License (`assets/fonts/OFL.txt`).

## Measurement limits

This is an acoustic evidence and human-annotation app, not a validated hollow-tile classifier, fill instrument or structural safety tool. No defect probabilities or confidence scores are fabricated. Reference deviation is an unvalidated screening rule; fill is conditional interpolation; fruit firmness is a research proxy. Synthetic examples are deterministic PCM passed through the analysis path, visibly labeled and excluded from microphone coverage. Demo anchors and microphone anchors are kept separate.

Requested microphone configuration is 48 kHz mono PCM16 with processing disabled where supported by the platform/plugin. Hardware behavior still requires device verification. The displayed decay is an observed 20 dB amplitude reduction, not a calibrated reverberation RT60. Levels are digital dBFS, not dB SPL.

Derived traces, metadata, projects, marks, compressed photo bytes and notes persist locally using SharedPreferences. References persist separately under `thud.anchors.v1`; marks and records use `thud.store.v1`. Photos are bounded to 2 MB each, and total capacity depends on platform quota; failed writes roll back and show an error. Image pins are normalized coordinates, not GPS or automatic camera localization. Raw full-resolution PCM is not retained after analysis. No server upload is implemented. Clearing app/site data removes local evidence.

## Verification

Web smoke at 390×844 exercised touch-swipe selection, Fruit mass proxy/save, Slab three-tap reference and contrast anomaly, Vessel 0/50/100 synthetic anchors with 50% interpolation, all Scope views, map creation, human marking, gallery upload, image pin, persistence after reload and unsaved-edit protection. PDF bytes were captured and parsed: map, synthetic provenance, saved context and photo page present. Browser native download was not observed. Reduce Motion navigation was exercised. Physical microphone accuracy, iPhone frame rate, native camera permissions, haptics and native sharing still require real-device verification. A Flutter web keyboard-event error occurred during automated text entry; subsequent photo/save/report paths completed.

## Design references

The project includes Meng To's `no-ai-design-slop`, `dither-background` and `high-contrast-skeuomorphic-clean` skills.

Interface refinement follows **Unslop by Damir Sokolovsky ([@heydamir](https://x.com/heydamir))**, [heydamir/unslop](https://github.com/heydamir/unslop), licensed CC BY 4.0. The installed skill and original attribution are preserved under `.agents/skills/unslop`. The later user-directed visual adaptation retains a focused cobalt listening scene, large rounded typography and state-driven motion; long explanations remain in help and evidence screens.
