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
- Results: a plain-language assessment and recommended action come first. Technical measurements, learned model distances, method and evidence are collapsed under Measurement details. Rejected readings have no save/map action; demo provenance remains visible.
- Slab: user-labelled Supported / Suspected Unsupported examples fit a local prototype model; normal-only comparison remains Inconclusive. Independent tile IDs enable leave-one-tile-out evaluation when enough examples exist. Experimental support classification is not certified defect detection.
- Material / Devices: few-shot reference enrollment, unknown rejection and normal-baseline differences. Devices separates powered-off taps from operating audio; never touch powered or moving parts. No specific failure/composition diagnosis.
- Vessel: three or more measured fill levels for the exact container train a regularized nonlinear regressor. Returns fill, empirical uncertainty envelope and calibration status; volume requires consistent capacity metadata. No universal monotonic frequency assumption.
- Fruit: labelled same-type acoustic similarity is an experimental relative firmness indicator, not exact ripeness, spoilage, sweetness or food safety.
- Dossiers: room grids, notes, review/defect/accepted marks, linked readings, camera/gallery photos and normalized pins. Slab result → dossier → cell → saved finding → Handover report. Report summary uses actual inspector marks; export includes results, metrics, references, room map and photos. Saved Slab readings link back to their report.
- Scope: live waveform; saved waveform, spectrum, decay envelope, bounded STFT spectrogram, modal frequencies/damping and pairwise neural signature comparison. DECAY is a peak-binned envelope, not RT60.
- Reports: coverage, human-marked map, measured evidence, intent/input context, photos and pins embedded in a Unicode-font PDF. Synthetic provenance and human/automatic distinctions carry into reports.
- Saved results and history reuse the mode color, concise verdict and essential metrics. Signal plots expand on demand. Rejected readings have no save/map action; demo provenance remains visible. Capture failures use short action-oriented messages that distinguish a missing neural embedding from a recorded tap rejected for acoustic quality. Settings contains feedback toggles and a local-storage note. Calibrated modes expose their reference library before the first sample; empty libraries explain the three-tap setup and offer a demo path.
- Consumer assessment is shared by live results, saved results and history. Ambiguous, incompatible and out-of-range captures say Insufficient evidence or Inconclusive. Reports distinguish automatic experimental assessments from human inspection marks.

The shell uses mode-specific color, the white `thud.` wordmark, rounded utility buttons and a capsule navigation dock. Dock icons and labels are vertically centered within the same inset bounds as the sliding selection capsule. Projects/Tools use light content sheets within the blue shell. Nunito is bundled locally under its SIL Open Font License (`assets/fonts/OFL.txt`).

## Measurement and storage

Captures and demos run the bundled joint YAMNet TFLite: 975 ms mono 16 kHz float32 → 1,024 pretrained features plus 521 AudioSet event scores. Isolated impacts are centered and coordinate-median pooled; operating audio uses distributed windows. Windowed-sinc polyphase resampling suppresses aliasing. Speech-family event scores scan the complete capture and gate contamination; they do not diagnose physical properties. Native Android/iOS use `tflite_flutter`; web uses a dedicated worker with locally bundled TFLite WASM. No inference CDN or cloud upload.

Calibration fits lightweight models on-device over frozen neural features and explicitly scaled resonance/damping features. Reference profiles persist in `thud.anchors.v1`; existing `thud.calibration.v2` profiles migrate into that library. Fitted parameters and exact selected references persist with saved inference context in `thud.store.v1`. Changed frontend version requires re-recording older captures; legacy records remain readable, not silently reused as current calibration.

Capture requests 48 kHz mono PCM16 with processing disabled. iOS configures `.record` / `.measurement`, rejects interrupted/route-changed takes and restores its previous category/mode. Shared analysis detects onsets, excludes closely spaced impacts, rejects clipping/poor impact SNR/speech, and measures FFT/STFT/MFCC, centroid, flatness, modal peaks and damping. Capture uncertainty is not a calibrated diagnostic probability. Demos prove software behavior only; physical accuracy, licensing and real-device measurements remain deployment gates. See [MODEL_APP_HANDOFF.md](MODEL_APP_HANDOFF.md).

The camera/gallery attaches inspection photos with normalized pins. Photos remain bounded to 2 MB each. Storage writes report errors. Full-resolution PCM is released after neural encoding; derived traces and embeddings persist locally.

### Reference library schema

Reference profiles are local, user-supplied data; THUD does not seed fabricated labels. Storage key: `thud.anchors.v1`. The Reference Library accepts either one anchor, an anchor array, or the exported wrapper:

```json
{"schema":"thud.reference-library","version":1,"anchors":[...]}
```

Each anchor contains `intent`, `object`, exactly three clean `taps`, and optional `level`, `label`, and `metadata`. Slab uses `Supported` / `Suspected Unsupported` (legacy Solid/Hollow labels are normalized). Fruit uses same-type labelled firmness examples. Devices normal baselines are separate from anomaly examples and operating/tap modes. Imports validate the whole batch before an atomic write; malformed data leaves the prior library unchanged. Demo and real profiles remain separate; incompatible old embeddings require re-recording.

### Architectural floor plans

Projects may store one validated JPEG, PNG or WebP floor plan under `thud.store.v1` (maximum 2 MB), including decoded pixel dimensions. Findings retain normalized `planX` / `planY` independently from normalized photo `x` / `y`. Replacing a plan requires explicit reset of existing plan pins; storage failures roll back the plan and annotations. Handover PDF rendering uses the same contain-fit geometry and numbered status-colored pins as the interactive map.

The floor-plan API is `ThudStore.updateFloorPlan(project, {bytes, name, resetPins})`. A plan pin requires a project plan and a finite normalized coordinate pair; a photo pin still requires its own photo and coordinate pair.

### Visible tile mapping

In a dossier's Inspection map, capture/upload a floor photo or plan, then open the tile editor. Set four clockwise perspective corners if needed; detect visible grout or create a manual grid. Review counts, select IDs, and add/remove/split/correct boundaries before saving. Detection is offline image processing, not a pretrained tile-count guarantee. Flat or obscured grout requires manual correction. Finding links use stable tile IDs; removing linked tiles or replacing the reference image requires explicit unlink/reset. Building, floor, unit, room, grid position and normalized geometry persist locally and appear in inspection reports. Optional AR tracking is not implemented.

## Verification

Current learned-calibration checks: `flutter analyze`; `flutter test` (preprocessing, learned fitting/OOD, persistence/migration, and UI verdict independence from acoustic scalar metrics). Encoder parity runs the actual TFLite on three real waveform fixtures in desktop TensorFlow and the bundled browser WASM runtime. Interactive app smoke evidence and platform limits are recorded in [MODEL_APP_HANDOFF.md](MODEL_APP_HANDOFF.md).

Earlier signal-only UI/PDF smoke results predate the learned-calibration cutover and are not evidence of current ML field accuracy.

Verdict-first smoke: an isolated real `SensingResult` rendered in Chrome desktop and 390×844 mobile with deterministic PCM passed good/hollow/ambiguous/Fruit policy checks. Fruit displayed Ripeness not determined with an uncertainty icon and no exposed metrics by default. This verifies presentation and policy routing, not ripeness or defect-detection accuracy.

## Design references

The project includes Meng To's `no-ai-design-slop`, `dither-background` and `high-contrast-skeuomorphic-clean` skills.

Interface refinement follows **Unslop by Damir Sokolovsky ([@heydamir](https://x.com/heydamir))**, [heydamir/unslop](https://github.com/heydamir/unslop), licensed CC BY 4.0. The installed skill and original attribution are preserved under `.agents/skills/unslop`. The current user-directed adaptation uses mode-specific colors, playful subject glyphs, swipe transitions and result-first typography.
