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
- Slab / Devices / Material: three-tap named references and regularized deviation from peak, ring time and HII. Slab matching/differing results lead to Add to handover; other comparisons lead to save/review. Material compares a sample against an original reference.
- Vessel: at least three strictly decreasing known-fill anchors for the same container; piecewise interpolation inside the measured range. Other readings show resonance and anchor setup.
- Fruit: resonance, ring time and mass-based firmness `peakHz² × massKg^(2/3)`. Fruit type and mass remain attached to the saved result.
- Dossiers: room grids, notes, review/defect/accepted marks, linked readings, camera/gallery photos and normalized pins. Slab result → dossier → cell → saved finding → Handover report. Report summary uses actual inspector marks; export includes results, metrics, references, room map and photos. Saved Slab readings link back to their report.
- Scope: WAVE / SPECTRUM / DECAY views of stored signal data. DECAY is explicitly a peak-binned absolute envelope, not a reconstructed Schroeder curve or RT60. Stored waveform time is relative because full recording duration is not retained.
- Reports: coverage, human-marked map, measured evidence, intent/input context, photos and pins embedded in a Unicode-font PDF. Synthetic provenance and human/automatic distinctions carry into reports.
- Saved results and history reuse the mode color, concise verdict and essential metrics. Signal plots expand on demand. Capture failures use short action-oriented messages; underlying measurements and flags remain stored. Settings contains feedback toggles and a local-storage note.

The shell uses mode-specific color, the white `thud.` wordmark, rounded utility buttons and a capsule navigation dock. Dock icons and labels are vertically centered within the same inset bounds as the sliding selection capsule. Projects/Tools use light content sheets within the blue shell. Nunito is bundled locally under its SIL Open Font License (`assets/fonts/OFL.txt`).

## Measurement and storage

The microphone drives acoustic metrics. The camera/gallery attaches inspection photos with normalized image pins. Slab, Devices and Material compare recorded signatures against a named reference; Vessel interpolates between known-fill anchors; Fruit computes the mass-based firmness index. Synthetic examples run deterministic PCM through the same analysis path and retain Demo provenance.

Capture requests 48 kHz mono PCM16 with processing disabled. The analyzer uses a 4096-point Hann FFT, band-energy ratio HII, observed 20 dB ring-down time, estimated SNR and digital dBFS levels.

Derived traces, metadata, projects, marks, compressed photos and notes persist locally using SharedPreferences. References use `thud.anchors.v1`; records use `thud.store.v1`. Photos are bounded to 2 MB each. Storage errors roll back writes. Image pins use normalized photo coordinates. Full-resolution PCM is released after analysis. Clearing app/site data removes saved records.

### Reference library schema

Reference profiles are local, user-supplied data; THUD does not seed fabricated labels. Storage key: `thud.anchors.v1`. The Reference Library accepts either one anchor, an anchor array, or the exported wrapper:

```json
{"schema":"thud.reference-library","version":1,"anchors":[...]}
```

Each anchor contains `intent`, `object`, exactly three clean `taps`, and optional `level`, `label`, and `metadata`. Slab labels are `Solid` / `Hollow`; Material labels are `Original` / `Suspect`; Devices labels are `Reference` / `Anomaly`. Legacy unlabeled anchors remain usable as a single `Reference` baseline. Imports validate the whole batch before an atomic local write; invalid data leaves the previous library unchanged. Results persist classifier method, label/status, score, margin, prototypes, selected profiles and tap IDs for auditability. Demo and microphone/imported references remain separate.

### Architectural floor plans

Projects may store one validated JPEG, PNG or WebP floor plan under `thud.store.v1` (maximum 2 MB), including decoded pixel dimensions. Findings retain normalized `planX` / `planY` independently from normalized photo `x` / `y`. Replacing a plan requires explicit reset of existing plan pins; storage failures roll back the plan and annotations. Handover PDF rendering uses the same contain-fit geometry and numbered status-colored pins as the interactive map.

The floor-plan API is `ThudStore.updateFloorPlan(project, {bytes, name, resetPins})`. A plan pin requires a project plan and a finite normalized coordinate pair; a photo pin still requires its own photo and coordinate pair.

## Verification

Web smoke at 390×844 exercised touch-swipe selection, Fruit mass proxy/save, Slab three-tap reference and contrast anomaly, Vessel 0/50/100 synthetic anchors with 50% interpolation, all Scope views, map creation, human marking, gallery upload, image pin, persistence after reload and unsaved-edit protection. PDF bytes were captured and parsed: map, synthetic provenance, saved context and photo page present. Browser native download was not observed. Reduce Motion navigation was exercised. Physical microphone accuracy, iPhone frame rate, native camera permissions, haptics and native sharing still require real-device verification. A Flutter web keyboard-event error occurred during automated text entry; subsequent photo/save/report paths completed.

Current compact-result smoke at 390×844: direct mode swipe; Fruit mass/type persistence; Devices reference match/difference; Material three-tap teaching; Slab result-to-handover, accepted cell mark and saved report navigation. An injected steady audio stream exercised the microphone pipeline's noisy retry state, with no metrics/save/map controls in the retry recap. Exported PDF blob (23,823 bytes) was parsed with pdftotext: handover counts, reference match, measured frequency/ring/HII/SNR, map and photo present. Analyzer and release web build passed; final browser run had no reported errors.

## Design references

The project includes Meng To's `no-ai-design-slop`, `dither-background` and `high-contrast-skeuomorphic-clean` skills.

Interface refinement follows **Unslop by Damir Sokolovsky ([@heydamir](https://x.com/heydamir))**, [heydamir/unslop](https://github.com/heydamir/unslop), licensed CC BY 4.0. The installed skill and original attribution are preserved under `.agents/skills/unslop`. The current user-directed adaptation uses mode-specific colors, playful subject glyphs, swipe transitions and result-first typography.
