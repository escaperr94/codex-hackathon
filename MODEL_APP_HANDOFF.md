# THUD learned ML/DL integration handoff

## Correction to the previous integration

The previous implementation only ran a YCB material hint while the main app still used handcrafted peak/decay/HII comparisons, fill interpolation, and a firmness formula. That did not fulfill the requested ML/DL integration. Those decision paths have now been replaced. The obsolete material-hint runtime and asset were removed; the research artifact itself remains in `models/`.

## What actually runs

1. Capture normalized mono PCM at 48 kHz, or generate explicitly marked deterministic demo audio. The shared analyzer detects isolated impacts or distributed operating windows, rejects malformed/clipped/low-quality captures, and retains attack/ring windows.
2. Compute FFT/STFT, MFCC, spectral centroid/flatness, modal peaks, damping and decay, with robust median/MAD aggregation and a bounded uncertainty index. Speech-family contamination is checked using event scores over the complete recording; no speech heuristic is used. In impact mode, a tap whose window (or an overlapping full-recording patch) is speech-classified is dropped from both the pooled embedding and the physical features; the take is rejected only when every tap is contaminated or kept taps still score ≥ 0.30. Operating recordings still reject on any speech-family score.
3. Invoke the finalized joint pretrained **YAMNet TFLite encoder**: `float32[15600] → float32[1545]` (`1024` embedding values plus `521` AudioSet event scores). Isolated impact embeddings are coordinate-median pooled; operating recordings use distributed windows. Input uses anti-aliased Blackman-windowed-sinc/polyphase resampling without amplitude normalization.
4. Fit lightweight local models over the frozen neural features plus eight bounded physical features (1,032-dimensional calibration vector). Persist fitted parameters, provenance and reference embeddings; no cloud inference or backpropagation occurs on device.

Encoder asset: `assets/models/yamnet-waveform-embedding-events-f32.tflite`. Version: `yamnet-waveform-embedding-events-v2`; SHA-256 `e8c20545218dc41fd888dbde06debf15d00e8cb6674a1742221c98b1cf1d71c6`; size 15,009,820 bytes. Android/iOS use `tflite_flutter 0.12.1`; web uses a dedicated worker with locally bundled TFLite WASM. `usable` requires actual event-classifier evidence and no speech contamination. Scope exposes the saved STFT, modal features, damping and pairwise neural comparison.

## Calibration use cases

| Mode | User teaching flow | Learned model and result |
|---|---|---|
| Slab | Name exact construction/reference; three clean taps | Labelled `Supported` / `Suspected Unsupported` examples fit a local prototype classifier. Normal-only references remain `Inconclusive`; specimen IDs enable leave-one-tile-out evaluation. Not certified bond/defect detection. |
| Material | Name original/accepted or suspect examples; three clean taps | Few-shot neural/DSP reference comparison with unknown rejection. No composition/authenticity claim. |
| Devices | Power off/unplug for tap references, or select safe operating capture; three clean taps/reference windows | Normal-baseline difference detection, with sustained audio treated separately from impact decay. No named-failure diagnosis. Never touch powered or moving parts. |
| Vessel | Same named container, at least three measured fill levels, three taps per level | Regularized local regression returns estimated fill, empirical support interval and calibration status. Volume appears only with consistent capacity metadata. |
| Fruit | Same fruit type/cultivar; labelled firmer/softer examples, at least two classes and three taps per class | Relative acoustic firmness/maturity similarity only. No exact ripeness, spoilage, sweetness, internal-defect or food-safety claim. |
| Scope | Any clean capture, optionally save a reference | General acoustic evidence surface: waveform, spectrum, STFT, resonance/damping and neural similarity. |

This is **on-device fitting over frozen features**, not weight training. Three taps are a UI minimum, not field-generalization evidence. Demo and microphone/imported references remain separate.

## Persistence and migration

- `Measurement` stores the current encoder version and 1,024-D embedding; context stores capture mode, uncertainty, serialized acoustic evidence, fitted calibration and provenance IDs.
- `thud.anchors.v1` is the serialized reference library. Legacy `thud.calibration.v2` data is migrated/read with compatibility checks. The fused calibration contract is versioned `yamnet-calibration-events-v3`; incompatible old embeddings require re-recording rather than silent reuse.
- `thud.store.v1` remains readable. Project metadata includes building/floor/unit, validated floor-plan tile geometry and stable tile IDs. Removing linked tiles or replacing a plan requires explicit unlink/reset.
- Full PCM is released after analysis/encoding. Embeddings, fitted parameters, traces and reports remain local; clearing app/site data deletes them.

## Existing trained models versus local training

The finalized durian/coconut heads and source-specific material/motor/leakage heads use incompatible engineered frontends or acquisition conditions. They remain research artifacts, not silently repurposed app diagnoses. Fruit therefore uses labelled local frozen-feature calibration and does not inherit public-dataset accuracy.

No finalized learned tile-bonding/static-fill physical-state model existed in the release. Local calibration does not create missing physical ground truth.

## Verification observed

- `flutter analyze`: passed with no errors after the merged source integration; existing style/lifecycle findings remain informational only.
- `flutter test`: the full suite exceeded the 600-second verification command window and was terminated without a result. Focused acoustic, tile, and persistence checks passed 19 tests; use the focused result as exercised evidence, not as a substitute for the full suite.
- `python -m pytest models/tests/test_audio_events.py`: passed (`1` test; TensorFlow Lite deprecation warning only). This validates the joint model shape, built-in operators, real Sound-of-Water fixture parity, unchanged 1,024-D prefix and official 521-score tail.
- Actual desktop TFLite parity on three existing real 975-ms waveform fixtures: maximum embedding errors `7.62939453125e-06`, `2.86102294921875e-06`, `2.384185791015625e-06`; joint input/output `[15600] → [1545]`; observed CPU invocation times 4.08, 4.97 and 3.41 ms in the verification environment.
- The model exporter rebuilds the joint asset from the cached official YAMNet SavedModel without a network download or new training: `python models/experiments/export_audio_events.py`.
- Tile mapping is offline projective/grout geometry with manual corners/grid and editable stable IDs; no MobileSAM or ARKit runtime was adopted. Counts are visible image regions and require review.
- No physical Android/iPhone capture, native delegate, battery/thermal, or field-accuracy validation was performed. Browser inference is the exercised runtime on this workstation.

## Model-selection and licensing decisions

The shipped path uses the smallest finalized artifact already validated for this app contract: YAMNet provides a 1,024-D embedding and AudioSet event scores in one local TFLite model. CLAP, BEATs, AST, MediaPipe-YAMNet, MobileSAM, YOLO/Core ML and ARKit were not added: no compatible cached weights/runtime plus app contract and no device evidence justified their size or integration cost. Research evidence bounds the choice: Sound-of-Water YAMNet shape classification was 58.33% versus a 50.00% majority baseline; frozen YAMNet DCASE valve AUC was 0.528; EfficientAT's best inspected DCASE variant was AUC 0.6095 with a wide CI. These do not validate THUD physical diagnoses.

Review exact YAMNet, TFLite runtime, dataset and bundled web-runtime redistribution obligations before product distribution. A successful local export/parity check is not license clearance.

## Remaining limits

- No physical Android/iPhone microphone accuracy, native inference/delegate, battery/thermal or route/device matrix validation in this session.
- Speech event scores are a conservative capture-quality gate, not calibrated speech probabilities. Capture uncertainty and fill intervals are not calibrated diagnostic confidence.
- Learned user labels, reference coverage and embedding support can be too broad or too narrow; no domain-shift detector proves all bad captures are rejected.
- Optional ARKit tracking/raycasting and MobileSAM segmentation were intentionally skipped. Offline normalized floor geometry is the required supported location path; textured, occluded or diagonal floors may need manual correction.
- The scheduled pull script remains from the earlier task; its first automatic fast-forward encountered divergent local/upstream history. The source was incorporated via an explicit merge while retaining local ML integration; no reset/clean/stash was used.
