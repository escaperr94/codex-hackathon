# THUD learned ML/DL integration handoff

## Correction to the previous integration

The previous implementation only ran a YCB material hint while the main app still used handcrafted peak/decay/HII comparisons, fill interpolation, and a firmness formula. That did not fulfill the requested ML/DL integration. Those decision paths have now been replaced. The obsolete material-hint runtime and asset were removed; the research artifact itself remains in `models/`.

## What actually runs

1. Capture normalized mono PCM at 48 kHz, or generate an explicitly marked demo waveform.
2. Select the strongest sample in the loudest 20 ms block, center a 975 ms window, zero-pad its edges, and resample to 16 kHz with anti-aliased Blackman-windowed sinc/polyphase filtering. No amplitude normalization is added.
3. Invoke the existing finalized pretrained **YAMNet TFLite neural encoder**: `float32[15600] → float32[1024]`.
4. Train a small local model using the user's encoded calibration examples. Persist fitted parameters and example embeddings; subsequent decisions come from that model, not acoustic scalar thresholds.

Encoder asset: `assets/models/yamnet-waveform-embedding-f32.tflite`, copied from `models/artifacts/final/research/`. Version: `yamnet-waveform-embedding-f32-v1`. Android/iOS use `tflite_flutter 0.12.1`; web runs the same model in a dedicated local WASM worker. There is no cloud inference or runtime CDN dependency. Scope remains an acoustic explorer, not an invented diagnostic classifier.

## Calibration use cases

Shipped/source artifact SHA-256 both equal `420472f3eb419c7950948be7b291a6599c91f8bfe326d080151ed0364cba271f`.

| Mode | User teaching flow | Learned model and result |
|---|---|---|
| Slab | Name exact construction/reference; three clean taps | Fit normal centroid in 1024D neural space and a leave-one-out distance threshold. Predict match/difference; not bonded/debonded ground truth. |
| Material | Name original/accepted sample; three clean taps | Same learned normal-reference model. Compare signatures; no universal composition/authenticity claim. |
| Devices | Power off/unplug; name casing/reference; three gentle taps | Same learned normal-reference model. Not operating-machine fault diagnosis. |
| Vessel | Same named container, at least three known fill percentages, three taps per percentage | Fit regularized ridge regression on neural embeddings. Reject embedding-space OOD and predictions outside calibrated label range. No peak-frequency interpolation. |
| Fruit | Same fruit type/cultivar name; known Unripe/Ripe/Overripe examples, at least two classes and three taps per class | Fit nearest-centroid classifier on normalized neural embeddings, with class-local leave-one-out support and ambiguity rejection. Further labelled fruits accumulate. No mass-based firmness verdict. |

This is **on-device training of lightweight models over frozen deep features**. YAMNet weights are pretrained and remain frozen; no unsupported claim of backpropagation/fine-tuning is made. Three taps are the minimum UI fit, not sufficient evidence of field generalization. Fruit training should use multiple independent fruits with independently known labels, not guesses from the model itself. Reference spread/support is not a calibrated confidence probability.

## Persistence and migration

- `Measurement` stores encoder version and 1024D embedding; intent/context changes and storage roundtrips retain them.
- `thud.calibration.v2` stores reference examples and fitted models in one write. Keys separate intent, exact object name, and demo/real provenance. Model JSON validates encoder/version/dimension and finite parameters.
- Existing `thud.anchors.v1` reference traces are preserved/readable but are not accepted as learned calibration. Their original PCM was discarded, so users must record new examples. No fake embeddings or scalar fallback are used.
- `thud.store.v1` saved records remain readable. New result metadata includes model status, method, example IDs, fitted distance/threshold, estimated fill or maturity label. Reports retain these distinctions.
- Full PCM is released after analysis/encoding. Embeddings, fitted parameters and traces stay local. Clearing app/site data deletes them.

## Existing trained models versus local training

The finalized durian/coconut heads **were trained** in the research work, as were source-specific material/motor/leakage heads. They are not interchangeable with app calibration. Durian requires a 112D multi-impact engineered frontend; coconut requires its 98D numeric-waveform frontend and three-ridge pooling. Neither accepts YAMNet's 1024D embedding. Their original datasets, contracts, OOF metrics and exports remain unchanged.

Rather than falsely feeding those heads unrelated features or presenting Apple/Pear/Melon mass formulas as ML, Fruit now has an actual user-labelled neural calibration flow. This is a locally fitted model, **not** the public-data durian/coconut classifier and does not inherit their reported accuracy.

No finalized learned tile-bonding/static-fill physical-state model existed in the release. Local deep-feature anomaly/regression learning now serves those comparison/calibration use cases, but does not create missing physical ground truth. Pouring/shaking fill classes, hydrophone/contact leaks, SDNET contact impact-echo, GPLA leak regimes and single-motor datasets are not silently repurposed as phone-tap diagnoses.

## Verification observed

- Installed temporary Flutter 3.50.0 / Dart 3.14.0 SDK for verification; official SDK SHA-256 matched the release manifest.
- `flutter pub get` resolved microphone, TFLite, web interop and test dependencies.
- `flutter analyze`: no issues.
- `flutter test`: all 9 tests passed. Coverage includes malformed/edge PCM, alignment, passband/alias rejection at 48/44.1 kHz, learned reference/fill/class predictions, OOD/ambiguity/JSON validation, calibration persistence/provenance separation, legacy migration, and a result UI test with identical acoustic metrics but different neural embeddings.
- `flutter build web --release --no-web-resources-cdn`: passed, including Wasm dry run.
- Actual TFLite desktop Interpreter ran three shipped real-audio parity fixtures: maximum embedding errors `7.63e-6`, `2.86e-6`, `2.38e-6` versus reference.
- Actual browser WASM TFLite ran the same three real-audio fixtures: maximum errors `6.79e-6`, `3.04e-6`, `3.58e-6`.
- Four real supplied motor recordings ran through the **app's Dart DSP → browser TFLite → Dart local reference fit → JSON restore → prediction**. This verified the executable chain, not diagnosis accuracy: the repaired recording remained within the fitted reference support. No success/accuracy claim is made from that result.
- Interactive app smoke initially exposed relative worker imports failing under a blob worker URL. Fixed by sending an absolute local runtime base; the blob-worker regression scenario now returns 1024 finite embedding values.
- Final compiled Flutter app smoke at 390×844: actual demo capture encoded successfully; three reference taps trained a Slab model, matched the original capture, saved the result, and retained the model/match after reload. Long-ring contrast produced `Different from learned reference`.
- Vessel UI: nine actual demo encoder invocations across 0/50/100% teaching levels trained/persisted the local ridge model; the 50% capture returned `50% estimated fill`.
- Fruit UI: taught Ripe and Unripe with identical demo waveforms to exercise contradictory/ambiguous data. Class model persisted, but prediction correctly displayed `Unknown fruit signature` with no save action. Tests independently cover supported class predictions with separated examples.
- Final browser run reported no app page errors; all runtime/model assets were served locally. Visual checks covered reference match, learned fill and unknown-fruit result surfaces.

## Runtime provenance and licensing

Official web runtime package: `@tensorflow/tfjs-tflite@0.0.1-alpha.10`, https://registry.npmjs.org/@tensorflow/tfjs-tflite/-/tfjs-tflite-0.0.1-alpha.10.tgz. Unmodified runtime files and Apache license live in `web/audio_encoder_runtime/`. YAMNet provenance and research model permission limits remain documented under `models/docs/`. Review exact model/checkpoint/dataset redistribution obligations before product distribution; a successful export is not license clearance.

## Remaining limits

- No physical Android/iPhone microphone accuracy, battery/thermal/latency, native inference or delegate validation in this session. Native Android/iOS implementations compile through Dart analysis but were not executed on a device.
- Windows native inference is not packaged: this repo has no Windows platform target or bundled `libtensorflowlite_c-win.dll`. Browser inference works on this workstation.
- Embedding support rejection cannot prove all domain shifts are caught. Local reference models can be too broad or too narrow; user-labelled data can be wrong.
- App preprocessing is explicit and versioned; it is not claimed byte-identical to every historical research frontend. Local training and inference use the same app frontend.
- Scheduled pull script remains from the earlier task; no additional source pull or commits were made for this correction.
