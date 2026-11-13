# THUD baseline implementation plan

Historical implementation plan, retained for provenance. The working directory is now `models/`, and Markdown documentation is under `docs/`. Later pretrained loaders, transfer heads and TFLite exports supersede the original unimplemented-status notes below; consult `README.md` and finalization reports for the current state.


## Scope and decision

Build a local Python research CLI for isolated microphone tap WAVs: physical/MFCC features, tile screening with L2 logistic regression, calibrated fill regression with ridge, group-held-out evaluation, and portable JSON linear-model inference. No existing code/callers/tests/config/exports exist to migrate. No recordings are uploaded; input WAVs are never modified. Do not build Flutter, claim device performance, or ship synthetic models as physical inspectors.

## Sequence and verification

1. Define contracts here and inspect available runtime. Verify that dependencies are available before implementing.
2. Implement `thud/audio.py`. Verify deterministic damped-tone frequency recovery, amplitude robustness, sample-rate handling, and rejection of silent/nonfinite/clipped/too-short audio.
3. Implement `thud/models.py` and `thud/__main__.py`. Verify train-only scaling, whole-object/session folds, manifest validation, JSON save/reload prediction parity, and both task outputs.
4. Run the actual CLI against a temporary synthetic WAV/CSV pilot: analyze, evaluate tile/fill, train, predict; exercise invalid input. Keep behavioral tests; delete generated pilot files afterward. Synthetic outcomes prove software execution only.
5. Document installation, manifest/capture contracts, commands, observed smoke evidence, and the real-data/device experiments still required.

## Shared implementation contracts

- Python >=3.10; existing NumPy, SciPy and scikit-learn. Standard-library argparse/CSV/JSON; pytest for development. No librosa, TensorFlow, pickle or model downloads required for this baseline.
- `thud/audio.py`: `FEATURE_VERSION = 1`, stable `FEATURE_NAMES`; `analyze_wav(path)` returns a JSON-safe dict with `sample_rate`, `duration_s`, `onset_s`, `features` (ordered name-to-float mapping), and quality metadata. One WAV represents one isolated tap with pre-impact baseline and sufficient ring-down. Reject invalid capture rather than emit a physical diagnosis. `ValueError` is the user-input error contract. Higher-rate PCM is read but never overwritten. DSP parameters are exploratory, not validated physical thresholds. Do not mislabel an arbitrary decay fit T20 or EDT.
- `thud/models.py`: `load_dataset(manifest, task)` returns `(records, X, y)` with one feature row per WAV; `fit_model(X, y, task, family=None)` returns a JSON-safe portable linear artifact; `predict_model(model, X)` returns a NumPy prediction array; `validate_model(model)` rejects malformed/incompatible artifacts; `evaluate(records, X, y, task, group_by)` returns JSON-safe held-out predictions and metrics. `group_by` is `object_id` or `session_id`. No hyperparameter search. Fit scaling within each training fold. Binary tile labels are exactly `bonded` / `debonded`; score/probability is not a calibrated physical defect probability. Fill target is volume-labelled percentage points in [0,100]; predictions are not silently clamped.
- CSV columns: `path,label,object_id,session_id,recording_id`; fill additionally requires `family`, one family per dataset. Paths resolve relative to the manifest. Require nonempty metadata, existing unique paths, finite valid targets, and consistent object/session identity per recording_id. Reject duplicate recording bytes to prevent aliases crossing folds. Metadata must identify original recordings before any cropping/augmentation; folds group all rows from the same held-out object/session. Cannot detect undeclared crop/augmentation leakage automatically.
- Model artifact: schema/version/task/feature version and ordered names, training mean/scale, coefficients/intercept, fixed tile class order or fill family. JSON only, finite arrays and strict shape checks; no executable deserialization. Export standardized linear coefficients from scikit-learn; inference needs NumPy only after features. Fill training requires at least three distinct calibration levels. Every tile training fold must contain both classes.
- CLI: `python -m thud analyze WAV`; `evaluate CSV --task tile|fill --group-by object_id|session_id`; `train CSV --task tile|fill --output MODEL.json`; `predict MODEL.json WAV [--family FAMILY]`. Fill prediction requires matching family explicitly. JSON stdout; errors to stderr and nonzero exit. Output model creation refuses overwrite. No network use.

## Research gates, not implemented experiments

- Real tile pilot: at least eight known-state specimens, two sessions/users, three three-tap bursts per point, safe external tool, phone 10–15 cm away. Split isolated taps while retaining original recording identity. Whole-specimen and held-out-session evaluation; report false positives/defect recall/coverage/repeatability.
- Real fill pilot: two safe unpressurized vessels at measured 0/25/50/75/100% volume, fixed contents/cap/support/tap point, two sessions; separate vessel-transfer and changed-placement stress results. Peak switching/nonidentifiability is a sensing limitation, not fixed by monotonic interpolation.
- Later pretrained comparison: YAMNet embeddings + the same heads/splits, then EfficientAT mn04_as. Preserve original PCM and resample only the encoder branch. Add a task CNN only if real held-out evidence warrants it.
- Device gate: converted candidate on physical iPhone 11 offline, desktop parity, 20 warm-ups + 100 timings, cold load/peak memory and 10-minute thermal/energy observation. Capture/window wait separate from compute. Not measured here.

## Deep-learning extension (implemented)

- `thud/deep.py` implements a task-trained PyTorch log-mel depthwise-separable CNN: fixed 16 kHz/1-second input, 64 mel bins, 512 FFT, 400-sample window, 160-sample hop, tile classification or one-family fill regression, checkpoint validation, grouped evaluation, and TorchScript export.
- `thud/encoders.py` implements no-download contracts/wrappers for already-loaded YAMNet and EfficientAT encoders. It enforces their sample-rate/frontend metadata and distinguishes YAMNet 521 scores from 1,024D embeddings.
- CLI commands: `deep-train`, `deep-evaluate`, `deep-predict`, and `deep-export`.
- Pretrained artifact loading, transfer-learning heads, Core ML/mobile runtime integration and physical-device measurements remain intentionally unimplemented. The task CNN's synthetic smoke behavior is not physical-task accuracy.
