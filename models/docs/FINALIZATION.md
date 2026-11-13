# Final model handoff

## Workspace

All first-party model code, experiments, tests, project configuration, local data and artifacts are in `models/`. All project Markdown documents and generated reports are in `models/docs/`. Upstream dependency README/license files inside downloaded source/data remain with their sources. Repository-level non-model files are not moved.

From the repository root, enter `models/` before running the commands below. Installation uses `models/pyproject.toml`. Finalized artifacts live in `artifacts/final/`, distinct from historical/synthetic pilot exports in `artifacts/`. The final folder is exempted from the artifact ignore rule; raw downloads/caches remain ignored. Files are local research outputs, **not automatically cleared for redistribution**.

## Adoption decisions

| Task | Finalized research candidate | Decision and evidence |
|---|---|---|
| Durian ripeness | `fruit/durian/deployment-float32.tflite` | 112D physical/MFCC features → 3 softmax scores, scaler included. Nested whole-fruit BA 82.94% versus original 81.54%; reused folds, exploratory, not a proven field improvement. |
| Coconut maturity | `fruit/coconut/deployment-float32.tflite` | 98D ridge features → 3 logits. Keep original balanced-logistic recipe, 69.87% fruit-held-out BA. Expanded selector was worse (66.63%) and is not the release policy. Pool three ridge logits per fruit. |
| Material recognition | `classifiers/{ycb,realimpact}_material_yamnet.tflite` | Standardized 1024D mean YAMNet features → LinearSVC margins/class index. Separate models/domains; not a universal material inspector. Complete one-second waveform versions are under `waveform/`. |
| Pucks / phone motor | `classifiers/pucks_mfcc-logmel_logistic.tflite`, `motor_mfcc-logmel_logistic.tflite` | Tiny spectral/MFCC heads. YAMNet/RBF comparators retained, not all recommended. Controlled fabrication or single-motor confounds remain. |
| ACM-S2 fill classes | `classifiers/acm_s2_fillclass_logistic.tflite` | Three pouring/shaking classes; not static tap percentage regression. External ACC benchmark remains historical, not converted by these exports. |
| Leakage | `classifiers/pipe_leak_{logistic,rbf}.tflite`, `compressed_air_{logistic,rbf}.tflite` | Separate source-specific models, not seal integrity. Pipe grouping corrected before evaluation by merging hash-linked filename groups. Missing session metadata still prevents independent-site claims. |
| Jackfruit / GPLA-12 / SDNET | Individually named heads under `classifiers/` | Retained research comparators only. Jackfruit has supplied features but no reproducible raw-audio frontend; GPLA labels are leak-condition regimes, SDNET uses contact impact-echo sensing. |
| Machine anomaly | Per-section normal-reference scorers under `research/` | Cosine/Gaussian scorers, not complete mobile encoders. Training normals only. DCASE2023 and MIMII DG section00 are identical audio, not independent replication. |
| Pouring shape | `research/pouring-shape-yamnet-three-bins.tflite` | Training-validation-selected comparator only; 33.33% heldout container accuracy is poor. Do not adopt as a fill estimator. |
| Tile bonding / static tap-fill / vacuum / moisture | No finalized learned model | Matching real ground truth remains absent or blocked. Synthetic tile/fill pilot exports are not in the final release. |

Filenames above are relative to `models/artifacts/final/`. Probability outputs are classifier scores, not calibrated physical-state confidence. SVC models with `class_index` must use that output: libsvm vote ties can differ from simple score argmax. Exact class order, feature names, input shape, preprocessing, training provenance and permission caveats are adjacent to each artifact.

## Review and tuning

Finite fruit feature-block/class-weight/regularization ablations use training-only inner selection with whole-fruit outer folds. Full-data deployment hyperparameters are selected separately from the outer-score ranking. Conservative coconut baseline is retained rather than replacing it with a worse expanded selector.

Pipe review found six identical complete-WAV hash sets crossing filename groups (12 clips), confirmed by decoded PCM. Transitive merging reduces 319 groups to 314; corrected whole-component OOF reports are retained beside the final models. An earlier review's approximate 26-clip estimate was not reproduced and is not a final claim. No grouping can recreate missing physical session/site identity.

Existing historical benchmark JSON files were not relabelled as new runs. Their hashes/scores remain historical provenance; final scripts/reports carry current source and artifact hashes. See [fruit evidence](FRUIT_FINALIZATION.md), [classifier evidence](CLASSIFIER_FINALIZATION.md), [research encoder/scorer evidence](RESEARCH_FINALIZATION.md) and [dataset inventory](DATASETS.md).

## Reproduce and verify

```powershell
# From models/, using the existing downloaded datasets/caches:
python -m experiments.finalize_fruit
python -m experiments.finalize_classifiers
python -m experiments.finalize_research
python -m experiments.finalize_waveform

# Offline release-only parity smoke; does not need original datasets:
python -m experiments.verify_release

# Existing regression suite:
python -m pytest -q
```

`verify_release` opens every final `.tflite`, checks expected model/fixture hashes, exact tensor shapes and dtype, rejects custom/Flex FlatBuffer operators, allocates and invokes the Interpreter, checks finite outputs, and compares compact real-input fixtures to desktop reference scores/class indices. Evaluation-fold fruit exports receive allocation/finite smoke independently; their complete held-out parity is recorded by the fruit producer. `release-verification.json` records observed results. This is numerical/software verification, not a new accuracy evaluation.

Float32 is the final default. No FP16/int8 file is called ready without representative-input parity; historical FP16 YAMNet drift is a reason not to claim quantization as an automatic upgrade. Head-only exports intentionally leave the exact DSP/encoder outside the TFLite and are not represented as complete phone capture pipelines. The waveform material models include YAMNet DSP and task head but still require the documented external decoding, resampling, crop and domain-specific normalization.

## Unmeasured gate

No physical iPhone/Android inference, Flutter integration, delegate support, minimum-device peak memory, energy or thermal validation was performed. TensorFlow 2.19 desktop Interpreter emits its deprecation warning; migration to a different LiteRT/mobile runtime requires re-running these parity fixtures. Dataset/checkpoint license review remains necessary before commercial distribution. Blocked tile/CCM/watermelon and unsupported physical-label tasks are documented in the dataset inventory, not silently replaced by unrelated classifiers.
