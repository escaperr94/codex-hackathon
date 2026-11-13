# Fruit model finalization

Run from `models/`:

```powershell
python -m experiments.finalize_fruit
```

No downloads or installation are performed. The script requires the existing fruit feature caches, source audio/workbook waveform cache, NumPy/SciPy/scikit-learn/TensorFlow/librosa and ffmpeg already used by the original benchmarks. Each process requests at most two CPU threads.

## Scope and selection

These are **research deployment feature heads, not raw-audio models**. Outputs are under `artifacts/final/fruit/{durian,coconut}/`. The script never edits the previous benchmark reports, recordings or caches; their before/after SHA-256 values are asserted and retained in `summary.json`.

The finite space is 18 multinomial L2-logistic candidates per task: three feature blocks × balanced/unweighted class fitting × C in {0.01,0.1,1}. Durian compares all 112 features, the 80 MFCC-only features and the 32 spectral-only features. Coconut compares all 98 features, 26 MFCC features and 95 features excluding log RMS/log peak/crest. This is an amplitude-confound ablation, not proof that the remaining spectral features are acquisition invariant.

The original whole-fruit outer splits are reused verbatim: durian five folds (seed 20261114), coconut four folds (seed 42). Three-fold train-only stratified unique-fruit CV selects each candidate without using the outer observations. Scaling is fitted inside each training fold. Durian preserves equal fruit weight and optional class balancing for both scaler and classifier; coconut preserves its original unweighted scaler and optional class-balanced classifier. Durian selects by pooled inner OOF fruit balanced accuracy; coconut by mean inner-fold fruit balanced accuracy, matching the respective original metrics. Ties use fixed block, weight, C order. Each fixed feature/weight policy and the full nested selector has separate outer predictions.

**Deployment selection is independent of the outer-score ranking:** after OOF evaluation, a fresh three-fold grouped selection on the full dataset chooses deployment hyperparameters, then a final head is fitted on all eligible observations. Coconut conservatively retains the **original all-feature balanced-logistic branch with C in {0.01,0.1,1,10}**; the expanded block/class-weight ablation remains exploratory and is not the coconut release policy. Its four `baseline-oof-fold-N` heads and `release_baseline` predictions are retained separately. Durian uses the exploratory nested logistic selector. Previously inspected outer scores mean neither is a new untouched confirmatory test set. The old RBF-SVM results remain a comparator, not an outer-score-selected deployment fallback.

## Model/input contracts

- **Durian:** 133 eligible iPhone recordings of 133 distinct fruits; class columns `[Unripe, Ripe, Overripe]` correspond to IDs `[0,1,2]`. Original Actual_Ripening_Status labels and ambiguity/contradictory-duplicate/terminal-stage/disease/missing-label exclusions are preserved. Decode compressed audio with ffmpeg to mono float32 16 kHz; reuse `durian_benchmark.impact_windows` and `acoustic_features`. The 100 Hz highpass onset detector selects up to 18 one-second windows (100 ms before / 900 ms after); 20 MFCC mean/std, spectral statistics, ZCR and six band ratios are computed per window, then mean/std pooled over impacts. Export input is **`float32[1,112]`**, not waveform. Output is **`float32[1,3]` softmax scores**; pool recording probabilities per fruit if repeated recordings are supplied. Scores have not been calibrated as physical ripeness probabilities.
- **Coconut:** 127 physical fruits / 381 real ridge waveforms; class columns `[im, m, om]` mean immature, mature, overmature with IDs `[0,1,2]`. All three ridge observations stay together. Reuse `coconut_benchmark.waveform` and `frontend`: contiguous source numeric audio at 44.1 kHz, trailing blank trim only, DC removal, `resample_poly(160,441)` to 16 kHz, whole available 0.25–3 second waveform. Features are 13 MFCC mean/std, 32 logmel mean/std and eight waveform/spectral descriptors. Export input is **`float32[1,98]`**, not waveform. Output is **`float32[1,3]` logistic decision logits, not probabilities**. For the original fruit protocol, evaluate each of its three ridges and average logits before argmax. Do not pool softmax probabilities instead.

Both exports include feature-index selection, the fitted StandardScaler, the linear numerical head and durian-only softmax. Exact indexed feature names, feature masks, labels, frontend source hashes and acquisition contracts are retained in each task's `results.json`. A byte-identical frontend implementation is the reference; changing library versions/defaults or onset behavior requires frontend parity revalidation. No YAMNet or raw-audio frontend is needed for these selected numerical heads.

## Evidence files

- `results.json`: full finite candidate scores and inner split provenance; outer row/fruit splits; per-recording and per-fruit OOF predictions; fixed-policy/majority/baseline metrics; deployment train provenance and contracts.
- `oof-fold-N.tflite`: each selected outer-training-only numerical head. Its `.json` contains SHA-256, shape/dtype/operator inspection, held-out row IDs, maximum error and decision parity. `.npz` retains fitted scaler/linear parameters and feature indices.
- `deployment-float32.tflite`: full-data fitted research deployment head, with separate all-real-cached-row parity evidence in its `.json` and fitted numerical parameters in `.npz`. Its apparent in-sample performance is not offered as generalization evidence.
- `real-audio-smoke.json`: at least one source recording per class decoded/recomputed through the reference frontend and passed to the deployment Interpreter; coconut checks all three ridges for each selected fruit. Includes source provenance, cache-feature differences and numerical predictions.
- `verification-fixture.npz` plus `.json`: six real cached feature rows (two per class), float64 reference scores, source labels/fruit IDs/row indices and output-column names; independently reusable without the source datasets. These probes check deployment parity, not accuracy. Metadata includes model/fixture hashes and score units.
- `summary.json`, `manifest.json`: compact outcomes, versions, unchanged source hashes and artifact byte lengths/SHA-256.

The direct experiment invokes TensorFlow Lite Interpreter on every held-out selected-head input and on every full-data deployment feature input. Assertions require finite float32 input/output with exact shapes, no custom/Flex operators, score maximum absolute error below `1e-4`, and unchanged recording/fruit argmax. A deterministic normalized-origin class-column check plus invalid dtype/shape/nonfinite rejection checks the numerical boundary. No synthetic task model is trained or shipped. Float16/int8 are deliberately not exported without additional real-input evidence.

## Measured desktop results

The direct experiment and real-input smoke completed successfully. The new nested selector is modestly better on the reused durian folds but worse on coconut balanced accuracy; there is **no general improvement claim**.

| Fruit-level OOF metric | Durian original MFCC/spectral | Durian nested ablation | Coconut original balanced logistic | Coconut nested ablation |
|---|---:|---:|---:|---:|
| Accuracy | 83.46% | 84.96% | 70.08% | 70.08% |
| Balanced accuracy | 81.54% | 82.94% | 69.87% | 66.63% |
| Macro F1 | 0.8147 | 0.8299 | 0.6557 | 0.6357 |

The retained conservative coconut release branch independently reproduced the original **70.08% accuracy / 69.87% balanced accuracy / 0.6557 macro F1** on the original outer folds. Its four exported baseline OOF heads had **zero** recording/pooled-fruit argmax changes and maximum logit error **9.27e-6**. This baseline retention is a conservative release-policy decision, not selection of a new outer-score winner.

Durian balanced MFCC-only reached 82.94% balanced accuracy; spectral-only balanced/unweighted reached 65.73%/68.08%. The original all-feature balanced policy was reproduced at 81.54%. Coconut MFCC-only balanced reached 67.80%, while excluding explicit amplitude features gave 67.15% balanced / 68.39% unweighted balanced accuracy. The exploratory regularization grid is `{0.01,0.1,1}`, whereas the conservative coconut release reproduces the original balanced-logistic `{0.01,0.1,1,10}` branch. The wider ablation selector did not improve coconut and must not be marketed as an upgrade. The durian difference is exploratory, not a statistically established improvement.

Fresh full-data grouped inner selection chose **durian all features, unweighted C=1** and **coconut all features, balanced C=0.01**. Their float32 deployment files are **3,896 bytes** and **3,464 bytes**, respectively, including the scaler/head but excluding frontend. These choices are not the best outer-score policy.

The nine exploratory OOF heads (five durian/four coconut) had zero recording and pooled-fruit argmax changes. Maximum OOF score error was **3.89e-7** for durian probabilities and **4.76e-6** for coconut logits. The four additional conservative coconut baseline OOF heads are separately verified and reported in `release_baseline.tflite_parity`. Separate full-data deployment parity exercised all **133 / 381** real cached feature rows with zero argmax changes and maximum errors **1.88e-7 / 1.60e-6**. Three compressed durian audio recordings and nine coconut waveform recordings were recomputed through the reference frontends and exercised in the Interpreter. All exports passed finite output, exact shape/dtype, builtin-only FlatBuffer audit (CUSTOM disallowed), and numerical-boundary/class-column assertions.

Only desktop CPU conversion/parity is established. TensorFlow emitted its `tf.lite.Interpreter` deprecation warning; no unapproved package install or runtime migration was attempted.

## Limits and licensing

These are small single-source fruit-family classifiers, not universal fruit ripeness, firmness, chemical composition or food-safety estimators. Durian collection date/maturity/device and coconut geometry/amplitude/acquisition conditions may confound labels. Coconut has only eight immature fruits and uses a mechanized rubber tap/condenser microphone/acoustic foam rather than phone audio. No independent orchard/farm/session/operator holdout or physical iPhone test has been performed. Head-only parity does not establish a working mobile frontend, phone latency, memory, battery behavior or field accuracy.

Durian source: [Mesa-Satina et al., Zenodo 18603796](https://doi.org/10.5281/zenodo.18603796), dataset CC BY 4.0. Coconut source: [Mendeley hxh8kd3snj version 1](https://data.mendeley.com/datasets/hxh8kd3snj/1), dataset CC BY 4.0; [article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9926183/) has separate CC BY-NC-ND terms. Preserve dataset attribution. No extra model license grant or commercial product suitability is implied; review exact redistribution obligations.
