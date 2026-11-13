# Research classifier finalization

Run `python -m experiments.finalize_classifiers` from `models/`. No downloads or historical report writes. Full machine-readable contracts, hashes and numerical evidence are in `artifacts/final/classifiers/inventory.json` and each adjacent model JSON.

## Export inventory

All models are builtin-only float32 **feature heads**, not raw-audio models. StandardScaler and complete score/prediction function are included. `scores` follows declared sklearn class order; use `class_index` for classification because SVC OVO vote ties need not equal OVR-score argmax. SVM scores are not probabilities. No phone hardware was tested.

| Artifact | Real feature rows | Bytes | Full-data score max absolute error |
|---|---:|---:|---:|
| `acm_s2_fillclass_logistic.tflite` | 21 | 3092 | 3.42653e-07 |
| `acm_s2_fillclass_rbf.tflite` | 21 | 18132 | 1.12336e-07 |
| `pipe_leak_logistic.tflite` | 1000 | 3200 | 1.11729e-06 |
| `pipe_leak_rbf.tflite` | 1000 | 117720 | 4.23337e-05 |
| `ycb_material_mfcc_logmel.tflite` | 131 | 7684 | 7.9922e-07 |
| `ycb_material_yamnet.tflite` | 131 | 42484 | 5.13976e-06 |
| `realimpact_material_mfcc_logmel.tflite` | 270 | 3964 | 7.65642e-07 |
| `realimpact_material_yamnet.tflite` | 270 | 17884 | 4.16052e-06 |
| `pucks_mfcc-logmel_logistic.tflite` | 405 | 3092 | 1.09001e-06 |
| `pucks_mfcc-logmel_rbf_svm.tflite` | 405 | 146932 | 2.46618e-06 |
| `pucks_yamnet_logistic.tflite` | 405 | 18036 | 8.77535e-07 |
| `pucks_yamnet_rbf_svm.tflite` | 405 | 1651268 | 5.50825e-06 |
| `motor_mfcc-logmel_logistic.tflite` | 60 | 3200 | 4.75214e-08 |
| `motor_mfcc-logmel_rbf_svm.tflite` | 60 | 16892 | 5.49517e-07 |
| `motor_yamnet_logistic.tflite` | 60 | 10672 | 8.49659e-08 |
| `motor_yamnet_rbf_svm.tflite` | 60 | 241588 | 1.44834e-06 |
| `compressed_air_logistic.tflite` | 48 | 3200 | 2.84021e-07 |
| `compressed_air_rbf.tflite` | 48 | 16892 | 8.47692e-07 |
| `jackfruit.tflite` | 253 | 2696 | 1.67789e-07 |
| `gpla12.tflite` | 684 | 254332 | 1.26241e-06 |
| `sdnet_spectral_svm.tflite` | 1936 | 516724 | 1.13296e-06 |
| `sdnet_small_spectral_mlp.tflite` | 1936 | 13644 | 3.44008e-07 |

## Selection and evaluation

Existing full-data Pucks/motor and secondary-table heads were reused. Both fixed logistic/RBF families retained where previously compared; no deployment family was chosen using outer held-out winner scores. YCB/RealImpact full-data LinearSVC C uses object-rank training CV. SDNET full-data SVM C/kernel uses training-deck CV; fixed tanh32 MLP retained as comparator. ACM-S2/pipe caches did not exist: real features were recomputed once; existing caches reused elsewhere.

`oof-parity.json` separates historical/replayed per-fold TensorFlow scorer parity from full-data **TFLite** parity. No OOF TFLite parity is claimed. All deployment TFLites were compared to sklearn on every real cached feature row, finite edge inputs and meaningful near-boundary probes. `scorer-selfcheck.json` records binary sign/zero tie, class order, OVO voting, logistic, tanh MLP and RBF regression cases. No full-fit accuracy is passed off as generalization.

## Pipe leakage correction

`pipe-leak-hash-group-audit.json` merges all filename groups connected by identical complete-file SHA256 before LOGO. New per-family OOF reports preserve the historical report untouched; grouping does not establish missing session/acquisition independence. See refreshed metrics there, not a deployment training score.

## Frontends, labels and limitations

Exact feature order, source reference, scaler, class labels, training provenance and real-input smoke evidence are adjacent to each TFLite. ACM-S2 is21pouring/shaking clips, codes0/1/2=empty/50%/90%, not tap-volume regression. YCB uses robot impact OGX; RealImpact uses force-deconvolved impulse responses, not raw PCM. Pucks known ABS/PA2200/PA12 materials remain manufacturing-process confounded; unknown puck_A excluded. Motor is one physical motor before/after repair, not general fault diagnosis. Compressed air is48selected emissions, not seal/vacuum inference. Pipe recordings may be hydrophone/contact logger, not verified phone airborne audio.

Jackfruit is an explicitly **supplied-feature-only** maturity head:27columns preserved from XLSX, no raw audio or exact reproducible acoustic frontend available, so no audio smoke or ready-to-capture claim. GPLA12 accepts44features from1460already-preprocessed numeric values;12classes are pressure/microphone/noise combinations among leak signals, not leak/no-leak. Their CV is record/content-disjoint only: unseen-fruit/run generalization blocked by missing acquisition IDs. SDNET accepts native102.4kHz impact-echo sensor features; sound/above-rebar/below-rebar classes are not phone/tile claims. CCNY is excluded because per-signal truth remains ambiguous.

## Permissions and device gate

Local research only. Pucks has conflicting metadata CC-BY-NC-ND4.0/PDF CC-BY-SA4.0 terms; compressed air CC-BY-NC-ND4.0. Motor CC-BY4.0. YCB source node license null. Jackfruit repository MIT does not establish independent dataset provenance; GPLA repository EPL-2.0 requires dataset review. Other dataset/model permissions must be reviewed separately before redistribution or product use. Model hashes do not grant rights. No physical iPhone latency/memory/energy, capture parity or held-out THUD product validation measured.

## Source links and leakage-count clarification

[ACM-S2](https://zenodo.org/records/4770439) and [pipe-leak audio](https://zenodo.org/records/18631450) dataset metadata state CC BY4.0; preserve attribution. [YCB](https://osf.io/4tcp6), [Pucks](https://zenodo.org/records/7551338), [phone motor](https://data.mendeley.com/datasets/bxcnxfm94c/5), [compressed air](https://zenodo.org/records/7551606), [jackfruit](https://github.com/Abhinandan-Hooli/jackfruit-acoustic-maturity-classification), [GPLA12](https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12), [SDNET](https://commons.und.edu/data/19/) and the [pinned structural derivative](https://github.com/ehoxha91/impact_echo_datasets) remain protocol-specific research sources.

The pipe audit found72clips in36exact-hash duplicate pairs overall, of which12clips/6hash sets cross filename groups. Decoded PCM and feature-row hashes reproduce the12cross-group count, not the previous reviewer26clip figure. All observed linked filename groups were merged319->314components. Refreshed exploratory LOGO accuracy/balanced accuracy: logistic91.4%, RBF94.3%; acquisition independence remains unknown. SDNET mirror class counts1378/494/64 retain the unresolved one-label discrepancy versus the published original count; no labels were changed.
