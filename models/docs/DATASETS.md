# Acoustic Dataset Benchmark Inventory

Workspace paths in this inventory resolve from `models/`. Model code is in `thud/` and `experiments/`; project documentation and generated Markdown reports are in `docs/`. Historical JSON results are preserved rather than rewritten after relocation. Final export verification is separate under `artifacts/final/`.


Status: accessible-corpus research benchmarks completed for the documented subsets; full-release and blocked-data limits remain explicit. Finalized research TFLites, ablations and current verification commands are in [FINALIZATION.md](FINALIZATION.md). Findings are protocol-specific; none establish universal THUD accuracy.

## Objective

Evaluate task-appropriate audio models on public datasets related to THUD, using object/group-held-out splits. Desktop CPU measurements are not edge-device measurements. Keep the sensing protocol and target label explicit; never transfer accuracy between tasks.

## Verified datasets

| Task | Dataset | Evidence and labels | Protocol match | Access / decision |
|---|---|---|---|---|
| Tile/debonding | [SDNET2021](https://commons.und.edu/data/19) | 1,936 field impact-echo signals from five bridge decks; three delamination classes and ground-truth maps | Concrete impact-echo; not tile or phone airborne audio | Candidate proxy; download only the impact-echo subset after license review (~1.81 GB reported) |
| Tile/debonding | [Impact-Echo with Ground-Truth Schematics](https://data.mendeley.com/datasets/997pwzvhgj/2) | 1,824 signals; labels derived from schematics, not per-file labels | Lab concrete impact-echo; not tile/phone | Best bounded structural proxy (~33 MB reported); derive labels with an auditable map |
| Tile/debonding | [Tile acoustic CNN study](https://doi.org/10.3390/buildings16040870) | 2,000 lab taps plus 1,800 noise-conditioned and 200 field recordings; intact/debonded | Exact tile tap task, but data available from author on request | Do not claim public download; request separately |
| Fill/content | [CORSMAL CCM](https://corsmal.github.io/containers_manip.html) | 1,140 recordings; 15 containers; empty/50%/90%; water/rice/pasta; annotations | Audio from pouring/shaking, not static taps | Strong fill proxy; full audio ~2.8 GB for training set; license is stated on project page |
| Fill/content | [ACM-S2](https://zenodo.org/records/4770439) | 21 WAVs; 19 pouring/2 shaking; four containers; 0/50/90%; CC BY 4.0 | Different microphone/room and interaction | **First fill benchmark**; 11.23 MB ZIP |
| Fill/content | [ACC models](https://zenodo.org/records/4770061) | Released action/pouring/shaking TensorFlow models; CC BY 4.0 | Trained for CORSMAL interactions | **First pretrained comparison**; 60.66 MB model ZIP |
| Fruit ripeness | [Durian phone-knock](https://zenodo.org/records/18603796) | Source describes 189 fruits; archive has 562 audio members from 175 IDs; label/duplicate audit required | Phone tap protocol matches, but Android contains contradictory duplicate labels | Tested strict 133-fruit iPhone terminal-stage subset; audio ZIP 1.62 GB verified; CC BY 4.0 |
| Fruit maturity | [Coconut acoustic](https://data.mendeley.com/datasets/hxh8kd3snj/1) | Workbook verified as raw 44.1 kHz waveforms: 127 fruits / 381 ridges, not advertised 129/387; five fruits have shorter contiguous recordings | Mechanized tapping, not phone capture | Completed fruit-held-out classification on all actual 127 fruits; CC BY 4.0 |
| Material | [YCB impact](https://osf.io/4tcp6) | Real airborne impact/friction recordings; 11/14 material labels, object IDs and actions | Robot vertical pokes, not manual phone tap | Tested 27-object/9-class/131-recording subset with object-disjoint splits; null OSF data license requires review |
| Material | [IDMT-ISA-Pucks](https://zenodo.org/records/7551338) | ZIP contains 225 WAVs, not documented 260; 135 known-material recordings scored in 45 complete-game folds | Airborne industrial audio, but material/manufacturing process and game/player confounded | Full 3,776,343,888-byte ZIP MD5 verified. License conflict: Zenodo CC BY-NC-ND 4.0 vs bundled PDF CC BY-SA 4.0; permission review required |
| Machine anomaly | [DCASE 2024 development](https://zenodo.org/records/10902294) | 990 source-normal + 10 target-normal training; 100 normal + 100 anomaly test per section | Operating sound, not THUD tap | Valve evaluated with frozen general encoders and released task-trained Jvachier/STgram models; not all machines |
| Machine anomaly | [DCASE 2024 additional training](https://zenodo.org/records/11259435) / [evaluation](https://zenodo.org/records/11363076) | Normal-only training; published test labels/domain maps joined by full filename | Operating sound | RoboticArm paired 1,000 train / 200 test evaluated; official evaluator executed unchanged; not all additional machines |
| Machine anomaly | [MIMII DG](https://zenodo.org/records/6529888) / [MIMII DUE](https://zenodo.org/records/4740355) | Multiple industrial machine/domain-shift sections | Operating sound, not tap | Stronger machine benchmark; license metadata needs review; avoid full download initially |
| Seal/leak proxy | [Water-distribution leak audio](https://zenodo.org/records/18631450) | 500 leak + 386 no-leak + 114 noise 1-second clips; pipe/pressure/flow/device labels; CC BY 4.0 | Pipe hydrophone/noise-logger recordings; not verified airborne phone or closure tap/vacuum | All 1,000 supplied clips downloaded and scored; three RAR files total 5,651,250 bytes |
| Seal/leak proxy | [IDMT compressed-air](https://zenodo.org/records/7551606) | Source describes 5,592 files; measurement microphones and multiple sessions; CC BY-NC-ND 4.0 | Audible leak, not phone tap | Tested 48 complete stratified recordings across all three archives using verified range retrieval; 194.9 MB transferred, no 22 GB full download |
| Moisture | No verified public phone-tap dataset found in checked sources | Nearby wood sensors/contact AE lack airborne tap + reference moisture labels | Protocol mismatch | Collect paired taps and moisture-meter/gravimetric labels; do not substitute |

## Benchmark matrix

| Dataset family | Task | First model | Strong comparison | Split | Primary metrics | Edge gate |
|---|---|---|---|---|---|---|
| Concrete impact-echo | defect class | log-mel + linear SVM | small depthwise CNN | slab/specimen/group held out | macro-F1, balanced accuracy, AUROC | fixed-shape CPU latency and model size |
| CORSMAL/ACM-S2 | content and level class | MFCC/log-mel + linear SVM | released ACC model | container-held out; ACM-S2 external setup | macro-F1, level accuracy, confusion matrix | inference time and peak process memory |
| Durian | ripeness class | MFCC/log-mel + logistic regression | small CNN; frozen YAMNet head | fruit-held out; device transfer | macro-F1, balanced accuracy, per-device score | 48-kHz capture separate from model cost; fixed-window CPU proxy |
| YCB impact | material class | log-mel + linear SVM | small CNN | object held out, action-stratified | macro-F1, balanced accuracy | 1-second fixed input, CPU timing |
| Leak audio | leak/no-leak | log-mel + logistic regression | small CNN/autoencoder | recording/session/device grouped | AUROC, AUPRC, balanced accuracy | 1-second fixed input |
| DCASE/MIMII | anomaly score | normal-only log-mel Mahalanobis | task-specific ASD model/STgram-MFN where compatible | official machine/section protocol | AUC, pAUC, source/target AUC | 10-second input, model size/CPU timing |

## Leakage rules

- Never split repeated taps, fruit orientations, recordings, or windows from one physical object across train and test.
- Fit scalers, PCA, covariance, thresholds and hyperparameters on training groups only.
- Released ACC models use ACM-S2 only as an external test. Separately labelled ACM-S2 container-held-out internal cross-validation is a different experiment, not the same external test.
- Do not use hidden DCASE evaluation labels.
- Report raw audio, processed arrays, checkpoint and dataset licenses separately.

## Storage policy

The workspace previously contained a 20.5 GiB accidental partial ObjectFolder download; it was deleted. Do not download ObjectFolder's 76.35 GiB archive or any multi-gigabyte dataset until a bounded subset is identified. Download manifests and hashes before data where possible. Keep all large downloads under `artifacts/datasets/`; never write temporary tails at the repository root.

## Known unrun items

- Exact tile recordings from the Buildings paper: author request required.
- Full YCB/RealImpact collections: selected object-held-out pilots completed, not entire corpora. OSF/RealImpact data licenses remain unstated; no production/redistribution rights inferred. RealImpact uses fully decoded rows from bounded compressed prefixes, not full-archive integrity verification.
- Full CORSMAL: released ACC evaluated on all ACM-S2 and seven official demos. HTTP/API routes returned an obfuscated JavaScript challenge; direct Chromium HTTPS returned ERR_EMPTY_RESPONSE, then HTTP-to-HTTPS Chromium navigation returned a visible Error 403 / Web Page Blocked. `artifacts/ccm-browser-access.json` records the blocker. Full CCM validation (137 published filenames) has not run; the runner requires the complete supplied validation list and does not silently score a partial subset.
- DCASE2024 valve: task-trained Jvachier and STgram-MFN encoders actually executed. STgram's current main branch is incompatible with released checkpoint; pinned historical source loaded strictly.
- Moisture and seal-integrity-from-phone-tap: no matching public labelled dataset found in checked sources.
- Physical iPhone 11/Core ML execution: unavailable; CPU fixed-shape benchmark is only an edge proxy until hardware testing.

## Executed benchmarks (updated during execution)

### ACM-S2 / CORSMAL ACC

- `experiments/acc_benchmark.py`: released CCM-trained ACC ensemble evaluated on all 21 ACM-S2 recordings with no fitting/tuning. Filling-level accuracy 13/21 (61.90%), macro-F1 0.4327; filling-type accuracy 13/21, macro-F1 0.4721; joint accuracy 9/21 (42.86%), macro-F1 0.3388. Both empty samples were missed. The seven official in-domain demos scored 7/7, separately reported as smoke checks only.
- `experiments/dataset_benchmark.py`: a separate exploratory internal ACM-S2 container-held-out experiment with 90 spectral/cepstral summary features. Logistic accuracy 17/21 (80.95%), macro-F1 0.8024; RBF accuracy 11/21 (52.38%), macro-F1 0.5460. Training and evaluation are on the small ACM-S2 population, so these are not directly comparable to frozen ACC's external transfer result.
- ACC architecture/weights loaded strictly and all tensors matched released HDF5 values. Original cascade and compatibility frontend were checked against source. Full historical TensorFlow environment equivalence was not measured.
- Reports: `artifacts/acc-benchmark.json`, `artifacts/acc-inspection.json`, `artifacts/acc-predictions.npz`, `artifacts/dataset-benchmarks.json`.

### Pipe leakage

- All 1,000 supplied clips (500 leak, 386 no-leak, 114 environmental noise) are scored with filename-condition-held-out classifiers. Identical WAV hashes are excluded from training whenever present in a test group. This excludes direct duplicate leakage but cannot recover absent recording/session identifiers.
- Device-transfer evaluation separately holds hydrophone/noise-logger type out. This is a necessary stress check: filename-condition cross-validation must not be presented as proof of new-device or new-site generalization.
- Both a fixed logistic classifier and fixed RBF SVM are evaluated; no hyperparameter selection based on the reported test scores. Full per-example predictions, hashes and group assignments are retained in `artifacts/dataset-benchmarks.json`.

### Hardware availability

- Host CPU: Intel Core i5-11400H, 6 cores / 12 logical processors, Windows. `adb devices -l` returned no attached Android devices. No iPhone device runner, Core ML execution or physical-device energy measurement is available in the current environment.
- Latest refreshed ACC CPU run: 20 warmups + 100 routed single-recording inferences; preprocessed p50/p95 36.51/40.60 ms. Frontend measured separately on five clips: p50/p95 112.12/164.96 ms. Ensemble has 16,482,637 parameters. Peak process working set 942,227,456 bytes includes Python/framework/dependencies, not model-only RAM.
- Timing while other workers are active is a desktop workload observation, not an isolated A13 benchmark. Prediction latency excludes capture/window wait unless explicitly stated.

### Durian phone taps: audited fruit-held-out classification

- `experiments/durian_benchmark.py` acquired only the sound ZIP (1,624,075,464 bytes, source MD5 verified). It found 105 exact-duplicate Android groups/211 members bearing contradictory stage labels. These members were excluded from supervised experiments.
- Terminal actual status cannot label earlier recordings. Primary experiment uses strict stage-matched iPhone recordings from 133 fruits (57 Unripe / 41 Ripe / 35 Overripe); Disease/missing IDs 19, 35, 163, 164 are excluded. This terminal-stage association remains a source assumption, not independent biological ground truth.
- Five outer fruit-held-out folds, training-only inner tuning, identical model splits. Balanced accuracy / macro-F1: majority 0.3333/0.2000; MFCC/spectral 0.8154/0.8147; frozen YAMNet 0.8068/0.8076; hybrid 0.8096/0.8093. Larger neural representations did not improve the primary task.
- Three-class cross-device evaluation is not estimable after the integrity exclusions: the Android primary subset contains 57 Unripe and no Ripe/Overripe examples. Do not present paired devices as demonstrated device transfer.
- Reports: `artifacts/durian-results.json`, `artifacts/durian-report.md`; full label/duplicate mappings retained. Single-farm, date, acquisition-stage and unannotated impact-selection confounds remain.

### Structural impact-echo

- `experiments/impact_echo_benchmark.py` processed all 1,824 CCNY signals descriptively. Classification is blocked by missing unambiguous row-index-to-defect geometry registration; no labels inferred from acoustic appearance or clustering.
- Official SDNET2021 downloads returned HTTP 403. A pinned author-maintained original-rate mirror supplied 1,936 signals and aligned labels at 102.4 kHz. Whole-deck-held-out nested SVM: balanced accuracy 0.5205, macro-F1 0.4619. Fixed 2,819-parameter MLP: balanced accuracy 0.4688, macro-F1 0.4693; iteration-cap warnings retained. The majority class is large: plain accuracy alone is misleading.
- This is contact impact-echo sensor evidence, not airborne phone-microphone tile detection. Mirror licensing remains unresolved and one class count differs from publication. Reports: `artifacts/impact-echo-results.json`, `artifacts/impact-echo-report.md`, predictions and split CSVs.

### Compressed-air audible leakage

- A 48-recording stratified pilot covers all three source archives, sessions 1/2, lab/work backgrounds, microphone configurations 1m/2m and balanced leak/no-leak conditions. Session-held-out RBF: balanced accuracy 0.8125, macro-F1 0.8124, AUC 0.9063. Microphone-held-out RBF: balanced accuracy 0.8958, macro-F1 0.8958, AUC 0.9497.
- HTTP 206 and exact Content-Range were validated before each body read. Selected ZIP member CRCs and SHA256 were verified; full-archive MD5 values are expected-source metadata only, not independently verified. Transfer 194,895,137 bytes; no full archives downloaded.
- Reports: `artifacts/compressed-air-benchmark.json`, `artifacts/compressed-air-manifest.json`, `artifacts/compressed-air-access.json`. Session 3, additional backgrounds/mics/settings and full-release coverage remain untested.

### DCASE2024 valve: released task-trained encoders

- `experiments/machine_task_benchmark.py`: official 1,000 normal train / 200 test, no train/test WAV digest overlap; float32 encoder features with float64 refitted covariance. Published 2020 model accuracy is not transferred as a claim about 2024.
- Jvachier seed0 original calibration: pooled AUC 0.4809, standardized pAUC 0.4889. Refit with only 2024 training normals: AUC 0.4704, pAUC 0.5121.
- STgram-MFN frozen 128D features plus 2024 normal-only Mahalanobis: AUC 0.7012, pAUC 0.5889, official-mask source/target AUC 0.6692/0.7332. Exact upstream feature parity and strict historical-checkpoint loading verified. Stronger than the prior frozen EfficientAT run on this inspected test set, but not independently confirmed and not universally best.
- Model parameters: Jvachier 413,440; STgram 1,350,335. Refreshed CPU full-clip processing p50/p95 29.06/37.60 ms and 28.97/46.48 ms, excluding capture wait. No 2020 threshold was transplanted to a new deployment.
- Reports: `artifacts/machine-task-summary.json`, `artifacts/machine-task-benchmark.json`, `artifacts/machine-task-stgram.json`.


### YCB impact material classification

- `experiments/material_benchmark.py`: 27 distinct objects, nine material classes, 131 recordings after four ambiguous exact-duplicate files were excluded. Three object-held-out outer folds; nested training-only linear-SVM tuning. Both representations use the same peak-centered one-second crops and folds.
- Recording balanced accuracy / macro-F1: MFCC/log-mel 0.3037/0.2882, frozen YAMNet 0.3481/0.3342. Both have 10/27 object accuracy; object macro-F1 0.3393/0.3407. Three objects per class is a small controlled pilot, not full-corpus coverage or universal material inference.
- CPU one-second preprocessed input feature+head p50/p95: 1.08/1.47 ms for features; 9.55/10.56 ms for YAMNet. Excludes decoding, resampling and event crop selection. Actual selected audio 46.29 MB, total payload transfer 50.41 MB. Data permissions remain unresolved; no raw redistribution.
- Reports: `artifacts/material-benchmark.json`, `artifacts/material-summary.json`, `artifacts/material-access.json`; recording/object predictions, fold memberships and source hashes retained.

### Full supplied water-leak release

- All 1,000 clips: fixed logistic AUC 0.9606, balanced accuracy 0.9150, macro-F1 0.9150; fixed RBF AUC 0.9769, balanced accuracy 0.9430, macro-F1 0.9430. These pooled out-of-fold scores combine separately fitted models, not one calibrated deployment detector.
- Device-type-held-out logistic on 998 known-device clips drops to AUC 0.7722, balanced accuracy 0.6419, macro-F1 0.6067. High filename-group scores therefore do not establish new-sensor generalization.
- Source device names include hydrophones and noise loggers; do not describe this as verified phone-airborne leak audio. Exact WAV hash duplicates are excluded from reference training when present in test. Independent session IDs remain unavailable.
- Finalization correction: merge all filename groups linked by identical complete-WAV hashes before LOGO (six cross-group duplicate sets, 12 clips; 319 groups become 314 connected components). New logistic accuracy 91.4%, RBF 94.3%; detailed metrics in `artifacts/final/classifiers/pipe_leak-*-oof.json`. Missing acquisition metadata still limits independence. Historical JSON above remains unchanged.


### Coconut raw-waveform maturity benchmark

- `experiments/coconut_benchmark.py` verified the 278,424,725-byte workbook SHA256 and parsed waveform samples, not assumed extracted features. Actual population 127 fruits/381 ridges: 8 premature, 36 mature, 83 overmature; fruit IDs 67 and 86 are absent. Five fruits have 22,500 contiguous samples per ridge; others 132,300 at source-documented 44.1 kHz. Trailing blanks are trimmed, not imputed; no internal missing cells or invalid-ridge exclusions.
- Four stratified fruit-held-out outer folds, three-fold training-only selection, average ridge decision scores per fruit. Balanced logistic: balanced accuracy 0.6987, macro-F1 0.6557; RBF: 0.6147/0.5941; fixed unweighted logistic: 0.6984/0.7370. Train-majority: balanced accuracy 0.3333, accuracy 0.6535. No universal best model is claimed: metric tradeoffs matter with only eight premature fruits.
- CPU full frontend+head p50/p95: one ridge 9.05/10.59 ms; whole three-ridge fruit 27.38/29.76 ms. Serialized all-data logistic pipeline 5,492 bytes, separate from out-of-fold evaluation and not including waveform acquisition or workbook parsing. Report: `artifacts/coconut-benchmark.json`.

### Portable inference check (not physical-device execution)

- Durian MFCC/spectral fold heads exported to builtin-only float32 TFLite: five 3,180-byte files, fixed input [1,112] and output [1,3], scaler + logistic + softmax. All 133 held-out probabilities compared against sklearn: maximum discrepancy 3.56e-7, zero class changes, balanced accuracy unchanged. Reports: `artifacts/durian-tflite-heads.json`, `artifacts/durian-mfcc-fold-*.tflite`.
- Desktop TFLite CPU head-only p50 0.0038–0.00485 ms, 20 warmups + 100 calls/fold. Audio decoding, impact selection and MFCC frontend are not converted; these are evaluation-fold heads, not production-trained mobile applications.

### RealImpact bounded material subset

- `python -m experiments.material_benchmark --realimpact`: 270 complete force-deconvolved recordings from nine objects (three Ceramic / three Plastic / three Wood), first 30 rows/object selected deterministically before modelling. Material names were matched to primary ObjectFolder numeric-ID material tables, not guessed solely from filenames.
- Three object-held-out folds with training-only nested SVM tuning. MFCC/log-mel recording accuracy 0.7778 / macro-F1 0.7593; YAMNet 0.8370 / 0.8355. Object results 7/9 versus 8/9 correct, macro-F1 0.7500 / 0.8857. The independent unit is only nine objects, not 270 microphone rows; nearby rows often share one impact vertex, and plastic bowls share a product family.
- Consumed 211,681,280 compressed audio bytes via validated HTTP ranges and early DEFLATE termination. Complete finite float32 rows and prefix hashes retained; small annotation members have CRC verification, but the truncated full audio member CRC/archive SHA is not verified.
- Preprocessed one-second CPU feature+head p50/p95 0.95/1.17 ms for MFCC/log-mel, 8.72/9.48 ms for YAMNet. Force deconvolution requires per-example peak normalization; this is not ordinary phone audio. Reports: `artifacts/material-realimpact-benchmark.json`, `artifacts/material-realimpact-protocol.json` and source manifest.


## Exact untested-data gates

- **Tile acoustic recordings:** Buildings 16(4):870 and Applied Sciences 15:1500 studies require author-provided data in the checked sources; the ECTI 79-recording study does not provide a verified public archive. These are not replaced by contact concrete IE accuracy.
- **CCNY and IE2020:** CCNY arrays are available and descriptively processed, but no unambiguous physical row-to-defect registration supports classification. IE2020 `pnyn2ccfhg/1` and bare-deck `44rb96872r/1` public file-list APIs returned empty lists; no file UUID/download was fabricated.
- **CORSMAL CCM:** source annotations label 684 training-pool recordings from nine containers. The 137 published validation filenames differ from an evaluator row-index list; the runner documents its exact chosen list. Public/private filling labels are -1 in inspected annotation assets. Browser/API access to the audio failed; seven demos are not full validation.
- **Seal/vacuum and moisture:** no matching phone-tap integrity or tap-plus-reference-moisture corpus was verified. Pipe/compressed-air leak classification and contact AE are different tasks. No requested physical labels were invented.
- **Hardware:** no phone is attached. Physical iPhone/Android execution, energy and thermal behavior remain not measured. TFLite head parity and desktop CPU inference are reachable substitutes, not falsely labelled on-device tests.
- Full-corpus claims are withheld wherever only a selected subset was evaluated. Size caps are storage protections, not evidence that an accessible corpus is unavailable.

### IDMT Pucks and phone motor

- `experiments/puck_motor_benchmark.py` covers all 135 known-material Pucks files with 45 complete-game-held-out folds. All three minute patches, stereo channels and nine selected windows per game remain together; unknown factory-puck material and background-only files are excluded. MFCC/log-mel logistic file accuracy 0.9111, complete-game accuracy 1.0000; RBF file accuracy 0.6444. Frozen YAMNet logistic file accuracy 0.4296. Complete-game pooling and small controlled categories must not be confused with general material recognition.
- The majority comparator scores zero because leaving out a single-class game changes the training-majority class. Zero is a consequence of this split/argmax rule, not chance accuracy; do not inflate claimed improvement by comparing against it alone.
- Phone motor: all 20 decoded WAVs, five speed-held-out folds, both directions/conditions. Feature logistic/RBF 20/20 correct; YAMNet logistic 17/20 and RBF 14/20. This is one motor before/after repair; speed transfer is not unseen-motor, independent-session or causal defect validation. M4A originals were excluded as duplicate source observations.
- Motor payload 78,025,990 bytes; source SHA256 verified. Pucks full ZIP 3,776,343,888 bytes with source MD5 verified; exact archive count 225 and license disagreement preserved.
- Four-second CPU frontend p50 2.55–2.69 ms; YAMNet frontend+encoder 13.83–15.52 ms, excluding decoding/resampling/capture. Logistic head storage 4,321 bytes motor / 5,793 bytes Pucks; this is not runtime RAM. Reports: `artifacts/puck-motor-pucks-report.json`, `artifacts/puck-motor-motor-report.json` and corresponding summaries.

### Cross-corpus valve and RoboticArm anomaly checks

- `experiments/multimachine_benchmark.py`: frozen EfficientAT `mn04_as_mAP_432` + cosine 1-NN, versus same-frontend log-mel Gaussian (Ledoit-Wolf). Both fit only on each section's normal training clips; no test label, threshold or statistic is used. Machine-specific valve (or RoboticArm) subsets only, not all-machine ASD.
- Hmean (cosine / Gaussian): DCASE2023 valve 55.82/55.13; DCASE2025 72.68/69.23; MIMII DG (3 sections) 57.33/54.41; MIMII DUE (3 sections) 55.07/53.71; DCASE2020 valve 67.62/60.08; DCASE2026 ValveEmu near channel 69.30/65.81. RoboticArm (official 2024 evaluator executed unchanged, exact match): 53.88/50.68.
- **DCASE2023 valve is byte-identical to MIMII DG section 00 (1,200 WAVs)**; it is not independent validation. Other pairs share Hitachi testbed sessions despite zero hash collisions.
- License: Zenodo metadata says CC BY 4.0 for MIMII DG and DCASE2023, while descriptions state CC BY-NC-SA 4.0; use the restrictive reading. 2026 far reference channel unused; DCASE2025 supplemental clean/noise files excluded. Archives verified by size and API checksum.
- Desktop CPU 28.2–33.3 ms per 10 s clip (2 threads); reference bank 1.54 MB float32. Not iPhone/Neural Engine.

### Task-trained DCASE2024 valve encoders

- Jvachier zero-shot AUC 0.4809, refit 0.4704; STgram-MFN refit 0.7012 (pAUC 0.58895). Test set was inspected before; refit uses 2024 normal-only clips; not unadapted performance. 2020-pretraining overlap not audited. CPU p50 29.06 ms (Jvachier), 28.97 ms (STgram).

### Secondary tabular datasets

- Jackfruit (253 spectral/MFCC rows, 12 exact-duplicate rows kept in one fold): balanced logistic 81.34% balanced accuracy, majority 50.00%. No specimen IDs exist, so this is not unseen-fruit evidence. Features only; no audio/phone test.
- GPLA-12 (684 rows): balanced logistic 80.56%. These are 12 sensor/pressure/noise regimes, not leak vs no-leak or seal integrity; no session IDs, so no unseen-rig claim.
- Seedless watermelon (doi:10.21950/OBYUCV): Dataverse API requires a guestbook response with personal identity; not submitted. No metrics. Total downloaded 3,822,006 bytes.



