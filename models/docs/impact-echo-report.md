# Impact-echo dataset experiments

## Scope and result

These are **contact/specialized impact-echo (IE) measurements of concrete, not phone microphone recordings**. Nothing here measures tile bonding, iPhone accuracy, Core ML conversion, device latency, RAM, energy, or battery use. No audio foundation model was applied by pretending that a 500 kHz IE sensor waveform is 16 kHz microphone PCM.

- **CCNY/Mendeley 997pwzvhgj version 2:** all six published files downloaded (33,880,205 bytes combined), with exact publisher SHA-256 verification. All 1,824 signals processed descriptively; defect classification deliberately not scored because exact index-to-schematic defect registration remains ambiguous.
- **SDNET2021:** official README and 1.81 GB IE archive requests both returned HTTP 403. Instead, all 1,936 original-rate, author-preprocessed mirror signals and aligned labels were downloaded from a pinned public author repository (154,899,584 bytes across 32 NPY files), without downloading GPR/IRT or the unavailable official archive. Five whole-deck-held-out experiments were run with spectral SVM and a 2,819-parameter spectral MLP.
- The SVM achieved **52.05% three-class balanced accuracy / 46.19% macro F1**. The MLP achieved **46.88% balanced accuracy / 46.93% macro F1**. The MLP hit its optimization iteration cap in every fold; all warnings are retained. These are finite baselines, not a search for a claimed state of the art.

## Sources, access, permissions and provenance

1. [Mendeley dataset version 2](https://data.mendeley.com/datasets/997pwzvhgj/2), DOI [10.17632/997pwzvhgj.2](https://doi.org/10.17632/997pwzvhgj.2), Hoxha, Feng, Salem and Xiao. Its page explicitly states that it has no labels and offers schematics for label derivation. Exact dataset license: **CC BY 4.0**. The [public file API](https://data.mendeley.com/public-api/datasets/997pwzvhgj/files?folder_id=root&version=2) returned six files, saved as `mendeley-manifest.json`. Every downloaded content hash matched its `content_details.sha256_hash`.
2. [Original RA-L paper](https://par.nsf.gov/servlets/purl/10667757), Hoxha, Feng, Sanakov and Xiao, *Robotic Inspection and Subsurface Defect Mapping Using Impact-Echo and Ground Penetrating Radar*, DOI [10.1109/LRA.2023.3290386](https://doi.org/10.1109/LRA.2023.3290386). Paper PDF and extracted text are preserved for inspection; IEEE copyright is distinct from the dataset's CC BY license. The paper describes a microphone in a dome **and a separate ground-contact PZT vibration sensor**. The released IE arrays must not be equated with the microphone channel. The paper reports 500 kHz/1,024 samples for DS2 and an IENet with six residual blocks, three BiLSTM layers and 534,498 parameters. Its training uses another dataset's Slabs 1–7, evaluating Slab 8 and qualitatively mapping DS2.
3. [SDNET2021 official record](https://commons.und.edu/data/19/), Ichi and Dorafshan, DOI [10.31356/data019](https://doi.org/10.31356/data019). The record describes five in-service bridge decks in Grand Forks, ND, with removal/chain-dragging-grounded delamination annotations. It offers a 286 kB README and 1,811,828 kB IE ZIP. `access-audit.json` records HTTP 403 for the exact official README and IE URLs. No access restriction was bypassed; no raw archive or original README was obtained. **Dataset reuse/redistribution license was not established from the accessible record; license review required.** A publisher paper's open-access license must not be substituted for a dataset license.
4. [Author-maintained preprocessing repository](https://github.com/ehoxha91/impact_echo_datasets), pinned commit **ff970edfa89637623e13cd89132c5266862c6344**. The README says its `IE_data_sdnet2021_original` data are trimmed native-rate signals with labels, 102.4 kHz and 10,000 samples each. It distinguishes these from a separate 500 kHz upsampled version. The README contains an apparent heading typo around that distinction; the benchmark uses only the `original` directory, not upsampled data. Each NPY download was bounded by the GitHub tree's exact byte size and verified against its Git blob SHA-1; local SHA-256s are also saved. This is an **author-preprocessed derivative**, not a verified byte-for-byte extraction of the unavailable official archive. No repository license file was present in its recursive tree; availability does not establish redistribution/commercial-training permission.
5. [Published-model implementation repository](https://github.com/ehoxha91/impact_echo_contrastive_learning), pinned commit **b17373a0c67ee0df8b9a1174f2e2c96140b9710c**. README, baseline code and evaluation source were inspected and preserved. It says its attached weights were trained for **one epoch only as usage samples**, not the original article's trained weights. We did not score these demonstration checkpoints as a published pretrained system, nor invent CCNY labels to train IENet. Its README claims MIT but the inspected tree has no LICENSE file; exact permission requires review.

Downloads used HTTP status validation, response-header Content-Length caps when available, enforced stream-byte caps, timeout bounds, and file removal on failed/oversized/inconsistent transfers. No Range-based assumption, dependency installation, app implementation, or cloud upload was used. `downloads.json` retains source/final URLs, byte sizes, SHA-256s and Git blob hashes where applicable. Page HTML, public API inventories, upstream code, original arrays and original schematic PDF are preserved under `artifacts/datasets/impact_echo/`.

## CCNY: descriptive processing, not fake defect prediction

### Signal contract

- `time_amplitude_all.npy` and `time_all.npy`: `(1824, 1024)`, float64.
- `freq_amplitude.npy` and `freq.npy`: `(1824, 124)`, float64.
- Loader documents **500,000 samples/s**. Time values run 0, 2, 4, …, 2,046; the microsecond interpretation is consistent with that sample rate. Full acquisition duration is **2.048 ms**, not 2.048 seconds. Signal amplitudes have no explicit engineering units in the array/loader and are not labelled pressure, PCM, or sound level.
- Upstream repository documentation identifies the provided frequency axis as **kHz**, approximately 0–60 kHz with rounded 0.49 kHz bin spacing. It is retained unchanged; it was not treated as Hz or resampled audio.
- Signals `[0:1178]` are Part 1, grid 31×38; `[1178:1824]` are Part 2, grid 19×34. The published Mendeley loader uses older filenames `time_amplitude.npy`/`time.npy`; the actual version-2 files have `_all` names. The runner uses the actual manifest filenames, without modifying upstream code.

### Exact classification blocker

The PDF depicts artificial voids, delamination, honeycombing, metal and cracks. Its annotated corner pairs span `(0,0)` to `(38,38)` for Part 1 and `(0,0)` to `(34,34)` for Part 2, while the documented arrays have **31×38 and 19×34** measurement grids. Approximate inch spacing in the loader is not an exact coordinate table. The author's frequency-map script reshapes to those grid shapes and flips axis 1, which establishes a display ordering but does not resolve calibrated physical origin, which grid axis spans which schematic dimension, endpoint inclusion, scan-to-schematic registration, or defect boundary membership. The dataset page explicitly says labels are absent; its role in the original paper is qualitative evaluation.

Consequently, no per-signal defect labels, defect probabilities, accuracy, SVM training, or published-model reproduction are claimed for these arrays. Merely rasterizing rectangles from the PDF into a guessed 31×38 grid would manufacture ground truth and invite spatial leakage. A corrected row/column-to-physical-coordinate mapping plus authoritative defect polygons or per-signal labels is the missing prerequisite. A second physically independent labelled slab is also needed for true slab-generalization evaluation; splitting the two parts of one slab is not independent-slab validation.

### Exercised descriptive scenario

All 1,824 complete native-rate traces were transformed into 84 spectral descriptors. Standardization, two-dimensional PCA and four-cluster KMeans were **fit only on Part 1** and applied without refitting to Part 2. Cluster IDs are descriptive partitions, **not defect classes**. They do not establish detection performance or anomaly probabilities.

- Batch feature time: **33.73 ms for 1,824 traces** on desktop CPU (not an individual-inference latency measurement).
- PCA variance captured by first two components: **19.54% + 8.41%**.
- Part 1 cluster counts: **362 / 207 / 527 / 82**; Part 2: **124 / 67 / 432 / 23**.
- Supplied-spectrum dominant frequency above the author's excluded first eight bins: min/Q1/median/Q3/max = **3.91 / 4.88 / 8.79 / 9.77 / 12.70 kHz**. These are spectrum summaries, not defect depth estimates.
- Per-signal outputs, explicit fit/project roles and unavailable defect labels: `impact-echo-ccny-descriptive.csv`. Features/projections: `impact-echo-ccny-features.npz`. Fitted descriptive pipeline: `impact-echo-ccny-descriptive.joblib`.

## SDNET2021: separately scored contact-IE derivative

### Labels, units and grouping

The mirror supplies 16 paired signal/label sections of **121 measurements each**, totalling **1,936 signals** across five decks. Each waveform has **10,000 original-rate samples at 102,400 Hz** (97.65625 ms retained by the mirror). We preserved all samples and source precision on disk. Amplitude engineering units, calibration and official raw-file extraction details are not established by the mirror, so native floating-point amplitudes are not described as Pa, microphone PCM, or acceleration units.

The official record defines class 1 sound concrete, class 2 delamination above the top rebar mat, and class 3 delamination below it. Downloaded mirror counts are **1,378 / 494 / 64**. This is **558 delaminated**, not the **557 delaminated / 1,379 sound** count reported in indexed text of the original dataset article; the one-label discrepancy is unresolved, so results are explicitly for this pinned derivative, not asserted identical to original SDNET2021 labels. Official raw README/ZIP access was unavailable for reconciliation. All aligned label values were used as released, never inferred from waveform shape.

Outer evaluation holds out **every section of one whole deck at once**, avoiding adjacent point/section mixing across train and test. Decks are `forest_river_nb`, `forest_river_sb`, `park_river_median`, `park_river_nb`, `park_river_sb`. This evaluates held-deck transfer, not unseen region, sensor, bridge family, or airborne-phone generalization; related decks at the same river remain in training. All samples belong to exactly one test fold. No exact waveform duplicates across decks were found in the native-rate mirror. Signal maxima repeatedly equal 10.513225 in all sections; this may indicate acquisition saturation [INFERENCE], retained rather than silently repaired. Counts are preserved in `sdnet-quality.json`.

### Fixed representations and finite methods

The feature frontend uses the first **20 ms / 2,048 samples** at native rate, zero-centering and a Hann FFT. Its 84 dimensions are 64 log relative-energy bands over 0.5–25 kHz, 16 DCT coefficients, spectral centroid, width, peak and flatness. No resizing to an audio encoder, loudness-derived labels, random windows or augmentation was used. For CCNY, the same frontend uses the entire available 1,024-sample trace rather than padding to 20 ms. Relative bands remove global gain, not impactor/contact/geometry or instrument-response confounds.

1. **Spectral SVM:** training-only StandardScaler plus class-balanced linear/RBF SVC. `C ∈ {0.1, 1, 10}` and kernel are selected by **inner leave-training-deck-out** balanced accuracy. The outer test deck never participates in tuning/scaling. This compares linear and nonlinear heads without a broad hyperparameter hunt.
2. **Small task-trained spectral MLP:** training-only scaler, 84→32 tanh→3 network (**2,819 parameters**), fixed L2 alpha 1, L-BFGS, 300-iteration cap, seed 42. No test-driven architecture/epoch selection. All five fits reached the iteration cap; this limits the conclusion and is not hidden by suppressing warnings.

### Results: pooled out-of-deck predictions

| Model | Accuracy | Balanced accuracy | Macro F1 | Class 1 recall | Class 2 recall | Class 3 recall |
|---|---:|---:|---:|---:|---:|---:|
| Spectral SVM | 65.60% | **52.05%** | 46.19% | 75.40% | 41.70% | 39.06% |
| Small spectral MLP | **71.38%** | 46.88% | **46.93%** | 83.82% | 44.33% | 12.50% |

A constant class-1 predictor has 71.18% accuracy, so raw accuracy is a poor success criterion. Higher MLP accuracy does **not** establish a better defect detector. Its very low deep-delamination recall is important; only 64 examples carry class 3. There are only five independent outer deck groups, not 1,936 independent structures. No narrow confidence interval based on independent individual traces is reported.

SVM confusion matrix (rows true, columns predicted; order 1,2,3):

```text
1039 242  97
 155 206 133
   8  31  25
```

MLP confusion matrix:

```text
1155 205 18
 228 219 47
  27  29  8
```

Held-deck balanced accuracy (SVM / MLP): Forest River NB **47.43% / 40.82%**; Forest River SB **54.39% / 49.62%**; Park River median **53.85% / 34.89%**; Park River NB **49.67% / 40.72%**; Park River SB **50.36% / 58.18%**. Pooled balanced accuracy above is computed across all out-of-fold predictions, not the unweighted mean of deck values.

### CPU proxy, not iPhone benchmark

The run used Python 3.12.9 on Windows 11, Intel64 Family 6 Model 141, one BLAS thread. Extracting features for all 1,936 traces took **62.07 ms** as a batch. Each fitted head was warmed up 20 times and timed for 100 single-feature-vector predictions:

- SVM head-only fold p50: **0.292–0.449 ms**; p95: **0.314–0.775 ms**. Joblib files **727,061–878,565 bytes**, including scaler and runtime serialization metadata; support vectors make the kernel SVM larger than a linear head.
- MLP head-only fold p50: **0.228–0.250 ms**; p95: **0.246–0.448 ms**. Each joblib file **43,523 bytes**; parameter count is 2,819. This is not a quantized/mobile-export size.

Head timing excludes feature extraction, capture/window wait, loading, conversion and device runtime. Batch feature times are not directly equivalent to single-sample overhead. Stored file bytes are not resident memory. The small MLP has a credible compact-compute path but its current generalization/optimization limits prevent a validated edge recommendation. None of these desktop timings predict A13/iPhone 11 latency.

## Additional IE2020 access audit

- [Bare-deck annotated IE dataset 44rb96872r/1](https://data.mendeley.com/datasets/44rb96872r/1): page accessible, **2,016 signals from eight laboratory slabs**, sound/defective labels, CC BY 4.0. The root public file endpoint returned HTTP 200 with an **empty array**; the attempted folder endpoints returned 404. No useful file inventory/label-preserving slab download was obtained, and no claim of benchmarking this dataset is made.
- [Annotated IE2020 pnyn2ccfhg/1](https://data.mendeley.com/datasets/pnyn2ccfhg/1): page accessible, describes text files with time/amplitude columns split into sound/defective folders, CC BY 4.0. The root public file endpoint likewise returned HTTP 200 with an empty array. No bulk archive or unsupported hidden download was attempted. Saved manifests and page HTML document the accessible evidence.

These datasets are not silently substituted for CCNY or SDNET2021, nor counted as tested because their metadata were read.

## Reproduction and output inventory

From the repository root, with the existing environment and preserved inputs:

```text
python experiments/impact_echo_benchmark.py
```

The runner performs input/schema/finite assertions and deterministic fold/group checks while exercising the dataset experiments; it downloads nothing and installs nothing. It is separate from other agents' experiments. Inputs come from `artifacts/datasets/impact_echo/`, using pinned manifests and the provenance listed above. No build, lint, unit-test suite, formatter or app smoke test was run for this assignment.

- `impact-echo-results.json`: full measured metrics, class counts, all folds, selected SVM parameters, runtime/size evidence and MLP warnings.
- `impact-echo-sdnet-predictions.csv`: **3,872 predictions**, both models, true/predicted labels, held-out deck, source section and per-section row index.
- `impact-echo-sdnet-splits.csv`: explicit **9,680** outer-fold memberships (1,936 signals × five folds).
- `impact-echo-sdnet-features.npz`: 84D features, original label values and deck IDs.
- `impact-echo-sdnet-*.joblib`: separate fitted models/scalers per held-out deck; not a single all-data production model.
- `impact-echo-ccny-descriptive.csv`, `.npz` and `.joblib`: all descriptive outputs with no fabricated class label.
- `impact-echo-input-hashes.json`: SHA-256/size inventory of preserved inputs. `impact-echo-artifact-hashes.json`: artifact integrity inventory, including script/report and generated predictions/splits/models.

## Remaining risk

The CCNY label-registration blocker is unresolved rather than guessed. SDNET results depend on an author-preprocessed mirror with an unresolved one-label difference and unverified dataset redistribution license. Contact IE performance does not establish airborne THUD observability or phone accuracy. No physical iPhone, mobile export, real tile-bonding dataset or microphone adaptation was tested.
