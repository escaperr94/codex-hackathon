# THUD baseline research code

Local research baseline for isolated microphone tap analysis, tile bonding screening, and container fill estimation.

**This is research software, not a certified physical inspector, structural safety tool, or completed mobile application.**

Dataset sources, access/license checks, executed grouped benchmarks, numerical exports and untested coverage are recorded in [DATASETS.md](DATASETS.md). Those public-data tasks include pouring, fruit taps, operating machines and contact impact-echo; their scores do not validate tile bonding or static tap-fill on phones.

## Consolidated workspace

All model code, tests, datasets and generated artifacts live under `models/`. Run installation, CLI, experiment and test commands from that directory (`cd models` from the repository root). Documentation lives in `models/docs/`; final research exports and their verification reports live in `models/artifacts/final/`. Historical reports are preserved, so paths inside them are relative to `models/`, not the repository root. Historical script hashes describe the source at the time of execution; moving a script does not make its old report a fresh run.

Benchmark scripts formerly under `artifacts/dcase/` and `artifacts/sound-of-water/` now live in `experiments/`: `run_valve_benchmark.py`, `compare_valve_encoders.py`, `sound_water_benchmark.py`, and `sound_water_efficientat.py`. Invoke them with `python -m experiments.<module>` from `models/`. Raw data and upstream dependency code remain in `artifacts/`; generated Markdown reports now belong in `docs/`.

Start with [FINALIZATION.md](FINALIZATION.md) for current model choices, release paths, finite ablations and verification commands. [FRUIT_FINALIZATION.md](FRUIT_FINALIZATION.md), [CLASSIFIER_FINALIZATION.md](CLASSIFIER_FINALIZATION.md) and [RESEARCH_FINALIZATION.md](RESEARCH_FINALIZATION.md) retain task-specific contracts and limits. Historical synthetic tile/fill exports below are software diagnostics, not finalized physical-state models.

For end-to-end Flutter app setup, automated checks, browser smoke testing, calibration workflows, tile-map testing and native-device limits, see [APP_TESTING.md](APP_TESTING.md).




## Scope and design

- **Inputs:** Isolated mono or stereo PCM/float WAV files (16–192 kHz, 0.4–10 s duration) containing a single manual or tool impact with at least 50 ms quiet lead-in and 250 ms ring-down.
- **Features:** 37 deterministic physical and spectral descriptors (`FEATURE_VERSION = 1`): dominant frequency peak, spectral centroid, spectral bandwidth, spectral flatness, four octave/sub-band energy ratios (100–500, 500–1500, 1500–4000, 4000–7500 Hz), 50% and 90% cumulative energy decay times within the fixed 250 ms analysis window, tail energy ratio, and 13 MFCC means and standard deviations.
- **Models:** Classical, low-dimensional linear models:
  - **Tile bonding:** L2-regularized logistic regression (`bonded` vs `debonded`). Outputs a screening class, not a structural void percentage or defect certification.
  - **Vessel fill:** Ridge regression on tracked spectral/MFCC features trained on volume-labelled percentage points ([0, 100]) for a single container family. Outputs unclamped percentage estimates.
- **Artifact format:** Self-contained, non-executable JSON schemas containing exact feature names, training standardization means and scales, linear coefficients, scalar intercept, and class or container-family tags.
- **Data integrity & leakage protection:** Manifest loading strictly enforces relative path resolution, duplicate path rejection, duplicate content hashing (SHA-256) rejection, consistent `recording_id` $\leftrightarrow$ `(object_id, session_id)` mapping, and strict training-only scaling during group-held-out cross-validation.

## Prerequisites and installation

Requires Python 3.10+ with standard scientific libraries (`numpy`, `scipy`, `scikit-learn`):

```bash
pip install -e .
# For running development tests:
pip install pytest
```

## CLI usage

All commands output strict JSON to standard output. Errors are written to standard error with a non-zero exit code.

### 1. Analyze an isolated tap WAV

Extracts capture metadata, quality checks, and all 37 features:

```bash
python -m thud analyze path/to/tap.wav
```

## Deep-learning baseline

The repository now includes a **task-trained** PyTorch model. This is not YAMNet, EfficientAT, CLAP, or a downloaded foundation model: its weights are learned from the THUD manifest labels.

The model uses:

- 16 kHz mono audio, resampled from each WAV without changing the source file;
- a fixed 1.0-second input window;
- a fixed 64-bin log-mel frontend (`n_fft=512`, 400-sample Hann window, 160-sample hop);
- a small depthwise-separable CNN with fewer than 200,000 parameters;
- a two-class tile head or scalar fill-regression head.

Run the task-trained CNN:

```bash
# Train a tile CNN; --epochs=1 is only a software smoke run.
python -m thud deep-train tiles_manifest.csv --task tile --output tile_cnn.pt --epochs 40

# Evaluate whole-object or session holdouts. Each fold trains from scratch.
python -m thud deep-evaluate tiles_manifest.csv --task tile --group-by object_id --epochs 40

# Predict one isolated tap.
python -m thud deep-predict tile_cnn.pt path/to/tap.wav

# Export fixed-shape TorchScript for a later runtime integration.
python -m thud deep-export tile_cnn.pt --output tile_cnn.ts
```

For fill:

```bash
python -m thud deep-train fill_manifest.csv --task fill --output fill_cnn.pt --epochs 40
python -m thud deep-predict fill_cnn.pt path/to/tap.wav --family steel-bottle-500ml
```

The fill family must match exactly. The CNN does not impose monotonicity or clamp predictions to `[0, 100]`.

### Pretrained encoder contracts

`thud/encoders.py` provides **contract-only adapters** for already-loaded encoders. It does not download weights or perform conversion:

- YAMNet: 16 kHz mono; 0.96-second patches with 0.48-second hop; 521 event scores versus 1,024-dimensional embeddings.
- EfficientAT: 32 kHz; 128 mel bins; 800-sample window; 320-sample hop; AudioSet logits.

Use `wrap_loaded_encoder(...)` only after the exact model artifact has been obtained, loaded and licensed. The wrapper validates the output tensor shape; it does not establish task accuracy, conversion success, latency or iPhone execution.

### Official pretrained encoders

YAMNet is loaded from Google's official TensorFlow SavedModel, not a recreated architecture. Its frozen 1,024D frame embeddings feed a standardized logistic-regression tile head or ridge fill head. The fitted coefficients are exported as non-executable JSON; the TFLite exporter implements the same normalization and linear head in TensorFlow.

```bash
pip install -e ".[pretrained]"
python -m thud yamnet-train https://tfhub.dev/google/yamnet/1 tiles_manifest.csv --task tile --output tile_yamnet.json
python -m thud yamnet-evaluate https://tfhub.dev/google/yamnet/1 tiles_manifest.csv --task tile --group-by object_id
python -m thud yamnet-predict https://tfhub.dev/google/yamnet/1 tile_yamnet.json path/to/tap.wav
```

The official URL downloads/caches weights on first use. Substitute a local SavedModel directory for offline use; a weights-only `.h5` is not accepted. Fill training uses `--task fill`, and prediction requires `--family` matching the trained family. Heads from the former recreated encoder must be retrained.

EfficientAT uses the official `mn04_as_mAP_432.pt` release (SHA-256 `899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c`) and unchanged upstream sources pinned to commit `a425fdce92572e602a1d5634799bd9f1f2efa806`. Loading checks the checkpoint hash, verifies source Git blob IDs, strictly loads every weight, freezes the network, and preserves its upstream frontend. It returns 384D embeddings. Source fetching is explicit; inference never downloads.

```bash
# Use matching torch/torchvision/torchaudio builds.
pip install -e ".[efficientat]"
python -m thud efficientat-fetch artifacts/efficientat-source
# Obtain mn04_as_mAP_432.pt from the official release linked below.
python -m thud efficientat-train artifacts/mn04_as_mAP_432.pt tiles_manifest.csv --source-dir artifacts/efficientat-source --task tile --output tile_efficientat.json
python -m thud efficientat-evaluate artifacts/mn04_as_mAP_432.pt tiles_manifest.csv --source-dir artifacts/efficientat-source --task tile --group-by object_id
python -m thud efficientat-predict artifacts/mn04_as_mAP_432.pt tile_efficientat.json path/to/tap.wav --source-dir artifacts/efficientat-source
```

Official artifacts: [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet), [EfficientAT release](https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1).

### TensorFlow → TFLite / LiteRT (preferred deployment path)

```bash
# Combined waveform frontend, official encoder, normalization and task head.
python -m thud yamnet-export https://tfhub.dev/google/yamnet/1 --model tile_yamnet.json --output tile_yamnet.tflite
# Omit --model to export just the 1,024D mean embedding.
python -m thud yamnet-export https://tfhub.dev/google/yamnet/1 --output yamnet_embedding.tflite
```

The export is float32, fixed-shape and uses **TFLite built-in operators only**, with no Select TF Ops dependency. It refuses to overwrite an existing file. Input is exactly **15,600 float32 mono samples at 16 kHz (975 ms), normalized to [-1, 1]**, with no batch dimension. This includes the STFT window needed for one complete 96-frame patch. The app must resample/downmix and validate captures before calling the model. Use the same capture alignment, crop and zero-padding when constructing training WAVs; training on longer, mean-pooled recordings and deploying a single patch is not prediction-equivalent.

Tile output is one raw decision margin: `> 0` means `debonded`, otherwise `bonded`; it is **not a calibrated probability**. Fill output is one unclamped value in percentage points and must only be used for the trained container family. Embedding-only output is a vector of length 1,024. Ship the head JSON alongside the model to retain class/family provenance; the `.tflite` does not enforce the family itself.

Desktop TensorFlow 2.19 conversion and interpreter execution were exercised on synthetic tones. Combined tile and fill files each measured **12,886,324 bytes**; maximum absolute differences against TensorFlow plus the JSON head were **0.000144 margin units** and **0.001993 percentage points**, respectively. Embedding-only export measured 12,872,896 bytes. Local reports live in ignored `artifacts/tflite-parity.json`. A float16 trial shifted tile margins by up to 0.70, so quantized export is intentionally not exposed. Int8 calibration and hardware delegate coverage are unmeasured.

[LiteRT](https://ai.google.dev/edge/litert) continues TensorFlow Lite and deploys `.tflite` models on Android/iOS and other supported edge platforms. Built-in-only conversion avoids the additional [Select TF Ops runtime](https://ai.google.dev/edge/litert/models/ops_select). This model is phone-class, not a verified fit for microcontrollers. The desktop checker uses the interpreter bundled with the pinned TensorFlow 2.19 extra; that API emits a deprecation warning in favor of `ai_edge_litert`.

### Does PyTorch support mobile?

**Yes: use [ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html)** for new deployments. PyTorch's mobile entry point redirects there. Export a supported model using `torch.export`, lower it to an appropriate backend, and serialize an ExecuTorch `.pte` program. This is distinct from the repository's existing desktop TorchScript `.ts` export.

- [iOS](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html): Swift/Objective-C/C++ integration, XNNPACK CPU and Core ML backends. The documented prebuilt Swift package requires **iOS 17+**. Accelerator support depends on the model operators and backend, not merely on successful export.
- [Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html): native runtime integration and CPU/hardware-specific backends.
- Neither the task CNN nor EfficientAT has been exported or executed with ExecuTorch here; their audio frontends also need export compatibility. No PyTorch rewrite is needed merely because mobile deployment exists.

Decision: use TensorFlow YAMNet → built-in-only TFLite first; keep EfficientAT/PyTorch as the research comparator rather than add another mobile runtime now.

### Measured pretrained behavior and remaining limits

Both official encoders loaded and ran locally. YAMNet produced `(13, 521)` scores and `(13, 1024)` embeddings for Google's reference `miaow_16k.wav`, with top mean class `Animal` (0.7132). EfficientAT strict-loaded 983,599 parameters; real-artifact tests exercised changed-waveform embeddings and upstream frontend equivalence. Actual CLI train/predict/object-held-out evaluation ran for both encoders.

The generated-frequency pilot uses 32 tile-labelled WAVs and 20 fill-labelled WAVs. **Labels encode synthetic frequency cues, not physical bonded state or measured water volume.** Both frozen encoders achieved 100% tile-class accuracy on object and session holdouts. These figures demonstrate the software path only:

| Synthetic fill head | Object-held-out MAE (percentage points) | Session-held-out MAE (percentage points) |
|---|---:|---:|
| Classical features | 2.8581 | 73.5159 |
| Official YAMNet | 1.2930 | 3.0198 |
| Official EfficientAT mn04_as | 0.5138 | 1.4330 |

Local evidence: ignored `artifacts/pretrained-comparison.json`. Rerun YAMNet with `python -m experiments.pretrained_pilot --output new_results.json`; the experiment's `run(checkpoint, efficientat_bundle)` accepts an explicitly loaded EfficientAT bundle for the same comparison. The task CNN's separate synthetic diagnostic uses a different generator and cannot be ranked against this table.

### Real public-data checks (adjacent tasks, not THUD validation)

The following experiments use real recordings and leakage-safe splits. They test whether the encoders carry signal for related tasks; they do **not** measure tile-bonding or static-vessel-fill accuracy.

| Dataset/task | Data and split | Frozen YAMNet | Frozen EfficientAT `mn04_as` | Simple comparator |
|---|---|---:|---:|---:|
| Sound of Water: pouring-container **shape**, held-out containers | 125 mono 48 kHz WAVs from 48 containers; 36 train / 12 test containers, zero overlap; 12 test containers | **58.33%** accuracy, balanced accuracy 41.11%, macro-F1 42.86% | **41.67%** accuracy, balanced accuracy 32.22%, macro-F1 26.46% | MFCC mean/std: 41.67% / 31.11% / 31.48%; majority accuracy 50.00% |
| DCASE 2024 Task 2: **valve anomaly**, one section | 1,200 ten-second clips; initial control fit on 990 source-normal training clips only; 200 labeled test clips | pooled AUC **0.528**, standardized pAUC@0.1 0.505; source-normal-vs-all-anomaly AUC 0.577; target 0.479 | Follow-up frozen-encoder ablations below | — |

Sound of Water used 125 of 790 unique metadata items from the pinned dataset revision and selected up to three recordings per container before feature extraction. The held-out shape test contains one bottleneck, six cylindrical and five semiconical containers; the bottleneck recall was therefore not informative. EfficientAT desktop encoder-forward p50/p95 was 26.91/42.37 ms on this run, not an iPhone measurement. Shape is a metadata label invariant per container; there are no static tap recordings or fill/volume labels. Reports and reproducible scripts: `artifacts/sound-water-benchmark.json`, `artifacts/sound-of-water/benchmark.py`, and `artifacts/sound-of-water/efficientat_benchmark.py`.

DCASE used an unsupervised Mahalanobis head over mean YAMNet embeddings. The result is near chance overall and below chance on the target-domain official-style comparison. It is one valve section, not the official aggregate score; YAMNet is not a valve-fault model. The verified archive is CC BY-NC-SA 4.0. Report and script: `artifacts/dcase-benchmark.json` and `artifacts/dcase/run_valve_benchmark.py`.

ObjectFolder Real was **not scored**. The official Dropbox material-classification archive is 76.35 GiB; the official selective first-10-object audio shard is still 33.87 GiB gzip-tar and cannot be safely random-extracted by HTTP range without downloading/indexing the stream. No audio/label mapping was assumed. Its README's published 51.02% Real result is not our measurement. Evidence: `artifacts/objectfolder-access.json` and `artifacts/objectfolder-benchmark.json`.

Decision from these checks: keep YAMNet as the first cross-platform pretrained baseline, but do not promote either encoder based on these adjacent tasks. The only useful next accuracy measurement for THUD remains a real, specimen-labelled tile and measured-fill vessel pilot with object/session holdouts.

#### Exploratory improvement ablations

These test sets had already been inspected before the ablations. Training-only fitting and tuning prevent direct label leakage, but do not make this round independent confirmation. The original controls and holdouts remain unchanged.

Sound of Water: six fixed configurations, with head hyperparameters selected by the original three container-disjoint training folds. The configuration selected by training macro-F1 was three ordered YAMNet bins, not the best-looking test result.

| Shape configuration | Training CV macro-F1 | Held-out correct / 12 | Held-out macro-F1 |
|---|---:|---:|---:|
| YAMNet mean control | 0.3791 | 7 | 0.4286 |
| YAMNet mean + standard deviation | 0.3586 | 6 | 0.3175 |
| Three ordered temporal bins (training-selected) | 0.5036 | 4 | 0.2262 |
| YAMNet + MFCC | 0.3793 | 7 | 0.4048 |
| YAMNet + EfficientAT | 0.4313 | 6 | 0.3485 |
| YAMNet mean + RBF SVM | 0.3732 | 3 | 0.1778 |

No tested variant improved over the original control on held-out accuracy or macro-F1. Every variant missed the single bottleneck container; each correct container changes accuracy by 8.33 percentage points. Do not promote temporal pooling based on training CV alone. Reproduce with `python experiments/water_ablation.py`; protocol, scores, hashes, and fitted head arrays are retained in `artifacts/water-ablation-protocol.json`, `artifacts/water-ablation.json`, and `artifacts/water-ablation-heads.npz`.

DCASE: 19 fixed variants compare temporal mean/std, inclusion of the ten target-normal training clips, two normal-domain centers, cosine nearest-neighbor scoring, log-mel/PCA reconstruction, feature fusion, and frozen EfficientAT. No test labels enter scorer fitting, and no anomaly-labelled training data are used. Source/target AUC below compares that domain's normal test clips against **all** test anomalies, not just same-domain anomalies.

| Valve configuration | Pooled AUC | Source AUC | Target AUC | Standardized pAUC@0.1 |
|---|---:|---:|---:|---:|
| YAMNet mean / source-normal Mahalanobis control | 0.5280 | 0.5770 | 0.4790 | 0.5047 |
| YAMNet mean / all-normal Mahalanobis | 0.5283 | 0.5772 | 0.4794 | 0.5032 |
| YAMNet mean/std / all-normal Mahalanobis | 0.4952 | 0.5142 | 0.4762 | 0.4932 |
| EfficientAT / all-normal Mahalanobis | 0.5680 | 0.5410 | 0.5950 | 0.5116 |
| EfficientAT / source-normal cosine 1-NN | 0.6085 | 0.5682 | 0.6488 | 0.5047 |
| EfficientAT / all-normal cosine 1-NN | 0.6095 | 0.5682 | 0.6508 | 0.5047 |

EfficientAT cosine 1-NN has the largest observed pooled AUC, but is **not a validated winner**. Its +0.0815 gain versus control has a descriptive paired-bootstrap 95% interval of [-0.0090, +0.1794] (1,000 label-stratified clip resamples, seed 42). This is not corrected for inspecting 19 variants, and correlated clips can make it optimistic. Low-false-positive pAUC does not improve. Adding ten target normals changes its AUC by only +0.0010; the observed gain mostly comes from the encoder/scorer combination, not adaptation. None of the tested temporal, domain-center, PCA or fusion variants beats the original pooled control.

Numerical caveat: regenerated YAMNet means match the original cache exactly. The original float32 Mahalanobis control reproduces AUC 0.5280, whereas casting the same features to float64 yields 0.5022; covariance inversion is precision-sensitive. The fixed matrix preserves original float32 inputs for comparable controls. Prefer the simpler cosine-distance candidate for independent validation rather than treating covariance precision as a tunable test-set hyperparameter.

Reproduce with `python -m experiments.dcase_ablation`; the fixed plan, feature/checkpoint identity, per-clip scores, metrics and uncertainty intervals are in `artifacts/dcase-ablation-plan.json`, `artifacts/dcase-ablation-features.npz`, and `artifacts/dcase-ablation.json`. The script includes scorer self-checks and checks audio hashes and control reproduction. Freeze EfficientAT cosine 1-NN before evaluating fresh machine data; use a separate normal-only calibration set before making threshold/false-alarm claims. Neither dataset establishes THUD tile-bonding or static-vessel-fill performance.


Still not implemented or measured: Apple AudioFeaturePrint, CLAP/PANNs integration; Sound-of-Water model inference beyond the audio-only benchmark; Core ML or ExecuTorch export; native/Flutter microphone capture and UI; physical iPhone/Android execution, latency, memory or battery; real THUD tile/fill accuracy. The first physical comparison remains classical features, frozen YAMNet, frozen EfficientAT and the task CNN on identical specimen/session splits.

Example JSON output:
```json
{
  "sample_rate": 48000,
  "duration_s": 0.7,
  "onset_s": 0.12,
  "feature_version": 1,
  "features": {
    "peak_hz": 1200.0,
    "centroid_hz": 1420.5,
    "bandwidth_hz": 530.2,
    "flatness": 0.0034,
    "band_100_500": 0.001,
    "band_500_1500": 0.985,
    "band_1500_4000": 0.012,
    "band_4000_7500": 0.002,
    "energy_t50_s": 0.032,
    "energy_t90_s": 0.081,
    "tail_energy_ratio": 0.00003,
    "mfcc_mean_1": -14.2,
    "mfcc_std_1": 2.1
  },
  "quality": {
    "impact_to_baseline_db": 74.5,
    "analysis_window_s": 0.25,
    "tail_energy_ratio": 0.00003,
    "warnings": []
  }
}
```

### 2. Group-held-out evaluation

Evaluates model generalization across whole objects or recording sessions using leave-one-group-out cross-validation. Scaling and model fitting occur strictly on the training partition of each fold:

```bash
# Tile bonding held-out evaluation across distinct physical objects
python -m thud evaluate tiles_manifest.csv --task tile --group-by object_id

# Vessel fill held-out evaluation across recording sessions
python -m thud evaluate fill_manifest.csv --task fill --group-by session_id
```

### 3. Train a portable JSON model

Fits a model on the full manifest (scaling parameters and linear weights) and saves it to a JSON file. The command uses exclusive file creation to prevent accidental overwrites:

```bash
# Train tile screening model
python -m thud train tiles_manifest.csv --task tile --output tile_model.json

# Train vessel fill model
python -m thud train fill_manifest.csv --task fill --output fill_model.json
```

### 4. Predict using a saved model

Loads and validates the JSON model schema, extracts features from the target WAV, and computes linear inference using NumPy:

```bash
# Predict tile bonding
python -m thud predict tile_model.json path/to/tap.wav

# Predict vessel fill (requires explicit --family matching the model)
python -m thud predict fill_model.json path/to/tap.wav --family steel-bottle-500ml
```

## Manifest format

Manifests are CSV files with relative WAV paths and metadata. Paths are resolved relative to the manifest file location.

### Tile manifest (`--task tile`)
```csv
path,label,object_id,session_id,recording_id
taps/tile0_tap1.wav,bonded,tile_0,session_1,rec_001
taps/tile0_tap2.wav,bonded,tile_0,session_1,rec_002
taps/tile1_tap1.wav,debonded,tile_1,session_1,rec_003
```

### Fill manifest (`--task fill`)
All rows in a fill manifest must belong to the same `family`. Labels represent known volume percentage points in `[0, 100]`:
```csv
path,label,object_id,session_id,recording_id,family
taps/vessel0_10pct.wav,10.0,vessel_0,session_1,rec_101,steel-bottle-500ml
taps/vessel0_50pct.wav,50.0,vessel_0,session_1,rec_102,steel-bottle-500ml
taps/vessel0_90pct.wav,90.0,vessel_0,session_1,rec_103,steel-bottle-500ml
```

## Running tests

Run the regression and behavioral test suite:

```bash
python -m pytest -v
```

## Limitations and experimental evidence boundaries

1. **Synthetic vs. real acoustic evidence:** The unit tests use synthetic damped sinusoids to verify mathematical consistency, DSP behavior, and fold isolation. Synthetic signals exercise software execution; they do not establish that microphone taps can detect delamination or container fill in real-world environments.
2. **Decay times vs. room acoustics:** The energy times `energy_t50_s` and `energy_t90_s` measure energy decay within a 250 ms window following an impact. They are not room-acoustic $T_{20}$, $T_{60}$, or $EDT$ estimates and must not be reported as such.
3. **Hardware & compute limits:** The JSON model format allows zero-dependency deployment on mobile devices (e.g., via Dart/Flutter or native Swift), but no physical iPhone 11 hardware execution, memory footprint, or battery impact was measured in this session.
4. **Physical data requirement for field evaluation:**
   - **Tile pilot:** At least 8 physically distinct specimens with known bonded/debonded ground truth, 2 recording sessions, 2 operators, and 3 bursts of 3 taps per point. Hold out whole specimens to verify cross-tile generalization.
   - **Fill pilot:** At least 2 safe rigid vessels with volume measured at 0/25/50/75/100%, fixed liquid, cap state, and tap point. Fit on session 1 and test on session 2.
