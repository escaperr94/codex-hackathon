# Durian knock-audio experiment

## Evidence and sources

Executed on desktop CPU, not an iPhone. Acoustic input only; no image/thermal/chemical features.
Dataset: [Zenodo 18603796](https://doi.org/10.5281/zenodo.18603796), [primary API](https://zenodo.org/api/records/18603796), [README](https://zenodo.org/api/records/18603796/files/README.md/content). CC BY 4.0; attribute Mesa-Satina et al. (2026). Downloaded only audio, characteristics CSV, README and API metadata.

Reproduce: `python experiments/durian_benchmark.py --prepare --run`. Verified source checksums and SHA-256 are in `artifacts/datasets/durian/manifest.json`. Model hashes, package versions, features, splits, tuning and predictions are retained in `artifacts/durian-*`.

## Population and protocol

```json
{
  "source_characteristic_fruits": 189,
  "recordings": 562,
  "recorded_fruits": 175,
  "unrecorded_fruit_ids": [
    4,
    68,
    88,
    106,
    127,
    134,
    135,
    136,
    137,
    138,
    154,
    161,
    163,
    164
  ],
  "nominal_fruit_counts": {
    "Unripe": 63,
    "Ripe": 63,
    "Overripe": 63
  },
  "actual_fruit_counts": {
    "Unripe": 63,
    "Ripe": 63,
    "Overripe": 59,
    "Disease": 2,
    "": 2
  },
  "excluded_actual_ids": [
    19,
    35,
    163,
    164
  ],
  "primary_recordings": 133,
  "primary_fruits": 133,
  "contradictory_duplicate_groups": 105,
  "contradictory_duplicate_members": 211,
  "usable_actual_by_device_class": {
    "android / Unripe": 57,
    "iphone / Unripe": 57,
    "iphone / Ripe": 41,
    "iphone / Overripe": 35
  },
  "usable_nominal_by_device_class": {
    "android / Unripe": 68,
    "iphone / Unripe": 171,
    "iphone / Ripe": 66,
    "iphone / Overripe": 44,
    "android / Overripe": 1,
    "android / Ripe": 1
  },
  "ambiguous_origin_members": [
    "dataset_clean_sound/durian_100_overripe_android.wav",
    "dataset_clean_sound/durian_100_unripe_android.wav",
    "dataset_clean_sound/durian_180_ripe_android.wav",
    "dataset_clean_sound/durian_180_unripe_android.wav"
  ],
  "nominal_by_actual_status": {
    "Unripe / Unripe": 63,
    "Ripe / Ripe": 55,
    "Overripe / Overripe": 52,
    "Overripe / Disease": 1,
    "Ripe / Disease": 1,
    "Overripe / Ripe": 8,
    "Ripe / Overripe": 7,
    "Overripe / ": 2
  },
  "class_by_actual_status": {
    "A / Unripe": 21,
    "A / Ripe": 21,
    "A / Overripe": 20,
    "A / Disease": 1,
    "B / Unripe": 21,
    "B / Ripe": 21,
    "B / Disease": 1,
    "B / Overripe": 18,
    "C / Unripe": 21,
    "C / Overripe": 21,
    "C / Ripe": 21,
    "B / ": 2
  },
  "actual_recording_counts": {
    "Unripe": 114,
    "Ripe": 214,
    "Overripe": 228,
    "Disease": 6
  },
  "device_counts": {
    "android": 281,
    "iphone": 281
  },
  "maturity_by_actual_status": {
    "Immature / Unripe": 21,
    "Immature / Ripe": 21,
    "Immature / Overripe": 19,
    "Immature / Disease": 2,
    "Mature / Unripe": 21,
    "Mature / Ripe": 21,
    "Mature / Overripe": 21,
    "Overmature / Unripe": 21,
    "Overmature / Ripe": 21,
    "Overmature / Overripe": 19,
    "Overmature / ": 2
  },
  "label_disagreements": 19,
  "retries_per_fruit_device": {
    "1": 140,
    "2": 208,
    "3": 2
  },
  "audio_duration_seconds": {
    "min": 15.2,
    "median": 31.60846875,
    "max": 53.4093125
  },
  "primary_excluded_fruit_ids": [
    4,
    11,
    15,
    17,
    18,
    19,
    21,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    50,
    52,
    54,
    59,
    60,
    62,
    63,
    68,
    76,
    78,
    82,
    84,
    88,
    93,
    101,
    106,
    113,
    114,
    121,
    126,
    127,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    142,
    146,
    154,
    155,
    159,
    161,
    163,
    164,
    166,
    177,
    178,
    181,
    183,
    184
  ],
  "primary_date_by_class": {
    "2025-09-06 / Unripe": 20,
    "2025-09-12 / Ripe": 14,
    "2025-09-15 / Overripe": 2,
    "2025-09-12 / Overripe": 3,
    "2025-09-15 / Ripe": 2,
    "2025-09-17 / Overripe": 1,
    "2025-09-19 / Unripe": 18,
    "2025-09-23 / Ripe": 15,
    "2025-09-25 / Ripe": 2,
    "2025-09-26 / Overripe": 6,
    "2025-10-02 / Overripe": 3,
    "2025-09-25 / Overripe": 8,
    "2025-10-03 / Overripe": 1,
    "2025-09-23 / Unripe": 5,
    "2025-09-24 / Unripe": 11,
    "2025-09-27 / Overripe": 1,
    "2025-09-29 / Overripe": 1,
    "2025-10-02 / Unripe": 3,
    "2025-10-06 / Ripe": 8,
    "2025-10-06 / Overripe": 1,
    "2025-10-07 / Overripe": 7,
    "2025-10-09 / Overripe": 1
  }
}
```

Full recording-level exclusions and date/label cross-tabs remain in results JSON.

The physical fruit identifier is the numeric replicate suffix in characteristics Code / sound sample_num, not the recording filename, phone, timestamp, retry or ripeness code. Every observed recording of a fruit stays together. Five stratified fruit folds; three inner fruit folds tune C ∈ {0.01, 0.1, 1} using training-only per-fruit balanced accuracy. The same outer/inner seed and fruit protocol are used for each representation. Scaling and regularized multinomial logistic regression are fitted only to each training fold; class/fruit-balanced fit weights prevent repeats dominating.

Primary experiment is iPhone-only: target is the separate `Actual_Ripening_Status` field, restricted to recording-stage labels matching the characteristics table's nominal `Ripeness`. This operational terminal-stage association is not a verified per-recording destructive-reference timestamp; biological accuracy remains conditional on that association. Earlier unripe audio is never assigned the fruit's terminal ripe/overripe status. Disease/missing statuses, ambiguous source-member associations and contradictory duplicate labels are excluded. A second iPhone-only task uses the sound CSV's per-recording `ripeness` stages across usable audio, including earlier sessions, as a nominal-label sensitivity analysis. Characteristic nominal and actual disagreements are audited separately (19 fruits); these are not corrected using model predictions.

The sound CSV cross-joins multiple originals onto repeated output filenames. Each actual ZIP member is counted once; archive uncompressed bytes must match one unique source original/timestamp. Four Android members with equal-sized but different possible originals are excluded from both supervised tasks. Full candidate rows, matching origins, exclusions and audio hashes are retained in recordings JSON. This does not recover retries that the cleaned archive did not preserve.

A byte-level SHA-256 audit found 105 contradictory Android groups (211 archive members): exactly identical audio bytes under two or three different stage filenames. All such members are excluded; no model is asked to distinguish identical recordings with contradictory labels. No exact duplicates cross fruit IDs. The cleaned files are therefore not 562 independent trustworthy labelled examples. See integrity-audit JSON.

All audio is decoded by FFmpeg to 16 kHz mono PCM. A fixed 100 Hz high-pass RMS peak detector selects up to 18 peaks at least 0.5 seconds apart; 1-second windows include 0.1 seconds before each selected peak. No supervised tap selection or test-dependent tuning. Some selected peaks may be handling/noise/speech: the dataset has no tap boundary annotations, so this is a limitation, not verified perfect segmentation.

MFCC branch: 20 MFCC means/SD, centroid/bandwidth/rolloff/flatness mean/SD, zero-crossing mean/SD, six normalized spectral bands; mean/SD across windows = 112 floats. YAMNet: official local Google SavedModel, frozen 1,024D frame embeddings mean-pooled across windows, not its 521 sound labels. Hybrid concatenates the two (1,136D). No neural backbone fine-tuning. A fresh tiny CNN was not trained: the small, noisy-label fruit population does not justify adding a new high-variance learner to this bounded comparison; this does not establish that CNNs cannot improve.

## Held-out results

Class order in all confusion matrices: Unripe, Ripe, Overripe. Majority is selected on training fruit only. Per-fruit scores average recording probability vectors, not taps counted as independent fruits. For the longitudinal nominal-stage task, aggregation is fruit × true recording-stage (a fruit can have multiple targets at different times); these groups are never averaged across incompatible labels. All stages still share a single held-out fruit fold. Primary and sensitivity task populations differ, so their scores are not a head-to-head biological-label comparison.

| Target / protocol | Model | Recording balanced accuracy | Fruit(/stage) balanced accuracy | Fruit(/stage) macro F1 |
|---|---|---:|---:|---:|
| actual_iphone_primary | majority | 0.333 | 0.333 | 0.200 |
| actual_iphone_primary | mfcc_spectral | 0.815 | 0.815 | 0.815 |
| actual_iphone_primary | yamnet | 0.807 | 0.807 | 0.808 |
| actual_iphone_primary | hybrid | 0.810 | 0.810 | 0.809 |
| actual_pooled_secondary_device_confounded | majority | 0.333 | 0.333 | 0.200 |
| actual_pooled_secondary_device_confounded | mfcc_spectral | 0.789 | 0.804 | 0.805 |
| actual_pooled_secondary_device_confounded | yamnet | 0.772 | 0.787 | 0.789 |
| actual_pooled_secondary_device_confounded | hybrid | 0.821 | 0.833 | 0.836 |
| nominal_iphone_stage_sensitivity | majority | 0.333 | 0.333 | 0.252 |
| nominal_iphone_stage_sensitivity | mfcc_spectral | 0.744 | 0.744 | 0.732 |
| nominal_iphone_stage_sensitivity | yamnet | 0.724 | 0.724 | 0.719 |
| nominal_iphone_stage_sensitivity | hybrid | 0.711 | 0.711 | 0.706 |

On the fixed 133-fruit primary task, MFCC/spectral balanced accuracy was 81.5%, YAMNet 80.7%, and hybrid 81.0%, versus 33.3% for majority. This small observed gap does not establish statistical superiority; the hybrid did not improve the primary comparison. The low-dimensional head is therefore a reasonable conservative edge starting point, conditional on new independent acquisition and reliable time-matched ground truth.


Cross-device three-class evaluation is **not estimable** after integrity exclusions: the Android primary actual-status subset contains only Unripe. Android nominal stage data have 68 Unripe, one Ripe and one Overripe; this does not support a reliable stratified training comparison. The pooled actual-status experiment is secondary and explicitly device/class-confounded. Its per-device scores below are descriptive, not evidence of Android three-class ripeness generalization. Different phone codecs/formats remain confounded with device identity.

### Secondary pooled per-device recording scores

| Model | Device | N | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| majority | android | 57 | 1.000 | 0.333 |
| majority | iphone | 133 | 0.333 | 0.200 |
| mfcc_spectral | android | 57 | 0.930 | 0.321 |
| mfcc_spectral | iphone | 133 | 0.786 | 0.786 |
| yamnet | android | 57 | 1.000 | 0.333 |
| yamnet | iphone | 133 | 0.758 | 0.758 |
| hybrid | android | 57 | 1.000 | 0.333 |
| hybrid | iphone | 133 | 0.810 | 0.809 |

## CPU inference proxy

Host: `Windows-11-10.0.22624-SP0` / `Intel64 Family 6 Model 141 Stepping 1, GenuineIntel`. Two math/TF threads requested. Timings include feature extraction and a fitted head, but exclude decode, capture wait and model cold-load; whole-recording timings include peak detection. Single-impact timings start from an already isolated 1-second segment. Twelve calls, first two discarded. Not a phone latency, memory, thermal or energy benchmark.

| Input / representation | Windows | p50 ms | p95 ms |
|---|---:|---:|---:|
| single_impact_1s/mfcc_spectral | 1 | 5.43 | 7.59 |
| single_impact_1s/yamnet | 1 | 17.24 | 19.90 |
| single_impact_1s/hybrid | 1 | 20.08 | 23.58 |
| whole_recording/mfcc_spectral | 18 | 97.91 | 108.23 |
| whole_recording/yamnet | 18 | 227.31 | 302.87 |
| whole_recording/hybrid | 18 | 244.70 | 273.10 |

## Interpretation and limits

This is one farm, one harvest season, controlled rubber-stick tapping at approximately five inches, iPhone 13 and Redmi Note 14; it is not iPhone 11 validation or arbitrary-fruit ripeness. Maturity stage, harvest batch, collection date/time, storage treatment, background and nominal classes can covary with ripeness. Date/class/maturity counts are preserved; group holding prevents same-fruit leakage but does not remove these acquisition confounds. A new farm/season/user/session and known reference measurement validation are required before causal or commercial ripeness claims.

The README describes temporal tracking yet the characteristics table provides one Code/Replicate per fruit, not a validated longitudinal identity map across distinct numeric IDs. Numeric fruit identity is therefore the strongest released grouping evidence; undisclosed identity reuse cannot be excluded. Class probabilities are uncalibrated. Five-fold estimates are not a product acceptance threshold or evidence that the best observed row is universally best. Cross-device plus fruit holdout is stricter than testing a second phone on the same training fruit.

No physical iPhone inference, Core ML conversion, energy, battery, peak RAM or unseen-farm generalization was measured. Model sizes are not runtime RAM. Raw data retain the source CC BY license; no other modality archives were downloaded.

## Portable TensorFlow Lite numerical heads

Reproduce: `python experiments/durian_benchmark.py --export-heads`. Five fixed-shape float32 builtin-only `.tflite` files preserve each primary outer fold's training-only chosen C. They include 112D feature scaling, linear logits and softmax (input `[1,112]`, output `[1,3]`). These are evaluation-fold heads, not a full-data production model. No runtime dependencies installed.

Across all 133 original held-out predictions: maximum probability absolute discrepancy 3.56e-07; class-decision changes 0. Desktop TFLite head calls include input/output tensor copies, 20 warm-ups plus 100 timed calls per fold, two CPU threads requested.

| Fold | Bytes | Head CPU p50 ms | p95 ms | Max probability difference |
|---|---:|---:|---:|---:|
| 0 | 3180 | 0.00390 | 0.00430 | 1.54e-07 |
| 1 | 3180 | 0.00485 | 0.00692 | 3.56e-07 |
| 2 | 3180 | 0.00390 | 0.00490 | 1.96e-07 |
| 3 | 3180 | 0.00380 | 0.00451 | 2.56e-07 |
| 4 | 3180 | 0.00400 | 0.00552 | 2.76e-07 |

Only the numerical head is converted. Audio decoding, impact selection and MFCC/spectral frontend still run in the desktop reference implementation; frontend parity and full-pipeline mobile conversion are unmeasured. These microsecond head times exclude features/capture and are not end-to-end iPhone latency. No physical iPhone deployment was exercised.
