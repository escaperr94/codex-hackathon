# Research encoders and anomaly scorers

Run from `models/`:

```powershell
python -m experiments.finalize_research
```

This uses preserved local downloads and real cached features; it does not download data. Outputs: `artifacts/final/research/verification.json`, 32 builtin-only float32 TFLites, and adjacent compact `.parity.npz` real-input/reference fixtures. Historical reports remain unchanged. Each export records SHA-256, tensor contract, reference errors and desktop head-only timing. TensorFlow 2.19's Interpreter is the exercised runtime, not a mobile runtime.

## Artifacts and limits

- **YAMNet waveform encoder:** `yamnet-waveform-embedding-f32.tflite`, 12,872,896 bytes. Input `float32[15600]` at 16 kHz, mono PCM in [-1,1], no batch axis. Output `float32[1024]`, mean embedding, not 521 event probabilities. DSP and encoder included; decoding/resampling/alignment outside. Three actual pouring-audio windows matched the SavedModel with maximum absolute error 7.63e-6. The 975 ms contract is deliberately not silently substituted for a one-second or whole-recording trained head.
- **Pouring shape head:** `pouring-shape-yamnet-three-bins.tflite`, input `float32[1,3072]`, output three softmax scores in `[bottleneck,cylindrical,semiconical]` order. Whole-recording YAMNet frame embeddings must be split chronologically into three near-equal bins and each bin mean concatenated. The exported scaler/logistic head is the original training-only fit chosen by training-container validation, not a refit on test examples. All 125 real cached rows retain their reference decisions; one complete source waveform was re-extracted to verify frontend parity. Held-out 12-container accuracy is only 33.33%, balanced accuracy 23.33%, macro-F1 0.2262: retain as a **research comparator, not an adoption recommendation**, and never call this fill or volume estimation.
- **28 per-section machine scorers:** every section in the core/extra cross-corpus valve/RoboticArm reports has separate EfficientAT cosine 1-NN and log-mel Gaussian exports. The first expects `float32[1,384]` frozen official EfficientAT embeddings and outputs nearest normal cosine distance in [0,2]; zero-vector cosine convention is distance 1. The second expects `float32[1,256]` mean/std of the official 128-bin frontend and outputs nonnegative squared Mahalanobis distance. Both fit only the section's normal training rows; no test labels/statistics enter fit or thresholds. They include numerical scoring, **not EfficientAT/frontend**. A Cholesky factor implements Gaussian sum-of-squares scoring. One actual WAV per section is decoded and re-extracted against both cached representations, including 2026's explicit near-channel-only rule.
- **Two task-trained valve scoring heads:** `valve2024-{jvachier,stgram}-mahalanobis.tflite` expect `float32[1,128]` from their different, noninterchangeable original encoders/frontends. Normal-only covariance refits are included; the encoders are not. Input preprocessing metadata and source report hashes are retained. These files do not turn either encoder into a converted mobile model.

All numerical scorer outputs are distances/scores, not calibrated defect probabilities or unique fault diagnoses. No test-derived anomaly threshold is shipped. A training-normal score is not a deployment accuracy measure. Reusing the same reference recordings is not an independent field test.

## Verification

The export command directly allocates and invokes every model on every cached real feature row, checks finite outputs/exact shapes/float32 dtype, audits FlatBuffer operator codes (CUSTOM disallowed, so no Flex/Select TF Ops), compares to float64 reference scoring, and exercises finite zero/mean feature boundaries. The maximum scaled numerical error was 3.80e-6; the largest pooled test AUC difference from float32 score ordering was 0.000025. Numerical parity is not identical rank ordering for arbitrarily close scores. Full reports contain the observed errors rather than a claim of perfect scoring identity.

Desktop timing covers numerical inference only and excludes source capture, file decode, resampling and external encoders unless the waveform encoder is explicitly being discussed. These artifacts have no physical iPhone, Core ML, GPU/Neural Engine, memory, battery or thermal verification.

## Provenance and permissions

The [dataset inventory](DATASETS.md) preserves original source links and restrictions. DCASE2023 valve is byte-identical to MIMII DG section00, so those are not independent replications. MIMII DG/DCASE2023 metadata and descriptions have conflicting license terms; apply the restrictive interpretation pending review. Original model/data permissions remain binding; exporting weights/reference banks is not a new redistribution grant. References retained here are local research artifacts, not commercially cleared app assets.
