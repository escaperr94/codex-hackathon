# Waveform-to-material TFLites

Run after the classifier finalizer, from `models/`:

```powershell
python -m experiments.finalize_waveform
```

Two real-data trained LinearSVC heads are combined with the official frozen YAMNet DSP/encoder. These are actual waveform-to-score models rather than only numerical feature heads. Both use fixed `float32[16000]` mono PCM at 16 kHz in [-1,1], no batch axis. Outputs are `scores: float32[1,K]` uncalibrated one-vs-rest margins and `class_index: int32[1]`. Class order is preserved in `artifacts/final/waveform/verification.json`.

| Model | Bytes | Actual recordings checked | Maximum score error | Class changes | Desktop p50 ms | Desktop p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| YCB material | 12,915,988 | 131 | 9.80e-5 | 0 | 3.29 | 3.49 |
| RealImpact material | 12,891,388 | 270 | 2.98e-5 | 0 | 3.32 | 3.58 |

Timing is desktop two-thread waveform-to-score processing, 20 warmups/100 calls, not phone timing. It excludes capture, source decoding, resampling and crop/normalization. Reference scores come from the selected full-data sklearn heads on real cached embeddings; this verifies conversion, not in-sample generalization or a fresh accuracy test. No synthetic task weights are shipped.

YCB external preprocessing: decode channels to mono mean, polyphase resample to 16 kHz, center a one-second crop on the loudest 20 ms block and zero-pad if needed. Exact reference: `experiments.material_benchmark.waveform`. The nine classes are Aluminium, Ceramic, Foam, Hard_Plastic, Other_Plastic, Paper, Soft_Plastic, Steel, Wood. Training uses robot impact sounds, not general phone taps.

RealImpact external preprocessing: original 48 kHz force-deconvolved signals → polyphase16 kHz → identical peak-centered one-second crop → normalize that crop to 0.95 peak. The three classes are Ceramic, Plastic, Wood. Force-deconvolved impulse responses are not raw phone audio; do not apply the file unchanged to arbitrary captures and claim validated recognition.

Full-data C=.01 heads are selected with training-object CV, separately from the original outer-test winner. Every real recording is run through the combined TFLite and compared to the original reference scores/decisions. FlatBuffer CUSTOM operator codes are disallowed: builtin-only, no Flex/Select TF Ops. Compact `.parity.npz` real waveform/reference fixtures and model/fixture SHA-256 are retained for offline release smoke.

The generic 15,600-sample YAMNet encoder under `research/` is a different input contract. Do not substitute its embedding for this 16,000-sample mean embedding without parity checks. These are research-ready numerical exports, not commercially cleared or physically phone-tested models. YCB/RealImpact data provenance and checkpoint redistribution terms require review; no Flutter/runtime delegate, memory, energy or field accuracy has been measured.
