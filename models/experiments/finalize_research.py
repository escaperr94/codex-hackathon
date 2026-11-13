"""Finalize existing research encoders/scorers; no test-driven model selection.

Run from models/: python -m experiments.finalize_research
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[key] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
OUT = ART / 'final/research'
CHECKPOINT = ART / 'tfhub/9616fd04ec2360621642ef9455b84f4b668e219e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(path, x, reference, *, decisions=False):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    assert len(inputs) == len(outputs) == 1
    assert inputs[0]['dtype'] == outputs[0]['dtype'] == np.float32
    assert tuple(inputs[0]['shape']) == (1, x.shape[1])
    from tensorflow.lite.python import schema_py_generated as schema
    data = Path(path).read_bytes()
    flat = schema.Model.GetRootAsModel(data, 0)
    fixture = path.with_suffix('.parity.npz')
    np.savez_compressed(fixture, inputs=np.asarray(x[:8], np.float32), reference=np.asarray(reference[:8], np.float64))
    assert all(flat.OperatorCodes(i).BuiltinCode() != schema.BuiltinOperator.CUSTOM for i in range(flat.OperatorCodesLength()))
    result = []
    for row in x:
        interpreter.set_tensor(inputs[0]['index'], np.asarray(row[None], np.float32))
        interpreter.invoke()
        result.append(interpreter.get_tensor(outputs[0]['index'])[0])
    result = np.asarray(result)
    assert np.isfinite(result).all() and result.shape == np.asarray(reference).shape
    delta = np.abs(result - reference)
    changes = int(np.sum(result.argmax(1) != np.asarray(reference).argmax(1))) if decisions else None
    if decisions:
        assert changes == 0, (path, changes)
    np.testing.assert_allclose(result, reference, rtol=3e-4, atol=3e-4)
    # The public boundary is finite feature rows, not arbitrary unbounded values.
    for row in (np.zeros(x.shape[1], np.float32), np.mean(x, axis=0).astype(np.float32)):
        interpreter.set_tensor(inputs[0]['index'], row[None])
        interpreter.invoke()
        assert np.isfinite(interpreter.get_tensor(outputs[0]['index'])).all()
    for _ in range(20):
        interpreter.invoke()
    timings = []
    for _ in range(100):
        start = time.perf_counter(); interpreter.invoke(); timings.append((time.perf_counter()-start)*1000)
    return {'file': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'fixture': str(fixture.relative_to(ROOT)), 'fixture_sha256': sha(fixture),
            'input_shape': inputs[0]['shape'].tolist(), 'output_shape': outputs[0]['shape'].tolist(),
            'dtype': 'float32', 'operators': 'TFLITE_BUILTINS only; FlatBuffer CUSTOM opcodes rejected',
            'real_feature_rows': len(x), 'max_absolute_error': float(delta.max()),
            'max_scaled_error': float(np.max(delta / np.maximum(1., np.abs(reference)))),
            'decision_changes': changes, 'desktop_cpu_head_only_p50_ms': float(np.median(timings)),
            'desktop_cpu_head_only_p95_ms': float(np.quantile(timings, .95)),
            'warmups': 20, 'timed_inferences': 100}, result


def export(module, path):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.infer.get_concrete_function()], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    path.write_bytes(converter.convert())


def gaussian(x, train, name):
    fit = LedoitWolf().fit(x[train].astype(np.float64))
    # Cholesky yields sum-of-squares scoring, preserving nonnegative distances.
    factor = np.linalg.cholesky(fit.precision_)
    dimensions = x.shape[1]

    class Score(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, dimensions], tf.float32, name='features')])
        def infer(self, features):
            centered = features - tf.constant(fit.location_, tf.float32)
            projected = tf.matmul(centered, tf.constant(factor, tf.float32))
            return {'squared_mahalanobis': tf.reduce_sum(tf.square(projected), axis=1, keepdims=True)}

    path = OUT / (name + '.tflite')
    export(Score(), path)
    ref = fit.mahalanobis(x.astype(np.float64))[:, None]
    report, score = verify(path, x, ref)
    report.update({'output_semantics': 'nonnegative squared Mahalanobis distance; larger is more anomalous; not probability',
                   'fit_rows': int(train.sum()), 'fit': 'LedoitWolf float64 on normal training rows only',
                   'frontend_included': False})
    assert np.min(score) >= 0
    return report, score[:, 0]


def cosine(x, train, name):
    reference = x[train].astype(np.float64)
    norm = np.linalg.norm(reference, axis=1, keepdims=True)
    assert np.all(norm > 0)
    normalized = (reference / norm).astype(np.float32)
    dimensions = x.shape[1]

    class Score(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, dimensions], tf.float32, name='efficientat_embedding')])
        def infer(self, features):
            unit = tf.math.divide_no_nan(features, tf.norm(features, axis=1, keepdims=True))
            similarity = tf.matmul(unit, tf.constant(normalized.T))
            distance = tf.clip_by_value(1. - tf.reduce_max(similarity, axis=1, keepdims=True), 0., 2.)
            return {'cosine_distance': distance}

    path = OUT / (name + '.tflite')
    export(Score(), path)
    reference_score = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute').fit(reference).kneighbors(x.astype(np.float64))[0]
    report, score = verify(path, x, reference_score)
    report.update({'output_semantics': 'nearest training-normal cosine distance in [0,2]; not fault probability',
                   'fit_rows': int(train.sum()), 'frontend_included': False,
                   'frontend': 'official EfficientAT mn04_as 32kHz frontend + frozen 384D encoder; experiments.multimachine_benchmark.extract',
                   'zero_vector': 'cosine distance 1, matching sklearn cosine convention'})
    return report, score[:, 0]


def waveform_encoder():
    from thud.pretrained import load_yamnet
    from thud.tflite import export_yamnet_tflite, run_tflite, INPUT_SAMPLES
    encoder = load_yamnet(CHECKPOINT)
    path = OUT / 'yamnet-waveform-embedding-f32.tflite'
    if path.exists():
        path.unlink()  # This finalizer owns this reproducible output only.
    export_yamnet_tflite(CHECKPOINT, path)
    sample = next((ART / 'sound-of-water/audios').glob('*.wav'))
    x, rate = sf.read(sample, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if rate != 16000:
        from scipy.signal import resample_poly
        divisor = np.gcd(rate, 16000); x = resample_poly(x, 16000 // divisor, rate // divisor)
    windows = [np.pad(x[start:start+INPUT_SAMPLES], (0, max(0, INPUT_SAMPLES-len(x[start:start+INPUT_SAMPLES]))))
               for start in (0, max(0, len(x)//2-INPUT_SAMPLES//2), max(0, len(x)-INPUT_SAMPLES))]
    errors = []
    for window in windows:
        reference = np.asarray(encoder(window.astype(np.float32))[1]).mean(axis=0)
        actual = run_tflite(path, window)
        np.testing.assert_allclose(actual, reference, rtol=3e-3, atol=3e-4)
        errors.append(float(np.max(np.abs(actual-reference))))
    fixture = path.with_suffix('.parity.npz')
    np.savez_compressed(fixture, inputs=np.asarray(windows, np.float32),
                        reference=np.asarray([np.asarray(encoder(window.astype(np.float32))[1]).mean(axis=0) for window in windows]))
    from tensorflow.lite.python import schema_py_generated as schema
    flat = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    assert all(flat.OperatorCodes(i).BuiltinCode() != schema.BuiltinOperator.CUSTOM for i in range(flat.OperatorCodesLength()))
    return {'file': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'input': 'float32 [15600], mono16kHz PCM in [-1,1]; capture/resampling outside model',
            'output': 'float32 [1024] mean embedding, not event scores or physical estimate',
            'frontend_included': True, 'operators': 'TFLITE_BUILTINS only; CUSTOM opcodes rejected',
            'real_audio': str(sample.relative_to(ROOT)), 'audio_sha256': sha(sample),
            'real_windows': len(windows), 'max_absolute_error': max(errors),
            'fixture': str(fixture.relative_to(ROOT)), 'fixture_sha256': sha(fixture),
            'scope': '975ms fixed YAMNet encoder; not interchangeable with whole-recording temporal pooling heads'}


def pouring_head():
    source = load(ART/'water-ablation.json')
    name = source['selected_configuration_train_only']
    assert name == 'yamnet_three_bins', 'Review changed training-only selection before exporting'
    with np.load(ART/'water-ablation-temporal.npz') as cache:
        x = cache['three_bins']
    with np.load(ART/'water-ablation-heads.npz') as arrays:
        prefix = name + '__'
        mean, scale, weights, bias = [arrays[prefix+k] for k in ('scaler_mean','scaler_scale','coefficients','intercept')]
        classes = arrays[prefix+'classes'].tolist()
    from thud.pretrained import load_yamnet
    items = source['protocol']['item_ids_in_cache_order']
    sample = ART / 'sound-of-water/audios' / (items[0] + '.wav')
    waveform, rate = sf.read(sample, dtype='float32', always_2d=True)
    waveform = waveform.mean(axis=1)
    from scipy.signal import resample_poly
    divisor = np.gcd(rate, 16000)
    waveform = resample_poly(waveform, 16000//divisor, rate//divisor).astype(np.float32) if rate != 16000 else waveform
    embeddings = np.asarray(load_yamnet(CHECKPOINT)(waveform)[1])
    reextracted = np.concatenate([part.mean(axis=0) for part in np.array_split(embeddings, 3)])
    np.testing.assert_allclose(reextracted, x[0], rtol=1e-5, atol=1e-6)
    logits = ((x-mean)/scale)@weights.T+bias
    from scipy.special import softmax
    reference = softmax(logits, axis=1)

    class Head(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 3072], tf.float32, name='yamnet_three_temporal_bins')])
        def infer(self, features):
            values = (features-tf.constant(mean,tf.float32))/tf.constant(scale,tf.float32)
            return {'probabilities': tf.nn.softmax(tf.matmul(values,tf.constant(weights.T,tf.float32))+tf.constant(bias,tf.float32))}

    path=OUT/'pouring-shape-yamnet-three-bins.tflite'
    export(Head(),path)
    report,_=verify(path,x,reference,decisions=True)
    report.update({'class_order':classes,'output_semantics':'container shape probabilities, not water volume/fill',
                   'frontend_included':False,'frontend':'whole-recording YAMNet frame embeddings pooled into three chronological bins; experiments.water_ablation.main lines for temporal extraction',
                   'selection':'existing training-container-only validation; retained original training-only fit, no test refit',
                   'held_out_container_metrics':source['results'][name]['held_out_container_metrics'],
                   'source_report_sha256':sha(ART/'water-ablation.json'), 'release_status':'research comparator only; poor heldout performance'})
    report['real_audio_frontend_check'] = {'audio':str(sample.relative_to(ROOT)), 'sha256':sha(sample),
                                          'max_feature_error':float(np.max(np.abs(reextracted-x[0])))}
    return report


def machine_frontend_smoke(manifest, cache_path, bundle):
    """Re-extract one actual waveform per section, including 2026 near-channel policy."""
    import librosa
    import torch
    from thud.efficientat import efficientat_wav_embedding
    row = manifest[0]
    with zipfile.ZipFile(ROOT / row['archive']) as archive:
        content = archive.read(row['member'])
    assert hashlib.sha256(content).hexdigest() == row['wav_sha256']
    audio, rate = sf.read(io.BytesIO(content), dtype='float32', always_2d=True)
    if row['corpus'] == 'DCASE2026dev':
        assert audio.shape[1] == 2
        audio = audio[:, :1]
    temporary = OUT / 'frontend-smoke.wav'
    try:
        if row['corpus'] == 'DCASE2026dev':
            sf.write(temporary, audio[:, 0], rate, subtype='FLOAT')
        else:
            temporary.write_bytes(content)
        vector = efficientat_wav_embedding(bundle, temporary)
        waveform, _ = librosa.load(temporary, sr=32000, mono=True)
        with torch.inference_mode():
            mel = bundle.frontend(torch.from_numpy(waveform)[None, :])[0].numpy()
        summary = np.r_[mel.mean(axis=1), mel.std(axis=1)]
        with np.load(cache_path) as cached:
            np.testing.assert_allclose(vector, cached['efficientat'][0], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(summary, cached['logmel'][0], rtol=1e-5, atol=1e-5)
        return {'member':row['member'], 'wav_sha256':row['wav_sha256'], 'real_waveform_reextraction':'matched both cached feature representations'}
    finally:
        temporary.unlink(missing_ok=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    report={'script_sha256':sha(__file__),'models':[], 'physical_phone':'not measured',
            'limits':['No universal tile/fill/seal/moisture detector trained or claimed.',
                      'Feature scorers require the exact external frontend/encoder; only YAMNet waveform encoder includes DSP.',
                      'Previously inspected tests; fixed methods, no test-driven selection.',
                      'Original dataset and checkpoint licenses still apply; not a production redistribution grant.']}
    report['models'].append(waveform_encoder())
    report['models'].append(pouring_head())
    import torch
    from thud.efficientat import load_efficientat
    torch.set_num_threads(2)
    bundle = load_efficientat(ART/'mn04_as_mAP_432.pt', ART/'efficientat-source')
    for source_name in ('multimachine-benchmark.json','multimachine-extra-benchmark.json'):
        source=load(ART/source_name)
        for key, section in source['sections'].items():
            manifest=load(ROOT/section['manifest']); train=np.array([r['split']=='train' for r in manifest])
            assert all(r['label']==0 for r in manifest if r['split']=='train')
            cache_path=ART/f"multimachine-{key.replace('/','-')}.npz"
            smoke = machine_frontend_smoke(manifest, cache_path, bundle)
            with np.load(cache_path) as cache:
                for feature, method in [('efficientat','cosine1NN'),('logmel','Gaussian_LedoitWolf')]:
                    x=cache[feature]
                    name=key.replace('/','-')+'-'+method
                    entry,scores=(cosine if feature=='efficientat' else gaussian)(x,train,name)
                    raw=load(ROOT/section['raw_scores']); y=np.array([r['label'] for r in raw])
                    reference=np.array([r[method] for r in raw]); actual=scores[~train]
                    np.testing.assert_allclose(actual,reference,rtol=3e-4,atol=3e-4)
                    entry.update({'dataset':key,'test_auc':float(roc_auc_score(y,actual)),
                                  'reference_test_auc':float(roc_auc_score(y,reference)),
                                  'cache_sha256':sha(cache_path),'manifest_sha256':sha(ROOT/section['manifest'])})
                    entry['real_audio_frontend_check'] = smoke
                    if feature=='logmel': entry['frontend']='official EfficientAT frontend logmel mean/std 256D; experiments.multimachine_benchmark.extract'
                    report['models'].append(entry)
                    print('verified',name,flush=True)
    for tag in ('', '-stgram'):
        src=load(ART/f'machine-task{tag if tag else "-benchmark"}.json')
        with np.load(ART/f'machine-task{tag}-embeddings.npz') as cache:
            x=cache['embeddings']
        train=np.array([r['split']=='train' for r in src['split_records']])
        entry,scores=gaussian(x,train,'valve2024-'+('stgram' if tag else 'jvachier')+'-mahalanobis')
        entry['frontend']=src['frontend'];entry['source_report_sha256']=sha(ART/f'machine-task{tag if tag else "-benchmark"}.json')
        report['models'].append(entry)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'verified_models':len(report['models']),'report':str(OUT/'verification.json')},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
