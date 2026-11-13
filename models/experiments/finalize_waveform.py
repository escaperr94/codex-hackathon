"""Combine verified material heads with the exact one-second YAMNet waveform encoder.

Run after finalize_classifiers: python -m experiments.finalize_waveform
Capture decoding/resampling/peak-centered crop remain outside this model.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.signal import resample_poly
import tensorflow as tf
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
OUT=ART/'final/waveform'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def waveforms(task, rows):
    from experiments.material_benchmark import waveform, DATA
    if task=='ycb':
        manifest = {row['local_path']: row for row in json.loads((DATA/'manifest.json').read_text())}
        for row in rows:
            assert sha(DATA/row['path']) == manifest[row['path']]['sha256']
        return np.stack([waveform(DATA/row['path'])[0] for row in rows])
    signals={}
    result=[]
    for row in rows:
        obj=row['object']
        if obj not in signals:
            signals[obj]=np.load(DATA/'realimpact-prefix'/obj/'audio-first30.npy')
        signal=resample_poly(signals[obj][row['source_row']],1,3).astype(np.float32)
        start=round(row['crop_start_seconds']*16000)
        x=signal[start:start+16000].copy()
        assert x.shape==(16000,) and np.max(np.abs(x))>0
        x*=.95/np.max(np.abs(x))
        result.append(x)
    return np.stack(result)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    encoder=tf.saved_model.load(str(ART/'tfhub/9616fd04ec2360621642ef9455b84f4b668e219e'))
    report={'script_sha256':sha(__file__),'models':[], 'physical_phone':'not measured',
            'scope':'real-data trained waveform-to-material margin models, not validated universal material detectors'}
    for task in ('ycb','realimpact'):
        source=ART/('material-benchmark.json' if task=='ycb' else 'material-realimpact-benchmark.json')
        original=json.loads(source.read_text())
        rows=original['results']['yamnet']['recording_predictions']
        waves=waveforms(task,rows)
        head_path=ART/f'final/classifiers/{task}_material_yamnet.joblib'
        head=joblib.load(head_path)
        scaler,linear=head.steps[0][1],head.steps[-1][1]
        assert linear.coef_.shape[1]==1024
        with np.load(ART/('material-features.npz' if task=='ycb' else 'material-realimpact-features.npz')) as cache:
            features=cache['yamnet']
        assert features.shape==(len(waves),1024)
        reference=head.decision_function(features)

        class Model(tf.Module):
            def __init__(self):
                super().__init__()
                self.encoder=encoder

            @tf.function(input_signature=[tf.TensorSpec([16000],tf.float32,name='waveform')])
            def infer(self,waveform):
                embedding=tf.reduce_mean(self.encoder(waveform)[1],axis=0,keepdims=True)
                scaled=(embedding-tf.constant(scaler.mean_,tf.float32))/tf.constant(scaler.scale_,tf.float32)
                scores=tf.matmul(scaled,tf.constant(linear.coef_.T,tf.float32))+tf.constant(linear.intercept_,tf.float32)
                return {'scores':scores,'class_index':tf.argmax(scores,axis=1,output_type=tf.int32)}

        module=Model()
        converter=tf.lite.TFLiteConverter.from_concrete_functions([module.infer.get_concrete_function()],module)
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
        path=OUT/f'{task}-material-waveform-f32.tflite'
        path.write_bytes(converter.convert())
        from tensorflow.lite.python import schema_py_generated as schema
        flat=schema.Model.GetRootAsModel(path.read_bytes(),0)
        assert all(flat.OperatorCodes(i).BuiltinCode()!=schema.BuiltinOperator.CUSTOM for i in range(flat.OperatorCodesLength()))
        interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=2)
        interpreter.allocate_tensors()
        signature=interpreter.get_signature_runner()
        scores=[]
        for waveform in waves:
            assert waveform.shape==(16000,) and np.isfinite(waveform).all() and np.max(np.abs(waveform))<=1
            value=signature(waveform=waveform.astype(np.float32))
            assert np.isfinite(value['scores']).all()
            assert int(value['class_index'][0])==int(value['scores'].argmax(1)[0])
            scores.append(value['scores'][0])
        scores=np.asarray(scores)
        np.testing.assert_allclose(scores,reference,atol=.003,rtol=.005)
        changes=int(np.sum(scores.argmax(1)!=reference.argmax(1)))
        assert changes==0,(task,changes)
        fixture=path.with_suffix('.parity.npz')
        np.savez_compressed(fixture,inputs=waves[:6],reference=reference[:6],classes=linear.classes_)
        times=[]
        for i in range(120):
            start=time.perf_counter(); signature(waveform=waves[i%len(waves)].astype(np.float32)); elapsed=(time.perf_counter()-start)*1000
            if i>=20: times.append(elapsed)
        entry={'file':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size,
               'fixture':str(fixture.relative_to(ROOT)),'fixture_sha256':sha(fixture),'class_order':linear.classes_.tolist(),
               'input_shape':[16000],'dtype':'float32','sample_rate':16000,
               'output_scores':'uncalibrated LinearSVC one-vs-rest margins [1,K]; larger wins, not probabilities',
               'frontend_included':'YAMNet DSP+encoder+scaler+linear head',
               'external_preprocessing':original['protocol']['input'] if task=='ycb' else original['preprocessing'],
               'real_recordings':len(waves),'zero_decision_changes':changes==0,
               'max_absolute_error':float(np.max(np.abs(scores-reference))),
               'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
               'timing_scope':'desktop2thread waveform-to-score; excludes decode/resample/crop/capture',
               'head_sha256':sha(head_path),'source_report_sha256':sha(source),
               'original_oof_metrics':original['results']['yamnet']['object_metrics'],
               'release_status':'research-only; dataset/checkpoint license review required; no phone accuracy or runtime measured'}
        report['models'].append(entry)
        print(task,'verified',len(waves),'real recordings',flush=True)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
