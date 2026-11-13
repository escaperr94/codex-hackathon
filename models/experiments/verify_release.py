"""Offline smoke of every finalized TFLite, using compact real-input fixtures.

Run: python -m experiments.verify_release (from models/).
Needs TensorFlow 2.19, NumPy and shipped final/ artifacts; no raw dataset downloads.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'artifacts/final'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path):
    return ROOT / str(path).replace('\\','/')


def fixture_for(path):
    """Read each producer's explicit real-input/reference fixture contract."""
    folder=path.parent
    if path.with_suffix('.parity.npz').exists():
        report=load(folder/'verification.json')
        entry=next(r for r in report['models'] if relative(r['file'])==path)
        assert entry['sha256']==sha(path)
        fixture=relative(entry['fixture']); assert sha(fixture)==entry['fixture_sha256']
        with np.load(fixture,allow_pickle=False) as data:
            return data['inputs'],data['reference'],None,'generic'
    metadata=path.with_suffix('.json')
    if folder.parent.name=='fruit':
        entry=load(metadata); assert entry['sha256']==sha(path)
        if path.name!='deployment-float32.tflite':
            return None,None,None,'evaluation_fold'
        info=load(folder/'verification-fixture.json')
        fixture=relative(info['file']); assert sha(fixture)==info['sha256']
        assert info['model_sha256']==sha(path)
        with np.load(fixture,allow_pickle=False) as data:
            return data['features'],data['reference_scores'],None,'fruit'
    entry=load(metadata); assert sha(path)==entry['sha256']
    info=entry['verification_fixture'];fixture=relative(info['path']);assert sha(fixture)==info['sha256']
    with np.load(fixture,allow_pickle=False) as data:
        return data['features'],data['scores'],data['class_index'],'classifier'


def exercise(path):
    flat=schema.Model.GetRootAsModel(path.read_bytes(),0)
    codes=[flat.OperatorCodes(i).BuiltinCode() for i in range(flat.OperatorCodesLength())]
    assert schema.BuiltinOperator.CUSTOM not in codes, f'Custom/Flex operation: {path}'
    interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=2)
    interpreter.allocate_tensors()
    inputs=interpreter.get_input_details();outputs=interpreter.get_output_details()
    assert len(inputs)==1 and inputs[0]['dtype']==np.float32
    shape=tuple(int(v) for v in inputs[0]['shape']);assert all(v>0 for v in shape)
    rows,reference,reference_index,kind=fixture_for(path)
    if rows is None:
        # Fold exports were verified on all their held-out rows by the producer.
        # This independent smoke checks allocation/finite output, not accuracy.
        rows=np.zeros((1,)+shape,np.float32)
    elif shape==tuple(rows.shape[1:]):
        pass
    elif len(shape)==2 and shape[0]==1 and tuple(rows.shape[1:])==(shape[1],):
        rows=rows[:,None,:]
    else:
        raise ValueError(f'{path}: fixture shape {rows.shape} does not match input {shape}')
    predictions=[];indices=[]
    signatures=interpreter.get_signature_list()
    for row in rows:
        assert tuple(row.shape)==shape and np.isfinite(row).all()
        interpreter.set_tensor(inputs[0]['index'],row.astype(np.float32))
        interpreter.invoke()
        values=[interpreter.get_tensor(out['index']) for out in outputs]
        assert all(np.isfinite(v).all() for v in values)
        float_values=[v for v in values if v.dtype==np.float32]
        assert len(float_values)==1
        predictions.append(float_values[0].reshape(-1))
        integer_values=[v for v in values if v.dtype.kind in 'iu']
        if integer_values:
            assert len(integer_values)==1
            indices.append(int(integer_values[0].reshape(-1)[0]))
    actual=np.asarray(predictions)
    max_error=None
    if reference is not None:
        expected=np.asarray(reference).reshape(actual.shape)
        # Waveform encoder/linear material models include accumulated encoder error.
        waveform=len(shape)==1
        atol,rtol=(.003,.005) if waveform else (.0003,.0003)
        np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol,err_msg=str(path))
        max_error=float(np.max(np.abs(actual-expected)))
        if reference_index is not None:
            assert np.array_equal(indices,np.asarray(reference_index)),path
        elif actual.shape[1]>1:
            assert np.array_equal(actual.argmax(1),expected.argmax(1)),path
    return {'file':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(path),'bytes':path.stat().st_size,
            'input_shape':list(shape),'output_shapes':[o['shape'].tolist() for o in outputs],
            'builtin_operator_codes':codes,'signature_names':list(signatures),
            'fixture_kind':kind,'real_fixture_rows':len(rows) if reference is not None else 0,
            'max_absolute_reference_error':max_error,'status':'passed'}


def main():
    paths=sorted(FINAL.rglob('*.tflite'))
    if not paths:
        raise RuntimeError('No finalized TFLites found')
    reports=[]
    for path in paths:
        reports.append(exercise(path));print('PASS',path.relative_to(FINAL),flush=True)
    result={'status':'passed','script_sha256':sha(__file__),'tensorflow':tf.__version__,
            'python':platform.python_version(),'platform':platform.platform(),'models':reports,
            'physical_phone':'not measured','network_or_raw_datasets_required':False,
            'limits':['Feature heads still require exact external DSP/encoder.',
                      'Numerical parity does not establish task validity, calibration, license permission or phone runtime.']}
    (FINAL/'release-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'passed_models':len(reports),'real_fixture_models':sum(r['real_fixture_rows']>0 for r in reports)}))


if __name__=='__main__':
    main()
