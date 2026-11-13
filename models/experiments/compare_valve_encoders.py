"""Compare fixed frozen representations on the exact valve split; no tuning."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
import json
import platform
import hashlib
from pathlib import Path
import importlib.metadata
import numpy as np
import soundfile as sf
from scipy.signal import stft, resample_poly
from sklearn.covariance import LedoitWolf
from experiments.run_valve_benchmark import ROOT, ART, REPORT, discover, checksum, manifest_fingerprint, metrics
sys.path.insert(0, str(ROOT))


def spectral(rows):
    output = []
    for row in rows:
        audio, rate = sf.read(row['path'], dtype='float32')
        assert audio.ndim == 1
        if rate != 16000:
            divisor = np.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        _, _, value = stft(audio, fs=16000, nperseg=512, noverlap=256, boundary=None, padded=False)
        logpower = np.log(np.maximum(np.abs(value)**2, 1e-10))
        band_frames = np.stack([part.mean(axis=0) for part in np.array_split(logpower, 64, axis=0)])
        output.append(np.concatenate([band_frames.mean(axis=1), band_frames.std(axis=1)]))
    return np.asarray(output, dtype=np.float32)


def run():
    rows = discover()
    fingerprint = manifest_fingerprint(rows)
    fit = np.array([r['split']=='train' and r['domain']=='source' and r['label']=='normal' for r in rows])
    test = np.array([r['split']=='test' for r in rows])
    label = np.array([int(r['label']=='anomaly') for r in rows])
    source = np.array([r['domain']=='source' for r in rows])
    target = np.array([r['domain']=='target' for r in rows])
    assert fit.sum() == 990 and test.sum() == 200
    all_anomaly = test & (label == 1)
    masks = {
        'pooled_all_test': test,
        'official_style_source_normals_vs_all_test_anomalies': (test & source & (label==0)) | all_anomaly,
        'official_style_target_normals_vs_all_test_anomalies': (test & target & (label==0)) | all_anomaly,
        'same_domain_source_only': test & source,
        'same_domain_target_only': test & target,
    }
    results = {}
    definitions = {
        'spectral_statistics': '128D: mean/std over time of 64 uniformly grouped log-power STFT bands; 16kHz; window512 hop256, log floor1e-10',
        'efficientat_mn04_as': '384D frozen official mn04_as; upstream 32kHz librosa waveform and 128mel frontend; strict checkpoint/source load via thud.efficientat',
    }
    for encoder in definitions:
        cache = ART / f'valve_{encoder}_features.npz'
        checkpoint_hash = checksum(ROOT/'artifacts/mn04_as_mAP_432.pt') if encoder == 'efficientat_mn04_as' else hashlib.sha256(definitions[encoder].encode()).hexdigest()
        key = hashlib.sha256(f'{encoder}\n{definitions[encoder]}\n{checkpoint_hash}\n{fingerprint}'.encode()).hexdigest()
        try:
            matrix = None
            if cache.exists():
                with np.load(cache) as saved:
                    if str(saved.get('fingerprint','')) == key:
                        matrix = saved['features']
            if matrix is None:
                if encoder == 'spectral_statistics':
                    matrix = spectral(rows)
                else:
                    import torch
                    torch.set_num_threads(2)
                    torch.set_num_interop_threads(2)
                    from thud.efficientat import load_efficientat, efficientat_wav_embedding
                    bundle = load_efficientat(ROOT/'artifacts/mn04_as_mAP_432.pt', ROOT/'artifacts/efficientat-source')
                    values = []
                    for index, row in enumerate(rows,1):
                        values.append(efficientat_wav_embedding(bundle, row['path']))
                        if index%100==0:
                            print(f'{encoder}: {index}/{len(rows)}', flush=True)
                    matrix = np.stack(values)
                assert matrix.shape[0] == len(rows) and np.isfinite(matrix).all()
                np.savez_compressed(cache, features=matrix, fingerprint=key, ordered_manifest_fingerprint=fingerprint, encoder=encoder)
            detector = LedoitWolf().fit(matrix[fit])
            scores = detector.mahalanobis(matrix)
            results[encoder] = {'status':'complete', 'definition':definitions[encoder], 'dimensions':int(matrix.shape[1]),
                                'fit_clips':990, 'target_training_normals_used':0, 'covariance_shrinkage':float(detector.shrinkage_),
                                'cache_fingerprint':key, 'checkpoint_or_definition_sha256':checkpoint_hash,
                                'metrics':{name:metrics(label[mask], scores[mask]) for name,mask in masks.items()}}
            np.savez_compressed(ART/f'valve_{encoder}_scores.npz', filenames=np.array([r['filename'] for r in rows]),
                                scores=scores, labels=label, fit_mask=fit, test_mask=test, ordered_manifest_fingerprint=fingerprint)
        except Exception as error:
            import traceback
            results[encoder] = {'status':'failed', 'error':f'{type(error).__name__}: {error}', 'traceback':traceback.format_exc()}
            print(results[encoder]['traceback'], flush=True)
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    report['comparisons'] = results
    report['ordered_manifest_fingerprint'] = fingerprint
    report['runtime'] = {'python':sys.version,'platform':platform.platform(),'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','tensorflow','tensorflow-hub','torch','torchaudio','librosa','soundfile']}}
    report['commands'] = [
        'mkdir -p artifacts/dcase && curl -L --fail --retry 2 -o artifacts/dcase/dev_valve.zip https://zenodo.org/api/records/10902294/files/dev_valve.zip/content',
        'python -m experiments.run_valve_benchmark',
        'python -m experiments.compare_valve_encoders',
    ]
    report['method']['target_training_normals_used'] = 0
    report['method']['ordered_manifest_fingerprint'] = fingerprint
    report['method']['checkpoint_saved_model_pb_sha256'] = checksum(ROOT/'artifacts/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e/saved_model.pb')
    report['method']['checkpoint_variables_sha256'] = checksum(ROOT/'artifacts/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e/variables/variables.data-00000-of-00001')
    report['caveats'].extend([
        'All encoders use precisely the same 990 source-domain normal training clips and 200 labeled test clips. Ten target-domain normal training clips are unused; no test-label tuning.',
        'The simple spectral comparator and EfficientAT were added as predefined comparisons after the initial YAMNet metrics, without hyperparameter optimization or best-result selection.',
        'This is a dataset benchmark, not a verified official challenge submission or a comparison against the official aggregate leaderboard.',
        'Dataset source/target differences include valve opening/closing patterns. Attributes are retained in filenames but not used for detector fitting.',
        'Frozen encoder code licensing and checkpoint redistribution rights must be reviewed separately from the dataset CC BY-NC-SA 4.0 license.',
    ])
    REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results, indent=2),flush=True)


if __name__=='__main__':
    run()
