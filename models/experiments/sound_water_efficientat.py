"""Append a frozen EfficientAT comparator using the completed pilot's exact splits."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '3'
import collections
import contextlib
import hashlib
import io
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('artifacts/sound-of-water').resolve()
REPORT = Path('artifacts/sound-water-benchmark.json').resolve()
SOURCE = Path('artifacts/efficientat-source').resolve()
CHECKPOINT = Path('artifacts/mn04_as_mAP_432.pt').resolve()
CACHE = ROOT / 'efficientat_embeddings_384.npz'
SCRIPT = Path(__file__).resolve()
SEED = 42
CS = [0.01, 0.1, 1.0, 10.0]
KEY = 'efficientat_mn04_mean_384'
started = time.perf_counter()
report = json.loads(REPORT.read_text(encoding='utf-8'))
classes = report['target']['classes']
items = [r['item_id'] for r in report['audio_file_metadata']]
original_recordings = report['split']['train']['item_ids'] + report['split']['test']['item_ids']
assert len(items) == len(set(items)) == len(original_recordings)
assert set(items) == set(original_recordings)
metadata = {}
import csv
for path in sorted((ROOT / 'splits').glob('*.csv')):
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            metadata[row['item_id']] = row
rows = [metadata[item] for item in items]
y = np.array([r['shape'] for r in rows])
index_by_item = {item: i for i, item in enumerate(items)}
train = np.array([index_by_item[item] for item in report['split']['train']['item_ids']])
test = np.array([index_by_item[item] for item in report['split']['test']['item_ids']])
assert not set(report['split']['train']['container_ids']) & set(report['split']['test']['container_ids'])
for name in ('train', 'test'):
    split = report['split'][name]
    assert {metadata[item]['container_id'] for item in split['item_ids']} == set(split['container_ids'])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(truth, prediction):
    return {'n': len(truth), 'accuracy': float(accuracy_score(truth, prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(truth, prediction)),
            'macro_f1': float(f1_score(truth, prediction, labels=classes, average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(truth, prediction, labels=classes).tolist(),
            'class_order': classes}


def aggregate(indices, probabilities, head_classes):
    grouped = collections.defaultdict(list)
    truth = {}
    for i, p in zip(indices, probabilities):
        group = rows[i]['container_id']
        grouped[group].append(p)
        if group in truth:
            assert truth[group] == rows[i]['shape']
        truth[group] = rows[i]['shape']
    records = []
    for group in sorted(grouped):
        mean = np.mean(grouped[group], axis=0)
        records.append({'container_id': group, 'truth': truth[group],
                        'prediction': str(head_classes[np.argmax(mean)]),
                        'mean_probabilities': {str(c): float(v) for c, v in zip(head_classes, mean)},
                        'recordings': len(grouped[group])})
    return records


def container_metrics(indices, probabilities, head_classes):
    records = aggregate(indices, probabilities, head_classes)
    return metrics([r['truth'] for r in records], [r['prediction'] for r in records])


try:
    import torch
    import torchaudio
    import torchvision
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    # The unchanged upstream helpers/utils.py opens metadata relative to cwd at import.
    sys.path.insert(0, str(SOURCE))
    previous_cwd = Path.cwd()
    try:
        os.chdir(SOURCE)
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
    finally:
        os.chdir(previous_cwd)
    checkpoint_hash = sha256(CHECKPOINT)
    manifest = [(item, sha256(ROOT / 'audios' / f'{item}.wav')) for item in items]
    fingerprint = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    assert fingerprint == report['reproducibility']['manifest_fingerprint']
    timing = {'clock': 'time.perf_counter', 'hardware': 'desktop CPU only',
              'notes': 'Single serial pass over full clips; no dedicated warm-up. Per-file forward timing excludes loading/resampling/frontend and includes the initial cold invocation. Not comparable to warmed device benchmarks.'}
    if CACHE.exists():
        saved = np.load(CACHE, allow_pickle=False)
        assert str(saved['manifest_fingerprint']) == fingerprint
        assert str(saved['checkpoint_sha256']) == checkpoint_hash
        assert saved['item_ids'].tolist() == items
        x = saved['embeddings']
        timing.update(json.loads(str(saved['extraction_timing_json'])))
        timing['feature_cache_reused_this_run'] = True
    else:
        load_started = time.perf_counter()
        # Explicit local strict load; no implicit network or fallback weights.
        with contextlib.redirect_stdout(io.StringIO()):
            model = get_model(width_mult=0.4, num_classes=527, pretrained_name=None)
            frontend = AugmentMelSTFT()
        state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        frontend.eval().requires_grad_(False)
        timing['model_and_frontend_load_seconds'] = time.perf_counter() - load_started
        features, file_timings = [], []
        extraction_started = time.perf_counter()
        with torch.inference_mode():
            for position, item in enumerate(items):
                file_started = time.perf_counter()
                waveform, rate = sf.read(ROOT / 'audios' / f'{item}.wav', dtype='float32', always_2d=True)
                mono = waveform.mean(axis=1)
                assert mono.size > 0 and np.isfinite(mono).all()
                divisor = math.gcd(rate, 32000)
                mono32 = resample_poly(mono, 32000 // divisor, rate // divisor).astype(np.float32) if rate != 32000 else mono
                tensor = torch.from_numpy(mono32).unsqueeze(0)
                frontend_started = time.perf_counter()
                mel = frontend(tensor).unsqueeze(1)
                frontend_seconds = time.perf_counter() - frontend_started
                forward_started = time.perf_counter()
                logits, embedding = model(mel)
                embedding = embedding.cpu().numpy()
                forward_seconds = time.perf_counter() - forward_started
                assert embedding.shape == (1, 384) and np.isfinite(embedding).all()
                features.append(embedding[0])
                file_timings.append({'item_id': item, 'audio_seconds': len(mono) / rate,
                                     'mel_frames': int(mel.shape[-1]), 'frontend_seconds': frontend_seconds,
                                     'encoder_forward_seconds': forward_seconds,
                                     'read_resample_frontend_encoder_seconds': time.perf_counter() - file_started})
                if position % 20 == 0:
                    print(json.dumps({'phase': 'efficientat', 'completed': position + 1, 'total': len(items)}), flush=True)
        x = np.asarray(features, dtype=np.float32)
        timing['feature_extraction_wall_seconds'] = time.perf_counter() - extraction_started
        timing['per_file'] = file_timings
        for field in ('frontend_seconds', 'encoder_forward_seconds', 'read_resample_frontend_encoder_seconds'):
            values = np.array([r[field] for r in file_timings])
            timing[field + '_summary'] = {'sum': float(values.sum()), 'mean': float(values.mean()),
                                         'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95))}
        timing['encoder_realtime_factor_sum_over_audio_seconds'] = sum(r['encoder_forward_seconds'] for r in file_timings) / sum(r['audio_seconds'] for r in file_timings)
        np.savez_compressed(CACHE, embeddings=x, item_ids=np.array(items),
                            manifest_fingerprint=fingerprint, checkpoint_sha256=checkpoint_hash,
                            extraction_timing_json=json.dumps(timing))
        timing['feature_cache_reused_this_run'] = False
    assert x.shape == (len(items), 384) and np.isfinite(x).all()
    fit_started = time.perf_counter()
    search = []
    for c in CS:
        scores = []
        for fold in report['split']['inner_validation']:
            tr = np.array([index_by_item[item] for item in fold['train']['item_ids']])
            va = np.array([index_by_item[item] for item in fold['validation']['item_ids']])
            assert not {rows[i]['container_id'] for i in tr} & {rows[i]['container_id'] for i in va}
            assert not (set(tr) | set(va)) & set(test)
            head = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=3000, random_state=SEED))
            head.fit(x[tr], y[tr])
            scores.append(container_metrics(va, head.predict_proba(x[va]), head.classes_)['macro_f1'])
        search.append({'C': c, 'fold_container_macro_f1': scores, 'mean_container_macro_f1': float(np.mean(scores))})
    best = max(search, key=lambda r: r['mean_container_macro_f1'])
    head = make_pipeline(StandardScaler(), LogisticRegression(C=best['C'], class_weight='balanced', max_iter=3000, random_state=SEED))
    head.fit(x[train], y[train])
    probabilities = head.predict_proba(x[test])
    predictions = head.classes_[np.argmax(probabilities, axis=1)]
    timing['head_cv_fit_and_test_predict_seconds'] = time.perf_counter() - fit_started
    timing['comparator_experiment_wall_seconds'] = time.perf_counter() - started
    report['results'][KEY] = {'selected_C': best['C'], 'hyperparameter_search_train_only': search,
                              'held_out_recording_metrics': metrics(y[test], predictions),
                              'held_out_container_metrics': container_metrics(test, probabilities, head.classes_),
                              'container_predictions': aggregate(test, probabilities, head.classes_),
                              'recording_predictions': [{'item_id': items[i], 'container_id': rows[i]['container_id'],
                                                        'truth': rows[i]['shape'], 'prediction': str(pred)} for i, pred in zip(test, predictions)],
                              'timing': timing}
    report['models']['efficientat'] = {'status': 'completed', 'source': 'https://github.com/fschmid56/EfficientAT',
                                       'checkpoint_release': 'https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt',
                                       'checkpoint_path': 'artifacts/mn04_as_mAP_432.pt', 'checkpoint_sha256': checkpoint_hash,
                                       'checkpoint_bytes': CHECKPOINT.stat().st_size, 'frozen': True,
                                       'features': 'Full provided audio, mean channels, scipy resample_poly to 32kHz; unchanged upstream AugmentMelSTFT eval mode: 128 mel bins, win_length=800, hopsize=320, n_fft=1024, preemphasis=.97, fmin=0/fmax=15000, log and fixed normalization (x+4.5)/5. Upstream mn04_as width=.4 returns spatially mean-pooled 384D convolution embeddings; 527 logits unused. No audio chunk cropping/padding, fine-tuning or augmentation.',
                                       'head_and_split': 'Identical original item IDs, outer container split, three training-only inner container folds and logistic-head C search. No test tuning.',
                                       'source_files': [{'path': str(p.relative_to(previous_cwd).as_posix()), 'sha256': sha256(p)} for p in sorted(SOURCE.rglob('*.py'))],
                                       'license_caveat': 'Upstream source MIT; exact checkpoint redistribution/commercial terms require independent review; no dataset rights implied.'}
    report.setdefault('additional_commands', []).append('python -m experiments.sound_water_efficientat')
    report['reproducibility']['efficientat'] = {'script_path': 'experiments/sound_water_efficientat.py',
                                               'script_sha256': sha256(SCRIPT), 'cache_path': 'artifacts/sound-of-water/efficientat_embeddings_384.npz',
                                               'cache_sha256': sha256(CACHE), 'cache_shape': list(x.shape),
                                               'manifest_fingerprint': fingerprint}
    report['environment']['efficientat_comparator'] = {'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
                                                      'torchvision': torchvision.__version__, 'threads': 3, 'interop_threads': 1,
                                                      'device': 'cpu', 'processor': platform.processor()}
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(REPORT), 'status': 'completed', 'result': report['results'][KEY]['held_out_container_metrics'],
                      'timing': {k: v for k, v in timing.items() if k != 'per_file'}}), flush=True)
except Exception as exc:
    blocker = {'stage': 'efficientat_comparator', 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    report['models']['efficientat'] = {'status': 'blocked', 'blocker': blocker,
                                       'command': 'python -m experiments.sound_water_efficientat'}
    report['failures'].append(blocker)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    raise
