"""Audio-only Sound-of-Water pilot; run from repository root with Python."""
import os
os.environ['OMP_NUM_THREADS'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '3'
os.environ['MKL_NUM_THREADS'] = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '3'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import platform
import sys
import time
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import scipy
import sklearn
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('artifacts/sound-of-water')
REPORT = Path('artifacts/sound-water-benchmark.json')
REVISION = '12575460ee39d6adaebbe5aff531a5f4a24a627b'
BASE = f'https://huggingface.co/datasets/bpiyush/sound-of-water/resolve/{REVISION}/'
SEED = 42
MAX_RECORDINGS_PER_CONTAINER = 3
CHECKPOINT = Path('artifacts/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e')
CLASSES = ['bottleneck', 'cylindrical', 'semiconical']
CS = [0.01, 0.1, 1.0, 10.0]
metadata_paths = ['splits/train.csv', 'splits/test_I.csv', 'splits/test_II.csv', 'splits/test_III.csv', 'annotations/localisation.csv']
failures = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(relative):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(BASE + relative, timeout=120) as response:
            content = response.read()
        path.write_bytes(content)
    return path


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def file_record(relative):
    path = ROOT / relative
    return {'path': str(path.as_posix()), 'dataset_relative_path': relative,
            'url': BASE + relative, 'bytes': path.stat().st_size, 'sha256': digest(path)}


def metrics(y, pred):
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, pred, labels=CLASSES).tolist(),
            'class_order': CLASSES}


def container_predictions(indices, probabilities, estimator_classes):
    groups = collections.defaultdict(list)
    for i, probability in zip(indices, probabilities):
        groups[selected[i]['container_id']].append(probability)
    predictions = []
    for group in sorted(groups):
        scores = np.mean(groups[group], axis=0)
        predictions.append({'container_id': group, 'truth': container_labels[group],
                            'prediction': str(estimator_classes[np.argmax(scores)]),
                            'mean_probabilities': {str(c): float(v) for c, v in zip(estimator_classes, scores)},
                            'recordings': len(groups[group])})
    return predictions


def container_score(indices, probabilities, estimator_classes):
    records = container_predictions(indices, probabilities, estimator_classes)
    return metrics([r['truth'] for r in records], [r['prediction'] for r in records])


def summarize_split(groups):
    indices = [i for i, r in enumerate(selected) if r['container_id'] in set(groups)]
    return {'container_ids': sorted(groups), 'containers': len(groups), 'recordings': len(indices),
            'container_class_counts': dict(collections.Counter(container_labels[g] for g in groups)),
            'recording_class_counts': dict(collections.Counter(selected[i]['shape'] for i in indices)),
            'item_ids': [selected[i]['item_id'] for i in indices]}


started = time.perf_counter()
ROOT.mkdir(parents=True, exist_ok=True)
for relative in metadata_paths:
    download(relative)
frames = {Path(relative).stem: read_csv(ROOT / relative) for relative in metadata_paths[:4]}
localisation = read_csv(ROOT / metadata_paths[-1])
unique = {}
for split, rows in frames.items():
    for row in rows:
        if row['item_id'] in unique:
            assert all(unique[row['item_id']][key] == row[key] for key in ['container_id', 'shape', 'material'])
        else:
            unique[row['item_id']] = row
by_container = collections.defaultdict(list)
for row in unique.values():
    by_container[row['container_id']].append(row)
container_labels = {}
for group, rows in by_container.items():
    assert len({r['shape'] for r in rows}) == 1
    container_labels[group] = rows[0]['shape']
selected = [row for group in sorted(by_container, key=lambda g: int(g.split('_')[1]))
            for row in sorted(by_container[group], key=lambda r: r['item_id'])[:MAX_RECORDINGS_PER_CONTAINER]]
assert len({r['item_id'] for r in selected}) == len(selected)
assert set(r['shape'] for r in selected) == set(CLASSES)
for row in selected:
    row['audio_relative_path'] = f"audios/{row['item_id']}.wav"

# These metadata-only choices precede downloads, feature extraction and scored predictions.
groups = np.array(sorted(container_labels))
group_y = np.array([container_labels[g] for g in groups])
train_groups, test_groups = train_test_split(groups, test_size=0.25, random_state=SEED, stratify=group_y)
assert not set(train_groups) & set(test_groups)
train_indices = np.array([i for i, r in enumerate(selected) if r['container_id'] in set(train_groups)])
test_indices = np.array([i for i, r in enumerate(selected) if r['container_id'] in set(test_groups)])
inner_folds = []
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for tr, va in inner.split(train_groups, [container_labels[g] for g in train_groups]):
    tr_groups, va_groups = set(train_groups[tr]), set(train_groups[va])
    assert not tr_groups & va_groups and not (tr_groups | va_groups) & set(test_groups)
    tr_indices = np.array([i for i in train_indices if selected[i]['container_id'] in tr_groups])
    va_indices = np.array([i for i in train_indices if selected[i]['container_id'] in va_groups])
    inner_folds.append((tr_indices, va_indices, tr_groups, va_groups))
print(json.dumps({'phase': 'download', 'recordings': len(selected), 'containers': len(groups),
                  'train_containers': len(train_groups), 'test_containers': len(test_groups)}), flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(download, r['audio_relative_path']): r for r in selected}
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            failures.append({'stage': 'download', 'path': futures[future]['audio_relative_path'],
                             'exception': type(exc).__name__, 'message': str(exc)})
if failures:
    REPORT.write_text(json.dumps({'status': 'blocked', 'dataset_revision': REVISION,
                                 'failures': failures,
                                 'downloaded_files': [file_record(p) for p in metadata_paths + [r['audio_relative_path'] for r in selected] if (ROOT / p).is_file()]}, indent=2))
    raise RuntimeError('Required audio download failed; see report; no split silently reduced.')

tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(1)
manifest_fingerprint = hashlib.sha256(json.dumps([(r['item_id'], digest(ROOT / r['audio_relative_path'])) for r in selected]).encode()).hexdigest()
cache = ROOT / 'features.npz'
if cache.exists():
    saved = np.load(cache, allow_pickle=False)
    assert str(saved['manifest_fingerprint']) == manifest_fingerprint
    yamnet_x, mfcc_x = saved['yamnet'], saved['mfcc']
    audio_info = json.loads(str(saved['audio_info_json']))
else:
    model = tf.saved_model.load(str(CHECKPOINT))
    yamnet_features, mfcc_features, audio_info = [], [], []
    for index, row in enumerate(selected):
        path = ROOT / row['audio_relative_path']
        waveform, rate = sf.read(path, dtype='float32', always_2d=True)
        channels = waveform.shape[1]
        mono = waveform.mean(axis=1)
        assert mono.size > 0 and np.isfinite(mono).all() and np.abs(mono).max() <= 1.000001
        divisor = math.gcd(rate, 16000)
        x = resample_poly(mono, 16000 // divisor, rate // divisor).astype(np.float32) if rate != 16000 else mono
        scores, embeddings, spectrogram = model(x)
        embeddings = np.asarray(embeddings)
        assert embeddings.ndim == 2 and embeddings.shape[1] == 1024 and np.isfinite(embeddings).all()
        yamnet_features.append(embeddings.mean(axis=0))
        mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=13, n_fft=1024, hop_length=320, n_mels=64, fmin=50, fmax=7600)
        mfcc_features.append(np.r_[mfcc.mean(axis=1), mfcc.std(axis=1)])
        audio_info.append({'item_id': row['item_id'], 'sample_rate_hz': rate, 'channels': channels,
                           'seconds': len(mono) / rate, 'yamnet_frames': int(len(embeddings))})
        if index % 10 == 0:
            print(json.dumps({'phase': 'features', 'completed': index + 1, 'total': len(selected)}), flush=True)
    yamnet_x, mfcc_x = np.asarray(yamnet_features), np.asarray(mfcc_features)
    np.savez_compressed(cache, yamnet=yamnet_x, mfcc=mfcc_x,
                        manifest_fingerprint=manifest_fingerprint, audio_info_json=json.dumps(audio_info))
assert yamnet_x.shape == (len(selected), 1024) and mfcc_x.shape == (len(selected), 26)
assert np.isfinite(yamnet_x).all() and np.isfinite(mfcc_x).all()
y = np.array([r['shape'] for r in selected])
results = {}
for name, x in [('yamnet_mean_1024', yamnet_x), ('mfcc_mean_std_26', mfcc_x)]:
    search = []
    for c in CS:
        fold_scores = []
        for tr, va, _, _ in inner_folds:
            head = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=3000, random_state=SEED))
            head.fit(x[tr], y[tr])
            fold_scores.append(container_score(va, head.predict_proba(x[va]), head.classes_)['macro_f1'])
        search.append({'C': c, 'fold_container_macro_f1': fold_scores, 'mean_container_macro_f1': float(np.mean(fold_scores))})
    best = max(search, key=lambda row: row['mean_container_macro_f1'])
    head = make_pipeline(StandardScaler(), LogisticRegression(C=best['C'], class_weight='balanced', max_iter=3000, random_state=SEED))
    head.fit(x[train_indices], y[train_indices])
    probabilities = head.predict_proba(x[test_indices])
    predictions = head.classes_[np.argmax(probabilities, axis=1)]
    results[name] = {'selected_C': best['C'], 'hyperparameter_search_train_only': search,
                     'held_out_recording_metrics': metrics(y[test_indices], predictions),
                     'held_out_container_metrics': container_score(test_indices, probabilities, head.classes_),
                     'container_predictions': container_predictions(test_indices, probabilities, head.classes_),
                     'recording_predictions': [{'item_id': selected[i]['item_id'], 'container_id': selected[i]['container_id'],
                                               'truth': selected[i]['shape'], 'prediction': str(pred)} for i, pred in zip(test_indices, predictions)]}
    print(json.dumps({'model': name, 'held_out_container_metrics': results[name]['held_out_container_metrics']}), flush=True)

dummy = DummyClassifier(strategy='most_frequent', random_state=SEED).fit(np.zeros((len(train_indices), 1)), y[train_indices])
dummy_p = dummy.predict_proba(np.zeros((len(test_indices), 1)))
results['most_frequent_dummy'] = {'held_out_recording_metrics': metrics(y[test_indices], dummy.predict(np.zeros((len(test_indices), 1)))),
                                  'held_out_container_metrics': container_score(test_indices, dummy_p, dummy.classes_)}
all_relative = metadata_paths + [r['audio_relative_path'] for r in selected]
downloaded = [file_record(relative) for relative in all_relative]
report = {
    'status': 'completed', 'experiment': 'Sound-of-Water audio-only unseen-container shape classification pilot',
    'command': 'python -m experiments.sound_water_benchmark', 'random_seed': SEED,
    'source': {'dataset': 'bpiyush/sound-of-water', 'url': 'https://huggingface.co/datasets/bpiyush/sound-of-water',
               'revision': REVISION, 'license': 'unverified: inspected dataset card/API has no license field; do not infer dataset rights from model/code license'},
    'target': {'field': 'shape', 'classes': CLASSES, 'definition': 'Container shape metadata labels; one invariant label per physical container_id. Audio-only classification, not time-varying water level.'},
    'coverage': {'official_csv_rows': {split: len(rows) for split, rows in frames.items()},
                 'official_csv_rows_sum_with_overlap': sum(map(len, frames.values())),
                 'unique_items_in_csv_union': len(unique), 'localisation_rows': len(localisation),
                 'unique_localisation_video_ids': len({r['video_id'] for r in localisation}),
                 'unique_split_video_ids': len({r['video_id'] for r in unique.values()}),
                 'unique_containers_in_union': len(groups), 'selected_recordings': len(selected),
                 'selection': 'Lexicographically first up to three unique item_id recordings per container from deduplicated union of four split CSVs. Metadata-only deterministic choice before feature extraction; custom pilot, not official paper split.',
                 'container_class_counts': dict(collections.Counter(container_labels.values())),
                 'selected_recording_class_counts': dict(collections.Counter(y)),
                 'audio_seconds_total': sum(r['seconds'] for r in audio_info),
                 'sample_rate_counts': dict(collections.Counter(r['sample_rate_hz'] for r in audio_info)),
                 'channel_counts': dict(collections.Counter(r['channels'] for r in audio_info)),
                 'downloaded_audio_bytes': sum(r['bytes'] for r in downloaded if r['dataset_relative_path'].endswith('.wav')),
                 'all_downloaded_bytes': sum(r['bytes'] for r in downloaded)},
    'split': {'method': 'Stratified 75/25 train/test split of container IDs, random_state=42; all recordings from each container stay in its split.',
              'group_overlap_count': len(set(train_groups) & set(test_groups)),
              'train': summarize_split(train_groups), 'test': summarize_split(test_groups),
              'inner_validation': [{'fold': fold, 'train': summarize_split(sorted(trg)), 'validation': summarize_split(sorted(vag))}
                                   for fold, (_, _, trg, vag) in enumerate(inner_folds)],
              'tuning': 'Three-fold stratified CV of training container IDs only, maximize unweighted mean per-fold container macro-F1; C=[0.01,0.1,1,10], ties choose first/smallest C. No test-set tuning. StandardScaler fits training recordings within each fold only.'},
    'models': {'yamnet': {'source': 'https://tfhub.dev/google/yamnet/1', 'saved_model_path': str(CHECKPOINT.as_posix()),
                           'frozen': True, 'features': 'Full provided audio, mono channel average, polyphase resampling to 16kHz, arithmetic mean of all 1024D frame embeddings. AudioSet 521 event scores unused.',
                           'checkpoint_files': [{'path': str(p.as_posix()), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(CHECKPOINT.rglob('*')) if p.is_file()]},
               'mfcc': {'features': 'Same full 16kHz mono audio; 13 librosa MFCC coefficients, per-coefficient temporal mean/std = 26D; n_fft=1024, hop_length=320, n_mels=64, fmin=50, fmax=7600; librosa remaining defaults.', 'frozen_encoder': False},
               'head': 'StandardScaler + L2 LogisticRegression, class_weight=balanced, max_iter=3000, seed=42',
               'aggregation': 'Mean class probabilities across a held-out container recordings; maximum selects shape. Primary metrics weight each container equally. Recording metrics are secondary and correlated.'},
    'results': results, 'downloaded_files': downloaded, 'audio_file_metadata': audio_info,
    'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'librosa': librosa.__version__,
                    'soundfile': sf.__version__, 'tensorflow': tf.__version__,
                    'thread_limits': {'tensorflow_intra': 3, 'tensorflow_inter': 1, 'OMP': 3, 'OPENBLAS': 3, 'MKL': 3}},
    'failures': failures,
    'caveats': ['This is pouring-water audio, not taps on a static vessel. Container shape labels are not fill-level/volume labels. The result validates neither THUD static-vessel-fill estimation nor tile bonding.',
                'No videos, bounding boxes, masks, visual embeddings, time-varying visual annotations, or video-derived feature inputs were downloaded or used. Shape metadata is the supervised label only. Provided audio is already trimmed; localisation timing is not used to crop it again.',
                'The union of official CSVs contains duplicate item IDs across official splits; deduplicated before sampling. Official test_I explicitly includes seen containers, so official train/test_I would not satisfy the requested container-held-out protocol.',
                '48 containers and 12 held-out containers make this a small pilot; it is not the original paper evaluation and only up to three recordings/container were used. Classes, material, room, recording session and vessel geometry may be confounded; container grouping does not remove collection-domain confounds.',
                'All settings, pooling and C search are fixed in this script; test labels do not tune preprocessing, scaling, regularization or selection. No confidence interval is claimed from correlated recordings.',
                'General AudioSet pretraining exposure to the dataset exact sounds is not independently audited; frozen extractor training is external, task head uses only the declared training containers.',
                'Dataset license is unverified because no license declaration was found in the inspected dataset card/API. Separate code/checkpoint licenses do not grant rights to this dataset.',
                'No iPhone runtime, Core ML/TFLite export, RAM, battery or hardware latency was measured. Desktop runtime below is experiment wall time including downloads/feature extraction/head fitting, not per-inference latency.'],
    'experiment_wall_seconds': time.perf_counter() - started,
    'reproducibility': {'script_path': str((ROOT / 'benchmark.py').as_posix()), 'script_sha256': digest(ROOT / 'benchmark.py'),
                        'feature_cache_path': str(cache.as_posix()), 'feature_cache_sha256': digest(cache),
                        'manifest_fingerprint': manifest_fingerprint,
                        'assertions': ['unique selected item IDs', 'invariant per-container target labels', 'disjoint outer/inner container splits', 'finite nonempty normalized audio', '1024D frozen YAMNet embedding contract', 'finite feature shapes', 'cache fingerprint matches exact downloaded audio']}}
REPORT.write_text(json.dumps(report, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else v), encoding='utf-8')
print(json.dumps({'report': str(REPORT), 'status': 'completed', 'seconds': report['experiment_wall_seconds']}), flush=True)
