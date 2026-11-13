"""Bounded Pucks/material and single-phone-motor experiments; CPU is not iPhone.

Run --download-pucks, --download-motor, then --benchmark. Outputs stay scoped.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import csv
import hashlib
import io
import json
import platform
import re
import time
import zipfile
from pathlib import Path
import joblib
import numpy as np
import requests
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'
PUCKS = OUT / 'datasets' / 'pucks'
MOTOR = OUT / 'datasets' / 'phone-motor'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fetch(url, path, cap, size=None, checksum=None, algorithm='sha256'):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if size is not None and path.stat().st_size != size:
            raise ValueError(f'Existing size mismatch: {path}')
        if checksum and digest(path, algorithm) != checksum:
            raise ValueError(f'Existing checksum mismatch: {path}')
        return
    temporary = path.with_suffix(path.suffix + '.part')
    h = hashlib.new(algorithm)
    count = 0
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        if int(response.headers.get('Content-Length', 0)) > cap:
            raise ValueError('Content-Length exceeds cap')
        with temporary.open('wb') as handle:
            for block in response.iter_content(1024 * 1024):
                count += len(block)
                if count > cap:
                    raise ValueError('Stream exceeds download cap')
                handle.write(block)
                h.update(block)
    if size is not None and count != size:
        raise ValueError(f'Size mismatch: {count} != {size}')
    if checksum and h.hexdigest() != checksum:
        raise ValueError(f'{algorithm} mismatch')
    temporary.rename(path)


def download_pucks():
    meta = requests.get('https://zenodo.org/api/records/7551338', timeout=60).json()
    save(PUCKS / 'release-metadata.json', meta)
    entry, = meta['files']
    assert entry['size'] == 3776343888 and entry['checksum'] == 'md5:b5af986a20cc45f4f90b2699d0c0c80e'
    path = PUCKS / entry['key']
    fetch(entry['links']['self'], path, 3900000000, entry['size'], entry['checksum'][4:], 'md5')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        save(PUCKS / 'archive-manifest.json', [{'name': x.filename, 'size': x.file_size, 'crc': x.CRC} for x in archive.infolist()])
        for name in names:
            if 'readme' in name.lower() or name.lower().endswith(('.csv', '.pdf')):
                if archive.getinfo(name).file_size < 1000000:
                    target = PUCKS / ('source-' + Path(name).name)
                    target.write_bytes(archive.read(name))
    save(PUCKS / 'download-verification.json', {'bytes': path.stat().st_size, 'md5': digest(path, 'md5'), 'expected_md5': entry['checksum'], 'license': meta['metadata']['license'], 'zip_members': len(names)})
    print('Pucks download verified', flush=True)


def download_motor():
    meta = requests.get('https://data.mendeley.com/public-api/datasets/bxcnxfm94c', timeout=60).json()
    assert meta['version'] == 5
    save(MOTOR / 'release-metadata.json', meta)
    selected = [x for x in meta['files'] if x['filename'].lower().endswith(('.wav', '.csv', '.md', '.txt'))]
    total = sum(x['size'] for x in selected)
    assert total <= 200000000
    verification = []
    for entry in selected:
        detail = entry['content_details']
        path = MOTOR / entry['filename']
        fetch(detail['download_url'], path, 200000000, entry['size'], detail['sha256_hash'])
        if path.suffix.lower() == '.wav':
            assert path.read_bytes()[:4] == b'RIFF'
            with sf.SoundFile(path) as audio:
                info = {'frames': len(audio), 'rate': audio.samplerate, 'channels': audio.channels}
        else:
            info = None
        verification.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path), 'audio': info})
    assert sum(x['file'].lower().endswith('.wav') for x in verification) == 20
    save(MOTOR / 'download-verification.json', {'bytes': total, 'files': verification, 'license': meta['data_licence'], 'm4a_excluded': 'Decoded WAVs are working copies of the 20 original M4A recordings, not independent samples.'})
    print('Motor WAVs and metadata verified', flush=True)


def mono16(audio, rate):
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 16000:
        g = int(np.gcd(rate, 16000))
        audio = resample_poly(audio, 16000 // g, rate // g)
    audio = np.asarray(audio, dtype=np.float32)
    assert audio.ndim == 1 and np.isfinite(audio).all() and len(audio) >= 400
    return audio


def features(audio):
    audio = audio - audio.mean()
    frames = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160]
    power = np.abs(np.fft.rfft(frames * np.hanning(400), n=512)) ** 2
    hz = np.fft.rfftfreq(512, 1 / 16000)
    edges = 700 * (10 ** (np.linspace(2595*np.log10(1+80/700), 2595*np.log10(1+7600/700), 34)/2595)-1)
    filters = np.maximum(0, np.minimum((hz[None,:]-edges[:-2,None])/np.diff(edges)[:-1,None], (edges[2:,None]-hz[None,:])/np.diff(edges)[1:,None]))
    assert (filters.sum(axis=1) > 0).all()
    logmel = np.log(np.maximum(power @ filters.T, 1e-10))
    cepstra = dct(logmel, axis=1, norm='ortho')[:,:13]
    value = np.r_[cepstra.mean(0), cepstra.std(0), logmel.mean(0), logmel.std(0)].astype(np.float32)
    assert value.shape == (90,) and np.isfinite(value).all()
    return value


def make_head(name):
    if name == 'majority':
        return DummyClassifier(strategy='most_frequent')
    estimator = LogisticRegression(C=1., max_iter=3000, class_weight='balanced', random_state=42) if name == 'logistic' else SVC(C=1., kernel='rbf', gamma='scale', class_weight='balanced')
    return make_pipeline(StandardScaler(), estimator)


def metric(y, prediction, labels):
    return {'accuracy': float(accuracy_score(y, prediction)), 'balanced_accuracy': float(balanced_accuracy_score(y, prediction)), 'macro_f1': float(f1_score(y, prediction, labels=labels, average='macro', zero_division=0)), 'confusion_matrix': confusion_matrix(y, prediction, labels=labels).tolist()}


def evaluate(rows, matrix, dataset, representation):
    y = np.array([r['label'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    labels = sorted(set(y))
    results = {}
    for name in ('majority', 'logistic', 'rbf_svm'):
        predictions = np.full(len(rows), '', dtype=object)
        folds = []
        for train, test in LeaveOneGroupOut().split(matrix, y, groups):
            assert not set(groups[train]) & set(groups[test])
            assert not {rows[i]['sha256'] for i in train} & {rows[i]['sha256'] for i in test}
            assert set(y[train]) == set(labels)
            model = make_head(name)
            start = time.perf_counter()
            model.fit(matrix[train], y[train])
            predictions[test] = model.predict(matrix[test])
            folds.append({'held_out': str(groups[test[0]]), 'train': len(train), 'test': len(test), 'fit_predict_ms': 1000*(time.perf_counter()-start), **metric(y[test], predictions[test], labels)})
            folds[-1]['held_out_source_sha256'] = sorted({rows[i]['sha256'] for i in test})
            folds[-1]['group_fingerprint_sha256'] = hashlib.sha256('\n'.join(folds[-1]['held_out_source_sha256']).encode()).hexdigest()
        # Windows are not independent: report majority-voted complete file and group outcomes too.
        file_y, file_p, file_rows = [], [], []
        for filename in sorted({r['file'] for r in rows}):
            indices = [i for i,r in enumerate(rows) if r['file'] == filename]
            truth, = set(y[indices])
            values, counts = np.unique(predictions[indices], return_counts=True)
            prediction = values[np.argmax(counts)]
            file_y.append(truth); file_p.append(prediction)
            file_rows.append({'file': filename, 'group': rows[indices[0]]['group'], 'sha256': rows[indices[0]]['sha256'], 'truth': truth, 'prediction': prediction, 'windows': len(indices)})
        group_y, group_p, group_rows = [], [], []
        for group in sorted(set(groups)):
            # Motor speed folds contain both labels; aggregate each source condition separately.
            for truth in sorted(set(y[groups == group])):
                indices = np.flatnonzero((groups == group) & (y == truth))
                values, counts = np.unique(predictions[indices], return_counts=True)
                prediction = values[np.argmax(counts)]
                group_y.append(truth); group_p.append(prediction)
                group_rows.append({'group': group, 'truth': truth, 'prediction': prediction, 'windows': len(indices)})
        model = make_head(name).fit(matrix, y)
        model_path = OUT / f'puck-motor-{dataset}-{representation}-{name}.joblib'
        joblib.dump(model, model_path)
        timings = []
        for _ in range(20):
            start = time.perf_counter(); model.predict(matrix[:1]); timings.append((time.perf_counter()-start)*1000)
        results[name] = {'window_metrics_descriptive_only': metric(y, predictions, labels), 'file_metrics': metric(file_y, file_p, labels), 'group_condition_metrics': metric(group_y, group_p, labels), 'folds': folds, 'file_predictions': file_rows, 'group_predictions': group_rows, 'window_predictions': [{**row, 'prediction': predictions[i]} for i,row in enumerate(rows)], 'deployment_refit': {'all_data_not_evaluation': True, 'path': str(model_path.relative_to(ROOT)), 'bytes': model_path.stat().st_size, 'sha256': digest(model_path), 'head_cpu_ms_p50': float(np.median(timings)), 'head_cpu_ms_p95': float(np.quantile(timings,.95))}}
    return results


def load_yamnet():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    path = OUT / 'tfhub' / '9616fd04ec2360621642ef9455b84f4b668e219e'
    assert (path / 'saved_model.pb').exists(), 'Official cached YAMNet SavedModel unavailable'
    model = tf.saved_model.load(str(path))
    return model, path


def run_dataset(dataset, yamnet=None):
    rows, acoustic, embeddings = [], [], []
    feature_times, embedding_times = [], []
    cache = OUT / f'puck-motor-{dataset}-features.npz'
    manifest = OUT / f'puck-motor-{dataset}-manifest.json'
    source_inventory = []
    if cache.exists() and manifest.exists():
        data = np.load(cache)
        rows = json.loads(manifest.read_text())
        acoustic = data['acoustic']
        embeddings = data['yamnet']
        timings = json.loads((OUT / f'puck-motor-{dataset}-extraction.json').read_text())
        if yamnet is not None and embeddings.shape[1] != 1024:
            raise ValueError('Cached features lack YAMNet embeddings; choose a fresh cache')
    else:
        if dataset == 'pucks':
            archive_path = PUCKS / 'IDMT_ISA_PUCKS.zip'
            archive = zipfile.ZipFile(archive_path)
            sources = sorted(x for x in archive.namelist() if x.lower().endswith('.wav'))
        else:
            archive = None
            sources = sorted(x.name for x in MOTOR.glob('*.wav'))
        motor_index = {}
        if dataset == 'motor':
            with (MOTOR / 'metadata.csv').open(newline='', encoding='utf-8-sig') as handle:
                metadata = list(csv.DictReader(handle))
            assert len(metadata) == 20 and len({r['acoustic_m4a'] for r in metadata}) == 20
            motor_index = {Path(r['acoustic_wav']).name: r for r in metadata}
            assert set(motor_index) == set(sources)
        for file in sources:
            if dataset == 'pucks':
                # The distributed PDF documents three-minute games split into minute patches.
                match = re.search(r'/puck_([AEGI])/', file)
                if not match or match[1].upper() == 'A':
                    continue
                label = {'E':'ABS','G':'PA2200','I':'PA12'}[match[1].upper()]
                identity = re.fullmatch(r'vol_(000|050|100)/puck_([EGI])/puck\2_game-(\d{2})_3min_vol\1_patch-([012])\.wav', file)
                if not identity:
                    raise ValueError(f'Unexpected documented game/patch identifier: {file}')
                noise, puck, game, patch = identity.groups()
                group = f'puck-{puck}-vol-{noise}-game-{game}'
                payload = archive.read(file)
                sha = hashlib.sha256(payload).hexdigest()
                audio, rate = sf.read(io.BytesIO(payload), dtype='float32', always_2d=True)
                assert rate == 44100 and audio.shape[1] == 2 and 59 <= len(audio)/rate <= 61
            else:
                match = re.fullmatch(r'audio_(debonded|repaired)_(\d+)rpm_(forward|reverse)\.wav', file)
                if not match:
                    raise ValueError(f'Unknown condition filename: {file}')
                metadata_row = motor_index[file]
                label, group = metadata_row['condition'], metadata_row['rotational_speed_rpm']
                assert (label, group, metadata_row['rotation_direction']) == match.groups()
                path = MOTOR / file
                sha = digest(path)
                audio, rate = sf.read(path, dtype='float32', always_2d=True)
            info = {'file': file, 'sha256': sha, 'original_rate': rate, 'channels': audio.shape[1], 'duration_seconds': len(audio)/rate, 'label': label, 'group': group}
            if dataset == 'motor':
                info.update(source_recording=motor_index[file]['acoustic_m4a'], direction=motor_index[file]['rotation_direction'])
            source_inventory.append(info)
            audio = mono16(audio, rate)
            # Three nonoverlapping, regularly spaced four-second windows per minute/source.
            # All windows of all minute chunks from one game remain in the same fold.
            duration = len(audio)/16000
            assert duration >= 12, 'Need three disjoint four-second windows per recording'
            starts = np.linspace(0, max(0,duration-4), 3)
            for start_seconds in starts:
                start = int(start_seconds*16000)
                clip = audio[start:start+64000]
                assert np.max(np.abs(clip)) <= 1, 'YAMNet requires normalized float waveform; do not silently clip source audio'
                t = time.perf_counter(); acoustic.append(features(clip)); feature_times.append((time.perf_counter()-t)*1000)
                if yamnet is not None:
                    t = time.perf_counter()
                    _, embedding, _ = yamnet(clip)
                    value = np.asarray(embedding).mean(0)
                    assert value.shape == (1024,) and np.isfinite(value).all()
                    embeddings.append(value)
                    embedding_times.append((time.perf_counter()-t)*1000)
                rows.append({**info, 'start_seconds': float(start_seconds), 'window_seconds': len(clip)/16000})
            print(dataset, len(source_inventory), file, flush=True)
        if archive:
            archive.close()
        acoustic = np.stack(acoustic)
        embeddings = np.stack(embeddings) if embeddings else np.empty((len(rows),0))
        np.savez_compressed(cache, acoustic=acoustic, yamnet=embeddings)
        save(manifest, rows)
        timings = {'feature_ms_p50': float(np.median(feature_times)), 'feature_ms_p95': float(np.quantile(feature_times,.95)), 'yamnet_ms_p50': float(np.median(embedding_times)) if embedding_times else None, 'yamnet_ms_p95': float(np.quantile(embedding_times,.95)) if embedding_times else None, 'files': source_inventory, 'includes': 'frontend only for acoustic; YAMNet frontend+encoder only; excludes disk I/O and resampling'}
        save(OUT / f'puck-motor-{dataset}-extraction.json', timings)
    assert len(rows) == len(acoustic) and len(set(r['sha256'] for r in rows)) == len(set(r['file'] for r in rows)), 'Duplicate source files must not cross folds'
    if dataset == 'pucks':
        files_per_group = {group: {r['file'] for r in rows if r['group'] == group} for group in {r['group'] for r in rows}}
        assert len(files_per_group) == 45 and all(len(files) == 3 for files in files_per_group.values())
        assert len({r['file'] for r in rows}) == 135
    if dataset == 'motor':
        pcm_hashes = {}
        for file in sorted({r['file'] for r in rows}):
            audio, rate = sf.read(MOTOR / file, dtype='float32', always_2d=True)
            pcm_hashes[file] = hashlib.sha256(str((rate, audio.shape)).encode() + audio.tobytes()).hexdigest()
        assert len(set(pcm_hashes.values())) == 20, 'Duplicate decoded PCM conditions are not independent'
        save(OUT / 'puck-motor-motor-duplicate-audit.json', {'source_files': 20, 'pcm_sha256': pcm_hashes, 'byte_identical_files': False, 'pcm_identical_files': False, 'm4a_included': False, 'source_pairing': 'Metadata maps each included WAV to one distinct excluded original M4A; formats are not extra observations.'})
    save(OUT / f'puck-motor-{dataset}-protocol.json', {
        'source': 'https://zenodo.org/records/7551338' if dataset == 'pucks' else 'https://data.mendeley.com/datasets/bxcnxfm94c/5',
        'script_sha256': digest(Path(__file__)),
        'dependency_versions': {name: __import__(name).__version__ for name in ('numpy', 'scipy', 'sklearn', 'soundfile')},
        'features_cache_sha256': digest(cache),
        'source_manifest_sha256': digest(manifest),
        'fixed_candidates': ['majority', 'standardized logistic C=1', 'standardized RBF-SVM C=1 gamma=scale'],
        'selection': 'No winner selected by held-out results; all fixed candidates reported.',
        'frontend': 'Stereo channel average then polyphase resampling to 16kHz; retain source files unchanged. Three 4s nonoverlapping windows equally spaced over each file.',
        'edge': 'CPU proxy only, two numerical threads; four-second acquisition wait distinct from frontend/encoder and head timings. No physical iPhone measured.',
        'aggregation': 'Deterministic majority vote per file and per held-out group/condition; ties resolve lexical label order, not calibrated probability.',
        'source_duplicates': 'SHA256 source identity checked across folds; M4A counterparts excluded, so decoded WAVs are not extra training recordings.',
        'units': 'Samples within file/game are dependent; window metrics descriptive only; primary file and complete-game metrics retained.',
    })
    result = {'status': 'complete', 'files': len(set(r['file'] for r in rows)), 'groups': len(set(r['group'] for r in rows)), 'windows': len(rows), 'labels': sorted(set(r['label'] for r in rows)), 'feature_contract': '90D: 13 MFCC + 32 logmel, each mean/std, 16kHz mono, 25ms Hann/10ms hop, 4-second nonoverlapping windows', 'tuning': 'Fixed C=1 logistic and RBF-SVM; no test-based tuning. StandardScaler fitted only inside training fold.', 'timings_cpu_not_phone': timings, 'models': {'mfcc_logmel': evaluate(rows, acoustic, dataset, 'mfcc-logmel')}}
    if embeddings.shape[1]:
        result['models']['yamnet'] = evaluate(rows, embeddings, dataset, 'yamnet')
    if dataset == 'pucks':
        result.update(protocol='Leave one complete three-minute game out; all three minute patches, averaged stereo channels and windows together. Game IDs combine puck, noise level and game number. Player/physical-puck identity mapping is not published.', decision='Exclude unknown-material factory puck_A and no_puck/background files; classify ABS, PA2200, PA12 only.', limitations=['Material is confounded with manufacturing process (ABS/FDM, PA2200/SLS, PA12/MJF), geometry and player/game conditions; not universal cross-object material identification.', 'The PDF says different pucks and player sets per game, but releases no persistent physical-puck/player IDs for stronger grouping.', 'Air-hockey continuous impacts at tabletop microphones, not controlled phone taps.', 'License conflict: Zenodo metadata cc-by-nc-nd-4.0; distributed PDF: The dataset is provided for evaluation purpose under the Creative Commons Attribution-ShareAlike 4.0 International License (by-sa). Local evaluation only; no raw redistribution; production/redistribution rights unresolved.', 'ZIP contains 225 WAV files (45 per category including background), not the 260 stated in metadata and PDF. Used all 135 known-material WAVs.'])
    else:
        result.update(protocol='Leave one speed out across both conditions and both rotation directions. Source WAV is one recording, three windows; no M4A included.', limitations=['Single motor, condition acquired before/after rebonding; recording/session/repair may be confounded.', 'Only five held-out speeds and one recording per condition/speed/direction; not a motor-held-out generalization result.', 'WAVs are decoded copies of M4A with first 2 seconds removed, not additional independent recordings.', 'Android microphone 1m away, no-load steady operation; not tap or iPhone evidence.'])
    save(OUT / f'puck-motor-{dataset}-report.json', result)
    print(json.dumps({'dataset': dataset, 'files': result['files'], 'groups': result['groups'], 'metrics': {representation: {name: model['file_metrics'] for name, model in models.items()} for representation, models in result['models'].items()}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-pucks', action='store_true')
    parser.add_argument('--download-motor', action='store_true')
    parser.add_argument('--benchmark', choices=['pucks','motor','both'])
    parser.add_argument('--yamnet', action='store_true')
    args = parser.parse_args()
    if args.download_pucks:
        download_pucks()
    if args.download_motor:
        download_motor()
    if args.benchmark:
        model, model_path = load_yamnet() if args.yamnet else (None,None)
        provenance = {'script_sha256':digest(Path(__file__)), 'python':platform.python_version(), 'platform':platform.platform(), 'thread_limit':2, 'edge_evidence':'Windows CPU only; no phone, conversion, memory or battery measured', 'yamnet':None}
        if model_path:
            provenance['yamnet'] = {'path':str(model_path.relative_to(ROOT)), 'saved_model_sha256':digest(model_path/'saved_model.pb'), 'parameter_files_bytes':sum(x.stat().st_size for x in model_path.rglob('*') if x.is_file()), 'contract':'Official frozen YAMNet; mono 16kHz float32; 1024D frame embeddings mean pooled within four seconds; no encoder fitting'}
            provenance['yamnet']['files_sha256'] = {str(x.relative_to(model_path)): digest(x) for x in model_path.rglob('*') if x.is_file()}
        save(OUT/'puck-motor-runtime.json',provenance)
        for dataset in (['pucks','motor'] if args.benchmark == 'both' else [args.benchmark]):
            run_dataset(dataset,model)


if __name__ == '__main__':
    main()
