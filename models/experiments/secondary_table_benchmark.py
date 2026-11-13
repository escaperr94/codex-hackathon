"""Auditable small table datasets; no invented waveform rates or acquisition groups."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/datasets/secondary-tables'
CAP = 15_000_000
MANIFEST = []


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=90)


def get_json(url):
    with request(url) as response:
        return json.load(response)


def download(url, path, size, kind, expected):
    assert size < CAP
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        count = 0
        with request(url) as source, path.open('wb') as target:
            while block := source.read(65536):
                count += len(block)
                if count > size or count > CAP:
                    raise ValueError('Bounded download exceeded advertised size/cap')
                target.write(block)
    content = path.read_bytes()
    assert len(content) == size
    actual = hashlib.md5(content).hexdigest() if kind == 'MD5' else hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    assert actual == expected, (path, actual, expected)
    MANIFEST.append({'path': str(path.relative_to(ROOT)), 'url': url, 'bytes': size, 'source_digest_kind': kind, 'source_digest': expected, 'sha256': hashlib.sha256(content).hexdigest()})
    assert sum(row['bytes'] for row in MANIFEST) < CAP
    return path


def github(repo, path, folder):
    meta = get_json(f'https://api.github.com/repos/{repo}/contents/{path}')
    return download(meta['download_url'], BASE / folder / Path(path).name, meta['size'], 'Git blob SHA1', meta['sha'])


def acquire():
    BASE.mkdir(parents=True, exist_ok=True)
    meta = get_json('https://edatos.consorciomadrono.es/api/datasets/:persistentId/?persistentId=doi:10.21950/OBYUCV')
    (BASE / 'watermelon-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    files = meta['data']['latestVersion']['files']
    for item in files:
        f = item['dataFile']
        if f['id'] in (4702, 4703, 4705):
            try:
                download(f"https://edatos.consorciomadrono.es/api/access/datafile/{f['id']}", BASE / 'watermelon' / f['filename'], f['filesize'], f['checksum']['type'], f['checksum']['value'])
            except urllib.error.HTTPError as error:
                blocker = {'file_id': f['id'], 'filename': f['filename'], 'http_status': error.code, 'response': error.read().decode(), 'advertised_bytes': f['filesize'], 'advertised_checksum': f['checksum']}
                (BASE / f"watermelon-blocker-{f['id']}.json").write_text(json.dumps(blocker, indent=2), encoding='utf-8')
    for path in ('dataset/jackfruit.xlsx', 'dataset/README.md', 'README.md', 'LICENSE'):
        try:
            github('Abhinandan-Hooli/jackfruit-acoustic-maturity-classification', path, 'jackfruit' if path != 'README.md' else 'jackfruit-root')
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
    repo = 'Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12'
    listing = get_json(f'https://api.github.com/repos/{repo}/git/trees/main?recursive=1')
    (BASE / 'gpla-tree.json').write_text(json.dumps(listing, indent=2), encoding='utf-8')
    for item in listing['tree']:
        path = item['path']
        if path in ('README.md', 'LICENSE') or ('data_v1/' in path and Path(path).name in ('data.csv', 'label.csv')):
            github(repo, path, 'gpla')
    (BASE / 'download-manifest.json').write_text(json.dumps(MANIFEST, indent=2), encoding='utf-8')
    print(json.dumps(MANIFEST, indent=2))


def measures(y, prediction):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    return {'n': len(y), 'accuracy': float(accuracy_score(y, prediction)),
        'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
        'macro_f1': float(f1_score(y, prediction, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y, prediction).tolist()}


def model(candidate):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    family, c = candidate
    head = SVC(C=c, class_weight='balanced', gamma='scale') if family == 'rbf_svm' else LogisticRegression(C=c, class_weight='balanced', max_iter=3000, random_state=42)
    return make_pipeline(StandardScaler(), head)


def evaluate(name, X, y, records, frontend=None, raw=None):
    import joblib
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    assert np.isfinite(X).all() and len(X) == len(y) == len(records)
    content = raw if raw is not None else X
    content_groups = np.array([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in content])
    candidates = [(family, c) for family in ('balanced_logistic', 'rbf_svm') for c in (.1, 1., 10.)]
    families = {'balanced_logistic': candidates[:3], 'rbf_svm': candidates[3:], 'nested_selected': candidates}
    predictions = {family: np.full(len(y), -1) for family in [*families, 'majority']}
    folds = []
    selected_all = []
    for fold, (train, test) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=42).split(X, y, content_groups)):
        assert not set(content_groups[train]) & set(content_groups[test])
        scores = {}
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=100 + fold).split(X[train], y[train], content_groups[train]))
        for candidate in candidates:
            values = []
            for a, b in inner:
                assert not set(content_groups[train[a]]) & set(content_groups[train[b]])
                estimator = model(candidate).fit(X[train[a]], y[train[a]])
                values.append(measures(y[train[b]], estimator.predict(X[train[b]]))['balanced_accuracy'])
            scores[candidate] = float(np.mean(values))
        chosen = {}
        for family, space in families.items():
            best = max(space, key=scores.get)
            chosen[family] = {'model': best[0], 'C': best[1]}
            estimator = model(best).fit(X[train], y[train])
            predictions[family][test] = estimator.predict(X[test])
        values, counts = np.unique(y[train], return_counts=True)
        predictions['majority'][test] = values[counts.argmax()]
        folds.append({'fold': fold, 'train_record_indices_zero_based': train.tolist(), 'test_record_indices_zero_based': test.tolist(),
            'inner_folds': [{'train_record_indices_zero_based': train[a].tolist(), 'validation_record_indices_zero_based': train[b].tolist()} for a, b in inner],
            'inner_scores': [{'model': c[0], 'C': c[1], 'balanced_accuracy': value} for c, value in scores.items()], 'selection': chosen})
        selected_all.append(scores)
    results = {family: {'metrics': measures(y, values), 'predictions': [{**row, 'truth': int(y[i]), 'prediction': int(values[i])} for i, row in enumerate(records)]} for family, values in predictions.items()}
    best = max(candidates, key=lambda c: np.mean([scores[c] for scores in selected_all]))
    deploy = model(best).fit(X, y)
    output = ROOT / f'artifacts/secondary-table-{name}-model.joblib'
    joblib.dump(deploy, output, compress=3)
    timings = []
    for iteration in range(55):
        started = time.perf_counter()
        feature = frontend(raw[0]) if frontend else X[0]
        deploy.predict(feature[None])
        if iteration >= 5:
            timings.append((time.perf_counter() - started) * 1000)
    return {'evaluation_status': 'exploratory_record_only_NOT_validated_unseen_acquisition_generalization',
        'split': '4fold stratified exact-content-hash-disjoint RECORD CV seed42, nested3fold training-only selection; content hashes prevent identical-row leakage but are NOT physical specimen/session IDs',
        'generalization_gate': 'BLOCKED: acquisition/specimen identities absent or undocumented; repeated samples could span train/test',
        'dimensions': X.shape[1], 'classes': {str(int(k)): int((y == k).sum()) for k in np.unique(y)}, 'results': results, 'folds': folds,
        'deployment_fit': {'candidate': best, 'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'path': str(output.relative_to(ROOT)), 'not_used_for_reported_predictions': True},
        'cpu_latency': {'unit': 'milliseconds per supplied feature row plus head' if frontend is None else 'milliseconds per1460point series including feature frontend and head',
            'p50': float(np.median(timings)), 'p95': float(np.quantile(timings, .95)), 'warmup': 5, 'repetitions': 50,
            'threads_max': 2, 'excludes': 'file decoding,capture wait; jackfruit acoustic frontend unavailable', 'iphone': 'not measured'}}


def gpla_frontend(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean()
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(x)))) ** 2
    total = max(spectrum.sum(), 1e-15)
    bands = np.array([part.sum() / total for part in np.array_split(spectrum, 32)])
    scale = max(np.std(x), 1e-12)
    z = centered / scale
    values = np.r_[np.mean(x), np.std(x), np.min(x), np.max(x), np.quantile(x, [.1, .25, .5, .75, .9]),
        np.mean(z ** 3), np.mean(z ** 4), np.mean(centered[:-1] * centered[1:] < 0), bands]
    assert len(values) == 44 and np.isfinite(values).all()
    return values


def benchmark():
    import platform
    import openpyxl
    import numpy as np
    import sklearn
    jack_path = BASE / 'jackfruit/jackfruit.xlsx'
    workbook = openpyxl.load_workbook(jack_path, read_only=True, data_only=True)
    assert len(workbook.worksheets) == 1
    data = list(workbook.active.values)
    workbook.close()
    header, entries = data[0], data[1:]
    assert header[:3] == ('Folder', 'Filename', 'Label') and len(header) == 30
    labels = sorted({row[2] for row in entries})
    X = np.asarray([row[3:] for row in entries], dtype=np.float64)
    y = np.array([labels.index(row[2]) for row in entries])
    rows = [{'record_index_zero_based': i, 'xlsx_row': i + 2, 'folder': row[0], 'filename': row[1],
        'feature_float64_sha256': hashlib.sha256(X[i].tobytes()).hexdigest()} for i, row in enumerate(entries)]
    jack = evaluate('jackfruit', X, y, rows)
    jack.update(source='https://github.com/Abhinandan-Hooli/jackfruit-acoustic-maturity-classification', label_mapping=dict(enumerate(labels)),
        schema=list(header), numeric_content='27precomputed acoustic features; no audio, no extractable capture rate',
        units='Frequency columns Hz, HNR_dB according column names; other units not documented',
        identity_audit='Folder encodes class and must not be used as predictor. Filename P1.1 etc suggests repeated takes but Standard recording filenames lack physical-fruit mapping; no invented specimen groups.',
        exact_duplicate_feature_rows=len(X) - len(np.unique(X, axis=0)), license='Repository MIT license; no independent acquisition or dataset-origin permission statement in checked dataset README')
    series = np.loadtxt(BASE / 'gpla/data.csv', delimiter=',')
    gy = np.loadtxt(BASE / 'gpla/label.csv', delimiter=',').astype(int)
    assert series.shape == (684, 1460) and gy.shape == (684,) and set(gy) == set(range(1, 13))
    gx = np.stack([gpla_frontend(row) for row in series])
    gr = [{'record_index_zero_based': i, 'csv_line_one_based': i + 1, 'series_float64_sha256': hashlib.sha256(row.tobytes()).hexdigest()} for i, row in enumerate(series)]
    gpla = evaluate('gpla-v1', gx, gy, gr, frontend=gpla_frontend, raw=series)
    gpla.update(source='https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12', paper='https://arxiv.org/html/2106.10277', edition='data_v1 only; v3placeholder not used',
        shape=list(series.shape), numeric_content='1460ordered preprocessed nonnegative time-series values per record, not standard PCM audio',
        features='12distribution/shape summaries +32normalized FFT energy bins in cycles/sample; no physical Hz or MFCC claim',
        units='Source acquisition numeric units not documented; paper says5second records preprocessed to1460values, not evidence of44100Hz PCM',
        label_mapping={i: {'microphone': 1 if i <= 6 else 2, 'pressure_MPa': [.2,.4,.5][((i - 1) % 6) // 2], 'noise': 'noiseless' if i % 2 else 'strongly_noisy'} for i in range(1, 13)},
        task_warning='12pressure/microphone/noise combinations among leakage signals; NOT leak-vs-no-leak or package seal integrity',
        identity_audit='CSV provides only signals and12class labels, no original run/session/pipeline identifiers. Cannot group independent acquisitions without inventing IDs.',
        exact_duplicate_series_rows=len(series) - len(np.unique(series, axis=0)), license='Repository EPL-2.0; review dataset rights separately before product use')
    blockers = [json.loads(path.read_text(encoding='utf-8')) for path in BASE.glob('watermelon-blocker-*.json')]
    watermelon = {'status': 'BLOCKED_download_requires_guestbook_response', 'source': 'https://doi.org/10.21950/OBYUCV', 'license': 'CC-BY-NC-SA-4.0',
        'metadata_only_claim': '158fruits,18measurements per fruit; acoustic spectral-band features and internal quality defects. Actual schema/IDs/labels cannot be confirmed without files.',
        'blockers': blockers, 'metrics': None, 'reason': 'Public API requires guestbookID2 response; no user identity or guestbook response invented/submitted. Metadata checksum advertised but downloaded contents cannot be verified.'}
    report = {'status': 'complete_with_explicit_access_and_generalization_blockers', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'download_manifest': MANIFEST, 'combined_download_bytes': sum(row['bytes'] for row in MANIFEST), 'cap_bytes': CAP,
        'platform': platform.platform(), 'processor': platform.processor(), 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'results': {'watermelon': watermelon, 'jackfruit': jack, 'gpla_v1': gpla},
        'limits': ['Record CV is an exploratory computation check, not unseen-fruit/session performance', 'No iPhone inference,Core ML export,or sensor acquisition measured', 'No universal best model claim; finite classifier comparison only']}
    path = ROOT / 'artifacts/secondary-table-benchmark.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(path), 'jackfruit': {k: v['metrics'] for k,v in jack['results'].items()}, 'gpla': {k: v['metrics'] for k,v in gpla['results'].items()}}, indent=2))


if __name__ == '__main__':
    acquire()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        benchmark()
