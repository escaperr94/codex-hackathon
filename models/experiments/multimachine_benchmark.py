"""Fixed normal-only acoustic ASD across four valve corpora and RoboticArm.

Run: python -m experiments.multimachine_benchmark [--acquire-only]
No test labels enter feature extraction, fitting, scorer selection or thresholds.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "2"

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import time
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ART / "datasets/multimachine"
CAP = 2_700_000_000
ARCHIVES = [
    ("DCASE2023dev", 7882613, "dev_valve.zip"),
    ("DCASE2025dev", 15097779, "dev_valve.zip"),
    ("MIMIIDG", 6529888, "valve.zip"),
    ("MIMIIDUE", 4740355, "dev_data_valve.zip"),
    ("DCASE2024eval", 11259435, "eval_data_RoboticArm_train_r2.zip"),
    ("DCASE2024eval", 11363076, "eval_data_RoboticArm_test.zip"),
]
EXTRA_ARCHIVES = [("DCASE2026dev", 19336329, "dev_valveEmu.zip"), ("DCASE2020dev", 3678171, "dev_data_valve.zip")]
EVALUATOR_COMMIT = "b5d17f63769bf4372a5412a7ede1f814c9590047"


def digest(path, kind="sha256"):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, kind).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def acquire(expanded=False):
    DATA.mkdir(parents=True, exist_ok=True)
    selected = []
    for corpus, record, name in (EXTRA_ARCHIVES if expanded else ARCHIVES):
        meta = json.loads((DATA / f"{record}-metadata.json").read_text(encoding="utf8"))
        entry = next(f for f in meta["files"] if f["key"] == name)
        selected.append((corpus, record, entry, meta))
    cap = 1_600_000_000 if expanded else CAP
    manifest_path = ART / ("multimachine-extra-downloads.json" if expanded else "multimachine-downloads.json")
    if expanded:
        import shutil
        assert shutil.disk_usage(DATA).free >= 12_000_000_000, "Approved expansion requires at least 12GB free"
    assert sum(x[2]["size"] for x in selected) <= cap
    manifest = []
    for corpus, record, entry, meta in selected:
        name = entry["key"]
        path = DATA / f"{record}-{name}"
        algorithm, expected = entry["checksum"].split(":")
        if not path.exists():
            partial = path.with_suffix(".partial")
            received = 0
            with urlopen(entry["links"]["self"], timeout=120) as response, partial.open("wb") as target:
                while chunk := response.read(1 << 20):
                    received += len(chunk)
                    if received > entry["size"]:
                        raise ValueError("Dataset exceeds API size; bounded download aborted")
                    target.write(chunk)
            assert received == entry["size"], (name, received, entry["size"])
            assert digest(partial, algorithm) == expected, "API checksum mismatch"
            partial.replace(path)
        assert path.stat().st_size == entry["size"] and digest(path, algorithm) == expected
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            wavs = [n for n in names if n.lower().endswith(".wav")]
            print(corpus, name, "WAVs", len(wavs), "examples", wavs[:2], wavs[-2:], flush=True)
        manifest.append({"corpus": corpus, "record": record, "archive": str(path.relative_to(ROOT)),
                         "url": entry["links"]["self"], "bytes": entry["size"],
                         "api_checksum": entry["checksum"], "sha256": digest(path),
                         "metadata_sha256": digest(DATA / f"{record}-metadata.json"),
                         "metadata_license": meta["metadata"].get("license"), "wav_count": len(wavs)})
        save(manifest_path, {"cap_bytes": cap, "selected_bytes": sum(x[2]["size"] for x in selected), "archives": manifest})
    return manifest


def parse_member(member):
    p = PurePosixPath(member)
    legacy = re.match(r"(normal|anomaly)_id_(\d+)_\d+\.wav$", p.name)
    if legacy:
        split = "train" if "train" in p.parts else "test"
        assert split in p.parts
        return {"section": legacy[2], "domain": "not_provided", "split": split, "label": int(legacy[1] == "anomaly")}
    match = re.match(r"section_(\d+)_(source|target)_(train|test)_(normal|anomaly)_", p.name)
    if match:
        section, domain, split, label = match.groups()
        assert split in p.parts or f"{domain}_{split}" in p.parts, member
        return {"section": section, "domain": domain, "split": split, "label": int(label == "anomaly")}
    match = re.match(r"section_(\d+)_(?:test_)?\d+\.wav$", p.name)
    if match:
        assert "test" in p.parts
        return {"section": match[1], "domain": None, "split": "test", "label": None}
    raise ValueError(f"Unrecognized audio member {member}")


def metric_values(y, scores, domains):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    result = {"pooled_auc": float(roc_auc_score(y, scores)),
              "pooled_standardized_pauc_fpr_0_1": float(roc_auc_score(y, scores, max_fpr=.1))}
    if set(domains) == {"not_provided"}:
        components = list(result.values())
        result["native_machine_id_auc_pauc_harmonic_mean"] = float(0 if min(components) == 0 else 2 / sum(1 / v for v in components))
        return result
    for domain in ("source", "target"):
        # Official domain AUC: domain's normals versus ALL section anomalies.
        mask = (domains == domain) | (y == 1)
        result[f"{domain}_auc_all_anomalies"] = float(roc_auc_score(y[mask], scores[mask]))
    components = [result["source_auc_all_anomalies"], result["target_auc_all_anomalies"], result["pooled_standardized_pauc_fpr_0_1"]]
    for domain in ("source", "target"):
        mask = domains == domain
        result[f"{domain}_only_auc"] = float(roc_auc_score(y[mask], scores[mask]))
        result[f"{domain}_only_standardized_pauc_fpr_0_1"] = float(roc_auc_score(y[mask], scores[mask], max_fpr=.1))
    result["three_component_harmonic_mean"] = float(0 if min(components) == 0 else len(components) / sum(1 / v for v in components))
    return result


def self_check():
    import numpy as np
    assert parse_member("valve/train/section_02_target_train_normal_0000.wav") == {"section": "02", "domain": "target", "split": "train", "label": 0}
    assert parse_member("RoboticArm/test/section_00_test_0001.wav")["label"] is None
    assert parse_member("valve/source_test/section_00_source_test_normal_0000.wav")["domain"] == "source"
    y = np.array([0, 0, 1, 1]); d = np.array(["source", "target", "source", "target"])
    result = metric_values(y, np.array([0., 1., 2., 3.]), d)
    assert all(v == 1 for v in result.values())
    # Removing target anomalies would incorrectly make this source AUC perfect.
    assert metric_values(y, np.array([1., 0., 2., .5]), d)["source_auc_all_anomalies"] == .5


def extract(archives, bundle):
    import librosa
    import numpy as np
    import torch
    from thud.efficientat import efficientat_wav_embedding
    groups = {}
    excluded = []
    for item in archives:
        with zipfile.ZipFile(ROOT / item["archive"]) as archive:
            for member in sorted(archive.namelist()):
                if not member.lower().endswith(".wav"):
                    continue
                if "supplemental" in PurePosixPath(member).parts:
                    excluded.append({"corpus": item["corpus"], "archive": item["archive"], "member": member, "wav_sha256": hashlib.sha256(archive.read(member)).hexdigest(), "reason": "Optional clean machine/noise supplementary audio excluded to keep fixed train-normal protocol identical across corpora"})
                    continue
                row = {**parse_member(member), "member": member, "archive": item["archive"], "corpus": item["corpus"], "wav_sha256": hashlib.sha256(archive.read(member)).hexdigest()}
                groups.setdefault((item["corpus"], row["section"]), []).append(row)
    suffix = "extra-" if any(a["corpus"] in ("DCASE2020dev", "DCASE2026dev") for a in archives) else ""
    save(ART / f"multimachine-{suffix}supplementary-excluded.json", excluded)
    caches = {}
    timing = []
    frontend_identity = [(str(p.relative_to(ART / "efficientat-source")), digest(p)) for p in sorted((ART / "efficientat-source").rglob("*.py"))]
    temporary = DATA / "current-audio.wav"
    try:
        for (corpus, section), rows in sorted(groups.items()):
            cache = ART / f"multimachine-{corpus}-section{section}.npz"
            manifest_path = ART / f"multimachine-{corpus}-section{section}-manifest.json"
            signature = hashlib.sha256(json.dumps([[(r["archive"], r["member"], r["wav_sha256"]) for r in rows], [(a["archive"], a["sha256"]) for a in archives if a["archive"] in {r["archive"] for r in rows}], bundle.source_commit, bundle.checkpoint_sha256, frontend_identity, "official128mel_meanstd_v1_near_ch0" if corpus == "DCASE2026dev" else "official128mel_meanstd_v1"]).encode()).hexdigest()
            if cache.exists() and manifest_path.exists():
                with np.load(cache, allow_pickle=False) as saved:
                    assert str(saved["signature"]) == signature
                caches[(corpus, section)] = (cache, json.loads(manifest_path.read_text(encoding="utf8")))
                continue
            vectors, mels = [], []
            handles = {a: zipfile.ZipFile(ROOT / a) for a in set(r["archive"] for r in rows)}
            started = time.perf_counter()
            try:
                for i, row in enumerate(rows):
                    content = handles[row["archive"]].read(row["member"])
                    row["wav_sha256"] = hashlib.sha256(content).hexdigest()
                    temporary.write_bytes(content)
                    if corpus == "DCASE2026dev":
                        import soundfile as sf
                        stereo, native_sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
                        assert stereo.shape[1] == 2, "2026 protocol requires synchronized near/far channels"
                        row["input_channels"] = 2
                        row["selected_channel"] = "near microphone, channel 0; far noise-reference channel deliberately unused"
                        sf.write(temporary, stereo[:, 0], native_sr, subtype="FLOAT")
                        content = temporary.read_bytes()
                    t = time.perf_counter()
                    embedding = efficientat_wav_embedding(bundle, temporary)
                    elapsed = time.perf_counter() - t
                    if len(timing) < 8:
                        timing.append({"corpus": corpus, "member": row["member"], "seconds_decode_resample_frontend_model": elapsed, "warmup": len(timing) == 0})
                    waveform, sr = librosa.load(io.BytesIO(content), sr=32000, mono=True)
                    assert sr == 32000 and np.isfinite(waveform).all()
                    row["duration_seconds"] = len(waveform) / sr
                    with torch.inference_mode():
                        mel = bundle.frontend(torch.from_numpy(waveform)[None, :])[0].numpy()
                    vectors.append(embedding)
                    mels.append(np.r_[mel.mean(axis=1), mel.std(axis=1)])
                    if (i + 1) % 100 == 0:
                        print(corpus, section, i + 1, "/", len(rows), flush=True)
            finally:
                for handle in handles.values():
                    handle.close()
            np.savez_compressed(cache, signature=signature, efficientat=np.stack(vectors), logmel=np.stack(mels))
            save(manifest_path, rows)
            caches[(corpus, section)] = (cache, rows)
            print("Extracted", corpus, section, time.perf_counter() - started, "seconds", flush=True)
        if timing:
            save(ART / "multimachine-cpu-timing.json", {"platform": platform.platform(), "processor": platform.processor(), "torch_threads": torch.get_num_threads(), "fixed_first_eight_clips": timing, "not_phone": True})
    finally:
        temporary.unlink(missing_ok=True)
    return caches


def official_labels(rows):
    """Join by full filename, never row position or guessed label order."""
    base = DATA / "evaluator"
    name = "ground_truth_RoboticArm_section_00_test.csv"
    def table(folder):
        with (base / folder / name).open(newline="", encoding="utf8") as f:
            result = {r[0]: int(r[1]) for r in csv.reader(f)}
        assert len(result) == 200
        return result
    labels, domains = table("ground_truth_data"), table("ground_truth_domain")
    tests = [r for r in rows if r["split"] == "test"]
    assert {PurePosixPath(r["member"]).name for r in tests} == set(labels) == set(domains)
    for r in tests:
        filename = PurePosixPath(r["member"]).name
        r["label"] = labels[filename]
        r["domain"] = {0: "source", 1: "target"}[domains[filename]]
    return {"repository": "https://github.com/nttcslab/dcase2024_task2_evaluator", "commit": EVALUATOR_COMMIT,
            "label_sha256": digest(base / "ground_truth_data" / name), "domain_sha256": digest(base / "ground_truth_domain" / name),
            "metric_equations": "Same masks and sklearn roc_auc_score(max_fpr=.1) as released evaluator lines 250-280; no arbitrary threshold decisions emitted"}


def run_released_evaluator(rows, scores, expected, thresholds):
    """Execute upstream evaluator unchanged on this one complete machine."""
    import importlib.util
    import numpy as np
    base = DATA / "evaluator"
    tree = json.loads((DATA / "evaluator-tree.json").read_text(encoding="utf8"))
    assert tree["sha"] == EVALUATOR_COMMIT
    for name in ("dcase2024_task2_evaluator.py", "tools/plot_anm_score.py", "tools/test_plots.py", "ground_truth_data/ground_truth_RoboticArm_section_00_test.csv", "ground_truth_domain/ground_truth_RoboticArm_section_00_test.csv", "ground_truth_attributes/ground_truth_RoboticArm_section_00_test.csv"):
        content = (base / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        assert blob == next(x["sha"] for x in tree["tree"] if x["path"] == name)
    cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(base)
        sys.path.insert(0, str(base))
        spec = importlib.util.spec_from_file_location("official_dcase2024_eval", base / "dcase2024_task2_evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = {}
        for method, values in scores.items():
            folder = base / "teams" / method
            folder.mkdir(parents=True, exist_ok=True)
            for prefix, sequence in (("anomaly_score", values), ("decision_result", (values > thresholds[method]).astype(int))):
                with (folder / f"{prefix}_RoboticArm_section_00_test.csv").open("w", newline="") as f:
                    csv.writer(f).writerows((PurePosixPath(r["member"]).name, float(value)) for r, value in zip(rows, sequence))
            _, table, _, _, _ = module.output_result(str(folder), ["RoboticArm"], ["section_00"], str(base / "results"), str(base / "additional-results"), 0, "", out_all=False)
            actual = float(table.loc[method, "official score"])
            assert abs(actual - expected[method]["three_component_harmonic_mean"]) < 1e-12
            result[method] = {"single_machine_official_evaluator_score": actual, "decision_threshold_normal_train_quantile_0_95": thresholds[method], "decision_metrics": "Not selected or used in comparison; upstream interface requires decisions. Normal-only 95th-percentile threshold, no test tuning.", "result_csv": str((base / "results" / f"{method}_result.csv").relative_to(ROOT))}
        return result
    finally:
        os.chdir(cwd)
        sys.path[:] = old_path


def run(archives, bundle, expanded=False):
    import numpy as np
    from sklearn.covariance import LedoitWolf
    from sklearn.neighbors import NearestNeighbors
    caches = extract(archives, bundle)
    report_path = ART / ("multimachine-extra-benchmark.json" if expanded else "multimachine-benchmark.json")
    report = {"protocol": "Fixed frozen mn04_as + cosine1NN; official-frontend logmel mean/std + LedoitWolf Gaussian Mahalanobis. Fit both source and target TRAIN NORMALS in each section independently. No test labels or test feature statistics enter fit; no cross-corpus tuning or thresholding.",
              "script_sha256": digest(__file__), "model_sha256": bundle.checkpoint_sha256,
              "model_bytes": (ART / "mn04_as_mAP_432.pt").stat().st_size,
              "source_commit": bundle.source_commit, "archives": archives, "sections": {}, "corpora": {},
              "limitations": ["Valve only in each of four corpora; RoboticArm only for paired 2024 additional/evaluation. Not all-machine evaluation or official challenge submission.", "Cross-year overlap is possible; these are not four fresh independent replications. MIMII DG is explicitly a DCASE2022 subset and DUE a DCASE2021 subset.", "No THUD tile, physical iPhone, mobile export, energy or phone latency measured.", "MIMII DG and DCASE2023 metadata say CC BY 4.0 but description conditions say CC BY-NC-SA 4.0: contradictions unresolved; use restrictive interpretation and seek permission.", "AudioSet-pretrained EfficientAT resource challenge eligibility not independently established; research comparison, not official leaderboard."]}
    if expanded:
        report["scope_note"] = "DCASE2020 complete valve archive, separate machine IDs without source/target labels; DCASE2026 complete ordinary valveEmu split using near microphone channel0 only, far synchronized noise-reference channel unused. Not a noise-aware/two-channel competition system."
        report["limitations"][0] = "Valve only for2020 and valveEmu only for2026; near channel alone, not all machines or a two-channel noise-aware system."
    for (corpus, section), (cache, rows) in sorted(caches.items()):
        train = np.array([r["split"] == "train" for r in rows])
        assert train.any() and (~train).any() and all(r["label"] == 0 for r in rows if r["split"] == "train")
        result = {"counts_train_normal_by_domain": dict(Counter(r["domain"] for r in rows if r["split"] == "train")), "train_count": int(train.sum()), "test_count": int((~train).sum()), "manifest": str((ART / f"multimachine-{corpus}-section{section}-manifest.json").relative_to(ROOT)), "manifest_sha256": digest(ART / f"multimachine-{corpus}-section{section}-manifest.json"), "features_sha256": digest(cache), "models": {}}
        scores = {}
        thresholds = {}
        with np.load(cache, allow_pickle=False) as features:
            for feature, method in (("efficientat", "cosine1NN"), ("logmel", "Gaussian_LedoitWolf")):
                x = features[feature].astype(np.float64)
                assert np.isfinite(x).all()
                start = time.perf_counter()
                if method == "cosine1NN":
                    scorer = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=1).fit(x[train])
                    normal_scores = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=1).fit(x[train]).kneighbors(x[train])[0][:, 1]
                    score = scorer.kneighbors(x[~train])[0][:, 0]
                else:
                    scorer = LedoitWolf().fit(x[train])
                    score = scorer.mahalanobis(x[~train])
                    normal_scores = scorer.mahalanobis(x[train])
                assert np.isfinite(score).all()
                scores[method] = score
                thresholds[method] = float(np.quantile(normal_scores, .95))
                result["models"][method] = {"dimensions": x.shape[1], "fit_score_seconds": time.perf_counter() - start}
                result["models"][method]["normal_reference_bytes_float32" if method == "cosine1NN" else "gaussian_location_covariance_precision_bytes_float64"] = int(x[train].size * 4 if method == "cosine1NN" else scorer.location_.nbytes + scorer.covariance_.nbytes + scorer.precision_.nbytes)
        # External withheld labels are loaded only after both scoring paths finish.
        if corpus == "DCASE2024eval":
            report["released_2024_label_provenance"] = official_labels(rows)
        tests = [r for r in rows if r["split"] == "test"]
        y = np.array([r["label"] for r in tests], dtype=int)
        domains = np.array([r["domain"] for r in tests])
        result["test_counts_domain_label"] = dict(Counter(f"{r['domain']}_{r['label']}" for r in tests))
        raw = [{**r, **{method: float(score[i]) for method, score in scores.items()}} for i, r in enumerate(tests)]
        raw_path = ART / f"multimachine-{corpus}-section{section}-scores.json"
        save(raw_path, raw)
        result["raw_scores"] = str(raw_path.relative_to(ROOT)); result["raw_scores_sha256"] = digest(raw_path)
        for method, score in scores.items():
            result["models"][method].update(metric_values(y, score, domains))
            if corpus == "MIMIIDUE":
                native = [result["models"][method][f"{domain}_{suffix}"] for domain in ("source", "target") for suffix in ("only_auc", "only_standardized_pauc_fpr_0_1")]
                result["models"][method]["DCASE2021_native_domain_auc_pauc_harmonic_mean"] = float(0 if min(native) == 0 else len(native) / sum(1 / v for v in native))
        if corpus == "DCASE2024eval":
            result["released_evaluator_execution"] = run_released_evaluator(tests, scores, result["models"], thresholds)
        report["sections"][f"{corpus}/section{section}"] = result
        print(corpus, section, result["models"], flush=True)
        save(report_path, report)
    for corpus in sorted(set(k[0] for k in caches)):
        sections = [v for k, v in report["sections"].items() if k.startswith(corpus + "/")]
        report["corpora"][corpus] = {"section_count": len(sections), "train_normal_count": sum(s["train_count"] for s in sections), "test_count": sum(s["test_count"] for s in sections), "models": {}}
        for method in ("cosine1NN", "Gaussian_LedoitWolf"):
            keys = ("pooled_auc", "pooled_standardized_pauc_fpr_0_1") if corpus == "DCASE2020dev" else ("source_auc_all_anomalies", "target_auc_all_anomalies", "pooled_standardized_pauc_fpr_0_1")
            vals = [s["models"][method][key] for s in sections for key in keys]
            report["corpora"][corpus]["models"][method] = {"harmonic_mean_native_machine_id_auc_pauc" if corpus == "DCASE2020dev" else "harmonic_mean_over_section_domain_auc_and_section_pauc": float(0 if min(vals) == 0 else len(vals) / sum(1 / v for v in vals)), "arithmetic_mean_section_pooled_auc": float(np.mean([s["models"][method]["pooled_auc"] for s in sections])), "arithmetic_mean_section_pauc": float(np.mean([s["models"][method]["pooled_standardized_pauc_fpr_0_1"] for s in sections]))}
            if corpus == "MIMIIDUE":
                native = [s["models"][method][f"{domain}_{suffix}"] for s in sections for domain in ("source", "target") for suffix in ("only_auc", "only_standardized_pauc_fpr_0_1")]
                report["corpora"][corpus]["models"][method]["DCASE2021_native_domain_auc_pauc_harmonic_mean"] = float(0 if min(native) == 0 else len(native) / sum(1 / v for v in native))
                report["corpora"][corpus]["metric_note"] = "Primary cross-corpus comparison uses 2022+ all-anomaly domain masks. Native 2021 metrics instead compare normals/anomalies within domain and include domain pAUC; reported separately, not called the same official score."
    identities = {}
    for (corpus, section), (_, rows) in caches.items():
        for row in rows:
            identities.setdefault(row["wav_sha256"], []).append((corpus, section, row["split"]))
    overlaps = Counter()
    for occurrences in identities.values():
        corpora = sorted(set(r[0] for r in occurrences))
        for i, left in enumerate(corpora):
            for right in corpora[i + 1:]:
                overlaps[f"{left} / {right}"] += 1
    report["exact_wav_hash_overlap_between_corpora"] = dict(overlaps)
    report["overlap_interpretation"] = "Counts unique identical WAV bytes across selected corpora; absence of byte overlap does not establish independent physical recordings. Each section is fit separately, never pooled cross-year."
    save(report_path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--available-only", action="store_true", help="Process only already checksum-verified archives in saved acquisition manifest")
    parser.add_argument("--expanded", action="store_true", help="Approved 2020/2026 bounded valve expansion")
    args = parser.parse_args()
    manifest = ART / ("multimachine-extra-downloads.json" if args.expanded else "multimachine-downloads.json")
    archives = json.loads(manifest.read_text(encoding="utf8"))["archives"] if args.available_only else acquire(args.expanded)
    if args.acquire_only:
        return
    self_check()
    import torch
    from thud.efficientat import load_efficientat
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    bundle = load_efficientat(ART / "mn04_as_mAP_432.pt", ART / "efficientat-source")
    run(archives, bundle, args.expanded)


if __name__ == "__main__":
    main()
