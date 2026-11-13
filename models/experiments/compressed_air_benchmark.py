"""Range-only IICA subset benchmark; never downloads the full 22 GB release.

Run: python experiments/compressed_air_benchmark.py
Outputs are confined to artifacts/compressed-air-* and datasets/compressed-air/.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

import hashlib
import io
import json
from pathlib import Path
import platform
import re
import struct
import time
import urllib.request
import zipfile
import zlib

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits
from dataset_benchmark import feature, grouped_cv, edge_probe

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "datasets" / "compressed-air"
PREFIX = ROOT / "artifacts" / "compressed-air-"
API = "https://zenodo.org/api/records/7551606"
CAP = 300_000_000
# Include the successful exploratory range bodies before this executable was saved.
PRIOR_BYTES = 70_000 + 194_461 + 30 + 855


def save(name, value):
    Path(str(PREFIX) + name + ".json").write_text(json.dumps(value, indent=2), encoding="utf-8")


class BoundedHTTP:
    def __init__(self):
        self.path = Path(str(PREFIX) + "access.json")
        if self.path.exists():
            self.evidence = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.evidence = {"hard_cap_bytes": CAP, "prior_exploratory_body_bytes": PRIOR_BYTES,
                             "body_bytes": PRIOR_BYTES, "requests": [],
                             "policy": "Validate 206 and exact Content-Range before reading archive bytes; reject 200 without reading."}

    def get(self, url, start=None, end=None, size=None):
        maximum = 1_000_000 if start is None else end - start + 1
        if self.evidence["body_bytes"] + maximum > CAP:
            raise RuntimeError("300 MB cumulative transfer cap would be exceeded")
        headers = {"Accept-Encoding": "identity", "User-Agent": "THUD-bounded-research/1.0"}
        if start is not None:
            headers["Range"] = f"bytes={start}-{end}"
        item = {"url": url, "range": headers.get("Range"), "body_bytes": 0}
        self.evidence["requests"].append(item)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                item.update(status=response.status, final_url=response.url,
                            headers=dict(response.headers.items()))
                if start is not None:
                    expected = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Unsafe range response: {response.status}, {response.headers.get('Content-Range')}; closed unread")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise RuntimeError("Unexpected encoded response")
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > maximum:
                    raise RuntimeError("Response Content-Length exceeds bounded request")
                pieces = []
                while item["body_bytes"] < maximum:
                    chunk = response.read(min(65536, maximum - item["body_bytes"]))
                    if not chunk:
                        break
                    pieces.append(chunk)
                    item["body_bytes"] += len(chunk)
                    self.evidence["body_bytes"] += len(chunk)
                body = b"".join(pieces)
                if start is not None and len(body) != maximum:
                    raise RuntimeError(f"Truncated range body: {len(body)} != {maximum}")
                return body
        except Exception as error:
            item["error"] = str(error)
            raise
        finally:
            save("access", self.evidence)


def directory(data):
    rows, position = [], 0
    while position < len(data):
        fields = struct.unpack_from("<4s6H3I5H2I", data, position)
        if fields[0] != b"PK\x01\x02":
            raise ValueError("Invalid central directory signature")
        n, e, c = fields[10:13]
        name = data[position + 46:position + 46 + n].decode("utf-8")
        extra = data[position + 46 + n:position + 46 + n + e]
        values = {"uncompressed_size": fields[9], "compressed_size": fields[8], "offset": fields[16]}
        p = 0
        while p < len(extra):
            kind, length = struct.unpack_from("<HH", extra, p)
            value = extra[p + 4:p + 4 + length]
            p += 4 + length
            if kind == 1:
                q = 0
                for key in ("uncompressed_size", "compressed_size", "offset"):
                    if values[key] == 0xffffffff:
                        values[key] = struct.unpack_from("<Q", value, q)[0]
                        q += 8
        if any(value == 0xffffffff for value in values.values()):
            raise ValueError("Unresolved ZIP64 value")
        rows.append(dict(name=name, method=fields[4], flags=fields[3], crc32=fields[7], **values))
        position += 46 + n + e + c
    assert position == len(data)
    return rows


class RemoteZip:
    def __init__(self, http, metadata):
        self.http, self.meta = http, metadata
        self.size, self.url = metadata["size"], metadata["links"]["self"]
        tail_offset = max(0, self.size - 70000)
        tail = self.range(tail_offset, self.size - 1)
        end = tail.rfind(b"PK\x05\x06")
        if end < 0:
            raise ValueError("Missing ZIP end record")
        record = struct.unpack_from("<4s4H2IH", tail, end)
        if record[1] or record[2]:
            raise ValueError("Multi-disk ZIP unsupported")
        cd_size, cd_offset = record[5:7]
        if cd_offset == 0xffffffff or cd_size == 0xffffffff:
            locator = struct.unpack_from("<4sIQI", tail, end - 20)
            assert locator[0] == b"PK\x06\x07" and locator[1] == 0 and locator[3] == 1
            zip64_offset = locator[2]
            local = zip64_offset - tail_offset
            z = tail[local:local + 56] if local >= 0 else self.range(zip64_offset, zip64_offset + 55)
            extended = struct.unpack("<4sQHHIIQQQQ", z)
            assert extended[0] == b"PK\x06\x06" and extended[4] == extended[5] == 0
            cd_size, cd_offset = extended[-2:]
        if cd_size > 4_000_000:
            raise ValueError("Unexpectedly large central directory")
        self.entries = directory(self.range(cd_offset, cd_offset + cd_size - 1))

    def range(self, start, end):
        if not 0 <= start <= end < self.size:
            raise ValueError("ZIP range outside source")
        return self.http.get(self.url, start, end, self.size)

    def member(self, entry):
        offset = entry["offset"]
        header = self.range(offset, offset + 29)
        local = struct.unpack("<4s5H3I2H", header)
        assert local[0] == b"PK\x03\x04" and local[3] == entry["method"]
        if local[2] & 1:
            raise ValueError("Encrypted member unsupported")
        beginning = offset + 30 + local[-2] + local[-1]
        compressed = self.range(beginning, beginning + entry["compressed_size"] - 1)
        if entry["method"] == 8:
            payload = zlib.decompress(compressed, -15)
        elif entry["method"] == 0:
            payload = compressed
        else:
            raise ValueError("Unsupported ZIP compression method")
        if len(payload) != entry["uncompressed_size"] or zlib.crc32(payload) != entry["crc32"]:
            raise ValueError("ZIP member size/CRC mismatch")
        return payload


def parser_selfcheck():
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("x/example.txt", b"CRC and directory self-check")
    blob = target.getvalue()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        info = archive.infolist()[0]
        rows = directory(blob[archive.start_dir:blob.rfind(b"PK\x05\x06")])
        assert rows[0]["crc32"] == info.CRC and rows[0]["offset"] == info.header_offset
        assert rows[0]["compressed_size"] == info.compress_size
        assert rows[0]["uncompressed_size"] == info.file_size


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    parser_selfcheck()
    http = BoundedHTTP()
    metadata_path = BASE / "zenodo-7551606.json"
    if not metadata_path.exists():
        metadata_path.write_bytes(http.get(API))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    save("sources", {"api": API, "license": metadata["metadata"]["license"], "archives": metadata["files"],
                     "source_expected_md5_verified": False,
                     "integrity": "Full source MD5 values are metadata only; only selected decompressed member CRC32 and local SHA256 are verified."})
    manifest = []
    for archive_meta in sorted(metadata["files"], key=lambda x: x["key"]):
        remote = RemoteZip(http, archive_meta)
        tag = archive_meta["key"].removesuffix(".zip")
        (BASE / (tag + "-directory.json")).write_text(json.dumps(remote.entries, indent=2), encoding="utf-8")
        readme = next(x for x in remote.entries if x["name"].lower().endswith("readme.md"))
        readme_path = BASE / (tag + "-README.txt")
        if not readme_path.exists():
            readme_path.write_bytes(remote.member(readme))
        assert zlib.crc32(readme_path.read_bytes()) == readme["crc32"]
        assert "_iO_ = no leak present" in readme_path.read_text(encoding="utf-8")
        # Balanced deterministic pilot: 2 sessions x 2 noises x 2 mics x 2 classes per leak type.
        for session in ("1", "2"):
            for background in ("lab", "work"):
                for mic in ("1m", "2m"):
                    for label_token, label in (("iO", 0), ("niO", 1)):
                        candidates = []
                        for entry in remote.entries:
                            parts = entry["name"].split("/")
                            match = re.fullmatch(r"([123])_(iO|niO)_([0-9.]+)n_([1-4][ml])_\.wav", parts[-1])
                            if len(parts) == 4 and parts[1:3] == [background, session] and match:
                                if match[1] == session and match[2] == label_token and match[4] == mic:
                                    candidates.append((float(match[3]), entry))
                        if not candidates:
                            raise RuntimeError(f"No labelled candidate: {tag}/{session}/{background}/{mic}/{label_token}")
                        candidates.sort(key=lambda x: (x[0], x[1]["name"]))
                        knob, selected = candidates[len(candidates) // 2]
                        target = BASE / selected["name"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            target.write_bytes(remote.member(selected))
                        payload = target.read_bytes()
                        if len(payload) != selected["uncompressed_size"] or zlib.crc32(payload) != selected["crc32"]:
                            raise ValueError("Cached member integrity failure")
                        audio_info = sf.info(str(target))
                        manifest.append({"path": str(target.relative_to(ROOT)), "member": selected["name"],
                                         "archive": tag, "source_url": remote.url, "source_expected_md5": archive_meta["checksum"],
                                         "source_full_md5_verified": False, "member_crc32_verified": True,
                                         "crc32": f"{selected['crc32']:08x}", "sha256": hashlib.sha256(payload).hexdigest(),
                                         "label": label, "label_name": "leak" if label else "no_leak",
                                         "session": session, "background": background, "mic_configuration": mic,
                                         "knob_rotations": knob, "duration_seconds": audio_info.duration,
                                         "samplerate": audio_info.samplerate, "channels": audio_info.channels,
                                         "uncompressed_bytes": len(payload), "compressed_bytes": selected["compressed_size"]})
                        save("manifest", manifest)
    assert len(manifest) == 48 and sum(x["label"] for x in manifest) == 24
    with threadpool_limits(limits=2):
        rows = [dict(row, path=ROOT / row["path"]) for row in manifest]
        started = time.perf_counter()
        features = np.stack([feature(row["path"]) for row in rows])
        extraction_ms = (time.perf_counter() - started) * 1000
        labels = np.array([row["label"] for row in rows])
        sessions = np.array([row["session"] for row in rows])
        microphones = np.array([row["mic_configuration"] for row in rows])
        np.savez_compressed(str(PREFIX) + "features.npz", X=features, y=labels, session=sessions, microphone=microphones)
        results = {"status": "complete", "dataset": metadata["metadata"]["title"], "record": API,
                   "task": "audible_compressed_air_leak_vs_no_leak", "label_evidence": "Archive README: iO=no leak, niO=leak",
                   "sampling": "48 whole recordings: all 3 leak-type archives, sessions 1/2, lab/work backgrounds, mic configurations 1m/2m, both labels. Choose median knob setting within each stratum, without listening or model scores.",
                   "scope_limit": "Only a small stratified subset; session 3, hydr/hydr_low/work_low, microphones 3/4 and other knob settings are not scored.",
                   "features": "90 dimensions: first 4 seconds resampled 48->16 kHz; 13 MFCC mean/std + 32 logmel mean/std. No recording/window random split.",
                   "protocol": "Separate leave-one-session-out and leave-one-microphone-configuration-out; complete recording per sample; same numeric session held out jointly across leak-type archives/backgrounds/mics; scaler fitted in each training fold. C=1 logistic/RBF and gamma=scale fixed, no test tuning; duplicate hashes removed from training.",
                   "counts": {"recordings": len(rows), "leak": int(labels.sum()), "no_leak": int((labels == 0).sum()),
                              "source_archive_types": 3, "sessions": 2, "microphone_configurations": 2, "backgrounds": 2},
                   "feature_extraction_total_ms": extraction_ms,
                   "session_disjoint": {method: grouped_cv(rows, features, labels, sessions, method) for method in ("logistic", "rbf")},
                   "microphone_disjoint": {method: grouped_cv(rows, features, labels, microphones, method) for method in ("logistic", "rbf")},
                   "cpu_feature_proxy": edge_probe(rows[0]["path"]),
                   "environment": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(), "max_cpu_threads": 2},
                   "transfer_body_bytes_including_exploration": http.evidence["body_bytes"],
                   "integrity": "Per-member CRC32 verified; local SHA256 manifested; complete source archive MD5 NOT verified (partial access).",
                   "caveats": ["Industrial measurement microphones, not phones; audible flowing leak, not seal/vacuum tap diagnosis.",
                               "Two sessions and two mic configurations only; no independent deployment test or hyperparameter search.",
                               "32-bit source converted by soundfile; features intentionally ignore audio after 4 seconds.",
                               "Runtime is desktop CPU, not iPhone/Core ML; no mobile conversion, latency, memory or energy measured.",
                               "CC BY-NC-ND 4.0 dataset; commercial/redistribution/derived-artifact permission requires review."]}
        save("benchmark", results)
        print(json.dumps({"status": results["status"], "counts": results["counts"], "bytes": http.evidence["body_bytes"],
                          "session": {key: {m: val[m] for m in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")} for key, val in results["session_disjoint"].items()},
                          "mic": {key: {m: val[m] for m in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")} for key, val in results["microphone_disjoint"].items()},
                          "feature_cpu": results["cpu_feature_proxy"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("failure", {"status": "blocked", "error": repr(exc),
                         "full_archives_downloaded": False, "phone_execution": "not measured"})
        raise
