"""Finite research-head finalization; never writes historical reports or downloads data.
Run from models/: python -m experiments.finalize_classifiers
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[_key] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import hashlib
import io
import json
from pathlib import Path
import platform
import warnings
import zipfile
import joblib
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from threadpoolctl import threadpool_limits
from experiments import dataset_benchmark as db
from experiments import material_benchmark as mb
from experiments import puck_motor_benchmark as pm
from experiments import secondary_table_benchmark as sb
from experiments import impact_echo_benchmark as ie

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "final/classifiers"
DOC = ROOT / "docs/CLASSIFIER_FINALIZATION.md"
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fit_linear(X, y, C=1.):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=42)).fit(X, y)


def fit_rbf(X, y, C=1.):
    return make_pipeline(StandardScaler(), SVC(C=C, gamma="scale", class_weight="balanced")).fit(X, y)


class Scorer(tf.Module):
    """Exact sklearn score semantics, including libsvm OVO votes (not OVR argmax)."""
    def __init__(self, model):
        super().__init__()
        scale, head = model[0], model[-1]
        self.mean = tf.constant(scale.mean_, tf.float32)
        self.scale = tf.constant(scale.scale_, tf.float32)
        self.head = head
        self.nclass = len(head.classes_)
        self.kind = type(head).__name__
        if isinstance(head, SVC):
            assert head.kernel in ("linear", "rbf") and not head.break_ties
            self.sv = tf.constant(head.support_vectors_, tf.float32)
            self.dual = np.asarray(head.dual_coef_, np.float32)
            self.intercepts = np.asarray(head.intercept_, np.float32)
            self.offset = np.r_[0, np.cumsum(head.n_support_)]
        elif isinstance(head, MLPClassifier):
            assert head.activation == "tanh" and head.out_activation_ in ("softmax", "logistic")
            self.weights = [tf.constant(w, tf.float32) for w in head.coefs_]
            self.biases = [tf.constant(b, tf.float32) for b in head.intercepts_]
        else:
            assert isinstance(head, (LogisticRegression, LinearSVC))
            self.weights = tf.constant(head.coef_.T, tf.float32)
            self.biases = tf.constant(head.intercept_, tf.float32)

    @tf.function
    def __call__(self, features):
        z = (features - self.mean) / self.scale
        if isinstance(self.head, SVC):
            if self.head.kernel == "rbf":
                # Direct differences avoid catastrophic cancellation at support vectors.
                kernel = tf.exp(-tf.constant(self.head._gamma, tf.float32) * tf.reduce_sum(tf.square(z[:, None, :] - self.sv[None, :, :]), axis=2))
            else:
                kernel = tf.matmul(z, self.sv, transpose_b=True)
            if self.nclass == 2:
                margin = tf.matmul(kernel, tf.constant(self.dual.T)) + self.intercepts
                scores = tf.concat([-margin, margin], axis=1)
                index = tf.cast(margin[:, 0] >= 0., tf.int32)
            else:
                votes = [tf.zeros_like(z[:, 0]) for _ in range(self.nclass)]
                confidence = [tf.zeros_like(z[:, 0]) for _ in range(self.nclass)]
                pair = 0
                for i in range(self.nclass):
                    for j in range(i + 1, self.nclass):
                        ai, bi = self.offset[i:i+2]
                        aj, bj = self.offset[j:j+2]
                        value = tf.reduce_sum(kernel[:, ai:bi] * self.dual[j-1, ai:bi], axis=1) + tf.reduce_sum(kernel[:, aj:bj] * self.dual[i, aj:bj], axis=1) + self.intercepts[pair]
                        votes[i] = votes[i] + tf.cast(value > 0., tf.float32)
                        votes[j] = votes[j] + tf.cast(value <= 0., tf.float32)
                        confidence[i] = confidence[i] + value
                        confidence[j] = confidence[j] - value
                        pair += 1
                vote = tf.stack(votes, axis=1)
                conf = tf.stack(confidence, axis=1)
                scores = vote + conf / (3. * (tf.abs(conf) + 1.))
                index = tf.argmax(vote, axis=1, output_type=tf.int32)
        elif isinstance(self.head, MLPClassifier):
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                z = tf.tanh(tf.matmul(z, w) + b)
            logits = tf.matmul(z, self.weights[-1]) + self.biases[-1]
            if self.nclass == 2:
                p = tf.sigmoid(logits)
                scores = tf.concat([1. - p, p], axis=1)
            else:
                scores = tf.nn.softmax(logits)
            index = tf.argmax(scores, axis=1, output_type=tf.int32)
        else:
            logits = tf.matmul(z, self.weights) + self.biases
            if isinstance(self.head, LogisticRegression):
                if self.nclass == 2:
                    p = tf.sigmoid(logits)
                    scores = tf.concat([1. - p, p], axis=1)
                    index = tf.cast(logits[:, 0] > 0., tf.int32)
                else:
                    scores = tf.nn.softmax(logits)
                    index = tf.argmax(logits, axis=1, output_type=tf.int32)
            elif self.nclass == 2:
                scores = tf.concat([-logits, logits], axis=1)
                index = tf.cast(logits[:, 0] > 0., tf.int32)
            else:
                scores = logits
                index = tf.argmax(logits, axis=1, output_type=tf.int32)
        return {"scores": scores, "class_index": index}


def reference_scores(model, X):
    if isinstance(model[-1], (LogisticRegression, MLPClassifier)):
        return model.predict_proba(X)
    score = model.decision_function(X)
    return np.column_stack([-score, score]) if score.ndim == 1 else score


def convert(model):
    module = Scorer(model)
    function = module.__call__.get_concrete_function(tf.TensorSpec([1, model.n_features_in_], tf.float32, name="features"))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([function], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def runtime(content):
    interpreter = tf.lite.Interpreter(model_content=content, num_threads=2)
    interpreter.allocate_tensors()
    ops = sorted({op["op_name"] for op in interpreter._get_ops_details()})
    assert not any("Flex" in x or "CUSTOM" in x for x in ops), ops
    runner = interpreter.get_signature_runner()
    from tensorflow.lite.python import schema_py_generated as schema
    flat = schema.Model.GetRootAsModel(content, 0)
    assert all(flat.OperatorCodes(i).BuiltinCode() != schema.BuiltinOperator.CUSTOM for i in range(flat.OperatorCodesLength())), "CUSTOM/Flex operator in serialized FlatBuffer"
    def run(X):
        values = [runner(features=np.asarray(row[None], np.float32)) for row in X]
        return np.concatenate([v["scores"] for v in values]), np.concatenate([v["class_index"] for v in values])
    return run, ops


def parity(model, X, run):
    X = np.asarray(X, np.float32)
    expected = reference_scores(model, X)
    scores, indices = run(X)
    actual = model.classes_[indices]
    predicted = model.predict(X)
    assert np.isfinite(scores).all()
    difference = float(np.max(np.abs(scores - expected)))
    assert np.allclose(scores, expected, atol=4e-3, rtol=3e-3), (type(model[-1]).__name__, difference)
    assert np.array_equal(predicted, actual), (type(model[-1]).__name__, int(np.sum(predicted != actual)))
    return {"rows": len(X), "maximum_absolute_score_error": difference, "class_disagreements": 0}


def boundary_inputs(model, X):
    scaler, head = model[0], model[-1]
    z = np.vstack([np.zeros(X.shape[1]), np.ones(X.shape[1]), -np.ones(X.shape[1]), np.full(X.shape[1], 10.), np.full(X.shape[1], -10.)])
    points = z * scaler.scale_ + scaler.mean_
    points = np.vstack([points, np.min(X, axis=0), np.max(X, axis=0)])
    if hasattr(head, "coef_") and not isinstance(head, SVC):
        w = head.coef_[0]
        norm = float(w @ w)
        if norm > 0:
            center = -head.intercept_[0] * w / norm
            # Test both sides, not an unknowable float64/float32 exact tie.
            points = np.vstack([points, (center + .01*w/norm)*scaler.scale_+scaler.mean_, (center-.01*w/norm)*scaler.scale_+scaler.mean_])
    return points.astype(np.float32)


def tensor_parity(model, X):
    module = Scorer(model)
    def run(values):
        result = module(tf.constant(np.asarray(values, np.float32)))
        return result["scores"].numpy(), result["class_index"].numpy()
    return parity(model, X, run)


def selfcheck():
    X = np.array([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.], [4., -2.], [3., -3.]], np.float32)
    checks = []
    for y in [np.array(["z", "z", "a", "a", "m", "m"]), np.array([7, 7, 2, 2, 7, 2])]:
        for head in [LogisticRegression(C=.7, max_iter=1000), LinearSVC(C=.7, dual="auto"), SVC(C=.7, gamma=.2), SVC(C=.7, kernel="linear"), MLPClassifier(hidden_layer_sizes=(3,), activation="tanh", solver="lbfgs", random_state=42, max_iter=1000)]:
            model = make_pipeline(StandardScaler(), head).fit(X, y)
            run, ops = runtime(convert(model))
            checks.append({"type": type(head).__name__, "classes": model.classes_.tolist(), "real_shape_fixture": parity(model, np.vstack([X, [[0, 0], [1e-4, -1e-4]]]), run), "ops": ops})
    # Regression: class-index output is not argmax(OVR) for SVC vote ties.
    model = make_pipeline(StandardScaler(), SVC(C=.7, gamma=.2)).fit(X, np.array(["z", "z", "a", "a", "m", "m"]))
    scorer = Scorer(model)
    z = model[0].transform(X).astype(np.float32)
    kernel = np.exp(-.2 * np.sum((z[:, None] - model[-1].support_vectors_[None]) ** 2, axis=2))
    assert np.allclose(np.diag(np.exp(-.2 * np.sum((z[:, None]-z[None])**2, axis=2))), 1.)
    assert np.isfinite(kernel).all() and scorer.nclass == 3
    probe = np.stack(np.meshgrid(np.linspace(-8, 8, 25), np.linspace(-8, 8, 25)), axis=-1).reshape(-1, 2).astype(np.float32)
    tensor_parity(model, probe)
    assert np.array_equal(model.predict(probe), model.classes_[scorer(tf.constant(probe))["class_index"].numpy()])
    # Binary margin orientation and exact-zero tie semantics differ by classifier.
    for head in [LinearSVC(dual="auto"), LogisticRegression(), SVC(kernel="linear")]:
        m = make_pipeline(StandardScaler(), head).fit(np.array([[-1.], [1.]], np.float32), np.array(["a", "z"]))
        parity(m, np.array([[-1e-3], [0], [1e-3]], np.float32), runtime(convert(m))[0])
    return {"status": "passed", "cases": checks, "coverage": ["nonlexical input labels sorted by sklearn", "binary positive margin maps class1", "binary exact zero tie", "multiclass libsvm OVO votes and OVR scores", "direct RBF squared-distance math", "logistic probabilities", "MLP tanh softmax"]}


def export(name, model, X, contract, smoke, provenance):
    content = convert(model)
    path = OUT / (name + ".tflite")
    path.write_bytes(content)
    head_path = OUT / (name + ".joblib")
    joblib.dump(model, head_path, compress=3)
    indices = np.linspace(0, len(X)-1, min(8, len(X)), dtype=int)
    fixture = OUT / (name + "-verification.npz")
    fixture_X = np.asarray(X[indices], np.float32)
    fixture_predictions = model.predict(fixture_X)
    np.savez_compressed(fixture, features=fixture_X, scores=reference_scores(model, fixture_X),
                        class_index=np.array([list(model.classes_).index(p) for p in fixture_predictions], np.int32),
                        classes=model.classes_, source_indices=indices)
    run, ops = runtime(content)
    real = parity(model, X, run)
    edges = parity(model, boundary_inputs(model, X), run)
    smokereport = None
    if smoke is not None:
        vector, evidence = smoke
        index = evidence.pop("cache_row", 0)
        delta = float(np.max(np.abs(np.asarray(vector) - X[index])))
        assert np.allclose(vector, X[index], rtol=2e-4, atol=2e-4), (name, delta)
        smokereport = {**evidence, "cache_row": index, "frontend_cache_maximum_absolute_error": delta, "runtime": parity(model, np.asarray(vector)[None], run)}
    result = {"name": name, "path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": len(content), "format": "builtin-only float32 TFLite", "operators": ops,
              "input": {"name": "features", "shape": [1, X.shape[1]], "dtype": "float32", "frontend_included": False, "scaler_included": True},
              "output": {"scores": "probabilities in classes order" if isinstance(model[-1], (LogisticRegression, MLPClassifier)) else "SVM margins (binary [-margin,+margin], multiclass sklearn OVR); NOT calibrated probabilities", "class_index": "index into classes; use this rather than score argmax for SVC vote ties", "classes": model.classes_.tolist()},
              "contract": contract, "training": provenance, "full_data_fit_parity_not_generalization": real, "finite_boundary_parity": edges, "real_input_smoke": smokereport,
              "physical_iphone": "not tested", "performance_claim": "Deployment fitted on every cached observation. No deployment training-set accuracy presented as held-out performance."}
    result["verification_fixture"] = {"path": str(fixture.relative_to(ROOT)), "sha256": digest(fixture), "schema": "features float32[N,D], scores float64[N,K], class_index int32[N], classes[K], source_indices[N]; <=8 real rows"}
    result["full_data_sklearn_head"] = {"path": str(head_path.relative_to(ROOT)), "sha256": digest(head_path)}
    save(OUT / (name + ".json"), result)
    print(json.dumps({"export": name, "rows": len(X), "bytes": len(content), "error": real["maximum_absolute_score_error"]}), flush=True)
    return result


def tune_material(X, rows):
    rows = [dict(r, label=r.get("truth", r.get("label"))) for r in rows]
    y = np.array([r.get("truth", r.get("label")) for r in rows])
    ranks = np.array([r.get("outer_fold", r.get("object_rank")) for r in rows])
    cs = [.01, .1, 1., 10.]
    scores = []
    for c in cs:
        values = []
        for held in sorted(set(ranks)):
            train, test = np.flatnonzero(ranks != held), np.flatnonzero(ranks == held)
            m = mb.fit(X[train], y[train], c)
            _, truth, pred = mb.object_predictions(m, X, rows, test)
            from sklearn.metrics import f1_score
            values.append(float(f1_score(truth, pred, average="macro", zero_division=0)))
        scores.append(float(np.mean(values)))
    chosen = cs[int(np.argmax(scores))]
    return mb.fit(X, y, chosen), {"selection": "all-data training-only object-rank CV; no outer-score family winner", "C_grid": cs, "mean_object_macro_f1": scores, "selected_C": chosen}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    historical = sorted(p for p in ART.glob("*") if p.is_file())
    hashes_before = {str(p.relative_to(ROOT)): digest(p) for p in historical}
    inventory, oof = [], []
    save(OUT / "scorer-selfcheck.json", selfcheck())
    # Cached YAMNet is used only for representative real-input parity, never refitted.
    encoder_path = ART / "tfhub/9616fd04ec2360621642ef9455b84f4b668e219e"
    encoder = tf.saved_model.load(str(encoder_path))
    encoder_ref = {"path": str(encoder_path.relative_to(ROOT)), "saved_model_sha256": digest(encoder_path / "saved_model.pb"), "pooling": "arithmetic mean of 1024D frame embeddings, not 521 event scores"}
    acoustic90 = {"feature_order": ["13 MFCC means", "13 MFCC population std", "32 natural-log-mel means", "32 natural-log-mel population std"], "frontend": "experiments/dataset_benchmark.py:feature", "frontend_sha256": digest(ROOT / "experiments/dataset_benchmark.py"), "capture": "mono mean, polyphase 16kHz, first4s, whole-recording mean removal, Hann400 hop160 FFT512; 32 mel80-7600Hz; ortho DCT13", "frontend_included": False}
    reports = load("dataset-benchmarks.json")["results"]
    for name, historical_report in zip(["acm_s2_fillclass", "pipe_leak"], reports):
        rows = historical_report["metrics"]["predictions"]
        # These datasets have no retained feature NPZ: recompute once from all real clips.
        X = np.stack([db.feature(ROOT / r["path"]) for r in rows])
        y = np.array([r["truth"] for r in rows])
        np.savez_compressed(OUT / (name + "-features.npz"), X=X, y=y)
        identities = np.array([digest(ROOT/r["path"]) for r in rows])
        if name == "pipe_leak":
            decoded_hashes = []
            for row in rows:
                audio, rate = sf.read(ROOT/row["path"], dtype="float32", always_2d=True)
                decoded_hashes.append(hashlib.sha256(str((rate, audio.shape)).encode()+audio.tobytes()).hexdigest())
            decoded_hashes = np.array(decoded_hashes)
        assert identities.tolist() == [r["sha256"] for r in rows]
        groups = np.array([r["group"] for r in rows])
        audit = None
        if name == "pipe_leak":
            parents = {g: g for g in set(groups)}
            def root(g):
                while parents[g] != g:
                    parents[g] = parents[parents[g]]
                    g = parents[g]
                return g
            links = []
            for sha in sorted(set(identities)):
                ix = np.flatnonzero(identities == sha)
                gs = sorted(set(groups[ix]))
                if len(gs) > 1:
                    links.append({"sha256": sha, "indices": ix.tolist(), "filename_groups": gs})
                    for g in gs[1:]:
                        parents[root(g)] = root(gs[0])
            merged = np.array([root(g) for g in groups])
            audit = {"linked_clips": sum(len(x["indices"]) for x in links), "hash_links": links, "original_groups": len(set(groups)), "merged_connected_components": len(set(merged)), "method": "union filename groups connected by an identical complete-file SHA256; transitive closure before LOGO"}
            from collections import Counter
            counts = Counter(identities)
            decoded_links = [np.flatnonzero(decoded_hashes == sha).tolist() for sha in sorted(set(decoded_hashes)) if len(set(groups[decoded_hashes == sha])) > 1]
            feature_hashes = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in X])
            feature_links = [np.flatnonzero(feature_hashes == sha).tolist() for sha in sorted(set(feature_hashes)) if len(set(groups[feature_hashes == sha])) > 1]
            audit.update(feature_cross_group_clips=sum(map(len, feature_links)), feature_cross_group_sets=feature_links)
            audit.update(all_duplicate_clips=sum(n for n in counts.values() if n>1), all_duplicate_hash_sets=sum(n>1 for n in counts.values()), decoded_pcm_cross_group_clips=sum(map(len, decoded_links)), decoded_pcm_cross_group_sets=decoded_links, previous_reviewer_26_clip_count="Not reproduced by complete-file SHA256, decoded float32 PCM SHA256 or exact feature-row hashing; observed12cross-group clips/6hash sets. All observed hash-connected groups are merged.")
            groups = merged
            save(OUT / "pipe-leak-hash-group-audit.json", audit)
        for family, fit in [("logistic", fit_linear), ("rbf", fit_rbf)]:
            fold_rows, predicted = [], np.full(y.shape, -1)
            for train, test in LeaveOneGroupOut().split(X, y, groups):
                assert not set(identities[train]) & set(identities[test])
                if len(set(y[train])) < 2:
                    continue
                m = fit(X[train], y[train])
                predicted[test] = m.predict(X[test])
                fold_rows.append({"held_out": str(groups[test[0]]), "test_indices": test.tolist(), "tensor_score_parity": tensor_parity(m, X[test])})
            valid = predicted >= 0
            refreshed = {"dataset": name, "family": family, "status": "exploratory", "protocol": "fixed C=1, gamma=scale; whole container LOGO" if name.startswith("acm") else "merged hash-connected filename-component LOGO, fixed C=1 gamma=scale; not verified acquisition/session independence", "n_scored": int(valid.sum()), "accuracy": float(accuracy_score(y[valid], predicted[valid])), "balanced_accuracy": float(balanced_accuracy_score(y[valid], predicted[valid])), "confusion_matrix": confusion_matrix(y[valid], predicted[valid]).tolist(), "truth": y.tolist(), "prediction": predicted.tolist(), "groups": groups.tolist(), "folds": fold_rows, "hash_group_audit": audit, "OOF_parity_scope": "TensorFlow scorer vs sklearn on each held-out fold; full deployment TFLite verified separately on all rows"}
            save(OUT / (name + "-" + family + "-oof.json"), refreshed)
            oof.append({"name": name+"_"+family, "folds": len(fold_rows), "parity": "per-fold TensorFlow score math; deployment TFLite separate"})
            contract = dict(acoustic90, task=historical_report["task"], labels={"0": "empty", "1": "50%", "2": "90%"} if name.startswith("acm") else {"0": "no leak including environmental noise", "1": "leak"}, protocol="pouring/shaking external setup, not taps" if name.startswith("acm") else "hydrophone/contact logger pipe monitoring, not verified airborne phone audio", license="Review original dataset permissions; local research only")
            contract["source"] = "https://zenodo.org/records/4770439" if name.startswith("acm") else "https://zenodo.org/records/18631450"
            contract["license"] = "Dataset CC BY4.0; preserve attribution; no model production-suitability grant inferred"
            smoke=(db.feature(ROOT/rows[0]["path"]), {"source": rows[0]["path"], "sha256": identities[0], "decoded_real_audio": True})
            inventory.append(export(name+"_"+family, fit(X,y), X, contract, smoke, {"fit": "all observations", "hyperparameters": "fixed C=1, balanced, gamma=scale for RBF", "feature_cache": str((OUT/(name+"-features.npz")).relative_to(ROOT)), "historical_report": "artifacts/dataset-benchmarks.json", "new_OOF_report": str((OUT/(name+"-"+family+"-oof.json")).relative_to(ROOT))}))
    for dataset, report_name, cache_name, head_name in [("ycb_material", "material-benchmark.json", "material-features.npz", "material-heads.joblib"), ("realimpact_material", "material-realimpact-benchmark.json", "material-realimpact-features.npz", "material-realimpact-heads.joblib")]:
        report = load(report_name)
        cache = np.load(ART/cache_name)
        heads = joblib.load(ART/head_name)
        rows = report["results"]["mfcc_logmel"]["recording_predictions"]
        if dataset.startswith("ycb"):
            source = ART/"datasets/material"/rows[0]["path"]
            wave, _ = mb.waveform(source)
            evidence = {"source": str(source.relative_to(ROOT)), "sha256": digest(source), "decoded_real_audio": True}
        else:
            source = ART/"datasets/material/realimpact-prefix"/rows[0]["object"]/"audio-first30.npy"
            raw = np.load(source)[rows[0]["source_row"]]
            signal = resample_poly(raw,1,3).astype(np.float32)
            start = round(rows[0]["crop_start_seconds"]*16000)
            wave = signal[start:start+16000].copy()
            wave *= .95/float(np.max(np.abs(wave)))
            evidence = {"source": str(source.relative_to(ROOT)), "sha256": digest(source), "source_row": rows[0]["source_row"], "real_deconvolved_audio": True, "not_raw_phone_PCM": True}
        for branch in ["mfcc_logmel", "yamnet"]:
            X = cache[branch]
            vector = mb.acoustic(wave) if branch == "mfcc_logmel" else encoder(wave)[1].numpy().mean(0)
            model, selection = tune_material(X, rows)
            folds = []
            for k in range(3):
                m = heads[f"{branch}_fold{k}"]
                test = np.array([i for i,r in enumerate(rows) if r.get("outer_fold",r.get("object_rank"))==k])
                folds.append({"fold": k, "tensor_score_parity": tensor_parity(m,X[test])})
            oof.append({"name": dataset+"_"+branch,"folds":folds,"source_heads":head_name,"parity":"saved historical OOF heads, TensorFlow scorer; deployment TFLite separate"})
            contract = {"task": "material_classification", "frontend": "experiments/material_benchmark.py:waveform + acoustic" if branch=="mfcc_logmel" else "same waveform crop + official frozen YAMNet frame-embedding mean", "frontend_sha256": digest(ROOT/"experiments/material_benchmark.py"), "feature_order": ["64 logmel means", "64 logmel population std", "13 MFCC means", "13 MFCC population std"] if branch=="mfcc_logmel" else "YAMNet embedding indices0..1023", "waveform": "1s peak20ms-centered16kHz; YCB decode and resample; RealImpact force-deconvolved48kHz->16kHz then .95 peak normalize", "encoder": encoder_ref if branch=="yamnet" else None, "license": "YCB source node_license null, rights review required" if dataset.startswith("ycb") else "RealImpact code/data/checkpoint permissions require separate review", "limits": report["limitations"]}
            inventory.append(export(dataset+"_"+branch,model,X,contract,(vector,dict(evidence)),dict(selection,source_cache=cache_name,source_cache_sha256=digest(ART/cache_name),historical_report=report_name)))
    for dataset in ["pucks", "motor"]:
        rows = load(f"puck-motor-{dataset}-manifest.json")
        cache = np.load(ART/f"puck-motor-{dataset}-features.npz")
        if dataset=="pucks":
            source=ART/"datasets/pucks/IDMT_ISA_PUCKS.zip"
            with zipfile.ZipFile(source) as archive:
                payload=archive.read(rows[0]["file"])
            audio, rate=sf.read(io.BytesIO(payload),dtype="float32",always_2d=True)
            evidence={"archive":str(source.relative_to(ROOT)),"member":rows[0]["file"],"sha256":hashlib.sha256(payload).hexdigest(),"decoded_real_audio":True}
        else:
            source=ART/"datasets/phone-motor"/rows[0]["file"]
            audio,rate=sf.read(source,dtype="float32",always_2d=True)
            evidence={"source":str(source.relative_to(ROOT)),"sha256":digest(source),"decoded_real_audio":True}
        mono=pm.mono16(audio,rate)
        start=int(rows[0]["start_seconds"]*16000)
        wave=mono[start:start+64000]
        for branch, key in [("mfcc-logmel","acoustic"),("yamnet","yamnet")]:
            X=cache[key]
            vector=pm.features(wave) if key=="acoustic" else encoder(wave)[1].numpy().mean(0)
            y=np.array([r["label"] for r in rows]); groups=np.array([r["group"] for r in rows])
            for family,fit in [("logistic",fit_linear),("rbf_svm",fit_rbf)]:
                path=ART/f"puck-motor-{dataset}-{branch}-{family}.joblib"
                model=joblib.load(path)
                folds=[]
                for train,test in LeaveOneGroupOut().split(X,y,groups):
                    m=fit(X[train],y[train]); folds.append({"held_out":str(groups[test[0]]),"tensor_score_parity":tensor_parity(m,X[test])})
                oof.append({"name":dataset+"_"+branch+"_"+family,"folds":folds,"parity":"replayed fixed historical training folds, TensorFlow scorer; deployment TFLite separate"})
                contract={"task":"material and manufacturing-process classification" if dataset=="pucks" else "same-motor debonded/repaired condition", "frontend":"experiments/puck_motor_benchmark.py:mono16 + features" if key=="acoustic" else "mono16 + four-second frozen YAMNet embedding mean", "frontend_sha256":digest(ROOT/"experiments/puck_motor_benchmark.py"),"feature_order":acoustic90["feature_order"] if key=="acoustic" else "YAMNet indices0..1023", "window":"three equally spaced4s nonoverlapping windows; model accepts one window feature vector; majority file/game vote outside model", "encoder":encoder_ref if key=="yamnet" else None,"license":load("puck-motor-source-audit.json")["pucks" if dataset=="pucks" else "phone_motor"],"limits":"Game grouping not physical-puck independence; unknown puck_A excluded" if dataset=="pucks" else "20 WAVs from one physical motor; M4A counterparts excluded; no unseen-motor validation"}
                inventory.append(export(dataset+"_"+branch+"_"+family,model,X,contract,(vector,dict(evidence)),{"existing_full_data_head":str(path.relative_to(ROOT)),"sha256":digest(path),"selection":"both fixed C=1 families retained, no outer-score winner selection","source_cache_sha256":digest(ART/f"puck-motor-{dataset}-features.npz")}))
    Xcache=np.load(ART/"compressed-air-features.npz"); X=Xcache["X"]; y=Xcache["y"]
    rows=load("compressed-air-manifest.json")
    for family,fit in [("logistic",fit_linear),("rbf",fit_rbf)]:
        folds=[]
        for axis in ["session","microphone"]:
            for train,test in LeaveOneGroupOut().split(X,y,Xcache[axis]):
                m=fit(X[train],y[train]); folds.append({"axis":axis,"held_out":str(Xcache[axis][test[0]]),"tensor_score_parity":tensor_parity(m,X[test])})
        oof.append({"name":"compressed_air_"+family,"folds":folds})
        contract=dict(acoustic90,task="audible_compressed_air_leak_vs_no_leak",labels={"0":"no_leak (_iO_)","1":"leak (_niO_)"},license="CC BY-NC-ND4.0; no commercial/redistribution rights inferred",limits="48selected whole recordings; no session3 or unknown factories/phone/seal-tap validation")
        inventory.append(export("compressed_air_"+family,fit(X,y),X,contract,(db.feature(ROOT/rows[0]["path"]),{"source":rows[0]["path"],"sha256":digest(ROOT/rows[0]["path"]),"decoded_real_audio":True}),{"fit":"all48cached real recordings","selection":"fixed C=1 gamma=scale, both families retained","cache_sha256":digest(ART/"compressed-air-features.npz")}))
    secondary=load("secondary-table-benchmark.json")["results"]
    import openpyxl
    path=ART/"datasets/secondary-tables/jackfruit/jackfruit.xlsx"
    workbook=openpyxl.load_workbook(path,read_only=True,data_only=True)
    entries=list(workbook.active.values); workbook.close()
    jackX=np.array([r[3:] for r in entries[1:]],dtype=np.float64)
    series=np.loadtxt(ART/"datasets/secondary-tables/gpla/data.csv",delimiter=",")
    gplaX=np.stack([sb.gpla_frontend(row) for row in series])
    for dataset,key,X in [("jackfruit","jackfruit",jackX),("gpla12","gpla_v1",gplaX)]:
        report=secondary[key]
        path=ART/("secondary-table-jackfruit-model.joblib" if dataset=="jackfruit" else "secondary-table-gpla-v1-model.joblib")
        model=joblib.load(path)
        preds=report["results"]["nested_selected"]["predictions"]
        y=np.array([r["truth"] for r in preds])
        folds=[]
        for fold in report["folds"]:
            train=np.array(fold["train_record_indices_zero_based"]); test=np.array(fold["test_record_indices_zero_based"])
            choice=fold["selection"]["nested_selected"]
            m=sb.model((choice["model"],choice["C"])).fit(X[train],y[train])
            assert np.array_equal(m.predict(X[test]),[preds[i]["prediction"] for i in test])
            folds.append({"fold":fold["fold"],"tensor_score_parity":tensor_parity(m,X[test])})
        oof.append({"name":dataset,"folds":folds,"parity":"replayed nested historical folds; TensorFlow score math"})
        contract={"task":"maturity Immature/Mature" if dataset=="jackfruit" else report["task_warning"],"labels":report["label_mapping"],"frontend":"UNAVAILABLE: supplied XLSX acoustic features; no raw audio/complete reproducible extraction" if dataset=="jackfruit" else "experiments/secondary_table_benchmark.py:gpla_frontend; source1460preprocessed numeric values, not PCM", "feature_order":report["schema"][3:] if dataset=="jackfruit" else ["mean","std","min","max","q10","q25","q50","q75","q90","standardized third moment","standardized fourth moment","centered sign crossing fraction","32normalized rFFT power bands in ascending cycles/sample"],"license":report["license"],"generalization_gate":report["generalization_gate"],"input_limit":"jackfruit cannot be invoked from phone audio without missing exact frontend" if dataset=="jackfruit" else "GPLA12 is12leak-only pressure/mic/noise classes, not leak detection"}
        smoke=None if dataset=="jackfruit" else (sb.gpla_frontend(series[0]),{"source":"artifacts/datasets/secondary-tables/gpla/data.csv","sha256":digest(ART/"datasets/secondary-tables/gpla/data.csv"),"real_preprocessed_time_series":True,"not_audio_PCM":True})
        inventory.append(export(dataset,model,X,contract,smoke,{"existing_full_data_head":str(path.relative_to(ROOT)),"sha256":digest(path),"selection":"historical mean nested INNER-CV scores, not outer-score winner","historical_report":"artifacts/secondary-table-benchmark.json"}))
    cache=np.load(ART/"impact-echo-sdnet-features.npz");X=cache["features"];y=cache["labels"];groups=cache["deck"]
    report=load("impact-echo-results.json")["sdnet2021"]
    # Deployment SVM chooses parameters only in all-data training-deck CV.
    from sklearn.model_selection import GridSearchCV
    search=GridSearchCV(make_pipeline(StandardScaler(),SVC(class_weight="balanced")),{"svc__C":[.1,1.,10.],"svc__kernel":["linear","rbf"]},scoring="balanced_accuracy",cv=LeaveOneGroupOut(),n_jobs=1)
    search.fit(X,y,groups=groups)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mlp=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,),activation="tanh",solver="lbfgs",alpha=1.,max_iter=300,random_state=42)).fit(X,y)
    first=sorted((ART/"datasets/impact_echo/repository/IE_data_sdnet2021_original").glob("*/*_origin_*.npy"))[0]
    wave=np.load(first)[0:1]
    vector=ie.spectral_features(wave,102400)[0]
    for family,model in [("spectral_svm",search.best_estimator_),("small_spectral_mlp",mlp)]:
        folds=[]
        for held in sorted(set(groups)):
            path=ART/f"impact-echo-sdnet-{family}-{held}.joblib"
            m=joblib.load(path); test=np.flatnonzero(groups==held)
            folds.append({"held_out_deck":str(held),"tensor_score_parity":tensor_parity(m,X[test])})
        oof.append({"name":"sdnet_"+family,"folds":folds,"parity":"saved historical deck-OOF heads; TensorFlow score math"})
        contract={"task":"structural impact-echo concrete class", "labels":report["label_meanings"],"frontend":"experiments/impact_echo_benchmark.py:spectral_features, native102400Hz sensor amplitude, first2048samples/20ms; not ordinary phone PCM", "frontend_sha256":digest(ROOT/"experiments/impact_echo_benchmark.py"),"feature_order":["64log relative spectral bands0.5-25kHz","DCT band coefficients1..16","relative centroid","relative width","peak","flatness"],"license":"Author-preprocessed SDNET mirror; original data and mirror permissions require review","limits":"Whole bridge decks held out historically; no phone or tile sensing claim"}
        contract["feature_order"] = ["64 natural-log normalized band energies, ascending500..25000Hz", "ortho DCT coefficients1..16", "centroid_Hz", "width_Hz", "peak_band_center_Hz", "flatness"]
        contract["source"] = "https://commons.und.edu/data/19/ ; pinned author derivative https://github.com/ehoxha91/impact_echo_datasets commit ff970edfa89637623e13cd89132c5266862c6344"
        contract["unresolved_label_provenance"] = "Pinned mirror counts1378/494/64 differ by one label from published1379sound/557delaminated; original raw archive unavailable; do not claim identical official labels"
        inventory.append(export("sdnet_"+family,model,X,contract,(vector,{"source":str(first.relative_to(ROOT)),"sha256":digest(first),"source_row":0,"real_sensor_waveform":True,"not_phone_PCM":True}),{"fit":"all1936real sensor records","selection":search.best_params_ if family=="spectral_svm" else "fixed tanh32 alpha1 MLP historical comparator, not outer-selected", "training_warnings":[] if family=="spectral_svm" else [str(w.message) for w in caught],"cache_sha256":digest(ART/"impact-echo-sdnet-features.npz")}))
    save(OUT/"oof-parity.json",{"scope":"Per-fold TensorFlow implementation versus sklearn on held-out real features. These are NOT TFLite OOF measurements; each full-data deployment TFLite independently evaluated on all cached real rows and boundaries.","models":oof})
    assert all(digest(ROOT/path)==sha for path,sha in hashes_before.items()), "Historical artifact changed"
    summary={"status":"complete_research_heads_with_explicit_frontend_and_generalization_limits","models":inventory,"excluded":{"ccny":"No unambiguous per-signal ground truth/index-to-geometry mapping; no supervised model exported.","durian_coconut":"Owned separately; deliberately outside this experiment."},"missing_prerequisites":{"jackfruit":"No raw audio and exact source acoustic frontend; exported supplied-feature research head only.","gpla12":"1460preprocessed values lack original sampling/calibration; no raw-PCM/audio inference claim.","all":"Physical iPhone runtime, independent target-phone acquisition validation and production license review not done."},"historical_report_hashes_preserved":hashes_before,"environment":{"python":platform.python_version(),"tensorflow":tf.__version__,"threads_max":2},"script_sha256":digest(__file__)}
    save(OUT/"inventory.json",summary)
    lines=["# Research classifier finalization","", "Run `python -m experiments.finalize_classifiers` from `models/`. No downloads or historical report writes. Full machine-readable contracts, hashes and numerical evidence are in `artifacts/final/classifiers/inventory.json` and each adjacent model JSON.","", "## Export inventory","", "All models are builtin-only float32 **feature heads**, not raw-audio models. StandardScaler and complete score/prediction function are included. `scores` follows declared sklearn class order; use `class_index` for classification because SVC OVO vote ties need not equal OVR-score argmax. SVM scores are not probabilities. No phone hardware was tested.","", "| Artifact | Real feature rows | Bytes | Full-data score max absolute error |","|---|---:|---:|---:|"]
    for item in inventory:
        lines.append(f"| `{item['name']}.tflite` | {item['full_data_fit_parity_not_generalization']['rows']} | {item['bytes']} | {item['full_data_fit_parity_not_generalization']['maximum_absolute_score_error']:.6g} |")
    lines += ["", "## Selection and evaluation", "", "Existing full-data Pucks/motor and secondary-table heads were reused. Both fixed logistic/RBF families retained where previously compared; no deployment family was chosen using outer held-out winner scores. YCB/RealImpact full-data LinearSVC C uses object-rank training CV. SDNET full-data SVM C/kernel uses training-deck CV; fixed tanh32 MLP retained as comparator. ACM-S2/pipe caches did not exist: real features were recomputed once; existing caches reused elsewhere.", "", "`oof-parity.json` separates historical/replayed per-fold TensorFlow scorer parity from full-data **TFLite** parity. No OOF TFLite parity is claimed. All deployment TFLites were compared to sklearn on every real cached feature row, finite edge inputs and meaningful near-boundary probes. `scorer-selfcheck.json` records binary sign/zero tie, class order, OVO voting, logistic, tanh MLP and RBF regression cases. No full-fit accuracy is passed off as generalization.", "", "## Pipe leakage correction", "", "`pipe-leak-hash-group-audit.json` merges all filename groups connected by identical complete-file SHA256 before LOGO. New per-family OOF reports preserve the historical report untouched; grouping does not establish missing session/acquisition independence. See refreshed metrics there, not a deployment training score.", "", "## Frontends, labels and limitations", "", "Exact feature order, source reference, scaler, class labels, training provenance and real-input smoke evidence are adjacent to each TFLite. ACM-S2 is21pouring/shaking clips, codes0/1/2=empty/50%/90%, not tap-volume regression. YCB uses robot impact OGX; RealImpact uses force-deconvolved impulse responses, not raw PCM. Pucks known ABS/PA2200/PA12 materials remain manufacturing-process confounded; unknown puck_A excluded. Motor is one physical motor before/after repair, not general fault diagnosis. Compressed air is48selected emissions, not seal/vacuum inference. Pipe recordings may be hydrophone/contact logger, not verified phone airborne audio.", "", "Jackfruit is an explicitly **supplied-feature-only** maturity head:27columns preserved from XLSX, no raw audio or exact reproducible acoustic frontend available, so no audio smoke or ready-to-capture claim. GPLA12 accepts44features from1460already-preprocessed numeric values;12classes are pressure/microphone/noise combinations among leak signals, not leak/no-leak. Their CV is record/content-disjoint only: unseen-fruit/run generalization blocked by missing acquisition IDs. SDNET accepts native102.4kHz impact-echo sensor features; sound/above-rebar/below-rebar classes are not phone/tile claims. CCNY is excluded because per-signal truth remains ambiguous.", "", "## Permissions and device gate", "", "Local research only. Pucks has conflicting metadata CC-BY-NC-ND4.0/PDF CC-BY-SA4.0 terms; compressed air CC-BY-NC-ND4.0. Motor CC-BY4.0. YCB source node license null. Jackfruit repository MIT does not establish independent dataset provenance; GPLA repository EPL-2.0 requires dataset review. Other dataset/model permissions must be reviewed separately before redistribution or product use. Model hashes do not grant rights. No physical iPhone latency/memory/energy, capture parity or held-out THUD product validation measured.", ""]
    lines += ["## Source links and leakage-count clarification", "", "[ACM-S2](https://zenodo.org/records/4770439) and [pipe-leak audio](https://zenodo.org/records/18631450) dataset metadata state CC BY4.0; preserve attribution. [YCB](https://osf.io/4tcp6), [Pucks](https://zenodo.org/records/7551338), [phone motor](https://data.mendeley.com/datasets/bxcnxfm94c/5), [compressed air](https://zenodo.org/records/7551606), [jackfruit](https://github.com/Abhinandan-Hooli/jackfruit-acoustic-maturity-classification), [GPLA12](https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12), [SDNET](https://commons.und.edu/data/19/) and the [pinned structural derivative](https://github.com/ehoxha91/impact_echo_datasets) remain protocol-specific research sources.", "", "The pipe audit found72clips in36exact-hash duplicate pairs overall, of which12clips/6hash sets cross filename groups. Decoded PCM and feature-row hashes reproduce the12cross-group count, not the previous reviewer26clip figure. All observed linked filename groups were merged319->314components. Refreshed exploratory LOGO accuracy/balanced accuracy: logistic91.4%, RBF94.3%; acquisition independence remains unknown. SDNET mirror class counts1378/494/64 retain the unresolved one-label discrepancy versus the published original count; no labels were changed.", ""]
    DOC.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"exported":len(inventory),"inventory":str((OUT/"inventory.json").relative_to(ROOT))}),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
