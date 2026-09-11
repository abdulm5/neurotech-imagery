"""Development-only comparisons, diagnostics, and sealed test evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import joblib
import mne
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, train_test_split
from threadpoolctl import threadpool_limits

from .data import (CHANNELS, CONFIG, SEED, manifest, load_prepared, prepare, write_json)
from .models import BandPower, make_model, spectral_classifier


def log_attempt(name, start, **details):
    record = {"experiment": name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.monotonic() - start, 3), **details}
    with open("results/experiments.jsonl", "a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def subject_scores(frame):
    records = []
    for subject, group in frame.groupby("subject"):
        records.append({"subject": int(subject), "n": len(group),
                        "accuracy": accuracy_score(group.label, group.prediction),
                        "balanced_accuracy": balanced_accuracy_score(group.label, group.prediction)
                        if group.label.nunique() == 2 else None})
    return pd.DataFrame(records)


def summarize(frame):
    scores = subject_scores(frame)
    values = scores.balanced_accuracy.dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    ci = np.quantile(rng.choice(values, (2000, len(values)), replace=True).mean(axis=1), [.025, .975]) if len(values) else [None, None]
    return {"trials": len(frame), "subjects": len(scores),
            "subjects_with_both_classes": len(values),
            "subjects_without_both_classes": scores.loc[scores.balanced_accuracy.isna(), "subject"].tolist(),
            "accuracy": accuracy_score(frame.label, frame.prediction),
            "balanced_accuracy": balanced_accuracy_score(frame.label, frame.prediction),
            "mean_subject_balanced_accuracy": float(values.mean()) if len(values) else None,
            "subject_bootstrap_95ci": list(ci),
            "confusion_matrix_left_right": confusion_matrix(frame.label, frame.prediction, labels=[0, 1]).tolist()}


def prediction_frame(meta, indices, prediction, fold, subset):
    result = meta.iloc[indices].copy()
    result["prediction"] = prediction
    result["fold"] = fold
    result["subset"] = subset
    return result


def experiment(name, x, meta, splits, factory):
    started = time.monotonic()
    frames = []
    y = meta.label.to_numpy()
    for fold, (train, valid) in enumerate(splits):
        assert not set(train) & set(valid)
        estimator = factory()
        estimator.fit(x[train], y[train])
        for indices, subset in [(train, "train"), (valid, "validation")]:
            frames.append(prediction_frame(meta, indices, estimator.predict(x[indices]), fold, subset))
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_csv(f"results/{name}_predictions.csv", index=False)
    validation = all_predictions[all_predictions.subset == "validation"]
    summary = summarize(validation)
    # Training predictions repeat across folds; average fold scores, not independent observations.
    train_frames = all_predictions[all_predictions.subset == "train"]
    summary["mean_fold_train_subject_balanced_accuracy"] = float(np.mean([
        summarize(g)["mean_subject_balanced_accuracy"] for _, g in train_frames.groupby("fold")]))
    clean = validation[~validation.high_amplitude]
    summary["unflagged_coverage"] = len(clean) / len(validation)
    summary["unflagged"] = summarize(clean) if len(clean) else None
    write_json(f"results/{name}_summary.json", summary)
    subject_scores(validation).to_csv(f"results/{name}_subjects.csv", index=False)
    log_attempt(name, started, status="completed", summary=summary)
    print(f"{name}: {summary['mean_subject_balanced_accuracy']:.3f}", flush=True)
    return summary


def development(permutations=100):
    if Path("results/test_summary.json").exists():
        raise ValueError("Holdout already evaluated; use a new study directory for further development")
    started = time.monotonic()
    split = manifest()
    x, meta = load_prepared("development")
    assert set(meta.subject).issubset(split["development"])
    assert not set(meta.subject) & set(split["test"])
    if meta.subject.nunique() < 5:
        raise ValueError("Need at least five usable development subjects")
    grouped = list(GroupKFold(n_splits=5).split(x, meta.label, groups=meta.subject))
    for train, valid in grouped:
        assert not set(meta.iloc[train].subject) & set(meta.iloc[valid].subject)
    selections = {}
    for name in ("spectral", "csp"):
        selections[name] = experiment(f"grouped_{name}", x, meta, grouped, lambda n=name: make_model(n))
    spectral = selections["spectral"]["mean_subject_balanced_accuracy"]
    csp = selections["csp"]["mean_subject_balanced_accuracy"]
    selected = "csp" if csp - spectral >= .01 else "spectral"
    write_json("artifacts/selection.json", {"selected_model": selected,
               "rule": "CSP only if mean subject balanced accuracy exceeds spectral by at least 0.01",
               "development_scores": {"spectral": spectral, "csp": csp},
               "test_evaluated": False, "seed": SEED})
    indices = np.arange(len(meta))
    random_split = [train_test_split(indices, test_size=.25, stratify=meta.label, random_state=SEED)]
    run_split = [(indices[meta.run.isin([4, 8])], indices[meta.run == 12])]
    for name in ("spectral", "csp"):
        experiment(f"random_{name}", x, meta, random_split, lambda n=name: make_model(n))
        experiment(f"later_run_{name}", x, meta, run_split, lambda n=name: make_model(n))
    dummy_x = np.zeros((len(x), 1))
    experiment("grouped_majority", dummy_x, meta, grouped, lambda: DummyClassifier(strategy="most_frequent"))
    metadata_x = meta[["run", "event_index"]].to_numpy(dtype=float)
    experiment("grouped_metadata", metadata_x, meta, grouped, spectral_classifier)

    # Label-blind audit follow-up: keep the original selection and folds unchanged.
    audit = json.loads(Path("results/audit_development_all.json").read_text())
    anomalous = sorted({a["subject"] for a in audit if a["status"] == "ok" and a["sampling_rate"] != CONFIG["sfreq"]})
    if anomalous:
        keep = ~meta.subject.isin(anomalous).to_numpy()
        sensitivity_splits = [(train[keep[train]], valid[keep[valid]]) for train, valid in grouped]
        sensitivity_splits = [(train, valid) for train, valid in sensitivity_splits if len(valid)]
        sensitivity = experiment("grouped_standard_rate_sensitivity", x, meta, sensitivity_splits,
                                 lambda: make_model("spectral"))
        write_json("results/sampling_rate_sensitivity.json", {"excluded_subjects": anomalous,
                   "reason": "Non-160-Hz EDF headers found in development-only audit; no score-based exclusion",
                   "selection_unchanged": True, "summary": sensitivity})

    features = BandPower().fit_transform(x)  # Deterministic per-trial calculation; no fitted population statistics.
    perm_started = time.monotonic()
    rng = np.random.default_rng(SEED)
    null = []
    blocks = list(meta.groupby(["subject", "run"]).indices.values())
    for iteration in range(permutations):
        perm_y = meta.label.to_numpy().copy()
        for block in blocks:
            perm_y[block] = rng.permutation(perm_y[block])
        predictions = np.empty(len(meta), dtype=int)
        for train, valid in grouped:
            estimator = spectral_classifier().fit(features[train], perm_y[train])
            predictions[valid] = estimator.predict(features[valid])
        null_frame = meta.copy()
        null_frame["label"] = perm_y
        null_frame["prediction"] = predictions
        null.append(float(subject_scores(null_frame).balanced_accuracy.mean()))
        if (iteration + 1) % 10 == 0:
            print(f"Permutation {iteration+1}/{permutations}", flush=True)
    perm_result = {"iterations": permutations, "observed": spectral, "null_scores": null,
                   "p_value": (1 + sum(score >= spectral for score in null)) / (1 + permutations),
                   "interpretation": "Within-subject/run exchangeability reference; not proof of neural causality"}
    write_json("results/permutation.json", perm_result)
    log_attempt("within_subject_run_permutations", perm_started, status="completed", **perm_result)

    identity_start = time.monotonic()
    train, valid = run_split[0]
    identity = spectral_classifier().fit(features[train], meta.subject.to_numpy()[train])
    predicted = identity.predict(features[valid])
    identity_result = {"accuracy": accuracy_score(meta.subject.to_numpy()[valid], predicted),
                       "balanced_accuracy": balanced_accuracy_score(meta.subject.to_numpy()[valid], predicted),
                       "subjects": int(meta.subject.nunique()), "uniform_chance": 1/meta.subject.nunique(),
                       "majority_accuracy": float((meta.subject.iloc[valid] == meta.subject.iloc[train].mode()[0]).mean())}
    identity_frame = meta.iloc[valid].copy()
    identity_frame["predicted_subject"] = predicted
    identity_frame.to_csv("results/identity_predictions.csv", index=False)
    write_json("results/identity.json", identity_result)
    log_attempt("subject_identity", identity_start, status="completed", **identity_result)

    for group in ("central", "occipital"):
        prepare("development", group=group)
        group_x, group_meta = load_prepared("development", group)
        # Keep identical trial population and folds for the regional comparison.
        lookup = {trial: i for i, trial in enumerate(group_meta.trial_id)}
        if not set(meta.trial_id).issubset(lookup):
            raise ValueError("Regional comparison has missing trials; reconcile explicitly before comparison")
        order = [lookup[t] for t in meta.trial_id]
        experiment(f"grouped_{group}", group_x[order], meta, grouped, lambda: make_model("spectral"))

    fit_start = time.monotonic()
    pipeline = make_model(selected).fit(x, meta.label)
    bundle = {"format_version": 1, "pipeline": pipeline, "model_name": selected,
              "preprocessing": CONFIG, "channels": CHANNELS, "classes": {0: "left", 1: "right"},
              "training_subjects": sorted(int(s) for s in meta.subject.unique()),
              "training_class_counts": {int(k): int(v) for k, v in meta.label.value_counts().items()},
              "training_majority_label": int(meta.label.mode()[0]),
              "reserved_test_subjects": split["test"], "seed": SEED,
              "versions": {p: version(p) for p in ["mne", "numpy", "scipy", "scikit-learn", "joblib"]},
              "python": platform.python_version()}
    joblib.dump(bundle, "artifacts/model.joblib", compress=3)
    write_json("artifacts/model_metadata.json", {k:v for k,v in bundle.items() if k != "pipeline"})
    log_attempt("final_development_fit", fit_start, status="completed", model=selected)
    log_attempt("development_complete", started, status="completed", selected_model=selected)


def test():
    output = Path("results/test_summary.json")
    if output.exists():
        raise ValueError("Test result already exists. Do not repeatedly inspect/reselect on the holdout.")
    started = time.monotonic()
    bundle = joblib.load("artifacts/model.joblib")
    split = manifest()
    assert not set(bundle["training_subjects"]) & set(split["test"])
    digest = hashlib.sha256(Path("artifacts/model.joblib").read_bytes()).hexdigest()
    write_json("artifacts/test_freeze.json", {"model_sha256": digest,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "selection": json.loads(Path("artifacts/selection.json").read_text())})
    prepare("test")
    x, meta = load_prepared("test")
    assert set(meta.subject).issubset(split["test"])
    frame = prediction_frame(meta, np.arange(len(meta)), bundle["pipeline"].predict(x), 0, "test")
    frame.to_csv("results/test_predictions.csv", index=False)
    result = summarize(frame)
    baseline = frame.copy()
    baseline["prediction"] = bundle["training_majority_label"]
    result["training_majority_baseline"] = summarize(baseline)
    clean = frame[~frame.high_amplitude]
    result.update(model_sha256=digest, model=bundle["model_name"], unflagged_coverage=len(clean)/len(frame),
                  unflagged=summarize(clean) if len(clean) else None)
    write_json(output, result)
    subject_scores(frame).to_csv("results/test_subjects.csv", index=False)
    log_attempt("sealed_subject_test", started, status="completed", summary=result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["development", "test"])
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    mne.set_log_level("ERROR")
    started = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            development(args.permutations) if args.command == "development" else test()
    except Exception as exc:
        log_attempt(args.command, started, status="failed", error=repr(exc))
        raise

if __name__ == "__main__":
    main()
