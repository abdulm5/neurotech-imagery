"""Behavioral checks for leakage barriers and raw-EDF inference."""
import json
from pathlib import Path
import subprocess
import sys

import joblib
import mne
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from neurotech.data import CHANNELS, CONFIG, extract_raw, extract_edf, infer_run, manifest, transform_trial
from neurotech.models import make_model
from neurotech.predict import predict_edf


def raw_fixture(sfreq=160):
    rng = np.random.default_rng(15)
    data = rng.normal(0, 8e-6, (64, int(sfreq * 30)))
    t = np.arange(data.shape[1]) / sfreq
    data[8] += 10e-6 * np.sin(2*np.pi*10*t)
    raw = mne.io.RawArray(data, mne.create_info(CHANNELS, sfreq, "eeg"), verbose="ERROR")
    raw.set_annotations(mne.Annotations([0, 6, 12, 18, 26], [4.1, 4.1, 4.1, 4.1, 2], ["T1", "T2", "T1", "T2", "T1"]))
    return raw


def save_bundle(path, x, labels):
    bundle = {"format_version": 1, "pipeline": make_model("spectral").fit(x, labels),
              "channels": CHANNELS, "preprocessing": CONFIG, "classes": {0:"left", 1:"right"}}
    joblib.dump(bundle, path)
    return bundle


def test_manifest_is_fixed_and_disjoint(tmp_path):
    path = tmp_path / "split.json"
    a = manifest(path)
    assert a == manifest(path)
    assert len(a["development"]) == 30 and len(a["test"]) == 10
    assert not set(a["development"]) & set(a["test"])


def test_run_semantics():
    assert infer_run("S001R04.edf") == 4
    assert infer_run("renamed.edf", 12) == 12
    for path, run in [("S001R06.edf", None), ("renamed.edf", None), ("S001R04.edf", 8)]:
        with pytest.raises(ValueError):
            infer_run(path, run)


def test_labels_order_quality_and_channel_order():
    raw = raw_fixture()
    x, rows, audit = extract_raw(raw.copy())
    assert x.shape == (4, 64, 480)
    assert [row["label"] for row in rows] == [0,1,0,1,0]
    assert rows[-1]["quality"] == "unavailable_incomplete"
    assert audit["valid_events"] == 4
    shuffled = raw.copy().reorder_channels(CHANNELS[::-1])
    reordered, _, _ = extract_raw(shuffled)
    np.testing.assert_array_equal(x, reordered)


def test_annotation_swap_cannot_change_features_or_predictions(tmp_path):
    raw = raw_fixture()
    x, rows, _ = extract_raw(raw.copy())
    bundle = save_bundle(tmp_path / "model.joblib", x, [0,1,0,1])
    raw.annotations.rename({"T1": "T2", "T2": "T1"})
    swapped_x, swapped_rows, _ = extract_raw(raw, include_labels=False)
    assert all("label" not in row for row in swapped_rows)
    np.testing.assert_array_equal(x, swapped_x)
    np.testing.assert_array_equal(bundle["pipeline"].predict(x), bundle["pipeline"].predict(swapped_x))


def test_trial_isolation():
    raw = raw_fixture()
    original, _, _ = extract_raw(raw.copy())
    raw._data[:, 7*160:10*160] *= 100
    changed, _, _ = extract_raw(raw)
    np.testing.assert_array_equal(original[0], changed[0])
    assert not np.array_equal(original[1], changed[1])


def test_invalid_and_flagged_events():
    raw = raw_fixture()
    raw._data[:, 1*160:4*160] = 0
    raw._data[0, 7*160] = np.nan
    raw._data[1, 13*160] = 1e-3
    x, rows, _ = extract_raw(raw)
    assert rows[0]["quality"] == "unavailable_flat"
    assert rows[1]["quality"] == "unavailable_nonfinite"
    assert rows[2]["high_amplitude"]
    assert len(x) == 2


def test_resampling_and_missing_channels():
    x, _, _ = extract_raw(raw_fixture(128))
    assert x.shape == (4,64,480) and np.isfinite(x).all()
    with pytest.raises(ValueError, match="Missing required"):
        extract_raw(raw_fixture().drop_channels(["C3"]))


def test_subject_folds():
    subjects = np.repeat(np.arange(10), 6)
    for train, valid in GroupKFold(5).split(subjects, groups=subjects):
        assert not set(subjects[train]) & set(subjects[valid])


def test_csp_fits_rank_deficient_referenced_trials(tmp_path):
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1e-5, (20, 64, 480))
    x -= x.mean(axis=1, keepdims=True)
    with threadpool_limits(limits=1):
        estimator = make_model("csp").fit(x, np.tile([0, 1], 10))
        predicted = estimator.predict(x)
    path = tmp_path / "csp.joblib"
    joblib.dump(estimator, path)
    np.testing.assert_array_equal(predicted, joblib.load(path).predict(x))


def test_no_task_annotations(tmp_path):
    raw = raw_fixture()
    raw.set_annotations(mne.Annotations([0], [30], ["T0"]))
    x, rows, _ = extract_raw(raw)
    assert x.shape == (0, 64, 480) and rows == []


def test_subject_metric_coverage_after_quality_filter():
    from neurotech.evaluate import summarize
    frame = pd.DataFrame({"subject": [1, 1, 2, 2], "label": [0, 1, 0, 0],
                          "prediction": [0, 1, 0, 0]})
    score = summarize(frame)
    assert score["subjects"] == 2
    assert score["subjects_with_both_classes"] == 1
    assert score["subjects_without_both_classes"] == [2]
    assert score["mean_subject_balanced_accuracy"] == 1


def test_edf_cli_serialization_and_label_swap(tmp_path):
    edf = tmp_path / "S001R04.edf"
    raw_fixture().export(edf, fmt="edf", verbose="ERROR")
    x, rows, _ = extract_edf(edf)
    model_path = tmp_path / "model.joblib"
    bundle = save_bundle(model_path, x, [0,1,0,1])
    direct = predict_edf(edf, model_path)
    expected = [bundle["classes"][int(y)] for y in bundle["pipeline"].predict(x)]
    assert direct.predicted_label.tolist() == expected + ["unavailable"]
    swapped = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    swapped.annotations.rename({"T1":"T2", "T2":"T1"})
    other = tmp_path / "swapped.edf"
    swapped.export(other, fmt="edf", verbose="ERROR")
    assert predict_edf(other, model_path, run=4).predicted_label.tolist() == direct.predicted_label.tolist()
    out = tmp_path / "predictions.csv"
    subprocess.run([sys.executable, "-m", "neurotech.predict", "--edf", str(edf),
                    "--model", str(model_path), "--output", str(out)], check=True)
    restored = pd.read_csv(out)
    restored["flat_channels"] = restored.flat_channels.fillna("")
    pd.testing.assert_frame_equal(restored, direct, check_dtype=False)
