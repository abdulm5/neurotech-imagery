"""Check the shipped model against saved holdout predictions without refitting."""
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from .data import edf_path, write_json
from .predict import predict_edf


def main():
    bundle = joblib.load("artifacts/model.joblib")
    freeze = json.loads(Path("artifacts/test_freeze.json").read_text())
    digest = hashlib.sha256(Path("artifacts/model.joblib").read_bytes()).hexdigest()
    assert digest == freeze["model_sha256"], "Model differs from evaluated artifact"
    heldout = pd.read_csv("results/test_predictions.csv")
    assert not set(heldout.subject) & set(bundle["training_subjects"])
    total = 0
    for (subject, run), expected in heldout.groupby(["subject", "run"]):
        actual = predict_edf(edf_path("data/raw", subject, run), "artifacts/model.joblib")
        actual = actual[actual.predicted_label != "unavailable"].sort_values("event_index")
        expected = expected.sort_values("event_index")
        np.testing.assert_array_equal(actual.event_index, expected.event_index)
        np.testing.assert_allclose(actual.onset, expected.onset, atol=1e-9)
        labels = expected.prediction.map(bundle["classes"])
        np.testing.assert_array_equal(actual.predicted_label, labels)
        total += len(actual)
    result = {"status": "passed", "python": platform.python_version(), "model_sha256": digest,
              "matched_raw_edf_predictions": total, "recordings": heldout.groupby(["subject", "run"]).ngroups,
              "test_subjects_excluded_from_training": True, "refitting": False}
    write_json("results/verification.json", result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
