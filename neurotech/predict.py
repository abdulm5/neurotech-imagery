"""Predict cue-aligned imagined fist labels from a supported raw EDF."""
import argparse
import joblib
import pandas as pd
from .data import extract_edf


def predict_edf(edf_path, model_path, run=None):
    """Return one row per task cue. Only load model files from a trusted source."""
    bundle = joblib.load(model_path)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model artifact format")
    x, rows, _ = extract_edf(edf_path, run=run, channels=bundle["channels"],
                             config=bundle["preprocessing"], include_labels=False)
    predicted = bundle["pipeline"].predict(x) if len(x) else []
    for row in rows:
        idx = row.pop("epoch_index")
        row["predicted_label"] = bundle["classes"][int(predicted[idx])] if idx >= 0 else "unavailable"
    columns = ["event_index", "onset", "duration", "predicted_label", "quality",
               "high_amplitude", "flat_channels", "peak_to_peak_uv"]
    return pd.DataFrame(rows, columns=columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", required=True)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--output")
    parser.add_argument("--run", type=int)
    args = parser.parse_args()
    try:
        result = predict_edf(args.edf, args.model, args.run)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")
    if args.output:
        result.to_csv(args.output, index=False)
    else:
        print(result.to_csv(index=False), end="")

if __name__ == "__main__":
    main()
