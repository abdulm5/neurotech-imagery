"""Download and audit EEGMMIDB; labels remain separate from signal processing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly
from fractions import Fraction

RUNS = (4, 8, 12)
SEED = 20260911
CONFIG = {"sfreq": 160, "tmin": 1.0, "tmax": 4.0, "band": [8, 30],
          "filter_order": 4, "padding_seconds": 1, "high_amplitude_volts": 200e-6,
          "flat_peak_to_peak_volts": 1e-12}
# Canonical 64-channel EEGMMIDB order, independent of any test recording.
CHANNELS = "Fc5 Fc3 Fc1 Fcz Fc2 Fc4 Fc6 C5 C3 C1 Cz C2 C4 C6 Cp5 Cp3 Cp1 Cpz Cp2 Cp4 Cp6 Fp1 Fpz Fp2 Af7 Af3 Afz Af4 Af8 F7 F5 F3 F1 Fz F2 F4 F6 F8 Ft7 Ft8 T7 T8 T9 T10 Tp7 Tp8 P7 P5 P3 P1 Pz P2 P4 P6 P8 Po7 Po3 Poz Po4 Po8 O1 Oz O2 Iz".split()
CENTRAL = "Fc3 Fc1 Fcz Fc2 Fc4 C3 C1 Cz C2 C4 Cp3 Cp1 Cpz Cp2 Cp4".split()
OCCIPITAL = "Po7 Po3 Poz Po4 Po8 O1 Oz O2".split()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def manifest(path="config/split.json"):
    path = Path(path)
    if path.exists():
        obj = json.loads(path.read_text())
    else:
        ids = np.random.default_rng(SEED).permutation(np.arange(1, 110))[:40].tolist()
        obj = {"seed": SEED, "development": ids[:30], "test": ids[30:], "runs": list(RUNS)}
        write_json(path, obj)
    all_ids = obj["development"] + obj["test"]
    if (len(obj["development"]) != 30 or len(obj["test"]) != 10
            or len(set(all_ids)) != 40 or not set(all_ids).issubset(range(1, 110))
            or obj["runs"] != list(RUNS)):
        raise ValueError("Manifest must contain 30 development and 10 distinct test subjects in 1–109, with runs 4/8/12")
    return obj


def edf_path(root, subject, run):
    return Path(root) / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"


def infer_run(path, run=None):
    match = re.fullmatch(r"S\d{3}R(\d{2})\.edf", Path(path).name, re.I)
    inferred = int(match.group(1)) if match else None
    if run is not None and inferred is not None and run != inferred:
        raise ValueError("Explicit run conflicts with EDF filename")
    run = run if run is not None else inferred
    if run not in RUNS:
        raise ValueError("Only left/right imagery runs 4, 8, 12 are supported; use --run for renamed EDFs")
    return run


def download(root="data/raw"):
    split = manifest()
    # Verify every file against the upstream published SHA256 list.
    base = "https://physionet.org/files/eegmmidb/1.0.0/"
    with urllib.request.urlopen(base + "SHA256SUMS.txt", timeout=60) as response:
        checksums = {line.split()[1].lstrip("./"): line.split()[0]
                     for line in response.read().decode().splitlines() if line.strip()}
    def fetch(item):
        subject, run = item
        path = edf_path(root, subject, run)
        rel = f"S{subject:03d}/{path.name}"
        started = time.monotonic()
        error = None
        for attempt in range(3):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    with urllib.request.urlopen(base + rel, timeout=90) as response:
                        payload = response.read()
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(payload)
                    tmp.replace(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != checksums[rel]:
                    path.unlink()
                    raise ValueError("SHA256 mismatch; deleted corrupt download")
                return {"subject": subject, "run": run, "status": "ok", "sha256": digest,
                        "seconds": round(time.monotonic() - started, 3), "attempts": attempt + 1}
            except Exception as exc:
                error = str(exc)
        return {"subject": subject, "run": run, "status": "failed", "error": error}
    items = [(s, r) for s in split["development"] + split["test"] for r in RUNS]
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for record in pool.map(fetch, items):
            records.append(record)
            if len(records) % 10 == 0:
                print(f"Checked {len(records)}/{len(items)} recordings", flush=True)
    write_json("results/download.json", records)
    failures = [x for x in records if x["status"] != "ok"]
    print(f"Downloaded/verified {len(records)-len(failures)}/{len(records)} EDFs", flush=True)
    if failures:
        raise RuntimeError("Download failures recorded in results/download.json; retry before training")


def normalize_channels(raw, channels=CHANNELS):
    lookup = {c.lower(): c for c in channels}
    mapping = {c: lookup.get(c.strip().strip(".").lower(), c.strip().strip(".")) for c in raw.ch_names}
    raw.rename_channels(mapping)
    missing = sorted(set(channels) - set(raw.ch_names))
    if missing:
        raise ValueError(f"Missing required EEG channels: {missing}")
    raw.pick(channels)
    raw.reorder_channels(channels)
    return raw


def transform_trial(signal, sfreq, config=CONFIG):
    """Reference, resample and filter one isolated [channels,time] trial."""
    signal = signal - signal.mean(axis=0, keepdims=True)
    if sfreq != config["sfreq"]:
        ratio = Fraction(config["sfreq"] / sfreq).limit_denominator(10000)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)
    expected = round((config["tmax"] - config["tmin"]) * config["sfreq"])
    signal = signal[:, :expected]
    if signal.shape[1] != expected:
        raise ValueError("Unexpected sample count after resampling")
    pad = round(config["padding_seconds"] * config["sfreq"])
    padded = np.pad(signal, ((0, 0), (pad, pad)), mode="reflect")
    sos = butter(config["filter_order"], config["band"], fs=config["sfreq"], btype="bandpass", output="sos")
    return sosfiltfilt(sos, padded, axis=-1)[:, pad:-pad]


def extract_raw(raw, channels=CHANNELS, config=CONFIG, include_labels=True):
    """Return valid epochs, all event records, audit. No filtering across trials."""
    original_names = list(raw.ch_names)
    normalize_channels(raw, channels)
    sfreq = float(raw.info["sfreq"])
    signals = raw.get_data()
    audit = {"sampling_rate": sfreq, "original_channels": original_names,
             "channel_order": list(raw.ch_names), "duration_seconds": raw.n_times / sfreq,
             "annotation_counts": {str(k): int(v) for k, v in zip(*np.unique(raw.annotations.description, return_counts=True))}}
    epochs, rows = [], []
    for onset, duration, annotation in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if annotation not in ("T1", "T2"):
            continue
        # Annotation onset is relative to raw origin; account for nonzero first sample.
        relative = float(onset - raw.first_time)
        row = {"event_index": len(rows), "onset": relative, "duration": float(duration),
               "epoch_index": -1, "quality": "ok", "high_amplitude": False,
               "flat_channels": "", "peak_to_peak_uv": None}
        if include_labels:
            row["label"] = 0 if annotation == "T1" else 1
        start = int(round((relative + config["tmin"]) * sfreq))
        stop = start + int(round((config["tmax"] - config["tmin"]) * sfreq))
        if duration + 1e-6 < config["tmax"] or start < 0 or stop > signals.shape[1]:
            row["quality"] = "unavailable_incomplete"
        else:
            trial = signals[:, start:stop]
            if not np.isfinite(trial).all():
                row["quality"] = "unavailable_nonfinite"
            else:
                ptp = np.ptp(trial, axis=1)
                row["peak_to_peak_uv"] = float(ptp.max() * 1e6)
                row["high_amplitude"] = bool(ptp.max() > config["high_amplitude_volts"])
                row["flat_channels"] = ",".join(c for c, p in zip(channels, ptp) if p <= config["flat_peak_to_peak_volts"])
                if np.all(ptp <= config["flat_peak_to_peak_volts"]):
                    row["quality"] = "unavailable_flat"
                else:
                    flags = []
                    if row["high_amplitude"]:
                        flags.append("high_amplitude")
                    if row["flat_channels"]:
                        flags.append("flat_channels")
                    row["quality"] = "+".join(flags) or "ok"
                    row["epoch_index"] = len(epochs)
                    epochs.append(transform_trial(trial, sfreq, config))
        rows.append(row)
    audit["events"] = len(rows)
    audit["valid_events"] = len(epochs)
    audit["annotation_durations"] = sorted(set(float(d) for d in raw.annotations.duration))
    samples = round((config["tmax"] - config["tmin"]) * config["sfreq"])
    return np.asarray(epochs, dtype=np.float64).reshape((-1, len(channels), samples)), rows, audit


def extract_edf(path, run=None, channels=CHANNELS, config=CONFIG, include_labels=True):
    infer_run(path, run)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    return extract_raw(raw, channels, config, include_labels)


def prepare(partition, root="data/raw", group="all"):
    split = manifest()
    channels = {"all": CHANNELS, "central": CENTRAL, "occipital": OCCIPITAL}[group]
    arrays, records, audits = [], [], []
    offset = 0
    for subject in split[partition]:
        for run in RUNS:
            path = edf_path(root, subject, run)
            try:
                x, rows, audit = extract_edf(path, channels=channels)
            except Exception as exc:
                audits.append({"subject": subject, "run": run, "status": "excluded", "reason": str(exc)})
                continue
            audit.update(subject=subject, run=run, status="ok")
            audits.append(audit)
            for row in rows:
                if row["epoch_index"] >= 0:
                    row["epoch_index"] += offset
                row.update(subject=subject, run=run, partition=partition,
                           trial_id=f"S{subject:03d}R{run:02d}E{row['event_index']:02d}")
                records.append(row)
            arrays.append(x)
            offset += len(x)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    name = f"{partition}_{group}"
    write_json(f"results/audit_{name}.json", audits)
    pd.DataFrame(records).to_csv(f"results/events_{name}.csv", index=False)
    if not arrays or offset == 0:
        raise ValueError("No usable trials; inspect data audit")
    np.save(out / f"{name}.npy", np.concatenate(arrays))
    print(f"Prepared {name}: {offset} valid trials", flush=True)


def load_prepared(partition, group="all"):
    name = f"{partition}_{group}"
    x = np.load(f"data/processed/{name}.npy", mmap_mode="r")
    meta = pd.read_csv(f"results/events_{name}.csv")
    meta = meta[meta.epoch_index >= 0].sort_values("epoch_index").reset_index(drop=True)
    assert np.array_equal(meta.epoch_index, np.arange(len(x)))
    return x, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["manifest", "download", "prepare"])
    parser.add_argument("--partition", choices=["development", "test"], default="development")
    parser.add_argument("--group", choices=["all", "central", "occipital"], default="all")
    parser.add_argument("--root", default="data/raw")
    args = parser.parse_args()
    if args.command == "manifest":
        print(json.dumps(manifest(), indent=2))
        write_json("config/preprocessing.json", CONFIG)
    elif args.command == "download":
        download(args.root)
    else:
        prepare(args.partition, args.root, args.group)

if __name__ == "__main__":
    main()
