# Imagined left/right fist classification

An offline EEGMMIDB classifier with subject-disjoint evaluation, explicit confound checks, and raw-EDF prediction. The study asks what generalizes to an unfamiliar person, not how high a random-split score can go.

**Measured result:** the frozen spectral model achieved **53.6% mean subject balanced accuracy** on 10 held-out people (450 trials), with a **48.3–60.2%** subject-bootstrap 95% interval. The interval includes chance; reliable above-chance generalization is not established. All 12 tests pass, and the raw-EDF interface reproduces all 450 saved holdout predictions in a separate locked environment. See [the report](docs/report.md) for the full evaluation and limitations.

## Quick start

Install [uv](https://docs.astral.sh/uv/) and Python 3.12, then:

```bash
uv sync --locked
uv run python -m neurotech.predict --edf /path/S099R04.edf --model artifacts/model.joblib --output predictions.csv
```

The EDF must be a left/right imagery recording: run 4, 8, or 12, with EEGMMIDB's 64 channels and T1/T2 task-onset annotations. For a renamed EDF, add `--run 4` (or 8/12). Execution and hands/feet runs are rejected. This is cue-aligned offline prediction, not rest detection or continuous streaming.

```python
from neurotech.predict import predict_edf
predictions = predict_edf("/path/S099R04.edf", "artifacts/model.joblib")
```

Each task event gets an onset, duration, `left`/`right` prediction, and quality status. Incomplete, nonfinite, or entirely flat events receive `unavailable`; high-amplitude but usable events are predicted and flagged. Missing required channels cause a clear error. Class annotation values are never classifier inputs. Only load trusted joblib artifacts (Python deserialization can execute code).

## Reproduce the study

Run commands from the repository root. Raw EDFs and processed arrays stay in ignored `data/`; model/results/configuration are included. Downloading verifies upstream SHA256 checksums. No GPU or external service credentials are needed for analysis.

```bash
uv run python -m neurotech.data manifest
uv run python -m neurotech.data download
uv run python -m neurotech.data prepare --partition development
uv run python -m neurotech.evaluate development --permutations 100
uv run python -m neurotech.evaluate test
uv run python -m neurotech.report
uv run pytest -q
uv run python -m neurotech.verify
```

The development command also prepares central/occipital channel groups, runs diagnostics, selects a model, and fits the deliverable on development subjects only. Test preparation occurs after selection. The test command refuses to overwrite a completed holdout evaluation, and development refuses to rerun after it. For an independent reproduction, copy the source/config into a fresh working directory without the saved result/model outputs, or move those outputs to a preserved archive first. Do not use repeated test scores to change the model.

Two fixed models are compared: log spectral power + logistic regression, and regularized four-component CSP + shrinkage LDA. Selection uses five subject-grouped folds and mean per-person balanced accuracy; spectral wins ties within one percentage point. Seed, selected subjects, time window, channel order, and quality criteria are saved. See [the full report](docs/report.md), [comparison table](results/comparison.csv), [experiment log](results/experiments.jsonl), and [demo script](docs/demo-script.md).

## Project map

- `neurotech/data.py`: fixed manifest, verified downloads, audit, shared trial extraction.
- `neurotech/models.py`: deterministic features and training-only sklearn transforms.
- `neurotech/evaluate.py`: split comparisons, controls, selection, frozen holdout.
- `neurotech/predict.py`: raw EDF function and CLI.
- `neurotech/report.py`: figures/report from saved results, without training.
- `neurotech/verify.py`: model-hash and prediction parity checks across all held-out EDFs, without refitting.
- `tests/`: real EDF round-trip, CLI parity, annotation-swap, channel order, trial isolation, invalid-input and split checks.
- `artifacts/`: trained pipeline, metadata, selection and model hash at test time.
- `results/`: audits, event tables, fold membership, predictions, metrics and experiment attempts.

## Interpretation

A random trial split shares people and recording runs across train/test. The headline uses held-out people. Later-run evaluation is not a separate-day/session test. Regional and metadata controls probe alternative explanations but do not prove the model decodes motor imagery alone. Visual targets are correlated with left/right task labels. No clinical, online, cross-device or universal-user performance is claimed.

Balanced accuracy averages left and right recall. The headline then averages that score across people so people with more valid trials do not dominate it. Confidence intervals bootstrap people, not individual EEG trials. The high-amplitude sensitivity result removes flagged evaluation trials without retraining and reports retained coverage.

## Data attribution

Data: [PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), Gerwin Schalk, DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P), licensed under Open Data Commons Attribution v1.0. Raw data are not redistributed here.

Schalk et al., *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System*, IEEE TBME 51(6):1034–1043 (2004), doi:10.1109/TBME.2004.827072. Goldberger et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23):e215–e220 (2000).
