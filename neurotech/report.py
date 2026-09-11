"""Generate figures and a source-linked report from saved results, without refitting."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import manifest


def read(path):
    return json.loads(Path(path).read_text())


def pct(value):
    return f"{100*value:.1f}%"


def main():
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False,
                         "axes.spines.right":False, "figure.dpi":150})
    names = ["random_spectral", "later_run_spectral", "grouped_spectral", "random_csp",
             "later_run_csp", "grouped_csp", "grouped_majority", "grouped_metadata",
             "grouped_central", "grouped_occipital"]
    rows = []
    for name in names:
        s = read(f"results/{name}_summary.json")
        rows.append({"experiment":name, "mean_subject_balanced_accuracy":s["mean_subject_balanced_accuracy"],
                     "accuracy":s["accuracy"], "train_score":s["mean_fold_train_subject_balanced_accuracy"],
                     "subjects":s["subjects"], "trials":s["trials"],
                     "ci_low":s["subject_bootstrap_95ci"][0], "ci_high":s["subject_bootstrap_95ci"][1]})
    table = pd.DataFrame(rows)
    table.to_csv("results/comparison.csv", index=False)
    selection = read("artifacts/selection.json")
    test = read("results/test_summary.json")
    per_subject = pd.read_csv("results/test_subjects.csv").sort_values("balanced_accuracy")
    perm = read("results/permutation.json")
    identity = read("results/identity.json")
    fig, axes = plt.subplots(1,2,figsize=(11,4), layout="constrained")
    positions = np.arange(3)
    for offset, model, color in [(-.16,"spectral","#237a87"),(.16,"csp","#d57939")]:
        subset = table.set_index("experiment").loc[[f"random_{model}",f"later_run_{model}",f"grouped_{model}"]]
        axes[0].bar(positions+offset, subset.mean_subject_balanced_accuracy, .3, label=model, color=color)
    axes[0].set_xticks(positions, ["Random trials","Later run","Unseen subjects"])
    axes[0].set(ylabel="Mean subject balanced accuracy",ylim=(0,1),title="Development: evaluation changes the question")
    axes[0].axhline(.5, color="gray",linestyle="--")
    axes[0].legend()
    axes[1].bar(per_subject.subject.astype(str), per_subject.balanced_accuracy,color="#237a87")
    axes[1].axhline(.5,color="gray",linestyle="--")
    axes[1].set(xlabel="Held-out subject",ylabel="Balanced accuracy",ylim=(0,1),title="Final test: variation across people")
    fig.savefig("docs/figures/generalization.png")
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,3.5), layout="constrained")
    axes[0].hist(perm["null_scores"],bins=15,color="#8cbec4")
    axes[0].axvline(perm["observed"],color="#c56b30",label="Observed")
    axes[0].set(title="Within-person/run shuffled labels",xlabel="Mean subject balanced accuracy",ylabel="Permutations")
    axes[0].legend()
    regional = table.set_index("experiment").loc[["grouped_central","grouped_occipital"]]
    axes[1].bar(["Central","Occipital"],regional.mean_subject_balanced_accuracy,color=["#237a87","#d57939"])
    axes[1].axhline(.5,color="gray",linestyle="--")
    axes[1].set(ylim=(0,1),title="Regional spectral models",ylabel="Mean subject balanced accuracy")
    cm = np.array(test["confusion_matrix_left_right"])
    axes[2].imshow(cm,cmap="Blues")
    for (i,j), value in np.ndenumerate(cm):
        axes[2].text(j,i,str(value),ha="center",va="center",color="white" if value > .6*cm.max() else "black")
    axes[2].set(xticks=[0,1],yticks=[0,1],xticklabels=["Left","Right"],yticklabels=["Left","Right"],
                xlabel="Prediction",ylabel="Truth",title="Held-out confusion matrix")
    fig.savefig("docs/figures/controls.png")
    plt.close(fig)
    audits = read("results/audit_development_all.json") + read("results/audit_test_all.json")
    events = pd.concat([pd.read_csv(f"results/events_{p}_all.csv") for p in ["development","test"]])
    excluded = events[events.epoch_index < 0]
    failed = [a for a in audits if a["status"] != "ok"]
    split = manifest()
    lines = ["# What does this motor imagery classifier establish?", "",
        f"The frozen **{selection['selected_model']}** model achieved **{pct(test['mean_subject_balanced_accuracy'])} mean per-subject balanced accuracy** on {test['subjects']} held-out people ({test['trials']} usable trials). The subject-bootstrap 95% interval is **{pct(test['subject_bootstrap_95ci'][0])}–{pct(test['subject_bootstrap_95ci'][1])}**. This interval resamples the observed people; it does not cover all sources of uncertainty.", "",
        ("The interval includes 50%. This small held-out test does not establish reliable above-chance generalization; the positive development permutation result does not override that uncertainty."
         if test['subject_bootstrap_95ci'][0] <= .5 <= test['subject_bootstrap_95ci'][1]
         else "Interpret this interval conditional on the observed subjects and frozen model; it does not resolve causal attribution or other datasets."), "",
        "This is an offline, cue-aligned, same-dataset experiment. It does not establish online control, cross-day performance, transfer to other devices, or that the source of every predictive feature is motor imagery.","",
        "## Data and preprocessing", "",
        f"A seeded permutation (seed {split['seed']}) selected 40 of 109 people before fitting: 30 development and 10 held out. All three imagery runs (4, 8, 12) were requested for each person. Development IDs: {split['development']}. Held-out IDs: {split['test']}.", "",
        f"The audit contains {len(audits)} recording entries, {len(failed)} unreadable/excluded recordings, {len(events)} task cues, {len(excluded)} unusable events, and {int(events.high_amplitude.sum())} high-amplitude flags. The raw sampling rates observed were {sorted(set(a['sampling_rate'] for a in audits if a['status']=='ok'))}. Every downloaded EDF was checked against PhysioNet's SHA256 manifest. Full durations, original channel names, counts, flags and exclusions are in results/audit_*.json and results/events_*.csv.", "",
        "T1 means left and T2 means right only in the supported unilateral imagery runs. We extract [1,4) seconds after the cue, require at least four seconds of annotated task duration, standardize channel order, average-reference each trial, resample that isolated trial to 160 Hz if necessary, and apply a fourth-order Butterworth 8–30 Hz bandpass forward and backward with one second of reflected padding on each side. No samples from adjacent trials enter filtering. Reflection creates edge assumptions; the one-second cue delay reduces early evoked responses but does not eliminate sustained cue information.","",
        "EEG is measured in volts. A raw-channel peak-to-peak value above 200 µV is flagged; finite complete trials remain in the primary score. Entirely flat and nonfinite trials are unavailable. Individual flat channels are flagged and retained. No manual cleaning, ICA, interpolation, or test-person calibration is performed. The flagged-trial sensitivity analysis reuses the fitted model and removes only flagged evaluation trials; it is not a separately trained clean-data model.","",
        "## Models and selection", "",
        "The spectral model uses Welch power (one-second Hann windows, 50% overlap), summed over 8≤f<13 and 13≤f<30 Hz per channel, then log power, training-fitted standardization, and logistic regression (C=1). The CSP model learns four spatial components with Ledoit–Wolf covariance regularization, then shrinkage LDA. CSP combines electrodes to emphasize class-dependent power; these are statistical patterns, not localized brain sources.", "",
        "Five development folds hold out entire people. Both model configurations were fixed in advance. CSP wins only if its mean subject balanced accuracy exceeds the spectral model by at least one percentage point. Every scaler, CSP fit, and classifier uses training data only. The selected pipeline is then fitted on all usable development trials, hashed, and evaluated once on the held-out people. Test people remain excluded from the shipped model.", "",
        "## Results", "",
        "| Development experiment | Subject balanced accuracy | Training score | Ordinary accuracy |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {pct(row['mean_subject_balanced_accuracy'])} | {pct(row['train_score'])} | {pct(row['accuracy'])} |")
    lines += ["", "![Generalization](figures/generalization.png)", "",
        "Random splits share people and runs; later-run splits share people; grouped splits share neither people nor their runs. The different scores describe different deployment conditions. Their difference is not a causal estimate of identity leakage because training sets and sample sizes also differ. Training scores are resubstitution scores averaged over folds and are optimistic by construction.","",
        f"Held-out ordinary accuracy is {pct(test['accuracy'])}; pooled balanced accuracy is {pct(test['balanced_accuracy'])}. The training-majority baseline achieves {pct(test['training_majority_baseline']['accuracy'])} ordinary accuracy and 50% balanced accuracy. Individual balanced accuracies range from {pct(per_subject.balanced_accuracy.min())} to {pct(per_subject.balanced_accuracy.max())}. Excluding high-amplitude test trials retains {pct(test['unflagged_coverage'])} of usable trials and yields {pct(test['unflagged']['mean_subject_balanced_accuracy']) if test['unflagged'] else 'no evaluable score'}. The remaining subset contains only {test['unflagged']['trials'] if test['unflagged'] else 0} trials from {test['unflagged']['subjects'] if test['unflagged'] else 0} people; this is not evidence of improvement from cleaning. Subject-level scores are in results/test_subjects.csv.","",
        "## Alternative explanations and the open-ended investigation", "",
        f"The fixed spectral pipeline's observed grouped score is {pct(perm['observed'])}. Across {perm['iterations']} within-person/run label permutations, the plus-one reference p-value is {perm['p_value']:.4f}. Scaling and classification are refitted in every fold for every permutation. This is a limited exchangeability reference: shuffling destroys temporal label structure, so it does not rule out order effects, artifacts, or visual information.","",
        f"Predicting person identity from runs 4/8 and evaluating on run 12 achieved {pct(identity['accuracy'])} accuracy, versus {pct(identity['uniform_chance'])} uniform chance and {pct(identity['majority_accuracy'])} majority accuracy. This quantifies persistent person/recording information, not its causal contribution to left/right predictions.","",
        "The metadata-only logistic model uses run number and task-event position. It tests a simple order/imbalance explanation; failure of this linear control does not exclude nonlinear schedules. The majority predictor is learned from each training fold. These controls are stronger context than a bare 50% reference but are not exhaustive.","",
        f"The central-only spectral model scored {pct(float(regional.iloc[0].mean_subject_balanced_accuracy))}; the occipital-only model scored {pct(float(regional.iloc[1].mean_subject_balanced_accuracy))}. Each uses its own within-group reference, the same trials, and the same held-out-person folds. Central channels are Fc3/Fc1/Fcz/Fc2/Fc4, C3/C1/Cz/C2/C4, Cp3/Cp1/Cpz/Cp2/Cp4. Occipital channels are Po7/Po3/Poz/Po4/Po8/O1/Oz/O2. The groups differ in size, so this is a diagnostic, not a controlled localization experiment.","",
        "The dataset's left/right visual target is correlated with the requested movement. Occipital predictability can be consistent with visual responses, volume conduction, or other shared signals; it cannot isolate their causes. A low occipital score would likewise not prove motor specificity. This cue/task entanglement is the weakest point in a claim of pure motor imagery decoding.","",
        "![Controls](figures/controls.png)", "",
        "## Design decision, limits, and next steps", "",
        "We chose two small power-based models instead of a neural network. Their transforms can be inspected, they fit quickly on CPU, and this preserves time for split comparisons and raw-EDF consistency checks. The alternative could learn richer structure, but extra capacity would not resolve cue confounding or make a random trial split a valid estimate for a new person.","",
        "The strongest defensible claim is limited to the observed held-out EEGMMIDB people and this preprocessing. Forty selected people are not a representative sample of every BCI user. Quality thresholds and the fixed time window affect coverage. The bootstrap is conditional on the selected model and observed subjects. The selected model's development score has selection optimism; the holdout is the final independent check. Class balance, temporal dependence, and residual artifacts remain limitations. No accuracy threshold was used to exclude a person.","",
        "The fixed raw-amplitude threshold flags many trials. The unflagged analysis can remove entire people and can leave only one class for others. Mean subject balanced accuracy excludes people without both remaining classes; JSON summaries explicitly report this coverage. Therefore its score is conditional on a changed cohort and should not be described as a controlled estimate of the benefit of cleaning.", "",
        "With more time: expand the prespecified cohort, compare independent recording days/devices, use a protocol separating cue direction from imagined hand, add stronger temporal/order controls, test reference and time-window sensitivity under nested subject validation, and then compare a compact neural model with the same splits. Separate calibration-enabled performance from truly unseen-person inference.","",
        "## Reproducibility and sources", "",
        "See README.md for locked installation and commands. results/experiments.jsonl records attempts and timings; config/split.json records subject selection; artifacts/test_freeze.json records the exact evaluated model hash. Trial-level predictions include subject/run/event/fold membership. Preprocessing and annotations used for extraction are shared with the prediction CLI; annotation class codes never enter its model features.","",
        "- [PhysioNet EEGMMIDB 1.0.0 and protocol](https://physionet.org/content/eegmmidb/1.0.0/)",
        "- [MNE run mapping](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html)",
        "- [MNE CSP example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html) — methodological reference; its hands/feet example is not our left/right task.",
        "- Schalk et al. (2004), BCI2000, IEEE TBME 51(6):1034–1043, doi:10.1109/TBME.2004.827072.",
        "- Goldberger et al. (2000), PhysioBank, PhysioToolkit, and PhysioNet, Circulation 101(23):e215–e220.", ""]
    sensitivity_path = Path("results/sampling_rate_sensitivity.json")
    if sensitivity_path.exists():
        sensitivity = read(sensitivity_path)
        where = lines.index("## Design decision, limits, and next steps")
        affected = sensitivity["excluded_subjects"]
        lines[where:where] = ["## Additional discovery: inconsistent sampling metadata", "",
            f"The development audit found non-160-Hz EDF headers for subjects {affected}. Subject 100's three runs report 128 Hz and task annotations around 5.1 seconds, compared with about 4.1 seconds in most recordings. This was found before model fitting. We follow the header's stated rate and resample; we cannot establish from these files alone whether the metadata or acquisition timing is wrong.", "",
            f"A prespecified-after-audit sensitivity check removes these subjects from both training and validation while preserving the original fold assignments for everyone else. The fixed spectral model scores {pct(sensitivity['summary']['mean_subject_balanced_accuracy'])} mean subject balanced accuracy. This is a descriptive robustness check on a different cohort, not a reason to replace the original primary result or select a different model. No held-out data informed this check.", ""]
    Path("docs/report.md").write_text("\n".join(lines))
    print("Generated docs/report.md, comparison.csv and figures")

if __name__ == "__main__":
    main()
