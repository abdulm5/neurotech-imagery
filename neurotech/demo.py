"""Build a 160-second captioned demo from actual saved results and predictions."""
import json
from pathlib import Path
import subprocess
import textwrap

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def main():
    test = read("results/test_summary.json")
    selected = read("artifacts/selection.json")["selected_model"]
    identity = read("results/identity.json")
    central = read("results/grouped_central_summary.json")["mean_subject_balanced_accuracy"]
    occipital = read("results/grouped_occipital_summary.json")["mean_subject_balanced_accuracy"]
    comparison = pd.read_csv("results/comparison.csv").set_index("experiment")
    predicted = pd.read_csv("results/example_predictions.csv")
    pct = lambda x: f"{100*x:.1f}%"
    scenes = [
        (16, "Can EEG predictions travel to a new person?", None,
         ["Left versus right imagined fist movement · PhysioNet EEGMMIDB",
          "40 fixed subjects: 30 for development, 10 held out until the model was frozen.",
          f"Final result: {pct(test['mean_subject_balanced_accuracy'])} mean subject balanced accuracy.",
          "This captioned demo summarizes measured results; no voiceover."],
         "I built an offline classifier for imagined left versus right fist movement. The central question is whether a model works for a person whose recordings it has never seen. I selected forty subjects before fitting: thirty for development and ten held out until the model was frozen."),
        (22, "One shared pipeline, from raw EDF to prediction", None,
         ["Only imagery runs 4, 8, 12; T1 = left, T2 = right.",
          "Use seconds 1–4 after each cue; normalize channel order and average-reference.",
          "Resample each isolated trial to 160 Hz; apply an 8–30 Hz bandpass.",
          "Flag >200 µV peak-to-peak; keep usable trials in the main score.",
          "Annotation class codes never enter the classifier.",
          "Audit finding: S100 reports 128 Hz and longer task annotations."],
         "The loader validates the run because the same annotation codes mean different movements in other runs. It extracts seconds one to four after each cue, references and filters each trial independently, and flags large amplitudes. Invalid events remain visible in the prediction output. Training and inference use the same preprocessing code."),
        (24, "The split changes what the score means", "docs/figures/generalization.png",
         [f"Held-out mean: {pct(test['mean_subject_balanced_accuracy'])}; subject-bootstrap 95% interval: "
          f"{pct(test['subject_bootstrap_95ci'][0])}–{pct(test['subject_bootstrap_95ci'][1])}.",
          "The interval includes 50%; reliable above-chance generalization is not established."],
         f"I compared random trials, a later run from familiar people, and entirely unfamiliar people. These are different deployment questions. The final held-out score was {pct(test['mean_subject_balanced_accuracy'])}, with substantial variation across individuals. The interval resamples subjects, not trials. Differences between split strategies are descriptive, not a pure causal estimate of identity leakage."),
        (24, "What else could explain the predictions?", "docs/figures/controls.png",
         [f"Subject identification across runs: {pct(identity['accuracy'])}; chance: {pct(identity['uniform_chance'])}.",
          "Also tested majority labels, run/trial-position metadata, and 100 label permutations."],
         f"I checked a majority predictor, a metadata-only model, one hundred within-person and run label permutations, and subject identification across runs. Identity accuracy was {pct(identity['accuracy'])}. That shows the features contain persistent person or recording information, but does not tell us exactly how much left-right decoding relies on it. The shuffled-label reference also cannot rule out visual cues or artifacts."),
        (20, f"Design decision: ship the {selected} model", None,
         [f"Spectral grouped score: {pct(comparison.loc['grouped_spectral','mean_subject_balanced_accuracy'])}.",
          f"CSP grouped score: {pct(comparison.loc['grouped_csp','mean_subject_balanced_accuracy'])}.",
          "Fixed rule: choose CSP only if it exceeds spectral by at least 1 percentage point.",
          "Two interpretable power-based models; no parameter search or test-person fitting."],
         f"I compared log spectral power with logistic regression against four regularized common spatial patterns with shrinkage LDA. The preset selection rule chose {selected}. I chose these small models over a neural network to preserve time for evaluation and make the transforms understandable. More model capacity would not fix a misleading split or cue confounding."),
        (20, "A working raw-EDF prediction interface", None,
         ["uv run python -m neurotech.predict --edf data/raw/S040/S040R04.edf " + "\\",
          "  --model artifacts/model.joblib --output predictions.csv", "",
          *[f"Cue {int(r.event_index):02d}, onset {r.onset:.1f}s → {r.predicted_label} [{r.quality}]"
            for r in predicted.head(3).itertuples()],
          "Shown: actual saved CLI output, verified against evaluation predictions."],
         "The deliverable includes a trained model and a raw EDF command. It returns a row for every task cue, including quality flags or unavailable status. I verified that the command agrees with the evaluation predictions, and that changing annotation class codes without changing onsets leaves predictions unchanged."),
        (20, "Weakest point: cue direction and task are entangled", None,
         [f"Central-only score: {pct(central)}. Occipital-only score: {pct(occipital)}.",
          "Left and right visual targets are correlated with the imagined hand.",
          "Regional models use separate within-group references and identical splits.",
          "Predictive occipital EEG is not proof of a visual source—or motor specificity."],
         f"The open-ended investigation compared central and occipital electrodes, scoring {pct(central)} and {pct(occipital)}. The visual target appears on the requested side, so cue and task are entangled. Regional predictability could reflect visual responses, volume conduction, or other signals. My weakest claim would be that the model decodes motor imagery alone; this protocol cannot establish that."),
        (14, "What I would do next", None,
         ["Separate cue direction from imagined hand in the experimental protocol.",
          "Test new recording days and devices, then expand the fixed subject cohort.",
          "Compare richer models under the same subject-disjoint evaluation.",
          "Code, trained model, trial predictions, audit, and report are included."],
         "Next I would separate cue direction from imagined hand, evaluate new recording days and devices, expand the cohort, and only then compare richer models under the same splits. The repository preserves the model, audit, predictions, and experiment log so every reported number can be checked.")]
    out = Path("data/demo")
    out.mkdir(parents=True, exist_ok=True)
    script = ["# Demo narration / recording script", "", "Target: 2 minutes 40 seconds. The accompanying demo.mp4 is captioned and has no audio. Use this script as a starting point, adapt it to your own words, and rehearse before the live defense.", ""]
    concat = []
    elapsed = 0
    for i, (duration, title, chart, captions, narration) in enumerate(scenes):
        fig = plt.figure(figsize=(12.8,7.2), dpi=100, facecolor="#f8f9f6")
        fig.text(.055,.94,"NEUROTECH / EEGMMIDB",fontsize=12,color="#237a87",weight="bold")
        fig.text(.055,.86,title,fontsize=25,weight="bold",color="#172d36")
        if chart:
            ax = fig.add_axes([.055,.27,.89,.51])
            ax.imshow(plt.imread(chart))
            ax.axis("off")
            y = .19
            fontsize = 15
        else:
            y = .72
            fontsize = 18 if i != 5 else 15
        for caption in captions:
            wrapped = textwrap.wrap(caption, width=103 if chart else 91) or [""]
            for line in wrapped:
                fig.text(.065,y,line,fontsize=fontsize,color="#253b44",family="DejaVu Sans")
                y -= .047 if chart else .057
            y -= .025 if not chart else .01
        fig.text(.055,.055,f"{i+1:02d} / {len(scenes):02d} · Captioned demonstration · Saved experimental results",fontsize=11,color="#62787d")
        frame = out / f"frame_{i:02d}.png"
        fig.savefig(frame,dpi=100)
        plt.close(fig)
        concat += [f"file '{frame.resolve()}'", f"duration {duration}"]
        start, end = f"{elapsed//60}:{elapsed%60:02d}", f"{(elapsed+duration)//60}:{(elapsed+duration)%60:02d}"
        script += [f"## {start}–{end} — {title}", "", narration, ""]
        elapsed += duration
    concat.append(f"file '{frame.resolve()}'")
    (out / "frames.txt").write_text("\n".join(concat)+"\n")
    Path("docs/demo-script.md").write_text("\n".join(script))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(out / "frames.txt"),
                    "-vf", "fps=12", "-t", str(elapsed), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", "docs/demo.mp4"], check=True)
    print(f"Created docs/demo.mp4 ({elapsed}s), docs/demo-script.md")

if __name__ == "__main__":
    main()
