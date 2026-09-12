"""Build a timed captioned demo from actual saved results and predictions."""
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


def build_scenes():
    test = read("results/test_summary.json")
    selected = read("artifacts/selection.json")["selected_model"]
    identity = read("results/identity.json")
    central = read("results/grouped_central_summary.json")["mean_subject_balanced_accuracy"]
    occipital = read("results/grouped_occipital_summary.json")["mean_subject_balanced_accuracy"]
    comparison = pd.read_csv("results/comparison.csv").set_index("experiment")
    predicted = pd.read_csv("results/example_predictions.csv")
    pct = lambda x: f"{100*x:.1f}%"
    scenes = [
        (22, "Can EEG predictions travel to a new person?", None,
         ["Left versus right imagined fist movement · PhysioNet EEGMMIDB",
          "40 fixed subjects: 30 for development, 10 held out until the model was frozen.",
          f"Final result: {pct(test['mean_subject_balanced_accuracy'])} mean subject balanced accuracy."],
         'I built a classifier for imagined left versus right fist movement. I used forty people: thirty for development and ten held out for testing. The headline result was 53.6 percent balanced accuracy, averaged across those ten people. That is close to chance, so the main finding is limited generalization.'),
        (24, "One shared pipeline, from raw EDF to prediction", None,
         ["Only imagery runs 4, 8, 12; T1 = left, T2 = right.",
          "Use seconds 1–4 after each cue; normalize channel order and average-reference.",
          "Resample each isolated trial to 160 Hz; apply an 8–30 Hz bandpass.",
          "Flag >200 µV peak-to-peak; keep usable trials in the main score.",
          "Annotation class codes never enter the classifier.",
          "Audit finding: S100 reports 128 Hz and longer task annotations."],
         'I used imagery runs four, eight, and twelve, taking seconds one to four after each cue. I standardized and filtered each trial independently, and flagged large signals. One unexpected finding was that subject one hundred reports 128 hertz instead of 160, with longer task annotations. I documented this and checked sensitivity separately.'),
        (27, "The split changes what the score means", "docs/figures/generalization.png",
         [f"Held-out mean: {pct(test['mean_subject_balanced_accuracy'])}; subject-bootstrap 95% interval: "
          f"{pct(test['subject_bootstrap_95ci'][0])}–{pct(test['subject_bootstrap_95ci'][1])}.",
          "The interval includes 50%; reliable above-chance generalization is not established."],
         'These charts show why the evaluation matters. The spectral model scored 51.3 percent with random trials, 57.8 percent on a later run, and 53.3 percent on unfamiliar development subjects. The final test scored 53.6 percent, with a 95 percent interval from 48.3 to 60.2. That includes chance, so reliable performance on new people remains unproven.'),
        (25, "What else could explain the predictions?", "docs/figures/controls.png",
         [f"Subject identification across runs: {pct(identity['accuracy'])}; chance: {pct(identity['uniform_chance'])}.",
          "Also tested majority labels, run/trial-position metadata, and 100 label permutations."],
         'I tested alternative explanations with majority labels, recording metadata, and one hundred label permutations. The first two controls were around fifty percent. The striking result was subject identification: 95.7 percent accuracy, versus 3.3 percent chance. The features clearly carry person or recording information. That does not prove the hand classifier relies on it.'),
        (20, f"Design decision: ship the {selected} model", None,
         [f"Spectral grouped score: {pct(comparison.loc['grouped_spectral','mean_subject_balanced_accuracy'])}.",
          f"CSP grouped score: {pct(comparison.loc['grouped_csp','mean_subject_balanced_accuracy'])}.",
          "Fixed rule: choose CSP only if it exceeds spectral by at least 1 percentage point.",
          "Two interpretable power-based models; no parameter search or test-person fitting."],
         'Both candidate models scored about 53.3 percent on unfamiliar development subjects. Our preset rule selected the simpler spectral model unless CSP improved by a full percentage point. I chose these interpretable models over a neural network so I could spend more effort checking evaluation assumptions and explaining the results.'),
        (17, "A working raw-EDF prediction interface", None,
         ["uv run python -m neurotech.predict --edf data/raw/S040/S040R04.edf " + "\\",
          "  --model artifacts/model.joblib --output predictions.csv", "",
          *[f"Cue {int(r.event_index):02d}, onset {r.onset:.1f}s → {r.predicted_label} [{r.quality}]"
            for r in predicted.head(3).itertuples()],
          "Shown: actual saved CLI output, verified against evaluation predictions."],
         'This command takes a raw EDF file and returns a prediction for each task cue, alongside quality flags. The examples shown are real outputs. All twelve tests passed, and a separate environment reproduced all four hundred and fifty held-out predictions exactly.'),
        (20, "Weakest point: cue direction and task are entangled", None,
         [f"Central-only score: {pct(central)}. Occipital-only score: {pct(occipital)}.",
          "Left and right visual targets are correlated with the imagined hand.",
          "Regional models use separate within-group references and identical splits.",
          "Predictive occipital EEG is not proof of a visual source—or motor specificity."],
         'My weakest point is that visual cues could contribute to the predictions. Central electrodes scored 54.4 percent, while occipital electrodes scored 53.9 percent. The visual target appears on the same side as the imagined hand. These results cannot establish that the model decodes motor imagery alone.'),
        (15, "What I would do next", None,
         ["Separate cue direction from imagined hand in the experimental protocol.",
          "Test new recording days and devices, then expand the fixed subject cohort.",
          "Compare richer models under the same subject-disjoint evaluation.",
          "Code, trained model, trial predictions, audit, and report are included."],
         'Next, I would separate visual cue direction from the imagined hand and test new recording days and devices. Then I would expand the cohort and compare richer models. The repository contains the code, model, and results needed to inspect the work.')]
    return scenes


def main():
    scenes = build_scenes()
    out = Path("data/demo")
    out.mkdir(parents=True, exist_ok=True)
    script = ["# Demo narration / recording script", "", "Target: about 2 minutes 50 seconds. Present the PowerPoint and advance manually. Read only the narration paragraphs. Switch after the last sentence of each section. Times are pacing guides, not exact deadlines.", ""]
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
        frame = out / f"frame_{i:02d}.png"
        fig.savefig(frame,dpi=100)
        plt.close(fig)
        concat += [f"file '{frame.resolve()}'", f"duration {duration}"]
        start, end = f"{elapsed//60}:{elapsed%60:02d}", f"{(elapsed+duration)//60}:{(elapsed+duration)%60:02d}"
        script += [f"## Slide {i+1} | {start}–{end} | {title}", "", narration, "", (f"**At about {end}, switch to slide {i+2}.**" if i < len(scenes)-1 else f"**At about {end}, stop recording.**"), ""]
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
