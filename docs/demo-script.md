# Recording script with slide cues

Target: about **2 minutes 50 seconds**. Present the PowerPoint and advance manually. Read only the narration paragraphs. **Switch after the last sentence of each section.** Times are pacing guides, not exact deadlines.

## Slide 1 | 0:00–0:22 | Can EEG predictions travel to a new person?

I built a classifier for imagined left versus right fist movement. I used forty people: thirty for development and ten held out for testing. The headline result was 53.6 percent balanced accuracy, averaged across those ten people. That is close to chance, so the main finding is limited generalization.

**At about 0:22, switch to slide 2.**

## Slide 2 | 0:22–0:46 | One shared pipeline, from raw EDF to prediction

I used imagery runs four, eight, and twelve, taking seconds one to four after each cue. I standardized and filtered each trial independently, and flagged large signals. One unexpected finding was that subject one hundred reports 128 hertz instead of 160, with longer task annotations. I documented this and checked sensitivity separately.

**At about 0:46, switch to slide 3.**

## Slide 3 | 0:46–1:13 | The split changes what the score means

These charts show why the evaluation matters. The spectral model scored 51.3 percent with random trials, 57.8 percent on a later run, and 53.3 percent on unfamiliar development subjects. The final test scored 53.6 percent, with a 95 percent interval from 48.3 to 60.2. That includes chance, so reliable performance on new people remains unproven.

**At about 1:13, switch to slide 4.**

## Slide 4 | 1:13–1:38 | What else could explain the predictions?

I tested alternative explanations with majority labels, recording metadata, and one hundred label permutations. The first two controls were around fifty percent. The striking result was subject identification: 95.7 percent accuracy, versus 3.3 percent chance. The features clearly carry person or recording information. That does not prove the hand classifier relies on it.

**At about 1:38, switch to slide 5.**

## Slide 5 | 1:38–1:58 | Design decision: ship the spectral model

Both candidate models scored about 53.3 percent on unfamiliar development subjects. Our preset rule selected the simpler spectral model unless CSP improved by a full percentage point. I chose these interpretable models over a neural network so I could spend more effort checking evaluation assumptions and explaining the results.

**At about 1:58, switch to slide 6.**

## Slide 6 | 1:58–2:15 | A working raw-EDF prediction interface

This command takes a raw EDF file and returns a prediction for each task cue, alongside quality flags. The examples shown are real outputs. All twelve tests passed, and a separate environment reproduced all four hundred and fifty held-out predictions exactly.

**At about 2:15, switch to slide 7.**

## Slide 7 | 2:15–2:35 | Weakest point: cue direction and task are entangled

My weakest point is that visual cues could contribute to the predictions. Central electrodes scored 54.4 percent, while occipital electrodes scored 53.9 percent. The visual target appears on the same side as the imagined hand. These results cannot establish that the model decodes motor imagery alone.

**At about 2:35, switch to slide 8.**

## Slide 8 | 2:35–2:50 | What I would do next

Next, I would separate visual cue direction from the imagined hand and test new recording days and devices. Then I would expand the cohort and compare richer models. The repository contains the code, model, and results needed to inspect the work.

**At about 2:50, stop recording.**
