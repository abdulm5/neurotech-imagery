# Demo narration / recording script

Target: 2 minutes 40 seconds. The accompanying demo.mp4 is captioned and has no audio. Use this script as a starting point, adapt it to your own words, and rehearse before the live defense.

## 0:00–0:16 — Can EEG predictions travel to a new person?

I built an offline classifier for imagined left versus right fist movement. The central question is whether a model works for a person whose recordings it has never seen. I selected forty subjects before fitting: thirty for development and ten held out until the model was frozen.

## 0:16–0:38 — One shared pipeline, from raw EDF to prediction

The loader validates the run because the same annotation codes mean different movements in other runs. It extracts seconds one to four after each cue, references and filters each trial independently, and flags large amplitudes. Invalid events remain visible in the prediction output. Training and inference use the same preprocessing code.

## 0:38–1:02 — The split changes what the score means

I compared random trials, a later run from familiar people, and entirely unfamiliar people. These are different deployment questions. The final held-out score was 53.6%, with substantial variation across individuals. The interval resamples subjects, not trials. Differences between split strategies are descriptive, not a pure causal estimate of identity leakage.

## 1:02–1:26 — What else could explain the predictions?

I checked a majority predictor, a metadata-only model, one hundred within-person and run label permutations, and subject identification across runs. Identity accuracy was 95.7%. That shows the features contain persistent person or recording information, but does not tell us exactly how much left-right decoding relies on it. The shuffled-label reference also cannot rule out visual cues or artifacts.

## 1:26–1:46 — Design decision: ship the spectral model

I compared log spectral power with logistic regression against four regularized common spatial patterns with shrinkage LDA. The preset selection rule chose spectral. I chose these small models over a neural network to preserve time for evaluation and make the transforms understandable. More model capacity would not fix a misleading split or cue confounding.

## 1:46–2:06 — A working raw-EDF prediction interface

The deliverable includes a trained model and a raw EDF command. It returns a row for every task cue, including quality flags or unavailable status. I verified that the command agrees with the evaluation predictions, and that changing annotation class codes without changing onsets leaves predictions unchanged.

## 2:06–2:26 — Weakest point: cue direction and task are entangled

The open-ended investigation compared central and occipital electrodes, scoring 54.4% and 53.9%. The visual target appears on the requested side, so cue and task are entangled. Regional predictability could reflect visual responses, volume conduction, or other signals. My weakest claim would be that the model decodes motor imagery alone; this protocol cannot establish that.

## 2:26–2:40 — What I would do next

Next I would separate cue direction from imagined hand, evaluate new recording days and devices, expand the cohort, and only then compare richer models under the same splits. The repository preserves the model, audit, predictions, and experiment log so every reported number can be checked.
