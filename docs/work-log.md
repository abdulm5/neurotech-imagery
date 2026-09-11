# Implementation and operational log

The study configuration was fixed before training: seed 20260911, 40 subjects, 30/10 person split, imagery runs 4/8/12, 1–4 second window, two fixed models, five subject-grouped folds, 100 spectral label permutations. No target accuracy or performance-based subject exclusions.

- The repository started empty. Implemented shared extraction, model pipelines, evaluation, auditing, inference, tests, and report generation.
- Dependency setup initially failed because the sandbox could not access uv's cache; moving the cache exposed a macOS system-configuration sandbox failure. Installing with the required access succeeded. Exact dependencies are in uv.lock, Python 3.12 in .python-version.
- The first data download attempt could not resolve the host in the network-restricted sandbox. Retried with network access. Downloads verify the upstream SHA256 list and use up to three attempts per file.
- GitHub authentication looked invalid in the sandbox but succeeded with network/keychain access. No login change was needed.
- First test run: 8 passed, 1 failed. The failure was the test's treatment of empty CSV fields: pandas reloads blank strings as missing values. Normalized the channel-flag field in the round-trip assertion. The corrected run passed all 9 tests.

Model experiment configurations, outcomes, timestamps, runtimes, and exceptions are written to results/experiments.jsonl. Download outcomes are in results/download.json. Tests use synthetic EEG exported to real EDF, without fitting on held-out study subjects.

- A pre-fitting header audit of development recordings found that subject 100's three runs report 128 Hz and task annotations around 5.1 seconds, unlike the nominal 160 Hz / approximately 4.1-second protocol. Added a development-only spectral sensitivity analysis excluding that subject from both sides of the existing folds. The primary cohort, model-selection rule, and held-out subjects remain unchanged. The files alone do not establish whether acquisition timing or header metadata is responsible.
- Expanded verification to 11 tests, including rank-deficient CSP and empty task annotations. All passed in the independent environment.
- Added a locked local encoder to produce a 160-second captioned demo using actual outputs. No synthetic voice or impersonated narration is used.

- All 30 development subjects yielded usable data: 1,341 trials, no unreadable files or event exclusions. The fixed raw-amplitude threshold flagged 1,037 trials (77.3%). Because removing those trials changes subject/class coverage, summaries explicitly count people with both evaluable classes. Added these descriptive coverage fields from saved predictions without refitting any model.
- Development completed with the original fixed models and 100 permutations. Spectral grouped mean subject balanced accuracy: 0.5334; CSP: 0.5326. Selected spectral under the preset rule. Random-split accuracy was not higher in this experiment; retained the result rather than seeking a more favorable split.

- All 120 EDFs completed checksum verification. Held-out evaluation ran once on 450 usable trials from all 10 reserved subjects. Mean subject balanced accuracy was 0.5358, with subject-bootstrap interval 0.4826–0.6023. The primary conclusion remains uncertain above chance. No model or preprocessing changes followed this result.
- The unflagged holdout contains only 70 trials from 2 people; the report explicitly prevents treating its higher score as a controlled cleaning benefit.
- Final verification: 12 tests passed. An independently installed locked environment reproduced all 450 predictions from all 30 held-out raw EDFs and checked the frozen model SHA256. The exact documented uv CLI also succeeded.
- Generated and visually inspected the report figures and demo frames. The final H.264 video is 1280×720, 12 fps, exactly 160 seconds, captioned without audio; a full decode passed. The narration script is a starting point for the participant's own recording and defense.
