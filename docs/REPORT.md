# StressSense: Wearable Stress Detection under Physical Activity Confounding

**Author:** Malya Aboun · UCSC Deep Learning · Final project report

## 1. Problem

Wearable devices are widely used to infer "stress" from physiology, but physical
activity raises the same signals (heart rate, electrodermal activity, skin
temperature) that stress does. A detector that has only ever seen "rest vs.
stress" will happily call a workout "stress." This project asks a sharper
question: **can a deep model distinguish genuine mental stress from physical
exercise**, and *where does it actually fail*?

I treat this as 4-class classification of 30-second windows of Empatica E4
signals into `baseline`, `stress`, `aerobic`, and `anaerobic`, and I analyze the
confusion structure rather than only headline accuracy.

## 2. Dataset

PhysioNet *Wearable Device Dataset from Induced Stress and Structured Exercise
Sessions* (v1.0.1): 36 participants, three session types (STRESS, AEROBIC,
ANAEROBIC). Each session contains Empatica E4 CSVs — EDA (4 Hz), TEMP (4 Hz),
BVP (64 Hz), HR (1 Hz), 3-axis ACC (32 Hz), IBI, and `tags.csv` event marks.

Labels are derived from the session type plus the protocol tags. The tag→stage
mapping is taken from the dataset's own `Wearable_Dataset.ipynb`: `stress`
windows come from the documented mental-stress tasks (Stroop / mental
arithmetic / opinion tasks; protocol V1 for `S..` subjects includes Stroop,
V2 for `f..` subjects does not). `aerobic`/`anaerobic` come from the active
(first-tag → last-tag) block of each exercise session. `baseline` is the rest
period immediately before the first tag of **any** session type — sourcing it
from all three session types makes baseline-vs-stress and baseline-vs-exercise
partly within-session contrasts.

Documented sensor problems are handled explicitly (e.g. `f07` STRESS excluded
because the PPG/TEMP sensors were covered; connection-split sessions read
independently but grouped per subject). After windowing (30 s, 15 s stride,
capped at 40 windows per class per session) the dataset is **3,994 windows**:
baseline 958, stress 641, aerobic 1,200, anaerobic 1,195.

## 3. Model

A compact 1D CNN: input `(4 channels, 120 samples)` → three
`Conv1d→BatchNorm→ReLU→MaxPool` blocks (32/64/128 filters, kernel 7) → global
average pooling → dropout + linear head → 4 logits (~90k parameters). The four
channels are EDA, TEMP, HR, and an **ACC motion-intensity envelope** (per-bin std
of the accelerometer magnitude, which removes gravity and captures wrist
movement). BVP is deliberately omitted as a raw channel because downsampling its
64 Hz waveform to the common 4 Hz grid would destroy it; HR (derived from BVP) is
used instead.

A 1D CNN suits the data because the discriminative cues are local,
translation-invariant shapes — an EDA rise, sustained vs. spiky motion — learnable
from a few thousand windows without the overfitting risk of a larger model.

## 4. Training

Adam (lr 1e-3, weight decay 1e-4), cross-entropy with **inverse-frequency class
weights** (baseline and stress are minority classes). The crucial design choice
is a **subject-level split** (25 train / 5 val / 6 test subjects): all windows
from one person stay in one split, so the test score measures generalization to
new people rather than memorized individuals. Normalization statistics are fit on
the training split only and stored in the checkpoint. The best checkpoint is
selected on validation macro-F1 with early stopping. Training takes ~2 minutes on
CPU.

## 5. Evaluation

Reported on the held-out test subjects (`S02, S03, S09, S14, S15, f17`; 652
windows): accuracy, macro-F1, per-class precision/recall/F1, a confusion matrix,
and a computed error-analysis narrative (`evaluate.py` writes all of these to
`results/`). Macro-F1 is the headline metric because the classes are imbalanced.

## 6. Results

| Metric | Value |
|---|---|
| Accuracy | 0.54 |
| Macro-F1 | 0.53 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| baseline | 0.44 | 0.65 | 0.52 |
| stress | 0.76 | 0.33 | 0.46 |
| aerobic | 0.61 | 0.62 | 0.62 |
| anaerobic | 0.54 | 0.48 | 0.51 |

Key findings from the confusion matrix:

- **Mental stress and physical activity are perfectly separated** — 0% of stress
  windows are misclassified as aerobic/anaerobic and vice versa. The motion
  channel cleanly distinguishes seated tasks from cycling.
- **Confusions are within-category.** Aerobic↔anaerobic (both cycling) are
  mutually confused ~30%. Stress is **under-detected**: 67% of stress windows are
  read as baseline (recall 0.33), but stress precision is high (0.76) — when the
  model commits to "stress" it is usually right.
- Easiest class: aerobic (F1 0.62). Hardest: stress (F1 0.46).

The validation macro-F1 (≈ 0.64) exceeds the test macro-F1 (≈ 0.53); this gap is
expected with only 6 test subjects and is itself evidence of how much
subject-level variance matters.

## 7. Limitations

Controlled lab data with few subjects (a 6-subject test set is high-variance);
"stress" means *performing a scripted stress task*, not a clinical state;
each activity is essentially one recording per subject, so some separability may
reflect per-session offsets rather than pure physiology; wearable signals are
motion-corrupted, especially during exercise. This is a research prototype, **not
a medical or diagnostic tool**.

## 8. Future work

Add a high-rate BVP/HRV branch; use subject-wise cross-validation for stable
estimates; explore per-subject normalization / domain adaptation for true
cross-person robustness; and regress the self-reported stress level instead of a
binary stress label.

## 9. Conclusion

A compact, reproducible pipeline answers the project's core question: under
honest subject-level evaluation, the model cleanly separates mental stress from
physical exercise, while the genuinely hard problems — detecting subtle mental
stress against rest, and telling two kinds of cycling apart — remain open. The
value is in the confusion analysis, not the headline accuracy.
