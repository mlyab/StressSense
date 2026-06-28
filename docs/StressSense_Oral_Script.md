# StressSense — Oral Presentation Script
**Malya Aboun · UCSC Deep Learning · Final Project**

Word-for-word script for a 5-minute presentation (11 slides, ~25–30 seconds each).

---

## Slide 1 — Title / Motivation

Hi everyone, I'm Malya Aboun, and this is my final project for UCSC Deep Learning.

The project is called StressSense. The idea is pretty simple: wearables are supposed to detect stress. They look at things like heart rate and skin conductance — your EDA. But if you think about it, exercise raises those exact same signals. Your heart rate goes up. You sweat. So the question I wanted to answer was: can a deep learning model actually tell mental stress apart from physical exercise?

*(advance to next slide)*

---

## Slide 2 — Problem

So here is the setup. I am using an Empatica E4 wristband dataset, and I frame this as four-class window classification: baseline rest, mental stress, aerobic exercise, and anaerobic exercise.

The real question is not just "can it detect stress" — it is specifically whether the model confuses stress with exercise. Those are the two cases that look the most alike physiologically.

*(advance to next slide)*

---

## Slide 3 — Dataset

The data comes from PhysioNet. 36 participants wore the E4 wristband across three types of sessions: a stress session with Stroop and mental arithmetic tasks, aerobic steady-state cycling, and anaerobic sprint intervals.

After windowing into 30-second chunks, I end up with about 3,994 windows total. Stress is the smallest class at 641 windows. The data was messier than it looks on paper — one subject had their sensors physically covered, two others had sessions split from dropped connections. I handled all of that explicitly in the preprocessing code.

*(advance to next slide)*

---

## Slide 4 — Pipeline

The full pipeline is four Python scripts. `dataset.py` parses the raw E4 CSVs, resamples everything to a common 4 Hz grid, and builds 30-second windows. `train.py` runs the model. `evaluate.py` computes the metrics and writes the confusion matrix. And `demo.py` launches the Gradio interface.

Everything is driven by a single config file and seeded for reproducibility. You can reproduce the whole thing from scratch.

*(advance to next slide)*

---

## Slide 5 — Model

The model is a compact 1D CNN with about 90,000 parameters. The input is a 4-by-120 tensor — that is 4 channels over 30 seconds at 4 Hz.

The four input channels are EDA, skin temperature, heart rate, and an accelerometer motion intensity envelope. That last one is important: I take the standard deviation of the 3-axis accelerometer magnitude per time bin, which removes the constant gravity component and leaves just how much the wrist is actually moving.

I deliberately leave out raw BVP. Downsampling a 64 Hz pulse waveform to 4 Hz just destroys it, so I use the derived heart rate instead.

A 1D CNN is the right choice here because the useful patterns are local shapes in time — an EDA rise, a burst of motion — not long-range dependencies.

*(advance to next slide)*

---

## Slide 6 — Training

Training uses Adam with class-weighted cross-entropy. Baseline and stress are minority classes, so they get higher weight to stop the model from just predicting aerobic all the time.

The most important decision in the whole project is the subject-level split. All windows from a given person stay in one split — 25 for training, 5 for validation, 6 for testing. That means the test score measures how well the model generalizes to new people, not just new windows from the same people it already saw. That distinction really changes what the numbers mean.

The best checkpoint is picked by validation macro-F1. It trains in about two minutes on CPU.

*(advance to next slide)*

---

## Slide 7 — Evaluation

For evaluation, I report accuracy and macro-F1 as the headline numbers, plus per-class precision, recall, and F1 for all four classes.

I use macro-F1 rather than accuracy because the classes are not balanced. Stress has 102 test windows while aerobic and anaerobic each have 200. If I just reported accuracy, a model that mostly ignored stress could still look decent.

And the confusion matrix is the most informative output of all — it shows not just how often the model is wrong, but what it is confusing things with.

*(advance to next slide)*

---

## Slide 8 — Demo

The demo is a Gradio app. You can pick one of the pre-exported example windows, or upload your own CSV with EDA, TEMP, HR, and ACC columns. It shows the predicted class, confidence scores for all four classes, and a plot of the four biosignals behind the decision.

*[PAUSE — run live demo here, or describe what the screen shows]*

If the live demo does not load, I can run the inference script directly on a saved example window from the command line and show the confusion matrix from the results folder.

*(advance to next slide)*

---

## Slide 9 — Results & Confusion Analysis

Here are the actual results on the 6 held-out test subjects, 652 windows total. Accuracy is 0.54, macro-F1 is 0.53.

But the headline number is honestly not the point. Look at the confusion matrix. Stress is never confused with exercise — zero percent of stress windows are predicted as aerobic or anaerobic, and vice versa. The motion intensity channel cleanly separates seated tasks from cycling. That was the central question of this project, and the answer is clean.

The hard cases are different. Aerobic and anaerobic confuse each other about 30 percent of the time — they are both cycling, and the physiology is similar. And 67 percent of stress windows get misread as baseline — subtle mental stress really does look a lot like just sitting quietly.

*(advance to next slide)*

---

## Slide 10 — Limitations & Future Work

Let me be honest about what this project is and what it is not. It is a controlled-lab prototype with a small dataset. Six test subjects gives you high variance — which is exactly why validation macro-F1 was 0.64 and the test number came out at 0.53.

"Stress" here means performing a scripted Stroop task, not anything like a clinical stress state. And each activity is essentially one recording per subject, so some of the signal separation might reflect per-session signal offsets rather than pure physiology.

Future work would add a separate BVP or HRV branch for cardiac waveform information, use subject-wise cross-validation for more stable estimates, and look at per-subject normalization for better cross-person robustness.

*(advance to next slide)*

---

## Slide 11 — Lessons Learned

Five things I learned from this project.

One: subject-level splitting changes the story. Validation was 0.64, test was 0.53. That gap is the point, not a bug.

Two: the confusion matrix is more useful than accuracy for understanding what is actually happening.

Three: deliberate preprocessing choices — the motion intensity envelope, dropping BVP — beat throwing every raw signal at the model.

Four: handling real-world data problems explicitly is most of the actual work.

And five: the value of this project is not the 54 percent number. It is showing exactly where the model works and where it does not. Thanks.

---

---

## 30-Second Speed Version

*Use this if you are running over time or need to give a quick summary.*

"StressSense is a 1D CNN that classifies 30-second windows of Empatica E4 wristband data as baseline, stress, aerobic exercise, or anaerobic exercise. The key result: the model cleanly separates mental stress from physical exercise — zero percent confusion both ways. The hard problem is subtle stress versus rest: 67 percent of stress windows are misread as baseline. Accuracy is 0.54, macro-F1 is 0.53, on 6 held-out test subjects. The value is in the confusion analysis, not the headline number."

---

## Demo Backup Line

> "If the live demo does not load, I can run the inference script on a saved example window and show the confusion matrix from the results folder."

```bash
python inference.py --input examples/example_stress_1.csv
```

---

## Timing Guide

| Slide | Target time | Cumulative |
|---|---|---|
| 01 · Title | 30 s | 0:30 |
| 02 · Problem | 25 s | 0:55 |
| 03 · Dataset | 30 s | 1:25 |
| 04 · Pipeline | 25 s | 1:50 |
| 05 · Model | 35 s | 2:25 |
| 06 · Training | 30 s | 2:55 |
| 07 · Evaluation | 25 s | 3:20 |
| 08 · Demo | 25 s | 3:45 |
| 09 · Results | 35 s | 4:20 |
| 10 · Limitations | 25 s | 4:45 |
| 11 · Lessons | 30 s | 5:15 |

*Total: ~5 minutes 15 seconds. If you need to cut, shorten slides 5 (Model) and 10 (Limitations) first.*
