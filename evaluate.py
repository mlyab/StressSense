"""Evaluate a trained StressSense model on the held-out test subjects.

Reuses the exact subject split stored in the checkpoint, so the test set is the
same one train.py never touched. Produces:
  * results/metrics.json            -- accuracy, macro-F1, per-class P/R/F1
  * results/classification_report.txt
  * results/confusion_matrix.png    -- counts + row-normalized
  * results/error_analysis.txt      -- computed narrative (stress vs. exercise)
and exports a few example windows per class to examples/ for the demo.

Usage:
    python evaluate.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from dataset import load_cache, load_config
from inference import Predictor, load_checkpoint
import torch


def _confusion_plot(cm: np.ndarray, classes: List[str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, title, fmt in (
        (axes[0], cm, "Confusion matrix (counts)", "d"),
        (axes[1], cm_norm, "Confusion matrix (row-normalized)", ".2f"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes,
               xlabel="predicted", ylabel="true", title=title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        thresh = mat.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _error_analysis(cm: np.ndarray, classes: List[str], report: dict) -> str:
    idx = {c: i for i, c in enumerate(classes)}
    row_sum = cm.sum(axis=1)

    def frac(true_cls: str, pred_set: List[str]) -> float:
        i = idx[true_cls]
        if row_sum[i] == 0:
            return float("nan")
        return sum(cm[i, idx[p]] for p in pred_set if p in idx) / row_sum[i]

    physical = [c for c in ("aerobic", "anaerobic") if c in idx]
    lines: List[str] = ["StressSense - error analysis (test set)", "=" * 42, ""]

    per_f1 = {c: report[c]["f1-score"] for c in classes if c in report}
    best = max(per_f1, key=per_f1.get)
    worst = min(per_f1, key=per_f1.get)
    lines.append(f"Per-class F1: " + ", ".join(f"{c}={per_f1[c]:.2f}" for c in classes))
    lines.append(f"Easiest class: {best} (F1={per_f1[best]:.2f}); "
                 f"hardest class: {worst} (F1={per_f1[worst]:.2f}).")
    lines.append("")

    if "stress" in idx and physical:
        s2p = frac("stress", physical)
        lines.append("Does the model separate mental stress from physical activity?")
        lines.append(f"  - {s2p*100:.1f}% of true STRESS windows were misread as "
                     f"physical activity ({'/'.join(physical)}).")
        for p in physical:
            p2s = frac(p, ["stress"])
            lines.append(f"  - {p2s*100:.1f}% of true {p.upper()} windows were misread as STRESS.")
        verdict = ("largely separates" if s2p < 0.15 else
                   "mostly separates" if s2p < 0.35 else "struggles to separate")
        lines.append(f"  => The model {verdict} stress from physical activity.")
        lines.append("")

    if "aerobic" in idx and "anaerobic" in idx:
        a2an = frac("aerobic", ["anaerobic"])
        an2a = frac("anaerobic", ["aerobic"])
        lines.append("Aerobic vs. anaerobic (both are cycling exercise):")
        lines.append(f"  - aerobic->anaerobic: {a2an*100:.1f}% | "
                     f"anaerobic->aerobic: {an2a*100:.1f}%")
        lines.append("")

    if "baseline" in idx and "stress" in idx:
        b2s = frac("baseline", ["stress"])
        s2b = frac("stress", ["baseline"])
        lines.append("Baseline vs. stress (within the same stress sessions):")
        lines.append(f"  - baseline->stress: {b2s*100:.1f}% | stress->baseline: {s2b*100:.1f}%")
        lines.append("")

    lines.append("Note: results come from a small, controlled lab dataset with few "
                 "subjects and\none session per activity, so absolute numbers should be "
                 "read with caution.")
    return "\n".join(lines)


def _export_examples(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                     channels: List[str], classes: List[str],
                     examples_dir: str, per_class: int = 3, seed: int = 0) -> None:
    os.makedirs(examples_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []
    for ci, cls in enumerate(classes):
        rows = np.where(y == ci)[0]
        if len(rows) == 0:
            continue
        pick = rng.choice(rows, size=min(per_class, len(rows)), replace=False)
        for k, r in enumerate(pick):
            fname = f"example_{cls}_{k+1}.csv"
            pd.DataFrame(X[r].T, columns=channels).to_csv(
                os.path.join(examples_dir, fname), index=False)
            manifest.append({"file": fname, "true_label": cls,
                             "subject": str(subjects[r])})
    pd.DataFrame(manifest).to_csv(os.path.join(examples_dir, "manifest.csv"), index=False)


def evaluate(cfg: dict) -> dict:
    cache = load_cache(cfg["data"]["cache_path"])
    classes = [str(c) for c in cache["classes"]]
    channels = [str(c) for c in cache["channels"]]
    ckpt_path = os.path.join(cfg["paths"]["models_dir"], "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assign = load_checkpoint(ckpt_path, device)["split_assignment"]
    predictor = Predictor(ckpt_path)

    test_idx = np.array([i for i, s in enumerate(cache["subject"])
                         if assign.get(str(s)) == "test"])
    if len(test_idx) == 0:
        raise RuntimeError("No test windows found; checkpoint split and cache disagree.")

    X_test = cache["X"][test_idx]
    y_test = cache["y"][test_idx]
    probs = predictor.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    acc = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    report = classification_report(y_test, preds, labels=range(len(classes)),
                                   target_names=classes, output_dict=True, zero_division=0)
    report_txt = classification_report(y_test, preds, labels=range(len(classes)),
                                       target_names=classes, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=range(len(classes)))

    test_subjects = sorted({str(s) for s in cache["subject"][test_idx]})
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    print(f"Test windows: {len(test_idx)}")
    print(f"\nAccuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}\n")
    print(report_txt)

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1,
                   "per_class": {c: report[c] for c in classes},
                   "test_subjects": test_subjects,
                   "n_test_windows": int(len(test_idx))}, f, indent=2)
    with open(os.path.join(results_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report_txt}\n")

    _confusion_plot(cm, classes, os.path.join(results_dir, "confusion_matrix.png"))

    narrative = _error_analysis(cm, classes, report)
    with open(os.path.join(results_dir, "error_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(narrative + "\n")
    print("\n" + narrative)

    _export_examples(X_test, y_test, cache["subject"][test_idx], channels, classes,
                     cfg["data"]["examples_dir"], seed=cfg["seed"])

    print(f"\nSaved metrics, report, confusion matrix and error analysis -> {results_dir}")
    print(f"Exported demo example windows -> {cfg['data']['examples_dir']}")
    return {"accuracy": acc, "macro_f1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the StressSense CNN.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
