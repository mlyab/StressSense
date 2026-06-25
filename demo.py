"""Interactive Gradio demo for StressSense.

Pick one of the exported example windows (or upload your own CSV) and the trained
CNN predicts baseline / stress / aerobic / anaerobic, shows the class confidences,
and plots the four biosignals that drove the decision.

Usage:
    python demo.py                 # launches a local web app
    python demo.py --share         # public Gradio link
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dataset import load_config
from inference import Predictor, read_window_csv

CHANNEL_LABELS = {
    "EDA": "EDA (uS)",
    "TEMP": "Skin temp (C)",
    "HR": "Heart rate (bpm)",
    "ACC": "Motion intensity (g)",
}


def _plot_window(window: np.ndarray, channels, target_fs: int, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(window.shape[1]) / target_fs
    fig, axes = plt.subplots(len(channels), 1, figsize=(8, 6), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch, row in zip(axes, channels, window):
        ax.plot(t, row, color="#3b6ea5")
        ax.set_ylabel(CHANNEL_LABELS.get(ch, ch), fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def build_demo(cfg: dict):
    import gradio as gr

    ckpt_path = os.path.join(cfg["paths"]["models_dir"], "best_model.pt")
    predictor = Predictor(ckpt_path)
    examples_dir = cfg["data"]["examples_dir"]

    manifest_path = os.path.join(examples_dir, "manifest.csv")
    examples: Dict[str, Tuple[str, str]] = {}
    if os.path.isfile(manifest_path):
        for _, row in pd.read_csv(manifest_path).iterrows():
            label = f"{row['true_label']} — subject {row['subject']} ({row['file']})"
            examples[label] = (os.path.join(examples_dir, row["file"]), row["true_label"])

    def _run(window: np.ndarray, title: str):
        pred, scores = predictor.predict(window)
        fig = _plot_window(window, predictor.channels, predictor.target_fs, title)
        return scores, fig, f"### Predicted: **{pred}**"

    def run_example(choice: Optional[str]):
        if not choice or choice not in examples:
            return None, None, "Select an example window."
        path, true_label = examples[choice]
        window = read_window_csv(path, predictor.channels, predictor.window_len)
        scores, fig, md = _run(window, f"Example — true label: {true_label}")
        return scores, fig, md + f"  \nTrue label: *{true_label}*"

    def run_upload(file_obj):
        if file_obj is None:
            return None, None, "Upload a CSV with columns: " + ", ".join(predictor.channels)
        try:
            window = read_window_csv(file_obj.name, predictor.channels, predictor.window_len)
        except Exception as exc:
            return None, None, f"Could not read window: {exc}"
        return _run(window, "Uploaded window")

    with gr.Blocks(title="StressSense") as demo:
        gr.Markdown(
            "# StressSense\n"
            "Wearable stress detection under physical-activity confounding. "
            "The model classifies a 30-second window of Empatica E4 signals "
            "(EDA, skin temperature, heart rate, wrist motion) as **baseline**, "
            "**stress**, **aerobic** or **anaerobic**."
        )
        with gr.Tabs():
            with gr.Tab("Example windows"):
                ex_dropdown = gr.Dropdown(sorted(examples), label="Example window")
                ex_btn = gr.Button("Classify example", variant="primary")
            with gr.Tab("Upload CSV"):
                up_file = gr.File(label="CSV with one column per channel "
                                        f"({', '.join(predictor.channels)})",
                                  file_types=[".csv"])
                up_btn = gr.Button("Classify upload", variant="primary")

        verdict = gr.Markdown()
        with gr.Row():
            scores_out = gr.Label(label="Class confidence", num_top_classes=4)
            plot_out = gr.Plot(label="Signals")

        ex_btn.click(run_example, ex_dropdown, [scores_out, plot_out, verdict])
        up_btn.click(run_upload, up_file, [scores_out, plot_out, verdict])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the StressSense demo.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--share", action="store_true", help="Create a public link.")
    args = parser.parse_args()
    demo = build_demo(load_config(args.config))
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
