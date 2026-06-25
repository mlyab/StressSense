"""Run a trained StressSense model on a single signal window.

The checkpoint written by train.py is self-contained, so this module needs only
the .pt file. It exposes a small `Predictor` used by both evaluate.py and the
Gradio demo, and a CLI for classifying a window stored as a CSV.

CSV format expected by `--input`: one column per channel (header names matching
the model's channels, e.g. EDA,TEMP,HR,ACC) and one row per time sample. The
window is linearly resampled to the model's expected length if needed.

Usage:
    python inference.py --input examples/example_stress.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from model import StressCNN


def load_checkpoint(path: str, device: torch.device) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nTrain a model first with `python train.py`."
        )
    return torch.load(path, map_location=device, weights_only=False)


def resample_to_len(window: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a (C, T) window along time to `target_len` samples."""
    c, t = window.shape
    if t == target_len:
        return window.astype(np.float32)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(dst, src, window[i]) for i in range(c)]).astype(np.float32)


def read_window_csv(path: str, channels: List[str], target_len: int) -> np.ndarray:
    """Load a CSV of per-channel columns into a (C, target_len) window."""
    df = pd.read_csv(path)
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {path} is missing channel column(s): {missing}. "
            f"Expected columns: {channels}"
        )
    window = df[channels].to_numpy(dtype=float).T  # (C, T)
    return resample_to_len(window, target_len)


class Predictor:
    """Wraps a trained model plus its channels, classes and normalization stats."""

    def __init__(self, ckpt_path: str, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.channels: List[str] = list(ckpt["channels"])
        self.classes: List[str] = list(ckpt["classes"])
        self.mean = np.asarray(ckpt["norm_mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["norm_std"], dtype=np.float32)
        self.target_fs = int(ckpt["target_fs"])
        self.window_sec = int(ckpt["window_sec"])
        self.window_len = self.target_fs * self.window_sec

        m = ckpt["model_cfg"]
        self.model = StressCNN(len(self.channels), len(self.classes),
                               conv_channels=list(m["conv_channels"]),
                               kernel_size=m["kernel_size"], dropout=m["dropout"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean[None, :, None]) / self.std[None, :, None]

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X is (C, T) or (N, C, T) of raw (un-normalized) windows -> softmax probs."""
        single = X.ndim == 2
        if single:
            X = X[None]
        Xn = self._normalize(X.astype(np.float32))
        logits = self.model(torch.from_numpy(Xn).to(self.device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs[0] if single else probs

    def predict(self, window: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Classify one (C, T) window -> (predicted_label, {class: probability})."""
        probs = self.predict_proba(window)
        scores = {c: float(p) for c, p in zip(self.classes, probs)}
        return self.classes[int(np.argmax(probs))], scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one signal window.")
    parser.add_argument("--input", required=True, help="CSV with one column per channel.")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint)
    window = read_window_csv(args.input, predictor.channels, predictor.window_len)
    label, scores = predictor.predict(window)
    print(f"Predicted: {label}")
    for cls, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:10s} {p:.3f}")


if __name__ == "__main__":
    main()
