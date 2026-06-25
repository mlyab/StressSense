"""Turn the raw Empatica E4 sessions into a cache of labeled signal windows.

The PhysioNet dataset is organized by activity (STRESS / AEROBIC / ANAEROBIC),
with one subfolder per session holding raw E4 CSVs (EDA, TEMP, HR, BVP, ACC,
IBI, tags). This module parses those sessions and produces fixed-length,
multi-channel windows, each with one of four labels:

    baseline, stress, aerobic, anaerobic

Labeling logic (full rationale in the README):
  * baseline : the rest period just before the first event tag of a session.
  * stress   : the documented mental-stress tasks inside a STRESS session. The
               tag -> task mapping is read straight from the dataset's own
               Wearable_Dataset.ipynb (`graph_multiple`), which shades the
               Stroop/TMCT/opinion/subtraction tasks.
  * aerobic  : the active block of an AEROBIC session (first tag -> last tag).
  * anaerobic: the active block of an ANAEROBIC session (first tag -> last tag).

Each E4 CSV stores the session start time (UTC) in row 0 and the sampling rate
in row 1; the signals run at different rates (EDA/TEMP 4 Hz, HR 1 Hz, ACC 32 Hz),
so every channel is resampled onto a common grid before windowing.

Run `python dataset.py --config config.yaml` to build the cache.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

ACTIVITIES = ["STRESS", "AEROBIC", "ANAEROBIC"]
_DT_FMT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- #
# Config + reproducibility
# --------------------------------------------------------------------------- #
def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml and resolve the important paths relative to it."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = os.path.dirname(path)

    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    cfg["data"]["dataset_root"] = _abs(cfg["data"]["dataset_root"])
    cfg["data"]["cache_path"] = _abs(cfg["data"]["cache_path"])
    cfg["data"]["examples_dir"] = _abs(cfg["data"]["examples_dir"])
    cfg["paths"]["models_dir"] = _abs(cfg["paths"]["models_dir"])
    cfg["paths"]["results_dir"] = _abs(cfg["paths"]["results_dir"])
    cfg["_base_dir"] = base
    return cfg


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Raw E4 file readers
# --------------------------------------------------------------------------- #
def _parse_dt(s: str) -> datetime:
    return datetime.strptime(str(s).strip(), _DT_FMT)


def read_e4_file(path: str) -> Tuple[datetime, float, np.ndarray]:
    """Read an Empatica CSV. Returns (start_time, sample_rate_hz, data).

    `data` is shape (n,) for single-channel files and (n, 3) for ACC.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing E4 file: {path}")
    df = pd.read_csv(path, header=None)
    start = _parse_dt(df.iloc[0, 0])
    fs = float(df.iloc[1, 0])
    data = df.iloc[2:].to_numpy(dtype=float)
    if data.shape[1] == 1:
        data = data[:, 0]
    return start, fs, data


def read_tags(path: str, t0: datetime) -> List[float]:
    """Return event-tag times in seconds relative to the session start `t0`."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    times: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                times.append((_parse_dt(line) - t0).total_seconds())
    return times


# --------------------------------------------------------------------------- #
# Resampling onto a common grid
# --------------------------------------------------------------------------- #
def _resample_slow(values: np.ndarray, offset: float, fs: float,
                   grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of a slow signal (EDA/TEMP/HR) onto `grid`.

    Samples outside the signal's own time span are edge-filled; the caller tracks
    those regions separately via the validity mask.
    """
    t = offset + np.arange(len(values)) / fs
    return np.interp(grid, t, values, left=values[0], right=values[-1])


def _acc_motion_intensity(acc: np.ndarray, offset: float, fs: float,
                          n_grid: int, target_fs: float) -> np.ndarray:
    """Convert 3-axis ACC into a motion-intensity envelope on the grid.

    ACC is stored in units of 1/64 g. We take the vector magnitude and, for each
    grid bin, the standard deviation of the magnitude samples that fall in it.
    The std removes the (near-constant) gravity component, so what remains is how
    much the wrist is moving in that 1/target_fs-second bin -- exactly the cue we
    expect to separate physical exercise from seated stress.
    """
    mag = np.sqrt((acc ** 2).sum(axis=1)) / 64.0  # -> g
    t = offset + np.arange(len(mag)) / fs
    bins = np.floor(t * target_fs).astype(int)
    keep = (bins >= 0) & (bins < n_grid)
    bins, mag = bins[keep], mag[keep]

    count = np.bincount(bins, minlength=n_grid).astype(float)
    s1 = np.bincount(bins, weights=mag, minlength=n_grid)
    s2 = np.bincount(bins, weights=mag * mag, minlength=n_grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, s1 / count, 0.0)
        var = np.where(count > 0, s2 / count - mean ** 2, 0.0)
    return np.sqrt(np.clip(var, 0.0, None))


def session_channels(session_dir: str, channels: List[str],
                     target_fs: float) -> Tuple[np.ndarray, np.ndarray, datetime]:
    """Build the resampled channel matrix for one session.

    Returns (X, valid_mask, t0) where X is (C, T) on a uniform `target_fs` grid
    that spans the EDA recording, and valid_mask is (T,) True where every channel
    is backed by real samples (not edge-filled).
    """
    file_map = {"EDA": "EDA.csv", "TEMP": "TEMP.csv", "HR": "HR.csv",
                "BVP": "BVP.csv", "ACC": "ACC.csv"}

    # EDA defines the session time reference (t0) and grid length.
    eda_start, eda_fs, eda_vals = read_e4_file(os.path.join(session_dir, "EDA.csv"))
    t0 = eda_start
    duration = len(eda_vals) / eda_fs
    n_grid = int(duration * target_fs)
    grid = np.arange(n_grid) / target_fs

    rows = np.empty((len(channels), n_grid), dtype=np.float32)
    valid = np.ones(n_grid, dtype=bool)

    for ci, ch in enumerate(channels):
        if ch not in file_map:
            raise ValueError(f"Unknown channel '{ch}' (known: {list(file_map)})")
        start, fs, data = read_e4_file(os.path.join(session_dir, file_map[ch]))
        offset = (start - t0).total_seconds()
        if ch == "ACC":
            rows[ci] = _acc_motion_intensity(data, offset, fs, n_grid, target_fs)
            ch_end = offset + len(data) / fs
        else:
            rows[ci] = _resample_slow(data, offset, fs, grid)
            ch_end = offset + len(data) / fs
        valid &= (grid >= offset) & (grid <= ch_end)

    return rows, valid, t0


# --------------------------------------------------------------------------- #
# Tag -> protocol-stage mapping
# --------------------------------------------------------------------------- #
def stress_task_spans(subject: str, tags: List[float]) -> List[Tuple[str, float, float]]:
    """Mental-stress task windows for a STRESS session.

    Index mapping is taken from the dataset's Wearable_Dataset.ipynb. V1 sessions
    ('S..') run Stroop/TMCT/opinion/subtraction; V2 sessions ('f..') drop Stroop.
    """
    if subject.startswith("S"):  # protocol V1
        plan = [("Stroop", 2, 3), ("TMCT", 4, 5), ("RealOpinion", 6, 7),
                ("OppositeOpinion", 8, 9), ("Subtract", 10, 11)]
        need = 12
    else:                        # protocol V2
        plan = [("TMCT", 1, 2), ("RealOpinion", 3, 4),
                ("OppositeOpinion", 5, 6), ("Subtract", 7, 8)]
        need = 9
    if len(tags) < need:
        return []
    return [(name, tags[i], tags[j]) for name, i, j in plan]


def session_regions(activity: str, subject: str, tags: List[float],
                    duration: float, cfg: dict) -> List[Tuple[str, float, float]]:
    """Return labeled (label, start_s, end_s) regions for one session."""
    seg = cfg["segmentation"]
    baseline_sec = cfg["windows"]["baseline_sec"]
    regions: List[Tuple[str, float, float]] = []

    # Baseline: rest just before the first tag of the session.
    if activity in seg["baseline_from"]:
        if tags:
            first = tags[0]
            regions.append(("baseline", max(0.0, first - baseline_sec), first))
        elif activity == "STRESS":
            # A STRESS folder with no tags is a baseline-only recording (f14_a).
            regions.append(("baseline", 0.0, duration))

    # Active part of the session.
    if activity == "STRESS":
        for name, a, b in stress_task_spans(subject, tags):
            if b > a:
                regions.append(("stress", a, b))
    elif activity in ("AEROBIC", "ANAEROBIC") and len(tags) >= 2:
        label = "aerobic" if activity == "AEROBIC" else "anaerobic"
        regions.append((label, tags[0], tags[-1]))

    return regions


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def base_subject(folder: str) -> str:
    """Strip a `_a`/`_b` continuation suffix so split files group as one subject."""
    return re.sub(r"_[ab]$", "", folder)


def make_windows(X: np.ndarray, valid: np.ndarray, regions: List[Tuple[str, float, float]],
                 cfg: dict, rng: np.random.Generator) -> List[Tuple[np.ndarray, str]]:
    """Slide windows over each labeled region, capping windows per class."""
    target_fs = cfg["signals"]["target_fs"]
    win = int(cfg["windows"]["window_sec"] * target_fs)
    stride = int(cfg["windows"]["stride_sec"] * target_fs)
    min_cov = cfg["windows"]["min_coverage"]
    cap = cfg["windows"]["max_windows_per_session_class"]

    by_label: Dict[str, List[np.ndarray]] = {}
    for label, start_s, end_s in regions:
        a = int(round(start_s * target_fs))
        b = int(round(end_s * target_fs))
        for s in range(a, b - win + 1, stride):
            if valid[s:s + win].mean() >= min_cov:
                by_label.setdefault(label, []).append(X[:, s:s + win].copy())

    out: List[Tuple[np.ndarray, str]] = []
    for label, wins in by_label.items():
        if len(wins) > cap:
            idx = rng.choice(len(wins), size=cap, replace=False)
            wins = [wins[i] for i in idx]
        out.extend((w, label) for w in wins)
    return out


# --------------------------------------------------------------------------- #
# Build + cache
# --------------------------------------------------------------------------- #
def build_dataset(cfg: dict, verbose: bool = True) -> dict:
    """Scan the whole dataset and return arrays of windows, labels and metadata."""
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    root = os.path.join(cfg["data"]["dataset_root"], "Wearable_Dataset")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Wearable_Dataset folder not found at: {root}\n"
            f"Check `data.dataset_root` in config.yaml and that the dataset is unzipped."
        )

    channels = cfg["signals"]["channels"]
    classes = cfg["labels"]["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    target_fs = cfg["signals"]["target_fs"]
    excluded = {(a, s) for a, s in cfg["segmentation"]["exclude"]}

    X_list, y_list, subj_list, act_list = [], [], [], []
    skipped: List[str] = []

    for activity in ACTIVITIES:
        adir = os.path.join(root, activity)
        if not os.path.isdir(adir):
            continue
        for folder in sorted(os.listdir(adir)):
            subj = base_subject(folder)
            if (activity, subj) in excluded:
                skipped.append(f"{activity}/{folder} (excluded by config)")
                continue
            session_dir = os.path.join(adir, folder)
            try:
                X, valid, t0 = session_channels(session_dir, channels, target_fs)
                tags = read_tags(os.path.join(session_dir, "tags.csv"), t0)
            except Exception as exc:  # keep building even if one session is broken
                skipped.append(f"{activity}/{folder} ({exc})")
                continue

            duration = X.shape[1] / target_fs
            regions = session_regions(activity, subj, tags, duration, cfg)
            windows = make_windows(X, valid, regions, cfg, rng)
            for w, label in windows:
                X_list.append(w)
                y_list.append(class_to_idx[label])
                subj_list.append(subj)
                act_list.append(activity)
            if verbose:
                counts = Counter(lbl for _, lbl in windows)
                print(f"  {activity}/{folder:8s} tags={len(tags):2d} -> "
                      f"{dict(counts)}")

    if not X_list:
        raise RuntimeError("No windows were produced. Check the dataset path and config.")

    data = {
        "X": np.stack(X_list).astype(np.float32),       # (N, C, T)
        "y": np.array(y_list, dtype=np.int64),
        "subject": np.array(subj_list),
        "activity": np.array(act_list),
        "channels": np.array(channels),
        "classes": np.array(classes),
        "target_fs": np.int64(target_fs),
        "window_sec": np.int64(cfg["windows"]["window_sec"]),
    }
    if verbose:
        print("\nClass counts:",
              {classes[i]: int((data["y"] == i).sum()) for i in range(len(classes))})
        print("Subjects:", len(set(subj_list)))
        if skipped:
            print("Skipped:", *skipped, sep="\n  ")
    return data


def save_cache(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)


def load_cache(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Window cache not found: {path}\nBuild it first with `python dataset.py`."
        )
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


# --------------------------------------------------------------------------- #
# Subject-level split + normalization
# --------------------------------------------------------------------------- #
def subject_split(subjects: np.ndarray, cfg: dict) -> Dict[str, str]:
    """Assign each subject to train/val/test. Windows never leak across splits."""
    uniq = sorted(set(subjects.tolist()))
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(cfg["split"]["train"] * n))
    n_val = int(round(cfg["split"]["val"] * n))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    assign = {}
    for i, s in enumerate(uniq):
        assign[s] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assign


def split_indices(data: dict, cfg: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Map the cache rows into train/val/test index arrays plus the subject map."""
    assign = subject_split(data["subject"], cfg)
    split_of = np.array([assign[s] for s in data["subject"]])
    idx = {name: np.where(split_of == name)[0] for name in ("train", "val", "test")}
    return idx, assign


def compute_channel_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over (windows, time). Fit on the training split only."""
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, :, None]) / std[None, :, None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the StressSense window cache.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Building windows from: {cfg['data']['dataset_root']}")
    data = build_dataset(cfg)
    save_cache(data, cfg["data"]["cache_path"])
    print(f"\nSaved {data['X'].shape[0]} windows "
          f"(shape {data['X'].shape}) -> {cfg['data']['cache_path']}")


if __name__ == "__main__":
    main()
