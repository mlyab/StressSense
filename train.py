"""Train the StressSense 1D CNN on the cached windows.

Pipeline: load cache -> subject-level split -> fit normalization on train only ->
train with an (optionally) class-weighted loss -> keep the checkpoint with the
best validation macro-F1. The checkpoint is self-contained (weights + channels +
classes + normalization stats + the subject split), so evaluate.py, inference.py
and demo.py need nothing else.

Usage:
    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from dataset import (compute_channel_stats, load_cache, load_config, normalize,
                     set_seed, split_indices)
from model import build_model


def _loaders(data: dict, idx: Dict[str, np.ndarray], mean, std, batch_size: int
             ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    loaders = {}
    for name in ("train", "val", "test"):
        X = normalize(data["X"][idx[name]], mean, std)
        y = data["y"][idx[name]]
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))
    return loaders["train"], loaders["val"], loaders["test"]


def _class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> Tuple[float, float, float]:
    model.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        n += len(yb)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = float((preds == trues).mean())
    macro_f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
    return total_loss / n, acc, macro_f1


def _plot_curves(history: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax[0].legend()
    ax[1].plot(epochs, history["val_acc"], label="val accuracy")
    ax[1].plot(epochs, history["val_f1"], label="val macro-F1")
    ax[1].set(title="Validation metrics", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_cache(cfg["data"]["cache_path"])
    classes = [str(c) for c in data["classes"]]
    channels = [str(c) for c in data["channels"]]
    idx, assign = split_indices(data, cfg)

    n_by_split = {k: len(v) for k, v in idx.items()}
    print(f"Device: {device}")
    print(f"Windows: {data['X'].shape}, channels={channels}, classes={classes}")
    print(f"Split (windows): {n_by_split}")
    subj_by_split = {s: sum(1 for k, v in assign.items() if v == s)
                     for s in ("train", "val", "test")}
    print(f"Split (subjects): {subj_by_split}")
    if min(n_by_split.values()) == 0:
        raise RuntimeError("A split is empty; with very few subjects, adjust split ratios.")

    mean, std = compute_channel_stats(data["X"][idx["train"]])
    train_loader, val_loader, test_loader = _loaders(
        data, idx, mean, std, cfg["train"]["batch_size"])

    model = build_model(cfg, in_channels=len(channels), num_classes=len(classes)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    weight = (_class_weights(data["y"][idx["train"]], len(classes)).to(device)
              if cfg["train"]["class_weighting"] else None)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])

    history = {k: [] for k in ("train_loss", "val_loss", "val_acc", "val_f1")}
    best_f1, best_state, best_epoch = -1.0, None, 0
    patience = cfg["train"]["early_stopping_patience"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
            n += len(yb)
        train_loss = running / n
        val_loss, val_acc, val_f1 = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_f1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (best val_f1 {best_f1:.3f} @ {best_epoch}).")
            break

    # Restore and report the best model on the held-out test set.
    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1 = _evaluate(model, test_loader, criterion, device)
    print(f"\nBest epoch {best_epoch}: val_f1={best_f1:.3f} | "
          f"test_acc={test_acc:.3f} test_f1={test_f1:.3f}")

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["paths"]["models_dir"], "best_model.pt")
    torch.save({
        "model_state": best_state,
        "channels": channels,
        "classes": classes,
        "norm_mean": mean,
        "norm_std": std,
        "model_cfg": cfg["model"],
        "target_fs": int(data["target_fs"]),
        "window_sec": int(data["window_sec"]),
        "split_assignment": assign,
    }, ckpt_path)

    _plot_curves(history, os.path.join(cfg["paths"]["results_dir"], "training_curves.png"))
    with open(os.path.join(cfg["paths"]["results_dir"], "training_history.json"),
              "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_epoch": best_epoch,
                   "val_f1": best_f1, "test_acc": test_acc, "test_f1": test_f1}, f, indent=2)

    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Saved curves     -> {os.path.join(cfg['paths']['results_dir'], 'training_curves.png')}")
    return {"best_epoch": best_epoch, "val_f1": best_f1,
            "test_acc": test_acc, "test_f1": test_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the StressSense CNN.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
