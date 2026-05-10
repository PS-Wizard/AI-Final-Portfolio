from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_show(fig, path: Path | None = None, dpi: int = 200) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.show()


'''
history.history -> accuray, val_accuracy, loss, val_loss
'''
def plot_history(history, title: str, path: Path | None = None) -> None:
    values = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(values["accuracy"], label="train")
    axes[0].plot(values["val_accuracy"], label="validation")
    axes[0].set_title(f"{title} — accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()

    axes[1].plot(values["loss"], label="train")
    axes[1].plot(values["val_loss"], label="validation")
    axes[1].set_title(f"{title} — loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    fig.tight_layout()
    save_show(fig, path)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(title)
    plt.xticks(rotation=30)
    fig.tight_layout()
    save_show(fig, path)


def plot_metric_comparison(results: pd.DataFrame, path: Path | None = None) -> None:
    ordered = results.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(ordered))))
    ax.barh(ordered["experiment_name"], ordered["macro_f1"], color="#1f77b4")
    ax.set_xlabel("macro F1")
    ax.set_title("Vision experiment comparison")
    ax.set_xlim(max(0, ordered["macro_f1"].min() - 0.05), 1.0)
    fig.tight_layout()
    save_show(fig, path)


def plot_training_time_vs_accuracy(results: pd.DataFrame, path: Path | None = None) -> None:
    size = results["trainable_parameters"].clip(lower=1) / results["trainable_parameters"].max() * 900
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results["training_time_seconds"], results["accuracy"], s=size, alpha=0.65)
    for _, row in results.iterrows():
        ax.annotate(row["experiment_name"], (row["training_time_seconds"], row["accuracy"]), fontsize=8)
    ax.set_xlabel("training time (seconds)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs training time")
    fig.tight_layout()
    save_show(fig, path)
