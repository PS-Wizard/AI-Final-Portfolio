from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.plots import save_show


def show_label_distribution(df: pd.DataFrame, path: Path | None = None) -> None:
    counts = df["label_clean"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title("Class distribution")
    ax.set_xlabel("class")
    ax.set_ylabel("tweets")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_show(fig, path)


def show_text_length_distribution(df: pd.DataFrame, path: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["text"].str.split().map(len).hist(bins=40, ax=axes[0], color="#ff7f0e")
    axes[0].set_title("Raw word count")
    axes[0].set_xlabel("words")
    df["clean_word_count"].hist(bins=40, ax=axes[1], color="#2ca02c")
    axes[1].set_title("Clean word count")
    axes[1].set_xlabel("words")
    fig.tight_layout()
    save_show(fig, path)


def top_words(df: pd.DataFrame, label: str | None = None, n: int = 25) -> pd.DataFrame:
    source = df if label is None else df.loc[df["label_clean"] == label]
    counter = Counter()
    for text in source["clean_text"]:
        counter.update(text.split())
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def show_top_words(words: pd.DataFrame, title: str, path: Path | None = None) -> None:
    ordered = words.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, len(ordered) * 0.24)))
    ax.barh(ordered["word"], ordered["count"], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("count")
    fig.tight_layout()
    save_show(fig, path)
