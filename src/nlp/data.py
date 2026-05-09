from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

TARGET_NAMES = ["hate_speech", "offensive_language", "neither"]
LABEL_NORMALIZATION = {
    "hate speech": "hate_speech",
    "hate speec": "hate_speech",
    "offensive language": "offensive_language",
    "neither": "neither",
}
LABEL_TO_ID = {label: idx for idx, label in enumerate(TARGET_NAMES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def load_dataset(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "hatevsoffensive_language.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"label": "label_raw", "text": "text"})
    df["label_clean"] = df["label_raw"].map(LABEL_NORMALIZATION)
    if df["label_clean"].isna().any():
        missing = sorted(df.loc[df["label_clean"].isna(), "label_raw"].unique())
        raise ValueError(f"unknown labels: {missing}")
    df["label_id"] = df["label_clean"].map(LABEL_TO_ID).astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def split_dataset(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label_id"],
        random_state=seed,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def class_weights(labels: pd.Series) -> dict[int, float]:
    classes = np.array(sorted(labels.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.to_numpy())
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return df["label_clean"].value_counts().rename_axis("label").reset_index(name="count")
