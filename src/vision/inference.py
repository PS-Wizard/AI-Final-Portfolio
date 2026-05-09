from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input

from src.plots import save_show
from src.vision.data import IMG_SIZE, TRANSFER_IMG_SIZE


def show_misclassified(predictions: pd.DataFrame, path: Path | None = None, title: str = "Misclassified examples", limit: int = 12) -> None:
    subset = predictions.loc[~predictions["correct"]].sort_values("confidence", ascending=False).head(limit)
    if subset.empty:
        print("No misclassified examples.")
        return
    _gallery(subset, title, path)


def show_prediction_gallery(
    predictions: pd.DataFrame,
    path: Path | None = None,
    title: str = "Prediction gallery",
    ascending: bool = False,
    limit: int = 12,
) -> None:
    _gallery(predictions.sort_values("confidence", ascending=ascending).head(limit), title, path)


def predict_unlabeled(
    model: tf.keras.Model,
    unlabeled: pd.DataFrame,
    id_to_label: dict[int, str],
    use_transfer_pipeline: bool,
) -> pd.DataFrame:
    if unlabeled.empty:
        return unlabeled.copy()
    images = np.stack([_preprocess_image(path, use_transfer_pipeline) for path in unlabeled["path"]])
    probabilities = model.predict(images, verbose=0)
    out = unlabeled.copy()
    out["predicted_label_id"] = probabilities.argmax(axis=1)
    out["predicted_class"] = out["predicted_label_id"].map(id_to_label)
    out["confidence"] = probabilities.max(axis=1)
    return out


def show_unlabeled_predictions(predictions: pd.DataFrame, path: Path | None = None, title: str = "Unlabeled test predictions") -> None:
    if predictions.empty:
        print("No unlabeled images found.")
        return
    rows = int(np.ceil(len(predictions) / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows))
    axes = np.array(axes).reshape(rows, 5)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), predictions.iterrows()):
        ax.imshow(Image.open(row["path"]).convert("RGB"))
        ax.set_title(f"{row['predicted_class']}\nconf={row['confidence']:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save_show(fig, path)


def _gallery(rows: pd.DataFrame, title: str, path: Path | None) -> None:
    ncols = 4
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        ax.imshow(Image.open(row["path"]).convert("RGB"))
        ax.set_title(
            f"true={row['display_class']}\npred={row['predicted_class']}\nconf={row['confidence']:.3f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save_show(fig, path)


def _preprocess_image(path: str, transfer: bool) -> np.ndarray:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    if transfer:
        image = tf.image.resize(image, TRANSFER_IMG_SIZE)
        return mobilenet_preprocess_input(image).numpy()
    image = tf.image.resize(image, IMG_SIZE)
    return (tf.cast(image, tf.float32) / 255.0).numpy()
