from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

from src.plots import save_show
from src.vision.data import IMG_SIZE


def dataset_summary(inventory: pd.DataFrame, corrupted: pd.DataFrame, unlabeled: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "metric": ["total_labeled_images", "total_classes", "corrupted_images", "unlabeled_test_images"],
        "value": [len(inventory), inventory["display_class"].nunique(), len(corrupted), len(unlabeled)],
    })


def show_sample_images(inventory: pd.DataFrame, seed: int, path: Path | None = None, samples_per_class: int = 5) -> None:
    rng = np.random.default_rng(seed)
    class_names = sorted(inventory["display_class"].unique())
    fig, axes = plt.subplots(len(class_names), samples_per_class, figsize=(15, 3 * len(class_names)))
    for row_idx, class_name in enumerate(class_names):
        paths = inventory.loc[inventory["display_class"] == class_name, "path"].to_numpy()
        chosen = rng.choice(paths, size=samples_per_class, replace=False)
        for col_idx, image_path in enumerate(chosen):
            ax = axes[row_idx, col_idx]
            ax.imshow(Image.open(image_path).convert("RGB"))
            ax.set_title(class_name, fontsize=9)
            ax.axis("off")
    fig.suptitle("Sample images per class", fontsize=14)
    fig.tight_layout()
    save_show(fig, path)


def show_dataset_eda(inventory: pd.DataFrame, path: Path | None = None) -> None:
    class_distribution = inventory["display_class"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_distribution.plot(kind="bar", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Class distribution")
    axes[0].set_xlabel("class")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].hist(inventory["width"], bins=30, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Image width distribution")
    axes[1].set_xlabel("width")
    axes[1].set_ylabel("frequency")

    axes[2].hist(inventory["aspect_ratio"], bins=30, color="#2ca02c", alpha=0.85)
    axes[2].set_title("Aspect ratio distribution")
    axes[2].set_xlabel("aspect ratio")
    fig.tight_layout()
    save_show(fig, path)


def show_split_distribution(distribution: pd.DataFrame, path: Path | None = None) -> None:
    ax = distribution.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Class distribution after stratified split")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=30)
    fig = ax.get_figure()
    fig.tight_layout()
    save_show(fig, path)


def make_augmentation(seed: int) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        layers.RandomRotation(0.06, seed=seed),
        layers.RandomZoom(0.08, seed=seed),
        layers.RandomTranslation(0.06, 0.06, seed=seed),
    ], name="traffic_sign_augmentation")


def show_augmentation_examples(inventory: pd.DataFrame, class_name: str, seed: int, path: Path | None = None) -> tf.keras.Sequential:
    augmentation = make_augmentation(seed)
    sample_path = inventory.loc[inventory["display_class"] == class_name, "path"].sample(1, random_state=seed).iloc[0]
    image = tf.io.read_file(sample_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("original")
    axes[0, 0].axis("off")
    for idx in range(1, 10):
        augmented = augmentation(tf.expand_dims(image, 0), training=True)[0]
        ax = axes[idx // 5, idx % 5]
        ax.imshow(tf.clip_by_value(augmented, 0.0, 1.0))
        ax.set_title(f"aug {idx}")
        ax.axis("off")
    fig.tight_layout()
    save_show(fig, path)
    return augmentation
