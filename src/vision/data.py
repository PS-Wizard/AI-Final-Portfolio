from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_CLASS_MAP = {"DIrection": "Direction"}
IMG_SIZE = (128, 128)
TRANSFER_IMG_SIZE = (224, 224)
SCRATCH_BATCH_SIZE = 32
TRANSFER_BATCH_SIZE = 16


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_class_name(name: str) -> str:
    return DISPLAY_CLASS_MAP.get(name, name)


def scan_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dir = dataset_dir / "Train"
    test_dir = dataset_dir / "Test"
    records: list[dict] = []
    corrupted: list[dict] = []

    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        raw_class = class_dir.name
        display_class = normalize_class_name(raw_class)
        for image_path in sorted(class_dir.rglob("*")):
            if not is_image_file(image_path):
                continue
            try:
                with Image.open(image_path) as img:
                    img.verify()
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode
                records.append({
                    "raw_class": raw_class,
                    "display_class": display_class,
                    "path": str(image_path.resolve()),
                    "filename": image_path.name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "size_pixels": width * height,
                    "mode": mode,
                })
            except (UnidentifiedImageError, OSError, IOError) as exc:
                corrupted.append({
                    "raw_class": raw_class,
                    "display_class": display_class,
                    "path": str(image_path.resolve()),
                    "error": str(exc),
                })

    unlabeled = [
        {"path": str(path.resolve()), "filename": path.name}
        for path in sorted(test_dir.rglob("*"))
        if is_image_file(path)
    ]
    return pd.DataFrame(records), pd.DataFrame(corrupted), pd.DataFrame(unlabeled)


def split_dataset(inventory: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        inventory,
        test_size=0.30,
        stratify=inventory["display_class"],
        random_state=seed,
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["display_class"],
        random_state=seed,
    )
    return _with_split(train, "train"), _with_split(val, "val"), _with_split(test, "test")


def _with_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["split"] = split
    return frame.reset_index(drop=True)


def add_labels(*frames: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str], dict[str, int], dict[int, str]]:
    labels = sorted(frames[0]["display_class"].unique())
    label_to_id = {name: idx for idx, name in enumerate(labels)}
    id_to_label = {idx: name for name, idx in label_to_id.items()}
    out = []
    for frame in frames:
        labelled = frame.copy()
        labelled["label"] = labelled["display_class"].map(label_to_id).astype(int)
        out.append(labelled)
    return out, labels, label_to_id, id_to_label


@tf.autograph.experimental.do_not_convert
def decode_scratch(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


@tf.autograph.experimental.do_not_convert
def decode_transfer(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, TRANSFER_IMG_SIZE)
    image = mobilenet_preprocess_input(image)
    return image, label


def make_dataset(dataframe: pd.DataFrame, loader, batch_size: int, seed: int, shuffle: bool = False) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["path"].to_numpy(), dataframe["label"].to_numpy()))
    dataset = dataset.map(loader, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(len(dataframe), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_scratch_datasets(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int):
    return (
        make_dataset(train, decode_scratch, SCRATCH_BATCH_SIZE, seed, shuffle=True),
        make_dataset(val, decode_scratch, SCRATCH_BATCH_SIZE, seed),
        make_dataset(test, decode_scratch, SCRATCH_BATCH_SIZE, seed),
    )


def make_transfer_datasets(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int):
    return (
        make_dataset(train, decode_transfer, TRANSFER_BATCH_SIZE, seed, shuffle=True),
        make_dataset(val, decode_transfer, TRANSFER_BATCH_SIZE, seed),
        make_dataset(test, decode_transfer, TRANSFER_BATCH_SIZE, seed),
    )


def class_weights(train: pd.DataFrame) -> dict[int, float]:
    classes = np.array(sorted(train["label"].unique()))
    weights = compute_class_weight("balanced", classes=classes, y=train["label"].to_numpy())
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def split_distribution(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "train": train["display_class"].value_counts().sort_index(),
        "val": val["display_class"].value_counts().sort_index(),
        "test": test["display_class"].value_counts().sort_index(),
    }).fillna(0).astype(int)
