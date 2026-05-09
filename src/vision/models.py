from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from src.vision.data import IMG_SIZE, TRANSFER_IMG_SIZE


def build_baseline_cnn(num_classes: int, augmentation: tf.keras.Model | None = None) -> tf.keras.Model:
    model_layers: list[tf.keras.layers.Layer] = [layers.Input(shape=(*IMG_SIZE, 3))]
    if augmentation is not None:
        model_layers.append(augmentation)
    model_layers.extend([
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return models.Sequential(model_layers, name="baseline_cnn")


def build_deeper_cnn(
    num_classes: int,
    augmentation: tf.keras.Model,
    use_batchnorm: bool = False,
    use_dropout: bool = False,
) -> tf.keras.Model:
    model_layers: list[tf.keras.layers.Layer] = [layers.Input(shape=(*IMG_SIZE, 3)), augmentation]
    for filters in (32, 64, 128):
        model_layers.append(layers.Conv2D(filters, 3, padding="same"))
        if use_batchnorm:
            model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.Activation("relu"))
        model_layers.append(layers.Conv2D(filters, 3, padding="same"))
        if use_batchnorm:
            model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.Activation("relu"))
        model_layers.append(layers.MaxPooling2D(2))
        if use_dropout:
            model_layers.append(layers.Dropout(0.25))

    model_layers.extend([
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
    ])
    if use_dropout:
        model_layers.append(layers.Dropout(0.4))
    model_layers.append(layers.Dense(num_classes, activation="softmax"))
    return models.Sequential(model_layers, name="deeper_cnn")


def build_mobilenetv2(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = MobileNetV2(input_shape=(*TRANSFER_IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=(*TRANSFER_IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2"), base


def enable_mobilenetv2_finetuning(base: tf.keras.Model, fine_tune_at: int = 130) -> None:
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
