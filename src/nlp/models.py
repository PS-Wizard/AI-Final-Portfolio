from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, SimpleRNN, SpatialDropout1D
from tensorflow.keras.models import Sequential


def build_simple_rnn(vocab_size: int, num_classes: int, embedding_dim: int = 128) -> tf.keras.Model:
    return Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        SimpleRNN(64),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ], name="simple_rnn")


def build_lstm(vocab_size: int, num_classes: int, embedding_dim: int = 128, use_dropout: bool = False) -> tf.keras.Model:
    layers = [Embedding(input_dim=vocab_size, output_dim=embedding_dim), LSTM(64)]
    if use_dropout:
        layers.append(Dropout(0.5))
    layers.append(Dense(32, activation="relu"))
    if use_dropout:
        layers.append(Dropout(0.3))
    layers.append(Dense(num_classes, activation="softmax"))
    return Sequential(layers, name="lstm")


def build_bilstm(vocab_size: int, num_classes: int, embedding_dim: int = 128, use_dropout: bool = False) -> tf.keras.Model:
    layers = [Embedding(input_dim=vocab_size, output_dim=embedding_dim), Bidirectional(LSTM(64))]
    if use_dropout:
        layers.append(Dropout(0.5))
    layers.append(Dense(32, activation="relu"))
    if use_dropout:
        layers.append(Dropout(0.3))
    layers.append(Dense(num_classes, activation="softmax"))
    return Sequential(layers, name="bilstm")


def build_small_regularized_lstm(vocab_size: int, num_classes: int, embedding_dim: int = 64) -> tf.keras.Model:
    return Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        SpatialDropout1D(0.30),
        LSTM(32, dropout=0.30, recurrent_dropout=0.20),
        Dense(16, activation="relu"),
        Dropout(0.30),
        Dense(num_classes, activation="softmax"),
    ], name="lstm_small_regularized")


def build_pretrained_lstm(
    embedding_matrix: np.ndarray,
    num_classes: int,
    trainable: bool = True,
) -> tf.keras.Model:
    return Sequential([
        Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=trainable,
        ),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="pretrained_lstm")


def build_pretrained_small_regularized_lstm(
    embedding_matrix: np.ndarray,
    num_classes: int,
    trainable: bool = True,
) -> tf.keras.Model:
    return Sequential([
        Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=trainable,
        ),
        SpatialDropout1D(0.30),
        LSTM(32, dropout=0.30, recurrent_dropout=0.20),
        Dense(16, activation="relu"),
        Dropout(0.30),
        Dense(num_classes, activation="softmax"),
    ], name="pretrained_lstm_small_regularized")
