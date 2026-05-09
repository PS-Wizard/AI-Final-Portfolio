from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.nlp.data import TARGET_NAMES
from src.plots import plot_confusion_matrix, plot_history


@dataclass(frozen=True)
class NlpExperiment:
    name: str
    build_model: Callable[[], tf.keras.Model]
    class_weight: dict[int, float] | None = None
    epochs: int = 15
    batch_size: int = 32


def run_experiment(
    experiment: NlpExperiment,
    x_train: np.ndarray,
    y_train: pd.Series,
    x_test: np.ndarray,
    y_test: pd.Series,
    test_reference: pd.DataFrame,
    outputs,
) -> dict:
    model = experiment.build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    _save_model_summary(model, outputs.summaries / f"{experiment.name}_summary.txt")

    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train.reset_index(drop=True),
        validation_split=0.20,
        epochs=experiment.epochs,
        batch_size=experiment.batch_size,
        class_weight=experiment.class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=1,
    )
    training_time = time.perf_counter() - start

    pd.DataFrame(history.history).to_csv(outputs.histories / f"{experiment.name}_history.csv", index=False)
    plot_history(history, _title(experiment.name), outputs.figures / "training_curves" / f"{experiment.name}_curves.png")

    probabilities = model.predict(x_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    predictions = test_reference[["text", "clean_text", "label_clean", "label_id"]].copy()
    predictions["predicted_label_id"] = y_pred
    predictions["predicted_label"] = [TARGET_NAMES[idx] for idx in y_pred]
    predictions["confidence"] = confidence
    predictions["correct"] = predictions["label_id"].to_numpy() == y_pred
    predictions.to_csv(outputs.predictions / f"{experiment.name}_predictions.csv", index=False)

    pd.DataFrame(report).transpose().to_csv(outputs.tables / f"{experiment.name}_classification_report.csv")
    pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES).to_csv(outputs.tables / f"{experiment.name}_confusion_matrix.csv")
    plot_confusion_matrix(cm, TARGET_NAMES, _title(experiment.name), outputs.figures / "confusion_matrices" / f"{experiment.name}_cm.png")

    metrics = {
        "experiment_name": experiment.name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "misclassified_samples": int((~predictions["correct"]).sum()),
        "epochs_trained": int(len(history.history["loss"])),
        "training_time_seconds": float(training_time),
        "trainable_parameters": int(np.sum([np.prod(w.shape) for w in model.trainable_weights])),
        "non_trainable_parameters": int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights])),
    }
    model.save(outputs.models / f"{experiment.name}.keras")
    (outputs.tables / f"{experiment.name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "predictions": predictions, "confusion_matrix": cm, "history": history, "model": model}


def run_many(experiments: list[NlpExperiment], x_train, y_train, x_test, y_test, test_reference, outputs) -> dict[str, dict]:
    results = {}
    for experiment in experiments:
        results[experiment.name] = run_experiment(experiment, x_train, y_train, x_test, y_test, test_reference, outputs)
    return results


def registry(results: dict[str, dict], path: Path | None = None) -> pd.DataFrame:
    frame = pd.DataFrame([result["metrics"] for result in results.values()])
    frame = frame.sort_values(["macro_f1", "accuracy", "training_time_seconds"], ascending=[False, False, True]).reset_index(drop=True)
    if path is not None:
        frame.to_csv(path, index=False)
    return frame


def comparison(results: dict[str, dict], names: list[str], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame([results[name]["metrics"] for name in names])
    frame = frame.sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _save_model_summary(model: tf.keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _title(name: str) -> str:
    return name.replace("_", " ")
