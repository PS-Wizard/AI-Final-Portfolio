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
from sklearn.metrics import classification_report, confusion_matrix

from src.plots import plot_confusion_matrix, plot_history
from src.vision.models import build_baseline_cnn, build_deeper_cnn, build_mobilenetv2, enable_mobilenetv2_finetuning


@dataclass(frozen=True)
class VisionExperiment:
    name: str
    build_model: Callable[[], tf.keras.Model]
    optimizer: Callable[[], tf.keras.optimizers.Optimizer]
    epochs: int
    class_weight: dict[int, float] | None = None
    use_transfer_data: bool = False


def baseline_experiments(num_classes: int, augmentation: tf.keras.Model, class_weight: dict[int, float]) -> list[VisionExperiment]:
    return [
        VisionExperiment(
            "baseline_cnn",
            lambda: build_baseline_cnn(num_classes),
            lambda: tf.keras.optimizers.Adam(1e-3),
            15,
        ),
        VisionExperiment(
            "baseline_cnn_augmentation",
            lambda: build_baseline_cnn(num_classes, augmentation),
            lambda: tf.keras.optimizers.Adam(1e-3),
            15,
        ),
        VisionExperiment(
            "baseline_cnn_class_weights",
            lambda: build_baseline_cnn(num_classes),
            lambda: tf.keras.optimizers.Adam(1e-3),
            15,
            class_weight=class_weight,
        ),
    ]


def deeper_experiments(num_classes: int, augmentation: tf.keras.Model) -> list[VisionExperiment]:
    return [
        VisionExperiment(
            "deeper_cnn_augmentation",
            lambda: build_deeper_cnn(num_classes, augmentation),
            lambda: tf.keras.optimizers.Adam(1e-3),
            15,
        ),
        VisionExperiment(
            "deeper_cnn_augmentation_batchnorm",
            lambda: build_deeper_cnn(num_classes, augmentation, use_batchnorm=True),
            lambda: tf.keras.optimizers.Adam(1e-3),
            15,
        ),
        VisionExperiment(
            "deeper_cnn_augmentation_dropout",
            lambda: build_deeper_cnn(num_classes, augmentation, use_dropout=True),
            lambda: tf.keras.optimizers.Adam(1e-3),
            15,
        ),
        VisionExperiment(
            "deeper_cnn_augmentation_batchnorm_dropout",
            lambda: build_deeper_cnn(num_classes, augmentation, use_batchnorm=True, use_dropout=True),
            lambda: tf.keras.optimizers.Adam(1e-3),
            15,
        ),
    ]


def optimizer_experiment(num_classes: int, augmentation: tf.keras.Model) -> VisionExperiment:
    return VisionExperiment(
        "deeper_cnn_augmentation_batchnorm_dropout_sgd",
        lambda: build_deeper_cnn(num_classes, augmentation, use_batchnorm=True, use_dropout=True),
        lambda: tf.keras.optimizers.SGD(1e-2, momentum=0.9),
        15,
    )


def run_experiment(
    experiment: VisionExperiment,
    train_ds,
    val_ds,
    test_ds,
    test_df: pd.DataFrame,
    labels: list[str],
    outputs,
    model: tf.keras.Model | None = None,
) -> dict:
    model = model or experiment.build_model()
    optimizer = experiment.optimizer()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    _save_model_summary(model, outputs.summaries / f"{experiment.name}_summary.txt")

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=experiment.epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        class_weight=experiment.class_weight,
        verbose=1,
    )
    training_time = time.perf_counter() - start

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(outputs.histories / f"{experiment.name}_history.csv", index=False)
    plot_history(history, _title(experiment.name), outputs.figures / "training_curves" / f"{experiment.name}_curves.png")

    metrics, predictions, cm, report = evaluate_classifier(model, test_ds, test_df, labels)
    predictions.to_csv(outputs.predictions / f"{experiment.name}_predictions.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(outputs.tables / f"{experiment.name}_classification_report.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(outputs.tables / f"{experiment.name}_confusion_matrix.csv")
    plot_confusion_matrix(cm, labels, _title(experiment.name), outputs.figures / "confusion_matrices" / f"{experiment.name}_cm.png")

    metrics.update({
        "experiment_name": experiment.name,
        "epochs_trained": int(len(history.history["loss"])),
        "training_time_seconds": float(training_time),
        "trainable_parameters": int(np.sum([np.prod(w.shape) for w in model.trainable_weights])),
        "non_trainable_parameters": int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights])),
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
        "model_path": str((outputs.models / f"{experiment.name}.keras").resolve()),
    })
    model.save(outputs.models / f"{experiment.name}.keras")
    (outputs.tables / f"{experiment.name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "predictions": predictions, "confusion_matrix": cm, "history": history, "model": model}


def run_many(experiments: list[VisionExperiment], datasets, test_df: pd.DataFrame, labels: list[str], outputs) -> dict[str, dict]:
    scratch = datasets["scratch"]
    transfer = datasets["transfer"]
    results = {}
    for experiment in experiments:
        train_ds, val_ds, test_ds = transfer if experiment.use_transfer_data else scratch
        results[experiment.name] = run_experiment(experiment, train_ds, val_ds, test_ds, test_df, labels, outputs)
    return results


def run_transfer_experiments(datasets, test_df: pd.DataFrame, labels: list[str], outputs) -> dict[str, dict]:
    train_ds, val_ds, test_ds = datasets["transfer"]
    model, base = build_mobilenetv2(len(labels))
    feature_extraction = VisionExperiment(
        "mobilenetv2_feature_extraction",
        lambda: model,
        lambda: tf.keras.optimizers.Adam(1e-3),
        12,
        use_transfer_data=True,
    )
    results = {
        feature_extraction.name: run_experiment(feature_extraction, train_ds, val_ds, test_ds, test_df, labels, outputs, model=model)
    }

    enable_mobilenetv2_finetuning(base)
    fine_tuning = VisionExperiment(
        "mobilenetv2_finetuning",
        lambda: model,
        lambda: tf.keras.optimizers.Adam(1e-5),
        8,
        use_transfer_data=True,
    )
    results[fine_tuning.name] = run_experiment(fine_tuning, train_ds, val_ds, test_ds, test_df, labels, outputs, model=model)
    return results


def evaluate_classifier(model: tf.keras.Model, dataset, dataframe: pd.DataFrame, labels: list[str]):
    test_loss, test_accuracy = model.evaluate(dataset, verbose=0)
    probabilities = model.predict(dataset, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    y_true = dataframe["label"].to_numpy()
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    predictions = dataframe[["path", "filename", "display_class", "label"]].copy()
    predictions["predicted_label_id"] = y_pred
    predictions["predicted_class"] = [labels[idx] for idx in y_pred]
    predictions["confidence"] = probabilities.max(axis=1)
    predictions["correct"] = predictions["label"].to_numpy() == y_pred

    metrics = {
        "accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "misclassified_samples": int((~predictions["correct"]).sum()),
    }
    return metrics, predictions, cm, report


def registry(results: dict[str, dict], path: Path | None = None) -> pd.DataFrame:
    frame = pd.DataFrame([result["metrics"] for result in results.values()])
    frame = frame.sort_values(["macro_f1", "accuracy", "training_time_seconds"], ascending=[False, False, True]).reset_index(drop=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
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
