from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.nlp.data import TARGET_NAMES
from src.nlp.text import clean_text, encode_texts


def predict_text(model, tokenizer, text: str, max_len: int, normalizer=None) -> pd.DataFrame:
    cleaned = clean_text(text, normalizer)
    encoded = encode_texts(tokenizer, [cleaned], max_len)
    probabilities = model.predict(encoded, verbose=0)[0]
    rows = [
        {"label": label, "probability": float(probabilities[idx])}
        for idx, label in enumerate(TARGET_NAMES)
    ]
    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def error_examples(predictions: pd.DataFrame, path: Path | None = None, per_class: int = 2) -> pd.DataFrame:
    mistakes = predictions.loc[~predictions["correct"]].copy()
    rows = [mistakes.loc[mistakes["label_clean"] == label].head(per_class) for label in TARGET_NAMES]
    out = pd.concat(rows, ignore_index=True) if rows else mistakes.head(6)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out
