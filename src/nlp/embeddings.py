from __future__ import annotations

import numpy as np
import pandas as pd


def load_embedding(name: str):
    import gensim.downloader as api

    return api.load(name)


def build_embedding_matrix(
    embedding_model,
    word_index: dict[str, int],
    vocab_size: int,
    embedding_dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.05, size=(vocab_size, embedding_dim)).astype("float32")
    matrix[0] = np.zeros(embedding_dim, dtype="float32")

    found = 0
    missing = 0
    missing_examples: list[str] = []
    for word, index in word_index.items():
        if index >= vocab_size:
            continue
        if word in embedding_model:
            matrix[index] = embedding_model[word]
            found += 1
        else:
            missing += 1
            if len(missing_examples) < 25:
                missing_examples.append(word)

    coverage = pd.DataFrame({
        "metric": ["found_words", "missing_words", "coverage_percent"],
        "value": [found, missing, round(found / max(found + missing, 1) * 100, 2)],
    })
    missing_df = pd.DataFrame({"missing_word_examples": missing_examples})
    return matrix, coverage, missing_df
