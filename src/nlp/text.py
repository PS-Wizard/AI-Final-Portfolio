from __future__ import annotations

import html
import re
from collections.abc import Iterable
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}

DOMAIN_STOPWORDS = {"rt"}


class Normalizer(Protocol):
    def normalize(self, word: str) -> str: ...


class StemNormalizer:
    def __init__(self) -> None:
        from nltk.stem import PorterStemmer

        self.stemmer = PorterStemmer()

    def normalize(self, word: str) -> str:
        return self.stemmer.stem(word)


class LemmaNormalizer:
    def __init__(self) -> None:
        import nltk
        from nltk.stem import WordNetLemmatizer

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            try:
                nltk.data.find("corpora/wordnet.zip")
            except LookupError as exc:
                raise RuntimeError(
                    "NLTK wordnet is missing. Run this once in a separate cell: "
                    "import nltk; nltk.download('wordnet')"
                ) from exc
        self.lemmatizer = WordNetLemmatizer()

    def normalize(self, word: str) -> str:
        return self.lemmatizer.lemmatize(word)


def clean_text(text: str, normalizer: Normalizer | None = None) -> str:
    text = html.unescape(str(text).lower())
    for source, target in CONTRACTIONS.items():
        text = text.replace(source, target)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [
        word
        for word in text.split()
        if word not in ENGLISH_STOP_WORDS and word not in DOMAIN_STOPWORDS and len(word) > 1
    ]
    if normalizer is not None:
        words = [normalizer.normalize(word) for word in words]
    return " ".join(words)


def add_clean_text(df: pd.DataFrame, text_col: str = "text", normalizer: Normalizer | None = None) -> pd.DataFrame:
    out = df.copy()
    out["clean_text"] = out[text_col].map(lambda value: clean_text(value, normalizer))
    out["clean_word_count"] = out["clean_text"].str.split().map(len)
    return out


def make_tokenizer(texts: Iterable[str], vocab_size: int, oov_token: str = "<OOV>") -> Tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def percentile_max_len(word_counts: pd.Series, percentile: int = 95) -> int:
    return max(1, int(np.percentile(word_counts, percentile)))


def encode_texts(tokenizer: Tokenizer, texts: Iterable[str], max_len: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
