"""Funciones de preprocesamiento de texto compartidas."""

import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download("stopwords", quiet=True)
STOPWORDS_ES = set(stopwords.words("spanish"))


def preprocess(text: str) -> str:
    """Limpia y normaliza un texto en español."""
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS_ES and len(w) > 2]
    return " ".join(tokens)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=str)
        cleaned = []
        for text in X:
            t = text.lower()
            t = re.sub(r"\d+", " ", t)
            t = re.sub(r"[^a-záéíóúüñ\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            if self.remove_stopwords:
                words = [w for w in t.split() if w not in STOPWORDS_ES]
                t = " ".join(words)
            cleaned.append(t)
        return cleaned