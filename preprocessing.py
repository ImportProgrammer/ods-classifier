"""Funciones de preprocesamiento de texto compartidas."""

import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS_ES = set(stopwords.words("spanish"))


def preprocess(text: str) -> str:
    """Limpia y normaliza un texto en español."""
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS_ES and len(w) > 2]
    return " ".join(tokens)
