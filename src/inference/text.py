"""Phase 5 building blocks: transcript -> MiniLM sentence embedding + text emotion.

Moved here from src/05_features_text.py (the behaviour is unchanged) so that src/predict.py
uses exactly the same steps. Both builders return a function that takes a list of transcripts
(as model_text() returns them) and gives one row per transcript; an empty transcript gives a
row of zeros, as in training. See the docstring of src/05_features_text.py.
"""
import re

import numpy as np
import torch

import utils
import config


def model_text(raw_text, missing):
    """Transcript as the transformer models read it: notes removed, punctuation and case kept."""
    if missing:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", " ", str(raw_text))).strip()


def build_minilm(device):
    """Returns embed(texts) -> array (n, 384)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.TEXT_EMBED_MODEL, device=str(device))
    model.eval()

    def embed(texts):
        out = np.zeros((len(texts), config.FEATURE_DIMS["text_minilm"]), dtype=np.float32)
        filled = [i for i, t in enumerate(texts) if t]                # empty text stays all-zero
        if filled:
            with torch.no_grad():
                out[filled] = model.encode([texts[i] for i in filled],
                                           batch_size=config.TEXT_BATCH_CLIPS,
                                           convert_to_numpy=True, show_progress_bar=False)
        return out

    return embed


def build_text_emotion(device):
    """Returns (analyse, labels). analyse(texts) -> array (n, 7) of emotion probabilities."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMOTION_MODEL)
    model = utils.load_pretrained(AutoModelForSequenceClassification, config.TEXT_EMOTION_MODEL)
    model = model.eval().to(device)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @torch.no_grad()
    def analyse(texts):
        out = np.zeros((len(texts), len(labels)), dtype=np.float32)
        filled = [i for i, t in enumerate(texts) if t]
        if filled:
            encoded = tokenizer([texts[i] for i in filled], padding=True, truncation=True,
                                max_length=config.TEXT_MAX_TOKENS, return_tensors="pt").to(device)
            # softmax over the logits keeps ALL classes (no top-k cut)
            out[filled] = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
        return out

    return analyse, labels
