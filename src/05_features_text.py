"""Phase 5 - Text features, from the transcripts in data/splits.csv.

1. TF-IDF (always) -> features/text_tfidf.npy (N, size of vocabulary) + features/tfidf_vocab.json
   Counts single words and word pairs of `text_clean`, weighted so that words used in few
   transcripts count more than words used everywhere. The vocabulary and the weights are
   learned from the TRAIN split only and then applied to all clips (no leakage).
   One-letter words are kept on purpose: "i" is a known personality cue.
   This feature is the separate text baseline of Phase 6 and the source of the word cues of
   Phase 9. It is recomputed on every run (fast, and it depends on the split).

2. Sentence embedding (always) -> features/text_minilm.npy (N, 384)
   `sentence-transformers/all-MiniLM-L6-v2`, FROZEN: one vector for the meaning of the whole
   transcript.

3. Text emotion (only if USE_HEAVY_MODELS) -> features/text_emotion.npy (N, 7)
   `j-hartmann/emotion-english-distilroberta-base`, FROZEN: softmax over ALL emotion classes,
   from the logits, in the model's fixed class order -> features/text_emotion_labels.json

The two transformer models were trained on normal sentences, so they read the transcript WITH
its punctuation and capital letters (only transcriber notes like "[inaudible 00:00:07]" are
removed). The bag-of-words TF-IDF uses the fully cleaned `text_clean`.
A clip without transcript (missing_text = 1) gets a row of zeros in every text feature.

Resume: an embedding / emotion row is reused only if the clip's transcript is still the same
text it was computed from (features/text_hashes.json), so only new or changed clips run
through the models.

Run:  python src/05_features_text.py
"""
import hashlib
import json
import os
import re
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

import utils
import config

log = utils.get_logger("05_text")


def model_text(raw_text, missing):
    """Transcript as the transformer models read it: notes removed, punctuation and case kept."""
    if missing:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", " ", str(raw_text))).strip()


# ----------------------------------------------------------------------------
# 1. TF-IDF
# ----------------------------------------------------------------------------
def tfidf_features(df, ids):
    vectorizer = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES,
                                 ngram_range=config.TFIDF_NGRAMS, min_df=config.TFIDF_MIN_DF,
                                 token_pattern=r"(?u)\b\w[\w']*\b")     # keeps "i" and "don't"
    train_text = df.loc[df["split"] == "train", "text_clean"]
    try:
        vectorizer.fit(train_text)                                    # TRAIN split only
    except ValueError:
        sys.exit("TF-IDF found no usable words in the train transcripts - are they all empty?")
    matrix = vectorizer.transform(df["text_clean"]).toarray()
    vocabulary = vectorizer.get_feature_names_out().tolist()
    utils.save_features("text_tfidf", ids, matrix)
    with open(config.TFIDF_VOCAB_JSON, "w") as f:
        json.dump(vocabulary, f, indent=1)
    joblib.dump(vectorizer, config.TFIDF_VECTORIZER_FILE)             # reused by the demo
    log.info(f"TF-IDF: fitted on {len(train_text)} train transcripts, "
             f"vocabulary of {len(vocabulary)} terms")


# ----------------------------------------------------------------------------
# 2. MiniLM sentence embedding
# ----------------------------------------------------------------------------
def build_minilm(device, texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.TEXT_EMBED_MODEL, device=str(device))
    model.eval()

    def compute_batch(batch_ids):
        out = np.zeros((len(batch_ids), config.FEATURE_DIMS["text_minilm"]), dtype=np.float32)
        filled = [i for i, v in enumerate(batch_ids) if texts[v]]     # empty text stays all-zero
        if filled:
            with torch.no_grad():
                out[filled] = model.encode([texts[batch_ids[i]] for i in filled],
                                           batch_size=config.TEXT_BATCH_CLIPS,
                                           convert_to_numpy=True, show_progress_bar=False)
        return {"text_minilm": out}

    return compute_batch


# ----------------------------------------------------------------------------
# 3. Text emotion model
# ----------------------------------------------------------------------------
def build_text_emotion(device, texts):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMOTION_MODEL)
    model = utils.load_pretrained(AutoModelForSequenceClassification, config.TEXT_EMOTION_MODEL)
    model = model.eval().to(device)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @torch.no_grad()
    def compute_batch(batch_ids):
        out = np.zeros((len(batch_ids), len(labels)), dtype=np.float32)
        filled = [i for i, v in enumerate(batch_ids) if texts[v]]
        if filled:
            encoded = tokenizer([texts[batch_ids[i]] for i in filled], padding=True, truncation=True,
                                max_length=config.TEXT_MAX_TOKENS, return_tensors="pt").to(device)
            # softmax over the logits keeps ALL classes (no top-k cut)
            out[filled] = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
        return {"text_emotion": out}

    return compute_batch, labels


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_model(label, names, ids, build, device, texts, hashes):
    """Load one model, extract its features for clips whose text is new or changed, report timing."""
    current = {v: hashlib.md5(texts[v].encode()).hexdigest() for v in ids}
    stored = hashes.get(names[0], {})

    def text_changed_time(video_id):
        # cached_extract reuses a row if its source is "not newer" than the feature file:
        # same text -> 0 (always older), different or unknown text -> infinity (always newer).
        return 0.0 if stored.get(video_id) == current[video_id] else float("inf")

    start = time.time()
    built = build(device, texts)
    compute_batch, extra = built if isinstance(built, tuple) else (built, None)
    load_sec = time.time() - start
    n_new, sec, replaced = utils.cached_extract(names, ids, compute_batch, text_changed_time,
                                                config.TEXT_BATCH_CLIPS, label)
    per_clip = f"{sec / n_new:.4f} s per clip" if n_new else "nothing new to compute"
    log.info(f"{label}: model loaded in {load_sec:.1f} s; {n_new} clips in {sec:.1f} s = {per_clip}; "
             f"NaN/inf replaced: {replaced}")
    for name in names:
        hashes[name] = current
    del built, compute_batch              # drop the model before the next one is loaded (low RAM)
    utils.free_memory()
    return extra


def main():
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()

    df = utils.load_splits()
    if "text_clean" not in df.columns:
        sys.exit("splits.csv has no text_clean column - run `python src/02_preprocess.py` first.")
    ids = df["video_id"].tolist()
    utils.write_ids(ids)                       # the row order every feature file must follow
    device = utils.get_device()
    n_missing = int(df["missing_text"].sum())
    log.info(f"{len(ids)} clips ({n_missing} without transcript -> zero rows), device = {device}, "
             f"USE_HEAVY_MODELS = {config.USE_HEAVY_MODELS}")

    texts = {v: model_text(t, m) for v, t, m in zip(df["video_id"], df["text"], df["missing_text"])}
    hashes = {}
    if os.path.exists(config.TEXT_HASHES_JSON):
        with open(config.TEXT_HASHES_JSON) as f:
            hashes = json.load(f)

    tfidf_features(df, ids)
    names = ["text_tfidf", "text_minilm"]
    run_model("MiniLM sentence embedding", ["text_minilm"], ids, build_minilm, device, texts, hashes)

    if config.USE_HEAVY_MODELS:
        labels = run_model("text emotion model", ["text_emotion"], ids, build_text_emotion,
                           device, texts, hashes)
        with open(config.TEXT_EMOTION_LABELS_JSON, "w") as f:
            json.dump(labels, f, indent=2)
        log.info(f"text_emotion column order: {labels}")
        names.append("text_emotion")
    else:
        log.info("USE_HEAVY_MODELS is off: text_emotion is not computed")
    with open(config.TEXT_HASHES_JSON, "w") as f:
        json.dump(hashes, f)

    # ---- acceptance check: shapes, row order == ids.csv, no NaN/inf ----
    shapes = utils.check_features(names)
    for name, shape in shapes.items():
        log.info(f"  {name}.npy  shape {shape}")
    stale = utils.stale_feature_files(skip=names)
    if stale:
        log.info(f"NOTE: other feature files do not match the current ids.csv and need their "
                 f"script re-run before use: {stale}")
    log.info(f"ACCEPTANCE CHECK PASSED: {len(names)} feature file(s) match features/ids.csv "
             f"({len(ids)} rows, same order), no NaN/inf")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
