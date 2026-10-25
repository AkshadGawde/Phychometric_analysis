"""Phase 4 - Audio features, from the cleaned wav files of Phase 2 (16 kHz mono).

1. Hand-crafted features (always) -> features/audio_mfcc.npy  (N, 87)
   Every clip is cut into short overlapping frames (64 ms, one every 16 ms); each feature is
   measured per frame and then summarised over the whole clip:
     mfcc1..40 mean + std   80   shape of the sound spectrum = voice timbre
     pitch_mean, pitch_std   2   how high the voice is and how much it moves (intonation)
     voiced_fraction         1   share of frames in which the vocal cords vibrate
     rms_mean, rms_std       2   loudness and how much it varies
     zcr_mean                1   zero-crossing rate: noisy / hissy sounds vs voiced sounds
     nonsilent_fraction      1   speaking-rate proxy: share of frames that are not a pause
   Pitch uses librosa.pyin, which also decides per frame whether it is voiced. Unvoiced frames
   are NaN, so the statistics are NaN-safe (nanmean / nanstd) and become 0 when a clip has no
   voiced frame at all.
   The column names are saved to features/audio_mfcc_cols.json (used by the cue analysis).

2. Speech model (only if USE_HEAVY_MODELS)
   Hugging Face `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`, FROZEN
   (eval() + torch.no_grad()). The waveform is normalised by the model's own feature extractor.
   (a) mean of the last hidden state over time -> features/audio_w2v.npy      (N, 1024)
   (b) softmax over ALL emotion classes, from the logits, in the model's fixed class order
       -> features/audio_emotion.npy  (N, 8) + features/audio_emotion_labels.json

   IMPORTANT: the emotion head stored in this checkpoint (classifier.dense, classifier.output)
   is not the layout the standard transformers classification model expects, so the standard
   loader silently replaces it with a RANDOM head. build_wav2vec2() therefore loads the
   wav2vec2 encoder normally and rebuilds the checkpoint's own head by hand:
       mean-pool -> dense (1024 -> 1024) -> tanh -> output (1024 -> 8)

Very short or near-silent audio never stops the run: short audio is padded with silence,
problems are logged and listed in features/audio_problems.csv.

Resume: rows computed earlier are reused; only new clips (or clips whose wav was re-made by
Phase 2) are analysed.

Run:  python src/04_features_audio.py
"""
import json
import os
import sys
import time

import librosa
import numpy as np
import pandas as pd

import utils
import config
from inference import audio
from inference.audio import MFCC_COLUMNS

log = utils.get_logger("04_audio")

problems = {}          # video_id -> text, for clips with unusual audio (filled while extracting)
analysed = set()       # clips whose audio was looked at in this run


def wav_mtime(video_id):
    """When the clip's wav was last written (features older than this are recomputed)."""
    return os.path.getmtime(utils.audio_path(video_id))


def load_audio(video_id):
    """The cleaned waveform of one clip, made safe to analyse. Problems are noted, not raised."""
    y, _ = librosa.load(utils.audio_path(video_id), sr=config.SAMPLE_RATE, mono=True)
    analysed.add(video_id)
    y, notes = audio.prepare_waveform(y)
    if notes:
        problems[video_id] = "; ".join(notes)
    return y


# ----------------------------------------------------------------------------
# 1. Hand-crafted features
# ----------------------------------------------------------------------------
# (the features themselves: src/inference/audio.py, shared with the app)
def handcrafted_batch(batch_ids):
    rows = []
    for video_id in batch_ids:
        try:
            rows.append(audio.handcrafted_features(load_audio(video_id)))
        except Exception as error:               # one bad clip must not stop the whole run
            problems[video_id] = f"hand-crafted features failed ({error}), row set to 0"
            rows.append(np.zeros(len(MFCC_COLUMNS), dtype=np.float32))
    return {"audio_mfcc": np.stack(rows)}


# ----------------------------------------------------------------------------
# 2. wav2vec2 speech-emotion model
# ----------------------------------------------------------------------------
# (the model itself: src/inference/audio.py, shared with the app)
def make_w2v_batch(analyse, n_labels):
    def w2v_batch(batch_ids):
        embeddings, probabilities = [], []
        for video_id in batch_ids:            # one clip at a time: no padding, so the mean is exact
            try:
                embedding, probs = analyse(load_audio(video_id))
            except Exception as error:
                problems[video_id] = f"wav2vec2 failed ({error}), row set to 0"
                embedding = np.zeros(config.FEATURE_DIMS["audio_w2v"], dtype=np.float32)
                probs = np.zeros(n_labels, dtype=np.float32)
            embeddings.append(embedding)
            probabilities.append(probs)
        utils.free_memory()                   # clips differ in length; keep the GPU cache small
        return {"audio_w2v": np.stack(embeddings), "audio_emotion": np.stack(probabilities)}
    return w2v_batch


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def extract(label, names, ids, compute_batch, batch_size, use_cache=True):
    n_new, sec, replaced = utils.cached_extract(names, ids, compute_batch, wav_mtime, batch_size,
                                                label, use_cache=use_cache)
    per_clip = f"{sec / n_new:.3f} s per clip" if n_new else "nothing new to compute"
    log.info(f"{label}: {n_new} clips in {sec:.1f} s = {per_clip}; NaN/inf replaced: {replaced}")
    return n_new


def save_problems(ids):
    """features/audio_problems.csv: clips with very short / near-silent / failed audio.
    Entries of clips that were not re-analysed in this run are kept from the earlier file."""
    known = {}
    if os.path.exists(config.AUDIO_PROBLEMS_CSV):
        old = pd.read_csv(config.AUDIO_PROBLEMS_CSV, dtype=str, keep_default_na=False)
        known = dict(zip(old.video_id, old.problem))
    known = {v: p for v, p in known.items() if v in set(ids) and v not in analysed}
    known.update(problems)
    pd.DataFrame({"video_id": list(known), "problem": list(known.values())}).to_csv(
        config.AUDIO_PROBLEMS_CSV, index=False)
    log.info(f"clips with unusual audio (very short / near-silent / failed): {len(known)}")
    for video_id, problem in known.items():
        log.info(f"  {video_id}: {problem}")


def main():
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()

    df = utils.load_splits()
    ids = df["video_id"].tolist()
    without_wav = [v for v in ids if not os.path.exists(utils.audio_path(v))]
    if without_wav:
        sys.exit(f"{len(without_wav)} clip(s) of splits.csv have no wav (e.g. {without_wav[0]}) "
                 f"- run `python src/02_preprocess.py` first.")

    utils.write_ids(ids)                       # the row order every feature file must follow
    log.info(f"{len(ids)} clips, USE_HEAVY_MODELS = {config.USE_HEAVY_MODELS}")

    # ---- 1. hand-crafted ----
    # If the list of columns changed since the file was made, the old rows mean something else.
    same_columns = (os.path.exists(config.AUDIO_MFCC_COLS_JSON)
                    and json.load(open(config.AUDIO_MFCC_COLS_JSON)) == MFCC_COLUMNS)
    extract("hand-crafted (MFCC, pitch, energy)", ["audio_mfcc"], ids, handcrafted_batch,
            config.AUDIO_BATCH_CLIPS, use_cache=same_columns)
    with open(config.AUDIO_MFCC_COLS_JSON, "w") as f:
        json.dump(MFCC_COLUMNS, f, indent=2)
    names = ["audio_mfcc"]

    # ---- 2. wav2vec2 ----
    if config.USE_HEAVY_MODELS:
        device = utils.get_device()
        start = time.time()
        analyse, labels = audio.build_wav2vec2(device)
        log.info(f"wav2vec2 loaded in {time.time() - start:.1f} s on {device}")
        extract("wav2vec2 (embedding + emotion)", ["audio_w2v", "audio_emotion"], ids,
                make_w2v_batch(analyse, len(labels)), config.AUDIO_BATCH_CLIPS)
        del analyse                   # the 1.2 GB model is no longer needed
        utils.free_memory()
        with open(config.AUDIO_EMOTION_LABELS_JSON, "w") as f:
            json.dump(labels, f, indent=2)
        log.info(f"audio_emotion column order: {labels}")
        names += ["audio_w2v", "audio_emotion"]
    else:
        log.info("USE_HEAVY_MODELS is off: audio_w2v / audio_emotion are not computed")

    save_problems(ids)

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
