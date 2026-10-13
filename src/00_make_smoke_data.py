"""SMOKE TEST ONLY - not part of the real pipeline and not called by run_all.sh.

Use this when you have a few .mp4 clips in data/raw/videos/ but NOT the official
ChaLearn label / transcript files yet, and only want to check that the code runs.

It does three things:
1. Restores the original ChaLearn filenames (e.g. `--Ymqszjv54.001.mp4`). Clips saved from the
   browser get generic names, but the pipeline needs the real name (transcript lookup, youtube_id).
   Each local file is matched by SHA-256 to the file list of the Hugging Face mirror.
   Every rename is logged in data/raw/video_rename_map.csv so it can be undone.
2. Fetches the REAL transcriptions of those clips from the Hugging Face mirror by streaming
   (Video(decode=False) -> only metadata is read, no video is downloaded)
   -> data/raw/transcripts/transcription_hf_subset.pkl
3. Writes a DUMMY annotation file with RANDOM Big Five scores
   -> data/raw/annotations/DUMMY_FAKE_annotation_RANDOM_SCORES.pkl

!!! The labels from step 3 are FAKE. Any metric computed from them is meaningless. !!!
As soon as the real annotation_training.pkl is in data/raw/annotations/, Phase 1 uses it instead.

Needs one extra package:  pip install datasets
Run:  python src/00_make_smoke_data.py
"""
import csv
import hashlib
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

import utils
import config

log = utils.get_logger("00_smoke")

HF_REPO = "yeray142/first-impressions-v2"
HF_SPLITS = ["train", "validation", "test"]
RENAME_MAP_CSV = os.path.join(config.RAW_DIR, "video_rename_map.csv")
TRANSCRIPT_PKL = os.path.join(config.TRANSCRIPTS_DIR, "transcription_hf_subset.pkl")
DUMMY_PKL = os.path.join(config.ANNOTATIONS_DIR, config.DUMMY_ANNOTATION_NAME)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Step 1: restore original filenames
# ----------------------------------------------------------------------------
def list_hub_videos():
    """File list of the mirror: {sha256: (split, filename)} and {filename: split}."""
    from huggingface_hub import HfApi
    api = HfApi()
    by_hash, by_name, sizes = {}, {}, set()
    for split in HF_SPLITS:
        files = api.list_repo_tree(HF_REPO, path_in_repo=split, repo_type="dataset")
        for f in tqdm(files, desc=f"list hub/{split}", unit="file"):
            name = os.path.basename(f.path)
            if not name.endswith(".mp4") or f.lfs is None:
                continue
            by_hash[f.lfs.sha256] = (split, name)
            by_name[name] = split
            sizes.add(f.size)
    return by_hash, by_name, sizes


def restore_filenames():
    """Rename local clips to their original names. Returns {original_filename: hub_split}."""
    local = sorted(f for f in os.listdir(config.VIDEOS_DIR) if f.lower().endswith(".mp4"))
    if not local:
        sys.exit(f"MISSING FILES: no .mp4 videos in {config.VIDEOS_DIR}/")
    by_hash, by_name, sizes = list_hub_videos()

    present = {}                       # original filename -> hub split
    renamed, duplicates, unmatched = [], [], []
    for fname in tqdm(local, desc="match videos"):
        path = os.path.join(config.VIDEOS_DIR, fname)
        if fname in by_name:                               # already has its original name (re-run)
            present[fname] = by_name[fname]
            continue
        # Cheap size test first, then the exact SHA-256 comparison.
        match = by_hash.get(sha256_of(path)) if os.path.getsize(path) in sizes else None
        if match is None:
            unmatched.append(fname)
            continue
        split, original = match
        target = os.path.join(config.VIDEOS_DIR, original)
        if os.path.exists(target):                         # same clip downloaded twice
            duplicates.append((fname, original))
            continue
        os.rename(path, target)
        renamed.append((fname, original))
        present[original] = split

    if renamed:
        new_file = not os.path.exists(RENAME_MAP_CSV)
        with open(RENAME_MAP_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["downloaded_name", "original_name"])
            writer.writerows(renamed)

    log.info(f"videos: {len(present)} usable ({len(renamed)} renamed now), "
             f"{len(duplicates)} duplicate(s), {len(unmatched)} unmatched")
    for fname, original in duplicates:
        log.info(f"  duplicate, left untouched: '{fname}' is the same file as {original}")
    for fname in unmatched:
        log.info(f"  not found on the hub, left untouched: '{fname}'")
    return present


# ----------------------------------------------------------------------------
# Step 2: real transcriptions, streamed from the hub
# ----------------------------------------------------------------------------
def fetch_transcripts(present):
    """present = {filename: hub_split}. Saves {filename: text} like the official pkl."""
    transcripts = {}
    if os.path.exists(TRANSCRIPT_PKL):
        with open(TRANSCRIPT_PKL, "rb") as f:
            transcripts = pickle.load(f)
    missing = set(present) - set(transcripts)
    if not missing:
        log.info(f"transcripts: all {len(present)} already cached in {os.path.basename(TRANSCRIPT_PKL)}")
        return

    from datasets import Video, load_dataset
    bar = tqdm(total=len(missing), desc="fetch transcripts")
    for split in HF_SPLITS:
        wanted = {name for name in missing if present[name] == split}
        if not wanted:
            continue
        ds = load_dataset(HF_REPO, split=split, streaming=True)
        ds = ds.cast_column("video", Video(decode=False))   # path only -> nothing is downloaded
        for row in ds:
            name = os.path.basename(row["video"]["path"])
            if name in wanted:
                transcripts[name] = row["transcription"] or ""
                wanted.discard(name)
                bar.update(1)
                if not wanted:
                    break
    bar.close()

    still_missing = set(present) - set(transcripts)
    with open(TRANSCRIPT_PKL, "wb") as f:
        pickle.dump(transcripts, f)
    log.info(f"transcripts: {len(present) - len(still_missing)} ok, {len(still_missing)} failed "
             f"-> {TRANSCRIPT_PKL}")


# ----------------------------------------------------------------------------
# Step 3: DUMMY labels
# ----------------------------------------------------------------------------
def make_dummy_annotations(present):
    """Random scores in [0, 1], same layout as the official file: {trait: {filename: score}}."""
    real = [n for n in config.REAL_ANNOTATION_NAMES
            if os.path.exists(os.path.join(config.ANNOTATIONS_DIR, n))]
    if real:
        sys.exit(f"Real labels are available ({real}) - refusing to create dummy labels.")

    rng = np.random.RandomState(config.SEED)
    names = sorted(present)
    dummy = {trait: {name: float(rng.uniform(0, 1)) for name in names}
             for trait in config.TRAITS + ["interview"]}
    dummy[config.DUMMY_KEY] = "RANDOM numbers for a smoke test. NOT real ChaLearn labels."
    with open(DUMMY_PKL, "wb") as f:
        pickle.dump(dummy, f)
    log.info(f"dummy labels for {len(names)} clips -> {DUMMY_PKL}")


def main():
    utils.set_seed()
    utils.ensure_dirs()
    present = restore_filenames()
    if not present:
        sys.exit("No local video could be matched to the Hugging Face mirror - nothing to do.")
    fetch_transcripts(present)
    make_dummy_annotations(present)
    utils.print_fake_labels_banner()


if __name__ == "__main__":
    main()
