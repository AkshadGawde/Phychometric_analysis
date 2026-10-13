"""Fetch more labelled clips and all transcripts from the Hugging Face mirror.

The official ChaLearn site needs a login for every download. The mirror
`yeray142/first-impressions-v2` holds the same 10,000 clips (same file names) plus their
transcripts, but NOT the trait labels - those come from the official annotation files that must
already be in data/raw/annotations/.

What it does
1. Transcripts: reads the three small metadata.csv files of the mirror and saves the
   transcript of every clip -> data/raw/transcripts/transcription_hf_mirror.pkl
   (same layout as the official file: {video_filename: text}).
2. Clips: downloads more .mp4 files into data/raw/videos/ until `--target` labelled clips are
   on disk. Only clips that have labels AND a non-empty transcript are chosen, in a fixed
   random order (SEED) over the mirror's train / validation / test folders.
   Before downloading it prints the free disk space and the estimated size, and stops if
   there is not enough room.

Resumable: clips already on disk are never downloaded again, and because the order is fixed
a re-run continues with exactly the clips that are still missing.

Run:  python src/00_fetch_more_clips.py --transcripts-only
      python src/00_fetch_more_clips.py --target 1500 [--workers 4]
"""
import argparse
import os
import pickle
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
import config

log = utils.get_logger("00_fetch")

HF_REPO = "yeray142/first-impressions-v2"
HF_SPLITS = ["train", "validation", "test"]
TRANSCRIPT_PKL = os.path.join(config.TRANSCRIPTS_DIR, "transcription_hf_mirror.pkl")
STAGING_DIR = os.path.join(config.RAW_DIR, "_hf_staging")     # partial downloads live here


def load_mirror_metadata():
    """One row per clip of the mirror: file_name, mirror split, transcription."""
    from huggingface_hub import hf_hub_download
    parts = []
    for split in HF_SPLITS:
        path = hf_hub_download(HF_REPO, f"{split}/metadata.csv", repo_type="dataset")
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        table["hf_split"] = split
        parts.append(table[["file_name", "hf_split", "transcription"]])
    meta = pd.concat(parts, ignore_index=True)
    log.info(f"mirror: {len(meta)} clips listed "
             f"({meta.hf_split.value_counts().reindex(HF_SPLITS).to_dict()})")
    return meta


def save_transcripts(meta):
    transcripts = dict(zip(meta.file_name, meta.transcription))
    with open(TRANSCRIPT_PKL, "wb") as f:
        pickle.dump(transcripts, f)
    empty = int((meta.transcription.str.strip() == "").sum())
    log.info(f"transcripts: {len(transcripts)} saved ({empty} of them empty) -> {TRANSCRIPT_PKL}")


def labelled_names():
    """File names that have a score for all five traits in the official annotation files."""
    names = None
    found = [n for n in config.REAL_ANNOTATION_NAMES
             if os.path.exists(os.path.join(config.ANNOTATIONS_DIR, n))]
    if not found:
        sys.exit(f"No official annotation file in {config.ANNOTATIONS_DIR}/ - labels are needed "
                 f"to know which clips are worth downloading.")
    labelled = set()
    for name in found:
        with open(os.path.join(config.ANNOTATIONS_DIR, name), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        names = set.intersection(*[set(data[t]) for t in config.TRAITS])
        labelled |= names
    log.info(f"labels: {len(labelled)} clips in {found}")
    return labelled


def choose_clips(meta, labelled, target):
    """The clips to download so that `target` labelled clips are on disk in the end."""
    on_disk = {f for f in os.listdir(config.VIDEOS_DIR) if f.lower().endswith(".mp4")}
    have = len(on_disk & labelled)
    usable = meta[meta.file_name.isin(labelled) & (meta.transcription.str.strip() != "")]
    log.info(f"mirror clips with labels: {int(meta.file_name.isin(labelled).sum())}, "
             f"of those with a non-empty transcript: {len(usable)}")

    # Fixed random order over ALL usable clips -> a re-run walks the same list again.
    order = sorted(usable.file_name)
    np.random.RandomState(config.SEED).shuffle(order)
    need = max(0, target - have)
    chosen = [name for name in order if name not in on_disk][:need]
    log.info(f"labelled clips on disk: {have}, target: {target}, to download: {len(chosen)}")
    split_of = dict(zip(meta.file_name, meta.hf_split))
    return [(split_of[name], name) for name in chosen]


def clip_sizes():
    """Size in bytes of every clip of the mirror (from the file listing, nothing is downloaded)."""
    from huggingface_hub import HfApi
    sizes = {}
    for split in HF_SPLITS:
        for item in tqdm(HfApi().list_repo_tree(HF_REPO, path_in_repo=split, repo_type="dataset"),
                         desc=f"list {split}", unit="file"):
            if item.path.endswith(".mp4"):
                sizes[os.path.basename(item.path)] = item.size
    return sizes


def check_disk_space(chosen, sizes):
    download_gb = sum(sizes.get(name, 0) for _, name in chosen) / 1e9
    processed_gb = len(chosen) * config.PROCESSED_MB_PER_CLIP / 1e3     # frames + wav + features
    free_gb = shutil.disk_usage(config.DATA_DIR).free / 1e9
    log.info(f"disk: {free_gb:.1f} GB free | estimated download {download_gb:.2f} GB "
             f"+ about {processed_gb:.2f} GB for frames/audio/features later")
    needed_gb = config.FETCH_FREE_SPACE_FACTOR * (download_gb + processed_gb)
    if free_gb < needed_gb:
        sys.exit(f"Not enough free disk space: {free_gb:.1f} GB free, want at least "
                 f"{needed_gb:.1f} GB. Free some space or lower --target.")


def download_one(split, name):
    """Download one clip into the staging folder, then move it into data/raw/videos/."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, f"{split}/{name}", repo_type="dataset", local_dir=STAGING_DIR)
    os.replace(path, os.path.join(config.VIDEOS_DIR, name))      # only complete files reach videos/


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=config.FETCH_TARGET_CLIPS,
                        help="labelled clips wanted on disk in total")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--transcripts-only", action="store_true")
    args = parser.parse_args()

    utils.set_seed()
    utils.ensure_dirs()
    meta = load_mirror_metadata()
    save_transcripts(meta)
    if args.transcripts_only:
        return

    chosen = choose_clips(meta, labelled_names(), args.target)
    if not chosen:
        log.info("nothing to download")
        return
    check_disk_space(chosen, clip_sizes())

    ok, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(download_one, split, name): name for split, name in chosen}
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="download", unit="clip"):
            try:
                job.result()
                ok += 1
            except Exception as error:                 # one failed clip must not stop the rest
                failed.append(jobs[job])
                tqdm.write(f"  FAILED {jobs[job]}: {error}")
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    log.info(f"download: {ok} ok, {len(failed)} failed "
             f"(re-run the same command to retry the failed ones)")
    on_disk = {f for f in os.listdir(config.VIDEOS_DIR) if f.lower().endswith(".mp4")}
    log.info(f"videos in {config.VIDEOS_DIR}: {len(on_disk)}")


if __name__ == "__main__":
    main()
