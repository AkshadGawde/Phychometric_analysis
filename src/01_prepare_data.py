"""Phase 1 - Data preparation.

1. Unzip everything in data/raw/zips/ (nested zips too) and sort the contents:
   .mp4 -> data/raw/videos/, annotation pkl -> annotations/, transcription pkl -> transcripts/
2. Load the two pkl files and build one table: one row per clip.
3. Split 70/15/15 by youtube_id (so one person never appears in two splits).
4. Binarise each trait with the TRAIN median.
5. Save data/splits.csv and run the acceptance check.

Run:  python src/01_prepare_data.py
"""
import glob
import os
import pickle
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import utils
import config

log = utils.get_logger("01_prepare")

EXTRACT_TMP = os.path.join(config.RAW_DIR, "_extracted")       # scratch folder, deleted at the end
DONE_FILE = os.path.join(config.ZIPS_DIR, ".extracted.txt")    # names of zips already unpacked


# ----------------------------------------------------------------------------
# Step 1: unzip
# ----------------------------------------------------------------------------
def extract_zip(zip_path, out_dir):
    """Extract one zip. Uses the CHALEARN_PW env var if the archive is encrypted."""
    password = os.environ.get("CHALEARN_PW")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir, pwd=password.encode() if password else None)
    except NotImplementedError:
        # Python's zipfile cannot read some compression/encryption methods -> use system unzip.
        # (Must come before RuntimeError: NotImplementedError is a subclass of it.)
        cmd = ["unzip", "-o", "-q"] + (["-P", password] if password else []) + [zip_path, "-d", out_dir]
        subprocess.run(cmd, check=True)
    except RuntimeError as e:
        # zipfile raises RuntimeError for a missing or wrong password
        if "password" in str(e).lower():
            sys.exit(f"{os.path.basename(zip_path)} is password protected. Set the password shown on the "
                     f"ChaLearn data page and re-run:\n    export CHALEARN_PW='...'")
        raise


def sort_extracted_files(folder):
    """Move videos and pkl files out of the scratch folder into their final place."""
    n_videos = 0
    for dirpath, _, filenames in os.walk(folder):
        for name in filenames:
            src = os.path.join(dirpath, name)
            lower = name.lower()
            if name.startswith("._"):                      # macOS resource-fork junk
                continue
            if lower.endswith(".mp4"):
                shutil.move(src, os.path.join(config.VIDEOS_DIR, name))
                n_videos += 1
            elif lower.endswith(".pkl") and "annotation" in lower:
                shutil.move(src, os.path.join(config.ANNOTATIONS_DIR, name))
            elif lower.endswith(".pkl") and "transcription" in lower:
                shutil.move(src, os.path.join(config.TRANSCRIPTS_DIR, name))
    return n_videos


def unzip_all():
    """Unpack every new zip in data/raw/zips/. Original zips are left untouched."""
    done = set()
    if os.path.exists(DONE_FILE):
        done = set(open(DONE_FILE).read().split("\n"))

    zips = sorted(glob.glob(os.path.join(config.ZIPS_DIR, "*.zip")))
    todo = [z for z in zips if os.path.basename(z) not in done]
    log.info(f"{len(zips)} zip(s) in {config.ZIPS_DIR}; {len(todo)} still to extract")

    ok, failed = 0, 0
    for zip_path in tqdm(todo, desc="unzip"):
        shutil.rmtree(EXTRACT_TMP, ignore_errors=True)
        os.makedirs(EXTRACT_TMP)
        try:
            extract_zip(zip_path, EXTRACT_TMP)
            # Nested zips: keep extracting until none are left inside the scratch folder.
            while True:
                inner = glob.glob(os.path.join(EXTRACT_TMP, "**", "*.zip"), recursive=True)
                if not inner:
                    break
                for inner_zip in inner:
                    extract_zip(inner_zip, os.path.dirname(inner_zip))
                    os.remove(inner_zip)
            n_videos = sort_extracted_files(EXTRACT_TMP)
            with open(DONE_FILE, "a") as f:
                f.write(os.path.basename(zip_path) + "\n")
            ok += 1
            tqdm.write(f"  {os.path.basename(zip_path)}: {n_videos} videos")
        except (zipfile.BadZipFile, subprocess.CalledProcessError, OSError, RuntimeError) as e:
            failed += 1
            tqdm.write(f"  FAILED {os.path.basename(zip_path)}: {e}")
    shutil.rmtree(EXTRACT_TMP, ignore_errors=True)
    log.info(f"unzip: {ok} ok, {failed} failed")


# ----------------------------------------------------------------------------
# Step 2: load labels + transcripts
# ----------------------------------------------------------------------------
def find_pkl(folder, expected_name):
    """Pick the pkl to use from `folder`, or stop with a precise message if there is none.

    Preference: the official file name > any other pkl > the smoke-test DUMMY file.
    So a dummy / partial file can never silently win over the real one.
    """
    found = sorted(glob.glob(os.path.join(folder, "*.pkl")))
    if not found:
        sys.exit(f"\nMISSING FILE: {expected_name}\n"
                 f"  Put it in: {folder}/\n"
                 f"  (or put the archive that contains it in {config.ZIPS_DIR}/ and re-run)")

    def rank(path):
        name = os.path.basename(path)
        if name == expected_name:
            return 0
        return 2 if "dummy" in name.lower() else 1

    chosen = min(found, key=rank)
    if len(found) > 1:
        log.info(f"several pkl files in {folder}, using {os.path.basename(chosen)}")
    return chosen


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    keys = list(data.keys())
    log.info(f"{os.path.basename(path)}: {len(keys)} top-level keys, first 6 = {keys[:6]}")
    return data


def build_table():
    """One row per clip that has a video on disk AND labels for all 5 traits."""
    annotation_path = find_pkl(config.ANNOTATIONS_DIR, config.REAL_ANNOTATION_NAME)
    annotations = load_pkl(annotation_path)
    transcripts = load_pkl(find_pkl(config.TRANSCRIPTS_DIR, config.REAL_TRANSCRIPT_NAME))

    # Smoke-test dummy labels (random numbers)? Warn loudly and leave a marker file next to
    # splits.csv so that every later phase repeats the warning.
    is_fake = (config.DUMMY_KEY in annotations
               or "dummy" in os.path.basename(annotation_path).lower())
    if is_fake:
        utils.print_fake_labels_banner()
        with open(config.FAKE_LABELS_MARKER, "w") as f:
            f.write("data/splits.csv was built from RANDOM dummy labels "
                    f"({os.path.basename(annotation_path)}).\nResults are meaningless.\n")
    elif os.path.exists(config.FAKE_LABELS_MARKER):
        os.remove(config.FAKE_LABELS_MARKER)

    # Expected layout: {trait_name: {video_filename: score}} (+ "interview", which we ignore)
    missing_traits = [t for t in config.TRAITS if t not in annotations]
    if missing_traits:
        sys.exit(f"Annotation pkl has no key(s) {missing_traits}. Keys found: {list(annotations.keys())}")

    videos = sorted(f for f in os.listdir(config.VIDEOS_DIR) if f.lower().endswith(".mp4"))
    if not videos:
        sys.exit(f"\nMISSING FILES: no .mp4 videos in {config.VIDEOS_DIR}/\n"
                 f"  Put the ChaLearn training video zips (training80_XX.zip) in {config.ZIPS_DIR}/ and re-run.")

    labelled = [v for v in videos if all(v in annotations[t] for t in config.TRAITS)]
    log.info(f"videos on disk: {len(videos)}, with labels: {len(labelled)}")
    if not labelled:
        sys.exit("None of the videos on disk appear in the annotation pkl - "
                 "are the videos from the TRAINING set (training80_XX.zip)?")

    # Cap at MAX_CLIPS with a seeded shuffle so the subset is random but repeatable.
    rng = np.random.RandomState(config.SEED)
    rng.shuffle(labelled)
    labelled = sorted(labelled[:config.MAX_CLIPS])

    rows = []
    for fname in labelled:
        text = transcripts.get(fname, "")
        row: dict = {
            "video_id": fname[:-len(".mp4")],
            "youtube_id": fname.split(".")[0],          # clips of one YouTube video share this
            "text": text if isinstance(text, str) else "",
        }
        for t in config.TRAITS:
            row[t] = float(annotations[t][fname])
        rows.append(row)
    df = pd.DataFrame(rows)
    log.info(f"clips kept: {len(df)} (MAX_CLIPS={config.MAX_CLIPS}), "
             f"unique youtube_ids: {df.youtube_id.nunique()}, "
             f"without transcript: {(df.text.str.strip() == '').sum()}")
    return df


# ----------------------------------------------------------------------------
# Step 3 + 4: group split and train-median binarisation
# ----------------------------------------------------------------------------
def add_split(df):
    """70/15/15 split by youtube_id. A random split per clip would leak identity."""
    groups = df["youtube_id"].values
    # First cut: train vs (val + test)
    gss = GroupShuffleSplit(n_splits=1, train_size=config.TRAIN_FRAC, random_state=config.SEED)
    train_idx, rest_idx = next(gss.split(df, groups=groups))
    # Second cut: split the remainder into val and test
    val_share = config.VAL_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=config.SEED)
    try:
        val_sub, test_sub = next(gss2.split(rest_idx, groups=groups[rest_idx]))
    except ValueError:
        sys.exit(f"Too few different youtube_ids ({df.youtube_id.nunique()}) to build "
                 f"train/val/test splits - use more clips (increase MAX_CLIPS).")

    df["split"] = ""
    df.iloc[train_idx, df.columns.get_loc("split")] = "train"
    df.iloc[rest_idx[val_sub], df.columns.get_loc("split")] = "val"
    df.iloc[rest_idx[test_sub], df.columns.get_loc("split")] = "test"
    return df


def add_binary_labels(df):
    """<trait>_bin = 1 if score >= median, where the median comes from TRAIN only."""
    train = df[df.split == "train"]
    for t in config.TRAITS:
        median = train[t].median()
        df[f"{t}_bin"] = (df[t] >= median).astype(int)
        log.info(f"  {t:<18} train median = {median:.4f}")
    return df


def report_split_ratio(df):
    """Print the clip-level train/val/test share. We split by youtube_id, so the share of
    CLIPS can drift from 70/15/15 - warn when it drifts too far."""
    share = df["split"].value_counts(normalize=True) * 100
    share = {s: float(share.get(s, 0.0)) for s in ["train", "val", "test"]}
    log.info("clip-level split ratio train/val/test = "
             f"{share['train']:.1f} / {share['val']:.1f} / {share['test']:.1f} %")
    for name, (low, high) in config.SPLIT_RATIO_OK.items():
        if not low <= share[name] <= high:
            log.info(f"WARNING: {name} holds {share[name]:.1f}% of the clips, "
                     f"outside the expected {low}-{high}%")


# ----------------------------------------------------------------------------
# Acceptance check
# ----------------------------------------------------------------------------
def acceptance_check(df):
    problems = []

    # (a) no youtube_id in more than one split
    splits_per_group = df.groupby("youtube_id")["split"].nunique()
    leaked = int((splits_per_group > 1).sum())
    if leaked:
        problems.append(f"{leaked} youtube_id(s) appear in more than one split")

    # (b) each trait's positive rate in train is 45-55%.
    # With very few train clips an exact 45-55% can be impossible (2 of 3 = 66.7%), so being
    # at most ONE clip away from an even split is also accepted. On real data this changes nothing.
    train = df[df.split == "train"]
    for t in config.TRAITS:
        n_positive = train[f"{t}_bin"].sum()
        rate = n_positive / len(train) * 100
        if not (45 <= rate <= 55 or abs(n_positive - len(train) / 2) <= 1):
            problems.append(f"{t}: train positive rate {rate:.1f}% is outside 45-55%")

    # (c) no NaN in trait columns
    trait_cols = config.TRAITS + [f"{t}_bin" for t in config.TRAITS]
    n_nan = int(df[trait_cols].isna().sum().sum())
    if n_nan:
        problems.append(f"{n_nan} NaN value(s) in trait columns")

    if problems:
        for p in problems:
            log.info(f"ACCEPTANCE FAILED: {p}")
        sys.exit(1)
    log.info("ACCEPTANCE CHECK PASSED: no group leakage, positive rates 45-55%, no NaN")


def main():
    utils.set_seed()
    utils.ensure_dirs()

    unzip_all()
    df = build_table()
    df = add_split(df)
    df = add_binary_labels(df)

    columns = (["video_id", "youtube_id", "split", "text"] + config.TRAITS
               + [f"{t}_bin" for t in config.TRAITS])
    df = df[columns]
    df.to_csv(config.SPLITS_CSV, index=False)
    log.info(f"saved {config.SPLITS_CSV}")

    # Console report: clips per split and % positive per trait
    report = df.groupby("split").agg(clips=("video_id", "count"), youtube_ids=("youtube_id", "nunique"))
    for t in config.TRAITS:
        report[f"{t[:5]}_pos%"] = (df.groupby("split")[f"{t}_bin"].mean() * 100).round(1)
    print(report.loc[["train", "val", "test"]].to_string())
    report_split_ratio(df)

    acceptance_check(df)
    utils.warn_if_fake_labels()      # repeat at the very end so it cannot be missed


if __name__ == "__main__":
    main()
