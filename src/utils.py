"""Shared helpers: device selection, seeding, logging, metrics."""
import logging
import os
import random
import sys

import numpy as np

# Make `import config` work no matter where a script is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402


def get_logger(name):
    """Console logger with a short, uniform format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed=config.SEED):
    """Seed python, numpy and (if installed) torch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Pick the fastest available device: cuda -> mps (Apple Silicon) -> cpu."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs():
    """Create every output folder the pipeline writes to."""
    for d in [config.ZIPS_DIR, config.VIDEOS_DIR, config.ANNOTATIONS_DIR, config.TRANSCRIPTS_DIR,
              config.FRAMES_DIR, config.AUDIO_DIR, config.FEATURES_DIR, config.RESULTS_DIR,
              config.MODELS_DIR]:
        os.makedirs(d, exist_ok=True)


def print_fake_labels_banner():
    """Loud warning used whenever the trait labels are the random smoke-test dummies."""
    line = "!" * 78
    print("\n".join([
        "", line, line,
        "!!" + "WARNING: THE BIG FIVE LABELS ARE FAKE".center(74) + "!!",
        "!!" + "(random numbers from src/00_make_smoke_data.py - smoke test only)".center(74) + "!!",
        "!!" + "Every score, split statistic and metric below is MEANINGLESS.".center(74) + "!!",
        "!!" + "Do NOT report these results. Add the real annotation_training.pkl".center(74) + "!!",
        "!!" + "to data/raw/annotations/ and re-run from src/01_prepare_data.py.".center(74) + "!!",
        line, line, "",
    ]), flush=True)


def labels_are_fake():
    """True while data/splits.csv was built from the dummy annotation file."""
    return os.path.exists(config.FAKE_LABELS_MARKER)


def warn_if_fake_labels():
    """Call at the start of every script that uses the labels."""
    if labels_are_fake():
        print_fake_labels_banner()


def load_splits():
    """Read data/splits.csv. Empty transcripts stay "" instead of becoming NaN."""
    import pandas as pd
    if not os.path.exists(config.SPLITS_CSV):
        sys.exit(f"Missing {config.SPLITS_CSV} - run `python src/01_prepare_data.py` first.")
    df = pd.read_csv(config.SPLITS_CSV, dtype={"video_id": str, "youtube_id": str})
    for col in ("text", "text_clean"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def save_splits(df):
    df.to_csv(config.SPLITS_CSV, index=False)


# ----------------------------------------------------------------------------
# Per-clip file locations (always derived from a video_id in splits.csv)
# ----------------------------------------------------------------------------
def video_path(video_id):
    return os.path.join(config.VIDEOS_DIR, f"{video_id}.mp4")


def frame_path(video_id, k):
    return os.path.join(config.FRAMES_DIR, f"{video_id}_{k}.jpg")


def audio_path(video_id):
    return os.path.join(config.AUDIO_DIR, f"{video_id}.wav")


# ----------------------------------------------------------------------------
# Feature files (Phases 3-5)
#
# features/ids.csv        the video_id order that EVERY feature file must follow
# features/<name>.npy     one row per clip
# features/<name>.ids.csv the video_id of each row of <name>.npy. It lets us (a) prove that a
#                         file really matches ids.csv, row by row, and (b) reuse rows from an
#                         earlier run, so only new clips are computed.
# ----------------------------------------------------------------------------
def feature_path(name):
    return os.path.join(config.FEATURES_DIR, f"{name}.npy")


def feature_ids_path(name):
    return os.path.join(config.FEATURES_DIR, f"{name}.ids.csv")


def write_ids(ids, path=None):
    import pandas as pd
    pd.DataFrame({"video_id": list(ids)}).to_csv(path or config.IDS_CSV, index=False)


def read_ids(path=None):
    import pandas as pd
    return pd.read_csv(path or config.IDS_CSV, dtype=str, keep_default_na=False)["video_id"].tolist()


def save_features(name, ids, matrix):
    """Save features/<name>.npy with its row ids. NaN/inf are replaced by 0 and counted."""
    matrix = np.asarray(matrix, dtype=np.float32)
    bad = ~np.isfinite(matrix)
    if bad.any():
        get_logger("features").info(f"WARNING: {name}: {int(bad.sum())} NaN/inf value(s) replaced by 0")
        matrix = np.where(bad, 0.0, matrix).astype(np.float32)
    write_ids(ids, feature_ids_path(name))
    np.save(feature_path(name), matrix)
    return int(bad.sum())


def load_feature_cache(name, source_mtime=None):
    """Rows computed in an earlier run, as {video_id: vector} ({} if there are none).
    source_mtime(video_id) = when the clip's input (frames / wav) was last written; a row is
    reused only if the feature file is newer, so re-made inputs are never paired with old features."""
    if not (os.path.exists(feature_path(name)) and os.path.exists(feature_ids_path(name))):
        return {}
    matrix, ids = np.load(feature_path(name)), read_ids(feature_ids_path(name))
    if len(matrix) != len(ids):
        return {}
    saved_at = os.path.getmtime(feature_path(name))
    return {v: matrix[i] for i, v in enumerate(ids)
            if source_mtime is None or source_mtime(v) <= saved_at}


def cached_extract(names, ids, compute_batch, source_mtime, batch_size, desc):
    """Fill features/<name>.npy for every name in `names` (all produced by the same model).

    compute_batch(batch_ids) -> {name: array of shape (len(batch_ids), dim)}
    Clips that already have an up-to-date row in all the files are skipped. Partial results are
    saved every FEATURE_CHECKPOINT_CLIPS clips, so an interrupted run continues where it stopped.
    Returns (number of clips computed now, seconds spent computing them, NaN/inf values replaced).
    """
    import time
    from tqdm import tqdm
    log = get_logger("features")
    cache = {name: load_feature_cache(name, source_mtime) for name in names}
    todo = [v for v in ids if not all(v in cache[name] for name in names)]
    log.info(f"{desc}: {len(ids)} clips, {len(ids) - len(todo)} already done, {len(todo)} to compute")

    def save(final):
        replaced = 0
        for name in names:
            rows = [v for v in ids if v in cache[name]]
            if rows and (final or len(rows) < len(ids)):
                replaced += save_features(name, rows, np.stack([cache[name][v] for v in rows]))
        return replaced

    start, since_checkpoint = time.time(), 0
    for i in tqdm(range(0, len(todo), batch_size), desc=desc, unit="batch"):
        batch = todo[i:i + batch_size]
        out = compute_batch(batch)
        for name in names:
            for video_id, row in zip(batch, np.asarray(out[name])):
                cache[name][video_id] = row
        since_checkpoint += len(batch)
        if since_checkpoint >= config.FEATURE_CHECKPOINT_CLIPS:
            save(final=False)
            since_checkpoint = 0
    seconds = time.time() - start
    replaced = save(final=True)
    return len(todo), seconds, replaced


def check_features(names):
    """Acceptance check shared by Phases 3-5. Each features/<name>.npy must have exactly the
    rows of ids.csv in the same order, the expected number of columns and no NaN/inf.
    Returns {name: shape}; stops the script if a file does not match."""
    ids = read_ids()
    shapes = {}
    for name in names:
        if not os.path.exists(feature_path(name)):
            sys.exit(f"ACCEPTANCE FAILED: {feature_path(name)} is missing")
        matrix = np.load(feature_path(name))
        if read_ids(feature_ids_path(name)) != ids or matrix.shape[0] != len(ids):
            sys.exit(f"ACCEPTANCE FAILED: rows of {name}.npy do not match features/ids.csv")
        expected = config.FEATURE_DIMS.get(name)
        if matrix.ndim != 2 or (expected is not None and matrix.shape[1] != expected):
            sys.exit(f"ACCEPTANCE FAILED: {name}.npy has shape {matrix.shape}, "
                     f"expected ({len(ids)}, {expected})")
        if not np.isfinite(matrix).all():
            sys.exit(f"ACCEPTANCE FAILED: {name}.npy contains NaN/inf")
        shapes[name] = matrix.shape
    return shapes


def stale_feature_files(skip=()):
    """Names of feature files in features/ whose rows do NOT match the current ids.csv
    (e.g. made by another script before splits.csv changed). Their script must be re-run."""
    ids = read_ids()
    stale = []
    for fname in sorted(os.listdir(config.FEATURES_DIR)):
        name = fname[:-len(".npy")]
        if not fname.endswith(".npy") or name in skip:
            continue
        ids_file = feature_ids_path(name)
        if not os.path.exists(ids_file) or read_ids(ids_file) != ids:
            stale.append(name)
    return stale


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_prob=None, name=""):
    """Binary metrics for one trait: acc, precision, recall, f1, auc.

    If the true labels contain only ONE class (happens in tiny smoke subsets), precision,
    recall, F1 and AUC are undefined. They are reported as NaN with a warning - never as a
    misleading 0 or 1. Accuracy is still valid and is kept.
    """
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y_true = np.asarray(y_true)
    out = {"acc": accuracy_score(y_true, y_pred),
           "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan}
    if len(np.unique(y_true)) < 2:
        get_logger("metrics").info(
            f"WARNING: only one class in the true labels{' of ' + name if name else ''} "
            f"-> precision/recall/F1/AUC = NaN")
        return out
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    if y_prob is not None:
        out["auc"] = roc_auc_score(y_true, y_prob)
    return out


def regression_metrics(y_true, y_pred):
    """MAE on raw trait scores and the ChaLearn challenge score (1 - MAE)."""
    mae = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    return {"mae": mae, "one_minus_mae": 1.0 - mae}


# ----------------------------------------------------------------------------
# Plot style shared by every figure in results/
# ----------------------------------------------------------------------------
PLOT = {
    "surface": "#fcfcfb",      # figure background
    "ink": "#0b0b0b",          # titles
    "ink2": "#52514e",         # axis labels, notes
    "muted": "#898781",        # tick labels
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    # Series colours, always used in this order (colour-blind-safe as neighbours).
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
}


def setup_plots():
    """Apply one quiet, consistent style and return pyplot (no window is opened)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "figure.facecolor": PLOT["surface"], "axes.facecolor": PLOT["surface"],
        "savefig.facecolor": PLOT["surface"], "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10,
        "text.color": PLOT["ink"], "axes.labelcolor": PLOT["ink2"],
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": PLOT["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": PLOT["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": PLOT["muted"], "ytick.color": PLOT["muted"],
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2, "legend.frameon": False,
        "axes.prop_cycle": cycler(color=PLOT["series"]),
    })
    return plt
