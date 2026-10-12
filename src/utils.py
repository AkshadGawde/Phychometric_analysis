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
