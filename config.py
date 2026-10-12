"""Central configuration: every path, constant and flag used by the pipeline.

All scripts in src/ import this file, so changing a value here changes it everywhere.
"""
import os

# ----------------------------------------------------------------------------
# Reproducibility + task constants
# ----------------------------------------------------------------------------
SEED = 42
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
N_FRAMES = 8
IMG_SIZE = 224
SAMPLE_RATE = 16000
MAX_CLIPS = 2000          # cap for speed
USE_HEAVY_MODELS = True   # wav2vec2 / ViT emotion models; set False for a fast run

# Optional overrides from the shell, so a smoke test needs no code edit:
#   MAX_CLIPS=50 USE_HEAVY_MODELS=0 ./run_all.sh
if "MAX_CLIPS" in os.environ:
    MAX_CLIPS = int(os.environ["MAX_CLIPS"])
if "USE_HEAVY_MODELS" in os.environ:
    USE_HEAVY_MODELS = os.environ["USE_HEAVY_MODELS"].lower() in ("1", "true", "yes")

# ----------------------------------------------------------------------------
# Paths (all relative to the project root = the folder containing this file)
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ZIPS_DIR = os.path.join(RAW_DIR, "zips")                  # user puts downloaded zips here
VIDEOS_DIR = os.path.join(RAW_DIR, "videos")              # all .mp4 files, flat
ANNOTATIONS_DIR = os.path.join(RAW_DIR, "annotations")    # annotation_training.pkl
TRANSCRIPTS_DIR = os.path.join(RAW_DIR, "transcripts")    # transcription_training.pkl

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FRAMES_DIR = os.path.join(PROCESSED_DIR, "frames")
AUDIO_DIR = os.path.join(PROCESSED_DIR, "audio")
SPLITS_CSV = os.path.join(DATA_DIR, "splits.csv")
PREPROCESS_LOG_CSV = os.path.join(PROCESSED_DIR, "preprocess_log.csv")   # one row per finished clip
FAILED_CLIPS_CSV = os.path.join(PROCESSED_DIR, "failed_clips.csv")       # clips dropped in Phase 2

# Label / transcript file names. The DUMMY file holds RANDOM scores and exists only for smoke
# tests (src/00_make_smoke_data.py). Phase 1 always prefers the real file when it is present.
REAL_ANNOTATION_NAME = "annotation_training.pkl"
REAL_TRANSCRIPT_NAME = "transcription_training.pkl"
DUMMY_ANNOTATION_NAME = "DUMMY_FAKE_annotation_RANDOM_SCORES.pkl"
DUMMY_KEY = "__FAKE_LABELS__"                                  # extra key inside the dummy pkl
# Written by Phase 1 while splits.csv is built from dummy labels, deleted when real labels are used.
FAKE_LABELS_MARKER = os.path.join(DATA_DIR, "FAKE_LABELS_DO_NOT_REPORT.txt")

FEATURES_DIR = os.path.join(ROOT, "features")
IDS_CSV = os.path.join(FEATURES_DIR, "ids.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# ----------------------------------------------------------------------------
# Split ratios (grouped by youtube_id, see src/01_prepare_data.py)
# ----------------------------------------------------------------------------
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Phase 1 warns (on real data) when the clip-level share of a split is outside these % ranges.
SPLIT_RATIO_OK = {"train": (65, 75), "val": (10, 20), "test": (10, 20)}

# ----------------------------------------------------------------------------
# Phase 2 preprocessing settings
# ----------------------------------------------------------------------------
FACE_MARGIN = 0.20          # extra border around the detected face, as a fraction of its size
FACE_DETECT_SIDE = 640      # frames are shrunk to this longest side for face detection (speed)
JPG_QUALITY = 95

NOISE_QUIET_PERCENT = 10    # the quietest 10% of frames are taken as "background noise"
NOISE_GATE_STRENGTH = 1.5   # a frequency bin must exceed 1.5 x the noise level to pass fully
NOISE_GATE_FLOOR = 0.1      # gated bins are turned down to 10% (not to 0, which sounds artificial)
TRIM_TOP_DB = 25            # leading/trailing audio quieter than peak - 25 dB counts as silence
PEAK_LEVEL = 0.95           # peak normalisation target
MIN_AUDIO_SEC = 0.5         # clips with less cleaned audio than this are dropped

FILLERS = ["um", "uh", "er", "ah", "umm", "uhh", "uhm", "erm"]   # always removed as whole words
COMMA_FILLERS = ["like"]    # removed only when set off by punctuation: "it was, like, great"

N_SAMPLE_CLIPS = 2          # clips shown in results/sample_frames.png

# ----------------------------------------------------------------------------
# Which feature files make up each modality (used from Phase 6 onwards).
# Each name is a file features/<name>.npy.
# ----------------------------------------------------------------------------
MODALITIES = {
    "visual": ["visual_cnn", "visual_emotion"],   # + "visual_vit" optional
    "audio": ["audio_mfcc", "audio_emotion"],     # + "audio_w2v" optional
    "text": ["text_minilm", "text_emotion"],      # tfidf used as a separate baseline
}

# Feature sets that only exist when the heavy pretrained models were run.
HEAVY_FEATURES = {"visual_emotion", "visual_vit", "audio_emotion", "audio_w2v", "text_emotion"}
if not USE_HEAVY_MODELS:
    MODALITIES = {m: [f for f in feats if f not in HEAVY_FEATURES] for m, feats in MODALITIES.items()}
