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
SPLITS_CSV = os.path.join(DATA_DIR, "splits_final.csv")   # the split every phase uses
# One row per clip: statistics + the settings its frames / audio were made with (used to resume).
PREPROCESS_LOG_CSV = os.path.join(PROCESSED_DIR, "preprocess_log.csv")
FAILED_CLIPS_CSV = os.path.join(PROCESSED_DIR, "failed_clips.csv")       # clips dropped in Phase 2

# Label / transcript file names. The DUMMY file holds RANDOM scores and exists only for smoke
# tests (src/00_make_smoke_data.py). Phase 1 always prefers the real files when they are present.
# ChaLearn ships one file per official part; every one that is present is used (we make our
# own train/val/test split by youtube_id, so the parts are pooled).
REAL_ANNOTATION_NAMES = ["annotation_training.pkl", "annotation_validation.pkl", "annotation_test.pkl"]
REAL_TRANSCRIPT_NAMES = ["transcription_training.pkl", "transcription_validation.pkl",
                         "transcription_test.pkl"]
DUMMY_ANNOTATION_NAME = "DUMMY_FAKE_annotation_RANDOM_SCORES.pkl"
DUMMY_KEY = "__FAKE_LABELS__"                                  # extra key inside the dummy pkl
# Written by Phase 1 while splits.csv is built from dummy labels, deleted when real labels are used.
FAKE_LABELS_MARKER = os.path.join(DATA_DIR, "FAKE_LABELS_DO_NOT_REPORT.txt")

FEATURES_DIR = os.path.join(ROOT, "features")
IDS_CSV = os.path.join(FEATURES_DIR, "ids.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# src/00_fetch_more_clips.py (optional download of more clips from the Hugging Face mirror)
FETCH_TARGET_CLIPS = 1500        # labelled clips wanted on disk in total
PROCESSED_MB_PER_CLIP = 0.8      # rough size of one clip's frames + wav + features
FETCH_FREE_SPACE_FACTOR = 2      # require this many times the estimated size to be free

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
MIN_FACE_FRAC = 0.10        # face boxes narrower than 10% of the picture width are ignored
FACE_OUTLIER_FRAC = 0.25    # a detection further than 25% of the picture width from the clip's
                            # median face box is replaced by that median box
MIN_FACES_FOR_MEDIAN = 3    # the median box is only trusted if >= 3 frames have a detection
BORDER_BLACK_LEVEL = 16     # rows/columns darker than this (0-255) in all frames = black bars
MIN_CONTENT_FRAC = 0.20     # if less than this would remain after removing bars, keep the full frame
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
# Phase 3-5 feature extraction settings
# ----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]     # normalisation the ResNet-50 weights were trained with
IMAGENET_STD = [0.229, 0.224, 0.225]
FACE_EMOTION_MODEL = "trpakov/vit-face-expression"
VISUAL_BATCH_CLIPS = 8                    # clips per batch (= 8 x N_FRAMES images)
FEATURE_CHECKPOINT_CLIPS = 200            # partial results are saved every 200 new clips
VISUAL_EMOTION_LABELS_JSON = os.path.join(FEATURES_DIR, "visual_emotion_labels.json")

# Phase 4 - audio. All hand-crafted features use the same analysis frames.
AUDIO_FRAME_LENGTH = 1024                 # 64 ms window at 16 kHz
AUDIO_HOP_LENGTH = 256                    # a new frame every 16 ms
N_MFCC = 40
PITCH_FMIN, PITCH_FMAX = 60, 400          # Hz, covers low male to high female speaking pitch
MIN_FEATURE_SAMPLES = 2 * AUDIO_FRAME_LENGTH   # shorter audio is padded with silence (and logged)
SILENCE_PEAK = 1e-4                       # audio whose loudest sample is below this is "near-silent"
AUDIO_BATCH_CLIPS = 8
SPEECH_EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
W2V_MAX_SEC = 30                          # longer audio is cut to this length for wav2vec2 (memory)
AUDIO_MFCC_COLS_JSON = os.path.join(FEATURES_DIR, "audio_mfcc_cols.json")
AUDIO_EMOTION_LABELS_JSON = os.path.join(FEATURES_DIR, "audio_emotion_labels.json")
AUDIO_PROBLEMS_CSV = os.path.join(FEATURES_DIR, "audio_problems.csv")   # short / near-silent clips

# Phase 5 - text
TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAMS = (1, 2)                     # single words and word pairs
TFIDF_MIN_DF = 2                          # a term must occur in at least 2 train transcripts
TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
TEXT_BATCH_CLIPS = 32
TEXT_MAX_TOKENS = 512                     # longer transcripts are cut for the emotion model
TFIDF_VOCAB_JSON = os.path.join(FEATURES_DIR, "tfidf_vocab.json")
TEXT_EMOTION_LABELS_JSON = os.path.join(FEATURES_DIR, "text_emotion_labels.json")
TEXT_HASHES_JSON = os.path.join(FEATURES_DIR, "text_hashes.json")   # which text each cached row was made from
TFIDF_VECTORIZER_FILE = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

# Expected number of columns of each feature file (checked after extraction).
# (text_tfidf is not listed: its width is the size of the vocabulary found in the train split.)
FEATURE_DIMS = {
    "visual_cnn": 2048, "visual_emotion": 7, "visual_vit": 768,
    "audio_mfcc": 2 * N_MFCC + 7,         # 40 MFCC means + 40 stds + 7 prosody values = 87
    "audio_w2v": 1024, "audio_emotion": 8,
    "text_minilm": 384, "text_emotion": 7,
}

# ----------------------------------------------------------------------------
# Which feature files make up each modality (used from Phase 6 onwards).
# Each name is a file features/<name>.npy.
# ----------------------------------------------------------------------------
MODALITIES = {
    "visual": ["visual_cnn", "visual_emotion"],   # + "visual_vit" optional
    "audio": ["audio_mfcc", "audio_emotion"],     # + "audio_w2v" optional
    "text": ["text_minilm", "text_emotion"],      # tfidf used as a separate baseline
}

# ----------------------------------------------------------------------------
# Phase 6-8 modelling settings
# ----------------------------------------------------------------------------
LOGREG_C_GRID = [0.01, 0.1, 1, 10]        # tuned on the validation split
RIDGE_ALPHA_GRID = [0.1, 1, 10, 100]      # tuned on the validation split
PCA_MIN_DIMS = 512                        # PCA is used only when a model has more inputs than this
PCA_VARIANCE = 0.95                       # keep enough components for 95% of the variance
MLP_HIDDEN = (128,)
# Probability calibration of every single-modality / early-fusion classifier (Platt scaling on
# 5-fold out-of-fold train predictions, folds grouped by YouTube video). Turned on after the first
# report: two face models gave P < 0.02 or > 0.98 for most held-out clips and dominated late fusion.
CALIBRATE = True
CALIBRATION_FOLDS = 5
CALIBRATION_MIN_SLOPE = 1e-6      # only for a slope <= 0 (useless out of fold): keeps the ranking, flat probabilities

# Intermediate fusion network (Phase 7)
FUSION_BRANCH_DIMS = (128, 64)            # each modality: d -> 128 -> 64
FUSION_HEAD_DIM = 64                      # concat -> 64 -> 5 traits
FUSION_DROPOUT = 0.3
FUSION_LR = 1e-3
FUSION_WEIGHT_DECAY = 1e-4
FUSION_BATCH_SIZE = 64
FUSION_MAX_EPOCHS = 100
FUSION_PATIENCE = 10                      # stop after 10 epochs without a better validation loss

UNIMODAL_CSV = os.path.join(RESULTS_DIR, "unimodal.csv")
FUSION_CSV = os.path.join(RESULTS_DIR, "fusion.csv")
ABLATION_CSV = os.path.join(RESULTS_DIR, "ablation.csv")
BEST_FUSION_JSON = os.path.join(RESULTS_DIR, "best_fusion.json")
FUSION_MODEL_FILE = os.path.join(MODELS_DIR, "intermediate_fusion.pt")

# ----------------------------------------------------------------------------
# Live app (src/10_export_bundle.py, src/predict.py, app/server.py) - see ROADMAP_WEB.md
# ----------------------------------------------------------------------------
BUNDLE_JSON = os.path.join(MODELS_DIR, "bundle.json")     # everything the app needs besides the models
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small.en")   # faster-whisper; "base.en" is faster
WINDOW_SEC = 15                   # long inputs are cut into windows as long as the training clips
WINDOW_MIN_TOTAL_SEC = 20         # ... but only inputs longer than this
EMOTION_STEP_SEC = 5              # the speech-emotion timeline shown in the app (display only)
APP_MAX_UPLOAD_MB = 25
APP_MAX_INPUT_SEC = 180
REPLAYS_DIR = os.path.join(ROOT, "app", "replays")      # recorded event streams for the demo mode

# ----------------------------------------------------------------------------
# Diagnostics (src/08b_diagnostics.py) and the final report (src/09_analysis.py)
# ----------------------------------------------------------------------------
DIAGNOSTICS_DIR = os.path.join(RESULTS_DIR, "diagnostics")
BOOTSTRAP_SAMPLES = 1000                  # resamples of the test set for the 95% confidence intervals
RANDOM_BASELINE_REPEATS = 100             # the stratified-random baseline is averaged over 100 draws
OVERFIT_GAP = 0.15                        # flag a model when train F1 - test F1 is larger than this
SHUFFLE_REPEATS = 3                       # label-shuffle test: retrain this many times on shuffled labels
SHUFFLE_AUC_TOLERANCE = 0.10              # shuffled-label AUC must stay within 0.5 +/- this
LEARNING_CURVE_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
LEARNING_CURVE_REPEATS = 3                # random train subsets per fraction (below 100%)
VISUAL_WEAK_F1 = 0.55                     # below this, the visual feature sets are compared one by one
# Reference point from the literature (not our result): the best published First Impressions
# systems reach about 0.91 mean 1-MAE (ChaLearn LAP 2016 challenge, Ponce-Lopez et al. 2016).
PUBLISHED_TOP_ONE_MINUS_MAE = 0.91
SUMMARY_MD = os.path.join(RESULTS_DIR, "summary.md")
REPORT_MD = os.path.join(RESULTS_DIR, "REPORT.md")

# Feature sets that only exist when the heavy pretrained models were run.
HEAVY_FEATURES = {"visual_emotion", "visual_vit", "audio_emotion", "audio_w2v", "text_emotion"}
if not USE_HEAVY_MODELS:
    MODALITIES = {m: [f for f in feats if f not in HEAVY_FEATURES] for m, feats in MODALITIES.items()}
