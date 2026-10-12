# VIP Task T1 — Multimodal Psychometric Assessment

Predicts the **Big Five personality traits** (openness, conscientiousness, extraversion,
agreeableness, neuroticism) from short talking-head videos of the
**ChaLearn First Impressions V2** dataset, using face (visual), voice (audio) and words (text),
on their own and fused.

Research prototype about *apparent* personality. **Not a diagnostic tool.**

- What each phase **produced** (files, counts, figures): see [OUTPUTS.md](OUTPUTS.md)
- The full specification: see [ROADMAP.md](ROADMAP.md)

## Status

| Phase | Script | Status |
|---|---|---|
| Setup | `config.py`, `src/utils.py`, `requirements.txt`, `run_all.sh` | Done |
| Smoke helper | `src/00_make_smoke_data.py` | Done |
| 1. Data preparation | `src/01_prepare_data.py` | Done — run on smoke data |
| 2. Preprocessing | `src/02_preprocess.py` | Done — run on smoke data |
| 3. Visual features | `src/03_features_visual.py` | Not started |
| 4. Audio features | `src/04_features_audio.py` | Not started |
| 5. Text features | `src/05_features_text.py` | Not started |
| 6. Unimodal models | `src/06_unimodal.py` | Not started |
| 7. Fusion | `src/07_fusion.py` | Not started |
| 8. Ablation | `src/08_ablation.py` | Not started |
| 9. Analysis + summary | `src/09_analysis.py` | Not started |
| 10. Streamlit demo | `app/streamlit_app.py` | Not started |
| 11. One-command run | `run_all.sh` | Written; stops after Phase 2 until 3–9 exist |

So far everything has run only on a **50-clip smoke subset with FAKE (random) labels**.

## Quick start

```bash
brew install ffmpeg                    # system dependency (macOS)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_all.sh                           # all phases, settings from config.py
python src/02_preprocess.py            # or one phase on its own
```

Python 3.10+. Device is picked automatically: `cuda` → `mps` (Apple Silicon) → `cpu`.

Data goes here (download from
<https://chalearnlap.cvc.uab.cat/dataset/24/data/41/description/>, login required):

```
data/raw/
├── zips/          downloaded .zip files, untouched (nested zips are handled)
├── videos/        .mp4 clips, flat (filled automatically from the zips)
├── annotations/   annotation_training.pkl
└── transcripts/   transcription_training.pkl
```

Videos must keep their ChaLearn names (e.g. `J4GQm9j0JZ0.003.mp4`). If a zip is password
protected: `export CHALEARN_PW='...'`.

## Switching to full data

1. Put the ChaLearn files in place — either
   - the downloaded zips in `data/raw/zips/`, **or**
   - the videos in `data/raw/videos/`, `annotation_training.pkl` in `data/raw/annotations/`
     and `transcription_training.pkl` in `data/raw/transcripts/`.
2. Run:

   ```bash
   MAX_CLIPS=2000 USE_HEAVY_MODELS=1 ./run_all.sh
   ```

Nothing else is needed and no smoke file has to be deleted: the real pkl files are always
preferred over the dummy ones, `data/FAKE_LABELS_DO_NOT_REPORT.txt` is removed automatically,
and clips already preprocessed are reused.

---

# What is implemented, phase by phase

## Setup (roadmap sections 2–3)

- **Folder structure** from the roadmap: `src/`, `app/`, `data/raw/...`, `data/processed/...`,
  `features/`, `results/models/`.
- **`requirements.txt`** — the roadmap list, plus `joblib` (model saving), `datasets`
  (smoke helper only) and the pin `opencv-python<5` (OpenCV 5 removed the Haar face detector).
- **`config.py`** — the single place for every setting:
  - constants: `SEED = 42`, `TRAITS`, `N_FRAMES = 8`, `IMG_SIZE = 224`, `SAMPLE_RATE = 16000`,
    `MAX_CLIPS = 2000`, `USE_HEAVY_MODELS = True`;
  - `MAX_CLIPS` and `USE_HEAVY_MODELS` can be overridden from the shell, so smoke and full
    runs use the same code;
  - all paths, the 70/15/15 split ratio, the Phase 2 settings (face margin, noise gate,
    silence threshold, filler words);
  - `MODALITIES` — which feature files belong to visual / audio / text; the heavy-model
    features are dropped automatically when `USE_HEAVY_MODELS` is off.
- **`src/utils.py`** — helpers shared by all scripts:
  - `set_seed()`, `get_device()`, `get_logger()`, `ensure_dirs()`;
  - `load_splits()` / `save_splits()` and the per-clip paths `video_path()`, `frame_path()`,
    `audio_path()`;
  - `classification_metrics()` — accuracy, precision, recall, F1, AUC. If a split contains
    only one class, precision / recall / F1 / AUC are returned as **NaN with a warning**,
    never 0 or 1;
  - `regression_metrics()` — MAE and the ChaLearn score 1 − MAE;
  - `warn_if_fake_labels()` — prints the big `LABELS ARE FAKE` banner;
  - `setup_plots()` — one shared look for every figure.
- **`run_all.sh`** — runs Phase 1 → 9 in order and activates `.venv` itself.

## Smoke helper — `src/00_make_smoke_data.py`

Not part of the real pipeline and not called by `run_all.sh`. It lets the code be tested
before the official label file is available.

1. **Restores original file names.** Clips saved from the browser get generic names. Each
   local file is matched by SHA-256 to the file list of the Hugging Face mirror
   `yeray142/first-impressions-v2` and renamed. Every rename is logged so it can be undone;
   duplicates and unmatched files are left untouched.
2. **Fetches the real transcripts** of those clips by streaming the mirror with
   `Video(decode=False)` — only metadata is read, no video is downloaded.
3. **Writes DUMMY labels**: seeded random scores in [0, 1] for the five traits, in the same
   layout as the official file. It refuses to do this if the real label file exists.

## Phase 1 — Data preparation — `src/01_prepare_data.py`

1. **Unzip.** Extracts every zip in `data/raw/zips/`, including zips inside zips, and sorts
   the contents: `.mp4` → `videos/`, annotation pkl → `annotations/`, transcription pkl →
   `transcripts/`. Uses `CHALEARN_PW` for encrypted zips. Zips already extracted are skipped.
2. **Load labels and transcripts.** Prints the top-level keys of each pkl first. The official
   file names are always preferred over any other pkl, and the dummy file is used only as a
   last resort.
3. **Build the clip table.** Keeps clips that have a video on disk and all five labels, capped
   at `MAX_CLIPS` with a seeded shuffle. `youtube_id` = file name up to the first dot.
4. **Split 70/15/15 by `youtube_id`** (`GroupShuffleSplit`), not by clip — clips of the same
   person never end up in two splits, which would leak identity.
5. **Binary labels.** For each trait the median is computed on **train only**;
   `<trait>_bin = 1` if the score is at or above it.
6. **Checks and warnings.**
   - Acceptance check: no `youtube_id` in two splits, every trait 45–55% positive in train,
     no NaN. The script exits with an error if it fails.
   - Prints the clip-level split ratio; on real data it warns if outside 65–75 / 10–20 / 10–20 %.
   - If the labels are the dummy ones, prints the FAKE banner and writes
     `data/FAKE_LABELS_DO_NOT_REPORT.txt`; with real labels that file is deleted.

`splits.csv` is rebuilt on every run (it takes about a second and is deterministic).

## Phase 2 — Preprocessing — `src/02_preprocess.py`

Works only on the `video_id`s listed in `data/splits.csv`.

**Video → frames**
- 8 evenly spaced frames per clip (`np.linspace` over the frame count), read in one pass.
- Largest face found by the OpenCV Haar cascade, cropped with a 20% margin; centre crop if
  no face is found. Detection runs on a downsized grey copy for speed.
- Resized to 224×224 and saved as JPG. ImageNet normalisation is left to Phase 3.

**Speech → cleaned audio**
- `ffmpeg` extracts the sound track, `librosa` loads it at 16 kHz mono.
- **Noise removal: spectral gating.** The quietest 10% of frames give a noise level per
  frequency; bins not clearly above it are turned down. Chosen over pre-emphasis + high-pass
  because pre-emphasis would distort the energy and zero-crossing cues measured in Phase 4.
- **Segmentation:** leading and trailing silence trimmed (`top_db = 25`).
- **Normalisation:** peak-normalised to 0.95, saved as 16-bit wav.

**Text → `text_clean`**
- Lowercase; transcriber notes such as `[inaudible 00:00:07]` removed; punctuation and extra
  spaces stripped (apostrophes inside words are kept: `don't`); whitespace tokens.
- Fillers `um, uh, er, ah` (and spelling variants) removed as whole words.
- `like` is removed only when set off by punctuation ("it was, like, great"), so the verb in
  "I like pizza" survives.
- Empty result → `text_clean = ""` and `missing_text = 1`.

**Robustness**
- Resume: a clip with all its frames and its wav is skipped. If `splits.csv` gains new clips,
  only those are processed. `--force` redoes everything.
- A clip that fails (unreadable video, no audio, silent) is logged, written to
  `failed_clips.csv` and removed from `splits.csv`; one bad clip never stops the run.
- Acceptance check: every clip left in `splits.csv` has 8 frames and 1 wav.

**Report figures:** sample frames, raw vs cleaned waveform, transcript length histogram.

## Phases 3–11 — not implemented yet

| Phase | Planned |
|---|---|
| 3 | Frozen ResNet-50 frame features; face-expression probabilities and ViT embedding (heavy) |
| 4 | MFCC, pitch, energy, zero-crossing, speaking-rate features; wav2vec2 embedding and emotion (heavy) |
| 5 | TF-IDF (fit on train), MiniLM sentence embeddings, text emotion (heavy); `features/ids.csv` |
| 6 | One model per modality and trait (logistic regression, MLP, Ridge); TF-IDF baseline |
| 7 | Early, late and intermediate (PyTorch) fusion, compared with the best unimodal model |
| 8 | Ablation: remove one modality at a time, ΔF1 per trait |
| 9 | Cue analysis, confusion matrices, `results/summary.md` |
| 10 | Streamlit demo tab "Big Five (our model)" |
| 11 | Final `README.md` and `run_all.sh` check |
