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
| Setup | `config.py`, `src/utils.py`, `src/modeling.py`, `requirements.txt`, `run_all.sh` | Done |
| Helpers | `src/00_make_smoke_data.py`, `src/00_fetch_more_clips.py` | Done |
| 1. Data preparation | `src/01_prepare_data.py` | Done |
| 2. Preprocessing | `src/02_preprocess.py` | Done |
| 3. Visual features | `src/03_features_visual.py` | Done |
| 4. Audio features | `src/04_features_audio.py` | Done |
| 5. Text features | `src/05_features_text.py` | Done |
| 6. Unimodal models | `src/06_unimodal.py` | Done |
| 7. Fusion | `src/07_fusion.py` | Done |
| 8. Ablation | `src/08_ablation.py` | Done |
| 8b. Diagnostics | `src/08b_diagnostics.py` | Done |
| 9. Analysis + reports | `src/09_analysis.py` | Done |
| 10. Live app (replaces the Streamlit demo) | `src/10_export_bundle.py`, `src/predict.py`, `app/server.py`, `web/` | Done — see [ROADMAP_WEB.md](ROADMAP_WEB.md) |
| 11. One-command run | `run_all.sh` | Done (runs 01 → 09 including 08b, then the bundle export) |

**Final run:** 1,500 real-labelled clips, heavy models on. Every phase reads and writes
`data/splits_final.csv`. The plain-English results are in
[results/REPORT.md](results/REPORT.md); every output file is listed in [OUTPUTS.md](OUTPUTS.md).

## Live pipeline visualizer

A web app that runs the trained pipeline on one new input (your voice, a 15 s webcam clip or
an uploaded file) and shows every stage working on it: voice, words, face, fusion, result.
Plan and acceptance checks: [ROADMAP_WEB.md](ROADMAP_WEB.md).

```bash
make install          # once: Python + web dependencies (+ headless Chromium for the e2e test)
./dev.sh              # API on :8000 + web on :5173 -> open http://localhost:5173
```

- One file from the command line: `python src/predict.py voice.mp3` (any audio/video format;
  `--json out.json` for every stage, `--record NAME` to save a demo replay).
- `make demo` serves the built app from the API alone on http://localhost:8000.
- `make test` runs the parity, input and fusion-maths tests; `make test-browser` the headless
  end-to-end test (needs `./dev.sh` running).
- First start downloads Whisper `small.en` (~480 MB) and warms the models up (~1 min, ~3 GB RAM).
- Deploying (Vercel for the web app, Render for the API): [DEPLOY.md](DEPLOY.md).

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
├── annotations/   annotation_training.pkl  (+ annotation_validation.pkl, annotation_test.pkl if you have them)
└── transcripts/   transcription_training.pkl  (+ the validation / test files if you have them)
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

**No videos, only the annotation files?** `python src/00_fetch_more_clips.py --target 1500`
downloads labelled clips and all transcripts from a public Hugging Face mirror of the dataset
(see below), then run the command in step 2.

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
  - all paths, the 70/15/15 split ratio, the Phase 2 settings (face margin, minimum face
    size, black-bar level, noise gate, silence threshold, filler words) and the feature
    settings (model names, batch size, expected feature dimensions);
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
  - feature-file helpers used by Phases 3–5: `write_ids()` / `read_ids()`,
    `cached_extract()` (computes only clips that have no up-to-date row yet and saves partial
    results during long runs), `save_features()` (replaces NaN/inf by 0 and counts them),
    `check_features()` (the shared acceptance check);
  - `load_features()` / `require_features_match_ids()` for the modelling phases (6–8): a
    **hard stop** if `ids.csv` does not list the clips of `splits.csv`, or if any feature file
    in `features/` does not have exactly the rows of `ids.csv` in the same order — otherwise
    one clip's features could be paired with another clip's labels;
  - `load_pretrained()` — loads a Hugging Face model and **stops** if any of its weights was
    not found in the checkpoint (transformers would otherwise fill them with random numbers);
  - `free_memory()` — releases a model after use, so only one heavy model is in memory at a
    time (the development machine has 8 GB of RAM);
  - `warn_if_fake_labels()` — prints the big `LABELS ARE FAKE` banner;
  - `setup_plots()` — one shared look for every figure.
- **`src/modeling.py`** — the model code shared by Phases 6–8: loading the aligned data,
  the scaler → PCA → classifier pipeline, model choice on validation, the three fusion types
  and the fusion network. Keeping it in one place guarantees that Phase 7's late fusion and
  Phase 8's ablation use exactly the same single-modality models as Phase 6.
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

## Download helper — `src/00_fetch_more_clips.py`

Optional. The official site needs a login for every download; the public mirror
`yeray142/first-impressions-v2` has the same 10,000 clips and their transcripts, but not the
trait labels. So the official annotation files must already be in `data/raw/annotations/`.

1. **Transcripts.** Reads the mirror's three small metadata files and saves the transcript of
   every clip to `data/raw/transcripts/transcription_hf_mirror.pkl`, in the official layout.
   If the official `transcription_*.pkl` files are present, Phase 1 uses those instead.
2. **Clips.** Downloads more `.mp4` files until `--target` labelled clips are on disk. Only
   clips that have labels and a non-empty transcript are chosen, in a fixed random order
   (seed 42) across the mirror's train / validation / test folders.
3. **Disk check.** Before downloading it prints the free space and the estimated size, and
   stops if less than twice the estimate is free.
4. **Resumable.** A clip only appears in `data/raw/videos/` once it is complete. Clips on
   disk are never downloaded again, and because the order is fixed a re-run continues with
   exactly the missing ones.

## Phase 1 — Data preparation — `src/01_prepare_data.py`

1. **Unzip.** Extracts every zip in `data/raw/zips/`, including zips inside zips, and sorts
   the contents: `.mp4` → `videos/`, annotation pkl → `annotations/`, transcription pkl →
   `transcripts/`. Uses `CHALEARN_PW` for encrypted zips. Zips already extracted are skipped.
2. **Load labels and transcripts.** Prints the top-level keys of each pkl first.
   - ChaLearn ships separate annotation files for its own training, validation and test parts.
     **All official files that are present are merged**, because step 4 makes our own split.
     If a clip appears in two files with different scores, the first file wins and a warning
     is printed.
   - Official files are always preferred over any other pkl; the dummy file is used only when
     nothing else is there.
3. **Build the clip table.** Keeps clips that have a video on disk and all five labels, capped
   at `MAX_CLIPS` with a seeded shuffle. `youtube_id` = file name up to the first dot.
4. **Split 70/15/15 by `youtube_id`** (`GroupShuffleSplit`), not by clip — clips of the same
   person never end up in two splits, which would leak identity.
5. **Binary labels.** For each trait the median is computed on **train only**;
   `<trait>_bin = 1` if the score is at or above it.
6. **Checks and warnings.**
   - Acceptance check: no `youtube_id` in two splits, every trait 45–55% positive in train,
     no NaN. The script exits with an error if it fails. With very few train clips an exact
     45–55% can be impossible (2 of 3 = 66.7%), so being at most one clip away from an even
     split is also accepted; on real data this changes nothing.
   - Prints the clip-level split ratio and warns if it is outside 65–75 / 10–20 / 10–20 %
     (also on smoke data).
   - If the labels are the dummy ones, prints the FAKE banner and writes
     `data/FAKE_LABELS_DO_NOT_REPORT.txt`; with real labels that file is deleted and the
     script says so.

`splits.csv` is rebuilt on every run (it takes about a second and is deterministic).

## Phase 2 — Preprocessing — `src/02_preprocess.py`

Works only on the `video_id`s listed in `data/splits.csv`.

**Video → frames**
- 8 evenly spaced frames per clip (`np.linspace` over the frame count), read in one pass.
- **Black bars removed first.** Rows and columns that are almost black in all 8 frames of a
  clip (letterbox / pillarbox bars) are cut away before anything else.
- Largest face found by the OpenCV Haar cascade, cropped with a 20% margin. Face boxes
  narrower than 10% of the picture width are ignored. If no face is left, the centre of the
  picture is used. Detection runs on a downsized grey copy for speed.
- **Outlier detections replaced.** A talking-head clip shows one person who stays in roughly
  the same place, so the median face box of the clip is computed. A detection whose centre is
  more than 25% of the picture width away from it is treated as a false detection and the
  median box is used for that frame instead. The median is only trusted when at least 3
  frames have a detection.
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
- `preprocess_log.csv` remembers which settings each clip's frames and audio were made with.
  If a frame setting (or an audio setting) changes, the frames (or audio) of every clip are
  made again automatically, so old and new outputs are never mixed.
- A file is only **rewritten when its content really differs**. Unchanged files keep their
  modification time, so the later phases recompute features only for the clips that changed.
- No extra noise-reduction library is used: spectral gating is written with `librosa` and
  `numpy`, both already in `requirements.txt`.
- A clip that fails (unreadable video, no audio, silent) is logged, written to
  `failed_clips.csv` and removed from `splits.csv`; one bad clip never stops the run.
- Acceptance check: every clip left in `splits.csv` has 8 frames and 1 wav.

**Report figures:** sample frames, raw vs cleaned waveform, transcript length histogram.

## Phase 3 — Visual features — `src/03_features_visual.py`

Both networks are **frozen feature extractors**: `model.eval()` and `torch.no_grad()`, nothing
is trained here. The device is chosen automatically (`mps` on Apple Silicon). A clip's
feature is the **mean over its 8 frames**.

1. **CNN features (always).** torchvision ResNet-50 with `IMAGENET1K_V2` weights. The final
   `fc` layer (the 1000-class ImageNet classifier) is replaced by `Identity`, so the network
   returns the 2048 numbers that describe the image just before classification. Frames are
   normalised with the ImageNet mean/std. → `visual_cnn` (N, 2048).
2. **Face-expression features (only if `USE_HEAVY_MODELS`).** Hugging Face ViT
   `trpakov/vit-face-expression`, with its own image processor.
   - Softmax over **all** 7 expression classes, taken straight from the model's logits (no
     top-k cut), in the model's fixed class order → `visual_emotion` (N, 7). The class order
     is saved to `visual_emotion_labels.json`.
   - CLS embedding = `last_hidden_state[:, 0]` → `visual_vit` (N, 768).
3. **Row order.** Writes `features/ids.csv` (the `video_id` order of `splits.csv`). Every
   feature file also has a `<name>.ids.csv` listing the clip of each row.
4. **Acceptance check.** Each `.npy` must have exactly the rows of `ids.csv` in the same
   order, the expected number of columns, and no NaN/inf.

**Robustness**
- Resume: rows computed earlier are reused; only new clips go through the networks.
- If Phase 2 re-made a clip's frames, that clip's features are recomputed (the feature file
  must be newer than the frames).
- Feature files made by another run that no longer match `ids.csv` are listed in a NOTE here;
  the modelling phases (6–8) refuse to start while such a file exists.

## Phase 4 — Audio features — `src/04_features_audio.py`

Works on the cleaned 16 kHz wav files of Phase 2.

**1. Hand-crafted features (always)** → `audio_mfcc` (N, 87)

Each clip is cut into short overlapping frames (64 ms long, one every 16 ms). Every feature is
measured per frame and then summarised over the clip:

| Columns | Count | What it captures |
|---|---|---|
| `mfcc1..40_mean`, `mfcc1..40_std` | 80 | Shape of the sound spectrum = voice timbre |
| `pitch_mean`, `pitch_std` | 2 | How high the voice is and how much it moves (intonation) |
| `voiced_fraction` | 1 | Share of frames in which the vocal cords vibrate |
| `rms_mean`, `rms_std` | 2 | Loudness and how much it varies |
| `zcr_mean` | 1 | Zero-crossing rate: hissy / noisy sounds vs voiced sounds |
| `nonsilent_fraction` | 1 | Speaking-rate proxy: share of frames that are not a pause |

- **Pitch uses `librosa.pyin`** (60–400 Hz), which also decides per frame whether it is voiced.
  Unvoiced frames are NaN, so the statistics are NaN-safe (`nanmean` / `nanstd`) and are set
  to 0 when a clip has no voiced frame. Plain `yin` returns a value for every frame, including
  pauses, which inflates the pitch mean and spread.
- The column names are saved to `audio_mfcc_cols.json` for the cue analysis in Phase 9.

**2. Speech model (only if `USE_HEAVY_MODELS`)** — Hugging Face
`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`, frozen (`eval()` + `no_grad()`).
The waveform is normalised by the model's own feature extractor at 16 kHz.

- Mean of the last hidden state over time → `audio_w2v` (N, 1024).
- Softmax over **all** 8 emotion classes, from the logits, in the model's fixed class order →
  `audio_emotion` (N, 8). The class order is saved to `audio_emotion_labels.json`.
- **The emotion head is rebuilt by hand.** The head stored in this checkpoint
  (`classifier.dense`, `classifier.output`) is not the layout the standard transformers
  classifier expects, so the standard loader silently swaps in a **random** head and the
  emotion probabilities become noise. The script loads the encoder normally and rebuilds the
  checkpoint's own head: mean-pool → dense (1024→1024) → tanh → output (1024→8). If those
  weights are ever missing, the script stops instead of continuing with a random head.
- Clips are passed one at a time, so no padding distorts the mean.

**Robustness**
- Very short audio is padded with silence and near-silent audio is flagged; neither stops the
  run. A clip whose analysis fails gets a row of zeros. All such clips are logged and listed
  in `audio_problems.csv`.
- NaN/inf values are replaced by 0 and counted.
- Resume: rows computed earlier are reused; only new clips, or clips whose wav was re-made by
  Phase 2, are analysed. If the list of hand-crafted columns changes, all rows are recomputed.
- Same acceptance check as Phase 3: every `.npy` matches `ids.csv` row by row, has the expected
  number of columns, and contains no NaN/inf.

## Phase 5 — Text features — `src/05_features_text.py`

1. **TF-IDF (always)** → `text_tfidf` (N, size of vocabulary). Single words and word pairs of
   `text_clean`, up to 2,000 terms, each needing at least 2 train transcripts. The vocabulary
   and weights are **learned from the train split only**. One-letter words are kept on
   purpose, because "i" is a known personality cue. It is recomputed on every run (it
   depends on the split) and serves as the separate text baseline of Phase 6 and the source of
   the word cues in Phase 9. Vocabulary → `tfidf_vocab.json`.
2. **Sentence embedding (always)** → `text_minilm` (N, 384), from the frozen
   `sentence-transformers/all-MiniLM-L6-v2`.
3. **Text emotion (only if `USE_HEAVY_MODELS`)** → `text_emotion` (N, 7), from the frozen
   `j-hartmann/emotion-english-distilroberta-base`: softmax over all classes from the logits,
   class order saved to `text_emotion_labels.json`.

- The two transformer models read the transcript **with** punctuation and capital letters
  (only transcriber notes are removed), because that is the kind of text they were trained on.
  TF-IDF uses the fully cleaned `text_clean`.
- A clip without transcript gets a row of zeros in every text feature.
- Resume: a row is reused only if the clip's transcript is still the text it was computed
  from (`text_hashes.json`).

## Phase 6 — Unimodal models — `src/06_unimodal.py`

One model per modality (visual, audio, text) and per trait.

- **Inputs.** The feature sets of the modality side by side, as listed in
  `config.MODALITIES`: visual = `visual_cnn` + `visual_emotion`, audio = `audio_mfcc` +
  `audio_emotion`, text = `text_minilm` + `text_emotion`.
- **Classification** (is the trait at or above the train median?).
  StandardScaler → PCA keeping 95% of the variance, only when there are more than 512 inputs →
  classifier. Candidates: LogisticRegression with balanced classes and
  C ∈ {0.01, 0.1, 1, 10}, and an MLP with one hidden layer of 128 units and early stopping.
  All are fitted on **train**; the one with the best **validation** F1 is kept and evaluated
  once on **test**.
- **Calibration** (`config.CALIBRATE`, added after the first report). The kept classifier's
  P(high) goes through Platt scaling, fitted on its out-of-fold predictions inside **train**
  (5 folds grouped by YouTube video, so one person is never on both sides). Validation and test
  stay untouched and each model's AUC is unchanged; only over-confidence is removed. Before it,
  the face models for agreeableness and stability gave P < 0.02 or > 0.98 for 84–89% of held-out
  clips and outvoted the other modalities in late fusion. A slope ≤ 0 (a model useless out of fold)
  is floored at 1e-6: ranking kept, probabilities flat around the base rate.
- **Regression** (the raw score). Same inputs → Ridge, alpha chosen on validation →
  MAE and the ChaLearn score 1 − MAE. Predictions are clipped to [0, 1].
- **Baseline.** Text TF-IDF + LogisticRegression, reported as modality `text_tfidf`.
- **Outputs.** `results/unimodal.csv`, `results/unimodal_f1.png`, and every fitted model in
  `results/models/`.
- **Acceptance check.** At least 15 rows, every metric in [0, 1]; prints the modalities
  ranked by mean F1.

## Phase 7 — Fusion — `src/07_fusion.py`

Three ways of combining the modalities, compared with the best single modality.

| Type | How it combines |
|---|---|
| Early | All features side by side → StandardScaler → PCA (95%) → LogisticRegression, C chosen on validation. One model sees everything. |
| Late | The best single-modality models of Phase 6 each predict; the probabilities are averaged (`late`) or averaged with weights proportional to each model's validation AUC (`late_weighted`). |
| Intermediate | A small PyTorch network: one branch per modality (d → 128 → ReLU → Dropout 0.3 → 64 → ReLU), the outputs joined (192) → 64 → ReLU → Dropout 0.3 → 5 outputs, one per trait. |

- **Intermediate network training.** BCEWithLogitsLoss, Adam (learning rate 1e-3, weight
  decay 1e-4), batch size 64, at most 100 epochs, stopped when the validation loss has not
  improved for 10 epochs; the weights of the best epoch are kept. Inputs are standardised
  with train statistics. It predicts all five traits at once and has no raw-score output, so
  its MAE columns are empty.
- **Choosing the best fusion type.** By mean **validation** F1, never by test results; saved
  to `results/best_fusion.json` for Phase 8.
- **Outputs.** `results/fusion.csv`, `results/unimodal_vs_fusion.png`,
  `results/intermediate_loss.png`, and the trained network in `results/models/`.
- **Acceptance check.** Prints, for each trait, the best single modality, the best fusion
  type, the difference, and whether fusion is better.

## Phase 8 — Ablation — `src/08_ablation.py`

Shows how much each modality contributes.

- Takes the fusion type chosen in Phase 7 and trains it again on every combination:
  all three (V+A+T), one removed (A+T, V+T, V+A), and each alone (V, A, T).
- For each trait: **ΔF1 = F1(V+A+T) − F1(subset)** on the test split. A large positive ΔF1
  for A+T means the model got clearly worse without the face, so the face matters. A value
  near zero or below means the removed modality added nothing, or hurt.
- **Outputs.** `results/ablation.csv`, `results/ablation.png`.
- **Acceptance check.** Prints "Removing <modality> causes the largest average F1 drop (x.xx)".

## Phase 8b — Diagnostics — `src/08b_diagnostics.py`

Checks whether the models really learn. All models are rebuilt with exactly the code and seeds
of Phases 6–7, then tested eight ways; every table goes to `results/diagnostics/`.

| Check | What it does | File |
|---|---|---|
| B1 Baselines | Models that learn nothing: always the majority class; random guessing with the train class balance (mean of 100 draws); always predicting the train mean score | `baselines.csv` |
| B2 Overfitting | Train vs validation vs test F1 of every model and trait; flagged when train − test > 0.15 | `overfitting.csv` |
| B3 Confidence intervals | 95% bootstrap intervals (1,000 resamples) of accuracy, F1, AUC and 1 − MAE, per trait and averaged; plus the interval of the gain of the best fusion over the best single modality. Test clips of one YouTube video are resampled together, because they show the same person | `bootstrap_ci.csv`, `fusion_vs_single.csv` |
| B4 Label-shuffle test | The best fusion model retrained three times on randomly shuffled train/val labels; its test AUC must fall to 0.5 ± 0.1, otherwise the script **stops** and reports possible leakage | `shuffle_test.csv` |
| B5 Learning curve | The best fusion model trained on 25 / 50 / 75 / 100% of the train clips (whole YouTube videos at a time, three random draws per size) | `learning_curve.csv`, `learning_curve.png` |
| B6 Network training | Train / validation loss per epoch of the fusion network and the early-stopping epoch | `intermediate_loss.csv` |
| B7 Best model | Per-trait table and confusion matrices of the model with the best validation F1 | `best_model_per_trait.csv`, `confusion_best.png` |
| B8 Face features | Only if the face model is weak (mean F1 ≤ 0.55): each face feature set tried on its own | `visual_feature_comparison.csv` |

The bootstrap metrics are computed with small vectorised functions; they were checked to give
exactly the same F1 and AUC as scikit-learn.

## Phase 9 — Analysis and reports — `src/09_analysis.py`

1. **Voice cues.** Logistic regression on the standardised, named audio features, without
   PCA so that every weight belongs to one feature; the top 5 weights per trait →
   `results/cues_audio.csv`.
2. **Word cues.** Logistic regression on the TF-IDF words; the top 10 words towards "high"
   and towards "low" per trait → `results/cues_text.csv`.
3. **Emotion links.** Correlations between the face, voice and text emotion probabilities
   and the raw trait scores on the training clips → `results/emotion_trait_corr.csv` and
   `emotion_trait_corr.png`.
4. **Halo effect.** Correlations between the five trait labels →
   `results/trait_correlations.csv`.
5. **Reports.** `results/REPORT.md` (full plain-English report: data, three tables with 95%
   intervals, diagnostics, cues, verdict, limitations) and `results/summary.md` (the
   roadmap's short version). **Every number in them is read from the result files**; the
   only fixed reference value is the published top score (≈ 0.91 1 − MAE), kept in
   `config.py`.

## Phases 10–11 — not implemented yet

| Phase | Planned |
|---|---|
| 10 | Streamlit demo tab "Big Five (our model)" |
| 11 | Final check of `README.md` and `run_all.sh` |
