# ROADMAP — VIP Task T1: Multimodal Psychometric Assessment

> **Instructions for Claude Code:** Read this entire file first. Build the project end-to-end exactly as specified, phase by phase. After each phase, run the phase's **Acceptance check** and do not move on until it passes. Ask the user only if a required input file is missing. Keep everything reproducible (`SEED = 42`).

---

## 0. Goal (one paragraph)

Investigate how **visual (face/video), acoustic (speech) and linguistic (text)** cues contribute to **psychometric assessment**. We predict the **Big Five personality traits** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from short talking-head videos in **ChaLearn First Impressions V2**. We:
1. preprocess each modality,
2. extract features (pretrained models as **frozen feature extractors** + classic hand-crafted features),
3. **train our own** model per modality,
4. train **early, intermediate and late fusion** models,
5. evaluate with accuracy / precision / recall / F1 / AUC (+ MAE on raw scores),
6. run an **ablation** (remove one modality at a time),
7. analyse which cues/modalities contribute most.

This is a research prototype, **not** a diagnostic tool.

---

## 1. Dataset — what to download and where to save it

**Dataset:** ChaLearn First Impressions V2 (CVPR'17).
**Site:** https://chalearnlap.cvc.uab.cat/dataset/24/description/ → register / log in → "Training" data page: https://chalearnlap.cvc.uab.cat/dataset/24/data/41/description/

**The USER downloads manually (Claude Code cannot log in):**

| What | Which files | How much |
|---|---|---|
| Training videos | **2–3 of the training video zip parts** (e.g. the first 2–3 `training80_XX.zip` parts) | ~1,500–2,000 clips is enough |
| Labels | the **training annotation** archive containing `annotation_training.pkl` | 1 file |
| Transcripts | the **training transcription** archive containing `transcription_training.pkl` | 1 file |

Zip passwords (if asked) are shown on the ChaLearn data page after login. Zips may be **nested** (zip inside zip).

**Save location (project root = the `VIP/` folder):**

```
VIP/
└── data/
    └── raw/
        ├── zips/            <- put ALL downloaded .zip files here, untouched
        ├── videos/          <- (created by script) all extracted .mp4 files, flat
        ├── annotations/     <- annotation_training.pkl
        └── transcripts/     <- transcription_training.pkl
```

> **Fallback if ChaLearn access fails:** stop and tell the user. Do NOT silently switch datasets or generate fake data.

---

## 2. Environment

- Local (VS Code + Jupyter extension) or Google Colab. Auto-select device: `cuda` → `mps` (Apple Silicon) → `cpu`.
- Python 3.10+. Create a venv and `requirements.txt`:

```
torch
torchvision
transformers
sentence-transformers
librosa
soundfile
moviepy
opencv-python
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
streamlit
pillow
```

- System dependency: `ffmpeg` (mac: `brew install ffmpeg`).

---

## 3. Project structure to create

```
VIP/
├── ROADMAP.md
├── requirements.txt
├── config.py                  # paths, SEED, TRAITS, N_FRAMES, SAMPLE_RATE, flags
├── src/
│   ├── 01_prepare_data.py     # unzip, load pkl, build splits.csv
│   ├── 02_preprocess.py       # frames, audio, text cleaning
│   ├── 03_features_visual.py
│   ├── 04_features_audio.py
│   ├── 05_features_text.py
│   ├── 06_unimodal.py
│   ├── 07_fusion.py
│   ├── 08_ablation.py
│   ├── 09_analysis.py         # cue analysis + all charts
│   └── utils.py               # device, seeding, metrics helpers, logging
├── data/ (raw/, processed/frames/, processed/audio/, splits.csv)
├── features/                  # *.npy + ids.csv
├── results/                   # *.csv, *.png, summary.md
├── app/streamlit_app.py       # demo (Phase 10)
└── run_all.sh                 # runs 01 → 09 in order
```

`config.py` must contain:
```python
SEED = 42
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
N_FRAMES = 8
IMG_SIZE = 224
SAMPLE_RATE = 16000
MAX_CLIPS = 2000          # cap for speed
USE_HEAVY_MODELS = True   # wav2vec2 / ViT emotion models; set False for a fast run
```

Every script must: be runnable standalone (`python src/0X_*.py`), skip work already done (cache/resume), show a `tqdm` progress bar, and log counts of successes/failures.

---

## Phase 1 — Data preparation (`01_prepare_data.py`)

**Implement**
1. Recursively unzip everything in `data/raw/zips/` (handle nested zips; prompt for password via env var `CHALEARN_PW` if needed). Move every `.mp4` flat into `data/raw/videos/`. Move found `.pkl` files into `annotations/` / `transcripts/`.
2. Load pkl files with `pickle.load(f, encoding="latin1")`. **Print the top-level keys first** and adapt — annotation pkl is expected to be `{trait_name: {video_filename: score}}` (may also contain `"interview"` — ignore it). Transcription pkl is expected to be `{video_filename: text}`.
3. Keep only videos that exist on disk (cap at `MAX_CLIPS`). Build a DataFrame: `video_id` (filename without `.mp4`), `youtube_id` (part before the first `.`), `text`, 5 raw trait scores.
4. **Split by `youtube_id` (GroupShuffleSplit), NOT by clip** — several clips come from the same YouTube video/person; a random split leaks identity. Ratio 70/15/15 train/val/test, `SEED`.
5. For each trait: median computed on **train only** → `<trait>_bin = (score >= median)` for all splits.
6. Save `data/splits.csv` with columns: `video_id, youtube_id, split, text, <5 raw traits>, <5 _bin traits>`.

**Expected output:** `data/splits.csv`; console prints counts per split and % positive per trait.

**Acceptance check:** no `youtube_id` appears in more than one split; every trait's positive rate in train is 45–55%; no NaN in trait columns.

---

## Phase 2 — Preprocessing (`02_preprocess.py`)

**Implement**
- **Video/image:** OpenCV → `N_FRAMES` evenly spaced frames (`np.linspace`). Detect the largest face with OpenCV Haar cascade (`haarcascade_frontalface_default.xml`); crop with 20% margin; if no face, use center crop. Resize to `IMG_SIZE×IMG_SIZE`, save JPG → `data/processed/frames/<video_id>_<k>.jpg`. (Normalization happens in Phase 3 with ImageNet mean/std.)
- **Speech:** extract audio with ffmpeg/moviepy → `librosa.load(sr=16000, mono=True)` → **noise removal:** simple spectral gating OR pre-emphasis + high-pass (document choice) → **segmentation:** `librosa.effects.trim(top_db=25)` to strip leading/trailing silence → **audio normalization:** peak normalize to 0.95 → save `data/processed/audio/<video_id>.wav`.
- **Text:** lowercase, strip punctuation/extra spaces, remove fillers (`um, uh, er, ah, like,` when standalone), tokenize (whitespace / `nltk`-free). Missing or empty transcript → `text_clean = ""`, `missing_text = 1`. Add `text_clean`, `missing_text` to `splits.csv`.
- Save 3 report figures to `results/`: `sample_frames.png` (grid of 8 frames for 2 clips), `sample_waveform.png` (raw vs cleaned), `text_length_hist.png`.

**Acceptance check:** every `video_id` in `splits.csv` has 8 frames + 1 wav (drop & log any that failed, and remove them from `splits.csv`). Print % missing transcripts.

---

## Phase 3 — Visual features (`03_features_visual.py`)

**Implement** (models are **frozen**, `torch.no_grad()`, batch processing):
1. **CNN features (required by brief):** torchvision `resnet50(weights=IMAGENET1K_V2)`, replace `fc` with Identity → 2048-d per frame → mean over 8 frames. Save `features/visual_cnn.npy` (N, 2048).
2. **Face-expression features (re-uses teammate's demo model):** if `USE_HEAVY_MODELS`: HF `trpakov/vit-face-expression` → for each frame take (a) softmax over **all** classes (`top_k=None`, sorted by fixed label order) → mean over frames → `features/visual_emotion.npy` (N, 7); (b) CLS embedding (last hidden state[:,0]) mean over frames → `features/visual_vit.npy` (N, 768).

**Acceptance check:** shapes correct, row count == number of ids, no NaN.

---

## Phase 4 — Audio features (`04_features_audio.py`)

**Implement**
1. **Hand-crafted (required by brief):** 40 MFCC (mean+std = 80), pitch via `librosa.yin(fmin=60,fmax=400)` (mean, std), RMS energy (mean, std), zero-crossing rate (mean), speaking-rate proxy = fraction of non-silent frames. → `features/audio_mfcc.npy` (N, ~87). Also save a column-name list `features/audio_mfcc_cols.json` (needed for cue analysis).
2. **Speech embeddings:** if `USE_HEAVY_MODELS`: HF `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` → (a) mean-pooled last hidden state → `features/audio_w2v.npy` (N, 1024); (b) softmax over **all** emotion classes, fixed order → `features/audio_emotion.npy` (N, 8).

**Acceptance check:** shapes correct, no NaN/inf (replace with 0 and log count).

---

## Phase 5 — Text features (`05_features_text.py`)

**Implement**
1. **TF-IDF:** `TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=2)` **fit on train only**, transform all → `features/text_tfidf.npy`; save vocabulary to `features/tfidf_vocab.json`.
2. **Transformer embeddings:** `sentence-transformers/all-MiniLM-L6-v2` → `features/text_minilm.npy` (N, 384). Empty text → zero vector.
3. **Text emotion:** if `USE_HEAVY_MODELS`: HF `j-hartmann/emotion-english-distilroberta-base`, `top_k=None`, fixed label order → `features/text_emotion.npy` (N, 7).
4. Save `features/ids.csv` — the `video_id` order that **all** `.npy` files follow (all scripts must use this exact order).

**Acceptance check:** all `.npy` row counts equal `len(ids.csv)`.

**Modality feature sets used downstream** (define in `config.py`):
```python
MODALITIES = {
  "visual": ["visual_cnn", "visual_emotion"],          # + "visual_vit" optional
  "audio":  ["audio_mfcc", "audio_emotion"],           # + "audio_w2v" optional
  "text":   ["text_minilm", "text_emotion"],           # tfidf used as a separate baseline
}
```
If `USE_HEAVY_MODELS=False`, drop the emotion/vit/w2v entries automatically.

---

## Phase 6 — Individual modality models (`06_unimodal.py`)

**Implement** — for each modality × each trait (binary `_bin` target):
- Pipeline: `StandardScaler` → optional `PCA(n_components=0.95)` for dims > 512 → classifier.
- Classifiers: (a) `LogisticRegression(max_iter=2000, class_weight="balanced")`, tune `C ∈ {0.01, 0.1, 1, 10}` on **val**; (b) `MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True)`.
- Pick best by val F1, refit on train, evaluate on **test**.
- Also a regression head (`Ridge`) on raw scores → report **MAE** and the ChaLearn score **1 − MAE**.
- Extra baseline row: text TF-IDF + LogisticRegression.

**Metrics (utils.py):** accuracy, precision, recall, F1 (binary), ROC-AUC, MAE, 1−MAE.

**Output:** `results/unimodal.csv` (columns: `modality, classifier, trait, acc, precision, recall, f1, auc, mae, one_minus_mae`), `results/unimodal_f1.png` (grouped bars: trait × modality), save fitted models to `results/models/` with `joblib`.

**Acceptance check:** 15+ rows; all metrics in [0,1]; print a ranking of modalities by mean F1.

---

## Phase 7 — Multimodal fusion (`07_fusion.py`)

Implement all three and compare with the best unimodal per trait:

1. **Early fusion:** concat scaled features of all modalities (`np.hstack`) → PCA(0.95) → LogisticRegression (tune C on val).
2. **Late fusion:** average the test-set `predict_proba` of the 3 best unimodal models; also a **weighted** version with weights = val-AUC (normalized).
3. **Intermediate fusion (PyTorch):**
   - One branch per modality: `Linear(d_m, 128) → ReLU → Dropout(0.3) → Linear(128, 64) → ReLU`.
   - Concatenate (192) → `Linear(192, 64) → ReLU → Dropout(0.3) → Linear(64, 5)` (multi-label, all 5 traits at once).
   - `BCEWithLogitsLoss`, Adam lr=1e-3, weight_decay=1e-4, batch 64, max 100 epochs, early stopping on val loss (patience 10). Standardize inputs using train stats.
   - Save loss curve `results/intermediate_loss.png`.

**Output:** `results/fusion.csv` (same columns as unimodal + `fusion_type`), `results/unimodal_vs_fusion.png` (per-trait F1: visual, audio, text, early, late, intermediate).

**Acceptance check:** table printed showing, for each trait, whether best fusion > best unimodal (and by how much).

---

## Phase 8 — Ablation (`08_ablation.py`)

**Implement:** using the **best fusion type** from Phase 7, retrain and test on:
`{V+A+T}`, `{A+T}` (no visual), `{V+T}` (no audio), `{V+A}` (no text) — and also each single modality for reference.
Compute **ΔF1 = F1(V+A+T) − F1(subset)** per trait.

**Output:** `results/ablation.csv`, `results/ablation.png` (bar chart of ΔF1 per removed modality, per trait).

**Acceptance check:** printed sentence like "Removing <modality> causes the largest average F1 drop (x.xx)".

---

## Phase 9 — Analysis & report material (`09_analysis.py`)

**Implement**
1. **Cue analysis:**
   - Audio: top-5 |coefficients| of the audio LogisticRegression per trait mapped to feature names (`audio_mfcc_cols.json`) — e.g. pitch_std, rms_mean.
   - Text: top-10 positive and negative TF-IDF words per trait.
   - Emotion features: correlation heatmap between emotion probabilities (face/voice/text) and raw trait scores → `results/emotion_trait_corr.png`.
2. Confusion matrix for the best overall model (one per trait, 1×5 grid) → `results/confusion_best.png`.
3. Write `results/summary.md` containing: dataset subset stats, final metrics table (unimodal vs fusion, mean over traits), ablation table, top cues per trait, 5-line conclusion answering:
   *"Do face, voice and words improve psychometric assessment when combined, and which modality contributes most?"*
   plus a Limitations section (subset of data, apparent-personality labels, possible demographic bias, not diagnostic) and Future Work (DAIC-WOZ for depression / PHQ-8).

**Acceptance check:** `summary.md` exists and every number in it is read from the CSVs (no hand-typed numbers).

---

## Phase 10 — Demo (`app/streamlit_app.py`) — only after Phases 1–9 pass

Extend the teammate's Streamlit demo with a **"Big Five (our model)"** tab:
- Upload one `.mp4` → run the same Phase 2 preprocessing + Phase 3–5 feature extraction on that single clip (audio and transcript from the **same** video; transcript via HF `openai/whisper-tiny` if no text provided).
- Load the saved intermediate-fusion model → show 5 trait probabilities as bars + which modality was used.
- Keep the disclaimer: research prototype, apparent personality, not diagnostic.
- Fix known issues in the teammate's app: image pipeline must use `top_k=None` (default returns only top 5 classes, which skews fusion); remove unused `speech_predict.py`.

---

## Phase 11 — One-command run

`run_all.sh`:
```bash
set -e
python src/01_prepare_data.py
python src/02_preprocess.py
python src/03_features_visual.py
python src/04_features_audio.py
python src/05_features_text.py
python src/06_unimodal.py
python src/07_fusion.py
python src/08_ablation.py
python src/09_analysis.py
echo "Done. See results/summary.md"
```

Also create `README.md` with: setup, where to put the data, how to run, what each output file means.

---

## Final deliverables checklist

- [ ] `data/splits.csv` (group split, no leakage)
- [ ] `features/*.npy` + `features/ids.csv`
- [ ] `results/unimodal.csv`, `results/fusion.csv`, `results/ablation.csv`
- [ ] Charts: `sample_frames.png`, `sample_waveform.png`, `unimodal_f1.png`, `unimodal_vs_fusion.png`, `intermediate_loss.png`, `ablation.png`, `emotion_trait_corr.png`, `confusion_best.png`
- [ ] `results/summary.md` (conclusion + limitations)
- [ ] `README.md`, `run_all.sh`, `requirements.txt`
- [ ] Streamlit demo tab (optional / last)

## Rules for Claude Code

- Never fabricate data, labels or results. If a file is missing, stop and say exactly which file and where it should go.
- Never evaluate on train data. Thresholds/medians/scalers/TF-IDF are fit on **train only**.
- Test the whole pipeline first with `MAX_CLIPS=50` and `USE_HEAVY_MODELS=False` (smoke test), then do the full run.
- Keep code simple, commented, and readable — students must explain it in a viva.
