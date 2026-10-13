# Phase outputs

What each phase produced in the **final run** (2026-10-14): the files it writes and its results.
What each phase *does* is described in [README.md](README.md). The plain-English results, with
95% confidence intervals and the verdict, are in [results/REPORT.md](results/REPORT.md).

**Final run settings:** `MAX_CLIPS=2000 USE_HEAVY_MODELS=1`, real ChaLearn labels, split file
`data/splits_final.csv`. All train-fitted objects (medians, TF-IDF, scalers, PCA, models) were
refitted on this split; only per-clip frames, audio and embeddings from earlier runs were reused.

| Phase | Result | Acceptance check |
|---|---|---|
| 1. Data preparation | 1,500 clips, 1,177 YouTube videos | Passed |
| 2. Preprocessing | 1,500 clips kept, 0 dropped | Passed |
| 3. Visual features | 3 files, 1,500 rows each | Passed |
| 4. Audio features | 3 files, 1,500 rows each | Passed |
| 5. Text features | 3 files, 1,500 rows each | Passed |
| 6. Unimodal models | 20 result rows, 40 saved models | Passed |
| 7. Fusion | 4 fusion types × 5 traits | Passed |
| 8. Ablation | 7 modality subsets × 5 traits | Passed |
| 8b. Diagnostics | 8 checks; label-shuffle test at chance level | Passed |
| 9. Analysis + reports | `REPORT.md`, `summary.md` | Passed |

---

## Data used

| | |
|---|---|
| Videos on disk | 1,502 (1,500 labelled + 2 duplicate downloads without a ChaLearn name) |
| Sources | 100 clips from `ChaLearn2016_tiny.zip`, 54 earlier clips, 1,346 clips downloaded from the Hugging Face mirror `yeray142/first-impressions-v2` (0 failed) |
| Labels | official `annotation_training.pkl`, `annotation_validation.pkl`, `annotation_test.pkl` (merged) |
| Transcripts | `transcription_hf_mirror.pkl` (all 10,000 clips of the mirror) |
| Fake-labels marker | absent — no dummy labels involved |

## Phase 1 — Data preparation

**File:** `data/splits_final.csv` — one row per clip (16 columns after Phase 2).

| Split | Clips | Share | YouTube videos |
|---|---|---|---|
| train | 1,041 | 69.4% | 823 |
| val | 238 | 15.9% | 177 |
| test | 221 | 14.7% | 177 |

| Trait | Train median | % high: train / val / test |
|---|---|---|
| openness | 0.5778 | 51.0 / 46.6 / 46.2 |
| conscientiousness | 0.5437 | 50.0 / 45.0 / 47.5 |
| extraversion | 0.4860 | 52.1 / 46.2 / 43.4 |
| agreeableness | 0.5604 | 52.3 / 47.1 / 44.8 |
| neuroticism | 0.5312 | 51.3 / 49.6 / 50.7 |

No YouTube video appears in two splits, and no split-ratio warning was raised.

## Phase 2 — Preprocessing

**Files:** `data/processed/frames/` (12,000 jpg), `data/processed/audio/` (1,500 wav),
`preprocess_log.csv`, `failed_clips.csv` (empty), `results/sample_frames.png`,
`sample_waveform.png`, `text_length_hist.png`.

| | |
|---|---|
| Processed now / reused | 696 / 804 |
| Failed or dropped | 0 |
| Faces detected | 11,173 of 12,000 frames (93.1%); 11 clips with no face in any frame |
| Detections replaced by the clip's median face box | 68 in 43 clips |
| Mean audio length | 15.3 s raw → 15.1 s after trimming silence |
| Missing transcripts | 0 (0.0%) |

## Phases 3–5 — Features

| File | Shape | NaN / inf | Time per new clip |
|---|---|---|---|
| `visual_cnn.npy` | (1500, 2048) | 0 / 0 | 0.072 s (ResNet-50) |
| `visual_emotion.npy` | (1500, 7) | 0 / 0 | 0.145 s (ViT, both files) |
| `visual_vit.npy` | (1500, 768) | 0 / 0 | |
| `audio_mfcc.npy` | (1500, 87) | 0 / 0 | 0.203 s |
| `audio_w2v.npy` | (1500, 1024) | 0 / 0 | 0.379 s (wav2vec2, both files) |
| `audio_emotion.npy` | (1500, 8) | 0 / 0 | |
| `text_tfidf.npy` | (1500, 2000) | 0 / 0 | refitted on the 1,041 train transcripts |
| `text_minilm.npy` | (1500, 384) | 0 / 0 | 0.002 s |
| `text_emotion.npy` | (1500, 7) | 0 / 0 | 0.004 s |

Also written: `ids.csv`, one `<name>.ids.csv` per file, the three emotion label-order files,
`audio_mfcc_cols.json`, `tfidf_vocab.json`, `audio_problems.csv` (0 clips),
`text_hashes.json`, `results/models/tfidf_vectorizer.joblib`.

## Phase 6 — Unimodal models

**Files:** `results/unimodal.csv` (20 rows), `results/unimodal_f1.png`, 40 models in
`results/models/`.

Mean test score over the five traits:

| Modality | F1 | AUC | Accuracy | 1 − MAE |
|---|---|---|---|---|
| Voice (audio) | 0.634 | 0.718 | 0.655 | 0.897 |
| Face (visual) | 0.604 | 0.660 | 0.622 | 0.890 |
| Words (text) | 0.533 | 0.552 | 0.536 | 0.875 |
| Words, TF-IDF baseline | 0.532 | 0.541 | 0.536 | 0.866 |

## Phase 7 — Fusion

**Files:** `results/fusion.csv`, `results/unimodal_vs_fusion.png`,
`results/intermediate_loss.png`, `results/best_fusion.json`,
`results/models/intermediate_fusion.pt` (+ scalers).

| Fusion type | Mean val F1 | Mean test F1 | Mean test AUC |
|---|---|---|---|
| Early | 0.624 | 0.615 | 0.712 |
| Late (average) | 0.622 | 0.608 | 0.697 |
| **Late (weighted)** — chosen on validation | **0.627** | **0.608** | **0.701** |
| Intermediate (network) | 0.598 | 0.625 | 0.738 |

- Fusion beat the best single modality on 3 of 5 traits.
- The fair comparison, with both choices made on validation, is late weighted vs voice:
  −0.026 mean test F1.
- The intermediate network stopped early; its best epoch was 3.

## Phase 8 — Ablation (late weighted fusion)

**Files:** `results/ablation.csv` (35 rows), `results/ablation.png`.

| Removed | Mean ΔF1 (positive = the model was better with it) |
|---|---|
| face | +0.002 |
| voice | +0.019 |
| words | −0.014 |

"Removing audio causes the largest average F1 drop (0.02)" — but every change is smaller than
the uncertainty of the F1 itself (about ±0.05).

## Phase 8b — Diagnostics

**Files in `results/diagnostics/`:** `baselines.csv`, `overfitting.csv`, `bootstrap_ci.csv`,
`fusion_vs_single.csv`, `shuffle_test.csv`, `learning_curve.csv`, `learning_curve.png`,
`intermediate_loss.csv`, `best_model_per_trait.csv`, `confusion_best.png` (also copied to
`results/`).

| Check | Result |
|---|---|
| Baselines (mean over traits) | always the majority class: acc 0.465, F1 0.635, AUC 0.500 · random guessing: acc 0.501, F1 0.494, AUC 0.503 · always the train mean: 1 − MAE 0.882 |
| Overfitting | 32 of 40 model/trait pairs flagged (train − test F1 > 0.15). Voice is the exception (mean gap 0.076). Validation and test F1 agree within 0.03 for every model |
| Fusion vs best single (paired bootstrap) | F1 −0.026 [−0.072, +0.020], AUC −0.017 [−0.058, +0.022]: not significant |
| Label-shuffle test (3 repeats) | test AUC 0.526, F1 0.511 → chance level, no sign of leakage |
| Learning curve (test AUC) | 261 clips 0.654 → 516 clips 0.674 → 777 clips 0.697 → 1,041 clips 0.701 |
| Best model (on validation) | late fusion (weighted) |
| Face features one by one | not needed (face mean F1 0.604 > 0.55) |
| Reproducibility | a second run of the diagnostics reproduced `bootstrap_ci.csv` byte for byte |

## Phase 9 — Analysis and reports

**Files:** `results/REPORT.md`, `results/summary.md`, `results/cues_audio.csv`,
`results/cues_text.csv`, `results/emotion_trait_corr.csv` + `.png`,
`results/trait_correlations.csv`.

- Halo effect: the five labels correlate on average at r = 0.71 (range 0.59–0.81).
- "Neuroticism" correlates positively with the other traits (0.72–0.81), consistent with it
  being coded as emotional stability.
- Strongest emotion link: face "happy" vs extraversion, r = +0.43.

---

## Earlier runs (superseded)

- **Smoke test.** 50 clips with random dummy labels, used only to check that the code runs.
- **First real run.** 154 clips (the tiny ChaLearn zip plus 54 earlier clips), used to build
  and test Phases 5–8.

The numbers above replace both. The old `data/splits.csv` from those runs is no longer used.
