# VIP Task T1 — Multimodal Psychometric Assessment

Predicts the **Big Five personality traits** (openness, conscientiousness, extraversion,
agreeableness, neuroticism) from short talking-head videos of the
**ChaLearn First Impressions V2** dataset, using three modalities — face (visual), voice (audio)
and words (text) — on their own and fused.

This is a research prototype about *apparent* personality. It is **not** a diagnostic tool.

> **Build status:** Phases 1–2 (data preparation, preprocessing) are implemented.
> Phases 3–9 are added one by one; until then `run_all.sh` stops after Phase 2.

## Setup

```bash
brew install ffmpeg                 # system dependency (macOS)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. The device is chosen automatically: `cuda` → `mps` (Apple Silicon) → `cpu`.

## Where to put the data

Download from the ChaLearn site (login required):
<https://chalearnlap.cvc.uab.cat/dataset/24/data/41/description/>

```
data/raw/
├── zips/          all downloaded .zip files, untouched (nested zips are handled)
├── videos/        .mp4 clips, flat   (filled automatically from the zips)
├── annotations/   annotation_training.pkl
└── transcripts/   transcription_training.pkl
```

Videos must keep their ChaLearn file names (e.g. `J4GQm9j0JZ0.003.mp4`): the part before the
first dot identifies the YouTube video and is what the train/val/test split is grouped by.
If a zip is password protected, set `export CHALEARN_PW='...'` first.

## How to run

```bash
./run_all.sh                          # everything, settings from config.py
python src/02_preprocess.py           # or any single phase
```

Every script can be re-run at any time: finished clips are skipped, and if `data/splits.csv`
gained new clips only those are processed. All settings live in `config.py`; `MAX_CLIPS` and
`USE_HEAVY_MODELS` can also be set from the shell.

## Smoke test (fake labels)

To check that the code runs before the real label file is available:

```bash
python src/00_make_smoke_data.py                 # needs a few .mp4 in data/raw/videos/
MAX_CLIPS=50 USE_HEAVY_MODELS=0 ./run_all.sh
```

`00_make_smoke_data.py` fetches the real transcripts of your clips from a Hugging Face mirror
and writes **random** trait scores to
`data/raw/annotations/DUMMY_FAKE_annotation_RANDOM_SCORES.pkl`.

**While these dummy labels are in use, every number produced is meaningless.** The file
`data/FAKE_LABELS_DO_NOT_REPORT.txt` exists, every script prints a `LABELS ARE FAKE` banner,
and `results/summary.md` starts with the same warning.

## Switching to full data

1. Put the ChaLearn files in place — either
   - the downloaded zips in `data/raw/zips/`, **or**
   - the videos in `data/raw/videos/`, `annotation_training.pkl` in `data/raw/annotations/`
     and `transcription_training.pkl` in `data/raw/transcripts/`.
2. Run:

   ```bash
   MAX_CLIPS=2000 USE_HEAVY_MODELS=1 ./run_all.sh
   ```

Nothing else is needed, and no smoke file has to be deleted:

- The real `annotation_training.pkl` / `transcription_training.pkl` are always preferred;
  the DUMMY label file and the Hugging Face transcript subset are then ignored.
- `data/FAKE_LABELS_DO_NOT_REPORT.txt` is deleted automatically and the banner disappears.
- Clips already preprocessed during the smoke test are reused; only the new clips are processed.
- Phase 1 prints the clip-level train/val/test ratio and warns if it is outside
  65–75 / 10–20 / 10–20 %.

## What each phase does

| Script | What it does | Main outputs |
|---|---|---|
| `src/01_prepare_data.py` | Unzips, loads labels + transcripts, splits 70/15/15 **by YouTube video** (no person in two splits), binarises each trait at the **train** median | `data/splits.csv` |
| `src/02_preprocess.py` | 8 face crops per clip; cleaned 16 kHz audio (spectral gating → silence trim → peak normalise); cleaned transcript | `data/processed/frames/`, `data/processed/audio/`, `text_clean` + `missing_text` columns |
| `src/03`–`09` | features, unimodal models, fusion, ablation, analysis | *(coming)* |

## Output files

| File | Meaning |
|---|---|
| `data/splits.csv` | One row per clip: `video_id`, `youtube_id`, `split`, raw `text`, 5 raw trait scores, 5 `_bin` labels (1 = at or above the train median), `text_clean`, `missing_text` |
| `data/processed/frames/<video_id>_<k>.jpg` | Frame `k` (0–7) of a clip, face-cropped, 224×224 |
| `data/processed/audio/<video_id>.wav` | Cleaned mono audio, 16 kHz |
| `data/processed/preprocess_log.csv` | Per clip: frames with a detected face, audio seconds before / after cleaning |
| `data/processed/failed_clips.csv` | Clips dropped in Phase 2 and why (they are also removed from `splits.csv`) |
| `results/sample_frames.png` | The 8 face crops of 2 sample clips |
| `results/sample_waveform.png` | Raw vs cleaned audio of one clip |
| `results/text_length_hist.png` | Transcript length in words |
