# ROADMAP — Live Pipeline Visualizer (web)

> **Instructions for Claude Code:** Read this whole file first. Build phase by phase. After each
> phase run its **Acceptance check** and do not move on until it passes. The research pipeline
> (`src/01` → `src/09`) must keep working unchanged; `./run_all.sh` must still pass.

## Status

| Phase | What | Status | Acceptance check |
|---|---|---|---|
| 0 | `src/inference/`, `src/10_export_bundle.py`, `tests/test_parity.py` | Done | bundle reproduces `unimodal.csv` + `ablation.csv`; parity on 20 raw test clips passes (face / text exact, voice ≤ 0.008 SD, P(high) ≤ 5.5e-4) |
| 1 | `src/predict.py` | Done | `tests/test_inputs.py`: mp3 ≤ 0.041 P(high) shift, 2-min input → 8 windows, silent / corrupt → clean error; warm 15 s clip ≈ 12 s |
| 2 | `app/server.py` | Done | SSE in order, `queued` for a 2nd job, upload deleted after the run, path traversal → 404 |
| 3 | `web/` skeleton | Done | four capture modes; no horizontal scroll at 390 px |
| 4 | pipeline view | Done | real data only, shared playhead, reduced-motion respected |
| 5 | result screen | Done | `tests/test_fusion_maths.py`: browser = Python on all 45 replay × subset × trait cases |
| 6 | "How it works" | Done | numbers from `results/*.csv` via `/api/model-card` |
| 7 | hardening | Done (local) | `./dev.sh`, `make demo`, `make test`, `web/scripts/e2e.mjs`; hosting not set up; `LAZY_VISUAL` not needed (peak ≈ 3 GB) |
| 8 | research hooks | Not started | |

Deviations from the plan below, on purpose:
- **Percentiles use the 459 held-out clips** (validation + test), not the training clips: the
  training predictions are over-confident (train F1 up to 0.93), a new person is not.
- **Voice and words run at the same time** (Whisper in a background thread): the UI shows both
  lanes running, which is what really happens.
- **Demo samples are synthetic** (macOS text-to-speech): no real person's voice ships with the app.
  They are far outside the training data, and their extreme scores show it, which is a useful lesson.
- **Plain CSS instead of Tailwind / Framer Motion**: the machine runs Node 18, which Tailwind 4
  does not support, and the few animations need no library.
- **Voice parity is measured in standard deviations**: Homebrew's ffmpeg 9 decodes AAC ~2 ms shorter
  than the version Phase 2 ran with; features from the stored wavs match exactly.
- **Probability calibration added to the research models** (found with the first live video): the
  face models for agreeableness and stability gave P < 0.02 or > 0.98 for 84–89% of held-out clips
  and outvoted the other lanes. Platt scaling on grouped out-of-fold train predictions
  (`config.CALIBRATE`, see README Phase 6) fixed it: weighted late fusion went from test AUC 0.701 to
  0.733 and F1 0.608 to 0.664; single-lane AUCs are unchanged. Phases 6–10 and the replays were re-run.

---

## 0. Goal

A web app where a person records their voice (or a 15 s webcam clip, or uploads a file) and
**watches the real pipeline run on their own input**, stage by stage, lane by lane —
Face, Voice, Words → Fusion — and then gets an honest result.

**The one design rule:** every animation on screen stands for a computation that really
happened, and every number or picture comes from the user's own input. No fake progress bars, no
decorative "neural network" graphics, no made-up confidence. That rule keeps it impressive
and keeps it from being cringe.

**Framing on every screen:** *"How would strangers rate your first impression — and how well can a
machine copy that judgement?"* It is apparent personality, not a personality test. The label
called "neuroticism" in the data is really **emotional stability** and is shown as such.

### Target layout

```
VIP/
├── src/
│   ├── inference/              <- NEW: importable code shared by research + app
│   │   ├── preprocess.py       (moved out of 02_preprocess.py)
│   │   ├── audio.py            (moved out of 04_features_audio.py)
│   │   ├── visual.py           (moved out of 03_features_visual.py)
│   │   ├── text.py             (moved out of 05_features_text.py)
│   │   ├── bundle.py           (loads models + bundle.json once)
│   │   ├── fuse.py             (late fusion with missing modalities)
│   │   └── explain.py          (per-input cues)
│   ├── 10_export_bundle.py     <- NEW
│   └── predict.py              <- NEW: CLI + stage generator
├── app/
│   └── server.py               <- NEW: FastAPI + Server-Sent Events
├── web/                        <- NEW: Vite + React + TypeScript frontend
└── results/models/bundle.json  <- NEW (git-ignored like the rest of results/models/)
```

---

## Phase 0 — Inference bundle and shared code

**Why:** the app must turn an input into features *exactly* as training did. A slightly different
cleaning step gives silently wrong predictions. Some numbers the app needs are also computed during
training and then thrown away.

1. **Move the extraction functions into `src/inference/`** and make the numbered scripts import
   them. They cannot be imported today, because their file names start with a digit. Move, don't
   copy: `spectral_gate`, `clean_audio`, `handcrafted_features`, `build_wav2vec2`, the frame /
   face-crop functions, `clean_text`, the MiniLM and text-emotion builders. Behaviour stays the same.
2. **`src/10_export_bundle.py`** writes `results/models/bundle.json`:
   - late-fusion weights per trait and modality (validation AUC, as in `modeling.late_fusion`);
   - **fallback rule when the face is missing:** on validation, compare *voice only* against
     *voice + words, re-weighted*, per trait; store the winner. (Words are near chance,
     AUC 0.55, so they may hurt more than help. Let validation decide, not taste.)
   - training-split distribution of every fused and unimodal prediction (sorted arrays, or 101
     quantiles) → used for **percentiles** (raw scores bunch around the middle and look the same
     for everyone);
   - training-split mean / std / quantiles of the interpretable audio features (pitch, pitch
     variation, loudness, voiced fraction, speaking-time share) → "your pitch variation is at the
     78th percentile";
   - per-trait test AUC and accuracy of the model actually used → reliability chip in the UI;
   - emotion label lists, model names, config values the features depend on, a `created_at` and the
     git commit.
3. Add `10_export_bundle.py` to `run_all.sh` after Phase 9.

**Acceptance check**
- `./run_all.sh` output (results tables) identical to before the refactor.
- **Parity test** `tests/test_parity.py`: for 20 test clips, run the raw `.mp4` through
  `src/inference/` and compare with the stored rows in `features/*.npy` (`np.allclose`,
  `atol=1e-4`) and with the stored test predictions. This is the most important test in the whole
  roadmap.

---

## Phase 1 — `src/predict.py`: the engine (this is also the "mp3 input" feature)

One function, `run(path) -> Iterator[Event]`, used by both the CLI and the server. It **yields one
event per finished stage**, so the UI can show each stage the moment it is done.

Input: any of `mp3 / wav / m4a / ogg / webm / mp4 / mov` (ffmpeg converts it). Audio-only input
skips the face lane.

| # | stage | what it sends to the UI (kept small: downsampled, not raw arrays) |
|---|---|---|
| 1 | `ingest` | duration, has_audio, has_video, sample rate, file type |
| 2 | `audio_clean` | raw + cleaned waveform (~2,000 points each), noise-floor curve, seconds trimmed, gain applied |
| 3 | `audio_features` | mel spectrogram (PNG, base64), pitch contour + voiced mask, loudness curve, the 87 hand-crafted values, speech-emotion probabilities per 5 s window + overall |
| 4 | `transcribe` | words with start/end times (faster-whisper `small.en`, `word_timestamps=True`), the transcript before/after `clean_text` (fillers marked), text-emotion probabilities |
| 5 | `visual` (video only) | the 8 sampled frames (small JPEGs) with the detected face box, the 224×224 crops, face-emotion per frame |
| 6 | `unimodal` | per modality × trait: high-probability (classifier) and score (ridge) |
| 7 | `fusion` | weights used, which modality was dropped and how the weights were renormalized, fused probability, percentile, spread |
| 8 | `explain` | per trait: top cues of *this* input (see below), modality contributions |
| — | `done` / `error` | total time, per-stage timings; or a readable error ("no speech found") |

**Long input:** audio longer than 20 s is cut into 15 s windows (same length as the training clips).
Each window is predicted, and the result is the mean, with the spread between windows shown as
uncertainty.

**Explanations (`explain.py`), kept honest:**
- modality contribution = weight × (probability − 0.5), per trait;
- for the voice cues already listed in `results/cues_audio.csv`: the user's value as a training
  percentile + the direction the model reads it in;
- for words: which of the user's words appear in `results/cues_text.csv`.
- No SHAP on the MLPs in v1; say "top linear cues" in the UI, not "why the model decided".

**Known mismatch to state in the UI's "about" page:** training transcripts were written by humans,
the app's come from Whisper.

CLI: `python src/predict.py path/to/voice.mp3 [--json out.json]` prints a stage table and the
final traits.

**Acceptance check**
- The audio track of 5 test clips, exported to mp3, gives the same voice predictions as stored
  (small tolerance for mp3 compression; log the difference).
- A 2-minute mp3 runs through (windowed); a silent file and a corrupt file end with a clean
  `error` event, not a traceback.
- Per-stage timings are printed; total for a 15 s clip **< 25 s** on the dev Mac (8 GB).

---

## Phase 2 — API (`app/server.py`)

FastAPI + Server-Sent Events.

- `POST /api/analyze` (multipart file) → `{job_id}`; limits: 25 MB, 3 min.
- `GET /api/jobs/{id}/events` → SSE stream of the Phase 1 events, in order.
- `GET /api/samples` and `GET /api/samples/{name}/events` → **replay mode** (below).
- `GET /api/model-card` → the study results (Phase 6) as JSON.
- Models loaded **once at startup** through `bundle.py`. One job runs at a time (8 GB RAM),
  others wait in a queue and get a `queued` event with their position.
- **Privacy:** uploads go to a temp dir and are deleted when the job ends (also on error). Nothing
  of the content is logged. The response says so.
- **Replay mode:** run 3–4 sample inputs offline and save their event streams as JSON. Replaying
  them streams the same events with the original timing. A demo then works with no microphone,
  no wait, on any laptop, even if the models are not loaded. Use your own recordings or friends'
  (with consent), **not** ChaLearn clips: those are YouTube videos of real people under a research
  licence.

**Acceptance check:** `curl -F file=@voice.mp3 .../api/analyze` then `curl -N .../events` shows
every stage arriving in order; a second job started meanwhile reports `queued`; the temp file is gone
afterwards.

---

## Phase 3 — Frontend skeleton and look

Vite + React + TypeScript + Tailwind + Framer Motion; D3 only for scales/paths; canvas for
waveforms and the spectrogram.

**Look: "lab instrument", not "cyberpunk".** Dark background with a faint grid, a monospace font
for every number, one colour per lane (face / voice / words / fusion) used the same way everywhere,
thin lines, generous spacing. Motion is used for one thing: showing data arriving.

Screens:
1. **Landing** — one sentence of framing, a "Try it" button, a "See how it works" link.
2. **Capture** — four options: record voice (live input level + timer, 15–30 s), record webcam
   15 s (countdown, framing guide: "face centred, talk about anything"), upload a file, pick a
   sample. A consent checkbox ("processed and deleted, never stored").
3. **Run** — the pipeline view (Phase 4).
4. **Result** — Phase 5.
5. **How it works** — Phase 6.

**Acceptance check:** all screens reachable; recording in Chrome and Safari produces a file the API
accepts (MediaRecorder gives webm/opus or mp4; ffmpeg reads both); layout works at phone width.

---

## Phase 4 — The pipeline view (the core)

Four horizontal lanes on desktop (stacked on phone). Each **stage card** has three states: *waiting*
→ *running* (live elapsed timer — the real one) → *done* (reveals its data). Cards light up as their
SSE event arrives.

**Voice lane**
1. *Cleaning* — the raw waveform drawn, then the cleaned one fades in over it; the noise-gated parts
   dim, the trimmed silence slides off the ends. Chips: "−1.8 s silence", "noise floor −52 dB".
2. *Spectrogram* — revealed by a left-to-right sweep, pitch contour drawn on top (gaps where the
   voice is unvoiced), loudness underneath.
3. *Voice features* — 4 readable chips (pitch, pitch variation, loudness variation, speaking-time
   share), each with its training percentile; "+ 81 more" expands into the raw MFCC values.
4. *Speech emotion* — 8 bars per window, animating from window to window.

**Words lane**
1. *Transcription* — words appear at their real timestamps.
2. *Cleaning* — fillers struck through ("um", ", like,"), lowercase / punctuation stripped.
3. *Cues* — words found in the model's cue lists underlined in the trait's colour.
4. *Text emotion* — 7 bars.

**Face lane** (video only; otherwise shown disconnected with "no video — lane skipped")
1. the 8 frames along a time strip → black bars cut away → face box drawn → zoom into the
   224×224 crop;
2. face-emotion per frame as a small strip chart.

**Fusion lane**
- Three probability bars per trait (one per lane) flow into a fusion node; line thickness = weight.
- A missing lane visibly disconnects and the remaining weights re-grow to sum to 1 (labelled:
  "face missing → weights renormalized").
- The fused value per trait settles last.

**Shared playhead (the main "wow"):** an audio player under the lanes. While it plays, a playhead
moves across the waveform, spectrogram and pitch line, the current word lights up in the
transcript, and the current emotion window highlights. All of it is real, time-aligned data.

**Acceptance check:** watching a run of each sample: each card shows only data from its own event;
nothing animates before its event arrives; the playhead stays in sync within ~50 ms; works with
`prefers-reduced-motion` (no sweeps, data just appears).

---

## Phase 5 — Result screen

- **Five traits** as horizontal bars (clearer than a radar), each with: percentile vs the training
  clips, a band for the spread across windows, and a **reliability chip** from `bundle.json`
  ("model right 65% of the time on this trait — chance is 50%"). "Neuroticism" shown as
  **Emotional stability**.
- **What drove this:** per trait, 2–3 cues from `explain`, in plain words
  ("your pitch moved more than 78% of training speakers — read as more extraverted").
- **Modality contribution** per trait as a small stacked bar.
- **Interactive ablation:** toggles for Face / Voice / Words. The browser re-fuses from the unimodal
  probabilities it already has (same weights, renormalized), so the result changes live. This is
  Phase 8 of the research turned into something people can play with.
- **Fine print, always visible:** apparent personality, research prototype, never for hiring,
  diagnosis or judging anyone.
- Optional: "save result card" as PNG (contains no audio and no face).

**Acceptance check:** toggling modalities in the browser gives the same numbers as
`predict.py` with that modality removed (unit test on the fusion maths, same code path as
`fuse.py`).

---

## Phase 6 — "How it works" (the research, visual)

Built from the existing CSVs in `results/` (export them to one JSON in `10_export_bundle.py`):

- the pipeline diagram with the real model names and feature sizes per lane;
- Table 1 as a chart: baselines vs single modalities vs fusion, with confidence intervals;
- ablation chart; learning curve; label-shuffle test ("trained on shuffled labels → chance, so no
  leakage");
- the emotion × trait heat map; the halo-effect correlation matrix;
- limits, written plainly (1,500 of 10,000 clips; crowd-rated first impressions; untested for
  demographic bias; speech-emotion model trained on acted speech).

This page is where reviewers see that the project is rigorous, not just pretty.

**Acceptance check:** every number on the page matches `results/REPORT.md`.

---

## Phase 7 — Hardening and deploy

- **Speed:** load models once; `mps` for wav2vec2 / ViT / ResNet, faster-whisper `int8` on CPU;
  run the voice and words lanes in parallel where memory allows. Expected peak memory ~3 GB (wav2vec2-large is the
  biggest); add `LAZY_VISUAL=1` to load face models only when video arrives.
- **Tests:** parity (Phase 0), fusion maths (Phase 5), API stream order (Phase 2), one Playwright
  test that replays a sample end to end.
- **Run locally with one command:** `make demo` → starts the API + the built frontend.
- **Hosting (optional):** Docker image → Hugging Face Spaces (CPU, slow but free), with replay mode
  as the default landing experience and "run on your voice" clearly marked as slower.

**Acceptance check:** fresh clone → `make demo` → sample replay works without any model
download; a live 15 s voice run finishes; all tests pass.

---

## Phase 8 (optional, research) — turn the app into an instrument

- **Validity study:** with consent (check whether your university needs ethics approval), participants
  record 30 s *and* fill in a real Big Five questionnaire (BFI-2 or IPIP-50). Correlate predicted
  apparent personality with self-report. Even ~30 people answer the question this project is built
  around.
- **Bias view:** per-group error on the ChaLearn gender / ethnicity annotations, shown on the
  "How it works" page.
- **Interpretable voice features:** add eGeMAPS (openSMILE) and speaking rate / pauses from Whisper
  timestamps; they replace MFCC indices in the explanations.

---

## Build order and rough effort

| phase | effort | depends on |
|---|---|---|
| 0 bundle + refactor + parity test | 1–2 days | — |
| 1 predict.py | 2 days | 0 |
| 2 API + replay | 1 day | 1 |
| 3 frontend skeleton | 1 day | 2 (can start with replay JSON) |
| 4 pipeline view | 3–4 days | 3 |
| 5 result screen | 1–2 days | 4 |
| 6 how it works | 1 day | 0 |
| 7 hardening | 1–2 days | all |

Phases 3–6 can be built against **replay JSON** before the live backend is fast, so frontend and
backend work don't block each other.
