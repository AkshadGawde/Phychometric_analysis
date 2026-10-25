"""Run the trained pipeline on ONE new input - a voice recording or a talking-head video.

    python src/predict.py path/to/voice.mp3                 table of the five traits
    python src/predict.py clip.mp4 --json out.json          + every stage's output as JSON
    python src/predict.py voice.wav --record my-sample      + save a replay for the app's demo mode

Any file ffmpeg can read works (mp3, wav, m4a, ogg, webm, mp4, mov). Audio-only input skips
the face lane; the fusion then uses the subset of modalities chosen on VALIDATION for exactly
that situation (results/models/bundle.json, written by src/10_export_bundle.py).

Engine.run(path) yields one event per stage, the moment the stage is finished, so the web app
(app/server.py) can show the pipeline working:

  ingest -> audio_clean -> audio_features -> transcribe -> visual -> unimodal -> fusion -> explain

Every feature is made by the SAME code as in training (src/inference/), including the 16-bit wav
and jpg round trips the training data went through. tests/test_parity.py checks this.
Inputs longer than WINDOW_MIN_TOTAL_SEC are cut into windows about as long as the training
clips; each window is predicted and the results are averaged (their spread is reported).

Apparent personality - how strangers would rate a first impression - not a personality test.
"""
import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import joblib
import librosa
import numpy as np
from PIL import Image

import modeling
import utils
import config
from inference import audio, preprocess, text, visual
from inference.audio import MFCC_COLUMNS

log = utils.get_logger("predict")

MODALITIES = ["visual", "audio", "text"]
STAGES = ["ingest", "audio_clean", "audio_features", "transcribe", "visual", "unimodal", "fusion",
          "explain"]
TRAIT_LABELS = {"openness": "Openness", "conscientiousness": "Conscientiousness",
                "extraversion": "Extraversion", "agreeableness": "Agreeableness",
                "neuroticism": "Emotional stability"}      # the dataset's "neuroticism" = stability
VOICE_LABELS = {"pitch_mean": "pitch", "pitch_std": "pitch variation",
                "voiced_fraction": "voiced share", "rms_mean": "loudness",
                "rms_std": "loudness variation", "zcr_mean": "breathiness / noise",
                "nonsilent_fraction": "speaking-time share"}


class InputError(Exception):
    """A problem with the input itself (no audio, silent, too long) - shown to the user as is."""


# ----------------------------------------------------------------------------
# Small helpers for the display data
# ----------------------------------------------------------------------------
def r(x, digits=4):
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


def envelope(y, n):
    """Peak level of n equal buckets: enough to draw a waveform."""
    if len(y) == 0:
        return []
    edges = np.linspace(0, len(y), min(n, len(y)) + 1).astype(int)
    return [r(np.max(np.abs(y[a:b])), 3) for a, b in zip(edges[:-1], edges[1:])]


def downsample(values, n, reduce=np.nanmean):
    """Reduce a per-frame curve to at most n points (NaN stays None)."""
    values = np.asarray(values, dtype=float)
    if len(values) <= n:
        return [r(v, 3) for v in values]
    edges = np.linspace(0, len(values), n + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = values[a:b]
        out.append(r(reduce(chunk), 3) if np.isfinite(chunk).any() else None)
    return out


def png_data_uri(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def jpg_data_uri(bgr, width=None, quality=80):
    if width and bgr.shape[1] > width:
        bgr = cv2.resize(bgr, (width, int(bgr.shape[0] * width / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    _, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def spectrogram_image(y, width=900, height=160):
    """Mel spectrogram (dB, magma colours, low pitch at the bottom) as a PNG data URI."""
    import matplotlib
    mel = librosa.feature.melspectrogram(y=y, sr=config.SAMPLE_RATE, n_fft=config.AUDIO_FRAME_LENGTH,
                                         hop_length=config.AUDIO_HOP_LENGTH, n_mels=96, fmax=8000)
    db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    level = ((db + 80.0) / 80.0)[::-1]                           # 0..1, low frequencies at the bottom
    level = cv2.resize(level.astype(np.float32), (min(width, level.shape[1]), height),
                       interpolation=cv2.INTER_AREA)
    rgb = (matplotlib.colormaps["magma"](np.clip(level, 0, 1))[..., :3] * 255).astype(np.uint8)
    return png_data_uri(rgb)


def percentile(value, quantiles):
    """Position of `value` among the held-out clips, in % (from 101 stored quantiles)."""
    q = np.asarray(quantiles)
    return float(np.clip(np.interp(value, q, np.linspace(0, 100, len(q))), 0, 100))


def readable_feature(name):
    if name.startswith("voice emotion: "):
        return f"sounds {name.split(': ')[1]}"
    if name in VOICE_LABELS:
        return VOICE_LABELS[name]
    band, stat = name[4:].split("_")                             # mfcc12_std -> 12, std
    return f"timbre band {band} ({'variation' if stat == 'std' else 'level'})"


def ffprobe(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=codec_type,codec_name:stream_disposition=attached_pic:format=duration,format_name",
           "-of", "json", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise InputError("the file could not be read (is it an audio or video file?)")
    info = json.loads(result.stdout or "{}")
    streams = info.get("streams", [])
    # A cover picture inside an mp3 shows up as a "video" stream: it is not a video.
    video = [s for s in streams if s.get("codec_type") == "video"
             and not s.get("disposition", {}).get("attached_pic")]
    duration = info.get("format", {}).get("duration")
    return {"has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "has_video": bool(video),
            "codecs": sorted({s.get("codec_name", "?") for s in streams}),
            "container": info.get("format", {}).get("format_name", "?"),
            "duration": float(duration) if duration not in (None, "N/A") else None}


def readable_video(path, tmp_dir):
    """OpenCV cannot always decode browser recordings (webm without a frame count): re-encode
    those to a plain mp4 first. The 8 frames are then taken exactly as in Phase 2."""
    frames = preprocess.read_frames(path)
    if frames is not None:
        return frames
    mp4 = os.path.join(tmp_dir, "video.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "18", mp4]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        return None
    return preprocess.read_frames(mp4)


# ----------------------------------------------------------------------------
# The engine: loads every model once, then runs inputs
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self):
        if not os.path.exists(config.BUNDLE_JSON):
            sys.exit(f"Missing {config.BUNDLE_JSON} - run `python src/10_export_bundle.py` first.")
        with open(config.BUNDLE_JSON) as f:
            self.bundle = json.load(f)
        self.device = utils.get_device()
        self.loaded = False

    def load(self, progress=lambda name: None):
        """Load all models. `progress(name)` is called before each one (the app shows it)."""
        if self.loaded:
            return
        self.classifiers, self.ridges = {}, {}
        for modality in MODALITIES:
            for trait in config.TRAITS:
                stem = os.path.join(config.MODELS_DIR, f"unimodal_{modality}_{trait}")
                self.classifiers[modality, trait] = joblib.load(stem + "_clf.joblib")
                self.ridges[modality, trait] = joblib.load(stem + "_ridge.joblib")
        progress("speech emotion (wav2vec2)")
        self.wav2vec2, _ = audio.build_wav2vec2(self.device)
        progress("face networks (ResNet-50, ViT)")
        self.resnet = visual.build_resnet(self.device)
        self.vit, _ = visual.build_vit(self.device)
        progress("text models (MiniLM, emotion)")
        self.minilm = text.build_minilm(self.device)
        self.text_emotion, _ = text.build_text_emotion(self.device)
        progress(f"speech recognition (Whisper {config.WHISPER_MODEL})")
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        self.loaded = True

    def warm_up(self):
        """Run every model once on a synthetic 4 s signal: the first real input would otherwise
        pay for just-in-time compilation (pitch tracker) and GPU start-up - over a minute."""
        sr = config.SAMPLE_RATE
        t = np.arange(4 * sr) / sr
        y = 0.3 * np.sin(2 * np.pi * (140 + 20 * np.sin(2 * np.pi * 0.5 * t)) * t) * (np.sin(2 * np.pi * 2 * t) > 0)
        y += 0.005 * np.random.default_rng(0).standard_normal(len(y))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "warm.wav")
            import soundfile as sf
            sf.write(path, y.astype(np.float32), sr)
            for _ in self.run(path):
                pass
        blank = [Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))] * config.N_FRAMES
        self.resnet(blank)
        self.vit(blank)

    # ------------------------------------------------------------------
    def run(self, path, transcript=None, collect=None):
        """Yield the events of one input. `transcript` (optional) replaces speech recognition
        (used by the parity test with the dataset's own transcripts). `collect` (optional dict)
        receives the raw feature rows."""
        self.load()
        start = time.time()
        timings = {}
        state = {}

        stages = [("ingest", self._ingest), ("audio_clean", self._audio_clean),
                  ("audio_features", self._audio_features), ("transcribe", self._transcribe),
                  ("visual", self._visual), ("unimodal", self._unimodal), ("fusion", self._fusion),
                  ("explain", self._explain)]
        started = {}
        with tempfile.TemporaryDirectory() as tmp_dir, ThreadPoolExecutor(max_workers=1) as pool:
            state.update(path=path, transcript=transcript, tmp_dir=tmp_dir)
            for name, step in stages:
                if name not in started:
                    started[name] = time.time()
                    yield {"type": "stage_start", "stage": name, "t": round(started[name] - start, 3)}
                if name == "audio_features" and transcript is None:
                    # Speech recognition only needs the raw audio: it runs in the background
                    # while the voice features are computed (the two lanes really run at once).
                    state["asr"] = pool.submit(self._recognise, state["raw"])
                    started["transcribe"] = time.time()
                    yield {"type": "stage_start", "stage": "transcribe",
                           "t": round(started["transcribe"] - start, 3)}
                try:
                    data = step(state)
                except InputError as error:
                    yield {"type": "error", "stage": name, "message": str(error)}
                    return
                except Exception as error:                          # noqa: BLE001 - shown to the user
                    log.exception(f"stage {name} failed")
                    yield {"type": "error", "stage": name, "message": f"{name} failed: {error}"}
                    return
                timings[name] = round(time.time() - started[name], 3)
                yield {"type": "stage", "stage": name, "seconds": timings[name],
                       "t": round(time.time() - start, 3), "data": data}
        if collect is not None:
            collect.update(state["features"])
        yield {"type": "done", "seconds": round(time.time() - start, 3), "timings": timings,
               "summary": state["summary"]}

    # ---- 1. ingest ------------------------------------------------------------------
    def _ingest(self, s):
        info = ffprobe(s["path"])
        if not info["has_audio"]:
            raise InputError("the file has no sound track - the voice lane needs speech")
        if info["duration"] and info["duration"] > config.APP_MAX_INPUT_SEC + 1:
            raise InputError(f"the input is {info['duration']:.0f} s long; the limit is "
                             f"{config.APP_MAX_INPUT_SEC} s")
        raw = preprocess.extract_raw_audio(s["path"])                 # 16 kHz mono, as in Phase 2
        s["raw"] = raw
        s["has_video"] = info["has_video"]
        info["duration"] = len(raw) / config.SAMPLE_RATE
        return {**info, "sample_rate": config.SAMPLE_RATE, "raw_sec": r(info["duration"], 2)}

    # ---- 2. cleaning -----------------------------------------------------------------
    def _audio_clean(self, s):
        raw, sr = s["raw"], config.SAMPLE_RATE
        details = {}
        try:
            clean = preprocess.clean_audio(raw, details)
        except RuntimeError:
            raise InputError("no usable speech found: the recording is silent or too short") from None
        clean = preprocess.as_saved_wav(clean)                         # 16-bit round trip, as in training
        s["clean"], s["trim_start"] = clean, details["trim_start"]
        # Background level: loudness of the quietest frames of the raw input, in dB full scale.
        frame_rms = librosa.feature.rms(y=raw, frame_length=512, hop_length=128)[0]
        noise_db = 20 * np.log10(np.percentile(frame_rms, config.NOISE_QUIET_PERCENT) + 1e-10)
        gate = details["gain_over_time"]
        n_points = 1200
        return {
            "raw_sec": r(len(raw) / sr, 2), "clean_sec": r(len(clean) / sr, 2),
            "trim_start_sec": r(details["trim_start"] / sr, 3), "trim_end_sec": r(details["trim_end"] / sr, 3),
            "removed_sec": r((len(raw) - len(clean)) / sr, 2),
            "gain": r(details["gain"], 3), "noise_floor_db": r(noise_db, 1),
            "gated_share": r(float(np.mean(gate < 0.5)), 3),
            "raw_envelope": envelope(raw, n_points),
            "gated_envelope": envelope(details["gated"], n_points),
            "clean_envelope": envelope(clean, max(1, int(n_points * len(clean) / len(raw)))),
            "gate_curve": downsample(gate, 600),
            "settings": {"noise_quiet_percent": config.NOISE_QUIET_PERCENT,
                         "gate_strength": config.NOISE_GATE_STRENGTH, "trim_top_db": config.TRIM_TOP_DB,
                         "peak_level": config.PEAK_LEVEL},
        }

    # ---- 3. voice features -------------------------------------------------------------
    def _audio_features(self, s):
        clean, sr = s["clean"], config.SAMPLE_RATE
        n = len(clean)
        if n / sr > config.WINDOW_MIN_TOTAL_SEC:
            n_windows = max(2, int(round(n / sr / config.WINDOW_SEC)))
        else:
            n_windows = 1                                              # the whole input, as in training
        bounds = np.linspace(0, n, n_windows + 1).astype(int)
        s["windows"] = list(zip(bounds[:-1], bounds[1:]))

        rows, emotions, curves = [], [], {"f0": [], "voiced": [], "rms": []}
        for a, b in s["windows"]:
            y, notes = audio.prepare_waveform(clean[a:b])
            frames = {}
            rows.append(audio.handcrafted_features(y, frames))
            emotions.append(self.wav2vec2(y)[1])
            for k in curves:
                curves[k].append(frames[k][:max(1, (b - a) // config.AUDIO_HOP_LENGTH + 1)])
        mfcc = np.nan_to_num(np.stack(rows))
        emotion = np.stack(emotions)
        s["audio_X"] = np.hstack([mfcc, emotion]) if "audio_emotion" in config.MODALITIES["audio"] else mfcc
        s.setdefault("features", {})["audio_mfcc"], s["features"]["audio_emotion"] = mfcc, emotion

        # Display-only: a speech-emotion timeline in short steps.
        step = int(config.EMOTION_STEP_SEC * sr)
        timeline = []
        for a in range(0, n, step):
            b = min(n, a + step)
            if b - a < sr and timeline:                                # a last scrap < 1 s: skip
                break
            probs = self.wav2vec2(audio.prepare_waveform(clean[a:b])[0])[1]
            timeline.append({"start": r(a / sr, 2), "end": r(b / sr, 2), "probs": [r(p, 4) for p in probs]})

        f0 = np.concatenate(curves["f0"])
        rms = np.concatenate(curves["rms"])
        loud_db = 20 * np.log10(np.maximum(rms, 1e-10) / max(rms.max(), 1e-10))
        values = mfcc.mean(axis=0)
        measures = []
        for name in VOICE_LABELS:
            value = float(values[MFCC_COLUMNS.index(name)])
            measures.append({"name": name, "label": VOICE_LABELS[name], "value": r(value, 4),
                             "percentile": r(percentile(value, self.bundle["voice_quantiles"][name]), 1)})
        s["voice_measures"] = {m["name"]: m for m in measures}
        hop_sec = config.AUDIO_HOP_LENGTH / sr
        n_points = 600
        return {
            "spectrogram": spectrogram_image(clean),
            "duration_sec": r(n / sr, 2),
            "pitch": downsample(f0, n_points), "loudness_db": downsample(loud_db, n_points),
            "curve_sec": r(len(f0) * hop_sec, 2),
            "voiced_share": r(float(np.mean(np.concatenate(curves["voiced"]))), 3),
            "measures": measures,
            "n_features": int(mfcc.shape[1]),
            "mfcc_means": [r(v, 2) for v in values[:config.N_MFCC]],
            "emotion_labels": self.bundle["labels"]["voice_emotion"],
            "emotion": [r(p, 4) for p in emotion.mean(axis=0)],
            "emotion_timeline": timeline,
            "windows": [{"start": r(a / sr, 2), "end": r(b / sr, 2)} for a, b in s["windows"]],
        }

    # ---- 4. words ----------------------------------------------------------------------
    def _recognise(self, raw):
        """Whisper on the raw (uncleaned) audio: words with their times in the raw timeline."""
        segments, _ = self.whisper.transcribe(raw, language="en", word_timestamps=True,
                                              condition_on_previous_text=False, beam_size=5)
        return [{"w": w.word, "start": r(w.start, 2), "end": r(w.end, 2), "p": r(w.probability, 3)}
                for segment in segments for w in (segment.words or [])]

    def _transcribe(self, s):
        sr = config.SAMPLE_RATE
        words = []
        if s["transcript"] is not None:                                 # given (parity test, samples)
            transcript = s["transcript"]
            source = "given"
        else:
            words = s["asr"].result()
            transcript = "".join(w["w"] for w in words).strip()
            source = f"whisper {config.WHISPER_MODEL}"
        missing = not transcript.strip()
        s["text_available"] = not missing

        # Text per window: words by their middle time (raw timeline -> cleaned timeline).
        window_texts = []
        if words and len(s["windows"]) > 1:
            for a, b in s["windows"]:
                lo, hi = (s["trim_start"] + a) / sr, (s["trim_start"] + b) / sr
                chosen = [w["w"] for w in words if lo <= (w["start"] + w["end"]) / 2 < hi]
                window_texts.append(text.model_text("".join(chosen).strip(), not chosen))
        else:
            window_texts = [text.model_text(transcript, missing)] * len(s["windows"])

        minilm = self.minilm(window_texts)
        emotion = self.text_emotion(window_texts)
        s["text_X"] = np.hstack([minilm, emotion]) if "text_emotion" in config.MODALITIES["text"] else minilm
        s["features"]["text_minilm"], s["features"]["text_emotion"] = minilm, emotion
        cleaned = preprocess.clean_text(transcript) if not missing else ""
        s["text_clean"] = cleaned

        cues = self.bundle["text_cues"]
        padded = f" {cleaned} "
        found = {t: {side: [w for w in cues[t][side] if f" {w} " in padded] for side in ("high", "low")}
                 for t in config.TRAITS}
        s["text_found"] = found
        cue_words = {}
        for t in config.TRAITS:
            for side in ("high", "low"):
                for w in found[t][side]:
                    if " " not in w:
                        cue_words.setdefault(w, []).append({"trait": t, "side": side})
        tokens = []
        if words:
            for w in words:
                c = preprocess.clean_text(w["w"])
                tokens.append({**w, "clean": c, "removed": c == "",
                               "cues": cue_words.get(c, [])})
        else:
            for raw_token in transcript.split():
                c = preprocess.clean_text(raw_token)
                tokens.append({"w": " " + raw_token, "start": None, "end": None, "clean": c,
                               "removed": c == "", "cues": cue_words.get(c, [])})
        return {
            "source": source, "available": not missing,
            "transcript": transcript, "clean": cleaned,
            "n_words": len(cleaned.split()), "tokens": tokens,
            "words_per_min": r(len(cleaned.split()) / max(1e-6, len(s["clean"]) / sr) * 60, 1),
            "cue_words_found": found,
            "emotion_labels": self.bundle["labels"]["text_emotion"],
            "emotion": [r(p, 4) for p in emotion.mean(axis=0)] if not missing else None,
            "embedding_dims": int(minilm.shape[1]),
            "note": None if not missing else "no speech recognised: the words lane is skipped",
        }

    # ---- 5. face -----------------------------------------------------------------------
    def _visual(self, s):
        if not s["has_video"]:
            s["visual_available"] = False
            return {"available": False, "reason": "audio-only input: no video, the face lane is skipped"}
        frames = readable_video(s["path"], s["tmp_dir"])
        if frames is None:
            s["visual_available"] = False
            return {"available": False, "reason": "the video could not be decoded"}
        full_h, full_w = frames[0].shape[:2]
        x0, y0, x1, y1 = preprocess.content_box(frames)
        frames = [f[y0:y1, x0:x1] for f in frames]
        detected = [preprocess.detect_face(f) for f in frames]
        boxes, replaced = preprocess.choose_face_boxes(frames)
        images, shown = [], []
        for frame, box, found in zip(frames, boxes, detected):
            crop = preprocess.crop_to_box(frame, box)
            jpg = preprocess.encode_crop(crop)                         # jpg round trip, as in training
            images.append(Image.open(io.BytesIO(jpg)).convert("RGB"))
            h, w = frame.shape[:2]
            cx0, cy0, cx1, cy1 = preprocess.crop_bounds(frame, box)
            shown.append({
                "frame": jpg_data_uri(frame, width=320),
                "crop": jpg_data_uri(crop, width=160),
                "face": None if box is None else [r(box[0] / w, 4), r(box[1] / h, 4), r(box[2] / w, 4), r(box[3] / h, 4)],
                "crop_box": [r(cx0 / w, 4), r(cy0 / h, 4), r((cx1 - cx0) / w, 4), r((cy1 - cy0) / h, 4)],
                "detected": found is not None,
                "replaced": found is not None and box is not None and not np.allclose(found, box),
            })
        cnn = visual.mean_over_frames(self.resnet(images), 1)
        probs, _ = self.vit(images)
        emotion = visual.mean_over_frames(probs, 1)
        x = np.hstack([cnn, emotion]) if "visual_emotion" in config.MODALITIES["visual"] else cnn
        s["visual_X"] = np.repeat(np.nan_to_num(x), len(s["windows"]), axis=0)
        s["features"]["visual_cnn"], s["features"]["visual_emotion"] = cnn, emotion
        s["visual_available"] = True
        per_frame = probs.cpu().numpy()
        for item, p in zip(shown, per_frame):
            item["emotion"] = [r(v, 4) for v in p]
        return {
            "available": True, "frames": shown, "n_frames": config.N_FRAMES,
            "faces_found": int(sum(d is not None for d in detected)), "replaced": int(replaced),
            "content_box": [r(x0 / full_w, 4), r(y0 / full_h, 4), r((x1 - x0) / full_w, 4), r((y1 - y0) / full_h, 4)],
            "bars_removed": (x0, y0, x1, y1) != (0, 0, full_w, full_h),
            "emotion_labels": self.bundle["labels"]["face_emotion"],
            "emotion": [r(v, 4) for v in emotion[0]],
            "cnn_dims": int(cnn.shape[1]),
        }

    # ---- 6. one model per modality ---------------------------------------------------------
    def _unimodal(self, s):
        inputs = {"audio": s["audio_X"]}
        if s.get("visual_available"):
            inputs["visual"] = s["visual_X"]
        if s.get("text_available"):
            inputs["text"] = s["text_X"]
        s["available"] = [m for m in MODALITIES if m in inputs]
        probs, scores = {}, {}
        for m, X in inputs.items():
            probs[m], scores[m] = {}, {}
            for t in config.TRAITS:
                probs[m][t] = modeling.positive_probability(self.classifiers[m, t], X)       # per window
                scores[m][t] = modeling.predict_score(self.ridges[m, t], X)
        s["probs"], s["scores"] = probs, scores
        return {
            "available": s["available"],
            "skipped": [m for m in MODALITIES if m not in inputs],
            "inputs": {m: int(X.shape[1]) for m, X in inputs.items()},
            "classifiers": {m: {t: modeling.estimator_name(self.classifiers[m, t]) for t in config.TRAITS}
                            for m in inputs},
            "prob": {m: {t: r(float(np.mean(p)), 4) for t, p in v.items()} for m, v in probs.items()},
            "score": {m: {t: r(float(np.mean(p)), 4) for t, p in v.items()} for m, v in scores.items()},
        }

    # ---- 7. fusion -----------------------------------------------------------------------
    def _fusion(self, s):
        available = s["available"]
        key = "+".join(available)
        choice = self.bundle["default_choice"][key]
        # Everything the browser needs to re-fuse any subset of the available modalities.
        options = {}
        for subset_key, per_trait in self.bundle["fusion"].items():
            if set(subset_key.split("+")) <= set(available):
                options[subset_key] = {t: {"weights": v["weights"], "quantiles": v["quantiles"],
                                           "test": v["test"], "val_auc": v["val_auc"]}
                                       for t, v in per_trait.items()}
        traits = {}
        for t in config.TRAITS:
            subset = choice[t]
            entry = self.bundle["fusion"][subset][t]
            fused = sum(w * s["probs"][m][t] for m, w in entry["weights"].items())   # per window
            p = float(np.mean(fused))
            score = float(np.mean(sum(w * s["scores"][m][t] for m, w in entry["weights"].items())))
            traits[t] = {
                "label": TRAIT_LABELS[t], "subset": subset, "weights": entry["weights"],
                "prob": r(p, 4), "percentile": r(percentile(p, entry["quantiles"]), 1),
                "spread": [r(float(fused.min()), 4), r(float(fused.max()), 4)] if len(fused) > 1 else None,
                "score": r(score, 4), "test": entry["test"],
            }
        s["fused"] = traits
        return {
            "available": available, "choice_rule": (
                "all three modalities: late fusion (weighted), the model chosen on validation"
                if len(available) == 3 else
                "a modality is missing: per trait, the subset of the available modalities with the "
                "best validation AUC"),
            "unimodal_prob": {m: {t: r(float(np.mean(v)), 4) for t, v in d.items()} for m, d in s["probs"].items()},
            "traits": traits, "options": options,
            "n_windows": len(s["windows"]), "held_out_clips": self.bundle["held_out_clips"],
        }

    # ---- 8. explanation ------------------------------------------------------------------------
    def _explain(self, s):
        names = self.bundle["audio_feature_names"]
        x = s["audio_X"].mean(axis=0)
        out = {}
        for t in config.TRAITS:
            fused = s["fused"][t]
            contributions = {m: r(w * (float(np.mean(s["probs"][m][t])) - 0.5), 4)
                             for m, w in fused["weights"].items()}
            cue = self.bundle["cue_models"][t]
            z = (x - np.array(cue["mean"])) / np.array(cue["scale"])
            push = np.array(cue["coef"]) * z
            top = []
            for i in np.argsort(-np.abs(push))[:3]:
                name = names[i]
                item = {"feature": name, "label": readable_feature(name), "push": r(push[i], 3),
                        "z": r(z[i], 2), "direction": "up" if push[i] > 0 else "down"}
                if name in s["voice_measures"]:
                    item["percentile"] = s["voice_measures"][name]["percentile"]
                top.append(item)
            out[t] = {"label": TRAIT_LABELS[t], "modality_contribution": contributions,
                      "voice_cues": top,
                      "words": s["text_found"][t] if s.get("text_available") else None}
        s["summary"] = {t: {"label": v["label"], "prob": v["prob"], "percentile": v["percentile"],
                            "subset": v["subset"]} for t, v in s["fused"].items()}
        s["summary"]["_available"] = s["available"]
        return {"traits": out,
                "note": "voice cues come from a readable linear model on the named voice features "
                        "(the cue model of the report), not from inside the classifier"}


# ----------------------------------------------------------------------------
# Replays (the app's demo mode)
# ----------------------------------------------------------------------------
def record_replay(events, path, name, title, description=""):
    """Save an event stream + a browser-playable copy of the input under app/replays/."""
    os.makedirs(config.REPLAYS_DIR, exist_ok=True)
    audio_name = f"{name}.mp3"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-b:a", "96k",
           os.path.join(config.REPLAYS_DIR, audio_name)]
    subprocess.run(cmd, check=True)
    video_name = None
    if ffprobe(path)["has_video"]:
        video_name = f"{name}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "28", "-vf", "scale=-2:360", "-c:a", "aac", "-b:a", "96k",
               os.path.join(config.REPLAYS_DIR, video_name)]
        subprocess.run(cmd, check=True)
    with open(os.path.join(config.REPLAYS_DIR, f"{name}.json"), "w") as f:
        json.dump({"name": name, "title": title, "description": description, "audio": audio_name,
                   "video": video_name, "events": events}, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="audio or video file")
    parser.add_argument("--json", help="write every event to this JSON file")
    parser.add_argument("--transcript", help="use this transcript instead of speech recognition")
    parser.add_argument("--record", metavar="NAME", help="save as a replay for the app's demo mode")
    parser.add_argument("--title", default=None, help="title of the replay")
    parser.add_argument("--description", default="", help="one line shown with the replay")
    args = parser.parse_args()
    if not os.path.exists(args.path):
        sys.exit(f"no such file: {args.path}")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found. Install it first (mac: `brew install ffmpeg`).")

    engine = Engine()
    t0 = time.time()
    engine.load(progress=lambda name: log.info(f"loading {name} ..."))
    log.info(f"models loaded in {time.time() - t0:.1f} s on {engine.device}")
    if args.record:                        # a replay should show warm timings, not the start-up
        engine.warm_up()

    events = []
    for event in engine.run(args.path, transcript=args.transcript):
        events.append(event)
        if event["type"] == "stage":
            log.info(f"  {event['stage']:<15} {event['seconds']:6.2f} s")
        elif event["type"] == "error":
            log.info(f"ERROR in {event['stage']}: {event['message']}")
            sys.exit(1)

    summary = events[-1]["summary"]
    print(f"\nModalities used: {', '.join(summary.pop('_available'))}")
    print(f"{'trait':<22}{'P(high)':>8}{'percentile':>12}   fused from")
    for t, v in summary.items():
        print(f"{v['label']:<22}{v['prob']:>8.2f}{v['percentile']:>11.0f}%   {v['subset']}")
    print("\nApparent personality: how strangers would rate this first impression. "
          "Research prototype - never use it to judge anyone.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(events, f)
        log.info(f"wrote {args.json}")
    if args.record:
        record_replay(events, args.path, args.record, args.title or args.record, args.description)
        log.info(f"saved replay '{args.record}' in {config.REPLAYS_DIR}")


if __name__ == "__main__":
    main()
