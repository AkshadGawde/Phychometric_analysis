"""Phase 2 building blocks: frames -> face crops, sound -> cleaned waveform, transcript -> text_clean.

Moved here from src/02_preprocess.py (the behaviour is unchanged) so that src/predict.py uses
exactly the same steps. See the docstring of src/02_preprocess.py for the reasoning behind
each choice.
"""
import io
import os
import re
import subprocess
import tempfile

import cv2
import librosa
import numpy as np
import soundfile as sf

import config

FACE_DETECTOR = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))


# ----------------------------------------------------------------------------
# VIDEO -> frames
# ----------------------------------------------------------------------------
def grab_frames(video_path, total):
    """Read the video once from the start and keep N_FRAMES evenly spaced frames.
    Returns (frames or None, number of frames that could really be decoded)."""
    wanted = np.linspace(0, total - 1, config.N_FRAMES).astype(int)
    wanted_set = set(wanted.tolist())
    kept = {}
    cap = cv2.VideoCapture(video_path)
    index = 0
    while index <= wanted[-1] and cap.grab():          # grab() = decode without copying (fast)
        if index in wanted_set:
            ok, frame = cap.retrieve()
            if ok:
                kept[index] = frame
        index += 1
    cap.release()
    if all(i in kept for i in wanted):
        return [kept[i] for i in wanted], index
    return None, index


def read_frames(video_path):
    """N_FRAMES evenly spaced frames, or None if the video cannot be decoded."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     # frame count according to the file header
    cap.release()
    if total <= 0:
        total = 10 ** 9                                # header has no count -> find out by reading
    frames, decoded = grab_frames(video_path, total)
    if frames is None and decoded > 0:                 # header was wrong: retry with the true count
        frames, _ = grab_frames(video_path, decoded)
    return frames


def content_box(frames):
    """Box (x0, y0, x1, y1) of the real picture, without black letterbox / pillarbox bars.
    A row or column counts as a bar only if it is (almost) black in ALL frames of the clip."""
    brightest = np.max([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], axis=0)
    height, width = brightest.shape
    cols = np.where(brightest.mean(axis=0) > config.BORDER_BLACK_LEVEL)[0]
    rows = np.where(brightest.mean(axis=1) > config.BORDER_BLACK_LEVEL)[0]
    if len(cols) == 0 or len(rows) == 0:                       # whole clip is dark: keep it all
        return 0, 0, width, height
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    if (x1 - x0) < config.MIN_CONTENT_FRAC * width or (y1 - y0) < config.MIN_CONTENT_FRAC * height:
        return 0, 0, width, height                             # implausible: do not crop
    return int(x0), int(y0), int(x1), int(y1)


def detect_face(frame):
    """Largest face box [x, y, w, h] in full-size pixels, or None if no face is found.
    `frame` is the picture with the black bars already removed."""
    height, width = frame.shape[:2]
    # Detect on a smaller grey copy: much faster and the detector does not need full HD.
    scale = min(1.0, config.FACE_DETECT_SIDE / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    # minSize: boxes narrower than MIN_FACE_FRAC of the picture width are not reported at all.
    min_side = max(1, int(config.MIN_FACE_FRAC * small.shape[1]))
    faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                           minSize=(min_side, min_side))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3]) / scale             # back to full-size pixels


def box_centre(box):
    return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])


def choose_face_boxes(frames):
    """One face box per frame (None = no face -> centre crop).

    A talking-head clip shows one person who stays in roughly the same place. So we take the
    MEDIAN box of the clip; a detection whose centre is further than FACE_OUTLIER_FRAC of the
    picture width from the median centre is almost certainly not the face (e.g. a pattern on
    clothing), and the median box is used for that frame instead.
    Returns (boxes, number of detections that were replaced)."""
    boxes = [detect_face(frame) for frame in frames]
    found = [box for box in boxes if box is not None]
    replaced = 0
    if len(found) >= config.MIN_FACES_FOR_MEDIAN:        # a median of 1-2 boxes says nothing
        median = np.median(found, axis=0)                # median x, y, w, h over the clip
        limit = config.FACE_OUTLIER_FRAC * frames[0].shape[1]
        for i, box in enumerate(boxes):
            if box is not None and np.linalg.norm(box_centre(box) - box_centre(median)) > limit:
                boxes[i] = median
                replaced += 1
    return boxes, replaced


def crop_bounds(frame, box):
    """(x0, y0, x1, y1) of the face box + margin; the centre square if there is no box."""
    height, width = frame.shape[:2]
    if box is not None:
        x, y, w, h = box
        mx, my = config.FACE_MARGIN * w, config.FACE_MARGIN * h
        x0, y0 = int(max(0, x - mx)), int(max(0, y - my))
        x1, y1 = int(min(width, x + w + mx)), int(min(height, y + h + my))
    else:
        side = min(height, width)
        x0, y0 = (width - side) // 2, (height - side) // 2
        x1, y1 = x0 + side, y0 + side
    return x0, y0, x1, y1


def crop_to_box(frame, box):
    """Cut the face box (+ margin) out of the frame; the centre square if there is no box."""
    x0, y0, x1, y1 = crop_bounds(frame, box)
    return cv2.resize(frame[y0:y1, x0:x1], (config.IMG_SIZE, config.IMG_SIZE),
                      interpolation=cv2.INTER_AREA)


def encode_crop(crop):
    """The crop as the jpg bytes Phase 2 writes to disk (Phase 3 reads these jpgs back)."""
    _, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, config.JPG_QUALITY])
    return jpg.tobytes()


# ----------------------------------------------------------------------------
# SPEECH -> cleaned waveform
# ----------------------------------------------------------------------------
def extract_raw_audio(video_path):
    """ffmpeg writes the sound track to a temporary wav, librosa loads it (16 kHz, mono)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav = os.path.join(tmp_dir, "raw.wav")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vn", wav]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(wav):
            raise RuntimeError("no audio track / ffmpeg failed: " + result.stderr.strip()[-150:])
        y, _ = librosa.load(wav, sr=config.SAMPLE_RATE, mono=True)
    return y


def spectral_gate(y, details=None):
    """Noise removal by simple spectral gating (see the docstring of src/02_preprocess.py).
    If a dict is passed as `details`, the noise profile and the mean gain are stored in it."""
    stft = librosa.stft(y, n_fft=512, hop_length=128)
    magnitude = np.abs(stft)
    # 1. Noise profile: average spectrum of the quietest frames of this clip.
    frame_energy = magnitude.sum(axis=0)
    quiet = frame_energy <= np.percentile(frame_energy, config.NOISE_QUIET_PERCENT)
    noise = magnitude[:, quiet].mean(axis=1, keepdims=True)
    # 2. Gain per time-frequency bin: ~1 where the signal is far above the noise, small otherwise.
    gain = 1.0 - config.NOISE_GATE_STRENGTH * noise / (magnitude + 1e-8)
    gain = np.clip(gain, config.NOISE_GATE_FLOOR, 1.0)
    if details is not None:
        details["noise_profile"] = noise[:, 0]
        details["gain_over_time"] = gain.mean(axis=0)            # 1 = untouched, low = gated
        details["gate_hop"] = 128
    return librosa.istft(stft * gain, hop_length=128, length=len(y))


def clean_audio(y, details=None):
    """noise removal -> segmentation (trim silence) -> peak normalisation.
    If a dict is passed as `details`, the intermediate results are stored in it (for the app)."""
    y = spectral_gate(y, details)
    if details is not None:
        details["gated"] = y
    y, (start, end) = librosa.effects.trim(y, top_db=config.TRIM_TOP_DB)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak == 0 or len(y) < config.MIN_AUDIO_SEC * config.SAMPLE_RATE:
        raise RuntimeError("audio is silent or shorter than MIN_AUDIO_SEC after cleaning")
    if details is not None:
        details.update(trim_start=int(start), trim_end=int(end), gain=float(config.PEAK_LEVEL / peak))
    return (config.PEAK_LEVEL * y / peak).astype(np.float32)


def to_wav_bytes(y):
    """The cleaned waveform as the 16-bit wav Phase 2 writes to disk."""
    buffer = io.BytesIO()
    sf.write(buffer, y, config.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def as_saved_wav(y):
    """The waveform after the same 16-bit round trip the training audio went through
    (Phase 2 writes a PCM_16 wav, Phase 4 reads it back)."""
    data, _ = sf.read(io.BytesIO(to_wav_bytes(y)), dtype="float32")
    return data


# ----------------------------------------------------------------------------
# TEXT -> text_clean
# ----------------------------------------------------------------------------
# "like" & co. count as fillers only between punctuation marks (or at the very start/end).
COMMA_FILLER_RE = re.compile(
    r"(^|[,.;:!?])\s*(?:" + "|".join(config.COMMA_FILLERS) + r")\s*(?=[,.;:!?]|$)")


def clean_text(text):
    """Lowercase, remove notes / fillers / punctuation, return space-joined tokens."""
    text = str(text).lower().replace("’", "'")
    text = re.sub(r"\[[^\]]*\]", " ", text)          # transcriber notes: [inaudible 00:00:07]
    text = COMMA_FILLER_RE.sub(r"\1 ", text)         # ", like," -> ","
    text = re.sub(r"[^a-z0-9'\s]", " ", text)        # punctuation -> space (keep don't, it's)
    tokens = [t.strip("'") for t in text.split()]    # whitespace tokenisation
    tokens = [t for t in tokens if t and t not in config.FILLERS]
    return " ".join(tokens)
