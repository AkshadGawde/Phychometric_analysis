"""Phase 2 - Preprocessing of the three modalities.

For every video_id listed in data/splits.csv (and only those):

VIDEO   N_FRAMES evenly spaced frames -> black letterbox / pillarbox bars removed
        -> largest face (OpenCV Haar cascade) cropped with a 20% margin. Face boxes narrower
        than 10% of the picture width are ignored (usually false detections); if no face is
        left, the centre of the picture is used. A detection far away from the clip's median
        face box is replaced by that median box -> resized to IMG_SIZE x IMG_SIZE
        -> data/processed/frames/<video_id>_<k>.jpg
        (ImageNet mean/std normalisation is applied later, in Phase 3.)

SPEECH  ffmpeg extracts the audio -> librosa loads it at 16 kHz mono
        -> noise removal -> silence trimming -> peak normalisation to 0.95
        -> data/processed/audio/<video_id>.wav
        Noise removal choice: SPECTRAL GATING. The quietest 10% of frames give a noise level
        per frequency; frequency bins that are not clearly above it are turned down.
        We chose this over "pre-emphasis + high-pass" because pre-emphasis changes the spectral
        tilt, which would distort the energy / zero-crossing cues of Phase 4 and does not match
        the audio the pretrained speech model was trained on.

TEXT    lowercase -> drop transcriber notes like "[inaudible 00:00:07]" -> drop fillers
        -> strip punctuation and extra spaces -> whitespace tokens.
        "like" is removed only when it is set off by punctuation ("it was, like, great"),
        so the verb in "I like pizza" is kept. Adds `text_clean` and `missing_text` to splits.csv.

Clips that fail (unreadable video, no audio, ...) are logged, written to
data/processed/failed_clips.csv and removed from splits.csv.

Resume: a clip that already has all its frames and its wav is skipped, so a re-run after new
clips were added to splits.csv only processes the new ones.
data/processed/preprocess_log.csv remembers which settings each clip's frames and audio were
made with. If a frame setting (or an audio setting) in config.py changes, the frames (or the
audio) of every clip are made again - but a file is only REWRITTEN when its content really
differs. Unchanged files keep their modification time, so the later phases recompute features
only for the clips that actually changed. --force re-makes everything in the same way.

Run:  python src/02_preprocess.py [--force]
"""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import utils
import config

log = utils.get_logger("02_preprocess")

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


def crop_to_box(frame, box):
    """Cut the face box (+ margin) out of the frame; the centre square if there is no box."""
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
    return cv2.resize(frame[y0:y1, x0:x1], (config.IMG_SIZE, config.IMG_SIZE),
                      interpolation=cv2.INTER_AREA)


def write_if_changed(path, data):
    """Write the bytes `data` to `path` only if they differ from the file that is already there.
    Returns True if the file was written. An untouched file keeps its modification time, which
    tells the later phases that this clip did not change."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            if f.read() == data:
                return False
    with open(path + ".tmp", "wb") as f:               # temporary name first: never a half-written file
        f.write(data)
    os.replace(path + ".tmp", path)
    return True


def process_frames(video_id):
    """Save the face crops of one clip. Returns (frames with a detected face,
    detections replaced by the median box, True if any jpg on disk changed)."""
    frames = read_frames(utils.video_path(video_id))
    if frames is None:
        raise RuntimeError("video could not be decoded")
    x0, y0, x1, y1 = content_box(frames)               # same bars in every frame of a clip
    frames = [frame[y0:y1, x0:x1] for frame in frames]
    boxes, replaced = choose_face_boxes(frames)
    changed = False
    for k, (frame, box) in enumerate(zip(frames, boxes)):
        _, jpg = cv2.imencode(".jpg", crop_to_box(frame, box),
                              [cv2.IMWRITE_JPEG_QUALITY, config.JPG_QUALITY])
        changed |= write_if_changed(utils.frame_path(video_id, k), jpg.tobytes())
    return sum(box is not None for box in boxes), replaced, changed


# ----------------------------------------------------------------------------
# SPEECH -> cleaned wav
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


def spectral_gate(y):
    """Noise removal by simple spectral gating (see the module docstring)."""
    stft = librosa.stft(y, n_fft=512, hop_length=128)
    magnitude = np.abs(stft)
    # 1. Noise profile: average spectrum of the quietest frames of this clip.
    frame_energy = magnitude.sum(axis=0)
    quiet = frame_energy <= np.percentile(frame_energy, config.NOISE_QUIET_PERCENT)
    noise = magnitude[:, quiet].mean(axis=1, keepdims=True)
    # 2. Gain per time-frequency bin: ~1 where the signal is far above the noise, small otherwise.
    gain = 1.0 - config.NOISE_GATE_STRENGTH * noise / (magnitude + 1e-8)
    gain = np.clip(gain, config.NOISE_GATE_FLOOR, 1.0)
    return librosa.istft(stft * gain, hop_length=128, length=len(y))


def clean_audio(y):
    """noise removal -> segmentation (trim silence) -> peak normalisation."""
    y = spectral_gate(y)
    y, _ = librosa.effects.trim(y, top_db=config.TRIM_TOP_DB)
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak == 0 or len(y) < config.MIN_AUDIO_SEC * config.SAMPLE_RATE:
        raise RuntimeError("audio is silent or shorter than MIN_AUDIO_SEC after cleaning")
    return (config.PEAK_LEVEL * y / peak).astype(np.float32)


def process_audio(video_id):
    """Save the cleaned wav of one clip.
    Returns (raw seconds, cleaned seconds, True if the wav on disk changed)."""
    raw = extract_raw_audio(utils.video_path(video_id))
    clean = clean_audio(raw)
    buffer = io.BytesIO()
    sf.write(buffer, clean, config.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    changed = write_if_changed(utils.audio_path(video_id), buffer.getvalue())
    return len(raw) / config.SAMPLE_RATE, len(clean) / config.SAMPLE_RATE, changed


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


# ----------------------------------------------------------------------------
# Report figures
# ----------------------------------------------------------------------------
def figure_sample_frames(plt, sample_ids):
    """Grid: one row per sample clip, one column per extracted frame."""
    rows, cols = len(sample_ids), config.N_FRAMES
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.95 * rows), squeeze=False)
    for r, video_id in enumerate(sample_ids):
        for k in range(cols):
            ax = axes[r, k]
            image = cv2.cvtColor(cv2.imread(utils.frame_path(video_id, k)), cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == rows - 1:                      # columns line up, so label them once
                ax.set_xlabel(f"frame {k}", fontsize=8, color=utils.PLOT["muted"])
        axes[r, 0].set_title(f"clip {video_id}", fontsize=10)
    fig.suptitle(f"Phase 2 output: {cols} evenly spaced face crops per clip "
                 f"({config.IMG_SIZE}×{config.IMG_SIZE} px)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(h_pad=1.5)
    fig.savefig(os.path.join(config.RESULTS_DIR, "sample_frames.png"))
    plt.close(fig)


def figure_sample_waveform(plt, video_id):
    """Raw vs cleaned audio of one clip, same time and amplitude scale."""
    raw = extract_raw_audio(utils.video_path(video_id))
    clean, _ = librosa.load(utils.audio_path(video_id), sr=config.SAMPLE_RATE, mono=True)
    panels = [
        ("Raw audio from the video (16 kHz mono)", raw, utils.PLOT["series"][0]),
        ("Cleaned: spectral gating, silence trimmed, peak-normalised "
         f"to {config.PEAK_LEVEL}", clean, utils.PLOT["series"][1]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 4.6), sharex=True, sharey=True)
    for ax, (title, y, color) in zip(axes, panels):
        seconds = np.arange(len(y)) / config.SAMPLE_RATE
        ax.plot(seconds, y, color=color, linewidth=0.6)
        ax.set_title(f"{title}  —  {len(y) / config.SAMPLE_RATE:.1f} s")
        ax.set_ylabel("amplitude")
        ax.set_ylim(-1.05, 1.05)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Audio preprocessing, clip {video_id}", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "sample_waveform.png"))
    plt.close(fig)


def figure_text_length(plt, df):
    """Histogram of transcript length in words (after cleaning)."""
    n_words = df["text_clean"].str.split().str.len()
    median = float(n_words.median())
    bin_width = 5
    bins = np.arange(0, n_words.max() + 2 * bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(n_words, bins=bins, color=utils.PLOT["series"][0],
            edgecolor=utils.PLOT["surface"], linewidth=1.5)      # surface-coloured gap between bars
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)                                           # room for the median label
    ax.axvline(median, color=utils.PLOT["ink2"], linewidth=1)
    ax.annotate(f"median {median:.0f} words", xy=(median, 1), xycoords=("data", "axes fraction"),
                xytext=(5, -4), textcoords="offset points", va="top",
                fontsize=9, color=utils.PLOT["ink2"])
    ax.set_xlabel(f"words per transcript after cleaning (bin width {bin_width})")
    ax.set_ylabel("number of clips")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_title(f"Transcript length  —  {len(df)} clips, "
                 f"{df['missing_text'].mean() * 100:.1f}% without transcript")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "text_length_hist.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
LOG_COLUMNS = ["video_id", "faces_found", "faces_replaced", "raw_sec", "clean_sec",
               "frames_settings", "audio_settings"]


def settings_id(settings):
    """Short fingerprint of a settings dict: it changes whenever any value in it changes."""
    return hashlib.md5(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]


def frames_settings_id():
    """Everything that decides what is inside the frame files."""
    return settings_id({
        "code_version": 3,            # raise when the frame-processing code itself changes
        "n_frames": config.N_FRAMES, "img_size": config.IMG_SIZE, "jpg_quality": config.JPG_QUALITY,
        "face_margin": config.FACE_MARGIN, "face_detect_side": config.FACE_DETECT_SIDE,
        "min_face_frac": config.MIN_FACE_FRAC, "face_outlier_frac": config.FACE_OUTLIER_FRAC,
        "min_faces_for_median": config.MIN_FACES_FOR_MEDIAN,
        "border_black_level": config.BORDER_BLACK_LEVEL, "min_content_frac": config.MIN_CONTENT_FRAC,
    })


def audio_settings_id():
    """Everything that decides what is inside the wav files."""
    return settings_id({
        "code_version": 1,            # raise when the audio-processing code itself changes
        "sample_rate": config.SAMPLE_RATE, "noise_quiet_percent": config.NOISE_QUIET_PERCENT,
        "noise_gate_strength": config.NOISE_GATE_STRENGTH, "noise_gate_floor": config.NOISE_GATE_FLOOR,
        "trim_top_db": config.TRIM_TOP_DB, "peak_level": config.PEAK_LEVEL,
        "min_audio_sec": config.MIN_AUDIO_SEC,
    })


def load_log():
    """preprocess_log.csv as {video_id: row}. Each row holds the clip's statistics and the
    settings fingerprints its frames and audio were made with."""
    if not os.path.exists(config.PREPROCESS_LOG_CSV):
        return {}
    table = pd.read_csv(config.PREPROCESS_LOG_CSV, dtype=str).reindex(columns=LOG_COLUMNS)
    table = table.drop_duplicates("video_id", keep="last")
    table.to_csv(config.PREPROCESS_LOG_CSV, index=False)     # one tidy row per clip
    return {row["video_id"]: row for row in table.to_dict("records")}


def append_log_row(row):
    new_file = not os.path.exists(config.PREPROCESS_LOG_CSV)
    pd.DataFrame([row]).reindex(columns=LOG_COLUMNS).to_csv(
        config.PREPROCESS_LOG_CSV, mode="a", header=new_file, index=False)


def main():
    force = "--force" in sys.argv
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found. Install it first (mac: `brew install ffmpeg`).")
    if FACE_DETECTOR.empty():
        sys.exit("OpenCV Haar cascade could not be loaded - install `opencv-python<5`.")

    df = utils.load_splits()
    ids = df["video_id"].tolist()

    # ---- which clips of splits.csv still need work? ----
    state = load_log()
    frames_id, audio_id = frames_settings_id(), audio_settings_id()

    def frames_ok(video_id):
        """All frames exist and were made with the current frame settings."""
        return (all(os.path.exists(utils.frame_path(video_id, k)) for k in range(config.N_FRAMES))
                and state.get(video_id, {}).get("frames_settings") == frames_id)

    def audio_ok(video_id):
        return (os.path.exists(utils.audio_path(video_id))
                and state.get(video_id, {}).get("audio_settings") == audio_id)

    todo = [v for v in ids if force or not (frames_ok(v) and audio_ok(v))]
    log.info(f"{len(ids)} clips in splits.csv: {len(ids) - len(todo)} up to date, "
             f"{len(todo)} to process")

    failures = []                                     # (video_id, reason)
    frames_changed, audio_changed = [], []            # clips whose files on disk really changed
    for video_id in tqdm(todo, desc="frames + audio"):
        row = dict(state.get(video_id, {}), video_id=video_id)
        try:
            if not os.path.exists(utils.video_path(video_id)):
                raise RuntimeError("video file not found in data/raw/videos/")
            if force or not frames_ok(video_id):
                found, replaced, changed = process_frames(video_id)
                row.update(faces_found=found, faces_replaced=replaced, frames_settings=frames_id)
                if changed:
                    frames_changed.append(video_id)
            if force or not audio_ok(video_id):
                raw_sec, clean_sec, changed = process_audio(video_id)
                row.update(raw_sec=round(raw_sec, 2), clean_sec=round(clean_sec, 2),
                           audio_settings=audio_id)
                if changed:
                    audio_changed.append(video_id)
        except Exception as error:                    # one bad clip must not stop the whole run
            failures.append((video_id, str(error)))
            tqdm.write(f"  FAILED {video_id}: {error}")
        state[video_id] = row                         # remember what was finished
        append_log_row(row)
    log.info(f"frames + audio: {len(todo) - len(failures)} processed now, {len(failures)} failed")
    log.info(f"files that really changed on disk: frames of {len(frames_changed)} clip(s), "
             f"audio of {len(audio_changed)} clip(s)")
    if frames_changed:
        log.info(f"  frames changed: {frames_changed[:10]}{' ...' if len(frames_changed) > 10 else ''}")

    # ---- drop every clip that does not have all frames + its wav ----
    complete = df["video_id"].map(lambda v: frames_ok(v) and audio_ok(v))
    known_reasons = dict(failures)
    failed = pd.DataFrame({"video_id": df.loc[~complete, "video_id"]})
    failed["reason"] = failed["video_id"].map(lambda v: known_reasons.get(v, "frames or wav missing"))
    failed.to_csv(config.FAILED_CLIPS_CSV, index=False)
    df = df[complete].reset_index(drop=True)
    if df.empty:
        sys.exit("No clip could be preprocessed - check data/raw/videos/ and ffmpeg.")

    # ---- text ----
    df["text_clean"] = df["text"].map(clean_text)
    df["missing_text"] = (df["text_clean"] == "").astype(int)
    utils.save_splits(df)

    # ---- report figures ----
    plt = utils.setup_plots()
    sample_ids = (df["video_id"].sample(n=min(config.N_SAMPLE_CLIPS, len(df)),
                                        random_state=config.SEED).tolist())
    figure_sample_frames(plt, sample_ids)
    figure_sample_waveform(plt, sample_ids[0])
    figure_text_length(plt, df)
    log.info(f"figures saved in {config.RESULTS_DIR}: sample_frames.png, sample_waveform.png, "
             f"text_length_hist.png")

    # ---- summary ----
    n_frames = sum(os.path.exists(utils.frame_path(v, k))
                   for v in df["video_id"] for k in range(config.N_FRAMES))
    n_wavs = sum(os.path.exists(utils.audio_path(v)) for v in df["video_id"])
    log.info(f"clips kept: {len(df)}   dropped: {len(failed)}")
    log.info(f"  frames: {n_frames} jpg ({config.N_FRAMES} per clip)")
    log.info(f"  audio : {n_wavs} wav")
    log.info(f"  text  : {int((df.missing_text == 0).sum())} with transcript, "
             f"{int(df.missing_text.sum())} missing "
             f"({df.missing_text.mean() * 100:.1f}% missing transcripts)")
    for _, row in failed.iterrows():
        log.info(f"  dropped {row.video_id}: {row.reason}")

    stats = pd.DataFrame([state[v] for v in df["video_id"]]).reindex(columns=LOG_COLUMNS)
    found = pd.to_numeric(stats.faces_found)
    replaced = pd.to_numeric(stats.faces_replaced)
    total = len(stats) * config.N_FRAMES
    log.info(f"  faces : detected in {int(found.sum())} of {total} frames "
             f"({found.sum() / total * 100:.1f}%), centre crop used for the rest; "
             f"{int((found == 0).sum())} clip(s) with no face at all")
    log.info(f"  faces : {int(replaced.sum())} detection(s) in {int((replaced > 0).sum())} clip(s) "
             f"were far from the clip's median face box and replaced by it")
    log.info(f"  audio length: mean {pd.to_numeric(stats.raw_sec).mean():.1f} s raw -> "
             f"{pd.to_numeric(stats.clean_sec).mean():.1f} s after trimming silence")

    # ---- acceptance check ----
    if n_frames != len(df) * config.N_FRAMES or n_wavs != len(df):
        log.info("ACCEPTANCE FAILED: some clips in splits.csv lack frames or audio")
        sys.exit(1)
    log.info(f"ACCEPTANCE CHECK PASSED: every clip in splits.csv has {config.N_FRAMES} frames + 1 wav")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
