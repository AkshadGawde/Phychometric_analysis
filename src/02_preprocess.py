"""Phase 2 - Preprocessing of the three modalities.

For every video_id listed in data/splits.csv (and only those):

VIDEO   N_FRAMES evenly spaced frames -> black letterbox / pillarbox bars removed
        -> largest face (OpenCV Haar cascade) cropped with a 20% margin. Face boxes narrower
        than 10% of the picture width are ignored (usually false detections); if no face is
        left, the centre of the picture is used -> resized to IMG_SIZE x IMG_SIZE
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
clips were added to splits.csv only processes the new ones. The settings used are stored in
data/processed/preprocess_settings.json; if a Phase 2 setting in config.py changes, every clip
is redone automatically. --force redoes everything as well.

Run:  python src/02_preprocess.py [--force]
"""
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


def crop_face(frame):
    """Crop the largest detected face (+ margin); centre crop if none. Returns (image, found).
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

    found = len(faces) > 0
    if found:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3]) / scale     # back to full-size pixels
        mx, my = config.FACE_MARGIN * w, config.FACE_MARGIN * h
        x0, y0 = int(max(0, x - mx)), int(max(0, y - my))
        x1, y1 = int(min(width, x + w + mx)), int(min(height, y + h + my))
    else:
        side = min(height, width)
        x0, y0 = (width - side) // 2, (height - side) // 2
        x1, y1 = x0 + side, y0 + side
    crop = cv2.resize(frame[y0:y1, x0:x1], (config.IMG_SIZE, config.IMG_SIZE),
                      interpolation=cv2.INTER_AREA)
    return crop, found


def process_frames(video_id):
    """Save the face crops of one clip. Returns the number of frames in which a face was found."""
    frames = read_frames(utils.video_path(video_id))
    if frames is None:
        raise RuntimeError("video could not be decoded")
    x0, y0, x1, y1 = content_box(frames)               # same bars in every frame of a clip
    faces_found = 0
    for k, frame in enumerate(frames):
        crop, found = crop_face(frame[y0:y1, x0:x1])
        faces_found += int(found)
        # Write to a temporary name first, so an interrupted run never leaves a half-written jpg.
        tmp = utils.frame_path(video_id, k).replace(".jpg", ".tmp.jpg")
        cv2.imwrite(tmp, crop, [cv2.IMWRITE_JPEG_QUALITY, config.JPG_QUALITY])
        os.replace(tmp, utils.frame_path(video_id, k))
    return faces_found


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
    """Save the cleaned wav of one clip. Returns (raw seconds, cleaned seconds)."""
    raw = extract_raw_audio(utils.video_path(video_id))
    clean = clean_audio(raw)
    tmp = utils.audio_path(video_id).replace(".wav", ".tmp.wav")
    sf.write(tmp, clean, config.SAMPLE_RATE, subtype="PCM_16")
    os.replace(tmp, utils.audio_path(video_id))
    return len(raw) / config.SAMPLE_RATE, len(clean) / config.SAMPLE_RATE


def audio_done(video_id):
    return os.path.exists(utils.audio_path(video_id))


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
def preprocess_settings():
    """Every setting that changes what is inside the frame / wav files."""
    return {
        "code_version": 2,            # raise when the processing code itself changes
        "n_frames": config.N_FRAMES, "img_size": config.IMG_SIZE,
        "face_margin": config.FACE_MARGIN, "face_detect_side": config.FACE_DETECT_SIDE,
        "min_face_frac": config.MIN_FACE_FRAC, "border_black_level": config.BORDER_BLACK_LEVEL,
        "min_content_frac": config.MIN_CONTENT_FRAC, "jpg_quality": config.JPG_QUALITY,
        "sample_rate": config.SAMPLE_RATE, "noise_quiet_percent": config.NOISE_QUIET_PERCENT,
        "noise_gate_strength": config.NOISE_GATE_STRENGTH, "noise_gate_floor": config.NOISE_GATE_FLOOR,
        "trim_top_db": config.TRIM_TOP_DB, "peak_level": config.PEAK_LEVEL,
        "min_audio_sec": config.MIN_AUDIO_SEC,
    }


def settings_timestamp(force):
    """Write preprocess_settings.json if the settings changed (or --force) and return its
    modification time. Frames / wav files older than that time are redone."""
    current, stored = preprocess_settings(), None
    if os.path.exists(config.PREPROCESS_SETTINGS_JSON):
        with open(config.PREPROCESS_SETTINGS_JSON) as f:
            stored = json.load(f)
    if force or stored != current:
        if os.listdir(config.FRAMES_DIR) or os.listdir(config.AUDIO_DIR):
            log.info("--force given: redoing all clips" if force else
                     "Phase 2 settings changed since the cached files were made: redoing all clips")
        with open(config.PREPROCESS_SETTINGS_JSON, "w") as f:
            json.dump(current, f, indent=2)
    return os.path.getmtime(config.PREPROCESS_SETTINGS_JSON)


def clip_is_current(video_id, since):
    """True if the clip has all its frames + wav and they were made with the current settings."""
    paths = [utils.frame_path(video_id, k) for k in range(config.N_FRAMES)]
    paths.append(utils.audio_path(video_id))
    return all(os.path.exists(p) and os.path.getmtime(p) >= since for p in paths)


def append_log_row(row):
    """One line per finished clip: face-detection and audio-length statistics."""
    new_file = not os.path.exists(config.PREPROCESS_LOG_CSV)
    pd.DataFrame([row]).to_csv(config.PREPROCESS_LOG_CSV, mode="a", header=new_file, index=False)


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

    # ---- frames + audio: only the clips of splits.csv that are not finished yet ----
    since = settings_timestamp(force)
    todo = [v for v in ids if not clip_is_current(v, since)]
    log.info(f"{len(ids)} clips in splits.csv: {len(ids) - len(todo)} already preprocessed, "
             f"{len(todo)} to process")
    failures = []                                     # (video_id, reason)
    for video_id in tqdm(todo, desc="frames + audio"):
        try:
            if not os.path.exists(utils.video_path(video_id)):
                raise RuntimeError("video file not found in data/raw/videos/")
            faces_found = process_frames(video_id)
            raw_sec, clean_sec = process_audio(video_id)
            append_log_row({"video_id": video_id, "faces_found": faces_found,
                            "raw_sec": round(raw_sec, 2), "clean_sec": round(clean_sec, 2)})
        except Exception as error:                    # one bad clip must not stop the whole run
            failures.append((video_id, str(error)))
            tqdm.write(f"  FAILED {video_id}: {error}")
    log.info(f"frames + audio: {len(todo) - len(failures)} processed now, {len(failures)} failed")

    # ---- drop every clip that does not have all frames + its wav ----
    complete = df["video_id"].map(lambda v: clip_is_current(v, since))
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
    n_wavs = sum(audio_done(v) for v in df["video_id"])
    log.info(f"clips kept: {len(df)}   dropped: {len(failed)}")
    log.info(f"  frames: {n_frames} jpg ({config.N_FRAMES} per clip)")
    log.info(f"  audio : {n_wavs} wav")
    log.info(f"  text  : {int((df.missing_text == 0).sum())} with transcript, "
             f"{int(df.missing_text.sum())} missing "
             f"({df.missing_text.mean() * 100:.1f}% missing transcripts)")
    for _, row in failed.iterrows():
        log.info(f"  dropped {row.video_id}: {row.reason}")

    if os.path.exists(config.PREPROCESS_LOG_CSV):
        stats = pd.read_csv(config.PREPROCESS_LOG_CSV, dtype={"video_id": str})
        stats = stats.drop_duplicates("video_id", keep="last")
        stats = stats[stats.video_id.isin(df.video_id)]
        if len(stats):
            log.info(f"  faces : found in {stats.faces_found.sum()} of {len(stats) * config.N_FRAMES} "
                     f"frames ({stats.faces_found.sum() / (len(stats) * config.N_FRAMES) * 100:.1f}%), "
                     f"centre crop used for the rest; "
                     f"{int((stats.faces_found == 0).sum())} clip(s) with no face at all")
            log.info(f"  audio length: mean {stats.raw_sec.mean():.1f} s raw -> "
                     f"{stats.clean_sec.mean():.1f} s after trimming silence")

    # ---- acceptance check ----
    if n_frames != len(df) * config.N_FRAMES or n_wavs != len(df):
        log.info("ACCEPTANCE FAILED: some clips in splits.csv lack frames or audio")
        sys.exit(1)
    log.info(f"ACCEPTANCE CHECK PASSED: every clip in splits.csv has {config.N_FRAMES} frames + 1 wav")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
