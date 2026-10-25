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
import shutil
import sys

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import utils
import config
from inference.preprocess import (FACE_DETECTOR, choose_face_boxes, clean_audio,  # noqa: F401
                                  clean_text, content_box, crop_to_box, extract_raw_audio,
                                  read_frames)

log = utils.get_logger("02_preprocess")


# ----------------------------------------------------------------------------
# Writing the frames and the cleaned wav (the processing steps themselves are in
# src/inference/preprocess.py, shared with the app)
# ----------------------------------------------------------------------------
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
