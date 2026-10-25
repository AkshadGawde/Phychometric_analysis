"""Parity test: the live app must turn a raw clip into EXACTLY the features training used.

For N test clips:

1. CODE parity (exact): the voice features computed by src/inference/ from the clip's stored
   cleaned wav (data/processed/audio) equal the stored row in features/audio_mfcc.npy /
   audio_emotion.npy.
2. END-TO-END parity: the raw .mp4 goes through src/predict.py (with the dataset's own
   transcript, so speech recognition is not involved). Every feature row is compared with the
   stored row, and the single-modality predictions with the saved models applied to the stored
   rows.
   Face and text features match exactly. Voice features may move a tiny bit when the installed
   ffmpeg is not the one Phase 2 ran with: ffmpeg 9 decodes the AAC sound track a few samples
   shorter than the version the stored wavs were made with, which shifts the silence trimming by
   ~2 ms. So voice features are compared in units of their standard deviation over the clips
   (must stay below MAX_Z_DIFF), and what really matters - the predicted probabilities - must agree
   to PROB_ATOL.

A failure here means the app silently scores something else than the reported models.

Run:  python tests/test_parity.py [N]        (default N = 20; about 5 s per clip)
"""
import os
import sys

import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import predict  # noqa: E402
import utils  # noqa: E402
import config  # noqa: E402

FEATURES = ["visual_cnn", "visual_emotion", "audio_mfcc", "audio_emotion", "text_minilm", "text_emotion"]
# Same numbers up to float noise of the GPU (batch of 8 clips in training vs one clip here).
RTOL, ATOL = 1e-3, 2e-4
MAX_Z_DIFF = 0.02            # voice features end to end: < 2% of a standard deviation
PROB_ATOL = 2e-3
VOICE = {"audio_mfcc", "audio_emotion"}


def main(n_clips=20):
    df = utils.load_splits()
    ids = utils.read_ids()
    row_of = {v: i for i, v in enumerate(ids)}
    stored = utils.load_features(FEATURES)
    clips = df[df["split"] == "test"].head(n_clips)

    engine = predict.Engine()
    engine.load()
    spread = {name: stored[name].std(axis=0) + 1e-8 for name in VOICE}
    worst = {name: 0.0 for name in FEATURES}
    worst_code, worst_prob = 0.0, 0.0
    failures = []
    for clip in clips.itertuples():
        i = row_of[clip.video_id]
        # 1. code parity: stored wav -> voice features
        y, _ = librosa.load(utils.audio_path(clip.video_id), sr=config.SAMPLE_RATE, mono=True)
        y, _ = predict.audio.prepare_waveform(y)
        code = {"audio_mfcc": predict.audio.handcrafted_features(y), "audio_emotion": engine.wav2vec2(y)[1]}
        for name, ours in code.items():
            diff = float(np.max(np.abs(np.nan_to_num(ours) - stored[name][i])))
            worst_code = max(worst_code, diff)
            if not np.allclose(np.nan_to_num(ours), stored[name][i], rtol=RTOL, atol=ATOL):
                failures.append(f"{clip.video_id} {name} (code parity): max |diff| {diff:.2e}")

        # 2. end to end: raw mp4 -> app
        collect = {}
        transcript = "" if clip.missing_text else clip.text
        events = list(engine.run(utils.video_path(clip.video_id), transcript=transcript, collect=collect))
        if events[-1]["type"] != "done":
            failures.append(f"{clip.video_id}: {events[-1]}")
            continue
        for name in FEATURES:
            ours, theirs = np.nan_to_num(collect[name][0]), stored[name][i]
            if name in VOICE:
                diff = float(np.max(np.abs(ours - theirs) / spread[name]))
                ok = diff <= MAX_Z_DIFF
            else:
                diff = float(np.max(np.abs(ours - theirs)))
                ok = np.allclose(ours, theirs, rtol=RTOL, atol=ATOL)
            worst[name] = max(worst[name], diff)
            if not ok:
                failures.append(f"{clip.video_id} {name}: difference {diff:.2e}")
        # predictions from the stored rows vs from the app's rows
        unimodal = next(e for e in events if e["type"] == "stage" and e["stage"] == "unimodal")["data"]["prob"]
        for modality in predict.MODALITIES:
            X = np.hstack([stored[n][i] for n in config.MODALITIES[modality]])[None]
            for trait in config.TRAITS:
                expected = predict.modeling.positive_probability(engine.classifiers[modality, trait], X)[0]
                worst_prob = max(worst_prob, abs(unimodal[modality][trait] - expected))
        print(f"  {clip.video_id}: ok")

    print(f"\n1. code parity, voice features from the stored wav: max |diff| {worst_code:.2e}")
    print("2. end to end, raw clip -> app (voice: in standard deviations; others: absolute):")
    for name, value in worst.items():
        print(f"  {name:<15} {value:.2e}" + ("  SD" if name in VOICE else ""))
    print(f"  {'P(high)':<15} {worst_prob:.2e}   (tolerance {PROB_ATOL})")
    if worst_prob > PROB_ATOL:
        failures.append(f"predicted probabilities differ by up to {worst_prob:.4f}")
    if failures:
        print("\nPARITY TEST FAILED:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print(f"\nPARITY TEST PASSED: {len(clips)} raw test clips -> the same features and predictions "
          f"as the stored training rows")


def test_parity():          # pytest entry point
    main(int(os.environ.get("PARITY_CLIPS", 5)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
