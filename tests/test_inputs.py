"""Phase 1 acceptance: src/predict.py on the kinds of input the app will get.

1. mp3: the sound track of test clips, exported to mp3, gives (almost) the same voice
   predictions as the stored rows (mp3 compression changes the audio a little; the difference
   is printed, not hidden).
2. a 2-minute mp3 runs through, cut into windows, with a spread per trait.
3. a silent file and a corrupt file end with a clean `error` event, not a traceback.

Run:  python tests/test_inputs.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import predict  # noqa: E402
import utils  # noqa: E402
import config  # noqa: E402


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def stage(events, name):
    return next(e for e in events if e["type"] == "stage" and e["stage"] == name)["data"]


def main():
    df = utils.load_splits()
    ids = utils.read_ids()
    stored = utils.load_features(config.MODALITIES["audio"])
    engine = predict.Engine()
    engine.load()
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        # 1. mp3 of five test clips
        worst = 0.0
        clips = df[df["split"] == "test"].head(5)
        for clip in clips.itertuples():
            mp3 = os.path.join(tmp, "clip.mp3")
            ffmpeg("-i", utils.video_path(clip.video_id), "-vn", "-b:a", "128k", mp3)
            events = list(engine.run(mp3, transcript=""))
            if events[-1]["type"] != "done":
                failures.append(f"mp3 {clip.video_id}: {events[-1]}")
                continue
            probs = stage(events, "unimodal")["prob"]["audio"]
            i = ids.index(clip.video_id)
            X = np.hstack([stored[n][i] for n in config.MODALITIES["audio"]])[None]
            for trait in config.TRAITS:
                expected = predict.modeling.positive_probability(engine.classifiers["audio", trait], X)[0]
                worst = max(worst, abs(probs[trait] - expected))
            if stage(events, "visual")["available"]:
                failures.append("mp3 input was treated as video")
        print(f"1. mp3 (128 kbit/s) vs stored rows: voice P(high) differs by at most {worst:.3f}")
        if worst > 0.05:
            failures.append(f"mp3 voice predictions differ by {worst:.3f} (> 0.05)")

        # 2. two minutes: four test clips' sound tracks one after the other
        parts = [utils.video_path(v) for v in df[df["split"] == "test"]["video_id"].iloc[5:13]]
        listing = os.path.join(tmp, "list.txt")
        wavs = []
        for k, part in enumerate(parts):
            wav = os.path.join(tmp, f"part{k}.wav")
            ffmpeg("-i", part, "-vn", "-ac", "1", "-ar", "16000", wav)
            wavs.append(wav)
        with open(listing, "w") as f:
            f.writelines(f"file '{w}'\n" for w in wavs)
        long_mp3 = os.path.join(tmp, "long.mp3")
        ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-t", "120", long_mp3)
        events = list(engine.run(long_mp3))
        if events[-1]["type"] != "done":
            failures.append(f"2-minute mp3: {events[-1]}")
        else:
            windows = stage(events, "audio_features")["windows"]
            spread = stage(events, "fusion")["traits"]["extraversion"]["spread"]
            print(f"2. 2-minute mp3: {len(windows)} windows, extraversion spread {spread}, "
                  f"{events[-1]['seconds']:.1f} s")
            if len(windows) < 2 or spread is None:
                failures.append("long input was not windowed")

        # 3. bad inputs
        silent = os.path.join(tmp, "silent.wav")
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "5", silent)
        corrupt = os.path.join(tmp, "corrupt.mp3")
        with open(corrupt, "wb") as f:
            f.write(os.urandom(4096))
        for name, path in [("silent", silent), ("corrupt", corrupt)]:
            last = list(engine.run(path))[-1]
            print(f"3. {name} file -> {last['type']}: {last.get('message')}")
            if last["type"] != "error":
                failures.append(f"{name} file did not give an error event")

    if failures:
        print("\nINPUT TESTS FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nINPUT TESTS PASSED")


def test_inputs():          # pytest entry point
    main()


if __name__ == "__main__":
    main()
