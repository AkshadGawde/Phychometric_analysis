"""Phase 4 building blocks: cleaned waveform -> hand-crafted features + wav2vec2 emotion.

Moved here from src/04_features_audio.py (the behaviour is unchanged) so that src/predict.py
uses exactly the same steps. See the docstring of src/04_features_audio.py for the details.
"""
import sys

import librosa
import numpy as np
import torch

import utils
import config

MFCC_COLUMNS = ([f"mfcc{i}_mean" for i in range(1, config.N_MFCC + 1)]
                + [f"mfcc{i}_std" for i in range(1, config.N_MFCC + 1)]
                + ["pitch_mean", "pitch_std", "voiced_fraction",
                   "rms_mean", "rms_std", "zcr_mean", "nonsilent_fraction"])


def prepare_waveform(y):
    """Make a cleaned waveform safe to analyse. Returns (waveform, list of problem notes)."""
    notes = []
    if len(y) < config.MIN_FEATURE_SAMPLES:
        notes.append(f"very short ({len(y) / config.SAMPLE_RATE:.2f} s), padded with silence")
        y = np.pad(y, (0, config.MIN_FEATURE_SAMPLES - len(y)))
    if np.max(np.abs(y)) < config.SILENCE_PEAK:
        notes.append("near-silent")
    return y, notes


def handcrafted_features(y, frames=None):
    """The 87 numbers described in src/04_features_audio.py, for one waveform.
    If a dict is passed as `frames`, the per-frame curves (pitch, voicing, loudness) are stored
    in it (for the app)."""
    sr, frame, hop = config.SAMPLE_RATE, config.AUDIO_FRAME_LENGTH, config.AUDIO_HOP_LENGTH

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=frame, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)[0]

    # Pitch: f0 is NaN in unvoiced frames, `voiced` says which frames are voiced.
    f0, voiced, _ = librosa.pyin(y, fmin=config.PITCH_FMIN, fmax=config.PITCH_FMAX, sr=sr,
                                 frame_length=frame, hop_length=hop)
    has_pitch = bool(np.any(np.isfinite(f0)))
    pitch_mean = float(np.nanmean(f0)) if has_pitch else 0.0       # 0 if the clip is all unvoiced
    pitch_std = float(np.nanstd(f0)) if has_pitch else 0.0

    # Speaking-rate proxy: frames whose loudness is within TRIM_TOP_DB of the loudest frame.
    loudest = rms.max()
    if loudest > config.SILENCE_PEAK:
        level_db = 20 * np.log10(np.maximum(rms, 1e-10) / loudest)
        nonsilent_fraction = float(np.mean(level_db > -config.TRIM_TOP_DB))
    else:
        nonsilent_fraction = 0.0                                   # the whole clip is silence

    if frames is not None:
        frames.update(f0=f0, voiced=voiced, rms=rms, hop=hop)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        [pitch_mean, pitch_std, float(np.mean(voiced)),
         float(rms.mean()), float(rms.std()), float(zcr.mean()), nonsilent_fraction],
    ]).astype(np.float32)


def build_wav2vec2(device):
    """Returns (analyse, labels). analyse(waveform) -> (embedding (1024,), probabilities (8,))."""
    import transformers
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2Model

    name = config.SPEECH_EMOTION_MODEL
    transformers.logging.set_verbosity_error()        # the "unused head weights" notice is expected
    extractor = AutoFeatureExtractor.from_pretrained(name)     # normalises the waveform for the model
    encoder = utils.load_pretrained(Wav2Vec2Model, name).eval().to(device)
    model_config = AutoConfig.from_pretrained(name)
    labels = [model_config.id2label[i] for i in range(len(model_config.id2label))]

    # Rebuild the emotion head exactly as it is stored in the checkpoint (see module docstring).
    with safe_open(hf_hub_download(name, "model.safetensors"), framework="pt") as f:
        head_keys = ["classifier.dense.weight", "classifier.dense.bias",
                     "classifier.output.weight", "classifier.output.bias"]
        if not all(key in f.keys() for key in head_keys):
            sys.exit(f"{name}: the checkpoint no longer contains the expected emotion head "
                     f"({head_keys}). Refusing to continue with a random head.")
        weights = {key: f.get_tensor(key) for key in head_keys}
    dense = torch.nn.Linear(*reversed(weights["classifier.dense.weight"].shape))
    output = torch.nn.Linear(*reversed(weights["classifier.output.weight"].shape))
    dense.load_state_dict({"weight": weights["classifier.dense.weight"],
                           "bias": weights["classifier.dense.bias"]})
    output.load_state_dict({"weight": weights["classifier.output.weight"],
                            "bias": weights["classifier.output.bias"]})
    dense, output = dense.eval().to(device), output.eval().to(device)
    if output.out_features != len(labels):
        sys.exit(f"{name}: head has {output.out_features} outputs but config lists {len(labels)} labels.")

    @torch.no_grad()
    def analyse(y):
        y = y[:int(config.W2V_MAX_SEC * config.SAMPLE_RATE)]
        inputs = extractor(y, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
        hidden = encoder(inputs["input_values"].to(device)).last_hidden_state   # (1, time, 1024)
        pooled = hidden.mean(dim=1)                                             # mean over time
        logits = output(torch.tanh(dense(pooled)))                              # one logit per class
        probabilities = torch.softmax(logits, dim=-1)                           # ALL classes
        return pooled[0].cpu().numpy(), probabilities[0].cpu().numpy()

    return analyse, labels
