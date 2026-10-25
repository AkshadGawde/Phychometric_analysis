"""Phase 10 - Inference bundle for the live app (src/predict.py, app/server.py).

Nothing is trained here: the fitted single-modality models of Phase 6 are loaded from
results/models/ and applied to the stored features, to collect what the app needs but the
research phases only computed on the fly:

  fusion      late-fusion weights per trait = validation AUC of each modality (as in
              modeling.late_fusion), for EVERY subset of the modalities, with the subset's
              validation AUC and test metrics (= the ablation of Phase 8);
  default     which subset to use when a modality is missing (face not filmed, no speech):
              all three -> the validation choice of Phase 7 (late fusion, weighted);
              otherwise the available subset with the best VALIDATION AUC, per trait;
  quantiles   101 quantiles of the fused "high" probability on the HELD-OUT clips (val + test)
              per subset and trait -> percentiles for a new person. Held-out, not train: the
              train predictions are over-confident (train F1 up to 0.93), a new person is not;
  voice       101 quantiles of the readable voice measures (pitch, loudness, ...) on all clips;
  cues        a readable linear cue model per trait (LogisticRegression on the standardised,
              named audio features, C on validation F1 - the same model as the cue table of
              Phase 9) + the TF-IDF word cues of Phase 9;
  model_card  the study's results (results/*.csv) as one JSON for the "How it works" page.

Acceptance check: the loaded models reproduce the test AUC of results/unimodal.csv, and the
subset metrics reproduce results/ablation.csv (so the app runs exactly the reported models).

Output: results/models/bundle.json
Run:    python src/10_export_bundle.py
"""
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import modeling
import utils
import config
from inference.audio import MFCC_COLUMNS

log = utils.get_logger("10_bundle")

MODALITIES = ["visual", "audio", "text"]
VOICE_MEASURES = ["pitch_mean", "pitch_std", "voiced_fraction", "rms_mean", "rms_std",
                  "zcr_mean", "nonsilent_fraction"]
QUANTILES = np.linspace(0, 1, 101)
R = config.RESULTS_DIR
D = config.DIAGNOSTICS_DIR


def subsets():
    """Every non-empty subset of the modalities, as sorted tuples in MODALITIES order."""
    return [s for k in range(1, len(MODALITIES) + 1) for s in itertools.combinations(MODALITIES, k)]


def key(subset):
    return "+".join(subset)


def auc(y, p):
    return utils.classification_metrics(y, (p >= 0.5).astype(int), p, quiet=True)["auc"]


def metrics(y, p):
    m = utils.classification_metrics(y, (p >= 0.5).astype(int), p, quiet=True)
    return {k: round(float(m[k]), 4) for k in ("acc", "f1", "auc")}


def load_probabilities(data):
    """prob[modality][trait] = P(high) of the saved Phase 6 classifier, for every clip."""
    prob = {}
    for modality in MODALITIES:
        X = data.matrix(modality)
        prob[modality] = {}
        for trait in config.TRAITS:
            clf = joblib.load(os.path.join(config.MODELS_DIR, f"unimodal_{modality}_{trait}_clf.joblib"))
            prob[modality][trait] = modeling.positive_probability(clf, X)
    return prob


def check_against_unimodal_csv(data, prob):
    table = pd.read_csv(config.UNIMODAL_CSV)
    worst = 0.0
    for modality in MODALITIES:
        for trait in config.TRAITS:
            reported = table[(table.modality == modality) & (table.trait == trait)]["auc"].iloc[0]
            now = auc(data.labels[trait][data.test], prob[modality][trait][data.test])
            worst = max(worst, abs(now - reported))
    if worst > 1e-6:
        sys.exit(f"ACCEPTANCE FAILED: the saved models give test AUCs up to {worst:.4f} away from "
                 f"results/unimodal.csv - re-run src/06_unimodal.py.")
    log.info("saved models reproduce results/unimodal.csv (test AUC, all 15 models)")


def fusion_tables(data, prob):
    held_out = data.val | data.test
    weights, out = {}, {}
    for trait in config.TRAITS:
        y = data.labels[trait]
        weights[trait] = {m: float(auc(y[data.val], prob[m][trait][data.val])) for m in MODALITIES}
    for subset in subsets():
        rows = {}
        for trait in config.TRAITS:
            y = data.labels[trait]
            w = np.array([weights[trait][m] for m in subset])
            w = w / w.sum()
            fused = sum(wi * prob[m][trait] for wi, m in zip(w, subset))
            rows[trait] = {
                "weights": {m: round(float(wi), 6) for m, wi in zip(subset, w)},
                "val_auc": round(float(auc(y[data.val], fused[data.val])), 4),
                "test": metrics(y[data.test], fused[data.test]),
                "quantiles": np.round(np.quantile(fused[held_out], QUANTILES), 5).tolist(),
            }
        out[key(subset)] = rows
    return weights, out


def check_against_ablation_csv(fusion):
    table = pd.read_csv(config.ABLATION_CSV)
    letters = {"V": "visual", "A": "audio", "T": "text"}
    worst = 0.0
    for row in table.itertuples():
        subset = tuple(m for m in MODALITIES if m in {letters[c] for c in row.subset.split("+")})
        worst = max(worst, abs(fusion[key(subset)][row.trait]["test"]["auc"] - row.auc))
    if worst > 1e-3:
        sys.exit(f"ACCEPTANCE FAILED: subset metrics differ from results/ablation.csv by {worst:.4f}")
    log.info(f"subset fusion reproduces results/ablation.csv ({len(table)} rows)")


def default_choice(fusion):
    """choice[available subset][trait] = the subset to fuse when only `available` exists."""
    full = key(MODALITIES)
    choice = {}
    for available in subsets():
        choice[key(available)] = {}
        for trait in config.TRAITS:
            if key(available) == full:
                choice[full][trait] = full                    # Phase 7's validation choice
                continue
            options = [s for s in subsets() if set(s) <= set(available)]
            best = max(options, key=lambda s: fusion[key(s)][trait]["val_auc"])
            choice[key(available)][trait] = key(best)
    return choice


def voice_quantiles(data):
    mfcc = data.features["audio_mfcc"]
    return {name: np.round(np.quantile(mfcc[:, MFCC_COLUMNS.index(name)], QUANTILES), 6).tolist()
            for name in VOICE_MEASURES}


def audio_feature_names():
    names = []
    for feature_set in config.MODALITIES["audio"]:
        if feature_set == "audio_mfcc":
            names += MFCC_COLUMNS
        else:
            names += [f"voice emotion: {e}" for e in json.load(open(config.AUDIO_EMOTION_LABELS_JSON))]
    return names


def cue_models(data):
    """The Phase 9 cue model (scaler -> LogisticRegression, C on validation F1), kept whole."""
    X = data.matrix("audio")
    out = {}
    for trait in config.TRAITS:
        y = data.labels[trait]
        best = None
        for C in config.LOGREG_C_GRID:
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
            model.fit(X[data.train], y[data.train])
            f1 = utils.classification_metrics(y[data.val], model.predict(X[data.val]), quiet=True)["f1"]
            score = f1 if not np.isnan(f1) else 0.0
            if best is None or score > best[1]:
                best = (model, score, C)
        scaler, logreg = best[0][0], best[0][-1]
        out[trait] = {"C": best[2], "mean": scaler.mean_.round(6).tolist(),
                      "scale": scaler.scale_.round(6).tolist(),
                      "coef": logreg.coef_[0].round(6).tolist()}
    return out


def text_cues():
    table = pd.read_csv(os.path.join(R, "cues_text.csv"))
    return {trait: {side: table[(table.trait == trait) & (table.side == side)]["word"].astype(str).tolist()
                    for side in ("high", "low")}
            for trait in config.TRAITS}


def model_card(data):
    """The numbers of results/REPORT.md, as JSON for the app's "How it works" page."""
    ci = pd.read_csv(os.path.join(D, "bootstrap_ci.csv"))
    ci = ci[ci.scope == "mean"]
    table1 = []
    names = [("baseline: majority class", "Always the majority class"),
             ("baseline: train mean", "Always the training average"),
             ("visual", "Face"), ("audio", "Voice"), ("text", "Words"),
             ("text_tfidf", "Words, TF-IDF baseline"), ("early", "Early fusion"),
             ("intermediate", "Intermediate fusion (network)"), ("late", "Late fusion (average)"),
             ("late_weighted", "Late fusion (weighted)")]
    for model, label in names:
        row = {"model": model, "label": label}
        for metric in ("acc", "f1", "auc", "one_minus_mae"):
            r = ci[(ci.model == model) & (ci.metric == metric)]
            if len(r) and not np.isnan(r.estimate.iloc[0]):
                row[metric] = [round(float(r[c].iloc[0]), 4) for c in ("estimate", "ci_low", "ci_high")]
        table1.append(row)
    baselines = pd.read_csv(os.path.join(D, "baselines.csv"))
    random = baselines[baselines.baseline.str.startswith("stratified")][["acc", "f1", "auc"]].mean()
    table1.insert(1, {"model": "baseline: random", "label": "Random guessing",
                      **{m: [round(float(random[m]), 4)] for m in ("acc", "f1", "auc")}})

    ablation = pd.read_csv(config.ABLATION_CSV)
    ablation = [{"subset": r.subset, "removed": r.removed, "kind": r.kind, "trait": r.trait,
                 "f1": round(r.f1, 4), "auc": round(r.auc, 4), "delta_f1": round(r.delta_f1, 4)}
                for r in ablation.itertuples()]
    curve = pd.read_csv(os.path.join(D, "learning_curve.csv"))
    curve = curve.groupby("fraction").mean(numeric_only=True).reset_index()
    shuffle = pd.read_csv(os.path.join(D, "shuffle_test.csv"))[["test_f1", "test_auc"]].mean()
    overfit = pd.read_csv(os.path.join(D, "overfitting.csv"))
    overfit = overfit.groupby("model")[["train_f1", "val_f1", "test_f1"]].mean().reset_index()
    traits = pd.read_csv(os.path.join(R, "trait_correlations.csv"), index_col=0)
    emotions = pd.read_csv(os.path.join(R, "emotion_trait_corr.csv"), index_col=0)
    gain = pd.read_csv(os.path.join(D, "fusion_vs_single.csv"))
    best = json.load(open(config.BEST_FUSION_JSON))
    split_counts = data.df["split"].value_counts().to_dict()
    youtube = data.df.groupby("split")["youtube_id"].nunique().to_dict()
    return {
        "dataset": {"clips": int(len(data.df)), "youtube_videos": int(data.df["youtube_id"].nunique()),
                    "splits": {s: {"clips": int(split_counts[s]), "videos": int(youtube[s])}
                               for s in ("train", "val", "test")}},
        "chosen_fusion": best["fusion_type"],
        "table1": table1,
        "ablation": ablation,
        "learning_curve": [{"fraction": float(r.fraction), "train_clips": int(round(r.train_clips)),
                            "test_f1": round(r.test_f1, 4), "test_auc": round(r.test_auc, 4)}
                           for r in curve.itertuples()],
        "shuffle_test": {"test_f1": round(float(shuffle.test_f1), 4),
                         "test_auc": round(float(shuffle.test_auc), 4)},
        "overfitting": [{"model": r.model, "train_f1": round(r.train_f1, 4), "val_f1": round(r.val_f1, 4),
                         "test_f1": round(r.test_f1, 4)} for r in overfit.itertuples()],
        "fusion_vs_single": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                             for r in gain.to_dict("records")],
        "trait_correlations": {"labels": list(traits.columns),
                               "matrix": traits.round(3).values.tolist()},
        "emotion_trait_corr": {"rows": list(emotions.index), "cols": list(emotions.columns),
                               "matrix": emotions.round(3).values.tolist()},
    }


def labels_or_empty(path):
    return json.load(open(path)) if os.path.exists(path) else []


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=config.ROOT).stdout.strip()
    except OSError:
        return ""


def main():
    utils.set_seed()
    utils.warn_if_fake_labels()
    data = modeling.Data()
    log.info(data.describe())

    prob = load_probabilities(data)
    check_against_unimodal_csv(data, prob)
    weights, fusion = fusion_tables(data, prob)
    check_against_ablation_csv(fusion)

    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "traits": config.TRAITS,
        "modalities": MODALITIES,
        "feature_sets": {m: config.MODALITIES[m] for m in MODALITIES},
        "labels": {   # empty when the heavy models were not run (smoke test)
            "voice_emotion": labels_or_empty(config.AUDIO_EMOTION_LABELS_JSON),
            "face_emotion": labels_or_empty(config.VISUAL_EMOTION_LABELS_JSON),
            "text_emotion": labels_or_empty(config.TEXT_EMOTION_LABELS_JSON),
        },
        "models": {"speech_emotion": config.SPEECH_EMOTION_MODEL, "face_emotion": config.FACE_EMOTION_MODEL,
                   "text_embedding": config.TEXT_EMBED_MODEL, "text_emotion": config.TEXT_EMOTION_MODEL,
                   "face_cnn": "torchvision ResNet-50 (IMAGENET1K_V2)"},
        "held_out_clips": int((data.val | data.test).sum()),
        "weights": weights,
        "fusion": fusion,
        "default_choice": default_choice(fusion),
        "voice_quantiles": voice_quantiles(data),
        "audio_feature_names": audio_feature_names(),
        "cue_models": cue_models(data),
        "text_cues": text_cues(),
        "model_card": model_card(data),
    }
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.BUNDLE_JSON, "w") as f:
        json.dump(bundle, f)

    for available, per_trait in bundle["default_choice"].items():
        log.info(f"  available {available:<18} -> " + ", ".join(f"{t[:5]}: {s}" for t, s in per_trait.items()))
    log.info(f"ACCEPTANCE CHECK PASSED: wrote {config.BUNDLE_JSON} "
             f"({os.path.getsize(config.BUNDLE_JSON) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
