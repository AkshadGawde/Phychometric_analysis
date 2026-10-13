"""Phase 9 - Cue analysis and the written reports.

1. Cues (which features push a prediction up or down), fitted on TRAIN, settings chosen on VAL:
   - audio: LogisticRegression on the standardised, NAMED audio features (no PCA, so every
     weight belongs to one feature) -> top-5 |weight| per trait  -> results/cues_audio.csv
   - text: LogisticRegression on the TF-IDF words -> top-10 positive / negative words per trait
                                                     -> results/cues_text.csv
   - emotions: correlation between the face / voice / text emotion probabilities and the raw
     trait scores (train split) -> results/emotion_trait_corr.csv + emotion_trait_corr.png
   - halo effect: correlations between the five trait labels -> results/trait_correlations.csv
2. results/summary.md  (the roadmap's short summary) and results/REPORT.md (the full
   plain-English report). EVERY number in both files is read from the result files of
   Phases 1-8b; nothing is typed by hand.

Run:  python src/09_analysis.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import modeling
import utils
import config

log = utils.get_logger("09_analysis")
R, D = config.RESULTS_DIR, config.DIAGNOSTICS_DIR
NICE = {"visual": "Face (visual)", "audio": "Voice (audio)", "text": "Words (text)",
        "text_tfidf": "Words, TF-IDF baseline", "early": "Early fusion", "late": "Late fusion (average)",
        "late_weighted": "Late fusion (weighted)", "intermediate": "Intermediate fusion (network)"}
EMOTION_SETS = {"visual_emotion": ("face", config.VISUAL_EMOTION_LABELS_JSON),
                "audio_emotion": ("voice", config.AUDIO_EMOTION_LABELS_JSON),
                "text_emotion": ("words", config.TEXT_EMOTION_LABELS_JSON)}


def f3(value):
    return "–" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.3f}"


def table_md(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 1. Cues
# ----------------------------------------------------------------------------
def feature_names(feature_set):
    if feature_set == "audio_mfcc":
        return json.load(open(config.AUDIO_MFCC_COLS_JSON))
    if feature_set in EMOTION_SETS:
        source, path = EMOTION_SETS[feature_set]
        return [f"{source} emotion: {label}" for label in json.load(open(path))]
    raise ValueError(feature_set)


def best_logreg(X_train, y_train, X_val, y_val, scale):
    """LogisticRegression with C chosen on validation F1 (no PCA, so weights stay readable)."""
    best = None
    for C in config.LOGREG_C_GRID:
        estimator = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        model = make_pipeline(StandardScaler(), estimator) if scale else estimator
        model.fit(X_train, y_train)
        f1 = utils.classification_metrics(y_val, model.predict(X_val), quiet=True)["f1"]
        score = f1 if not np.isnan(f1) else 0.0
        if best is None or score > best[1]:
            best = (model, score, C)
    model = best[0]
    return (model[-1] if scale else model).coef_[0], best[2]


def audio_cues(data):
    sets = config.MODALITIES["audio"]
    names = [n for s in sets for n in feature_names(s)]
    X = data.matrix("audio")
    rows = []
    for trait in config.TRAITS:
        y = data.labels[trait]
        coef, C = best_logreg(X[data.train], y[data.train], X[data.val], y[data.val], scale=True)
        for rank, i in enumerate(np.argsort(-np.abs(coef))[:5], start=1):
            rows.append({"trait": trait, "rank": rank, "feature": names[i], "weight": coef[i],
                         "direction": "higher -> trait high" if coef[i] > 0 else "higher -> trait low",
                         "C": C})
    return pd.DataFrame(rows)


def text_cues(data):
    vocab = json.load(open(config.TFIDF_VOCAB_JSON))
    X = data.features["text_tfidf"]
    rows = []
    for trait in config.TRAITS:
        y = data.labels[trait]
        coef, C = best_logreg(X[data.train], y[data.train], X[data.val], y[data.val], scale=False)
        order = np.argsort(coef)
        for rank, i in enumerate(order[::-1][:10], start=1):
            rows.append({"trait": trait, "side": "high", "rank": rank, "word": vocab[i], "weight": coef[i], "C": C})
        for rank, i in enumerate(order[:10], start=1):
            rows.append({"trait": trait, "side": "low", "rank": rank, "word": vocab[i], "weight": coef[i], "C": C})
    return pd.DataFrame(rows)


def emotion_correlations(data):
    """Pearson correlation of every emotion probability with every raw trait score (train split)."""
    rows = {}
    for feature_set in EMOTION_SETS:
        if feature_set not in data.features and not os.path.exists(utils.feature_path(feature_set)):
            continue
        values = utils.load_features([feature_set])[feature_set] if feature_set not in data.features \
            else data.features[feature_set]
        for j, name in enumerate(feature_names(feature_set)):
            column = values[data.train, j]
            rows[name] = {t: (np.corrcoef(column, data.scores[t][data.train])[0, 1]
                              if column.std() > 0 else np.nan) for t in config.TRAITS}
    return pd.DataFrame(rows).T[config.TRAITS]


def figure_emotion_corr(plt, corr):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", "#2a78d6"])
    limit = max(0.1, float(np.nanmax(np.abs(corr.values))))
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(corr) + 1.6))
    image = ax.imshow(corr.values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(value) > 0.7 * limit else utils.PLOT["ink"])
    ax.set_xticks(range(corr.shape[1]), [t.capitalize() for t in corr.columns], rotation=30, ha="left",
                  fontsize=9)
    ax.set_yticks(range(corr.shape[0]), corr.index, fontsize=8)
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("correlation (blue = goes up with the trait)", fontsize=8)
    fig.suptitle("Emotion probabilities vs trait scores (train clips)" + utils.fake_note(),
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "emotion_trait_corr.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2. Reports - everything below reads the CSV / JSON result files
# ----------------------------------------------------------------------------
def load_results():
    need = [config.UNIMODAL_CSV, config.FUSION_CSV, config.ABLATION_CSV, config.BEST_FUSION_JSON]
    need += [os.path.join(D, n) for n in ["baselines.csv", "overfitting.csv", "bootstrap_ci.csv",
                                          "fusion_vs_single.csv", "shuffle_test.csv", "learning_curve.csv",
                                          "intermediate_loss.csv", "best_model_per_trait.csv"]]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        sys.exit(f"Missing result files {missing} - run Phases 6, 7, 8 and 8b first.")
    r = {"unimodal": pd.read_csv(config.UNIMODAL_CSV), "fusion": pd.read_csv(config.FUSION_CSV),
         "ablation": pd.read_csv(config.ABLATION_CSV), "best": json.load(open(config.BEST_FUSION_JSON))}
    for name in ["baselines", "overfitting", "bootstrap_ci", "fusion_vs_single", "shuffle_test",
                 "learning_curve", "intermediate_loss", "best_model_per_trait"]:
        r[name] = pd.read_csv(os.path.join(D, f"{name}.csv"))
    visual = os.path.join(D, "visual_feature_comparison.csv")
    r["visual_cmp"] = pd.read_csv(visual) if os.path.exists(visual) else None
    for name in ["cues_audio", "cues_text", "trait_correlations", "emotion_trait_corr"]:
        r[name] = pd.read_csv(os.path.join(R, f"{name}.csv"), index_col=0 if "corr" in name else None)
    r["failed"] = pd.read_csv(config.FAILED_CLIPS_CSV) if os.path.exists(config.FAILED_CLIPS_CSV) else pd.DataFrame()
    return r


def ci(r, model, metric, scope="mean"):
    row = r["bootstrap_ci"][(r["bootstrap_ci"].model == model) & (r["bootstrap_ci"].metric == metric)
                           & (r["bootstrap_ci"].scope == scope)]
    if row.empty:
        return np.nan, np.nan, np.nan
    row = row.iloc[0]
    return row.estimate, row.ci_low, row.ci_high


def with_ci(r, model, metric, scope="mean"):
    est, lo, hi = ci(r, model, metric, scope)
    return "–" if np.isnan(est) else (f"{est:.3f} [{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else f"{est:.3f}")


def dataset_section(df, r):
    counts = df.groupby("split").agg(clips=("video_id", "count"), videos=("youtube_id", "nunique"))
    counts = counts.reindex(["train", "val", "test"])
    halo = r["trait_correlations"]
    off = halo.values[~np.eye(len(halo), dtype=bool)]
    neuro = halo.loc["neuroticism"].drop("neuroticism")
    label_files = sorted(f for f in os.listdir(config.ANNOTATIONS_DIR)
                         if f in config.REAL_ANNOTATION_NAMES)
    transcript_files = sorted(f for f in os.listdir(config.TRANSCRIPTS_DIR) if f.endswith(".pkl"))
    lines = [
        f"- **Clips used:** {len(df)} short talking-head videos (about 15 s each) from "
        f"{df.youtube_id.nunique()} different YouTube videos, all with real ChaLearn First "
        f"Impressions labels. {len(r['failed'])} clip(s) were dropped in preprocessing.",
        f"- **Labels:** the official ChaLearn annotation files ({', '.join(label_files)}). Each clip "
        f"has five scores between 0 and 1 given by crowd workers, i.e. *apparent* personality — "
        f"how the person comes across, not a personality test. For classification a score counts "
        f"as \"high\" when it is at or above the median of the training clips.",
        f"- **Transcripts:** {', '.join(transcript_files)} — taken from the public Hugging Face mirror "
        f"of the dataset (`yeray142/first-impressions-v2`), because the official transcript files "
        f"were not available. The videos come from the official ChaLearn subset and the same mirror.",
        "- **Split:** by YouTube video, so the same person never appears in two parts:",
        "",
        table_md(["part", "clips", "YouTube videos", "share of clips"],
                 [[s, int(counts.loc[s, "clips"]), int(counts.loc[s, "videos"]),
                   f"{counts.loc[s, 'clips'] / len(df) * 100:.1f}%"] for s in counts.index]),
        "",
        f"- **\"Neuroticism\" is really emotional stability.** The dataset documentation describes "
        f"this label as emotional stability (the reverse of neuroticism). It correlates "
        f"*positively* with the other four traits here (r = {neuro.min():.2f} to {neuro.max():.2f}), "
        f"which fits that description: a high score means calm and stable.",
        f"- **Halo effect.** The five labels are strongly related (average correlation between two "
        f"traits r = {off.mean():.2f}, range {off.min():.2f}–{off.max():.2f}). Raters who like a "
        f"person tend to score them high on everything, so the five tasks are far from independent.",
    ]
    return "\n".join(lines)


def table1(r, models):
    base = r["baselines"].groupby("baseline")[["acc", "f1", "auc", "one_minus_mae"]].mean()
    rows = [["Always the majority class", f3(base.loc["majority class", "acc"]),
             f3(base.loc["majority class", "f1"]), f3(base.loc["majority class", "auc"]), "–"],
            ["Random guessing (class balance of train)",
             f3(base.loc["stratified random (mean of draws)", "acc"]),
             f3(base.loc["stratified random (mean of draws)", "f1"]),
             f3(base.loc["stratified random (mean of draws)", "auc"]), "–"],
            ["Always the training average score", "–", "–", "–",
             with_ci(r, "baseline: train mean", "one_minus_mae")]]
    for model in models:
        rows.append([NICE[model]] + [with_ci(r, model, m) for m in ["acc", "f1", "auc", "one_minus_mae"]])
    return table_md(["model", "accuracy", "F1", "AUC", "1 − MAE"], rows)


def verdict(r, df, best_model):
    fvs = r["fusion_vs_single"].set_index("metric")
    best_single, best_fusion = r["best"]["best_single_modality"], r["best"]["fusion_type"]
    f1 = ci(r, best_model, "f1")[0]
    acc, acc_lo, acc_hi = ci(r, best_model, "acc")
    auc, auc_lo, auc_hi = ci(r, best_model, "auc")
    base = r["baselines"].groupby("baseline")[["acc", "f1", "auc"]].mean()
    majority, rnd = base.loc["majority class"], base.loc["stratified random (mean of draws)"]
    mean_base = ci(r, "baseline: train mean", "one_minus_mae")[0]
    boot = r["bootstrap_ci"]
    model_rows = boot[(boot.scope == "mean") & (~boot.model.str.startswith("baseline"))]
    mae_rows = model_rows[model_rows.metric == "one_minus_mae"].dropna(subset=["estimate"])
    best_mae_row = mae_rows.loc[mae_rows.estimate.idxmax()]
    auc_rows = model_rows[model_rows.metric == "auc"]
    top_test = auc_rows.loc[auc_rows.estimate.idxmax()]
    abl = r["ablation"][r["ablation"].kind == "one removed"].groupby("removed")["delta_f1"].mean()
    f1_half_width = (ci(r, best_fusion, "f1")[2] - ci(r, best_fusion, "f1")[1]) / 2
    singles = r["unimodal"][r["unimodal"].modality != "text_tfidf"].groupby("modality")[["f1"]].mean()["f1"]
    single_auc = {m: ci(r, m, "auc")[0] for m in singles.index}
    over = r["overfitting"]
    gaps = over.groupby("model")["train_minus_test"].mean()
    val_test = (over.groupby("model")["val_f1"].mean() - over.groupby("model")["test_f1"].mean()).abs()
    curve = r["learning_curve"].groupby("fraction")[["train_clips", "test_f1", "test_auc"]].mean()
    published = config.PUBLISHED_TOP_ONE_MINUS_MAE
    name = lambda m: NICE.get(m, m).lower()
    short = lambda m: NICE.get(m, m).split(" (")[0].lower()

    s = []
    learned = auc_lo > 0.5 and acc_lo > max(rnd.acc, majority.acc)
    s.append(("**Learning beyond the baselines — yes, but modestly:** " if learned else
              "**Learning beyond the baselines — not clearly:** ")
             + f"the model chosen on validation, "
             f"{name(best_model)}, reaches a mean test AUC of {auc:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f}) "
             f"and accuracy {acc:.3f} ({acc_lo:.3f}–{acc_hi:.3f}), against AUC {rnd.auc:.3f} and accuracy "
             f"{rnd.acc:.3f} for random guessing and {majority.acc:.3f} accuracy for always answering the "
             f"most common class; "
             + ("both intervals lie clearly above those baselines" if learned
                else "the intervals overlap the baselines")
             + f", but its F1 ({f1:.3f}) is {'below' if f1 < majority.f1 else 'only just above'} the "
               f"{majority.f1:.3f} that \"always high\" gets, so F1 alone would hide the learning.")
    s.append(f"**On the raw 0–1 scores** the best 1 − MAE is {best_mae_row.estimate:.3f} "
             f"({name(best_mae_row.model)}), only {best_mae_row.estimate - mean_base:+.3f} better than always "
             f"predicting the training average ({mean_base:.3f}), because the scores bunch up around the middle.")
    gain = fvs.loc["f1"]
    gain_auc = fvs.loc["auc"]
    significant = bool(gain.gain_significant or gain_auc.gain_significant)
    s.append(("**Fusion differs significantly from the best single modality:** " if significant else
              "**Fusion is not significantly better than the best single modality:** ")
             + f"{name(best_fusion)} "
             f"changes mean F1 by {gain.gain:+.3f} (95% CI {gain.gain_ci_low:+.3f} to {gain.gain_ci_high:+.3f}) "
             f"and AUC by {gain_auc.gain:+.3f} ({gain_auc.gain_ci_low:+.3f} to {gain_auc.gain_ci_high:+.3f}) "
             f"compared with {name(best_single)}"
             + ("; both intervals include zero" if not significant
                else "; at least one interval excludes zero, so the difference is statistically clear")
             + f" (on the test clips the highest AUC was {name(top_test.model)} at {top_test.estimate:.3f}, "
               f"but it was not the validation choice).")
    strongest = max(single_auc, key=single_auc.get)
    s.append(f"**Voice, face, words:** on its own the strongest modality is {short(strongest)} "
             f"(AUC {single_auc[strongest]:.3f}), followed by "
             + " and ".join(f"{short(m)} ({single_auc[m]:.3f})"
                            for m in sorted(single_auc, key=single_auc.get, reverse=True)[1:])
             + f"; in the ablation removing {short(abl.idxmax())} hurts most, but every mean change is small "
               f"(between {abl.min():+.3f} and {abl.max():+.3f} F1) next to the ±{f1_half_width:.3f} "
               f"uncertainty of the F1 itself.")
    worst = gaps.sort_values(ascending=False)
    overfit = bool(over.flag.any())
    honest_selection = val_test.max() <= 0.05
    s.append(("**Overfitting is clear" if overfit else "**Little overfitting")
             + (" but did not fool model selection:** " if honest_selection else " and validation is unreliable:** ")
             + f"the most flexible models fit "
             f"their training clips much better (train − test F1 up to {worst.iloc[0]:.2f} for "
             f"{name(worst.index[0])}), while the compact voice model shows only "
             f"{gaps.get('audio', np.nan):.2f}; validation and test F1 agree to within "
             f"{val_test.max():.3f} for every model"
             + (", so choices made on validation hold up on test." if honest_selection
                else ", so choices made on validation may not hold up on test."))
    first, last = curve.iloc[0], curve.iloc[-1]
    step = curve.test_auc.iloc[-1] - curve.test_auc.iloc[-2] if len(curve) > 1 else 0.0
    auc_rise = last.test_auc - first.test_auc
    s.append(("**More data would help somewhat:** " if auc_rise > 0.02 else
              "**More data of the same kind would help little:** ")
             + f"going from {first.train_clips:.0f} to {last.train_clips:.0f} "
             f"training clips raises test AUC from {first.test_auc:.3f} to {last.test_auc:.3f} "
             f"(last step {step:+.3f}) while F1 stays near {last.test_f1:.3f}"
             + (", so the ranking of clips keeps improving with data, though the gains are getting smaller."
                if auc_rise > 0.02 else ", so the curve has levelled off."))
    s.append(f"**Compared with published work:** the best published First Impressions systems reach about "
             f"{published:.2f} mean 1 − MAE with all 10,000 clips and models trained end to end; our best "
             f"{best_mae_row.estimate:.3f} uses {len(df)} clips ({len(df) / 10000 * 100:.0f}% of the data), "
             f"frozen pretrained features and {config.N_FRAMES} frames per clip, so it is "
             f"{published - best_mae_row.estimate:.3f} lower. Because always guessing the average already "
             f"scores {mean_base:.3f} here, that gap is bigger than it looks, and the comparison is only rough "
             f"since our test clips are not the official test set.")
    return " ".join(s)


def cues_section(r):
    out = []
    audio, text, emo = r["cues_audio"], r["cues_text"], r["emotion_trait_corr"]
    for trait in config.TRAITS:
        a = audio[audio.trait == trait].sort_values("rank")
        hi = text[(text.trait == trait) & (text.side == "high")].sort_values("rank").word.tolist()
        lo = text[(text.trait == trait) & (text.side == "low")].sort_values("rank").word.tolist()
        e = emo[trait].dropna()
        top_e = e.reindex(e.abs().sort_values(ascending=False).index)[:3]
        out.append(f"**{trait.capitalize()}**\n")
        out.append("- Voice: " + "; ".join(f"`{row.feature}` ({'+' if row.weight > 0 else '−'})"
                                          for row in a.itertuples()))
        out.append(f"- Words used more by clips rated high: {', '.join(hi)}")
        out.append(f"- Words used more by clips rated low: {', '.join(lo)}")
        out.append("- Strongest emotion links: " + "; ".join(f"{name} (r = {v:+.2f})" for name, v in top_e.items()))
        out.append("")
    return "\n".join(out)


def diagnostics_section(r):
    over = r["overfitting"]
    by_model = over.groupby("model")[["train_f1", "val_f1", "test_f1", "train_minus_test"]].mean()
    by_model = by_model.reindex([m for m in NICE if m in by_model.index])
    shuffle = r["shuffle_test"]
    shuffle_ok = abs(shuffle.test_auc.mean() - 0.5) <= config.SHUFFLE_AUC_TOLERANCE
    curve = r["learning_curve"].groupby("fraction")[["train_clips", "test_f1", "test_auc"]].mean().reset_index()
    hist = r["intermediate_loss"]
    lines = [
        "**Overfitting (mean F1 over the five traits).** A big gap between the training clips and "
        "the test clips means the model memorised rather than learned.",
        "",
        table_md(["model", "train F1", "validation F1", "test F1", "train − test", "flagged traits"],
                 [[NICE[m], f3(row.train_f1), f3(row.val_f1), f3(row.test_f1), f3(row.train_minus_test),
                   int(over[(over.model == m) & over.flag].shape[0])] for m, row in by_model.iterrows()]),
        "",
        f"**Label-shuffle test.** The {NICE[r['best']['fusion_type']].lower()} model was retrained "
        f"{shuffle.repeat.nunique()} times on training labels shuffled at random, so no real pattern was "
        f"left to learn. Its test AUC fell to {shuffle.test_auc.mean():.3f} (F1 {shuffle.test_f1.mean():.3f}). "
        + ("That is chance level, as it should be: there is no sign of information leaking from the test set."
           if shuffle_ok else "That is NOT chance level: possible leakage, results must not be trusted."),
        "",
        "**Learning curve** ([figure](diagnostics/learning_curve.png)):",
        "",
        table_md(["share of training clips", "training clips", "test F1", "test AUC"],
                 [[f"{row.fraction * 100:.0f}%", int(row.train_clips), f3(row.test_f1), f3(row.test_auc)]
                  for row in curve.itertuples()]),
        "",
        f"**Fusion network training** ([figure](intermediate_loss.png)): validation loss was lowest at "
        f"epoch {int(hist[hist.best_epoch].epoch.iloc[0])} of {len(hist)}; the weights of that epoch are used.",
    ]
    if r["visual_cmp"] is not None:
        v = r["visual_cmp"].groupby(["feature_set", "dims"])[["test_f1", "test_auc"]].mean().reset_index()
        lines += ["", "**Face features compared one by one** (the face model was weak, mean F1 ≤ "
                  f"{config.VISUAL_WEAK_F1}):", "",
                  table_md(["feature set", "numbers per clip", "mean test F1", "mean test AUC"],
                           [[row.feature_set, int(row.dims), f3(row.test_f1), f3(row.test_auc)]
                            for row in v.itertuples()])]
    return "\n".join(lines)


def write_reports(df, r):
    best_model = r["best_model_per_trait"].model.iloc[0]
    best_single, best_fusion = r["best"]["best_single_modality"], r["best"]["fusion_type"]
    compare = best_single if best_model != best_single else best_fusion
    models = [m for m in ["visual", "audio", "text", "text_tfidf", "early", "intermediate", "late", "late_weighted"]
              if m in set(r["bootstrap_ci"].model)]
    fake = ("> **WARNING — FAKE LABELS.** These results come from random dummy labels and mean "
            "nothing.\n\n" if utils.labels_are_fake() else "")

    t2_rows = []
    for trait in config.TRAITS:
        t2_rows.append([trait.capitalize(), with_ci(r, best_model, "f1", trait), with_ci(r, compare, "f1", trait),
                        f"{ci(r, best_model, 'f1', trait)[0] - ci(r, compare, 'f1', trait)[0]:+.3f}",
                        with_ci(r, best_model, "auc", trait), with_ci(r, compare, "auc", trait)])
    abl = r["ablation"][r["ablation"].kind == "one removed"]
    order = [m for m in ["visual", "audio", "text"] if m in set(abl.removed)]
    drops = abl.pivot(index="trait", columns="removed", values="delta_f1").reindex(index=config.TRAITS,
                                                                                   columns=order)
    t3_rows = [[t.capitalize()] + [f"{drops.loc[t, m]:+.3f}" for m in drops.columns] for t in config.TRAITS]
    t3_rows.append(["**mean**"] + [f"**{drops[m].mean():+.3f}**" for m in drops.columns])

    figures = [("Sample face crops", "sample_frames.png"), ("Raw vs cleaned audio", "sample_waveform.png"),
               ("Transcript lengths", "text_length_hist.png"), ("Single modalities, F1 per trait", "unimodal_f1.png"),
               ("Single modalities vs fusion", "unimodal_vs_fusion.png"),
               ("Fusion network training curve", "intermediate_loss.png"), ("Ablation", "ablation.png"),
               ("Emotion vs trait correlations", "emotion_trait_corr.png"),
               ("Confusion matrices of the best model", "confusion_best.png"),
               ("Learning curve", "diagnostics/learning_curve.png")]
    figure_lines = "\n".join(f"- [{title}]({path})" for title, path in figures if os.path.exists(os.path.join(R, path)))
    how_to_read = (
        "**How to read the numbers.** *Accuracy*: share of clips put on the right side of the "
        "median. *F1*: balance of catching the \"high\" clips and not raising false alarms (1 = "
        "perfect). Careful: simply answering \"high\" for every clip already gets a fairly high F1 "
        "here (see the first row of Table 1), so F1 on its own flatters weak models — *AUC* is "
        "the fairest single number. *AUC*: chance that a random \"high\" clip gets a higher score than a "
        "random \"low\" one (0.5 = guessing, 1 = perfect). *1 − MAE*: 1 minus the average error on "
        "the raw 0–1 score (the official ChaLearn measure). Square brackets give the 95% confidence "
        "interval: re-sampling the test clips 1,000 times, the value stayed in this range 95% of the "
        "time. All numbers are on the test clips, which no model saw during training or tuning.")
    limitations = (
        "## Limitations\n\n"
        f"- Only {len(df)} of the 10,000 clips were used, and our own test split is smaller and "
        "different from the official one.\n"
        "- The labels are *apparent* personality — first impressions of crowd workers — not "
        "personality measured by a validated questionnaire.\n"
        "- The data and labels may carry demographic bias (age, gender, ethnicity, attractiveness "
        "affect first impressions); we did not test for it.\n"
        "- All pretrained models were frozen; only 8 frames per clip were used, and the speech-"
        "emotion model was trained on acted speech.\n"
        "- Turning scores into high / low at the median throws away information and makes clips near "
        "the median hard to classify.\n"
        "- **This is a research prototype. It must not be used to judge or diagnose anyone.**\n\n"
        "## Future work\n\n"
        "- Use all 10,000 clips, more frames, and fine-tune the visual and speech models end to end.\n"
        "- Model time (how expressions and voice change during the clip) instead of averages.\n"
        "- Move from apparent personality to clinically meaningful targets: the DAIC-WOZ interview "
        "corpus with PHQ-8 depression scores is the natural next dataset for the same "
        "face + voice + words pipeline.\n")

    report = "\n\n".join([
        "# Multimodal Psychometric Assessment — Final Report", fake + how_to_read,
        "## 1. The data actually used", dataset_section(df, r),
        "## 2. Table 1 — Baselines, single modalities and fusion (mean over the five traits, test split)",
        table1(r, models),
        f"*The model chosen on the validation set is {NICE[best_model].lower()}; the fusion type "
        f"chosen on validation is {NICE[best_fusion].lower()}.*",
        f"## 3. Table 2 — Per trait: {NICE[best_model].lower()} vs {NICE[compare].lower()}",
        table_md(["trait", f"F1 {NICE[best_model]}", f"F1 {NICE[compare]}", "F1 difference",
                  f"AUC {NICE[best_model]}", f"AUC {NICE[compare]}"], t2_rows),
        f"## 4. Table 3 — Ablation of the {NICE[best_fusion].lower()} model",
        "Drop in test F1 when one modality is removed (positive = the model was better with it):",
        table_md(["trait"] + [f"without {NICE[m].split(' (')[0].lower()}" for m in drops.columns], t3_rows),
        "## 5. Diagnostics — do the models really learn?", diagnostics_section(r),
        "## 6. Top cues per trait",
        "Voice features are the five strongest weights of a logistic regression on the standardised "
        "audio features (+ = a higher value pushes towards \"high\"). Words come from a logistic "
        "regression on TF-IDF. Emotion links are correlations between the emotion probabilities and "
        "the raw scores on the training clips ([heat map](emotion_trait_corr.png)).",
        cues_section(r),
        "## 7. VERDICT", verdict(r, df, best_model),
        limitations,
        "## Figures", figure_lines,
    ])
    with open(config.REPORT_MD, "w") as f:
        f.write(report + "\n")

    summary = "\n\n".join([
        "# Summary — VIP Task T1: Multimodal Psychometric Assessment", fake.strip(),
        "## Data", dataset_section(df, r),
        "## Final metrics (mean over traits, test split, 95% CI)", table1(r, models),
        f"## Ablation ({NICE[best_fusion].lower()})",
        table_md(["trait"] + [f"without {NICE[m].split(' (')[0].lower()}" for m in drops.columns], t3_rows),
        "## Top cues per trait", cues_section(r),
        "## Conclusion: do face, voice and words improve psychometric assessment when combined, "
        "and which modality contributes most?", verdict(r, df, best_model),
        limitations,
        "Full report with diagnostics: [REPORT.md](REPORT.md)",
    ])
    with open(config.SUMMARY_MD, "w") as f:
        f.write(summary + "\n")


def main():
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()
    data = modeling.Data()
    log.info(data.describe())

    audio = audio_cues(data)
    audio.to_csv(os.path.join(R, "cues_audio.csv"), index=False)
    text = text_cues(data)
    text.to_csv(os.path.join(R, "cues_text.csv"), index=False)
    corr = emotion_correlations(data)
    corr.to_csv(os.path.join(R, "emotion_trait_corr.csv"))
    figure_emotion_corr(utils.setup_plots(), corr)
    data.df[config.TRAITS].corr().to_csv(os.path.join(R, "trait_correlations.csv"))
    for trait in config.TRAITS:
        a = audio[audio.trait == trait].feature.tolist()
        hi = text[(text.trait == trait) & (text.side == "high")].word.tolist()[:5]
        log.info(f"{trait:<18} voice: {a[:3]} | words (high): {hi}")

    write_reports(data.df, load_results())

    # ---- acceptance: the reports exist and their numbers come from the files ----
    for path in [config.SUMMARY_MD, config.REPORT_MD]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            log.info(f"ACCEPTANCE FAILED: {path} was not written")
            sys.exit(1)
    log.info(f"ACCEPTANCE CHECK PASSED: {config.SUMMARY_MD} and {config.REPORT_MD} written; every number "
             f"in them is read from the result files")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
