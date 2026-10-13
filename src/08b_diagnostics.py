"""Phase 8b - Diagnostics: how well do the models really learn?

Runs after Phase 8. Every model is rebuilt with exactly the code of Phases 6-7 (same seeds),
then checked in eight ways. All tables go to results/diagnostics/.

  B1 baselines.csv           models that learn nothing: always the majority class, random
                             guessing with the train class balance (mean of 100 draws), and
                             "always predict the train mean score" (MAE / 1-MAE).
  B2 overfitting.csv         train vs validation vs test F1 of every model and trait; flagged
                             when train F1 - test F1 > OVERFIT_GAP.
  B3 bootstrap_ci.csv        95% confidence intervals (1,000 resamples of the test set) of
     fusion_vs_single.csv    acc / F1 / AUC / 1-MAE, per trait and averaged over the traits, and
                             of the gain of the best fusion over the best single modality.
                             Test clips of the same YouTube video are resampled together,
                             because they are not independent (same person).
  B4 shuffle_test.csv        the best fusion model retrained on SHUFFLED train/val labels.
                             With no real signal left, its test AUC must fall to about 0.5;
                             if not, information leaks from the test set and the script stops.
  B5 learning_curve.csv/png  the best fusion model trained on 25 / 50 / 75 / 100% of the train
                             clips (whole YouTube videos at a time).
  B6 intermediate_loss.csv   train / validation loss per epoch of the fusion network (the figure
                             is results/intermediate_loss.png from Phase 7).
  B7 best_model_per_trait.csv + confusion_best.png   the best model (chosen on validation).
  B8 visual_feature_comparison.csv   if the visual models are weak (mean F1 <= VISUAL_WEAK_F1),
                             each visual feature set is tried on its own.

Run:  python src/08b_diagnostics.py
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.dummy import DummyClassifier

import modeling
import utils
import config

log = utils.get_logger("08b_diag")
OUT = config.DIAGNOSTICS_DIR
METRICS = ["acc", "f1", "auc", "one_minus_mae"]


def save(table, name):
    table.to_csv(os.path.join(OUT, name), index=False)
    log.info(f"  saved diagnostics/{name}")


# ----------------------------------------------------------------------------
# Fast metrics for the bootstrap (same definitions as sklearn, vectorised)
# ----------------------------------------------------------------------------
def fast_f1(y, pred):
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def fast_auc(y, prob):
    """Area under the ROC curve = probability that a random positive clip gets a higher score
    than a random negative clip (ties count half)."""
    n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(prob)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def point_metrics(y, prob, s_true, s_pred):
    pred = (prob >= 0.5).astype(int)
    single_class = len(np.unique(y)) < 2
    out = {"acc": float(np.mean(pred == y)),
           "f1": np.nan if single_class else fast_f1(y, pred),
           "auc": fast_auc(y, prob)}
    out["one_minus_mae"] = (1 - float(np.mean(np.abs(s_true - s_pred)))) if s_pred is not None else np.nan
    return out


# ----------------------------------------------------------------------------
# All models of Phases 6-7, with their predictions on every split
# ----------------------------------------------------------------------------
def build_models(data, modalities):
    models = {}
    for name in modalities + ["text_tfidf"]:
        models[name] = {t: modeling.unimodal(data, name, t) for t in config.TRAITS}
    for fusion_type in modeling.FUSION_TYPES:
        models[fusion_type] = {t: modeling.fuse(data, fusion_type, modalities, t) for t in config.TRAITS}
    return models


def model_metrics(data, result, trait, split):
    mask = modeling.split_mask(data, split)
    score = result.get(f"score_{split}")
    return point_metrics(data.labels[trait][mask], result[f"prob_{split}"],
                         data.scores[trait][mask], score)


# ----------------------------------------------------------------------------
# B1 baselines
# ----------------------------------------------------------------------------
def baselines(data):
    rows, predictions = [], {}
    for trait in config.TRAITS:
        y_train, y_test = data.labels[trait][data.train], data.labels[trait][data.test]
        s_train, s_test = data.scores[trait][data.train], data.scores[trait][data.test]
        dummy_x_train, dummy_x_test = np.zeros((len(y_train), 1)), np.zeros((len(y_test), 1))

        majority = DummyClassifier(strategy="most_frequent").fit(dummy_x_train, y_train)
        prob = majority.predict_proba(dummy_x_test)[:, list(majority.classes_).index(1)] \
            if 1 in majority.classes_ else np.zeros(len(y_test))
        m = utils.classification_metrics(y_test, majority.predict(dummy_x_test), prob)
        rows.append({"baseline": "majority class", "trait": trait, **m, "mae": np.nan,
                     "one_minus_mae": np.nan})

        draws = []
        for seed in range(config.RANDOM_BASELINE_REPEATS):
            rand = DummyClassifier(strategy="stratified", random_state=seed).fit(dummy_x_train, y_train)
            draws.append(utils.classification_metrics(
                y_test, rand.predict(dummy_x_test), rand.predict_proba(dummy_x_test)[:, 1], quiet=True))
        rows.append({"baseline": "stratified random (mean of draws)", "trait": trait,
                     **{k: float(np.nanmean([d[k] for d in draws])) for k in draws[0]},
                     "mae": np.nan, "one_minus_mae": np.nan})

        mean_score = np.full(len(s_test), s_train.mean())
        reg = utils.regression_metrics(s_test, mean_score)
        rows.append({"baseline": "always the train mean", "trait": trait, "acc": np.nan,
                     "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan, **reg})
        predictions[trait] = {"majority": prob, "train_mean": mean_score}
    return pd.DataFrame(rows), predictions


# ----------------------------------------------------------------------------
# B3 bootstrap
# ----------------------------------------------------------------------------
def bootstrap_indices(data):
    """BOOTSTRAP_SAMPLES resamples of the test clips, drawing whole YouTube videos."""
    rng = np.random.RandomState(config.SEED)
    test_positions = np.where(data.test)[0]
    groups = data.df["youtube_id"].values[test_positions]
    members = {g: np.where(groups == g)[0] for g in np.unique(groups)}   # positions inside test
    names = list(members)
    samples = []
    for _ in range(config.BOOTSTRAP_SAMPLES):
        chosen = rng.choice(len(names), size=len(names), replace=True)
        samples.append(np.concatenate([members[names[i]] for i in chosen]))
    return samples


def bootstrap_table(data, predictions, samples):
    """predictions: {model: {trait: (prob_test, score_test or None)}}.
    Returns the CI table and the per-sample mean-over-traits values (for paired differences)."""
    rows, per_sample = [], {}
    for model, by_trait in predictions.items():
        values = {m: np.full((len(samples), len(config.TRAITS)), np.nan) for m in METRICS}
        for k, trait in enumerate(config.TRAITS):
            y = data.labels[trait][data.test]
            s = data.scores[trait][data.test]
            prob, score = by_trait[trait]
            for b, idx in enumerate(samples):
                res = point_metrics(y[idx], prob[idx], s[idx], None if score is None else score[idx])
                for m in METRICS:
                    values[m][b, k] = res[m]
            full = point_metrics(y, prob, s, score)
            for m in METRICS:
                low, high = np.nanpercentile(values[m][:, k], [2.5, 97.5]) \
                    if not np.isnan(values[m][:, k]).all() else (np.nan, np.nan)
                rows.append({"model": model, "scope": trait, "metric": m, "estimate": full[m],
                             "ci_low": low, "ci_high": high})
        per_sample[model] = {m: np.nanmean(values[m], axis=1) for m in METRICS}
        for m in METRICS:
            estimate = np.nanmean([r["estimate"] for r in rows
                                   if r["model"] == model and r["metric"] == m and r["scope"] != "mean"])
            sample_means = per_sample[model][m]
            low, high = np.nanpercentile(sample_means, [2.5, 97.5]) \
                if not np.isnan(sample_means).all() else (np.nan, np.nan)
            rows.append({"model": model, "scope": "mean", "metric": m, "estimate": estimate,
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows), per_sample


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def figure_learning_curve(plt, curve, fusion_type):
    # Repeats of one fraction draw different videos, so their clip counts differ slightly:
    # group by fraction and plot at the average number of clips.
    summary = curve.groupby("fraction")[["test_f1", "test_auc"]].agg(["mean", "min", "max"])
    x = curve.groupby("fraction")["train_clips"].mean().values
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for metric, color, label in [("test_f1", utils.PLOT["series"][0], "mean test F1"),
                                 ("test_auc", utils.PLOT["series"][1], "mean test AUC")]:
        ax.fill_between(x, summary[(metric, "min")], summary[(metric, "max")], color=color, alpha=0.1,
                        linewidth=0)
        ax.plot(x, summary[(metric, "mean")], color=color, marker="o", markersize=6,
                markeredgecolor=utils.PLOT["surface"], markeredgewidth=1.5, label=label)
        ax.annotate(f"{summary[(metric, 'mean')].iloc[-1]:.3f}", xy=(x[-1], summary[(metric, 'mean')].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                    color=utils.PLOT["ink2"])
    ax.axhline(0.5, color=utils.PLOT["axis"], linewidth=1)
    ax.annotate("0.5 = chance level for AUC", xy=(x[0], 0.5), xytext=(0, -12),
                textcoords="offset points", fontsize=8, color=utils.PLOT["muted"])
    ax.set_xlabel("number of training clips")
    ax.set_ylabel("score on the test split")
    ax.set_xlim(0, x[-1] * 1.12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Learning curve of the {fusion_type.replace('_', ' ')} fusion model "
                 f"(band = range over random subsets)"
                 + utils.fake_note(), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curve.png"))
    plt.close(fig)


def figure_confusion(plt, data, result_by_trait, model_name):
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, len(config.TRAITS), figsize=(3.0 * len(config.TRAITS), 3.4))
    for ax, trait in zip(axes, config.TRAITS):
        y = data.labels[trait][data.test]
        pred = (result_by_trait[trait]["prob_test"] >= 0.5).astype(int)
        matrix = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for i in range(2):
            for j in range(2):
                dark = matrix[i, j] > matrix.max() * 0.6
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if dark else utils.PLOT["ink"])
        ax.set_xticks([0, 1], ["low", "high"])
        ax.set_yticks([0, 1], ["low", "high"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true" if trait == config.TRAITS[0] else "")
        ax.set_title(trait.capitalize(), fontsize=10)
        ax.grid(False)
    fig.suptitle(f"Confusion matrices of the best model ({model_name.replace('_', ' ')}) on the test split"
                 + utils.fake_note(), x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT, "confusion_best.png")
    fig.savefig(path)
    plt.close(fig)
    shutil.copyfile(path, os.path.join(config.RESULTS_DIR, "confusion_best.png"))   # roadmap location


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)    # "mean of empty slice" for metrics that are NaN by design
    start = time.time()
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()
    if not os.path.exists(config.BEST_FUSION_JSON):
        sys.exit(f"Missing {config.BEST_FUSION_JSON} - run `python src/07_fusion.py` first.")
    with open(config.BEST_FUSION_JSON) as f:
        chosen = json.load(f)
    best_fusion, best_single = chosen["fusion_type"], chosen["best_single_modality"]
    data = modeling.Data()
    modalities = data.modalities
    log.info(data.describe())
    log.info(f"best fusion (chosen on validation in Phase 7): {best_fusion}; "
             f"best single modality: {best_single}")

    models = build_models(data, modalities)
    log.info(f"rebuilt {len(models)} models x {len(config.TRAITS)} traits ({time.time() - start:.0f} s)")

    # ---- B1 baselines ----
    base, base_pred = baselines(data)
    save(base, "baselines.csv")

    # ---- B2 overfitting ----
    rows = []
    for name, by_trait in models.items():
        for trait, result in by_trait.items():
            f1 = {split: model_metrics(data, result, trait, split)["f1"] for split in ["train", "val", "test"]}
            rows.append({"model": name, "trait": trait, "train_f1": f1["train"], "val_f1": f1["val"],
                         "test_f1": f1["test"], "train_minus_test": f1["train"] - f1["test"]})
    over = pd.DataFrame(rows)
    over["flag"] = over["train_minus_test"] > config.OVERFIT_GAP
    save(over, "overfitting.csv")
    flagged = over[over.flag]
    log.info(f"B2 overfitting: {len(flagged)} of {len(over)} model/trait pairs have "
             f"train F1 - test F1 > {config.OVERFIT_GAP}")

    # ---- B3 bootstrap ----
    samples = bootstrap_indices(data)
    predictions = {name: {t: (r["prob_test"], r.get("score_test")) for t, r in by_trait.items()}
                   for name, by_trait in models.items()}
    predictions["baseline: majority class"] = {t: (base_pred[t]["majority"], None) for t in config.TRAITS}
    predictions["baseline: train mean"] = {
        t: (np.full(int(data.test.sum()), 0.5), base_pred[t]["train_mean"]) for t in config.TRAITS}
    ci, per_sample = bootstrap_table(data, predictions, samples)
    ci.loc[(ci.model == "baseline: train mean") & (ci.metric != "one_minus_mae"),
           ["estimate", "ci_low", "ci_high"]] = np.nan     # only its MAE is meaningful
    save(ci, "bootstrap_ci.csv")
    rows = []
    for metric in ["f1", "auc"]:
        diff = per_sample[best_fusion][metric] - per_sample[best_single][metric]
        low, high = np.nanpercentile(diff, [2.5, 97.5])
        est = lambda model: float(ci[(ci.model == model) & (ci.scope == "mean") & (ci.metric == metric)].estimate.iloc[0])
        single_ci = ci[(ci.model == best_single) & (ci.scope == "mean") & (ci.metric == metric)].iloc[0]
        rows.append({"metric": metric, "fusion": best_fusion, "single": best_single,
                     "fusion_estimate": est(best_fusion), "single_estimate": est(best_single),
                     "gain": est(best_fusion) - est(best_single), "gain_ci_low": low, "gain_ci_high": high,
                     "gain_significant": bool(low > 0 or high < 0),
                     "fusion_outside_single_ci": bool(est(best_fusion) > single_ci.ci_high
                                                      or est(best_fusion) < single_ci.ci_low)})
    versus = pd.DataFrame(rows)
    save(versus, "fusion_vs_single.csv")

    # ---- B4 label-shuffle test ----
    rng = np.random.RandomState(config.SEED)
    fit_positions = np.where(data.train | data.val)[0]
    rows = []
    for repeat in range(config.SHUFFLE_REPEATS):
        order = rng.permutation(fit_positions)         # one permutation of whole rows: all traits move together
        labels = {t: v.copy() for t, v in data.labels.items()}
        scores = {t: v.copy() for t, v in data.scores.items()}
        for t in config.TRAITS:
            labels[t][fit_positions] = data.labels[t][order]
            scores[t][fit_positions] = data.scores[t][order]
        shuffled = data.variant(labels=labels, scores=scores)   # test labels stay the real ones
        for trait in config.TRAITS:
            result = modeling.fuse(shuffled, best_fusion, modalities, trait)
            m = model_metrics(shuffled, result, trait, "test")
            rows.append({"repeat": repeat + 1, "trait": trait, "test_f1": m["f1"], "test_auc": m["auc"],
                         "test_acc": m["acc"]})
        log.info(f"B4 shuffle repeat {repeat + 1}/{config.SHUFFLE_REPEATS} done ({time.time() - start:.0f} s)")
    shuffle = pd.DataFrame(rows)
    save(shuffle, "shuffle_test.csv")
    shuffle_auc, shuffle_f1 = shuffle.test_auc.mean(), shuffle.test_f1.mean()
    shuffle_ok = abs(shuffle_auc - 0.5) <= config.SHUFFLE_AUC_TOLERANCE

    # ---- B5 learning curve ----
    rows = []
    train_positions = np.where(data.train)[0]
    videos = data.df["youtube_id"].values[train_positions]
    unique_videos = np.unique(videos)
    for fraction in config.LEARNING_CURVE_FRACTIONS:
        repeats = 1 if fraction >= 1 else config.LEARNING_CURVE_REPEATS
        for repeat in range(repeats):
            if fraction >= 1:
                subset = data
            else:
                keep = np.random.RandomState(config.SEED + repeat).choice(
                    unique_videos, size=max(2, int(round(fraction * len(unique_videos)))), replace=False)
                mask = np.zeros(len(data.df), dtype=bool)
                mask[train_positions[np.isin(videos, keep)]] = True
                subset = data.variant(train=mask)
            per_trait = [model_metrics(subset, modeling.fuse(subset, best_fusion, modalities, t), t, "test")
                         for t in config.TRAITS]
            rows.append({"fraction": fraction, "repeat": repeat + 1, "train_clips": int(subset.train.sum()),
                         "test_f1": np.nanmean([m["f1"] for m in per_trait]),
                         "test_auc": np.nanmean([m["auc"] for m in per_trait]),
                         "test_one_minus_mae": np.nanmean([m["one_minus_mae"] for m in per_trait])})
        log.info(f"B5 learning curve {int(fraction * 100)}% done ({time.time() - start:.0f} s)")
    curve = pd.DataFrame(rows)
    save(curve, "learning_curve.csv")
    plt = utils.setup_plots()
    figure_learning_curve(plt, curve, best_fusion)

    # ---- B6 intermediate network loss history ----
    net = modeling.train_intermediate(data, modalities)
    history = pd.DataFrame({"epoch": np.arange(1, len(net["history"]["train"]) + 1),
                            "train_loss": net["history"]["train"], "val_loss": net["history"]["val"]})
    history["best_epoch"] = history.epoch == net["best_epoch"]
    save(history, "intermediate_loss.csv")

    # ---- B7 best model (chosen on validation among all single modalities and fusion types) ----
    val_f1 = over.groupby("model")["val_f1"].mean()
    candidates = val_f1.drop(labels=["text_tfidf"], errors="ignore")
    best_model = candidates.idxmax()
    rows = []
    for trait in config.TRAITS:
        m = model_metrics(data, models[best_model][trait], trait, "test")
        full = modeling.result_rows(data, models[best_model][trait], trait, "test", name=f"{best_model}/{trait}")
        rows.append({"model": best_model, "trait": trait, "acc": full["acc"], "precision": full["precision"],
                     "recall": full["recall"], "f1": full["f1"], "auc": full["auc"],
                     "one_minus_mae": m["one_minus_mae"]})
    save(pd.DataFrame(rows), "best_model_per_trait.csv")
    figure_confusion(plt, data, models[best_model], best_model)

    # ---- B8 visual feature sets one by one ----
    visual_mean_f1 = over[over.model == "visual"].test_f1.mean() if "visual" in models else np.nan
    if not np.isnan(visual_mean_f1) and visual_mean_f1 <= config.VISUAL_WEAK_F1:
        names = [n for n in ["visual_cnn", "visual_vit", "visual_emotion"]
                 if os.path.exists(utils.feature_path(n))]
        extra = utils.load_features(names)
        rows = []
        for name in names:
            for trait in config.TRAITS:
                result = modeling.train_on_matrix(data, extra[name], trait)
                m = model_metrics(data, result, trait, "test")
                rows.append({"feature_set": name, "dims": extra[name].shape[1], "trait": trait,
                             "classifier": result["classifier_name"], "test_f1": m["f1"],
                             "test_auc": m["auc"], "test_one_minus_mae": m["one_minus_mae"]})
        save(pd.DataFrame(rows), "visual_feature_comparison.csv")
        log.info(f"B8 visual mean F1 {visual_mean_f1:.3f} <= {config.VISUAL_WEAK_F1}: feature sets compared")
    else:
        log.info(f"B8 visual mean F1 {visual_mean_f1:.3f} > {config.VISUAL_WEAK_F1}: comparison not needed")

    # ---- summary + acceptance ----
    mean = lambda model, metric: float(ci[(ci.model == model) & (ci.scope == "mean") & (ci.metric == metric)].estimate.iloc[0])
    log.info(f"best model on validation: {best_model}; mean test F1 {mean(best_model, 'f1'):.3f}, "
             f"AUC {mean(best_model, 'auc'):.3f}")
    log.info(f"shuffle test: mean test AUC {shuffle_auc:.3f}, F1 {shuffle_f1:.3f} with shuffled labels")
    log.info(f"finished in {time.time() - start:.0f} s")
    if not shuffle_ok:
        log.info(f"STOP - POSSIBLE LEAKAGE: with shuffled training labels the best fusion model still "
                 f"reaches mean test AUC {shuffle_auc:.3f} (expected 0.5 +/- {config.SHUFFLE_AUC_TOLERANCE}).")
        sys.exit(1)
    log.info(f"ACCEPTANCE CHECK PASSED: shuffle test AUC {shuffle_auc:.3f} is within "
             f"0.5 +/- {config.SHUFFLE_AUC_TOLERANCE}; all diagnostics saved in {OUT}")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
