"""Phase 6 - One model per modality and per trait.

For each modality (visual, audio, text) and each Big Five trait:

  classification (trait above / below the train median)
    inputs -> StandardScaler -> PCA(95% variance) if more than 512 inputs -> classifier
    candidates: LogisticRegression (balanced classes) with C in {0.01, 0.1, 1, 10}, and an
    MLP with one hidden layer of 128 units. All are fitted on TRAIN; the one with the best
    VALIDATION F1 is kept and evaluated once on TEST.

  regression (raw trait score 0..1)
    same inputs -> Ridge, alpha chosen on validation -> MAE and the ChaLearn score 1 - MAE.

Extra baseline: text TF-IDF + LogisticRegression (modality "text_tfidf").

Outputs
  results/unimodal.csv     modality, classifier, trait, acc, precision, recall, f1, auc, mae, one_minus_mae
  results/unimodal_f1.png  test F1 per trait and modality
  results/models/unimodal_<modality>_<trait>_clf.joblib / _ridge.joblib   the fitted models

Run:  python src/06_unimodal.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

import modeling
import utils
import config

log = utils.get_logger("06_unimodal")


def figure_unimodal_f1(plt, table, modalities, n_test):
    """Grouped bars: one group per trait, one bar per modality (+ the TF-IDF baseline)."""
    series = modalities + ["text_tfidf"]
    names = {"text_tfidf": "text TF-IDF baseline"}
    x = np.arange(len(config.TRAITS))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    f1 = table.pivot(index="trait", columns="modality", values="f1").reindex(config.TRAITS)
    for i, modality in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, f1[modality].values, width, color=utils.SERIES_COLORS[modality],
               label=names.get(modality, modality), edgecolor=utils.PLOT["surface"], linewidth=1.5)
    # Label only the best bar of each trait: enough to read the chart, without clutter.
    for k, trait in enumerate(config.TRAITS):
        row = f1.loc[trait, series].astype(float)
        if row.notna().any():
            best = row.idxmax()
            ax.annotate(f"{row[best]:.2f}",
                        xy=(k + (series.index(best) - (len(series) - 1) / 2) * width, row[best]),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                        color=utils.PLOT["ink2"])
    ax.set_xticks(x, [t.capitalize() for t in config.TRAITS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 on the test split")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series), fontsize=9)
    fig.suptitle(f"Single-modality models: test F1 per trait ({n_test} test clips)"
                 + utils.fake_note(), x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "unimodal_f1.png"))
    plt.close(fig)


def main():
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()
    data = modeling.Data()
    log.info(data.describe())

    rows = []
    for modality in data.modalities + ["text_tfidf"]:
        for trait in config.TRAITS:
            result = modeling.unimodal(data, modality, trait)
            metrics = modeling.result_rows(data, result, trait, "test", name=f"{modality}/{trait}/test")
            rows.append({"modality": modality, "classifier": result["classifier_name"],
                         "trait": trait, **metrics})
            stem = os.path.join(config.MODELS_DIR, f"unimodal_{modality}_{trait}")
            joblib.dump(result["classifier"], stem + "_clf.joblib")
            joblib.dump(result["ridge"], stem + "_ridge.joblib")
        log.info(f"{modality}: done")

    table = pd.DataFrame(rows)[["modality", "classifier", "trait"] + modeling.METRIC_COLUMNS]
    table.to_csv(config.UNIMODAL_CSV, index=False)
    print(table.round(3).to_string(index=False))

    figure_unimodal_f1(utils.setup_plots(), table, data.modalities, int(data.test.sum()))

    # ---- ranking + acceptance check ----
    ranking = table.groupby("modality")[["f1", "auc", "acc", "one_minus_mae"]].mean()
    ranking = ranking.sort_values("f1", ascending=False)
    log.info("modalities ranked by mean test F1 over the five traits:")
    for position, (modality, row) in enumerate(ranking.iterrows(), start=1):
        log.info(f"  {position}. {modality:<11} F1 {row.f1:.3f}   AUC {row.auc:.3f}   "
                 f"acc {row.acc:.3f}   1-MAE {row.one_minus_mae:.3f}")

    in_range, n_nan = modeling.check_metric_range(table)
    if len(table) < 15 or not in_range:
        log.info(f"ACCEPTANCE FAILED: {len(table)} rows, all metrics in [0,1]: {in_range}")
        sys.exit(1)
    log.info(f"ACCEPTANCE CHECK PASSED: {len(table)} rows, all metrics in [0, 1]"
             + (f" ({n_nan} NaN, see warnings above)" if n_nan else ", no NaN"))
    log.info(f"saved {config.UNIMODAL_CSV}, unimodal_f1.png and "
             f"{2 * len(table)} models in {config.MODELS_DIR}")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
