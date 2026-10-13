"""Phase 7 - Multimodal fusion: combine face, voice and words, compare with the best single modality.

Three ways of combining the modalities:

  EARLY         all features side by side -> StandardScaler -> PCA (95% variance)
                -> LogisticRegression (C chosen on validation). One model sees everything.
  LATE          the best single-modality models of Phase 6 each make a prediction; the
                probabilities are averaged ("late"), or averaged with weights proportional to
                each model's validation AUC ("late_weighted").
  INTERMEDIATE  a small PyTorch network: one branch per modality
                (d -> 128 -> ReLU -> Dropout 0.3 -> 64 -> ReLU), the three 64-number outputs
                are joined (192) -> 64 -> ReLU -> Dropout 0.3 -> 5 outputs, one per trait.
                BCEWithLogitsLoss, Adam (lr 1e-3, weight decay 1e-4), batch 64, at most 100
                epochs, stopped early when the validation loss has not improved for 10 epochs.
                Inputs are standardised with train statistics.

The "best fusion type" used by Phase 8 is chosen by mean VALIDATION F1, never by test results.

Outputs
  results/fusion.csv               fusion_type + the columns of unimodal.csv (test split)
  results/unimodal_vs_fusion.png   test F1 per trait: visual, audio, text, early, late, intermediate
  results/intermediate_loss.png    training curve of the intermediate network
  results/best_fusion.json         the fusion type chosen on validation
  results/models/intermediate_fusion.pt (+ _scalers.joblib)   the trained network

Run:  python src/07_fusion.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

import modeling
import utils
import config

log = utils.get_logger("07_fusion")
SCALERS_FILE = config.FUSION_MODEL_FILE.replace(".pt", "_scalers.joblib")


def figure_loss(plt, net):
    history = net["history"]
    epochs = np.arange(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(epochs, history["train"], color=utils.PLOT["series"][0], label="train loss")
    ax.plot(epochs, history["val"], color=utils.PLOT["series"][1], label="validation loss")
    best = net["best_epoch"]
    ax.scatter([best], [history["val"][best - 1]], s=50, color=utils.PLOT["series"][1],
               edgecolor=utils.PLOT["surface"], linewidth=2, zorder=3)
    ax.annotate(f"best epoch {best} (weights kept)", xy=(best, history["val"][best - 1]),
                xytext=(8, 8), textcoords="offset points", fontsize=9, color=utils.PLOT["ink2"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy loss")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Intermediate fusion network: stopped after {len(epochs)} epochs" + utils.fake_note())
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "intermediate_loss.png"))
    plt.close(fig)


def figure_unimodal_vs_fusion(plt, f1, series, n_test):
    """f1: DataFrame trait x series (test F1)."""
    x = np.arange(len(config.TRAITS))
    width = 0.84 / len(series)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, name in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, f1[name].values, width, color=utils.SERIES_COLORS[name], label=name,
               edgecolor=utils.PLOT["surface"], linewidth=1.5)
    for k, trait in enumerate(config.TRAITS):               # label only the best bar per trait
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
    fig.suptitle(f"Single modalities vs fusion: test F1 per trait ({n_test} test clips)"
                 + utils.fake_note(), x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "unimodal_vs_fusion.png"))
    plt.close(fig)


def main():
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()
    data = modeling.Data()
    log.info(data.describe())
    modalities = data.modalities
    if len(modalities) < 2:
        sys.exit("Fusion needs at least two modalities with features.")

    # ---- single modalities (same models as Phase 6), for the comparison ----
    test_f1 = pd.DataFrame(index=config.TRAITS, dtype=float)
    val_f1 = pd.DataFrame(index=config.TRAITS, dtype=float)
    for modality in modalities:
        for trait in config.TRAITS:
            result = modeling.unimodal(data, modality, trait)
            test_f1.loc[trait, modality] = modeling.result_rows(data, result, trait, "test")["f1"]
            val_f1.loc[trait, modality] = modeling.evaluate(
                data.labels[trait][data.val], result["prob_val"])["f1"]

    # ---- the fusion types ----
    rows = []
    for fusion_type in modeling.FUSION_TYPES:
        for trait in config.TRAITS:
            result = modeling.fuse(data, fusion_type, modalities, trait)
            metrics = modeling.result_rows(data, result, trait, "test",
                                           name=f"{fusion_type}/{trait}/test")
            rows.append({"fusion_type": fusion_type, "modality": "+".join(modalities),
                         "classifier": result["classifier_name"], "trait": trait, **metrics})
            test_f1.loc[trait, fusion_type] = metrics["f1"]
            val_f1.loc[trait, fusion_type] = modeling.evaluate(
                data.labels[trait][data.val], result["prob_val"])["f1"]
        log.info(f"{fusion_type}: done")
    table = pd.DataFrame(rows)[["fusion_type", "modality", "classifier", "trait"]
                               + modeling.METRIC_COLUMNS]
    table.to_csv(config.FUSION_CSV, index=False)
    print(table.drop(columns="modality").round(3).to_string(index=False))

    # ---- intermediate network: loss curve + saved model ----
    net = modeling.train_intermediate(data, modalities)
    torch.save({"state_dict": net["model"].state_dict(), "modalities": modalities,
                "input_dims": {m: data.matrix(m).shape[1] for m in modalities},
                "feature_sets": {m: config.MODALITIES[m] for m in modalities},
                "traits": config.TRAITS}, config.FUSION_MODEL_FILE)
    joblib.dump(net["scalers"], SCALERS_FILE)
    plt = utils.setup_plots()
    figure_loss(plt, net)
    figure_unimodal_vs_fusion(plt, test_f1, modalities + ["early", "late", "intermediate"],
                              int(data.test.sum()))

    # ---- choose the best fusion type and the best modality on VALIDATION ----
    val_mean = val_f1.mean(axis=0, skipna=True)
    best_fusion = val_mean[modeling.FUSION_TYPES].idxmax()
    best_modality = val_mean[modalities].idxmax()
    with open(config.BEST_FUSION_JSON, "w") as f:
        json.dump({"fusion_type": best_fusion, "best_single_modality": best_modality,
                   "chosen_by": "mean validation F1 over the five traits",
                   "mean_val_f1": {k: round(float(v), 4) for k, v in val_mean.items()},
                   "mean_test_f1": {k: round(float(v), 4) for k, v in test_f1.mean().items()}},
                  f, indent=2)

    # ---- acceptance: per trait, is the best fusion better than the best single modality? ----
    print("\nPer trait (test F1): best single modality vs best fusion type")
    print(f"{'trait':<18} {'best single':<22} {'best fusion':<26} {'difference':>10}  fusion better?")
    wins = 0
    for trait in config.TRAITS:
        single, fused = test_f1.loc[trait, modalities], test_f1.loc[trait, modeling.FUSION_TYPES]
        if single.isna().all() or fused.isna().all():
            print(f"{trait:<18} F1 not defined on this test split (one class only)")
            continue
        diff = fused.max() - single.max()
        wins += diff > 0
        print(f"{trait:<18} {single.idxmax() + f' {single.max():.3f}':<22} "
              f"{fused.idxmax() + f' {fused.max():.3f}':<26} {diff:>+10.3f}  "
              f"{'yes' if diff > 0 else 'no'}")
    print(f"fusion beats the best single modality on {wins} of {len(config.TRAITS)} traits")

    means = test_f1.mean()
    log.info("mean test F1 over the five traits: "
             + ", ".join(f"{name} {value:.3f}" for name, value in means.items()))
    log.info(f"chosen on VALIDATION: best fusion type = {best_fusion} "
             f"(val F1 {val_mean[best_fusion]:.3f}, test F1 {means[best_fusion]:.3f}); "
             f"best single modality = {best_modality} "
             f"(val F1 {val_mean[best_modality]:.3f}, test F1 {means[best_modality]:.3f})")
    log.info(f"fair comparison (both chosen on validation): {best_fusion} fusion vs {best_modality} "
             f"alone = {means[best_fusion] - means[best_modality]:+.3f} mean test F1")

    in_range, n_nan = modeling.check_metric_range(table)
    if not in_range:
        log.info("ACCEPTANCE FAILED: a metric is outside [0, 1]")
        sys.exit(1)
    log.info(f"ACCEPTANCE CHECK PASSED: comparison table printed, {len(table)} rows, metrics in [0, 1]"
             f" ({n_nan} NaN: the intermediate network has no raw-score output, so its MAE is empty)")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
