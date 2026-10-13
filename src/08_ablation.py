"""Phase 8 - Ablation: how much does each modality contribute?

Takes the best fusion type of Phase 7 (chosen on validation, results/best_fusion.json) and
trains it again on every combination of modalities:

    V+A+T                 all three (the reference)
    A+T, V+T, V+A         one modality removed
    V, A, T               one modality alone (for reference)

For each trait:   delta F1 = F1(V+A+T) - F1(subset)   on the test split.
A large positive delta for "A+T" means the model got clearly worse without the visual
features, i.e. the face matters. A delta near zero (or negative) means the removed modality
added nothing (or even hurt).

Outputs
  results/ablation.csv   subset, removed, kind, fusion_type, trait, acc, precision, recall, f1, auc, delta_f1
  results/ablation.png   delta F1 per removed modality and trait

Run:  python src/08_ablation.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import modeling
import utils
import config

log = utils.get_logger("08_ablation")


def short(modalities):
    """['visual', 'audio', 'text'] -> 'V+A+T'"""
    return "+".join(m[0].upper() for m in modalities)


def figure_ablation(plt, drops, modalities, fusion_type, n_test):
    """drops: DataFrame (trait + 'mean') x removed modality, values = delta F1."""
    groups = list(drops.index)
    x = np.arange(len(groups))
    width = 0.8 / len(modalities)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, modality in enumerate(modalities):
        offset = (i - (len(modalities) - 1) / 2) * width
        bars = ax.bar(x + offset, drops[modality].values, width, color=utils.SERIES_COLORS[modality],
                      label=f"without {modality}", edgecolor=utils.PLOT["surface"], linewidth=1.5)
        value = drops.loc["mean", modality]                 # label only the mean group
        if not np.isnan(value):
            ax.annotate(f"{value:+.2f}", xy=(bars[-1].get_x() + width / 2, value),
                        xytext=(0, 3 if value >= 0 else -10), textcoords="offset points",
                        ha="center", fontsize=8, color=utils.PLOT["ink2"])
    ax.axhline(0, color=utils.PLOT["axis"], linewidth=1)
    ax.set_xticks(x, [g.capitalize() for g in groups])
    ax.set_ylabel("drop in test F1 when the modality is removed")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(modalities), fontsize=9)
    fig.suptitle(f"Ablation of the {fusion_type} fusion model ({n_test} test clips): "
                 f"above 0 = the modality helped" + utils.fake_note(),
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "ablation.png"))
    plt.close(fig)


def main():
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()
    if not os.path.exists(config.BEST_FUSION_JSON):
        sys.exit(f"Missing {config.BEST_FUSION_JSON} - run `python src/07_fusion.py` first.")
    with open(config.BEST_FUSION_JSON) as f:
        fusion_type = json.load(f)["fusion_type"]
    data = modeling.Data()
    log.info(data.describe())
    modalities = data.modalities
    if len(modalities) < 2:
        sys.exit("Ablation needs at least two modalities with features.")
    log.info(f"fusion type from Phase 7 (chosen on validation): {fusion_type}")

    subsets = [("full", "none", modalities)]
    subsets += [("one removed", m, [k for k in modalities if k != m]) for m in modalities]
    subsets += [("alone", "all but " + m, [m]) for m in modalities]

    rows = []
    for kind, removed, subset in subsets:
        for trait in config.TRAITS:
            result = modeling.fuse(data, fusion_type, subset, trait)
            metrics = modeling.result_rows(data, result, trait, "test",
                                           name=f"{short(subset)}/{trait}/test")
            rows.append({"subset": short(subset), "removed": removed, "kind": kind,
                         "fusion_type": fusion_type, "trait": trait,
                         **{k: metrics[k] for k in ["acc", "precision", "recall", "f1", "auc"]}})
        log.info(f"{short(subset)}: done")
    table = pd.DataFrame(rows)
    full_f1 = table[table.kind == "full"].set_index("trait")["f1"]
    table["delta_f1"] = table["trait"].map(full_f1) - table["f1"]
    table.to_csv(config.ABLATION_CSV, index=False)

    f1_wide = table.pivot(index="subset", columns="trait", values="f1")[config.TRAITS]
    f1_wide["mean"] = f1_wide.mean(axis=1)
    print("\nTest F1 per subset of modalities:")
    print(f1_wide.reindex([short(s) for _, _, s in subsets]).round(3).to_string())

    removed = table[table.kind == "one removed"]
    drops = removed.pivot(index="trait", columns="removed", values="delta_f1")
    drops = drops.reindex(index=config.TRAITS, columns=modalities)
    drops.loc["mean"] = drops.mean(axis=0, skipna=True)
    print("\nDrop in test F1 when one modality is removed (F1 of V+A+T minus F1 without it):")
    print(drops.round(3).to_string())

    figure_ablation(utils.setup_plots(), drops, modalities, fusion_type, int(data.test.sum()))

    # ---- acceptance: the summary sentence ----
    mean_drop = drops.loc["mean"]
    if mean_drop.isna().all():
        log.info("ACCEPTANCE FAILED: F1 is not defined on this test split (one class only)")
        sys.exit(1)
    worst = mean_drop.idxmax()
    log.info(f"Removing {worst} causes the largest average F1 drop ({mean_drop[worst]:.2f}).")
    for modality in modalities:
        if modality != worst:
            effect = "drop" if mean_drop[modality] >= 0 else "GAIN (the model is better without it)"
            log.info(f"  removing {modality}: average F1 {effect} of {abs(mean_drop[modality]):.2f}")
    log.info(f"ACCEPTANCE CHECK PASSED: saved {config.ABLATION_CSV} ({len(table)} rows) and ablation.png")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
