"""Shared modelling code for Phases 6-8 (unimodal models, fusion, ablation).

Rules that hold everywhere in this file:
- models are FIT on the train split only,
- every choice (C, alpha, classifier type, fusion type, when to stop training) is made on the
  VALIDATION split,
- the test split is used once, to report the final numbers.
"""
import copy
import itertools

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import utils
import config

log = utils.get_logger("modeling")
FUSION_TYPES = ["early", "late", "late_weighted", "intermediate"]
_data_versions = itertools.count()          # every Data object gets its own number (see variant())


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class Data:
    """splits.csv + all feature files, checked to be aligned row by row."""

    def __init__(self):
        self.df = utils.load_splits()
        self.modalities = [m for m, sets in config.MODALITIES.items() if sets]
        names = sorted({n for m in self.modalities for n in config.MODALITIES[m]} | {"text_tfidf"})
        self.features = utils.load_features(names)          # hard check: ids.csv == splits.csv
        self.train = (self.df["split"] == "train").values
        self.val = (self.df["split"] == "val").values
        self.test = (self.df["split"] == "test").values
        self.labels = {t: self.df[f"{t}_bin"].values for t in config.TRAITS}     # 0 / 1
        self.scores = {t: self.df[t].values for t in config.TRAITS}             # raw, 0..1
        self.version = next(_data_versions)

    def variant(self, train=None, labels=None, scores=None):
        """A copy with a smaller train split and/or replaced labels (used by the diagnostics:
        learning curve, label-shuffle test). It gets a new version number, so no model trained
        on the original data can be reused for it."""
        other = copy.copy(self)
        if train is not None:
            other.train = train
        if labels is not None:
            other.labels = labels
        if scores is not None:
            other.scores = scores
        other.version = next(_data_versions)
        return other

    def matrix(self, modality):
        """All feature sets of one modality side by side (or the TF-IDF baseline)."""
        if modality == "text_tfidf":
            return self.features["text_tfidf"]
        return np.hstack([self.features[n] for n in config.MODALITIES[modality]])

    def describe(self):
        sizes = {m: self.matrix(m).shape[1] for m in self.modalities}
        return (f"{len(self.df)} clips (train {self.train.sum()}, val {self.val.sum()}, "
                f"test {self.test.sum()}); inputs per modality: {sizes}")


# ----------------------------------------------------------------------------
# Building blocks: scaler -> (PCA) -> model, chosen on validation
# ----------------------------------------------------------------------------
def pipeline(estimator, n_inputs, force_pca=False):
    """StandardScaler -> PCA(95% variance) if there are many inputs -> estimator."""
    steps = [StandardScaler()]
    if force_pca or n_inputs > config.PCA_MIN_DIMS:
        steps.append(PCA(n_components=config.PCA_VARIANCE, random_state=config.SEED))
    return make_pipeline(*steps, estimator)


def fit_classifier(X_train, y_train, X_val, y_val, use_mlp=True, force_pca=False):
    """Fit LogisticRegression for every C (and optionally an MLP) on train; keep the candidate
    with the best validation F1. Returns (fitted model, its name, its validation F1)."""
    candidates = [(f"logreg(C={C})",
                   LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
                  for C in config.LOGREG_C_GRID]
    if use_mlp:
        candidates.append(("mlp", MLPClassifier(hidden_layer_sizes=config.MLP_HIDDEN,
                                                early_stopping=True, max_iter=500,
                                                random_state=config.SEED)))
    best = None
    for name, estimator in candidates:
        model = pipeline(estimator, X_train.shape[1], force_pca)
        try:
            model.fit(X_train, y_train)
        except ValueError as error:                    # e.g. too few clips for the MLP's own split
            log.info(f"  {name} skipped: {error}")
            continue
        metrics = utils.classification_metrics(y_val, model.predict(X_val), quiet=True)
        # F1 is undefined when validation has one class only -> fall back to accuracy.
        score = metrics["f1"] if not np.isnan(metrics["f1"]) else metrics["acc"]
        if best is None or score > best[2]:
            best = (model, name, score)
    if best is None:
        raise RuntimeError("no classifier could be fitted")
    return best


def fit_ridge(X_train, s_train, X_val, s_val, force_pca=False):
    """Ridge regression on the raw trait score; alpha chosen by validation MAE."""
    best = None
    for alpha in config.RIDGE_ALPHA_GRID:
        model = pipeline(Ridge(alpha=alpha), X_train.shape[1], force_pca).fit(X_train, s_train)
        mae = float(np.mean(np.abs(predict_score(model, X_val) - s_val)))
        if best is None or mae < best[1]:
            best = (model, mae)
    return best[0]


def positive_probability(model, X):
    return model.predict_proba(X)[:, 1]


def predict_score(model, X):
    return np.clip(model.predict(X), 0.0, 1.0)         # trait scores live in [0, 1]


def evaluate(y_true, prob, s_true=None, s_pred=None, name=""):
    """Metric row from probabilities (decision at 0.5) and, if given, predicted raw scores."""
    row = utils.classification_metrics(y_true, (prob >= 0.5).astype(int), prob, name=name)
    if s_pred is not None:
        row.update(utils.regression_metrics(s_true, s_pred))
    else:
        row.update({"mae": np.nan, "one_minus_mae": np.nan})
    return row


# ----------------------------------------------------------------------------
# One model on one input matrix (used for a single modality and for early fusion)
# ----------------------------------------------------------------------------
def train_on_matrix(data, X, trait, use_mlp=True, force_pca=False):
    """Returns a dict with the fitted models and their validation / test predictions."""
    y, s = data.labels[trait], data.scores[trait]
    clf, name, _ = fit_classifier(X[data.train], y[data.train], X[data.val], y[data.val],
                                  use_mlp=use_mlp, force_pca=force_pca)
    ridge = fit_ridge(X[data.train], s[data.train], X[data.val], s[data.val], force_pca=force_pca)
    return {
        "classifier": clf, "classifier_name": name, "ridge": ridge,
        "prob_train": positive_probability(clf, X[data.train]),
        "prob_val": positive_probability(clf, X[data.val]),
        "prob_test": positive_probability(clf, X[data.test]),
        "score_train": predict_score(ridge, X[data.train]),
        "score_val": predict_score(ridge, X[data.val]),
        "score_test": predict_score(ridge, X[data.test]),
    }


_unimodal_cache = {}


def unimodal(data, modality, trait):
    """Best model of one modality for one trait (trained once, then remembered)."""
    key = (data.version, modality, trait)
    if key not in _unimodal_cache:
        use_mlp = modality != "text_tfidf"             # the TF-IDF baseline is LogisticRegression only
        _unimodal_cache[key] = train_on_matrix(data, data.matrix(modality), trait, use_mlp=use_mlp)
    return _unimodal_cache[key]


# ----------------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------------
def early_fusion(data, modalities, trait):
    """All modalities' features side by side -> scaler -> PCA(95%) -> LogisticRegression."""
    X = np.hstack([data.matrix(m) for m in modalities])
    out = train_on_matrix(data, X, trait, use_mlp=False, force_pca=True)
    out["classifier_name"] = "early: " + out["classifier_name"]
    return out


def late_fusion(data, modalities, trait, weighted):
    """Average the predictions of the best unimodal models.
    weighted=True: each model counts in proportion to its validation AUC."""
    parts = [unimodal(data, m, trait) for m in modalities]
    weights = np.ones(len(parts))
    if weighted:
        y_val = data.labels[trait][data.val]
        aucs = np.array([utils.classification_metrics(y_val, p["prob_val"] >= 0.5, p["prob_val"],
                                                      quiet=True)["auc"] for p in parts])
        if not np.isnan(aucs).any():                   # AUC undefined (one class in val) -> equal weights
            weights = aucs
    weights = weights / weights.sum()
    mix = lambda key: sum(w * p[key] for w, p in zip(weights, parts))
    return {
        "classifier_name": ("late weighted: " if weighted else "late average: ")
                           + ", ".join(f"{m}={w:.2f}" for m, w in zip(modalities, weights)),
        "prob_train": mix("prob_train"), "prob_val": mix("prob_val"), "prob_test": mix("prob_test"),
        "score_train": mix("score_train"), "score_val": mix("score_val"),
        "score_test": mix("score_test"),
    }


class FusionNet(torch.nn.Module):
    """Intermediate fusion: one small branch per modality, joined, then a shared head that
    predicts all five traits at once (multi-label)."""

    def __init__(self, input_dims):
        super().__init__()
        hidden, out = config.FUSION_BRANCH_DIMS
        self.branches = torch.nn.ModuleDict({
            modality: torch.nn.Sequential(
                torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(config.FUSION_DROPOUT),
                torch.nn.Linear(hidden, out), torch.nn.ReLU())
            for modality, dim in input_dims.items()})
        self.head = torch.nn.Sequential(
            torch.nn.Linear(out * len(input_dims), config.FUSION_HEAD_DIM), torch.nn.ReLU(),
            torch.nn.Dropout(config.FUSION_DROPOUT),
            torch.nn.Linear(config.FUSION_HEAD_DIM, len(config.TRAITS)))

    def forward(self, inputs):
        joined = torch.cat([self.branches[m](inputs[m]) for m in self.branches], dim=1)
        return self.head(joined)                       # logits, one per trait


_intermediate_cache = {}


def train_intermediate(data, modalities):
    """Train the fusion network on train, stop on validation loss. Returns a dict with the
    model, the per-modality scalers, the loss history and val / test probabilities (N, 5)."""
    key = (data.version, tuple(modalities))
    if key in _intermediate_cache:
        return _intermediate_cache[key]
    utils.set_seed()
    scalers = {m: StandardScaler().fit(data.matrix(m)[data.train]) for m in modalities}  # train stats

    def tensors(mask):
        return {m: torch.tensor(scalers[m].transform(data.matrix(m)[mask]), dtype=torch.float32)
                for m in modalities}

    def targets(mask):
        return torch.tensor(np.column_stack([data.labels[t][mask] for t in config.TRAITS]),
                            dtype=torch.float32)

    x_train, x_val, x_test = tensors(data.train), tensors(data.val), tensors(data.test)
    x_train_eval = {m: x.clone() for m, x in x_train.items()}
    y_train, y_val = targets(data.train), targets(data.val)

    model = FusionNet({m: data.matrix(m).shape[1] for m in modalities})   # small: runs on the CPU
    optimiser = torch.optim.Adam(model.parameters(), lr=config.FUSION_LR,
                                 weight_decay=config.FUSION_WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, waited = float("inf"), None, 0, 0
    n_train = len(y_train)
    for epoch in range(1, config.FUSION_MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n_train)
        total = 0.0
        for start in range(0, n_train, config.FUSION_BATCH_SIZE):
            batch = order[start:start + config.FUSION_BATCH_SIZE]
            optimiser.zero_grad()
            loss = loss_fn(model({m: x_train[m][batch] for m in modalities}), y_train[batch])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        history["train"].append(total / n_train)
        history["val"].append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config.FUSION_PATIENCE:       # early stopping
                break
    model.load_state_dict(best_state)                  # go back to the best epoch
    model.eval()
    with torch.no_grad():
        prob_train = torch.sigmoid(model(x_train_eval)).numpy()
        prob_val = torch.sigmoid(model(x_val)).numpy()
        prob_test = torch.sigmoid(model(x_test)).numpy()
    out = {"model": model, "scalers": scalers, "history": history, "best_epoch": best_epoch,
           "prob_train": prob_train, "prob_val": prob_val, "prob_test": prob_test,
           "modalities": list(modalities)}
    _intermediate_cache[key] = out
    return out


def intermediate_fusion(data, modalities, trait):
    net = train_intermediate(data, modalities)
    column = config.TRAITS.index(trait)
    return {"classifier_name": f"intermediate net (best epoch {net['best_epoch']})",
            "prob_train": net["prob_train"][:, column],
            "prob_val": net["prob_val"][:, column], "prob_test": net["prob_test"][:, column],
            "score_train": None, "score_val": None, "score_test": None}   # no raw-score output


def fuse(data, fusion_type, modalities, trait):
    """Run one fusion type on a set of modalities for one trait."""
    if fusion_type == "early":
        return early_fusion(data, modalities, trait)
    if fusion_type == "late":
        return late_fusion(data, modalities, trait, weighted=False)
    if fusion_type == "late_weighted":
        return late_fusion(data, modalities, trait, weighted=True)
    if fusion_type == "intermediate":
        return intermediate_fusion(data, modalities, trait)
    raise ValueError(f"unknown fusion type {fusion_type}")


def split_mask(data, split):
    return {"train": data.train, "val": data.val, "test": data.test}[split]


def result_rows(data, result, trait, split, name=""):
    """Metric dict of a result (from unimodal() or fuse()) on 'train', 'val' or 'test'."""
    mask = split_mask(data, split)
    return evaluate(data.labels[trait][mask], result[f"prob_{split}"],
                    data.scores[trait][mask], result[f"score_{split}"], name=name)


# ----------------------------------------------------------------------------
# Checks shared by the result tables
# ----------------------------------------------------------------------------
METRIC_COLUMNS = ["acc", "precision", "recall", "f1", "auc", "mae", "one_minus_mae"]


def check_metric_range(table):
    """Acceptance: every metric is in [0, 1] (NaN is allowed and counted)."""
    values = table[METRIC_COLUMNS].to_numpy(dtype=float)
    finite = values[~np.isnan(values)]
    ok = bool(((finite >= 0) & (finite <= 1)).all())
    return ok, int(np.isnan(values).sum())
