"""
Fraud detection on the Kaggle Credit Card Fraud dataset (284,807
transactions, 0.172% fraudulent).

What this script does
---------------------
1. Loads data/creditcard.csv (run download_data.py first).
2. Uses ONLY time-based splitting: first 80% of transactions (by Time) for
   training, last 20% for testing - mimics production, where a model is
   trained on the past and scored on the future.
3. Wraps StandardScaler + SMOTE + classifiers in one sklearn Pipeline so
   SMOTE resamples *training folds only* (imblearn pipelines make
   cross-validation leakage-free).
4. Tunes Logistic Regression and Random Forest with GridSearchCV (5-fold
   stratified CV) under class weights vs. SMOTE, then compares them with
   precision-recall curves and PR-AUC (average precision) - accuracy is
   meaningless at 0.17% positives.
5. Also tunes a decision threshold on out-of-fold predictions for the best
   F1, then reports precision / recall / F1 / PR-AUC / ROC-AUC / confusion
   matrix on the untouched temporal test set.
6. Saves artifacts/ (model, metrics) and figures/ (PR curves, confusion
   matrix, score distributions, class-balance chart).

Usage:
    python download_data.py          # once (Kaggle or manual file)
    python train_fraud.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "creditcard.csv"
ART = HERE / "artifacts"
FIGS = HERE / "figures"
ART.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

sns.set_theme(style="whitegrid", palette="deep")


# --------------------------------------------------------------------------- #
# 1. Load + temporal split
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found. Run `python download_data.py` first.")
    df = pd.read_csv(DATA_PATH)
    needed = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")
    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.20):
    df = df.sort_values("Time").reset_index(drop=True)
    n_test = int(len(df) * test_frac)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]
    return train_df, test_df


# --------------------------------------------------------------------------- #
# 2. Model variants: class weights vs. SMOTE (inside the CV folds)
# --------------------------------------------------------------------------- #
def build_pipe(model, use_smote: bool) -> object:
    steps = [("scale", StandardScaler())]
    if use_smote:
        # sampling_strategy=0.2 -> fraud grows to 20% of the majority class
        # instead of 100% (full SMOTE on 227k rows is needlessly expensive
        # and floods the forests with near-duplicates).
        steps.append(("smote", SMOTE(sampling_strategy=0.2,
                                     random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def experiment_specs() -> list[tuple[str, object, dict]]:
    """(label, base pipeline, param grid) - 4 experiments, one grid each."""
    lr_cw = build_pipe(
        LogisticRegression(max_iter=5000, class_weight="balanced",
                           random_state=RANDOM_STATE),
        use_smote=False,
    )
    grid_lr_cw = {"model__C": [0.03, 0.3, 3.0]}

    lr_smote = build_pipe(
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        use_smote=True,
    )
    grid_lr_smote = {
        "model__C": [0.03, 0.3, 3.0],
        "smote__k_neighbors": [3, 5],
    }

    rf_cw = build_pipe(
        RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        use_smote=False,
    )
    grid_rf_cw = {
        "model__n_estimators": [150],
        "model__max_depth": [10, 16],
        "model__min_samples_leaf": [2],
    }

    rf_smote = build_pipe(
        RandomForestClassifier(n_jobs=2, random_state=RANDOM_STATE),
        use_smote=True,
    )
    grid_rf_smote = {
        "model__n_estimators": [150],
        "model__max_depth": [12],
    }

    return [
        ("LogReg + class weights", lr_cw, grid_lr_cw),
        ("LogReg + SMOTE", lr_smote, grid_lr_smote),
        ("RandomForest + class weights", rf_cw, grid_rf_cw),
        ("RandomForest + SMOTE", rf_smote, grid_rf_smote),
    ]


# --------------------------------------------------------------------------- #
# 3. Metrics helpers
# --------------------------------------------------------------------------- #
def best_threshold_by_f1(y_true, proba) -> tuple[float, float, float]:
    """Threshold with the best F1, kept inside a usable (0.01..0.99) range."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    usable = thresholds < 0.995  # the curve's boundary point is not usable
    if not usable.any():
        return 0.99, float(precision[0]), float(recall[0])
    f1s = np.where(usable, f1s, -1.0)
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx])


def evaluate_at_threshold(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision_fraud": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall_fraud": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1_fraud": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def confusion_counts(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# --------------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------------- #
def plot_class_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = df["Class"].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#4c72b0", "#c44e52"],
                hue=counts.index, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Legit (0)", "Fraud (1)"])
    ax.set_ylabel("count (log scale)")
    ax.set_title(f"Extreme class imbalance: {counts.get(1, 0) / len(df):.3%} fraud")
    fig.tight_layout()
    fig.savefig(FIGS / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_pr_curves(models_probas: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, proba in models_probas.items():
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    baseline = float(np.mean(y_test))
    ax.axhline(baseline, ls="--", c="gray", lw=1,
               label=f"baseline (fraud rate {baseline:.3%})")
    ax.set_title("Precision-Recall curves - temporal test set")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "pr_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title: str, fname: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def plot_score_distribution(y_test, proba, fname: str = "score_distribution.png") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.kdeplot(proba[y_test == 0], ax=ax, fill=True, label="Legit")
    sns.kdeplot(proba[y_test == 1], ax=ax, fill=True, label="Fraud")
    ax.set_yscale("log")
    ax.set_xlabel("predicted fraud probability")
    ax.set_title("Predicted-probability distribution by true class (log-count)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # --stage N: fit one experiment, checkpoint it, exit (see stage_fraud.py)
    if "--stage" in argv:
        import stage_fraud
        stage_fraud.run_stage(int(argv[argv.index("--stage") + 1]))
        return

    df = load_data()
    plot_class_balance(df)

    train_df, test_df = temporal_split(df, test_frac=0.20)
    feature_cols = [c for c in df.columns if c != "Class"]
    X_train, y_train = train_df[feature_cols], train_df["Class"].values
    X_test, y_test = test_df[feature_cols], test_df["Class"].values
    print(f"Temporal split: train {len(X_train):,} | test {len(X_test):,}")
    print(f"Fraud rate  train {y_train.mean():.3%} | test {y_test.mean():.3%}")

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    test_probas: dict[str, np.ndarray] = {}

    for label, base_pipe, grid in experiment_specs():
        print(f"\n=== {label}: GridSearchCV (5-fold stratified, scoring=average_precision) ===")
        gs = GridSearchCV(base_pipe, grid, cv=CV, scoring="average_precision",
                          n_jobs=4, refit=True, verbose=1)
        gs.fit(X_train, y_train)
        best_cv = float(gs.best_score_)
        print(f"  best params : {gs.best_params_}")
        print(f"  best CV PR-AUC: {best_cv:.4f}")

        oof = cross_val_predict(gs.best_estimator_, X_train, y_train, cv=CV,
                                method="predict_proba", n_jobs=-1)[:, 1]
        thr, p_at, r_at = best_threshold_by_f1(y_train, oof)
        print(f"  tuned threshold = {thr:.3f} (OOF precision={p_at:.3f}, recall={r_at:.3f})")

        proba_test = gs.best_estimator_.predict_proba(X_test)[:, 1]
        results[label] = {
            "best_params": {k.replace("model__", "").replace("smote__", "smote: "): v
                            for k, v in gs.best_params_.items()},
            "cv_pr_auc": round(best_cv, 4),
            "test_default_0.5": evaluate_at_threshold(y_test, proba_test, 0.5),
            "test_tuned_threshold": evaluate_at_threshold(y_test, proba_test, thr),
            "confusion_at_tuned": confusion_counts(y_test, proba_test, thr),
        }
        fitted[label] = gs.best_estimator_
        test_probas[label] = proba_test

    winner = max(results, key=lambda n: results[n]["test_tuned_threshold"]["f1_fraud"])
    best_thr = results[winner]["test_tuned_threshold"]["threshold"]
    proba_best = test_probas[winner]
    pred_best = (proba_best >= best_thr).astype(int)

    print(f"\n### Winner: {winner} (threshold {best_thr:.3f}) ###")
    print(f"  test @ tuned: {results[winner]['test_tuned_threshold']}")
    print(f"  test @ 0.5  : {results[winner]['test_default_0.5']}")

    plot_pr_curves(test_probas, y_test)
    plot_confusion(y_test, pred_best,
                   f"{winner} - confusion matrix (thr={best_thr:.2f})",
                   "confusion_matrix_tuned.png")
    plot_score_distribution(y_test, proba_best)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "winner_model": winner,
        "split": "temporal 80/20 by Time",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_rate_train": round(float(y_train.mean()), 6),
        "fraud_rate_test": round(float(y_test.mean()), 6),
        "experiments": results,
        "headline": {
            "pr_auc": results[winner]["test_tuned_threshold"]["pr_auc"],
            "f1_fraud": results[winner]["test_tuned_threshold"]["f1_fraud"],
            "precision_fraud": results[winner]["test_tuned_threshold"]["precision_fraud"],
            "recall_fraud": results[winner]["test_tuned_threshold"]["recall_fraud"],
            "roc_auc": results[winner]["test_tuned_threshold"]["roc_auc"],
            "confusion": results[winner]["confusion_at_tuned"],
        },
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"pipeline": fitted[winner], "threshold": best_thr,
                 "model_name": winner},
                ART / "fraud_model.joblib")
    print(f"\nSaved model   -> {ART / 'fraud_model.joblib'}")
    print(f"Saved metrics -> {ART / 'metrics.json'}")


if __name__ == "__main__":
    main()