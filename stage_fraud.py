"""
Stage 1: fit ONE experiment and save its result to artifacts/partial_*.json.

Called by train_fraud.py --stage once per experiment so a slow machine can
checkpoint between experiments (results are merged later by --collect).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_predict

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

import train_fraud as tf  # noqa: E402


def run_stage(idx: int) -> None:
    label, base_pipe, grid = tf.experiment_specs()[idx]
    print(f"[stage] {idx}: {label} | grid={grid}", flush=True)

    df = tf.load_data()
    train_df, test_df = tf.temporal_split(df, test_frac=0.20)
    feature_cols = [c for c in df.columns if c != "Class"]
    X_train, y_train = train_df[feature_cols], train_df["Class"].values
    X_test, y_test = test_df[feature_cols], test_df["Class"].values

    gs = GridSearchCV(base_pipe, grid, cv=tf.CV, scoring="average_precision",
                      n_jobs=4, refit=True)
    gs.fit(X_train, y_train)
    best_cv = float(gs.best_score_)
    print(f"[stage] best params: {gs.best_params_} | CV PR-AUC {best_cv:.4f}",
          flush=True)

    oof = cross_val_predict(gs.best_estimator_, X_train, y_train, cv=tf.CV,
                            method="predict_proba", n_jobs=4)[:, 1]
    thr, p_at, r_at = tf.best_threshold_by_f1(y_train, oof)
    print(f"[stage] threshold {thr:.3f} (OOF P={p_at:.3f} R={r_at:.3f})", flush=True)

    proba_test = gs.best_estimator_.predict_proba(X_test)[:, 1]

    payload = {
        "label": label,
        "best_params": {k.replace("model__", "").replace("smote__", "smote: "): v
                        for k, v in gs.best_params_.items()},
        "cv_pr_auc": round(best_cv, 4),
        "test_default_0.5": tf.evaluate_at_threshold(y_test, proba_test, 0.5),
        "test_tuned_threshold": tf.evaluate_at_threshold(y_test, proba_test, thr),
        "confusion_at_tuned": tf.confusion_counts(y_test, proba_test, thr),
    }
    (ART / f"partial_{idx}.json").write_text(json.dumps(payload, indent=2))
    joblib_dump(gs.best_estimator_, ART / f"partial_{idx}.joblib", thr, label)
    print(f"[stage] saved partial_{idx}.*", flush=True)


def joblib_dump(pipe, path: Path, thr: float, label: str) -> None:
    import joblib
    joblib.dump({"pipeline": pipe, "threshold": thr, "model_name": label}, path)


if __name__ == "__main__":
    run_stage(int(sys.argv[1]))
