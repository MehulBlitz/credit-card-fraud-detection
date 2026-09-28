"""
Stage 2: merge checkpointed experiments (artifacts/partial_*.json), pick the
winner by tuned-threshold F1, draw the figures, and write the final
metrics.json + fraud_model.joblib.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

import train_fraud as tf

ART = tf.ART


def main() -> None:
    df = tf.load_data()
    train_df, test_df = tf.temporal_split(df, test_frac=0.20)
    feature_cols = [c for c in df.columns if c != "Class"]
    X_test, y_test = test_df[feature_cols], test_df["Class"].values

    partials = sorted(ART.glob("partial_*.json"))
    if not partials:
        raise SystemExit("No partial_*.json found - run the staged fits first.")

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    test_probas: dict[str, np.ndarray] = {}
    for p in partials:
        idx = p.stem.split("_")[1]
        payload = json.loads(p.read_text())
        label = payload["label"]
        results[label] = {k: v for k, v in payload.items() if k != "label"}
        bundle = joblib.load(ART / f"partial_{idx}.joblib")
        fitted[label] = bundle["pipeline"]
        test_probas[label] = fitted[label].predict_proba(X_test)[:, 1]
        print(f"[collect] {label}: CV PR-AUC {results[label]['cv_pr_auc']}, "
              f"test F1 {results[label]['test_tuned_threshold']['f1_fraud']}")

    winner = max(results, key=lambda n: results[n]["test_tuned_threshold"]["f1_fraud"])
    best_thr = results[winner]["test_tuned_threshold"]["threshold"]
    proba_best = test_probas[winner]
    pred_best = (proba_best >= best_thr).astype(int)

    print(f"\n### Winner: {winner} (threshold {best_thr:.3f}) ###")
    print(f"  test @ tuned: {results[winner]['test_tuned_threshold']}")
    print(f"  test @ 0.5  : {results[winner]['test_default_0.5']}")

    tf.plot_class_balance(df)
    tf.plot_pr_curves(test_probas, y_test)
    tf.plot_confusion(y_test, pred_best,
                      f"{winner} - confusion matrix (thr={best_thr:.2f})",
                      "confusion_matrix_tuned.png")
    tf.plot_score_distribution(y_test, proba_best)

    metrics = {
        "winner_model": winner,
        "split": "temporal 80/20 by Time",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "fraud_rate_train": round(float(train_df["Class"].mean()), 6),
        "fraud_rate_test": round(float(test_df["Class"].mean()), 6),
        "experiments": results,
        "headline": {
            "pr_auc": results[winner]["test_tuned_threshold"]["pr_auc"],
            "f1_fraud": results[winner]["test_tuned_threshold"]["f1_fraud"],
            "precision_fraud":
                results[winner]["test_tuned_threshold"]["precision_fraud"],
            "recall_fraud": results[winner]["test_tuned_threshold"]["recall_fraud"],
            "roc_auc": results[winner]["test_tuned_threshold"]["roc_auc"],
            "confusion": results[winner]["confusion_at_tuned"],
        },
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"pipeline": fitted[winner], "threshold": best_thr,
                 "model_name": winner}, ART / "fraud_model.joblib")

    # Remove the per-experiment checkpoints (kept JSONs are tiny, models aren't)
    for jb in ART.glob("partial_*.joblib"):
        jb.unlink()

    print(f"\nSaved model   -> {ART / 'fraud_model.joblib'}")
    print(f"Saved metrics -> {ART / 'metrics.json'}")


if __name__ == "__main__":
    main()
