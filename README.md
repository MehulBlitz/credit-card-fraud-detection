# Credit Card Fraud Detection — Imbalanced Classification

[![Retrain & refresh metrics](https://github.com/MehulBlitz/credit-card-fraud-detection/actions/workflows/update-metrics.yml/badge.svg)](https://github.com/MehulBlitz/credit-card-fraud-detection/actions/workflows/update-metrics.yml)
[![Live metrics dashboard](https://img.shields.io/badge/dashboard-MehulBlitz.github.io-2ea44f)](https://mehulblitz.github.io)

Fraud detection on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, **492 frauds = 0.173%**), with the class-imbalance
toolkit applied the way it should be: resampling *inside* cross-validation
folds, and PR curves instead of accuracy.

**Stack:** Python · Scikit-learn · imbalanced-learn · Pandas · Matplotlib · Gradio

## Results (reproduce with `python train_fraud.py`, seed = 42)

**Temporal split** — first 80% of transactions (by `Time`) for training,
last 20% (56,961 rows, 75 frauds) as the untouched test set, mimicking
production where a model is trained on the past and scored on the future.

| Experiment | 5-fold CV PR-AUC | Test PR-AUC | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC |
|---|---|---|---|---|---|---|
| LogReg + class weights | 0.7537 | 0.7533 | 0.667 | 0.773 | 0.716 | **0.986** |
| LogReg + SMOTE | 0.7620 | 0.7274 | 0.757 | 0.707 | 0.731 | 0.983 |
| **RandomForest + class weights** | 0.8488 | **0.8232** | **0.982** | 0.733 | **0.840** | 0.960 |
| RandomForest + SMOTE | 0.8561 | 0.8155 | 0.949 | **0.747** | 0.836 | 0.991 |

Winner by test F1: **Random Forest + class weights**, decision threshold
**0.386** (tuned on out-of-fold predictions, not on the test set).

**Headline: PR-AUC 0.823 and F1 0.840 on the held-out test set.**
Confusion matrix at the tuned threshold: **55 of 75 frauds caught
(73.3% recall) with exactly 1 false alarm in 56,961 legit transactions**
(98.2% precision).

### What the numbers say

- **Accuracy is useless here** — a model that predicts "legit" for
  everything scores 99.87%. The PR curve (and average precision) is the
  honest metric for rare events; ROC-AUC 0.98+ looks glamorous for *every*
  model and hides the precision differences that matter.
- **Class weights beat SMOTE for both models** on test PR-AUC — SMOTE's
  synthetic minority points helped CV scores slightly but generalized a
  touch worse. Good reminder that resampling is a hyperparameter-level
  choice to be validated, not a default.
- **Threshold tuning matters:** recall 70.7% → 73.3% with precision
  essentially unchanged (0.9815 → 0.9821) just by moving the cut from 0.5
  to 0.386 using out-of-fold predictions.
- Note on the LogReg thresholds (~0.98): with heavy class weighting the
  model's probabilities are extremely polarized, so its best-F1 operating
  point lives near the top of the score range — the operating point
  (P = 0.69 / R = 0.77) is the meaningful number, not the cut position.

## What this project demonstrates

- **Leakage-free resampling** — `StandardScaler` + `SMOTE` + model live in
  one `imblearn` Pipeline, so SMOTE sees *only the training part of each
  CV fold* and synthetic points never leak into validation.
- **Temporal splitting** — a random split would let near-duplicate
  transactions straddle train/test; splitting by `Time` is the realistic
  protocol.
- **Class-weights vs SMOTE compared** in a controlled grid
  (`GridSearchCV`, 5-fold stratified CV, scoring = average precision).
- **Threshold tuning on out-of-fold predictions** — the test set is
  touched exactly once.
- **Deployment** — interactive Gradio app with 30 feature inputs,
  a threshold slider, and one-click sampling of real legit/fraud
  transactions from the dataset.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python download_data.py           # Kaggle CLI -> public mirrors -> --synthetic-fallback
python train_fraud.py             # trains + tunes + saves artifacts/ and figures/
python app.py                     # interactive Gradio demo on http://127.0.0.1:7860
```

`download_data.py` works **without a Kaggle account**: it first tries the
Kaggle CLI (if you have `~/.kaggle/kaggle.json`), then public mirrors of
the dataset. Only with `--synthetic-fallback` does it generate a clearly
marked stand-in (never report metrics from that as real results).

Slow machine? Training is checkpointed per experiment:

```bash
python train_fraud.py --stage 0   # ... 1 2 3: one experiment each, resumable
python collect_fraud.py           # merge checkpoints -> metrics.json + model
```

## Project structure

```
2_credit_fraud/
├── download_data.py        # Kaggle / mirror / fallback loader + validation
├── train_fraud.py          # imblearn pipelines, GridSearchCV, threshold tuning
├── stage_fraud.py          # checkpoint helper for one experiment
├── collect_fraud.py        # merge checkpoints, figures, final artifacts
├── app.py                  # Gradio demo (feature sliders, sampler, threshold)
├── requirements.txt
├── data/                   # creditcard.csv (committed so Actions can retrain)
├── artifacts/              # fraud_model.joblib, metrics.json
└── figures/                # class balance, PR curves, confusion, score dist
```

## Résumé bullet (numbers verified by this repo)

> Developed a fraud detection model on 284,807 transactions (0.17% fraud)
> with class weights and SMOTE applied inside cross-validation folds to
> avoid leakage; selected models with precision-recall curves and PR-AUC;
> compared Logistic Regression and Random Forest, achieving **PR-AUC 0.82
> and F1 0.84** on a held-out test set (73% of frauds caught at 98%
> precision).

See [SETUP_GITHUB.md](SETUP_GITHUB.md) to publish this folder as a repo.
