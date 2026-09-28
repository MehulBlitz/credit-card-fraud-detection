# Publishing this project on GitHub

## 1. Get the code onto GitHub

### Option A — command line (from inside this project folder)

```bash
git init
git add .
git commit -m "Credit card fraud: imbalanced classification, SMOTE-in-CV, PR-AUC"
```

Create the empty repo on github.com first (**New repository** → name it
`credit-card-fraud-detection` → skip the README), then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/credit-card-fraud-detection.git
git push -u origin main
```

### Option B — GitHub Desktop

**File → Add local repository** → **Create a repository** → **Publish**.

## 2. What gets committed (and what doesn't)

The shipped `.gitignore` excludes `.venv/`, `__pycache__/`, `*.zip` and
`artifacts/*.joblib` (regenerable via `python train_fraud.py`).

Two deliberate exceptions:

- `data/creditcard.csv` (~67 MB) **is committed** so the scheduled GitHub
  Actions workflow can retrain without Kaggle credentials. (Kaggle's terms
  discourage redistribution — if you prefer strict compliance, remove the
  file from the repo and give the workflow a `KAGGLE_USERNAME` /
  `KAGGLE_KEY` secret instead.)
- `figures/` and `artifacts/metrics.json` are committed so recruiters can
  see results without training anything.

## 3. Optional: Hugging Face Space demo

This app's `artifacts/fraud_model.joblib` may exceed free-file comfort and
the app wants `data/creditcard.csv` for the sampling buttons. Either commit
both to the Space (they are small enough) or strip the sampler. The Space
setup is identical to the churn project:

```yaml
---
title: Credit Card Fraud Detector
emoji: 💳
sdk: gradio
app_file: app.py
---
```

## 4. Housekeeping

- **Description:** `Fraud detection on 284,807 transactions (0.17% fraud) —
  SMOTE inside CV folds, class-weight vs SMOTE comparison, PR-AUC-first
  evaluation, temporal holdout, Gradio demo.`
- **Topics:** `machine-learning`, `imbalanced-data`, `smote`,
  `fraud-detection`, `scikit-learn`, `gradio`
- Consider adding a short "Why not accuracy?" section to the README (it is
  already there) — it is the most-asked interview question on this project.
