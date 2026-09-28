"""
Load the Credit Card Fraud Detection dataset (284,807 transactions,
0.172% fraud) from Kaggle.

Two ways to get the data
------------------------
A) Automatic via the Kaggle CLI (recommended):
     1. Create a free account at https://www.kaggle.com
     2. kaggle.com -> Settings -> API -> "Create New Token"
        (downloads kaggle.json)
     3. Put kaggle.json in ~/.kaggle/ (Linux/macOS) or
        C:\\Users\\<you>\\.kaggle\\ (Windows)
     4. pip install kaggle
     5. python download_data.py

B) Manual download:
     1. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
     2. Download and unzip; place creditcard.csv into data/
     3. python download_data.py  (validates and reports class balance)

If the dataset is missing and you have no Kaggle credentials, this script
can generate a *synthetic stand-in* (clearly marked, same schema) so the
pipeline is runnable end-to-end anywhere. Never report metrics from the
synthetic data as real results.

Usage:
    python download_data.py [--synthetic-fallback]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_PATH = DATA_DIR / "creditcard.csv"

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"

# Public mirrors of the same creditcard.csv (tried before giving up).
RAW_URLS = [
    "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
    "https://raw.githubusercontent.com/pik1989/MLProjects/main/Dataset/creditcard.csv",
]

EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)


def has_local_file() -> bool:
    if not DATA_DIR.exists():
        return False
    for cand in sorted(DATA_DIR.glob("*.csv")):
        try:
            head = pd.read_csv(cand, nrows=2)
        except Exception:
            continue
        if all(c in head.columns for c in EXPECTED_COLUMNS):
            if cand.resolve() != OUT_PATH.resolve():
                df = pd.read_csv(cand)
                df.to_csv(OUT_PATH, index=False)
                print(f"[ok] Found local copy {cand.name}; standardized to {OUT_PATH.name}")
            else:
                print(f"[ok] Dataset already present: {cand}")
            return True
    return False


def download_with_kaggle() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[warn] kaggle package not installed (pip install kaggle)")
        return False

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # missing ~/.kaggle/kaggle.json etc.
        print(f"[warn] Kaggle authentication failed: {exc}")
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "creditcardfraud.zip"
    try:
        print(f"Downloading '{KAGGLE_DATASET}' from Kaggle ... (~66 MB)")
        api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), quiet=False)
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(DATA_DIR)
            zip_path.unlink()
    except Exception as exc:
        print(f"[warn] Kaggle download failed: {exc}")
        return False
    return OUT_PATH.exists()


def download_from(url: str) -> pd.DataFrame:
    import io
    import urllib.request

    print(f"  Trying mirror {url} ...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw))


def try_mirrors() -> bool:
    for url in RAW_URLS:
        try:
            df = download_from(url)
        except Exception as exc:  # noqa: BLE001 - try every mirror
            print(f"  [warn] mirror failed: {exc}")
            continue
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing or len(df) < 250_000:
            print(f"  [warn] mirror content invalid (rows={len(df)}, missing={missing})")
            continue
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_PATH, index=False)
        print(f"[ok] Downloaded real dataset from mirror -> {OUT_PATH}")
        return True
    return False


def make_synthetic_fallback(n_normal: int = 200_000, n_fraud: int = 344) -> None:
    """Schema-compatible stand-in so the code is runnable end-to-end.

    PCA 'V' columns are simulated as independent gaussians; fraud rows get
    shifted means on a few components and much smaller amounts of noise, so
    the signal is learnable but the class ratio matches the real data.
    """
    rng = np.random.default_rng(42)
    total = n_normal + n_fraud

    v_normal = rng.normal(0, 1.0, size=(n_normal, 28))
    v_fraud = rng.normal(0, 0.55, size=(n_fraud, 28))
    for j in (10, 11, 12, 14):
        v_fraud[:, j] += 1.6  # deliberately mild signal so scores stay realistic

    time = np.sort(rng.integers(0, 172_792, size=total)).astype(float)
    amount = np.concatenate([
        np.abs(rng.lognormal(mean=3.2, sigma=1.0, size=n_normal)),
        np.abs(rng.lognormal(mean=4.6, sigma=0.7, size=n_fraud)),
    ])
    cls = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

    df = pd.DataFrame(np.vstack([v_normal, v_fraud]), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = cls
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(
        f"[warn] SYNTHETIC stand-in written to {OUT_PATH} "
        f"({len(df):,} rows, fraud {cls.mean():.3%}). "
        "Replace with the real Kaggle file for genuine results."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-fallback", action="store_true",
                    help="Generate a synthetic stand-in if Kaggle data is unavailable")
    args = ap.parse_args()

    if has_local_file():
        _report()
        return 0

    print("Dataset not found locally; trying Kaggle ...")
    if download_with_kaggle():
        _report()
        return 0

    print("Kaggle unavailable; trying public mirrors ...")
    if try_mirrors():
        _report()
        return 0

    if args.synthetic_fallback:
        make_synthetic_fallback()
        _report()
        return 0

    print(
        "\n[error] Could not obtain the dataset.\n"
        f"Manual option: download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        f"and place creditcard.csv in {DATA_DIR}\n"
        "Or rerun with --synthetic-fallback for a runnable stand-in.\n"
    )
    return 1


def _report() -> None:
    df = pd.read_csv(OUT_PATH)
    counts = df["Class"].value_counts()
    fraud_pct = counts.get(1, 0) / len(df)
    print(f"[ok] {len(df):,} rows | fraud: {counts.get(1, 0):,} ({fraud_pct:.3%}) | "
          f"legit: {counts.get(0, 0):,}")
    print("     Columns:", ", ".join(df.columns[:5]), "...", ", ".join(df.columns[-3:]))
    if not np.isclose(len(df), 284_807, rtol=0.01):
        print("     [note] Row count differs from the real dataset (284,807).")


if __name__ == "__main__":
    sys.exit(main())
