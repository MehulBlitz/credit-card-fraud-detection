"""
Interactive demo for the credit card fraud detector.

Run (after `python train_fraud.py` has produced artifacts/fraud_model.joblib):
    python app.py
Then open the local URL Gradio prints (default http://127.0.0.1:7860).

The app scores one transaction at a time. Since the dataset ships PCA
features (V1..V28), there is also a one-click "Random legit / Random fraud"
sampler that draws a real transaction from data/creditcard.csv so you can
probe the model without inventing 30 numbers.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "artifacts" / "fraud_model.joblib"
DATA_PATH = HERE / "data" / "creditcard.csv"

if not MODEL_PATH.exists():
    raise SystemExit(
        "Model not found. Run `python download_data.py` and "
        "`python train_fraud.py` first, then relaunch this app."
    )

BUNDLE = joblib.load(MODEL_PATH)
PIPE = BUNDLE["pipeline"]
THRESHOLD_DEFAULT = float(BUNDLE["threshold"])

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Rough display ranges taken from the real dataset (5th-95th percentile).
BOUNDS: dict[str, tuple[float, float]] = {
    "Time": (0.0, 172_792.0),
    "Amount": (0.0, 250.0),
    **{f"V{i}": (-5.0, 5.0) for i in range(1, 29)},
}
DEFAULTS: dict[str, float] = {
    "Time": 40_000.0,
    "Amount": 60.0,
    **{f"V{i}": 0.0 for i in range(1, 29)},
}

_DATA_CACHE: pd.DataFrame | None = None


def _load_data() -> pd.DataFrame:
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = pd.read_csv(DATA_PATH)
    return _DATA_CACHE


def score_transaction(threshold: float, *values: float) -> tuple:
    x = pd.DataFrame([dict(zip(FEATURES, values))])
    proba = float(PIPE.predict_proba(x)[0, 1])
    verdict = "FRAUD" if proba >= threshold else "LEGIT"
    md = (
        f"### Prediction: **{verdict}**  \n"
        f"fraud probability **{proba:.4f}** vs threshold {threshold:.2f}"
    )
    return proba, md


def sample_row(label: str) -> dict:
    """Return default-slider values for a random row of the chosen class."""
    df = _load_data()
    cls = 1 if label == "Random fraud transaction" else 0
    row = df[df["Class"] == cls].sample(n=1, random_state=None).iloc[0]
    return {f: float(row[f]) for f in FEATURES}


def sample_and_score(label: str, threshold: float):
    vals = sample_row(label)
    proba, md = score_transaction(threshold, *[vals[f] for f in FEATURES])
    return *[vals[f] for f in FEATURES], proba, md


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Credit Card Fraud Detector") as demo:
        gr.Markdown(
            "# Credit Card Fraud Detection\n"
            "Scikit-learn + imbalanced-learn pipeline (scaling -> "
            "SMOTE/class-weights -> model) deployed with Gradio. "
            "The model was trained on the first 80% of transactions by time "
            "and evaluated on the last 20%."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Transaction features (V1-V28 are PCA components)")
                with gr.Row():
                    time_s = gr.Slider(*BOUNDS["Time"], value=DEFAULTS["Time"],
                                       step=1.0, label="Time (s)")
                    amount_s = gr.Slider(0, 500, value=DEFAULTS["Amount"],
                                         step=0.01, label="Amount ($)")
                with gr.Accordion("PCA features V1-V28", open=True):
                    v_sliders: dict[str, gr.Slider] = {}
                    for i in range(1, 29):
                        v_sliders[f"V{i}"] = gr.Slider(
                            -6.0, 6.0, value=0.0, step=0.01, label=f"V{i}"
                        )
                with gr.Row():
                    btn_legit = gr.Button("Sample random LEGIT transaction")
                    btn_fraud = gr.Button("Sample random FRAUD transaction")
                btn_score = gr.Button("Score transaction", variant="primary")
            with gr.Column():
                threshold = gr.Slider(0.01, 0.995, value=THRESHOLD_DEFAULT,
                                      step=0.005,
                                      label="Decision threshold "
                                            "(lower = catch more fraud)")
                proba_out = gr.Number(label="Fraud probability", precision=4)
                verdict = gr.Markdown("")
                gr.Markdown(
                    "**Tips**\n\n"
                    "- Click *Sample random FRAUD transaction* a few times: "
                    "real fraud rows sit in a different part of the PCA space, "
                    "so scores are usually high.\n"
                    "- For rare-event problems judge the model on the "
                    "PR curve, not accuracy."
                )

        all_inputs = [time_s, amount_s, *v_sliders.values()]

        btn_score.click(score_transaction,
                        inputs=[threshold, *all_inputs],
                        outputs=[proba_out, verdict])
        threshold.change(
            lambda t, *vals: score_transaction(t, *vals),
            inputs=[threshold, *all_inputs],
            outputs=[proba_out, verdict],
        )
        for btn, label in ((btn_legit, "Random legit transaction"),
                           (btn_fraud, "Random fraud transaction")):
            btn.click(
                sample_and_score,
                inputs=[gr.State(label), threshold],
                outputs=[*all_inputs, proba_out, verdict],
            )
    return demo


if __name__ == "__main__":
    build_ui().launch()
