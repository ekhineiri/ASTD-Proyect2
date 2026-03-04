"""
Tools for fitting Theta models to the M4 dataset.

This module mirrors scripts.arima_models but replaces ARIMA with the classic
Theta method (theta=2 by default), using:
  - theta=0 line: linear trend (OLS regression on time)
  - theta=theta line: SES (Simple Exponential Smoothing) on the theta-transformed series
and combines forecasts with equal weights (0.5 / 0.5) by default.

Usage examples
--------------

pip install -U pip
pip install statsmodels
----------------------------------

Run every frequency with default settings:

    python -m scripts.theta_models

Fit only weekly data:

    python -m scripts.theta_models --frequencies weekly

Use validation partition if available:

    python -m scripts.theta_models --use-validation

Change theta and weights:

    python -m scripts.theta_models --theta 2.0 --w-trend 0.5
"""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
THETA_OUTPUT_DIR = ROOT / "reports" / "theta_models"

FREQUENCIES = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_pred) + np.abs(y_true)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.mean(np.where(denom == 0, 0.0, num / denom)) * 100.0
    return float(val)


def fit_theta_model(
    y: pd.Series,
    steps: int = 0,
    theta: float = 2.0,
    w_trend: float = 0.5,
):
    """
    Fit classic Theta model:
      - Fit linear trend: y_t ≈ a + b t
      - Theta-transformed series: y_t(theta) = theta*y_t + (1-theta)*(a+bt)
      - Fit SES on y_t(theta)
      - Forecast:
          f0(h) = a + b (n+h)
          ftheta(h) = SES_forecast_on_transformed(h)
          f(h) = w_trend*f0(h) + (1-w_trend)*ftheta(h)

    Returns dict with fitted values, forecast, and a small summary.
    """
    if y.empty:
        raise ValueError("Empty series")

    y = y.astype(float)
    n = len(y)
    t = np.arange(1, n + 1, dtype=float)

    # 1) Linear trend via OLS (numpy polyfit degree 1) 
        # polyfit gives slope b and intercept a for y ≈ b*t + a
    b, a = np.polyfit(t, y.values, deg=1)
    trend_fitted = a + b * t  # in-sample

    # 2) Theta-transformed series 
        # y_theta(t) = theta*y(t) + (1-theta)*(a + b*t)
    y_theta = theta * y.values + (1.0 - theta) * trend_fitted

    # 3) SES on theta-series 
    ses_model = SimpleExpSmoothing(pd.Series(y_theta, index=y.index), initialization_method="estimated")
    ses_res = ses_model.fit(optimized=True)

    theta_fitted = ses_res.fittedvalues.values  # in-sample on transformed series

    # 4) Combine in-sample fitted values 
    w_trend = float(w_trend)
    w_trend = max(0.0, min(1.0, w_trend))
    combined_fitted = w_trend * trend_fitted + (1.0 - w_trend) * theta_fitted

    # 5) Forecast if requested 
    forecast = np.array([], dtype=float)
    if steps > 0:
        h = np.arange(1, steps + 1, dtype=float)
        trend_forecast = a + b * (n + h)
        theta_forecast = ses_res.forecast(steps).values
        forecast = w_trend * trend_forecast + (1.0 - w_trend) * theta_forecast

    summary = {
        "theta": float(theta),
        "w_trend": float(w_trend),
        "a_intercept": float(a),
        "b_slope": float(b),
        "ses_alpha": float(getattr(ses_res, "params", {}).get("smoothing_level", np.nan)),
        "ses_sse": float(getattr(ses_res, "sse", np.nan)),
        "n_obs": int(n),
    }

    return {
        "summary": summary,
        "fitted": pd.Series(combined_fitted, index=y.index),
        "forecast": forecast,
    }


def fit_theta_for_frequency(
    freq: str,
    theta: float = 2.0,
    w_trend: float = 0.5,
    use_validation: bool = False,
) -> None:
    """Fit Theta model to all series of a given frequency (M4-style wide CSV)."""

    history_path = DATA_DIR / "validation_last_block" / f"history_{freq}.csv"
    validation_path = DATA_DIR / "validation_last_block" / f"validation_{freq}.csv"

    if use_validation and history_path.exists() and validation_path.exists():
        train_df = pd.read_csv(history_path)
        valid_df = pd.read_csv(validation_path)
        series_id_col = train_df.columns[0]
        train_vals = train_df.drop(columns=[series_id_col]).apply(pd.to_numeric, errors="coerce")
        valid_vals = valid_df.drop(columns=[series_id_col]).apply(pd.to_numeric, errors="coerce")
    else:
        if use_validation:
            print("[WARN] Validation files not found; falling back to full train set")
        train_df = pd.read_csv(DATA_DIR / f"train_{freq}.csv")
        if train_df.empty:
            print(f"[WARN] Empty file: {DATA_DIR / f'train_{freq}.csv'}")
            return
        series_id_col = train_df.columns[0]
        train_vals = train_df[train_df.columns[1:]].apply(pd.to_numeric, errors="coerce")
        valid_vals = pd.DataFrame()

    THETA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for row_idx in range(len(train_df)):
        series_id = str(train_df.iloc[row_idx][series_id_col])
        train_series = train_vals.iloc[row_idx].dropna()
        if train_series.empty:
            continue

        valid_series = pd.Series(dtype=float)
        if not valid_vals.empty:
            valid_series = valid_vals.iloc[row_idx].dropna()

        try:
            res = fit_theta_model(
                y=train_series,
                steps=(len(valid_series) if not valid_series.empty else 0),
                theta=theta,
                w_trend=w_trend,
            )
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[ERROR] Theta failed for {series_id} (freq={freq}): {exc}")
            continue

        # write summary text with key info about the fitted model and parameters
        summary_path = THETA_OUTPUT_DIR / f"{freq}_{series_id}_theta_summary.txt"
        with open(summary_path, "w") as f:
            s = res["summary"]
            f.write("Theta model summary\n")
            f.write("-------------------\n")
            f.write(f"freq: {freq}\n")
            f.write(f"series_id: {series_id}\n")
            f.write(f"theta: {s['theta']}\n")
            f.write(f"w_trend: {s['w_trend']}\n")
            f.write(f"trend: y ~= a + b*t\n")
            f.write(f"  a (intercept): {s['a_intercept']:.6f}\n")
            f.write(f"  b (slope):     {s['b_slope']:.6f}\n")
            f.write(f"SES (on theta-series):\n")
            f.write(f"  alpha: {s['ses_alpha']}\n")
            f.write(f"  SSE:   {s['ses_sse']}\n")
            f.write(f"n_obs: {s['n_obs']}\n")

        # ---- diagnostic plot ----
        plt.figure(figsize=(10, 4))
        plt.plot(train_series.index, train_series.values, label="observed")
        fitted = res["fitted"]
        plt.plot(fitted.index, fitted.values, label="fitted", linestyle="--")
        plt.title(f"Theta(theta={theta}, w_trend={w_trend}) fit for {series_id} ({freq})")
        plt.legend()
        plt.tight_layout()
        plot_path = THETA_OUTPUT_DIR / f"{freq}_{series_id}_theta_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # evaluation on validation 
        if not valid_series.empty:
            forecast = res["forecast"]
            score = smape(valid_series.values, forecast)
            eval_path = THETA_OUTPUT_DIR / f"{freq}_{series_id}_eval.txt"
            with open(eval_path, "w") as f:
                f.write(f"sMAPE: {score:.4f}%\n")
            print(
                f"[OK] fitted Theta(theta={theta}) for {series_id} ({freq}) "
                f"with validation sMAPE={score:.4f}%"
            )
        else:
            print(f"[OK] fitted Theta(theta={theta}) for {series_id} ({freq})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Theta models to the project dataset."
    )
    parser.add_argument(
        "--frequencies",
        nargs="*",
        choices=FREQUENCIES,
        default=FREQUENCIES,
        help="List of frequencies to process (default: all)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=2.0,
        help="Theta parameter (classic Theta uses 2.0)",
    )
    parser.add_argument(
        "--w-trend",
        type=float,
        default=0.5,
        help="Weight for trend (theta=0) line in the final combination (0..1).",
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="read history/validation files generated by create_validation_sets",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for freq in args.frequencies:
        fit_theta_for_frequency(
            freq=freq,
            theta=args.theta,
            w_trend=args.w_trend,
            use_validation=args.use_validation,
        )