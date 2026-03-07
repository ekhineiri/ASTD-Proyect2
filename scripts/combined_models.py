"""
Combine ARIMA and Theta forecasts using a simple ensemble.

Each frequency will produce a CSV file:
hourly_1.csv, daily_1.csv, weekly_1.csv, etc.

Also generates diagnostic plots and summaries for each series.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Import the already written theta implementations
from scripts.theta_models import fit_theta_model
from scripts.theta_models import smape

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = ROOT / "submissions"
COMBINED_OUTPUT_DIR = ROOT / "reports" / "combined_models"  # For diagnostics

FREQUENCIES = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
# Add required forecast horizons
HORIZONS = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "weekly": 13,
    "daily": 14,
    "hourly": 48
}

def combine_forecasts(arima_fc, theta_fc):
    """Average ensemble."""
    return (np.array(arima_fc) + np.array(theta_fc)) / 2.0


def fit_combined_for_frequency(freq: str, order=(1, 1, 1), use_validation=False, model_number=1):
    """Fit ARIMA + Theta, combine forecasts, and save CSVs with diagnostics."""

    history_path = DATA_DIR / "validation_last_block" / f"history_{freq}.csv"
    validation_path = DATA_DIR / "validation_last_block" / f"validation_{freq}.csv"

    # Get the required forecast horizon for this frequency
    required_horizon = HORIZONS.get(freq, 1)
    
    if use_validation and history_path.exists() and validation_path.exists():
        train_df = pd.read_csv(history_path)
        valid_df = pd.read_csv(validation_path)

        series_id_col = train_df.columns[0]
        train_vals = train_df.drop(columns=[series_id_col]).apply(pd.to_numeric, errors="coerce")
        valid_vals = valid_df.drop(columns=[series_id_col]).apply(pd.to_numeric, errors="coerce")
        
        # For validation, we forecast the validation period length
        forecast_steps = len(valid_vals.columns)
        print(f"[INFO] {freq}: Using validation mode with {forecast_steps} steps")
    else:
        train_df = pd.read_csv(DATA_DIR / f"train_{freq}.csv")
        series_id_col = train_df.columns[0]
        train_vals = train_df.drop(columns=[series_id_col]).apply(pd.to_numeric, errors="coerce")
        valid_vals = pd.DataFrame()
        
        # For submission, we forecast the required horizon
        forecast_steps = required_horizon
        print(f"[INFO] {freq}: Using submission mode with {forecast_steps} steps (required horizon)")

    # Create directories
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Store all forecasts for this frequency
    all_forecasts: list[list] = []

    for row_idx in range(len(train_df)):
        series_id = str(train_df.iloc[row_idx][series_id_col])
        train_series = train_vals.iloc[row_idx].dropna()

        if train_series.empty:
            # Add row with NaNs for empty series
            all_forecasts.append([series_id] + [np.nan] * forecast_steps)
            continue

        valid_series = pd.Series(dtype=float)
        if not valid_vals.empty:
            valid_series = valid_vals.iloc[row_idx].dropna()

        try:
            # ---------- ARIMA ----------
            arima_model = ARIMA(train_series, order=order)
            arima_res = arima_model.fit()
            arima_forecast = arima_res.forecast(steps=forecast_steps).values
            arima_fitted = arima_res.fittedvalues

            # ---------- THETA ----------
            theta_res = fit_theta_model(
                y=train_series,
                steps=forecast_steps,
                theta=2.0,
                w_trend=0.5
            )
            theta_forecast = theta_res["forecast"]
            theta_fitted = theta_res["fitted"]

            # ---------- COMBINE ----------
            combined_forecast = combine_forecasts(arima_forecast, theta_forecast)
            
            # For fitted values, we need to align them (they might have different lengths)
            # Use the shorter length for combination
            min_len = min(len(arima_fitted), len(theta_fitted))
            combined_fitted = (arima_fitted.values[:min_len] + theta_fitted.values[:min_len]) / 2.0

        except Exception as exc:
            print(f"[ERROR] Combined model failed for {series_id}: {exc}")
            # Fallback to naive forecast
            combined_forecast = np.full(forecast_steps, train_series.values[-1])
            # Skip diagnostic files for failed models
            row = [series_id] + combined_forecast.tolist()
            all_forecasts.append(row)
            continue

        # Add to CSV rows
        row = [series_id] + combined_forecast.tolist()
        all_forecasts.append(row)

        # Generate diagnostic plots and summaries for first 10 series or if validation available
        if row_idx < 10 or (use_validation and not valid_series.empty):
            
            # ---------- Save summary text ----------
            summary_path = COMBINED_OUTPUT_DIR / f"{freq}_{series_id}_combined_summary.txt"
            with open(summary_path, "w") as f:
                f.write("Combined ARIMA + Theta Model Summary\n")
                f.write("====================================\n")
                f.write(f"Frequency: {freq}\n")
                f.write(f"Series ID: {series_id}\n")
                f.write(f"Model Number: {model_number}\n")
                f.write(f"ARIMA Order: {order}\n")
                f.write(f"Theta parameters: theta=2.0, w_trend=0.5\n\n")
                
                f.write("ARIMA Summary:\n")
                f.write(f"  AIC: {arima_res.aic:.4f}\n")
                f.write(f"  BIC: {arima_res.bic:.4f}\n")
                f.write(f"  Log Likelihood: {arima_res.llf:.4f}\n\n")
                
                f.write("Theta Summary:\n")
                s = theta_res["summary"]
                f.write(f"  Theta: {s['theta']}\n")
                f.write(f"  w_trend: {s['w_trend']}\n")
                f.write(f"  Trend: y = {s['a_intercept']:.4f} + {s['b_slope']:.4f}*t\n")
                f.write(f"  SES alpha: {s['ses_alpha']:.4f}\n")
                f.write(f"  SES SSE: {s['ses_sse']:.4f}\n\n")
                
                f.write(f"Training observations: {len(train_series)}\n")
                f.write(f"Forecast horizon: {forecast_steps}\n")
                
                if not valid_series.empty:
                    score = smape(valid_series.values, combined_forecast)
                    f.write(f"\nValidation sMAPE: {score:.4f}%\n")
                
                f.write(f"\nFirst 5 forecast values:\n")
                for i, val in enumerate(combined_forecast[:5]):
                    f.write(f"  f{i+1}: {val:.4f}\n")

            # ---------- Save diagnostic plot ----------
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Time series with fitted values
            plt.subplot(2, 1, 1)
            plt.plot(train_series.index, train_series.values, 'b-', label="Observed", linewidth=1.5)
            plt.plot(train_series.index[:min_len], combined_fitted, 'r--', label="Combined Fitted", linewidth=1.5, alpha=0.7)
            plt.plot(train_series.index[:len(arima_fitted)], arima_fitted.values, 'g:', label="ARIMA Fitted", linewidth=1, alpha=0.5)
            plt.plot(train_series.index[:len(theta_fitted)], theta_fitted.values, 'y:', label="Theta Fitted", linewidth=1, alpha=0.5)
            
            # Add forecast if validation available
            if not valid_series.empty:
                forecast_idx = range(len(train_series), len(train_series) + len(valid_series))
                plt.plot(forecast_idx, combined_forecast, 'm--', label="Combined Forecast", linewidth=2)
                plt.plot(forecast_idx, valid_series.values, 'k-', label="Actual Validation", linewidth=1.5)
            
            plt.title(f"Combined Model: {freq.upper()} - Series {series_id}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Forecast vs components
            plt.subplot(2, 1, 2)
            x_forecast = np.arange(1, forecast_steps + 1)
            plt.plot(x_forecast, arima_forecast, 'g^--', label="ARIMA Forecast", markersize=4)
            plt.plot(x_forecast, theta_forecast, 'ys--', label="Theta Forecast", markersize=4)
            plt.plot(x_forecast, combined_forecast, 'ro-', label="Combined Forecast", linewidth=2, markersize=6)
            
            if not valid_series.empty:
                plt.plot(x_forecast, valid_series.values, 'k*-', label="Actual", linewidth=1.5, markersize=8)
            
            plt.title(f"Forecast Comparison - {freq.upper()} - Series {series_id}")
            plt.xlabel("Forecast Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = COMBINED_OUTPUT_DIR / f"{freq}_{series_id}_combined_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Print validation score if available
        if use_validation and not valid_series.empty and len(valid_series) == forecast_steps:
            score = smape(valid_series.values, combined_forecast)
            print(f"[OK] combined ARIMA{order} + Theta for {series_id} ({freq}) sMAPE={score:.4f}%")
        elif (row_idx + 1) % 50 == 0:
            print(f"[OK] Processed {row_idx + 1} series for {freq}")

    # Save submission CSV
    output_csv = SUBMISSION_DIR / f"{freq}_{model_number}.csv"
    
    # Create DataFrame with ID column and forecast columns
    columns = [series_id_col] + [f"f{i+1}" for i in range(forecast_steps)]
    df = pd.DataFrame(all_forecasts, columns=columns)
    df.to_csv(output_csv, index=False)
    
    print(f"[INFO] Saved submission file for {freq} to {output_csv} ({len(all_forecasts)} series)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frequencies",
        nargs="*",
        choices=FREQUENCIES,
        default=FREQUENCIES,
        help="Frequencies to process"
    )
    parser.add_argument(
        "--order",
        type=str,
        default="1,1,1",
        help="ARIMA order as p,d,q"
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="Use validation mode (forecast validation period instead of required horizon)"
    )
    parser.add_argument(
        "--model-number",
        type=int,
        default=1,
        choices=[1, 2],
        help="Model number for submission files (1 or 2)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    order = tuple(int(x) for x in args.order.split(","))

    for freq in args.frequencies:
        fit_combined_for_frequency(
            freq=freq,
            order=order,
            use_validation=args.use_validation,
            model_number=args.model_number
        )