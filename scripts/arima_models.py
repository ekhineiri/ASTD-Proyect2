"""Tools for fitting ARIMA models to the M4 dataset.

This module is intentionally lightweight; it reads the same CSV files used by
:mod:`visualize_all_data`, fits an ``ARIMA`` model to each non‑empty series, and
writes a textual summary plus a diagnostic plot for each fit.

Usage examples
--------------

Run every frequency with the default order:

    python -m scripts.arima_models --order 1,1,1

Fit only the weekly data:

    python -m scripts.arima_models --frequencies weekly --order 2,1,2
"""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# delayed import inside functions to avoid pulling statsmodels until needed

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARIMA_OUTPUT_DIR = ROOT / "reports" / "arima_models"

FREQUENCIES = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]


def fit_arima_for_frequency(
    freq: str,
    order: tuple[int, int, int],
    use_validation: bool = False,
) -> None:
    """Fit ARIMA(p,d,q) to all series of a given frequency.

    The source data is normally ``data/train_{freq}.csv``.  If ``use_validation``
    is true and the validation partition has already been created via
    :mod:`create_validation_sets`, the function will instead read
    ``data/validation_last_block/history_{freq}.csv`` for training and
    ``.../validation_{freq}.csv`` for the held-out block.  When validation data
    are available, the script forecasts the hold‑out and reports MAE/RMSE.

    Parameters
    ----------
    freq : str
        Frequency name such as ``"daily"``.
    order : tuple[int, int, int]
        The (p, d, q) order to pass to ``statsmodels.tsa.arima.model.ARIMA``.
    use_validation : bool, optional
        Whether to look for history/validation files; if false the original
        train file is used and no forecasting evaluation is performed.
    """

    # decide which files to read
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

    ARIMA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # iterate over each series in the training frame
    for row_idx in range(len(train_df)):
        series_id = str(train_df.iloc[row_idx][series_id_col])
        train_series = train_vals.iloc[row_idx].dropna()
        if train_series.empty:
            continue

        valid_series = pd.Series(dtype=float)
        if not valid_vals.empty:
            valid_series = valid_vals.iloc[row_idx].dropna()

        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(train_series, order=order)
            res = model.fit()
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[ERROR] ARIMA failed for {series_id} (freq={freq}): {exc}")
            continue

        # write summary text
        summary_path = ARIMA_OUTPUT_DIR / f"{freq}_{series_id}_arima_summary.txt"
        with open(summary_path, "w") as f:
            f.write(res.summary().as_text())

        # diagnostic plot
        plt.figure(figsize=(10, 4))
        plt.plot(train_series.index, train_series.values, label="observed")
        fitted = res.fittedvalues
        plt.plot(fitted.index, fitted.values, label="fitted", linestyle="--")
        plt.title(f"ARIMA{order} fit for {series_id} ({freq})")
        plt.legend()
        plt.tight_layout()
        plot_path = ARIMA_OUTPUT_DIR / f"{freq}_{series_id}_arima_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        if not valid_series.empty:
            forecast = res.forecast(steps=len(valid_series))
            # symmetric mean absolute percentage error
            num = np.abs(forecast.values - valid_series.values)
            denom = (np.abs(forecast.values) + np.abs(valid_series.values)) / 2
            with np.errstate(divide='ignore', invalid='ignore'):
                smape = np.mean(np.where(denom == 0, 0.0, num / denom)) * 100
            eval_path = ARIMA_OUTPUT_DIR / f"{freq}_{series_id}_eval.txt"
            with open(eval_path, "w") as f:
                f.write(f"sMAPE: {smape:.4f}%\n")
            print(
                f"[OK] fitted ARIMA{order} for {series_id} ({freq}) "
                f"with validation sMAPE={smape:.4f}%"
            )
        else:
            print(f"[OK] fitted ARIMA{order} for {series_id} ({freq})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit ARIMA models to the project dataset."
    )
    parser.add_argument(
        "--frequencies",
        nargs="*",
        choices=FREQUENCIES,
        default=FREQUENCIES,
        help="List of frequencies to process (default: all)",
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
        help="read history/validation files generated by create_validation_sets",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    order = tuple(int(x) for x in args.order.split(","))
    if len(order) != 3:
        raise ValueError("order must have three comma-separated integers")

    for freq in args.frequencies:
        fit_arima_for_frequency(freq, order, use_validation=args.use_validation)
