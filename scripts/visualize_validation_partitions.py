from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARTITION_DIR = ROOT / "data" / "validation_last_block"
OUTPUT_DIR = ROOT / "reports" / "validation_partitions_plots"

FREQUENCIES = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]


def periods_per_year(freq: str) -> float:
    mapping = {
        "hourly": 24.0 * 365.25,
        "daily": 365.25,
        "weekly": 52.1775,
        "monthly": 12.0,
        "quarterly": 4.0,
        "yearly": 1.0,
    }
    return mapping[freq]


def row_to_values(row: pd.Series) -> np.ndarray:
    return pd.to_numeric(row, errors="coerce").dropna().to_numpy(dtype=float)


def plot_frequency_partition(freq: str) -> None:
    history_path = PARTITION_DIR / f"history_{freq}.csv"
    validation_path = PARTITION_DIR / f"validation_{freq}.csv"

    if not history_path.exists() or not validation_path.exists():
        print(f"[WARN] Missing partition files for {freq}")
        return

    history_df = pd.read_csv(history_path)
    validation_df = pd.read_csv(validation_path)

    if history_df.empty or validation_df.empty:
        print(f"[WARN] Empty partition file(s) for {freq}")
        return

    history_id_col = history_df.columns[0]
    validation_id_col = validation_df.columns[0]

    history_value_cols = history_df.columns[1:]
    validation_value_cols = validation_df.columns[1:]

    ppy = periods_per_year(freq)
    plt.figure(figsize=(14, 7))

    total_series = min(len(history_df), len(validation_df))
    for idx in range(total_series):
        series_id = str(history_df.iloc[idx][history_id_col])
        val_id = str(validation_df.iloc[idx][validation_id_col])
        if val_id != series_id:
            print(f"[WARN] ID mismatch at row {idx}: {series_id} vs {val_id}")
            continue

        history_values = row_to_values(history_df.iloc[idx][history_value_cols])
        validation_values = row_to_values(validation_df.iloc[idx][validation_value_cols])

        if len(history_values) == 0 and len(validation_values) == 0:
            continue

        x_history = np.arange(len(history_values), dtype=float) / ppy
        x_validation = (
            np.arange(len(history_values), len(history_values) + len(validation_values), dtype=float)
            / ppy
        )

        if len(history_values) > 0:
            plt.plot(
                x_history,
                history_values,
                linewidth=0.9,
                alpha=0.55,
                color="steelblue",
            )

        if len(validation_values) > 0:
            plt.plot(
                x_validation,
                validation_values,
                linewidth=1.1,
                alpha=0.85,
                color="tomato",
            )

            if len(history_values) > 0:
                plt.plot(
                    [x_history[-1], x_validation[0]],
                    [history_values[-1], validation_values[0]],
                    linewidth=0.8,
                    alpha=0.4,
                    color="tomato",
                )

    plt.title(f"{freq.capitalize()} partition: history (blue) + validation last block (red)")
    plt.xlabel("Years")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"partition_{freq}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Saved: {out_path}")


def main() -> None:
    for freq in FREQUENCIES:
        plot_frequency_partition(freq)


if __name__ == "__main__":
    main()