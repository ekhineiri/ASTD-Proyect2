from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "reports" / "plots_by_frequency"


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


def plot_all_series_for_frequency(freq: str) -> None:
	file_path = DATA_DIR / f"train_{freq}.csv"
	if not file_path.exists():
		print(f"[WARN] Missing file: {file_path}")
		return

	df = pd.read_csv(file_path)
	if df.empty:
		print(f"[WARN] Empty file: {file_path}")
		return

	series_id_col = df.columns[0]
	value_cols = df.columns[1:]
	values = df[value_cols].apply(pd.to_numeric, errors="coerce")

	ppy = periods_per_year(freq)

	plt.figure(figsize=(14, 7))
	for row_idx in range(len(df)):
		series_id = str(df.iloc[row_idx][series_id_col])
		row_values = values.iloc[row_idx].to_numpy(dtype=float)
		valid_mask = ~np.isnan(row_values)
		if not valid_mask.any():
			continue

		x = np.arange(len(row_values), dtype=float) / ppy
		plt.plot(
			x[valid_mask],
			row_values[valid_mask],
			linewidth=0.9,
			alpha=0.85,
			label=series_id,
		)

	plt.title(f"All {freq.capitalize()} time series (x-axis in years)")
	plt.xlabel("Years")
	plt.ylabel("Value")
	plt.grid(True, alpha=0.3)

	if len(df) <= 20:
		plt.legend(ncol=2, fontsize=8)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	out_path = OUTPUT_DIR / f"{freq}_all_series_years.png"
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()

	print(f"[OK] Saved: {out_path}")


def main() -> None:
	for freq in FREQUENCIES:
		plot_all_series_for_frequency(freq)


if __name__ == "__main__":
	main()
