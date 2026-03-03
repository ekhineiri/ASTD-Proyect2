from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "validation_last_block"


HORIZONS = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "weekly": 13,
    "daily": 14,
    "hourly": 48,
}


def to_numeric_series(row: pd.Series) -> list[float]:
    values = pd.to_numeric(row, errors="coerce").dropna()
    return values.tolist()


def pad_rows(rows: Iterable[list[float]], prefix: str) -> pd.DataFrame:
    rows = list(rows)
    width = max((len(row) for row in rows), default=0)
    columns = [f"{prefix}{idx}" for idx in range(1, width + 1)]
    padded = [row + [pd.NA] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, columns=columns)


def split_last_block(values: list[float], horizon: int) -> tuple[list[float], list[float]]:
    if len(values) <= horizon:
        raise ValueError(
            f"Series length ({len(values)}) must be greater than horizon ({horizon})"
        )
    return values[:-horizon], values[-horizon:]


def process_frequency(freq: str, horizon: int, input_dir: Path, output_dir: Path) -> None:
    source = input_dir / f"train_{freq}.csv"
    if not source.exists():
        print(f"[WARN] Missing file: {source}")
        return

    raw = pd.read_csv(source)
    if raw.empty:
        print(f"[WARN] Empty file: {source}")
        return

    id_col = raw.columns[0]
    value_cols = raw.columns[1:]

    history_rows: list[list[float]] = []
    validation_rows: list[list[float]] = []
    ids: list[str] = []

    for _, row in raw.iterrows():
        series_id = str(row[id_col])
        series_values = to_numeric_series(row[value_cols])

        history, validation = split_last_block(series_values, horizon)
        ids.append(series_id)
        history_rows.append(history)
        validation_rows.append(validation)

    history_df = pad_rows(history_rows, "V")
    validation_df = pd.DataFrame(
        validation_rows,
        columns=[f"V{idx}" for idx in range(1, horizon + 1)],
    )

    history_df.insert(0, "ID", ids)
    validation_df.insert(0, "ID", ids)

    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / f"history_{freq}.csv"
    validation_path = output_dir / f"validation_{freq}.csv"

    history_df.to_csv(history_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    print(
        f"[OK] {freq}: horizon={horizon}, series={len(ids)} -> "
        f"{history_path.name}, {validation_path.name}"
    )


def main() -> None:
    for freq, horizon in HORIZONS.items():
        process_frequency(freq, horizon, DATA_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()