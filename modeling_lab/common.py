from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "modeling_lab" / "outputs"


def read_processed(filename: str) -> pd.DataFrame:
    path = PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing processed input: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().removeprefix("\ufeff") for column in frame.columns]
    return frame


def write_output(frame: pd.DataFrame, filename: str) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / filename
    frame.to_csv(path, index=False)
    return path


def actual_home_win_probability(row: pd.Series) -> float:
    if row["actual_winner"] == row["home_team"]:
        return 1.0
    if row["actual_winner"] == row["away_team"]:
        return 0.0
    return 0.5


def model_home_probability(row: pd.Series, winner_column: str, probability_column: str) -> float:
    probability = float(row[probability_column])
    if row[winner_column] == row["home_team"]:
        return probability
    if row[winner_column] == row["away_team"]:
        return 1.0 - probability
    return 0.5
