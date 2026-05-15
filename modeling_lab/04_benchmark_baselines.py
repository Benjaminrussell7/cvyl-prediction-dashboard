from __future__ import annotations

import pandas as pd

from common import OUTPUTS, model_home_probability, write_output


INPUT = "historical_pregame_matchups.csv"
OUTPUT = "baseline_predictions.csv"


def benchmark_baselines(matchups: pd.DataFrame) -> pd.DataFrame:
    rows = matchups[
        [
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "actual_winner",
            "predicted_winner",
            "predicted_win_probability",
            "power_v3_recency_predicted_winner",
            "power_v3_recency_win_probability",
            "power_v3_calibrated_predicted_winner",
            "power_v3_calibrated_win_probability",
        ]
    ].copy()
    rows["elo_home_probability"] = matchups.apply(
        lambda row: model_home_probability(row, "predicted_winner", "predicted_win_probability"),
        axis=1,
    )
    rows["power_v3_recency_home_probability"] = matchups.apply(
        lambda row: model_home_probability(
            row,
            "power_v3_recency_predicted_winner",
            "power_v3_recency_win_probability",
        ),
        axis=1,
    )
    rows["power_v3_calibrated_home_probability"] = matchups.apply(
        lambda row: model_home_probability(
            row,
            "power_v3_calibrated_predicted_winner",
            "power_v3_calibrated_win_probability",
        ),
        axis=1,
    )
    return rows.sort_values(["game_date", "game_id"], ignore_index=True)


def main() -> None:
    matchups = pd.read_csv(OUTPUTS / INPUT)
    output_path = write_output(benchmark_baselines(matchups), OUTPUT)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
