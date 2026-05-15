from __future__ import annotations

import pandas as pd

from common import read_processed, write_output


OUTPUT = "historical_pregame_matchups.csv"


def build_historical_pregame_matchups() -> pd.DataFrame:
    backtest = read_processed("cvyl_backtest.csv")
    comparison = read_processed("cvyl_model_comparison_v4_calibrated.csv")

    rows = backtest.merge(
        comparison[
            [
                "game_date",
                "home_team",
                "away_team",
                "power_v3_recency_predicted_winner",
                "power_v3_recency_win_probability",
                "power_v3_calibrated_predicted_winner",
                "power_v3_calibrated_win_probability",
            ]
        ],
        on=["game_date", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )
    rows["game_date"] = pd.to_datetime(rows["game_date"], errors="coerce")
    rows = rows.sort_values(["game_date", "game_id"], ignore_index=True)
    return rows


def main() -> None:
    output_path = write_output(build_historical_pregame_matchups(), OUTPUT)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
