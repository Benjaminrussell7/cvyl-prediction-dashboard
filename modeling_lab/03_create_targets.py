from __future__ import annotations

import pandas as pd

from common import OUTPUTS, actual_home_win_probability, write_output


INPUT = "historical_pregame_matchups.csv"
OUTPUT = "targets.csv"


def create_targets(matchups: pd.DataFrame) -> pd.DataFrame:
    targets = matchups[
        [
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "actual_winner",
        ]
    ].copy()
    targets["home_win"] = matchups.apply(actual_home_win_probability, axis=1)
    targets["home_margin"] = targets["home_score"] - targets["away_score"]
    targets["total_goals"] = targets["home_score"] + targets["away_score"]
    return targets.sort_values(["game_date", "game_id"], ignore_index=True)


def main() -> None:
    matchups = pd.read_csv(OUTPUTS / INPUT)
    output_path = write_output(create_targets(matchups), OUTPUT)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
