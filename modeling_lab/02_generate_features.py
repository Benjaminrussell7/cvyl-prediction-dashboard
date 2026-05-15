from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUTPUTS, write_output


INPUT = "historical_pregame_matchups.csv"
OUTPUT = "leakage_safe_features.csv"


def generate_leakage_safe_features(matchups: pd.DataFrame) -> pd.DataFrame:
    rows = matchups.sort_values(["game_date", "game_id"], ignore_index=True).copy()
    histories: dict[str, list[dict[str, float]]] = {}
    feature_rows: list[dict[str, object]] = []

    for _, game in rows.iterrows():
        home = str(game["home_team"])
        away = str(game["away_team"])
        home_history = histories.get(home, [])
        away_history = histories.get(away, [])
        home_profile = team_profile(home_history)
        away_profile = team_profile(away_history)

        feature_row = base_feature_row(game)
        feature_row.update(matchup_features(home_profile, away_profile))
        feature_rows.append(feature_row)

        update_histories(histories, game, home_profile, away_profile)

    return pd.DataFrame(feature_rows).sort_values(["game_date", "game_id"], ignore_index=True)


def base_feature_row(game: pd.Series) -> dict[str, object]:
    elo_home_probability = model_home_probability(
        game,
        "predicted_winner",
        "predicted_win_probability",
    )
    power_v3_home_probability = model_home_probability(
        game,
        "power_v3_recency_predicted_winner",
        "power_v3_recency_win_probability",
    )
    calibrated_home_probability = model_home_probability(
        game,
        "power_v3_calibrated_predicted_winner",
        "power_v3_calibrated_win_probability",
    )
    return {
        "game_id": game["game_id"],
        "game_date": game["game_date"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "pregame_home_elo": float(game["pregame_home_elo"]),
        "pregame_away_elo": float(game["pregame_away_elo"]),
        "predicted_win_probability": float(game["predicted_win_probability"]),
        "power_v3_recency_win_probability": float(game["power_v3_recency_win_probability"]),
        "power_v3_calibrated_win_probability": float(game["power_v3_calibrated_win_probability"]),
        "elo_diff_home": float(game["pregame_home_elo"]) - float(game["pregame_away_elo"]),
        "elo_home_probability": elo_home_probability,
        "power_v3_recency_home_probability": power_v3_home_probability,
        "power_v3_calibrated_home_probability": calibrated_home_probability,
    }


def matchup_features(home: dict[str, float], away: dict[str, float]) -> dict[str, float]:
    return {
        "rolling_margin_std_last_3": home["margin_std_3"] - away["margin_std_3"],
        "rolling_margin_std_last_5": home["margin_std_5"] - away["margin_std_5"],
        "rolling_goals_for_std_last_5": home["goals_for_std_5"] - away["goals_for_std_5"],
        "rolling_goals_against_std_last_5": home["goals_against_std_5"] - away["goals_against_std_5"],
        "close_game_rate": home["close_game_rate"] - away["close_game_rate"],
        "upset_rate": home["upset_rate"] - away["upset_rate"],
        "rolling_margin_last_3": home["margin_avg_3"] - away["margin_avg_3"],
        "rolling_margin_last_5": home["margin_avg_5"] - away["margin_avg_5"],
        "recent_vs_season_margin_delta": home["recent_vs_season_margin_delta"] - away["recent_vs_season_margin_delta"],
        "offensive_trend": home["offensive_trend"] - away["offensive_trend"],
        "defensive_trend": home["defensive_trend"] - away["defensive_trend"],
        "elo_trend": home["elo_trend"] - away["elo_trend"],
        "power_rating_trend": home["power_rating_trend"] - away["power_rating_trend"],
        "rolling_trend_slope": home["rolling_trend_slope"] - away["rolling_trend_slope"],
        "offense_vs_opponent_defense_gap": home["goals_for_avg_5"] - away["goals_against_avg_5"],
        "defense_vs_opponent_offense_gap": away["goals_for_avg_5"] - home["goals_against_avg_5"],
        "projected_pace_interaction": (home["pace_avg_5"] + away["pace_avg_5"]) / 2.0,
        "volatility_mismatch": abs(home["margin_std_5"] - away["margin_std_5"]),
        "strength_gap_absolute": abs(home["season_margin"] - away["season_margin"]),
        "offense_defense_ratio": safe_ratio(home["goals_for_avg_5"], away["goals_against_avg_5"]),
        "offensive_sos": home["offensive_sos"] - away["offensive_sos"],
        "defensive_sos": home["defensive_sos"] - away["defensive_sos"],
        "recent_opponent_avg_power": home["recent_opponent_avg_power"] - away["recent_opponent_avg_power"],
        "recent_opponent_avg_elo": home["recent_opponent_avg_elo"] - away["recent_opponent_avg_elo"],
        "opponent_win_pct": home["opponent_win_pct"] - away["opponent_win_pct"],
        "games_played": home["games_played"],
        "opponent_games_played": away["games_played"],
        "rating_stability": home["rating_stability"] - away["rating_stability"],
        "recent_result_volatility": home["recent_result_volatility"] - away["recent_result_volatility"],
    }


def team_profile(history: list[dict[str, float]]) -> dict[str, float]:
    margins = values(history, "margin")
    goals_for = values(history, "goals_for")
    goals_against = values(history, "goals_against")
    total_goals = goals_for + goals_against if len(goals_for) and len(goals_against) else np.array([])
    expected = values(history, "expected_win_probability")
    wins = values(history, "win")
    opponent_power = values(history, "opponent_power_proxy")
    opponent_elo = values(history, "opponent_pregame_elo")
    opponent_win_pct = values(history, "opponent_win_pct")
    postgame_elo = values(history, "postgame_elo")

    season_margin = mean_or_zero(margins)
    margin_avg_3 = mean_last(margins, 3)
    margin_avg_5 = mean_last(margins, 5)
    prior_goals_for = mean_prior_window(goals_for, recent=3)
    recent_goals_for = mean_last(goals_for, 3)
    prior_goals_against = mean_prior_window(goals_against, recent=3)
    recent_goals_against = mean_last(goals_against, 3)
    return {
        "games_played": float(len(history)),
        "margin_std_3": std_last(margins, 3),
        "margin_std_5": std_last(margins, 5),
        "goals_for_std_5": std_last(goals_for, 5),
        "goals_against_std_5": std_last(goals_against, 5),
        "close_game_rate": rate(np.abs(margins) <= 2) if len(margins) else 0.0,
        "upset_rate": rate((wins == 1.0) & (expected < 0.5)) if len(wins) else 0.0,
        "win_pct": mean_or_zero(wins),
        "margin_avg_3": margin_avg_3,
        "margin_avg_5": margin_avg_5,
        "season_margin": season_margin,
        "recent_vs_season_margin_delta": margin_avg_3 - season_margin,
        "offensive_trend": recent_goals_for - prior_goals_for,
        "defensive_trend": prior_goals_against - recent_goals_against,
        "elo_trend": trend_delta(postgame_elo),
        "power_rating_trend": margin_avg_3 - mean_prior_window(margins, recent=3),
        "rolling_trend_slope": slope_last(margins, 5),
        "goals_for_avg_5": mean_last(goals_for, 5),
        "goals_against_avg_5": mean_last(goals_against, 5),
        "pace_avg_5": mean_last(total_goals, 5),
        "offensive_sos": mean_last(values(history, "opponent_goals_against_avg"), 5),
        "defensive_sos": mean_last(values(history, "opponent_goals_for_avg"), 5),
        "recent_opponent_avg_power": mean_last(opponent_power, 5),
        "recent_opponent_avg_elo": mean_last(opponent_elo, 5),
        "opponent_win_pct": mean_last(opponent_win_pct, 5),
        "rating_stability": 1.0 / (1.0 + std_last(margins, 5)),
        "recent_result_volatility": std_last(margins, 5),
    }


def update_histories(
    histories: dict[str, list[dict[str, float]]],
    game: pd.Series,
    home_profile: dict[str, float],
    away_profile: dict[str, float],
) -> None:
    home = str(game["home_team"])
    away = str(game["away_team"])
    home_expected = elo_probability(float(game["pregame_home_elo"]), float(game["pregame_away_elo"]))
    away_expected = 1.0 - home_expected
    append_game_history(
        histories,
        team=home,
        goals_for=float(game["home_score"]),
        goals_against=float(game["away_score"]),
        win=1.0 if game["actual_winner"] == home else 0.0,
        expected_win_probability=home_expected,
        pregame_elo=float(game["pregame_home_elo"]),
        opponent_pregame_elo=float(game["pregame_away_elo"]),
        postgame_elo=np.nan,
        opponent_profile=away_profile,
    )
    append_game_history(
        histories,
        team=away,
        goals_for=float(game["away_score"]),
        goals_against=float(game["home_score"]),
        win=1.0 if game["actual_winner"] == away else 0.0,
        expected_win_probability=away_expected,
        pregame_elo=float(game["pregame_away_elo"]),
        opponent_pregame_elo=float(game["pregame_home_elo"]),
        postgame_elo=np.nan,
        opponent_profile=home_profile,
    )


def append_game_history(
    histories: dict[str, list[dict[str, float]]],
    *,
    team: str,
    goals_for: float,
    goals_against: float,
    win: float,
    expected_win_probability: float,
    pregame_elo: float,
    opponent_pregame_elo: float,
    postgame_elo: float,
    opponent_profile: dict[str, float],
) -> None:
    histories.setdefault(team, []).append(
        {
            "goals_for": goals_for,
            "goals_against": goals_against,
            "margin": goals_for - goals_against,
            "win": win,
            "expected_win_probability": expected_win_probability,
            "pregame_elo": pregame_elo,
            "opponent_pregame_elo": opponent_pregame_elo,
            "postgame_elo": pregame_elo if np.isnan(postgame_elo) else postgame_elo,
            "opponent_power_proxy": opponent_profile["season_margin"],
            "opponent_goals_for_avg": opponent_profile["goals_for_avg_5"],
            "opponent_goals_against_avg": opponent_profile["goals_against_avg_5"],
            "opponent_win_pct": opponent_profile.get("win_pct", 0.0),
        }
    )


def model_home_probability(game: pd.Series, winner_column: str, probability_column: str) -> float:
    probability = float(game[probability_column])
    if game[winner_column] == game["home_team"]:
        return probability
    if game[winner_column] == game["away_team"]:
        return 1.0 - probability
    return 0.5


def elo_probability(team_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - team_elo) / 400.0))


def values(history: list[dict[str, float]], key: str) -> np.ndarray:
    return np.array([row.get(key, 0.0) for row in history], dtype=float)


def mean_or_zero(values_array: np.ndarray) -> float:
    return float(values_array.mean()) if len(values_array) else 0.0


def mean_last(values_array: np.ndarray, window: int) -> float:
    if len(values_array) == 0:
        return 0.0
    return float(values_array[-window:].mean())


def mean_prior_window(values_array: np.ndarray, *, recent: int) -> float:
    if len(values_array) <= recent:
        return mean_or_zero(values_array)
    return float(values_array[:-recent][-recent:].mean())


def std_last(values_array: np.ndarray, window: int) -> float:
    if len(values_array) < 2:
        return 0.0
    return float(values_array[-window:].std(ddof=0))


def rate(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def trend_delta(values_array: np.ndarray) -> float:
    if len(values_array) < 2:
        return 0.0
    return float(values_array[-1] - values_array[0])


def slope_last(values_array: np.ndarray, window: int) -> float:
    if len(values_array) < 2:
        return 0.0
    y = values_array[-window:]
    x = np.arange(len(y), dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 0.1))


def main() -> None:
    matchups = pd.read_csv(OUTPUTS / INPUT)
    output_path = write_output(generate_leakage_safe_features(matchups), OUTPUT)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
