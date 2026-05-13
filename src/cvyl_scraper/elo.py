from __future__ import annotations

from math import log

import pandas as pd


DEFAULT_STARTING_ELO = 1500.0
DEFAULT_K_FACTOR = 20.0
DEFAULT_RECENCY_MIN_MULTIPLIER = 0.85
DEFAULT_RECENCY_GROWTH_GAMES = 20.0

ELO_RATING_COLUMNS = ["team", "elo", "games_played"]
ELO_HISTORY_COLUMNS = [
    "game_id",
    "team",
    "opponent",
    "points_for",
    "points_against",
    "win",
    "game_date",
    "season",
    "division",
    "status",
    "pregame_elo",
    "opponent_pregame_elo",
    "postgame_elo",
]


def build_elo_outputs(
    team_games: pd.DataFrame,
    *,
    starting_elo: float = DEFAULT_STARTING_ELO,
    k_factor: float = DEFAULT_K_FACTOR,
    recency_min_multiplier: float = DEFAULT_RECENCY_MIN_MULTIPLIER,
    recency_growth_games: float = DEFAULT_RECENCY_GROWTH_GAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings: dict[str, float] = {}
    games_played: dict[str, int] = {}
    history_rows: list[dict[str, object]] = []

    completed = _completed_team_games(team_games)
    game_index = 0
    for _, game_rows in _chronological_games(completed):
        if len(game_rows) != 2:
            continue

        first = game_rows.iloc[0]
        second = game_rows.iloc[1]
        first_team = str(first["team"])
        second_team = str(second["team"])
        first_pregame = ratings.get(first_team, starting_elo)
        second_pregame = ratings.get(second_team, starting_elo)

        game_recency_multiplier = recency_multiplier(
            game_index,
            min_multiplier=recency_min_multiplier,
            growth_games=recency_growth_games,
        )
        first_postgame, second_postgame = updated_elo_pair(
            first_pregame,
            second_pregame,
            int(first["points_for"]),
            int(second["points_for"]),
            k_factor=k_factor,
            recency_multiplier_value=game_recency_multiplier,
        )
        ratings[first_team] = first_postgame
        ratings[second_team] = second_postgame
        games_played[first_team] = games_played.get(first_team, 0) + 1
        games_played[second_team] = games_played.get(second_team, 0) + 1

        history_rows.append(
            _history_row(first, first_pregame, second_pregame, first_postgame)
        )
        history_rows.append(
            _history_row(second, second_pregame, first_pregame, second_postgame)
        )
        game_index += 1

    ratings_frame = pd.DataFrame(
        [
            {"team": team, "elo": rating, "games_played": games_played[team]}
            for team, rating in ratings.items()
        ],
        columns=ELO_RATING_COLUMNS,
    ).sort_values(by=["elo", "team"], ascending=[False, True], ignore_index=True)

    history = pd.DataFrame(history_rows, columns=ELO_HISTORY_COLUMNS)
    return ratings_frame, history


def _completed_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame(columns=ELO_HISTORY_COLUMNS)

    completed = team_games[team_games["status"] == "completed"].copy()
    if completed.empty:
        return completed

    completed["game_date"] = pd.to_datetime(completed["game_date"], errors="coerce")
    completed["points_for"] = pd.to_numeric(completed["points_for"], errors="coerce")
    completed["points_against"] = pd.to_numeric(completed["points_against"], errors="coerce")
    return completed.dropna(
        subset=["game_id", "team", "opponent", "game_date", "points_for", "points_against"]
    )


def _chronological_games(team_games: pd.DataFrame):
    sorted_games = team_games.sort_values(
        by=["game_date", "game_id", "team"],
        ascending=[True, True, True],
        na_position="last",
    )
    return sorted_games.groupby("game_id", sort=False)


def _expected_score(team_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_elo - team_elo) / 400.0))


def recency_multiplier(
    game_index: int,
    *,
    min_multiplier: float = DEFAULT_RECENCY_MIN_MULTIPLIER,
    growth_games: float = DEFAULT_RECENCY_GROWTH_GAMES,
) -> float:
    if game_index < 0:
        raise ValueError("game_index must be non-negative.")
    if not 0 < min_multiplier <= 1:
        raise ValueError("min_multiplier must be greater than 0 and no more than 1.")
    if growth_games <= 0:
        raise ValueError("growth_games must be greater than 0.")

    progress = game_index / (game_index + growth_games)
    return min_multiplier + (1.0 - min_multiplier) * progress


def updated_elo_pair(
    first_elo: float,
    second_elo: float,
    first_score: int,
    second_score: int,
    *,
    k_factor: float = DEFAULT_K_FACTOR,
    recency_multiplier_value: float = 1.0,
) -> tuple[float, float]:
    first_actual = _actual_score(first_score, second_score)
    second_actual = 1.0 - first_actual
    first_expected = _expected_score(first_elo, second_elo)
    second_expected = 1.0 - first_expected
    margin_multiplier = _margin_multiplier(abs(first_score - second_score))
    update_size = k_factor * margin_multiplier * recency_multiplier_value
    first_delta = update_size * (first_actual - first_expected)
    second_delta = update_size * (second_actual - second_expected)
    return first_elo + first_delta, second_elo + second_delta


def _actual_score(points_for: int, points_against: int) -> float:
    if points_for > points_against:
        return 1.0
    if points_for < points_against:
        return 0.0
    return 0.5


def _margin_multiplier(margin: int) -> float:
    return max(1.0, log(margin + 1))


def _history_row(
    row: pd.Series,
    pregame_elo: float,
    opponent_pregame_elo: float,
    postgame_elo: float,
) -> dict[str, object]:
    return {
        "game_id": row["game_id"],
        "team": row["team"],
        "opponent": row["opponent"],
        "points_for": int(row["points_for"]),
        "points_against": int(row["points_against"]),
        "win": bool(row["win"]),
        "game_date": row["game_date"].date().isoformat(),
        "season": row["season"],
        "division": row["division"],
        "status": row["status"],
        "pregame_elo": pregame_elo,
        "opponent_pregame_elo": opponent_pregame_elo,
        "postgame_elo": postgame_elo,
    }
