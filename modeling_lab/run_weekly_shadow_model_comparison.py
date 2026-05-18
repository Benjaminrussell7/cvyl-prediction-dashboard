from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "modeling_lab"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LAB_DIR))

from cvyl_scraper.backtesting import _completed_games
from cvyl_scraper.hybrid import DEFAULT_POWER_V2_LOGISTIC_SCALE, power_v2_win_probability
from cvyl_scraper.model_comparison_v3 import _team_game_rows
from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency
from run_opponent_adjusted_experiment import build_opponent_adjusted_ratings


DEFAULT_GAMES_CSV = ROOT / "data" / "processed" / "cvyl_games.csv"
OUTPUT_DIR = ROOT / "modeling_lab" / "outputs"
REFERENCE_MODEL = "baseline_power_v3"
CANDIDATE_MODEL = "opponent_adjusted_power_v4"
LOOKAHEAD_DAYS = 7


RatingBuilder = Callable[[pd.DataFrame, dict[str, float]], dict[str, float]]


@dataclass(frozen=True)
class ShadowModel:
    name: str
    label: str
    rating_builder: RatingBuilder


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    games = _load_games(DEFAULT_GAMES_CSV)
    models = _model_registry()

    ranking_changes = build_weekly_ranking_changes(games, models)
    matchup_comparison = build_weekly_model_comparison(games, models, ranking_changes)
    disagreements = build_biggest_disagreements(matchup_comparison)
    template = build_interpretation_template(matchup_comparison, disagreements, ranking_changes)

    matchup_comparison.to_csv(OUTPUT_DIR / "weekly_model_comparison.csv", index=False)
    disagreements.to_csv(OUTPUT_DIR / "weekly_biggest_disagreements.csv", index=False)
    ranking_changes.to_csv(OUTPUT_DIR / "weekly_ranking_changes.csv", index=False)
    (OUTPUT_DIR / "weekly_model_comparison_interpretation_template.md").write_text(
        template,
        encoding="utf-8",
    )

    print(f"Wrote weekly shadow model comparison outputs to {OUTPUT_DIR}")


def build_weekly_ranking_changes(
    games: pd.DataFrame,
    models: list[ShadowModel],
) -> pd.DataFrame:
    rows = []
    for week_end in _weekly_cutoffs(games):
        ratings_by_model = _ratings_by_model_as_of(games, week_end, models)
        ranks_by_model = {
            model.name: _rank_ratings(ratings_by_model[model.name]) for model in models
        }
        all_teams = sorted({team for ranks in ranks_by_model.values() for team in ranks})
        for team in all_teams:
            reference_rank = ranks_by_model[REFERENCE_MODEL].get(team)
            candidate_rank = ranks_by_model[CANDIDATE_MODEL].get(team)
            reference_rating = ratings_by_model[REFERENCE_MODEL].get(team)
            candidate_rating = ratings_by_model[CANDIDATE_MODEL].get(team)
            rows.append(
                {
                    "week_start": _week_start(week_end).date().isoformat(),
                    "week_end": week_end.date().isoformat(),
                    "team": team,
                    "baseline_rank": reference_rank,
                    "power_v4_rank": candidate_rank,
                    "ranking_delta": _rank_delta(reference_rank, candidate_rank),
                    "baseline_rating": reference_rating,
                    "power_v4_rating": candidate_rating,
                    "rating_delta": _none_safe_delta(candidate_rating, reference_rating),
                }
            )
    return pd.DataFrame(rows)


def build_weekly_model_comparison(
    games: pd.DataFrame,
    models: list[ShadowModel],
    ranking_changes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for week_end in _weekly_cutoffs(games):
        ratings_by_model = _ratings_by_model_as_of(games, week_end, models)
        upcoming = _games_in_lookahead(games, week_end)
        if upcoming.empty:
            continue
        ranks_for_week = ranking_changes[ranking_changes["week_end"] == week_end.date().isoformat()]
        rank_lookup = ranks_for_week.set_index("team").to_dict("index") if not ranks_for_week.empty else {}

        for _, game in upcoming.iterrows():
            model_predictions = {
                model.name: _predict_game(game, ratings_by_model[model.name])
                for model in models
            }
            reference = model_predictions[REFERENCE_MODEL]
            candidate = model_predictions[CANDIDATE_MODEL]
            favorite_probability_delta = (
                candidate["favorite_probability"] - reference["favorite_probability"]
            )
            spread_delta = candidate["implied_spread"] - reference["implied_spread"]
            rows.append(
                {
                    "week_start": _week_start(week_end).date().isoformat(),
                    "week_end": week_end.date().isoformat(),
                    "target_game_date": game["game_date"].date().isoformat(),
                    "game_id": game["game_id"],
                    "status": game["status"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "home_score": game.get("home_score"),
                    "away_score": game.get("away_score"),
                    "baseline_home_rating": reference["home_rating"],
                    "baseline_away_rating": reference["away_rating"],
                    "power_v4_home_rating": candidate["home_rating"],
                    "power_v4_away_rating": candidate["away_rating"],
                    "baseline_home_win_probability": reference["home_probability"],
                    "power_v4_home_win_probability": candidate["home_probability"],
                    "home_probability_delta": candidate["home_probability"] - reference["home_probability"],
                    "baseline_favorite": reference["favorite"],
                    "power_v4_favorite": candidate["favorite"],
                    "baseline_favorite_probability": reference["favorite_probability"],
                    "power_v4_favorite_probability": candidate["favorite_probability"],
                    "probability_delta": favorite_probability_delta,
                    "baseline_implied_spread": reference["implied_spread"],
                    "power_v4_implied_spread": candidate["implied_spread"],
                    "spread_delta": spread_delta,
                    "baseline_home_rank": _team_rank(rank_lookup, game["home_team"], "baseline_rank"),
                    "baseline_away_rank": _team_rank(rank_lookup, game["away_team"], "baseline_rank"),
                    "power_v4_home_rank": _team_rank(rank_lookup, game["home_team"], "power_v4_rank"),
                    "power_v4_away_rank": _team_rank(rank_lookup, game["away_team"], "power_v4_rank"),
                    "home_ranking_delta": _team_rank(rank_lookup, game["home_team"], "ranking_delta"),
                    "away_ranking_delta": _team_rank(rank_lookup, game["away_team"], "ranking_delta"),
                    "ranking_delta": _matchup_ranking_delta(rank_lookup, game),
                    "predicted_winner_disagreement": reference["favorite"] != candidate["favorite"],
                    "upset_disagreement_flag": _upset_disagreement(reference, candidate),
                    "large_probability_disagreement": abs(favorite_probability_delta) >= 0.08,
                    "large_spread_disagreement": abs(spread_delta) >= 1.0,
                    "actual_winner": _actual_winner(game),
                    "baseline_correct": _prediction_correct(reference["favorite"], game),
                    "power_v4_correct": _prediction_correct(candidate["favorite"], game),
                }
            )
    return pd.DataFrame(rows)


def build_biggest_disagreements(matchup_comparison: pd.DataFrame) -> pd.DataFrame:
    if matchup_comparison.empty:
        return pd.DataFrame()
    disagreements = matchup_comparison.copy()
    disagreements["absolute_probability_delta"] = disagreements["probability_delta"].abs()
    disagreements["absolute_spread_delta"] = disagreements["spread_delta"].abs()
    disagreements["absolute_ranking_delta"] = disagreements["ranking_delta"].abs()
    disagreements["disagreement_score"] = (
        disagreements["absolute_probability_delta"] * 100
        + disagreements["absolute_spread_delta"] * 4
        + disagreements["absolute_ranking_delta"].fillna(0)
        + disagreements["predicted_winner_disagreement"].astype(int) * 20
    )
    return disagreements.sort_values(
        by=[
            "predicted_winner_disagreement",
            "upset_disagreement_flag",
            "disagreement_score",
            "absolute_probability_delta",
        ],
        ascending=[False, False, False, False],
        ignore_index=True,
    ).head(50)


def build_interpretation_template(
    matchup_comparison: pd.DataFrame,
    disagreements: pd.DataFrame,
    ranking_changes: pd.DataFrame,
) -> str:
    total_matchups = len(matchup_comparison)
    winner_disagreements = (
        int(matchup_comparison["predicted_winner_disagreement"].sum())
        if not matchup_comparison.empty
        else 0
    )
    mean_probability_delta = (
        float(matchup_comparison["probability_delta"].mean()) if not matchup_comparison.empty else 0.0
    )
    mean_spread_delta = (
        float(matchup_comparison["spread_delta"].mean()) if not matchup_comparison.empty else 0.0
    )
    largest_rank_moves = _largest_rank_moves(ranking_changes)
    top_disagreements = _top_disagreement_lines(disagreements.head(10))

    return "\n".join(
        [
            "# Weekly Shadow Model Comparison Template",
            "",
            "Scope: governance-only comparison between baseline Power v3 and the opponent-adjusted Power v4 candidate. This is not a production promotion.",
            "",
            "## Snapshot Summary",
            "",
            f"- Matchups compared: {total_matchups}",
            f"- Predicted winner disagreements: {winner_disagreements}",
            f"- Mean favorite probability delta, Power v4 minus baseline: {mean_probability_delta:+.3f}",
            f"- Mean implied spread delta, Power v4 minus baseline: {mean_spread_delta:+.2f}",
            "",
            "## Where Models Differ Most",
            "",
            top_disagreements,
            "",
            "## Directional Read",
            "",
            "- Does Power v4 appear directionally stronger this week? [Fill in after reviewing correctness, ranking deltas, and known team context.]",
            "- Are disagreements concentrated around isolated schedules, volatile teams, or specific divisions? [Fill in.]",
            "",
            "## Suspicious Predictions",
            "",
            "- Review rows with `predicted_winner_disagreement`, `upset_disagreement_flag`, `large_probability_disagreement`, or `large_spread_disagreement` in `weekly_biggest_disagreements.csv`.",
            "- Note any predictions where Power v4 moves strongly against recent observable results.",
            "",
            "## Calibration Observations",
            "",
            "- Check whether Power v4 probabilities are more or less compressed than baseline in the current week.",
            "- Check whether spread deltas are directionally plausible or too aggressive.",
            "",
            "## Largest Ranking Moves",
            "",
            largest_rank_moves,
            "",
            "## Outputs",
            "",
            "- `weekly_model_comparison.csv`",
            "- `weekly_biggest_disagreements.csv`",
            "- `weekly_ranking_changes.csv`",
        ]
    ) + "\n"


def _model_registry() -> list[ShadowModel]:
    return [
        ShadowModel(
            name=REFERENCE_MODEL,
            label="Baseline Power v3",
            rating_builder=_baseline_power_v3_ratings,
        ),
        ShadowModel(
            name=CANDIDATE_MODEL,
            label="Opponent-adjusted Power v4 candidate",
            rating_builder=_opponent_adjusted_power_v4_ratings,
        ),
    ]


def _baseline_power_v3_ratings(team_games: pd.DataFrame, _: dict[str, float]) -> dict[str, float]:
    if team_games.empty:
        return {}
    ratings = build_power_ratings_v3_recency(team_games)
    return dict(zip(ratings["team"], ratings["power_rating_v3_recency"], strict=False))


def _opponent_adjusted_power_v4_ratings(
    team_games: pd.DataFrame,
    baseline_ratings: dict[str, float],
) -> dict[str, float]:
    if team_games.empty:
        return {}
    adjusted, _ = build_opponent_adjusted_ratings(team_games, baseline_ratings)
    return adjusted


def _ratings_by_model_as_of(
    games: pd.DataFrame,
    cutoff: pd.Timestamp,
    models: list[ShadowModel],
) -> dict[str, dict[str, float]]:
    team_games = _team_games_as_of(games, cutoff)
    baseline_ratings = _baseline_power_v3_ratings(team_games, {})
    ratings_by_model: dict[str, dict[str, float]] = {}
    for model in models:
        if model.name == REFERENCE_MODEL:
            ratings_by_model[model.name] = baseline_ratings
        else:
            ratings_by_model[model.name] = model.rating_builder(team_games, baseline_ratings)
    return ratings_by_model


def _team_games_as_of(games: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    completed = _completed_games(games)
    completed = completed[completed["game_date"] <= cutoff]
    rows = []
    for _, game in completed.iterrows():
        rows.extend(_team_game_rows(game))
    return pd.DataFrame(rows)


def _predict_game(game: pd.Series, ratings: dict[str, float]) -> dict[str, object]:
    home_team = str(game["home_team"])
    away_team = str(game["away_team"])
    home_rating = float(ratings.get(home_team, 0.0))
    away_rating = float(ratings.get(away_team, 0.0))
    rating_diff = home_rating - away_rating
    home_probability = power_v2_win_probability(
        rating_diff,
        scale=DEFAULT_POWER_V2_LOGISTIC_SCALE,
    )
    favorite = home_team if home_probability >= 0.5 else away_team
    return {
        "home_rating": home_rating,
        "away_rating": away_rating,
        "rating_diff": rating_diff,
        "home_probability": home_probability,
        "favorite": favorite,
        "favorite_probability": max(home_probability, 1.0 - home_probability),
        "implied_spread": abs(rating_diff),
    }


def _weekly_cutoffs(games: pd.DataFrame) -> list[pd.Timestamp]:
    completed = _completed_games(games)
    if completed.empty:
        return []
    first = completed["game_date"].min().to_period("W-SUN").end_time.normalize()
    last = games["game_date"].max().to_period("W-SUN").end_time.normalize()
    cutoffs = pd.date_range(first, last, freq="W-SUN")
    return [pd.Timestamp(cutoff).normalize() for cutoff in cutoffs]


def _games_in_lookahead(games: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    start = cutoff + pd.Timedelta(days=1)
    end = cutoff + pd.Timedelta(days=LOOKAHEAD_DAYS)
    return games[
        (games["game_date"] >= start)
        & (games["game_date"] <= end)
        & games["home_team"].notna()
        & games["away_team"].notna()
    ].sort_values(["game_date", "game_time", "game_id"], na_position="last", ignore_index=True)


def _rank_ratings(ratings: dict[str, float]) -> dict[str, int]:
    ranked = sorted(ratings.items(), key=lambda item: (-item[1], item[0]))
    return {team: rank for rank, (team, _) in enumerate(ranked, start=1)}


def _actual_winner(game: pd.Series) -> str | None:
    if str(game.get("status")) != "completed":
        return None
    home_score = pd.to_numeric(game.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(game.get("away_score"), errors="coerce")
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return str(game["home_team"])
    if away_score > home_score:
        return str(game["away_team"])
    return "Tie"


def _prediction_correct(predicted_winner: str, game: pd.Series) -> bool | None:
    actual = _actual_winner(game)
    if actual is None:
        return None
    return predicted_winner == actual


def _upset_disagreement(reference: dict[str, object], candidate: dict[str, object]) -> bool:
    return bool(
        reference["favorite"] != candidate["favorite"]
        and max(reference["favorite_probability"], candidate["favorite_probability"]) >= 0.55
    )


def _load_games(path: Path) -> pd.DataFrame:
    games = pd.read_csv(path)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    return games.dropna(subset=["game_date"])


def _week_start(week_end: pd.Timestamp) -> pd.Timestamp:
    return week_end - pd.Timedelta(days=6)


def _rank_delta(reference_rank: int | None, candidate_rank: int | None) -> int | None:
    if reference_rank is None or candidate_rank is None:
        return None
    return candidate_rank - reference_rank


def _none_safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _team_rank(
    rank_lookup: dict[str, dict[str, object]],
    team: str,
    column: str,
) -> object:
    row = rank_lookup.get(str(team), {})
    return row.get(column)


def _matchup_ranking_delta(rank_lookup: dict[str, dict[str, object]], game: pd.Series) -> float:
    home_delta = _team_rank(rank_lookup, str(game["home_team"]), "ranking_delta")
    away_delta = _team_rank(rank_lookup, str(game["away_team"]), "ranking_delta")
    values = [value for value in [home_delta, away_delta] if value is not None and not pd.isna(value)]
    if not values:
        return 0.0
    return float(max(values, key=lambda value: abs(float(value))))


def _largest_rank_moves(ranking_changes: pd.DataFrame) -> str:
    if ranking_changes.empty:
        return "- No ranking data available."
    latest_week = ranking_changes["week_end"].max()
    latest = ranking_changes[ranking_changes["week_end"] == latest_week].copy()
    latest["absolute_ranking_delta"] = latest["ranking_delta"].abs()
    movers = latest.sort_values(
        ["absolute_ranking_delta", "team"],
        ascending=[False, True],
    ).head(10)
    if movers.empty:
        return "- No ranking movement found."
    return "\n".join(
        f"- {row['team']}: baseline rank {int(row['baseline_rank'])}, "
        f"Power v4 rank {int(row['power_v4_rank'])}, delta {int(row['ranking_delta']):+d}"
        for _, row in movers.iterrows()
        if not pd.isna(row["ranking_delta"])
    )


def _top_disagreement_lines(disagreements: pd.DataFrame) -> str:
    if disagreements.empty:
        return "- No matchup disagreements found."
    lines = []
    for _, row in disagreements.iterrows():
        lines.append(
            f"- {row['target_game_date']} {row['home_team']} vs {row['away_team']}: "
            f"baseline {row['baseline_favorite']} {row['baseline_favorite_probability']:.1%}, "
            f"Power v4 {row['power_v4_favorite']} {row['power_v4_favorite_probability']:.1%}, "
            f"spread delta {row['spread_delta']:+.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
