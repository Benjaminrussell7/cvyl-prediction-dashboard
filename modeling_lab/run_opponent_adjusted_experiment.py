from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvyl_scraper.backtesting import _actual_winner, _completed_games
from cvyl_scraper.hybrid import DEFAULT_POWER_V2_LOGISTIC_SCALE, power_v2_win_probability
from cvyl_scraper.model_comparison_v3 import _team_game_rows
from cvyl_scraper.power_v2 import DEFAULT_MARGIN_CAP, DEFAULT_SHRINKAGE_FACTOR
from cvyl_scraper.power_v3_recency import (
    DEFAULT_RECENCY_MAX_WEIGHT,
    DEFAULT_RECENCY_MIN_WEIGHT,
    add_recency_weights,
    build_power_ratings_v3_recency,
)


DEFAULT_GAMES_CSV = ROOT / "data" / "processed" / "cvyl_games.csv"
OUTPUT_DIR = ROOT / "modeling_lab" / "outputs"
ROLLING_WINDOW_SIZE = 25
DEFAULT_TOTAL_GOALS = 12.0

VARIANT_BASELINE = "baseline_power_v3"
VARIANT_OPPONENT_ADJUSTED = "opponent_adjusted_performance"
VARIANT_LABELS = {
    VARIANT_BASELINE: "Baseline Power v3",
    VARIANT_OPPONENT_ADJUSTED: "Opponent-adjusted performance",
}
ODDITY_TERMS = {
    "RHAM": ["RHAM"],
    "Westfield": ["Westfield"],
    "Avon": ["Avon"],
    "Suffield": ["Suffield"],
    "Somers": ["Somers"],
    "West Hartford Green": ["West", "Hartford", "Green"],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(DEFAULT_GAMES_CSV)

    predictions, performance_games = run_rolling_backtests(games)
    summary = summarize_variants(predictions)
    rolling = summarize_rolling_windows(predictions)
    rankings = build_final_rankings(games)
    interpretation = build_interpretation(summary, rolling, rankings)

    predictions.to_csv(OUTPUT_DIR / "opponent_adjusted_predictions.csv", index=False)
    performance_games.to_csv(
        OUTPUT_DIR / "opponent_adjusted_performance_games.csv",
        index=False,
    )
    summary.to_csv(OUTPUT_DIR / "opponent_adjusted_summary.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "opponent_adjusted_rolling_windows.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "opponent_adjusted_final_rankings.csv", index=False)
    (OUTPUT_DIR / "opponent_adjusted_interpretation.md").write_text(
        interpretation,
        encoding="utf-8",
    )

    print(f"Wrote opponent-adjusted lab outputs to {OUTPUT_DIR}")


def run_rolling_backtests(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = _completed_games(games)
    prior_team_game_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    performance_snapshots: list[pd.DataFrame] = []

    for game_number, (_, game) in enumerate(completed.iterrows(), start=1):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        actual_home_result = _home_result(home_score, away_score)
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        total_goals_prediction = _prior_total_goals(prior_team_game_rows)
        ratings_by_variant, performance_games = _pregame_ratings(prior_team_game_rows)

        if not performance_games.empty:
            snapshot = performance_games.copy()
            snapshot.insert(0, "prediction_game_number", game_number)
            performance_snapshots.append(snapshot)

        for variant, ratings in ratings_by_variant.items():
            home_rating = ratings.get(home_team, 0.0)
            away_rating = ratings.get(away_team, 0.0)
            predicted_margin = home_rating - away_rating
            home_probability = power_v2_win_probability(
                predicted_margin,
                scale=DEFAULT_POWER_V2_LOGISTIC_SCALE,
            )
            predicted_winner = home_team if home_probability >= 0.5 else away_team
            actual_winner = _actual_winner(home_team, away_team, home_score, away_score)

            prediction_rows.append(
                {
                    "game_number": game_number,
                    "game_id": game["game_id"],
                    "game_date": game["game_date"].date().isoformat(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "home_rating": home_rating,
                    "away_rating": away_rating,
                    "rating_diff": predicted_margin,
                    "home_win_probability": home_probability,
                    "predicted_winner": predicted_winner,
                    "actual_winner": actual_winner,
                    "prediction_correct": predicted_winner == actual_winner,
                    "home_score": home_score,
                    "away_score": away_score,
                    "actual_home_result": actual_home_result,
                    "predicted_margin": predicted_margin,
                    "actual_margin": actual_margin,
                    "predicted_total_goals": total_goals_prediction,
                    "actual_total_goals": actual_total,
                }
            )

        prior_team_game_rows.extend(_team_game_rows(game))

    performance = (
        pd.concat(performance_snapshots, ignore_index=True)
        if performance_snapshots
        else pd.DataFrame(columns=_performance_columns())
    )
    return pd.DataFrame(prediction_rows), performance


def _pregame_ratings(
    prior_team_game_rows: list[dict[str, object]],
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    if not prior_team_game_rows:
        return {VARIANT_BASELINE: {}, VARIANT_OPPONENT_ADJUSTED: {}}, pd.DataFrame()

    team_games = pd.DataFrame(prior_team_game_rows)
    baseline = build_power_ratings_v3_recency(team_games)
    baseline_ratings = dict(
        zip(baseline["team"], baseline["power_rating_v3_recency"], strict=False)
    )
    adjusted, performance_games = build_opponent_adjusted_ratings(team_games, baseline_ratings)
    return {
        VARIANT_BASELINE: {team: float(rating) for team, rating in baseline_ratings.items()},
        VARIANT_OPPONENT_ADJUSTED: adjusted,
    }, performance_games


def build_opponent_adjusted_ratings(
    team_games: pd.DataFrame,
    baseline_ratings: dict[str, float],
    *,
    margin_cap: float = DEFAULT_MARGIN_CAP,
    shrinkage_factor: float = DEFAULT_SHRINKAGE_FACTOR,
    recency_min_weight: float = DEFAULT_RECENCY_MIN_WEIGHT,
    recency_max_weight: float = DEFAULT_RECENCY_MAX_WEIGHT,
) -> tuple[dict[str, float], pd.DataFrame]:
    completed = _completed_team_games(team_games)
    if completed.empty:
        return {}, pd.DataFrame(columns=_performance_columns())

    weighted = add_recency_weights(
        completed,
        recency_min_weight=recency_min_weight,
        recency_max_weight=recency_max_weight,
    )
    weighted["raw_margin"] = weighted["points_for"] - weighted["points_against"]
    weighted["capped_margin"] = weighted["raw_margin"].clip(lower=-margin_cap, upper=margin_cap)
    weighted["opponent_power_v3_rating"] = (
        weighted["opponent"].map(baseline_ratings).fillna(0.0).astype(float)
    )
    weighted["expected_margin_vs_opponent"] = -weighted["opponent_power_v3_rating"]
    weighted["performance_above_expectation"] = (
        weighted["capped_margin"] - weighted["expected_margin_vs_opponent"]
    )
    weighted["performance_above_expectation"] = weighted[
        "performance_above_expectation"
    ].clip(lower=-margin_cap, upper=margin_cap)

    ratings = (
        weighted.groupby("team", sort=True)
        .apply(_opponent_adjusted_team_row, include_groups=False)
        .reset_index()
    )
    ratings["games_played"] = ratings["games_played"].astype(int)
    ratings["shrinkage_multiplier"] = ratings["games_played"] / (
        ratings["games_played"] + shrinkage_factor
    )
    ratings["opponent_adjusted_rating"] = (
        ratings["avg_performance_above_expectation"] * ratings["shrinkage_multiplier"]
    )
    adjusted = dict(zip(ratings["team"], ratings["opponent_adjusted_rating"], strict=False))

    performance_games = weighted[
        [
            "game_id",
            "game_date",
            "team",
            "opponent",
            "points_for",
            "points_against",
            "raw_margin",
            "capped_margin",
            "opponent_power_v3_rating",
            "expected_margin_vs_opponent",
            "performance_above_expectation",
            "recency_weight",
        ]
    ].copy()
    return {team: float(rating) for team, rating in adjusted.items()}, performance_games


def summarize_variants(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in predictions.groupby("variant", sort=False):
        rows.append(_metric_row(group, variant=variant, rolling_window=None))
    summary = pd.DataFrame(rows)
    baseline = summary[summary["variant"] == VARIANT_BASELINE].iloc[0]

    summary["delta_winner_accuracy"] = summary["winner_accuracy"] - baseline["winner_accuracy"]
    summary["delta_brier_score"] = summary["brier_score"] - baseline["brier_score"]
    summary["delta_log_loss"] = summary["log_loss"] - baseline["log_loss"]
    summary["delta_margin_mae"] = summary["margin_mae"] - baseline["margin_mae"]
    summary["delta_total_goals_mae"] = summary["total_goals_mae"] - baseline["total_goals_mae"]
    summary["promotion_candidate"] = summary.apply(_promotion_candidate, axis=1)
    return summary


def summarize_rolling_windows(predictions: pd.DataFrame) -> pd.DataFrame:
    windows = []
    max_game = int(predictions["game_number"].max()) if not predictions.empty else 0
    for start in range(1, max_game + 1, ROLLING_WINDOW_SIZE):
        end = min(max_game, start + ROLLING_WINDOW_SIZE - 1)
        window = predictions[
            (predictions["game_number"] >= start) & (predictions["game_number"] <= end)
        ]
        for variant, group in window.groupby("variant", sort=False):
            windows.append(_metric_row(group, variant=variant, rolling_window=f"{start}-{end}"))
    return pd.DataFrame(windows)


def build_final_rankings(games: pd.DataFrame) -> pd.DataFrame:
    completed = _completed_games(games)
    prior_rows: list[dict[str, object]] = []
    for _, game in completed.iterrows():
        prior_rows.extend(_team_game_rows(game))

    ratings_by_variant, _ = _pregame_ratings(prior_rows)
    rows = []
    for variant, ratings in ratings_by_variant.items():
        ranked = sorted(ratings.items(), key=lambda item: (-item[1], item[0]))
        for rank, (team, rating) in enumerate(ranked, start=1):
            rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "team": team,
                    "rank": rank,
                    "rating": rating,
                }
            )
    return pd.DataFrame(rows)


def build_interpretation(
    summary: pd.DataFrame,
    rolling: pd.DataFrame,
    rankings: pd.DataFrame,
) -> str:
    baseline = summary[summary["variant"] == VARIANT_BASELINE].iloc[0]
    adjusted = summary[summary["variant"] == VARIANT_OPPONENT_ADJUSTED].iloc[0]
    candidates = summary[summary["promotion_candidate"]]

    if bool(adjusted["promotion_candidate"]):
        recommendation = "promote the opponent-adjusted variant under the stated criteria."
    else:
        recommendation = "do not promote the opponent-adjusted variant under the stated criteria."

    lines = [
        "# Opponent-Adjusted Performance Power v3 Lab",
        "",
        "Scope: lab-only rolling backtest using existing canonical games data. No Streamlit or production prediction code was modified.",
        "",
        "Variant definition: each pregame window first builds the exact baseline Power v3 ratings. The opponent-adjusted variant evaluates every prior team-game by capped actual margin, expected margin for an average team against that opponent based on the opponent's prior Power v3 rating, and performance above or below that expectation. Team ratings are the recency-weighted average of those performance values with the same shrinkage behavior used by the baseline.",
        "",
        "## Overall Metrics",
        "",
        _markdown_table(
            summary[
                [
                    "variant",
                    "winner_accuracy",
                    "brier_score",
                    "log_loss",
                    "margin_mae",
                    "total_goals_mae",
                    "games",
                    "delta_winner_accuracy",
                    "delta_brier_score",
                    "delta_log_loss",
                    "delta_margin_mae",
                    "promotion_candidate",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        (
            f"- Baseline: winner_accuracy {baseline['winner_accuracy']:.3f}, "
            f"Brier {baseline['brier_score']:.3f}, log_loss {baseline['log_loss']:.3f}, "
            f"margin_mae {baseline['margin_mae']:.3f}."
        ),
        (
            f"- Opponent-adjusted: winner_accuracy {adjusted['winner_accuracy']:.3f} "
            f"({adjusted['delta_winner_accuracy']:+.3f}), Brier {adjusted['brier_score']:.3f} "
            f"({adjusted['delta_brier_score']:+.3f}), log_loss {adjusted['log_loss']:.3f} "
            f"({adjusted['delta_log_loss']:+.3f}), margin_mae {adjusted['margin_mae']:.3f} "
            f"({adjusted['delta_margin_mae']:+.3f})."
        ),
        f"- Promotion recommendation: {recommendation}",
        "",
        "## Ranking Oddities",
        "",
    ]

    for label, terms in ODDITY_TERMS.items():
        lines.append(_ranking_note(label, _team_rows(rankings, terms)))

    lines.extend(
        [
            "",
            "## Rolling Windows",
            "",
            f"- Rolling window size: {ROLLING_WINDOW_SIZE} games.",
            f"- Rolling windows evaluated: {rolling['rolling_window'].nunique() if not rolling.empty else 0}.",
            "- Full window metrics are in `opponent_adjusted_rolling_windows.csv`.",
            "",
            "## Files",
            "",
            "- `opponent_adjusted_predictions.csv`",
            "- `opponent_adjusted_performance_games.csv`",
            "- `opponent_adjusted_summary.csv`",
            "- `opponent_adjusted_rolling_windows.csv`",
            "- `opponent_adjusted_final_rankings.csv`",
            "- `opponent_adjusted_interpretation.md`",
        ]
    )
    if not candidates.empty:
        lines.append("")
        lines.append(f"Promotion candidates: {', '.join(candidates['variant'].tolist())}.")
    return "\n".join(lines) + "\n"


def _metric_row(
    group: pd.DataFrame,
    *,
    variant: str,
    rolling_window: str | None,
) -> dict[str, object]:
    probabilities = group["home_win_probability"].clip(1e-6, 1 - 1e-6)
    results = group["actual_home_result"]
    row = {
        "variant": variant,
        "variant_label": group["variant_label"].iloc[0],
        "winner_accuracy": float(group["prediction_correct"].mean()),
        "brier_score": float(((probabilities - results) ** 2).mean()),
        "log_loss": float(
            -(
                results * probabilities.map(math.log)
                + (1 - results) * (1 - probabilities).map(math.log)
            ).mean()
        ),
        "margin_mae": float((group["predicted_margin"] - group["actual_margin"]).abs().mean()),
        "total_goals_mae": float(
            (group["predicted_total_goals"] - group["actual_total_goals"]).abs().mean()
        ),
        "games": int(len(group)),
    }
    if rolling_window is not None:
        row["rolling_window"] = rolling_window
    return row


def _opponent_adjusted_team_row(group: pd.DataFrame) -> pd.Series:
    weights = group["recency_weight"].astype(float)
    performance = group["performance_above_expectation"].astype(float)
    return pd.Series(
        {
            "games_played": len(group),
            "avg_actual_margin": _weighted_average(group["raw_margin"], weights),
            "avg_capped_margin": _weighted_average(group["capped_margin"], weights),
            "avg_expected_margin_vs_opponent": _weighted_average(
                group["expected_margin_vs_opponent"],
                weights,
            ),
            "avg_performance_above_expectation": _weighted_average(performance, weights),
        }
    )


def _promotion_candidate(row: pd.Series) -> bool:
    if row["variant"] == VARIANT_BASELINE:
        return False
    return bool(
        row["delta_brier_score"] < 0
        and row["delta_log_loss"] < 0
        and row["delta_winner_accuracy"] >= -0.01
        and row["delta_margin_mae"] <= 0
    )


def _completed_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame()

    completed = team_games[team_games["status"] == "completed"].copy()
    completed["points_for"] = pd.to_numeric(completed["points_for"], errors="coerce")
    completed["points_against"] = pd.to_numeric(completed["points_against"], errors="coerce")
    completed["game_date"] = pd.to_datetime(completed.get("game_date"), errors="coerce")
    return completed.dropna(subset=["team", "opponent", "points_for", "points_against"])


def _ranking_note(team_label: str, rows: pd.DataFrame) -> str:
    if rows.empty:
        return f"- {team_label}: not present in final ranking output."

    baseline = rows[rows["variant"] == VARIANT_BASELINE]
    adjusted = rows[rows["variant"] == VARIANT_OPPONENT_ADJUSTED]
    if baseline.empty or adjusted.empty:
        return f"- {team_label}: present, but missing one variant row."

    baseline_row = baseline.iloc[0]
    adjusted_row = adjusted.iloc[0]
    rank_delta = int(adjusted_row["rank"]) - int(baseline_row["rank"])
    return (
        f"- {team_label}: baseline rank {int(baseline_row['rank'])}, rating "
        f"{baseline_row['rating']:.2f}; opponent-adjusted rank {int(adjusted_row['rank'])}, "
        f"rating {adjusted_row['rating']:.2f}; rank change {rank_delta:+d}."
    )


def _team_rows(rankings: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    if rankings.empty:
        return rankings
    mask = pd.Series(True, index=rankings.index)
    for term in terms:
        mask = mask & rankings["team"].str.contains(term, case=False, na=False)
    return rankings[mask]


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_markdown_cell(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float((values.astype(float) * weights.astype(float)).sum() / weights.astype(float).sum())


def _prior_total_goals(prior_team_game_rows: list[dict[str, object]]) -> float:
    if not prior_team_game_rows:
        return DEFAULT_TOTAL_GOALS
    prior = pd.DataFrame(prior_team_game_rows)
    totals = pd.to_numeric(prior["points_for"], errors="coerce") + pd.to_numeric(
        prior["points_against"],
        errors="coerce",
    )
    return float(totals.dropna().mean())


def _home_result(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if away_score > home_score:
        return 0.0
    return 0.5


def _performance_columns() -> list[str]:
    return [
        "prediction_game_number",
        "game_id",
        "game_date",
        "team",
        "opponent",
        "points_for",
        "points_against",
        "raw_margin",
        "capped_margin",
        "opponent_power_v3_rating",
        "expected_margin_vs_opponent",
        "performance_above_expectation",
        "recency_weight",
    ]


if __name__ == "__main__":
    main()
