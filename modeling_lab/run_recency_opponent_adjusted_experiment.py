from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "modeling_lab"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LAB_DIR))

from cvyl_scraper.backtesting import _actual_winner, _completed_games
from cvyl_scraper.hybrid import DEFAULT_POWER_V2_LOGISTIC_SCALE, power_v2_win_probability
from cvyl_scraper.model_comparison_v3 import _team_game_rows
from cvyl_scraper.power_v2 import DEFAULT_SHRINKAGE_FACTOR
from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency
from run_opponent_adjusted_experiment import (
    ODDITY_TERMS,
    ROLLING_WINDOW_SIZE,
    _home_result,
    _markdown_table,
    _prior_total_goals,
    _team_rows,
    build_opponent_adjusted_ratings,
)


DEFAULT_GAMES_CSV = ROOT / "data" / "processed" / "cvyl_games.csv"
OUTPUT_DIR = ROOT / "modeling_lab" / "outputs"

VARIANT_BASELINE = "baseline_power_v3"
VARIANT_OPPONENT_ADJUSTED = "opponent_adjusted_performance"


@dataclass(frozen=True)
class RecencyVariant:
    name: str
    label: str
    half_life_games: float


RECENCY_VARIANTS = [
    RecencyVariant("recency_mild", "Opponent-adjusted recency mild", 6.0),
    RecencyVariant("recency_moderate", "Opponent-adjusted recency moderate", 3.0),
    RecencyVariant("recency_aggressive", "Opponent-adjusted recency aggressive", 1.5),
]

VARIANT_LABELS = {
    VARIANT_BASELINE: "Baseline Power v3",
    VARIANT_OPPONENT_ADJUSTED: "Opponent-adjusted performance",
    **{variant.name: variant.label for variant in RECENCY_VARIANTS},
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(DEFAULT_GAMES_CSV)

    predictions = run_rolling_backtests(games)
    summary = summarize_variants(predictions)
    rolling = summarize_rolling_windows(predictions)
    rankings = build_final_rankings(games)
    interpretation = build_interpretation(summary, rolling, rankings)

    predictions.to_csv(OUTPUT_DIR / "recency_opponent_adjusted_predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "recency_opponent_adjusted_summary.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "recency_opponent_adjusted_rolling_windows.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "recency_opponent_adjusted_final_rankings.csv", index=False)
    (OUTPUT_DIR / "recency_opponent_adjusted_interpretation.md").write_text(
        interpretation,
        encoding="utf-8",
    )

    print(f"Wrote recency opponent-adjusted lab outputs to {OUTPUT_DIR}")


def run_rolling_backtests(games: pd.DataFrame) -> pd.DataFrame:
    completed = _completed_games(games)
    prior_team_game_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    for game_number, (_, game) in enumerate(completed.iterrows(), start=1):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        actual_home_result = _home_result(home_score, away_score)
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        total_goals_prediction = _prior_total_goals(prior_team_game_rows)
        ratings_by_variant = _pregame_ratings(prior_team_game_rows)

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

            rows.append(
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

    return pd.DataFrame(rows)


def _pregame_ratings(prior_team_game_rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    if not prior_team_game_rows:
        return {variant: {} for variant in VARIANT_LABELS}

    team_games = pd.DataFrame(prior_team_game_rows)
    baseline = build_power_ratings_v3_recency(team_games)
    baseline_ratings = {
        str(team): float(rating)
        for team, rating in zip(
            baseline["team"],
            baseline["power_rating_v3_recency"],
            strict=False,
        )
    }
    opponent_adjusted_ratings, performance_games = build_opponent_adjusted_ratings(
        team_games,
        baseline_ratings,
    )

    ratings_by_variant = {
        VARIANT_BASELINE: baseline_ratings,
        VARIANT_OPPONENT_ADJUSTED: opponent_adjusted_ratings,
    }
    for variant in RECENCY_VARIANTS:
        ratings_by_variant[variant.name] = build_recency_weighted_ratings(
            performance_games,
            half_life_games=variant.half_life_games,
        )
    return ratings_by_variant


def build_recency_weighted_ratings(
    performance_games: pd.DataFrame,
    *,
    half_life_games: float,
    shrinkage_factor: float = DEFAULT_SHRINKAGE_FACTOR,
) -> dict[str, float]:
    if performance_games.empty:
        return {}
    if half_life_games <= 0:
        raise ValueError("half_life_games must be positive.")

    weighted = performance_games.copy()
    weighted["game_date"] = pd.to_datetime(weighted["game_date"], errors="coerce")
    weighted = weighted.sort_values(
        by=["team", "game_date", "game_id"],
        kind="mergesort",
        ignore_index=True,
    )
    weighted["team_game_index"] = weighted.groupby("team").cumcount()
    team_max_index = weighted.groupby("team")["team_game_index"].transform("max")
    weighted["team_games_ago"] = team_max_index - weighted["team_game_index"]
    weighted["exponential_recency_weight"] = 0.5 ** (
        weighted["team_games_ago"].astype(float) / half_life_games
    )
    weighted["combined_weight"] = (
        weighted["recency_weight"].astype(float) * weighted["exponential_recency_weight"]
    )

    ratings = (
        weighted.groupby("team", sort=True)
        .apply(_recency_team_row, include_groups=False)
        .reset_index()
    )
    ratings["games_played"] = ratings["games_played"].astype(int)
    ratings["shrinkage_multiplier"] = ratings["games_played"] / (
        ratings["games_played"] + shrinkage_factor
    )
    ratings["rating"] = (
        ratings["avg_performance_above_expectation"] * ratings["shrinkage_multiplier"]
    )
    return dict(zip(ratings["team"], ratings["rating"], strict=False))


def summarize_variants(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in predictions.groupby("variant", sort=False):
        rows.append(_metric_row(group, variant=variant, rolling_window=None))
    summary = pd.DataFrame(rows)
    baseline = summary[summary["variant"] == VARIANT_BASELINE].iloc[0]
    opponent_adjusted = summary[summary["variant"] == VARIANT_OPPONENT_ADJUSTED].iloc[0]

    summary["delta_winner_accuracy"] = summary["winner_accuracy"] - baseline["winner_accuracy"]
    summary["delta_brier_score"] = summary["brier_score"] - baseline["brier_score"]
    summary["delta_log_loss"] = summary["log_loss"] - baseline["log_loss"]
    summary["delta_margin_mae"] = summary["margin_mae"] - baseline["margin_mae"]
    summary["delta_total_goals_mae"] = summary["total_goals_mae"] - baseline["total_goals_mae"]
    summary["delta_accuracy_vs_opponent_adjusted"] = (
        summary["winner_accuracy"] - opponent_adjusted["winner_accuracy"]
    )
    summary["delta_brier_vs_opponent_adjusted"] = (
        summary["brier_score"] - opponent_adjusted["brier_score"]
    )
    summary["delta_log_loss_vs_opponent_adjusted"] = (
        summary["log_loss"] - opponent_adjusted["log_loss"]
    )
    summary["delta_margin_mae_vs_opponent_adjusted"] = (
        summary["margin_mae"] - opponent_adjusted["margin_mae"]
    )
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

    ratings_by_variant = _pregame_ratings(prior_rows)
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
    best = _best_variant(summary)
    opponent_adjusted = summary[summary["variant"] == VARIANT_OPPONENT_ADJUSTED].iloc[0]
    recency_rows = summary[summary["variant"].isin([variant.name for variant in RECENCY_VARIANTS])]
    best_recency = _best_variant(recency_rows)
    promoted = recency_rows[recency_rows["promotion_candidate"]]

    if promoted.empty:
        recommendation = "do not make a recency-weighted variant the leading Power v4 candidate."
    else:
        promoted_best = _best_variant(promoted)
        recommendation = (
            f"make {promoted_best['variant']} the leading recency-weighted Power v4 candidate."
        )

    instability_note = _instability_note(summary, rankings)

    lines = [
        "# Recency-Weighted Opponent-Adjusted Power v4 Lab",
        "",
        "Scope: lab-only rolling backtest using existing canonical games data. Tournament data was not used, and no Streamlit or production prediction code was modified.",
        "",
        "Approach: baseline Power v3 and the existing opponent-adjusted model are preserved exactly. The recency variants reuse the same opponent-adjusted per-game performance values, then apply a team-specific exponential decay by team-game age before averaging. Mild uses a 6-game half-life, moderate uses 3 games, and aggressive uses 1.5 games. The newest game for each team has full decay weight; older games decay by half every half-life interval.",
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
                    "delta_accuracy_vs_opponent_adjusted",
                    "delta_brier_vs_opponent_adjusted",
                    "delta_log_loss_vs_opponent_adjusted",
                    "delta_margin_mae_vs_opponent_adjusted",
                    "promotion_candidate",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        (
            f"- Best overall by Brier/log-loss/margin sort: {best['variant']} "
            f"(Brier {best['brier_score']:.4f}, log_loss {best['log_loss']:.4f}, "
            f"margin_mae {best['margin_mae']:.4f}, accuracy {best['winner_accuracy']:.4f})."
        ),
        (
            f"- Existing opponent-adjusted reference: Brier {opponent_adjusted['brier_score']:.4f}, "
            f"log_loss {opponent_adjusted['log_loss']:.4f}, "
            f"margin_mae {opponent_adjusted['margin_mae']:.4f}, "
            f"accuracy {opponent_adjusted['winner_accuracy']:.4f}."
        ),
        (
            f"- Best recency variant: {best_recency['variant']} "
            f"(Brier {best_recency['brier_score']:.4f}, log_loss {best_recency['log_loss']:.4f}, "
            f"margin_mae {best_recency['margin_mae']:.4f}, accuracy {best_recency['winner_accuracy']:.4f})."
        ),
        _recency_improvement_sentence(recency_rows),
        f"- Instability/overreaction: {instability_note}",
        f"- Recommendation: {recommendation}",
        "",
        "## Ranking Movement",
        "",
    ]

    for label, terms in ODDITY_TERMS.items():
        lines.append(_ranking_note(label, _team_rows(rankings, terms)))

    lines.extend(
        [
            "",
            "## Material Movers",
            "",
            _material_movers(rankings),
            "",
            "## Rolling Windows",
            "",
            f"- Rolling window size: {ROLLING_WINDOW_SIZE} games.",
            f"- Rolling windows evaluated: {rolling['rolling_window'].nunique() if not rolling.empty else 0}.",
            "- Full window metrics are in `recency_opponent_adjusted_rolling_windows.csv`.",
            "",
            "## Files",
            "",
            "- `recency_opponent_adjusted_predictions.csv`",
            "- `recency_opponent_adjusted_summary.csv`",
            "- `recency_opponent_adjusted_rolling_windows.csv`",
            "- `recency_opponent_adjusted_final_rankings.csv`",
            "- `recency_opponent_adjusted_interpretation.md`",
        ]
    )
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


def _recency_team_row(group: pd.DataFrame) -> pd.Series:
    weights = group["combined_weight"].astype(float)
    return pd.Series(
        {
            "games_played": len(group),
            "avg_performance_above_expectation": _weighted_average(
                group["performance_above_expectation"],
                weights,
            ),
            "average_decay_weight": float(group["exponential_recency_weight"].mean()),
        }
    )


def _promotion_candidate(row: pd.Series) -> bool:
    if row["variant"] in {VARIANT_BASELINE, VARIANT_OPPONENT_ADJUSTED}:
        return False
    improved_metric = (
        row["delta_brier_vs_opponent_adjusted"] < 0
        or row["delta_log_loss_vs_opponent_adjusted"] < 0
        or row["delta_margin_mae_vs_opponent_adjusted"] < 0
    )
    return bool(improved_metric and row["delta_accuracy_vs_opponent_adjusted"] >= -0.01)


def _best_variant(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(
        by=["brier_score", "log_loss", "margin_mae", "winner_accuracy"],
        ascending=[True, True, True, False],
    ).iloc[0]


def _recency_improvement_sentence(recency_rows: pd.DataFrame) -> str:
    improved = recency_rows[
        (recency_rows["delta_brier_vs_opponent_adjusted"] < 0)
        | (recency_rows["delta_log_loss_vs_opponent_adjusted"] < 0)
        | (recency_rows["delta_margin_mae_vs_opponent_adjusted"] < 0)
    ]
    if improved.empty:
        return "- No recency-weighted variant improved Brier, log loss, or margin MAE versus the existing opponent-adjusted model."
    names = ", ".join(improved["variant"].tolist())
    return f"- Recency variants improving at least one core metric versus opponent-adjusted: {names}."


def _instability_note(summary: pd.DataFrame, rankings: pd.DataFrame) -> str:
    aggressive = summary[summary["variant"] == "recency_aggressive"].iloc[0]
    opponent = summary[summary["variant"] == VARIANT_OPPONENT_ADJUSTED].iloc[0]
    rank_changes = _rank_changes(rankings, "recency_aggressive")
    large_moves = int((rank_changes.abs() >= 5).sum()) if not rank_changes.empty else 0
    if aggressive["winner_accuracy"] < opponent["winner_accuracy"] or large_moves >= 5:
        return (
            "aggressive weighting shows overreaction risk, with "
            f"{large_moves} teams moving at least five ranking spots."
        )
    return (
        "no obvious overreaction in aggregate accuracy; "
        f"{large_moves} teams moved at least five ranking spots under aggressive weighting."
    )


def _ranking_note(team_label: str, rows: pd.DataFrame) -> str:
    if rows.empty:
        return f"- {team_label}: not present in final ranking output."

    parts = []
    for variant in VARIANT_LABELS:
        match = rows[rows["variant"] == variant]
        if match.empty:
            continue
        row = match.iloc[0]
        parts.append(f"{variant} rank {int(row['rank'])}, rating {row['rating']:.2f}")
    return f"- {team_label}: " + "; ".join(parts) + "."


def _material_movers(rankings: pd.DataFrame) -> str:
    changes = _rank_changes(rankings, "recency_aggressive")
    if changes.empty:
        return "- No ranking movement data available."
    movers = changes.reindex(changes.abs().sort_values(ascending=False).head(10).index)
    parts = [f"{team} {change:+d}" for team, change in movers.items() if change != 0]
    if not parts:
        return "- No teams moved in final aggressive-recency rankings versus opponent-adjusted."
    return "- Largest aggressive-recency rank changes versus opponent-adjusted: " + "; ".join(parts) + "."


def _rank_changes(rankings: pd.DataFrame, variant: str) -> pd.Series:
    reference = rankings[rankings["variant"] == VARIANT_OPPONENT_ADJUSTED][["team", "rank"]]
    target = rankings[rankings["variant"] == variant][["team", "rank"]]
    merged = reference.merge(target, on="team", suffixes=("_reference", "_target"))
    if merged.empty:
        return pd.Series(dtype="int64")
    changes = merged["rank_target"].astype(int) - merged["rank_reference"].astype(int)
    changes.index = merged["team"]
    return changes


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float((values.astype(float) * weights.astype(float)).sum() / weights.astype(float).sum())


if __name__ == "__main__":
    main()
