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
class BlendVariant:
    name: str
    label: str
    baseline_weight: float
    opponent_adjusted_weight: float


BLENDS = [
    BlendVariant("blend_90_10", "Blend 90% baseline / 10% opponent-adjusted", 0.90, 0.10),
    BlendVariant("blend_80_20", "Blend 80% baseline / 20% opponent-adjusted", 0.80, 0.20),
    BlendVariant("blend_70_30", "Blend 70% baseline / 30% opponent-adjusted", 0.70, 0.30),
    BlendVariant("blend_60_40", "Blend 60% baseline / 40% opponent-adjusted", 0.60, 0.40),
]

VARIANT_LABELS = {
    VARIANT_BASELINE: "Baseline Power v3",
    VARIANT_OPPONENT_ADJUSTED: "Opponent-adjusted performance",
    **{blend.name: blend.label for blend in BLENDS},
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(DEFAULT_GAMES_CSV)

    predictions = run_rolling_backtests(games)
    summary = summarize_variants(predictions)
    rolling = summarize_rolling_windows(predictions)
    rankings = build_final_rankings(games)
    interpretation = build_interpretation(summary, rolling, rankings)

    predictions.to_csv(OUTPUT_DIR / "blended_power_v4_predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "blended_power_v4_summary.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "blended_power_v4_rolling_windows.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "blended_power_v4_final_rankings.csv", index=False)
    (OUTPUT_DIR / "blended_power_v4_interpretation.md").write_text(
        interpretation,
        encoding="utf-8",
    )

    print(f"Wrote blended Power v4 lab outputs to {OUTPUT_DIR}")


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
    opponent_adjusted_ratings, _ = build_opponent_adjusted_ratings(team_games, baseline_ratings)

    ratings_by_variant = {
        VARIANT_BASELINE: baseline_ratings,
        VARIANT_OPPONENT_ADJUSTED: opponent_adjusted_ratings,
    }
    for blend in BLENDS:
        ratings_by_variant[blend.name] = _blend_ratings(
            baseline_ratings,
            opponent_adjusted_ratings,
            baseline_weight=blend.baseline_weight,
            opponent_adjusted_weight=blend.opponent_adjusted_weight,
        )
    return ratings_by_variant


def _blend_ratings(
    baseline_ratings: dict[str, float],
    opponent_adjusted_ratings: dict[str, float],
    *,
    baseline_weight: float,
    opponent_adjusted_weight: float,
) -> dict[str, float]:
    teams = set(baseline_ratings) | set(opponent_adjusted_ratings)
    return {
        team: (baseline_weight * baseline_ratings.get(team, 0.0))
        + (opponent_adjusted_weight * opponent_adjusted_ratings.get(team, 0.0))
        for team in teams
    }


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
    blend_rows = summary[summary["variant"].isin([blend.name for blend in BLENDS])]
    best_blend = _best_variant(blend_rows)
    promoted_blends = blend_rows[blend_rows["promotion_candidate"]]
    improved_over_pure = blend_rows[
        (blend_rows["delta_brier_vs_opponent_adjusted"] < 0)
        & (blend_rows["delta_log_loss_vs_opponent_adjusted"] < 0)
        & (blend_rows["delta_margin_mae_vs_opponent_adjusted"] < 0)
    ]

    if promoted_blends.empty:
        recommendation = "do not make a blended variant the leading Power v4 candidate."
    elif improved_over_pure.empty:
        recommendation = (
            "do not make a blended variant the leading Power v4 candidate; "
            f"{best_blend['variant']} is the best blend versus baseline, but pure "
            "opponent-adjusted remains stronger on Brier, log loss, and margin MAE."
        )
    else:
        promoted_best = _best_variant(promoted_blends)
        recommendation = (
            f"make {promoted_best['variant']} the leading blended Power v4 candidate."
        )

    lines = [
        "# Blended Power v4 Modeling Lab",
        "",
        "Scope: lab-only rolling backtest using existing canonical games data. No Streamlit or production prediction code was modified.",
        "",
        "Variant definition: baseline Power v3 and the opponent-adjusted performance model are preserved exactly. Blended variants are linear combinations of those two pregame team ratings at 90/10, 80/20, 70/30, and 60/40 weights.",
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
            f"- Best blended variant: {best_blend['variant']} "
            f"(Brier {best_blend['brier_score']:.4f}, log_loss {best_blend['log_loss']:.4f}, "
            f"margin_mae {best_blend['margin_mae']:.4f}, accuracy {best_blend['winner_accuracy']:.4f})."
        ),
        (
            f"- Pure opponent-adjusted reference: Brier {opponent_adjusted['brier_score']:.4f}, "
            f"log_loss {opponent_adjusted['log_loss']:.4f}, "
            f"margin_mae {opponent_adjusted['margin_mae']:.4f}."
        ),
        _blend_improvement_sentence(improved_over_pure),
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
            "## Rolling Windows",
            "",
            f"- Rolling window size: {ROLLING_WINDOW_SIZE} games.",
            f"- Rolling windows evaluated: {rolling['rolling_window'].nunique() if not rolling.empty else 0}.",
            "- Full window metrics are in `blended_power_v4_rolling_windows.csv`.",
            "",
            "## Files",
            "",
            "- `blended_power_v4_predictions.csv`",
            "- `blended_power_v4_summary.csv`",
            "- `blended_power_v4_rolling_windows.csv`",
            "- `blended_power_v4_final_rankings.csv`",
            "- `blended_power_v4_interpretation.md`",
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


def _promotion_candidate(row: pd.Series) -> bool:
    if row["variant"] in {VARIANT_BASELINE, VARIANT_OPPONENT_ADJUSTED}:
        return False
    return bool(
        row["delta_brier_score"] < 0
        and row["delta_log_loss"] < 0
        and row["delta_margin_mae"] < 0
        and row["delta_winner_accuracy"] >= -0.01
    )


def _best_variant(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(
        by=["brier_score", "log_loss", "margin_mae", "winner_accuracy"],
        ascending=[True, True, True, False],
    ).iloc[0]


def _blend_improvement_sentence(improved_over_pure: pd.DataFrame) -> str:
    if improved_over_pure.empty:
        return "- No blended variant improved Brier, log loss, and margin MAE versus pure opponent-adjusted."
    names = ", ".join(improved_over_pure["variant"].tolist())
    return f"- Blended variants improving all three metrics versus pure opponent-adjusted: {names}."


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


if __name__ == "__main__":
    main()
