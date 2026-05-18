from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "modeling_lab" / "outputs"
PREDICTIONS_CSV = OUTPUT_DIR / "opponent_adjusted_predictions.csv"

TARGET_VARIANT = "opponent_adjusted_performance"
BASELINE_VARIANT = "baseline_power_v3"
PROBABILITY_BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
PROBABILITY_LABELS = [
    "50-55%",
    "55-60%",
    "60-65%",
    "65-70%",
    "70-75%",
    "75-80%",
    "80-85%",
    "85%+",
]
SPREAD_BUCKETS = [0, 1, 2, 3, 4, 5, 100]
SPREAD_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS_CSV)
    prepared = prepare_predictions(predictions)

    summary = build_summary(prepared)
    team_volatility = build_team_volatility(prepared)
    interpretation = build_interpretation(prepared, summary, team_volatility)

    summary.to_csv(OUTPUT_DIR / "matchup_calibration_summary.csv", index=False)
    team_volatility.to_csv(OUTPUT_DIR / "matchup_calibration_team_volatility.csv", index=False)
    (OUTPUT_DIR / "matchup_calibration_interpretation.md").write_text(
        interpretation,
        encoding="utf-8",
    )

    print(f"Wrote matchup calibration outputs to {OUTPUT_DIR}")


def prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions[
        predictions["variant"].isin([BASELINE_VARIANT, TARGET_VARIANT])
    ].copy()
    selected["favorite_probability"] = selected["home_win_probability"].where(
        selected["home_win_probability"] >= 0.5,
        1.0 - selected["home_win_probability"],
    )
    selected["favorite_team"] = selected["home_team"].where(
        selected["home_win_probability"] >= 0.5,
        selected["away_team"],
    )
    selected["underdog_team"] = selected["away_team"].where(
        selected["home_win_probability"] >= 0.5,
        selected["home_team"],
    )
    selected["favorite_actual_margin"] = selected["actual_margin"].where(
        selected["favorite_team"] == selected["home_team"],
        -selected["actual_margin"],
    )
    selected["favorite_won"] = selected["favorite_actual_margin"] > 0
    selected["favorite_implied_spread"] = selected["predicted_margin"].abs()
    selected["favorite_spread_error"] = (
        selected["favorite_implied_spread"] - selected["favorite_actual_margin"]
    )
    selected["absolute_margin_error"] = (
        selected["predicted_margin"] - selected["actual_margin"]
    ).abs()
    selected["absolute_total_error"] = (
        selected["predicted_total_goals"] - selected["actual_total_goals"]
    ).abs()
    selected["probability_bucket"] = pd.cut(
        selected["favorite_probability"],
        bins=PROBABILITY_BUCKETS,
        labels=PROBABILITY_LABELS,
        include_lowest=True,
        right=False,
    )
    selected["spread_bucket"] = pd.cut(
        selected["favorite_implied_spread"],
        bins=SPREAD_BUCKETS,
        labels=SPREAD_LABELS,
        include_lowest=True,
        right=False,
    )
    return selected


def build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target = predictions[predictions["variant"] == TARGET_VARIANT].copy()

    for bucket, group in target.groupby("probability_bucket", observed=False):
        if group.empty:
            rows.append(_empty_bucket_row("probability_bucket", str(bucket)))
            continue
        predicted = float(group["favorite_probability"].mean())
        actual = float(group["favorite_won"].mean())
        rows.append(
            {
                "section": "probability_bucket",
                "bucket": str(bucket),
                "variant": TARGET_VARIANT,
                "games": int(len(group)),
                "predicted_win_rate": predicted,
                "actual_win_rate": actual,
                "calibration_error": actual - predicted,
                "mean_implied_spread": float(group["favorite_implied_spread"].mean()),
                "mean_actual_favorite_margin": float(group["favorite_actual_margin"].mean()),
                "mean_spread_error": float(group["favorite_spread_error"].mean()),
                "margin_mae": float(group["absolute_margin_error"].mean()),
                "notes": _confidence_note(actual - predicted),
            }
        )

    for bucket, group in target.groupby("spread_bucket", observed=False):
        if group.empty:
            rows.append(_empty_bucket_row("spread_bucket", str(bucket)))
            continue
        rows.append(
            {
                "section": "spread_bucket",
                "bucket": str(bucket),
                "variant": TARGET_VARIANT,
                "games": int(len(group)),
                "predicted_win_rate": float(group["favorite_probability"].mean()),
                "actual_win_rate": float(group["favorite_won"].mean()),
                "calibration_error": float(
                    group["favorite_won"].mean() - group["favorite_probability"].mean()
                ),
                "mean_implied_spread": float(group["favorite_implied_spread"].mean()),
                "mean_actual_favorite_margin": float(group["favorite_actual_margin"].mean()),
                "mean_spread_error": float(group["favorite_spread_error"].mean()),
                "margin_mae": float(group["absolute_margin_error"].mean()),
                "notes": _spread_note(float(group["favorite_spread_error"].mean())),
            }
        )

    rows.extend(_overall_rows(predictions))
    rows.extend(_upset_rows(target))
    return pd.DataFrame(rows)


def build_team_volatility(predictions: pd.DataFrame) -> pd.DataFrame:
    target = predictions[predictions["variant"] == TARGET_VARIANT].copy()
    team_rows = []
    for _, row in target.iterrows():
        team_rows.append(_team_game_row(row, home=True))
        team_rows.append(_team_game_row(row, home=False))
    team_games = pd.DataFrame(team_rows)
    team_games = team_games.sort_values(["team", "game_number", "game_id"], ignore_index=True)
    team_games["rating_change"] = team_games.groupby("team")["pregame_rating"].diff()
    team_games["week"] = pd.to_datetime(team_games["game_date"]).dt.to_period("W").astype(str)
    weekly = (
        team_games.groupby(["team", "week"], as_index=False)
        .agg(weekly_rating=("pregame_rating", "last"))
        .sort_values(["team", "week"], ignore_index=True)
    )
    weekly["weekly_rating_change"] = weekly.groupby("team")["weekly_rating"].diff()
    weekly_volatility = weekly.groupby("team")["weekly_rating_change"].std().rename(
        "weekly_rating_volatility"
    )

    summary = (
        team_games.groupby("team", as_index=False)
        .agg(
            games=("game_id", "size"),
            winner_accuracy=("favorite_side_correct", "mean"),
            mean_abs_margin_error=("absolute_margin_error", "mean"),
            max_abs_margin_error=("absolute_margin_error", "max"),
            outcome_margin_std=("actual_team_margin", "std"),
            rating_volatility=("rating_change", "std"),
            mean_pregame_rating=("pregame_rating", "mean"),
            final_observed_rating=("pregame_rating", "last"),
            upset_losses_as_favorite=("upset_loss_as_favorite", "sum"),
            upset_wins_as_underdog=("upset_win_as_underdog", "sum"),
        )
        .merge(weekly_volatility, on="team", how="left")
    )
    summary["prediction_difficulty_score"] = (
        summary["mean_abs_margin_error"].fillna(0)
        + summary["outcome_margin_std"].fillna(0)
        + summary["rating_volatility"].fillna(0)
    )
    return summary.sort_values(
        by=["prediction_difficulty_score", "mean_abs_margin_error", "team"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def build_interpretation(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    team_volatility: pd.DataFrame,
) -> str:
    target = predictions[predictions["variant"] == TARGET_VARIANT]
    probability_rows = summary[summary["section"] == "probability_bucket"]
    spread_rows = summary[summary["section"] == "spread_bucket"]
    overall = summary[
        (summary["section"] == "overall") & (summary["variant"] == TARGET_VARIANT)
    ].iloc[0]
    baseline = summary[
        (summary["section"] == "overall") & (summary["variant"] == BASELINE_VARIANT)
    ].iloc[0]
    valid_probability = probability_rows[probability_rows["games"] > 0]
    mean_abs_calibration_error = float(valid_probability["calibration_error"].abs().mean())
    high_confidence = target[target["favorite_probability"] >= 0.75]
    large_favorites = target[target["favorite_implied_spread"] >= 4.0]
    favorite_spread_bias = float(target["favorite_spread_error"].mean())

    hardest = team_volatility.head(8)
    volatile = team_volatility.sort_values(
        by=["weekly_rating_volatility", "rating_volatility"],
        ascending=[False, False],
        ignore_index=True,
    ).head(8)
    largest_errors = team_volatility.sort_values(
        by=["mean_abs_margin_error", "max_abs_margin_error"],
        ascending=[False, False],
        ignore_index=True,
    ).head(8)

    lines = [
        "# Matchup Calibration Analysis",
        "",
        "Scope: lab-only analytical evaluation using saved rolling predictions for baseline Power v3 and opponent_adjusted_performance. No tournament data, UI changes, or production prediction changes were used.",
        "",
        "## Probability Calibration",
        "",
        (
            f"- Opponent-adjusted overall favorite accuracy was {overall['actual_win_rate']:.3f} "
            f"with mean favorite probability {overall['predicted_win_rate']:.3f}; "
            f"mean absolute bucket calibration error was {mean_abs_calibration_error:.3f}."
        ),
        (
            f"- Baseline comparison: favorite accuracy {baseline['actual_win_rate']:.3f}, "
            f"mean favorite probability {baseline['predicted_win_rate']:.3f}."
        ),
        _calibration_conclusion(valid_probability),
        "",
        "## Spread Realism",
        "",
        (
            f"- Average implied favorite spread was {overall['mean_implied_spread']:.2f}; "
            f"average actual favorite margin was {overall['mean_actual_favorite_margin']:.2f}; "
            f"mean spread error was {favorite_spread_bias:+.2f} goals."
        ),
        _large_favorite_sentence(large_favorites),
        _spread_conclusion(spread_rows),
        "",
        "## Upsets And Difficult Matchups",
        "",
        _top_games_sentence("Biggest predicted upsets", _biggest_predicted_upsets(target)),
        _top_games_sentence("Biggest actual upsets", _biggest_actual_upsets(target)),
        "- Hardest teams to model: " + _team_list(hardest, "prediction_difficulty_score"),
        "- Most volatile weekly ratings: " + _team_list(volatile, "weekly_rating_volatility"),
        "- Largest average prediction errors: " + _team_list(largest_errors, "mean_abs_margin_error"),
        "",
        "## Recommendation",
        "",
        _readiness_recommendation(mean_abs_calibration_error, favorite_spread_bias, high_confidence),
        "- Future modeling directions: calibrate probability scale, consider a spread-to-probability calibration layer, and add team-level uncertainty so volatile teams get less extreme probabilities.",
        "",
        "## Output Files",
        "",
        "- `matchup_calibration_summary.csv`",
        "- `matchup_calibration_team_volatility.csv`",
        "- `matchup_calibration_interpretation.md`",
    ]
    return "\n".join(lines) + "\n"


def _overall_rows(predictions: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for variant, group in predictions.groupby("variant", sort=False):
        rows.append(
            {
                "section": "overall",
                "bucket": "all",
                "variant": variant,
                "games": int(len(group)),
                "predicted_win_rate": float(group["favorite_probability"].mean()),
                "actual_win_rate": float(group["favorite_won"].mean()),
                "calibration_error": float(
                    group["favorite_won"].mean() - group["favorite_probability"].mean()
                ),
                "mean_implied_spread": float(group["favorite_implied_spread"].mean()),
                "mean_actual_favorite_margin": float(group["favorite_actual_margin"].mean()),
                "mean_spread_error": float(group["favorite_spread_error"].mean()),
                "margin_mae": float(group["absolute_margin_error"].mean()),
                "notes": "overall model comparison",
            }
        )
    return rows


def _upset_rows(target: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for label, group in [
        ("biggest_predicted_upsets", _biggest_predicted_upsets(target)),
        ("biggest_actual_upsets", _biggest_actual_upsets(target)),
        ("largest_margin_errors", target.sort_values("absolute_margin_error", ascending=False).head(10)),
    ]:
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            rows.append(
                {
                    "section": label,
                    "bucket": str(rank),
                    "variant": TARGET_VARIANT,
                    "games": 1,
                    "predicted_win_rate": float(row["favorite_probability"]),
                    "actual_win_rate": float(row["favorite_won"]),
                    "calibration_error": float(row["favorite_won"] - row["favorite_probability"]),
                    "mean_implied_spread": float(row["favorite_implied_spread"]),
                    "mean_actual_favorite_margin": float(row["favorite_actual_margin"]),
                    "mean_spread_error": float(row["favorite_spread_error"]),
                    "margin_mae": float(row["absolute_margin_error"]),
                    "notes": (
                        f"{row['game_date']}: {row['favorite_team']} favored over "
                        f"{row['underdog_team']}, actual {row['home_team']} "
                        f"{int(row['home_score'])}-{int(row['away_score'])} {row['away_team']}"
                    ),
                }
            )
    return rows


def _biggest_predicted_upsets(target: pd.DataFrame) -> pd.DataFrame:
    upsets = target[~target["favorite_won"]].copy()
    return upsets.sort_values(
        by=["favorite_probability", "favorite_implied_spread", "favorite_actual_margin"],
        ascending=[False, False, True],
    ).head(5)


def _biggest_actual_upsets(target: pd.DataFrame) -> pd.DataFrame:
    upsets = target[~target["favorite_won"]].copy()
    return upsets.sort_values(
        by=["favorite_actual_margin", "favorite_implied_spread"],
        ascending=[True, False],
    ).head(5)


def _team_game_row(row: pd.Series, *, home: bool) -> dict[str, object]:
    if home:
        team = row["home_team"]
        opponent = row["away_team"]
        pregame_rating = row["home_rating"]
        actual_team_margin = row["actual_margin"]
        predicted_team_margin = row["predicted_margin"]
    else:
        team = row["away_team"]
        opponent = row["home_team"]
        pregame_rating = row["away_rating"]
        actual_team_margin = -row["actual_margin"]
        predicted_team_margin = -row["predicted_margin"]

    favorite_side_correct = row["favorite_won"]
    is_favorite = row["favorite_team"] == team
    won = actual_team_margin > 0
    return {
        "game_number": row["game_number"],
        "game_id": row["game_id"],
        "game_date": row["game_date"],
        "team": team,
        "opponent": opponent,
        "pregame_rating": float(pregame_rating),
        "actual_team_margin": float(actual_team_margin),
        "predicted_team_margin": float(predicted_team_margin),
        "absolute_margin_error": abs(float(predicted_team_margin - actual_team_margin)),
        "favorite_side_correct": bool(favorite_side_correct),
        "is_favorite": bool(is_favorite),
        "won": bool(won),
        "upset_loss_as_favorite": bool(is_favorite and not won),
        "upset_win_as_underdog": bool((not is_favorite) and won),
    }


def _empty_bucket_row(section: str, bucket: str) -> dict[str, object]:
    return {
        "section": section,
        "bucket": bucket,
        "variant": TARGET_VARIANT,
        "games": 0,
        "predicted_win_rate": None,
        "actual_win_rate": None,
        "calibration_error": None,
        "mean_implied_spread": None,
        "mean_actual_favorite_margin": None,
        "mean_spread_error": None,
        "margin_mae": None,
        "notes": "no games",
    }


def _confidence_note(calibration_error: float) -> str:
    if calibration_error <= -0.08:
        return "overconfident"
    if calibration_error >= 0.08:
        return "underconfident"
    return "roughly calibrated"


def _spread_note(spread_error: float) -> str:
    if spread_error >= 1.0:
        return "favorites underperformed implied spread"
    if spread_error <= -1.0:
        return "favorites outperformed implied spread"
    return "spread roughly aligned"


def _calibration_conclusion(probability_rows: pd.DataFrame) -> str:
    overconfident = probability_rows[probability_rows["calibration_error"] <= -0.08]
    underconfident = probability_rows[probability_rows["calibration_error"] >= 0.08]
    parts = []
    if not overconfident.empty:
        parts.append("overconfident in " + ", ".join(overconfident["bucket"].astype(str)))
    if not underconfident.empty:
        parts.append("underconfident in " + ", ".join(underconfident["bucket"].astype(str)))
    if not parts:
        return "- Probability buckets are broadly calibrated, though sample sizes are thin in high-confidence ranges."
    return "- Calibration flags: " + "; ".join(parts) + "."


def _large_favorite_sentence(large_favorites: pd.DataFrame) -> str:
    if large_favorites.empty:
        return "- There were no favorites with implied spreads of 4+ goals."
    return (
        f"- Large favorites, implied spread 4+ goals, went "
        f"{int(large_favorites['favorite_won'].sum())}-{int((~large_favorites['favorite_won']).sum())}; "
        f"mean spread error was {large_favorites['favorite_spread_error'].mean():+.2f} goals."
    )


def _spread_conclusion(spread_rows: pd.DataFrame) -> str:
    valid = spread_rows[spread_rows["games"] > 0]
    inflated = valid[valid["mean_spread_error"] >= 1.0]
    if inflated.empty:
        return "- Favorite inflation is not broad across spread buckets."
    return "- Favorite inflation appears in spread buckets: " + ", ".join(inflated["bucket"]) + "."


def _top_games_sentence(label: str, games: pd.DataFrame) -> str:
    if games.empty:
        return f"- {label}: none."
    parts = []
    for _, row in games.iterrows():
        parts.append(
            f"{row['game_date']} {row['underdog_team']} over {row['favorite_team']} "
            f"({row['favorite_probability']:.0%} favorite, actual favorite margin "
            f"{row['favorite_actual_margin']:+.0f})"
        )
    return f"- {label}: " + "; ".join(parts) + "."


def _team_list(frame: pd.DataFrame, value_column: str) -> str:
    parts = []
    for _, row in frame.iterrows():
        value = row[value_column]
        if pd.isna(value):
            continue
        parts.append(f"{row['team']} ({value:.2f})")
    return "; ".join(parts) + "."


def _readiness_recommendation(
    mean_abs_calibration_error: float,
    favorite_spread_bias: float,
    high_confidence: pd.DataFrame,
) -> str:
    high_confidence_error = (
        abs(float(high_confidence["favorite_won"].mean() - high_confidence["favorite_probability"].mean()))
        if not high_confidence.empty
        else 0.0
    )
    if mean_abs_calibration_error >= 0.12 or high_confidence_error >= 0.15:
        return "- Production readiness: use the model cautiously; calibration needs a probability scaling layer before high-confidence matchup probabilities are surfaced strongly."
    if abs(favorite_spread_bias) >= 1.0:
        return "- Production readiness: probabilities are usable for lab promotion, but implied spreads need calibration before being presented as precise goal margins."
    return "- Production readiness: no major calibration issue was discovered; still present probabilities with uncertainty because high-confidence samples are limited."


if __name__ == "__main__":
    main()
