from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "modeling_lab" / "outputs"
PREDICTIONS_CSV = OUTPUT_DIR / "opponent_adjusted_predictions.csv"

TARGET_VARIANT = "opponent_adjusted_performance"
BASELINE_VARIANT = "baseline_power_v3"
CURRENT_SCALE = 4.0
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


@dataclass(frozen=True)
class CalibrationVariant:
    name: str
    label: str
    probability_scale: float
    margin_scale: float
    prequential_bucket_calibration: bool = False


VARIANTS = [
    CalibrationVariant("current_conversion", "Current conversion", 4.0, 1.0),
    CalibrationVariant("logistic_scale_3", "Logistic scale 3.0", 3.0, 1.0),
    CalibrationVariant("logistic_scale_2_5", "Logistic scale 2.5", 2.5, 1.0),
    CalibrationVariant("logistic_scale_2", "Logistic scale 2.0", 2.0, 1.0),
    CalibrationVariant("margin_scale_1_25", "Margin scale 1.25", 4.0, 1.25),
    CalibrationVariant("margin_scale_1_5", "Margin scale 1.50", 4.0, 1.50),
    CalibrationVariant("combined_2_5_1_25", "Logistic 2.5 + margin 1.25", 2.5, 1.25),
    CalibrationVariant("combined_2_5_1_5", "Logistic 2.5 + margin 1.50", 2.5, 1.50),
    CalibrationVariant("combined_2_0_1_5", "Logistic 2.0 + margin 1.50", 2.0, 1.50),
    CalibrationVariant(
        "prequential_bucket_calibration",
        "Prequential bucket calibration",
        4.0,
        1.0,
        prequential_bucket_calibration=True,
    ),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(PREDICTIONS_CSV)
    target = source[source["variant"] == TARGET_VARIANT].copy()

    predictions = build_calibrated_predictions(target)
    summary = summarize_variants(predictions)
    buckets = summarize_probability_buckets(predictions)
    interpretation = build_interpretation(summary, buckets)

    predictions.to_csv(OUTPUT_DIR / "calibrated_matchup_predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "calibrated_matchup_summary.csv", index=False)
    buckets.to_csv(OUTPUT_DIR / "calibrated_matchup_probability_buckets.csv", index=False)
    (OUTPUT_DIR / "calibrated_matchup_interpretation.md").write_text(
        interpretation,
        encoding="utf-8",
    )

    print(f"Wrote calibrated matchup outputs to {OUTPUT_DIR}")


def build_calibrated_predictions(target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = target.sort_values(["game_number", "game_id"], ignore_index=True)
    bucket_history = _new_bucket_history()

    for _, game in ordered.iterrows():
        rating_diff = float(game["rating_diff"])
        actual_home_result = float(game["actual_home_result"])
        for variant in VARIANTS:
            home_probability = _home_probability(rating_diff, variant)
            if variant.prequential_bucket_calibration:
                home_probability = _prequential_bucket_probability(
                    game,
                    home_probability,
                    bucket_history,
                )
            predicted_margin = rating_diff * variant.margin_scale
            predicted_winner = (
                str(game["home_team"])
                if home_probability >= 0.5
                else str(game["away_team"])
            )
            actual_winner = str(game["actual_winner"])
            favorite_probability = max(home_probability, 1.0 - home_probability)
            favorite_team = (
                str(game["home_team"])
                if home_probability >= 0.5
                else str(game["away_team"])
            )
            favorite_actual_margin = (
                float(game["actual_margin"])
                if favorite_team == str(game["home_team"])
                else -float(game["actual_margin"])
            )
            rows.append(
                {
                    "game_number": int(game["game_number"]),
                    "game_id": game["game_id"],
                    "game_date": game["game_date"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "probability_scale": variant.probability_scale,
                    "margin_scale": variant.margin_scale,
                    "rating_diff": rating_diff,
                    "home_win_probability": home_probability,
                    "favorite_probability": favorite_probability,
                    "favorite_team": favorite_team,
                    "predicted_winner": predicted_winner,
                    "actual_winner": actual_winner,
                    "prediction_correct": predicted_winner == actual_winner,
                    "actual_home_result": actual_home_result,
                    "home_score": int(game["home_score"]),
                    "away_score": int(game["away_score"]),
                    "predicted_margin": predicted_margin,
                    "actual_margin": float(game["actual_margin"]),
                    "favorite_actual_margin": favorite_actual_margin,
                    "favorite_won": favorite_actual_margin > 0,
                    "implied_spread": abs(predicted_margin),
                    "spread_error": abs(predicted_margin) - favorite_actual_margin,
                    "margin_error": predicted_margin - float(game["actual_margin"]),
                    "absolute_margin_error": abs(predicted_margin - float(game["actual_margin"])),
                }
            )

        _update_bucket_history(game, bucket_history)

    return pd.DataFrame(rows)


def summarize_variants(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    current = predictions[predictions["variant"] == "current_conversion"]
    current_metrics = _metric_values(current)
    current_ece = _expected_calibration_error(current)

    for variant, group in predictions.groupby("variant", sort=False):
        metrics = _metric_values(group)
        ece = _expected_calibration_error(group)
        rows.append(
            {
                "variant": variant,
                "variant_label": group["variant_label"].iloc[0],
                "winner_accuracy": metrics["winner_accuracy"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "margin_mae": metrics["margin_mae"],
                "favorite_accuracy": metrics["favorite_accuracy"],
                "mean_favorite_probability": metrics["mean_favorite_probability"],
                "mean_implied_spread": metrics["mean_implied_spread"],
                "mean_actual_favorite_margin": metrics["mean_actual_favorite_margin"],
                "mean_spread_error": metrics["mean_spread_error"],
                "large_favorite_games": metrics["large_favorite_games"],
                "large_favorite_accuracy": metrics["large_favorite_accuracy"],
                "large_favorite_mean_spread_error": metrics["large_favorite_mean_spread_error"],
                "expected_calibration_error": ece,
                "games": int(len(group)),
                "delta_winner_accuracy": metrics["winner_accuracy"]
                - current_metrics["winner_accuracy"],
                "delta_brier_score": metrics["brier_score"] - current_metrics["brier_score"],
                "delta_log_loss": metrics["log_loss"] - current_metrics["log_loss"],
                "delta_margin_mae": metrics["margin_mae"] - current_metrics["margin_mae"],
                "delta_ece": ece - current_ece,
            }
        )
    summary = pd.DataFrame(rows)
    summary["promotion_candidate"] = summary.apply(_promotion_candidate, axis=1)
    return summary.sort_values(
        by=["brier_score", "log_loss", "expected_calibration_error", "margin_mae"],
        ascending=[True, True, True, True],
        ignore_index=True,
    )


def summarize_probability_buckets(predictions: pd.DataFrame) -> pd.DataFrame:
    bucketed = predictions.copy()
    bucketed["probability_bucket"] = pd.cut(
        bucketed["favorite_probability"],
        bins=PROBABILITY_BUCKETS,
        labels=PROBABILITY_LABELS,
        include_lowest=True,
        right=False,
    )
    rows = []
    for variant, variant_group in bucketed.groupby("variant", sort=False):
        for bucket, group in variant_group.groupby("probability_bucket", observed=False):
            if group.empty:
                rows.append(
                    {
                        "variant": variant,
                        "variant_label": variant_group["variant_label"].iloc[0],
                        "bucket": str(bucket),
                        "games": 0,
                        "predicted_win_rate": None,
                        "actual_win_rate": None,
                        "calibration_error": None,
                        "mean_implied_spread": None,
                        "mean_actual_favorite_margin": None,
                        "mean_spread_error": None,
                    }
                )
                continue
            predicted = float(group["favorite_probability"].mean())
            actual = float(group["favorite_won"].mean())
            rows.append(
                {
                    "variant": variant,
                    "variant_label": group["variant_label"].iloc[0],
                    "bucket": str(bucket),
                    "games": int(len(group)),
                    "predicted_win_rate": predicted,
                    "actual_win_rate": actual,
                    "calibration_error": actual - predicted,
                    "mean_implied_spread": float(group["implied_spread"].mean()),
                    "mean_actual_favorite_margin": float(
                        group["favorite_actual_margin"].mean()
                    ),
                    "mean_spread_error": float(group["spread_error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_interpretation(summary: pd.DataFrame, buckets: pd.DataFrame) -> str:
    current = summary[summary["variant"] == "current_conversion"].iloc[0]
    best = summary.iloc[0]
    promoted = summary[summary["promotion_candidate"]]
    best_promoted = promoted.iloc[0] if not promoted.empty else None
    best_bucket_rows = buckets[buckets["variant"] == best["variant"]]
    current_bucket_rows = buckets[buckets["variant"] == "current_conversion"]

    if best_promoted is None:
        recommendation = "do not promote a calibrated variant under the stated criteria."
    else:
        recommendation = (
            f"promote {best_promoted['variant']} as the leading Power v5 calibration candidate."
        )

    lines = [
        "# Probability And Spread Calibration Lab",
        "",
        "Scope: lab-only post-processing experiment over saved opponent_adjusted_performance rolling predictions. Baseline Power v3, the opponent-adjusted rating engine, and the current conversion logic were preserved; no tournament data, UI, or production prediction code was used.",
        "",
        "Approach: variants adjust only the translation layer from rating differential to probability and implied spread. Logistic-scale variants sharpen probabilities by reducing the logistic denominator. Margin-scale variants multiply the implied spread while leaving winner side unchanged. Combined variants do both. The prequential bucket variant recalibrates probabilities from earlier games only, with shrinkage toward the current probability.",
        "",
        "## Best Variant",
        "",
        (
            f"- Best by Brier/log-loss/ECE sort: {best['variant']} "
            f"(Brier {best['brier_score']:.4f}, log_loss {best['log_loss']:.4f}, "
            f"ECE {best['expected_calibration_error']:.4f}, margin_mae {best['margin_mae']:.4f}, "
            f"accuracy {best['winner_accuracy']:.4f})."
        ),
        (
            f"- Current conversion reference: Brier {current['brier_score']:.4f}, "
            f"log_loss {current['log_loss']:.4f}, ECE {current['expected_calibration_error']:.4f}, "
            f"margin_mae {current['margin_mae']:.4f}, accuracy {current['winner_accuracy']:.4f}."
        ),
        "",
        "## Summary",
        "",
        _markdown_table(
            summary[
                [
                    "variant",
                    "winner_accuracy",
                    "brier_score",
                    "log_loss",
                    "margin_mae",
                    "mean_favorite_probability",
                    "favorite_accuracy",
                    "mean_implied_spread",
                    "mean_actual_favorite_margin",
                    "mean_spread_error",
                    "expected_calibration_error",
                    "delta_brier_score",
                    "delta_log_loss",
                    "delta_margin_mae",
                    "delta_ece",
                    "promotion_candidate",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        _probability_sentence(current, best),
        _spread_sentence(current, best),
        _large_favorite_sentence(current, best),
        _bucket_sentence("Current conversion", current_bucket_rows),
        _bucket_sentence(f"Best variant ({best['variant']})", best_bucket_rows),
        f"- Recommendation: {recommendation}",
        "",
        "## Output Files",
        "",
        "- `calibrated_matchup_predictions.csv`",
        "- `calibrated_matchup_summary.csv`",
        "- `calibrated_matchup_probability_buckets.csv`",
        "- `calibrated_matchup_interpretation.md`",
    ]
    return "\n".join(lines) + "\n"


def _home_probability(rating_diff: float, variant: CalibrationVariant) -> float:
    return _logistic(rating_diff / variant.probability_scale)


def _prequential_bucket_probability(
    game: pd.Series,
    base_home_probability: float,
    bucket_history: dict[str, dict[str, float]],
) -> float:
    favorite_probability = max(base_home_probability, 1.0 - base_home_probability)
    bucket = _bucket_label(favorite_probability)
    history = bucket_history[bucket]
    if history["games"] < 12:
        calibrated_favorite_probability = favorite_probability
    else:
        empirical = history["wins"] / history["games"]
        shrinkage = min(0.65, history["games"] / (history["games"] + 30.0))
        calibrated_favorite_probability = (
            (1.0 - shrinkage) * favorite_probability
        ) + (shrinkage * empirical)
    calibrated_favorite_probability = min(max(calibrated_favorite_probability, 0.500001), 0.95)
    if base_home_probability >= 0.5:
        return calibrated_favorite_probability
    return 1.0 - calibrated_favorite_probability


def _update_bucket_history(game: pd.Series, bucket_history: dict[str, dict[str, float]]) -> None:
    favorite_probability = max(float(game["home_win_probability"]), 1.0 - float(game["home_win_probability"]))
    bucket = _bucket_label(favorite_probability)
    favorite_won = (
        float(game["actual_home_result"])
        if float(game["home_win_probability"]) >= 0.5
        else 1.0 - float(game["actual_home_result"])
    )
    bucket_history[bucket]["games"] += 1
    bucket_history[bucket]["wins"] += favorite_won


def _new_bucket_history() -> dict[str, dict[str, float]]:
    return {label: {"games": 0.0, "wins": 0.0} for label in PROBABILITY_LABELS}


def _bucket_label(favorite_probability: float) -> str:
    for lower, upper, label in zip(
        PROBABILITY_BUCKETS[:-1],
        PROBABILITY_BUCKETS[1:],
        PROBABILITY_LABELS,
        strict=False,
    ):
        if lower <= favorite_probability < upper:
            return label
    return PROBABILITY_LABELS[-1]


def _metric_values(group: pd.DataFrame) -> dict[str, float]:
    probabilities = group["home_win_probability"].clip(1e-6, 1 - 1e-6)
    results = group["actual_home_result"]
    large_favorites = group[group["favorite_probability"] >= 0.75]
    return {
        "winner_accuracy": float(group["prediction_correct"].mean()),
        "brier_score": float(((probabilities - results) ** 2).mean()),
        "log_loss": float(
            -(
                results * probabilities.map(math.log)
                + (1 - results) * (1 - probabilities).map(math.log)
            ).mean()
        ),
        "margin_mae": float(group["absolute_margin_error"].mean()),
        "favorite_accuracy": float(group["favorite_won"].mean()),
        "mean_favorite_probability": float(group["favorite_probability"].mean()),
        "mean_implied_spread": float(group["implied_spread"].mean()),
        "mean_actual_favorite_margin": float(group["favorite_actual_margin"].mean()),
        "mean_spread_error": float(group["spread_error"].mean()),
        "large_favorite_games": int(len(large_favorites)),
        "large_favorite_accuracy": (
            float(large_favorites["favorite_won"].mean()) if not large_favorites.empty else 0.0
        ),
        "large_favorite_mean_spread_error": (
            float(large_favorites["spread_error"].mean()) if not large_favorites.empty else 0.0
        ),
    }


def _expected_calibration_error(group: pd.DataFrame) -> float:
    bucketed = group.copy()
    bucketed["probability_bucket"] = pd.cut(
        bucketed["favorite_probability"],
        bins=PROBABILITY_BUCKETS,
        labels=PROBABILITY_LABELS,
        include_lowest=True,
        right=False,
    )
    total = len(bucketed)
    error = 0.0
    for _, bucket_group in bucketed.groupby("probability_bucket", observed=False):
        if bucket_group.empty:
            continue
        predicted = float(bucket_group["favorite_probability"].mean())
        actual = float(bucket_group["favorite_won"].mean())
        error += (len(bucket_group) / total) * abs(actual - predicted)
    return float(error)


def _promotion_candidate(row: pd.Series) -> bool:
    if row["variant"] == "current_conversion":
        return False
    return bool(
        row["delta_brier_score"] < 0
        and row["delta_log_loss"] < 0
        and row["delta_ece"] < 0
        and row["delta_winner_accuracy"] >= -0.01
        and row["delta_margin_mae"] <= 0.05
    )


def _probability_sentence(current: pd.Series, best: pd.Series) -> str:
    direction = "less compressed" if best["mean_favorite_probability"] > current["mean_favorite_probability"] else "not less compressed"
    return (
        f"- Probabilities became {direction}: mean favorite probability changed from "
        f"{current['mean_favorite_probability']:.3f} to {best['mean_favorite_probability']:.3f}, "
        f"while favorite accuracy was {best['favorite_accuracy']:.3f}."
    )


def _spread_sentence(current: pd.Series, best: pd.Series) -> str:
    return (
        f"- Spreads changed from {current['mean_implied_spread']:.2f} implied goals to "
        f"{best['mean_implied_spread']:.2f}; actual favorite margin was "
        f"{best['mean_actual_favorite_margin']:.2f}. Mean spread error moved from "
        f"{current['mean_spread_error']:+.2f} to {best['mean_spread_error']:+.2f}."
    )


def _large_favorite_sentence(current: pd.Series, best: pd.Series) -> str:
    return (
        f"- Large favorites under the best variant: {int(best['large_favorite_games'])} games, "
        f"accuracy {best['large_favorite_accuracy']:.3f}, mean spread error "
        f"{best['large_favorite_mean_spread_error']:+.2f}; current had "
        f"{int(current['large_favorite_games'])} large-favorite games."
    )


def _bucket_sentence(label: str, rows: pd.DataFrame) -> str:
    valid = rows[rows["games"] > 0].copy()
    if valid.empty:
        return f"- {label}: no populated probability buckets."
    worst = valid.iloc[valid["calibration_error"].abs().argmax()]
    return (
        f"- {label}: worst bucket was {worst['bucket']} with predicted "
        f"{worst['predicted_win_rate']:.3f}, actual {worst['actual_win_rate']:.3f}, "
        f"error {worst['calibration_error']:+.3f} over {int(worst['games'])} games."
    )


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


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


if __name__ == "__main__":
    main()
