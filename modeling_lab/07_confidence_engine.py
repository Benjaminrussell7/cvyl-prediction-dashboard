from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUTPUTS, actual_home_win_probability, write_output


FEATURES_INPUT = "leakage_safe_features.csv"
PREDICTIONS_INPUT = "baseline_predictions.csv"
TARGETS_INPUT = "targets.csv"
CONFIDENCE_OUTPUT = "confidence_predictions.csv"
UPSET_RISK_OUTPUT = "upset_risk_analysis.csv"
AGREEMENT_OUTPUT = "prediction_agreement_analysis.csv"


def load_confidence_frame() -> pd.DataFrame:
    features = pd.read_csv(OUTPUTS / FEATURES_INPUT)
    predictions = pd.read_csv(OUTPUTS / PREDICTIONS_INPUT)
    targets = pd.read_csv(OUTPUTS / TARGETS_INPUT)
    frame = features.merge(
        predictions[
            [
                "game_id",
                "actual_winner",
                "elo_home_probability",
                "power_v3_recency_home_probability",
                "power_v3_calibrated_home_probability",
            ]
        ],
        on="game_id",
        how="inner",
        suffixes=("", "_baseline"),
    )
    frame = frame.merge(
        targets[["game_id", "home_score", "away_score", "actual_winner"]],
        on="game_id",
        how="inner",
        suffixes=("", "_target"),
    )
    frame["actual_home_win"] = frame.apply(actual_home_win_probability, axis=1)
    return frame.sort_values(["game_date", "game_id"], ignore_index=True)


def build_confidence_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    confidence = frame.copy()
    confidence["team_stability"] = confidence.apply(team_stability_score, axis=1)
    confidence["model_agreement"] = confidence.apply(model_agreement_score, axis=1)
    confidence["volatility_score"] = confidence["recent_result_volatility"].map(inverse_scaled_score)
    confidence["strength_differential_score"] = confidence["strength_gap_absolute"].map(
        lambda value: scaled_score(value, midpoint=4.0, spread=4.0)
    )
    confidence["matchup_consistency"] = confidence.apply(matchup_consistency_score, axis=1)
    confidence["confidence_score"] = confidence.apply(prediction_confidence_score, axis=1)
    confidence["upset_risk"] = confidence["confidence_score"].map(upset_risk_category)
    confidence["favorite_probability"] = confidence["power_v3_calibrated_home_probability"].map(
        lambda value: max(float(value), 1.0 - float(value))
    )
    confidence["predicted_winner"] = np.where(
        confidence["power_v3_calibrated_home_probability"] >= 0.5,
        confidence["home_team"],
        confidence["away_team"],
    )
    confidence["prediction_correct"] = confidence["predicted_winner"] == confidence["actual_winner_target"]
    confidence["confidence_tier"] = confidence["confidence_score"].map(confidence_tier)

    confidence_output = confidence[
        [
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "predicted_winner",
            "actual_winner_target",
            "favorite_probability",
            "confidence_score",
            "confidence_tier",
            "upset_risk",
            "team_stability",
            "model_agreement",
            "volatility_score",
            "strength_differential_score",
            "matchup_consistency",
            "prediction_correct",
        ]
    ].rename(columns={"actual_winner_target": "actual_winner"})

    upset_output = build_upset_risk_analysis(confidence_output)
    agreement_output = build_prediction_agreement_analysis(confidence)
    calibration_output = build_confidence_calibration(confidence_output)
    agreement_output = pd.concat([agreement_output, calibration_output], ignore_index=True)
    return confidence_output, upset_output, agreement_output


def team_stability_score(row: pd.Series) -> float:
    rating_stability = normalized_0_1(row.get("rating_stability", 0.0))
    scoring_consistency = inverse_scaled_score(abs(row.get("rolling_goals_for_std_last_5", 0.0)))
    defensive_consistency = inverse_scaled_score(abs(row.get("rolling_goals_against_std_last_5", 0.0)))
    upset_frequency = inverse_scaled_score(abs(row.get("upset_rate", 0.0)), midpoint=0.25, spread=0.25)
    close_game_frequency = inverse_scaled_score(abs(row.get("close_game_rate", 0.0)), midpoint=0.35, spread=0.35)
    return weighted_average(
        [
            rating_stability,
            scoring_consistency,
            defensive_consistency,
            upset_frequency,
            close_game_frequency,
        ],
        [0.30, 0.20, 0.20, 0.15, 0.15],
    )


def model_agreement_score(row: pd.Series) -> float:
    probabilities = np.array(
        [
            float(row["elo_home_probability"]),
            float(row["power_v3_recency_home_probability"]),
            float(row["power_v3_calibrated_home_probability"]),
        ]
    )
    directional_votes = probabilities >= 0.5
    agreement_rate = max(directional_votes.mean(), 1.0 - directional_votes.mean())
    spread_penalty = min(float(probabilities.std()) * 4.0, 1.0)
    return float(np.clip((agreement_rate * 0.7) + ((1.0 - spread_penalty) * 0.3), 0.0, 1.0))


def matchup_consistency_score(row: pd.Series) -> float:
    trend_disagreement = abs(float(row.get("offensive_trend", 0.0)) - float(row.get("defensive_trend", 0.0)))
    conflicting_signal_count = count_conflicting_signals(row)
    trend_score = inverse_scaled_score(trend_disagreement, midpoint=3.0, spread=3.0)
    conflict_score = max(0.0, 1.0 - conflicting_signal_count / 4.0)
    volatility_score = inverse_scaled_score(float(row.get("volatility_mismatch", 0.0)), midpoint=4.0, spread=4.0)
    return weighted_average([trend_score, conflict_score, volatility_score], [0.35, 0.35, 0.30])


def count_conflicting_signals(row: pd.Series) -> int:
    signals = [
        float(row.get("elo_diff_home", 0.0)),
        float(row.get("rolling_margin_last_3", 0.0)),
        float(row.get("recent_vs_season_margin_delta", 0.0)),
        float(row.get("offense_vs_opponent_defense_gap", 0.0)),
        float(row.get("defense_vs_opponent_offense_gap", 0.0)),
    ]
    positive = sum(value > 0 for value in signals)
    negative = sum(value < 0 for value in signals)
    return min(positive, negative)


def prediction_confidence_score(row: pd.Series) -> float:
    score = weighted_average(
        [
            row["team_stability"],
            row["model_agreement"],
            row["volatility_score"],
            row["strength_differential_score"],
            row["matchup_consistency"],
        ],
        [0.22, 0.28, 0.18, 0.18, 0.14],
    )
    return round(float(np.clip(score * 100.0, 0.0, 100.0)), 2)


def upset_risk_category(confidence_score: float) -> str:
    if confidence_score >= 75:
        return "low"
    if confidence_score >= 60:
        return "moderate"
    if confidence_score >= 45:
        return "elevated"
    return "high"


def confidence_tier(confidence_score: float) -> str:
    if confidence_score >= 75:
        return "high"
    if confidence_score >= 60:
        return "medium"
    if confidence_score >= 45:
        return "low"
    return "very_low"


def build_upset_risk_analysis(confidence: pd.DataFrame) -> pd.DataFrame:
    grouped = confidence.groupby("upset_risk", as_index=False).agg(
        games=("game_id", "count"),
        average_confidence=("confidence_score", "mean"),
        average_favorite_probability=("favorite_probability", "mean"),
        prediction_accuracy=("prediction_correct", "mean"),
    )
    grouped["upset_rate"] = 1.0 - grouped["prediction_accuracy"]
    return grouped.sort_values("average_confidence", ascending=False, ignore_index=True)


def build_prediction_agreement_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["elo_pick_home"] = output["elo_home_probability"] >= 0.5
    output["power_v3_pick_home"] = output["power_v3_recency_home_probability"] >= 0.5
    output["calibrated_pick_home"] = output["power_v3_calibrated_home_probability"] >= 0.5
    output["agreement_count"] = output[
        ["elo_pick_home", "power_v3_pick_home", "calibrated_pick_home"]
    ].sum(axis=1)
    output["agreement_level"] = output["agreement_count"].map(
        lambda count: "full_agreement" if count in {0, 3} else "split_signal"
    )
    return output.groupby("agreement_level", as_index=False).agg(
        games=("game_id", "count"),
        average_confidence=("confidence_score", "mean"),
        prediction_accuracy=("prediction_correct", "mean"),
        average_model_agreement=("model_agreement", "mean"),
    )


def build_confidence_calibration(confidence: pd.DataFrame) -> pd.DataFrame:
    grouped = confidence.groupby("confidence_tier", as_index=False).agg(
        games=("game_id", "count"),
        average_confidence=("confidence_score", "mean"),
        average_favorite_probability=("favorite_probability", "mean"),
        prediction_accuracy=("prediction_correct", "mean"),
    )
    grouped["agreement_level"] = "confidence_tier:" + grouped["confidence_tier"]
    grouped["average_model_agreement"] = np.nan
    return grouped[
        [
            "agreement_level",
            "games",
            "average_confidence",
            "prediction_accuracy",
            "average_model_agreement",
            "average_favorite_probability",
        ]
    ]


def weighted_average(values: list[float], weights: list[float]) -> float:
    return float(np.average(np.clip(values, 0.0, 1.0), weights=weights))


def normalized_0_1(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def scaled_score(value: float, *, midpoint: float = 1.0, spread: float = 1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-(float(value) - midpoint) / max(spread, 1e-6))))


def inverse_scaled_score(value: float, *, midpoint: float = 2.0, spread: float = 2.0) -> float:
    return 1.0 - scaled_score(value, midpoint=midpoint, spread=spread)


def main() -> None:
    frame = load_confidence_frame()
    confidence, upset, agreement = build_confidence_outputs(frame)
    confidence_path = write_output(confidence, CONFIDENCE_OUTPUT)
    upset_path = write_output(upset, UPSET_RISK_OUTPUT)
    agreement_path = write_output(agreement, AGREEMENT_OUTPUT)
    print(f"Wrote {confidence_path}")
    print(f"Wrote {upset_path}")
    print(f"Wrote {agreement_path}")


if __name__ == "__main__":
    main()
