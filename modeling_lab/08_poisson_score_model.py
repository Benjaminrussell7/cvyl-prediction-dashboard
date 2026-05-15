from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import OUTPUTS, write_output


FEATURES_INPUT = "leakage_safe_features.csv"
TARGETS_INPUT = "targets.csv"
RAW_PREDICTIONS_OUTPUT = "poisson_predictions_raw.csv"
CALIBRATED_PREDICTIONS_OUTPUT = "poisson_predictions_calibrated.csv"
LEGACY_PREDICTIONS_OUTPUT = "poisson_predictions.csv"
LEGACY_SIMULATION_OUTPUT = "poisson_simulation_summary.csv"
CALIBRATED_SIMULATION_OUTPUT = "poisson_simulation_summary_calibrated.csv"
CALIBRATION_COMPARISON_OUTPUT = "poisson_calibration_comparison.csv"
DISTRIBUTION_OUTPUT = "score_distribution_analysis.csv"

RANDOM_SEED = 2026
MIN_TRAINING_GAMES = 30
SIMULATIONS_PER_MATCHUP = 10_000
POISSON_STEPS = 650
POISSON_LEARNING_RATE = 0.025
POISSON_L2 = 0.02
GOAL_SHRINKAGE_MULTIPLIER = 0.72
TOTAL_BUCKET_ADJUSTMENT_MULTIPLIER = 0.50
MARGIN_BUCKET_ADJUSTMENT_MULTIPLIER = 0.40


@dataclass(frozen=True)
class PreparedWindow:
    feature_columns: list[str]
    means: pd.Series
    stds: pd.Series


def load_score_modeling_frame() -> pd.DataFrame:
    features = pd.read_csv(OUTPUTS / FEATURES_INPUT)
    targets = pd.read_csv(OUTPUTS / TARGETS_INPUT)
    frame = features.merge(targets, on=["game_id", "game_date", "home_team", "away_team"], how="inner")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame.sort_values(["game_date", "game_id"], ignore_index=True)


def score_feature_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "pregame_home_elo",
        "pregame_away_elo",
        "elo_diff_home",
        "rolling_margin_last_3",
        "rolling_margin_last_5",
        "offensive_trend",
        "defensive_trend",
        "offense_vs_opponent_defense_gap",
        "defense_vs_opponent_offense_gap",
        "projected_pace_interaction",
        "offense_defense_ratio",
        "offensive_sos",
        "defensive_sos",
        "recent_opponent_avg_power",
        "recent_opponent_avg_elo",
        "rolling_goals_for_std_last_5",
        "rolling_goals_against_std_last_5",
        "rolling_margin_std_last_5",
        "recent_result_volatility",
        "volatility_mismatch",
        "strength_gap_absolute",
        "games_played",
        "opponent_games_played",
        "rating_stability",
    ]
    return [column for column in preferred if column in frame.columns]


def prepare_window(train: pd.DataFrame, feature_columns: list[str]) -> PreparedWindow:
    means = train[feature_columns].astype(float).mean()
    stds = train[feature_columns].astype(float).std().replace(0.0, 1.0).fillna(1.0)
    return PreparedWindow(feature_columns=feature_columns, means=means, stds=stds)


def matrix(frame: pd.DataFrame, window: PreparedWindow) -> np.ndarray:
    values = frame[window.feature_columns].astype(float).fillna(window.means)
    standardized = (values - window.means) / window.stds
    return np.column_stack([np.ones(len(standardized)), standardized.to_numpy(dtype=float)])


def fit_poisson_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights = np.zeros(x.shape[1], dtype=float)
    weights[0] = np.log(max(float(y.mean()), 0.25))
    for _ in range(POISSON_STEPS):
        linear = np.clip(x @ weights, -5.0, 4.0)
        expected = np.exp(linear)
        gradient = (x.T @ (expected - y)) / len(y)
        gradient[1:] += POISSON_L2 * weights[1:] / len(y)
        weights -= POISSON_LEARNING_RATE * gradient
    return weights


def predict_expected_goals(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x @ weights, -5.0, 4.0))


def expanding_poisson_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = score_feature_columns(frame)
    rows = []
    for row_index in range(MIN_TRAINING_GAMES, len(frame)):
        train = frame.iloc[:row_index].copy()
        test = frame.iloc[[row_index]].copy()
        window = prepare_window(train, feature_columns)
        x_train = matrix(train, window)
        x_test = matrix(test, window)
        home_weights = fit_poisson_regression(x_train, train["home_score"].astype(float).to_numpy())
        away_weights = fit_poisson_regression(x_train, train["away_score"].astype(float).to_numpy())
        expected_home = float(predict_expected_goals(x_test, home_weights)[0])
        expected_away = float(predict_expected_goals(x_test, away_weights)[0])
        game = test.iloc[0]
        rows.append(
            {
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "training_games": row_index,
                "expected_home_goals": expected_home,
                "expected_away_goals": expected_away,
                "expected_total_goals": expected_home + expected_away,
                "expected_margin": expected_home - expected_away,
            }
        )
    return pd.DataFrame(rows)


def soft_clip(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return float(max(value, 0.05))
    midpoint = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    return float(midpoint + half_width * np.tanh((value - midpoint) / half_width))


def total_bucket(value: float) -> str:
    if value < 6:
        return "0-6"
    if value < 9:
        return "6-9"
    if value < 12:
        return "9-12"
    if value < 15:
        return "12-15"
    return "15+"


def margin_bucket(value: float) -> str:
    absolute = abs(float(value))
    if absolute < 2:
        return "0-2"
    if absolute < 5:
        return "2-5"
    if absolute < 8:
        return "5-8"
    return "8+"


def bucket_bias(prior: pd.DataFrame, bucket_column: str, error_column: str, bucket: str) -> float:
    if prior.empty:
        return 0.0
    bucket_rows = prior.loc[prior[bucket_column] == bucket]
    if len(bucket_rows) < 4:
        bucket_rows = prior
    return float(bucket_rows[error_column].mean()) if len(bucket_rows) else 0.0


def side_goal_bounds(history: pd.DataFrame, side: str) -> tuple[float, float]:
    scores = history[f"{side}_score"].astype(float)
    average = float(scores.mean())
    lower = max(0.75, average * 0.35)
    upper = min(14.0, max(average * 1.85, float(scores.quantile(0.90)) * 1.10))
    return lower, max(upper, lower + 2.0)


def calibrate_poisson_predictions(raw_predictions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior_calibrated: list[dict[str, object]] = []
    sorted_raw = raw_predictions.sort_values(["game_date", "game_id"], ignore_index=True)
    for _, row in sorted_raw.iterrows():
        training_games = int(row["training_games"])
        history = frame.iloc[:training_games]
        home_average = float(history["home_score"].astype(float).mean())
        away_average = float(history["away_score"].astype(float).mean())

        expected_home = home_average + GOAL_SHRINKAGE_MULTIPLIER * (
            float(row["expected_home_goals"]) - home_average
        )
        expected_away = away_average + GOAL_SHRINKAGE_MULTIPLIER * (
            float(row["expected_away_goals"]) - away_average
        )

        home_lower, home_upper = side_goal_bounds(history, "home")
        away_lower, away_upper = side_goal_bounds(history, "away")
        expected_home = soft_clip(expected_home, home_lower, home_upper)
        expected_away = soft_clip(expected_away, away_lower, away_upper)

        preliminary_total = expected_home + expected_away
        preliminary_margin = expected_home - expected_away
        prior = pd.DataFrame(prior_calibrated)
        total_bias = bucket_bias(
            prior,
            "expected_total_bucket",
            "total_goals_error",
            total_bucket(preliminary_total),
        )
        margin_bias = bucket_bias(
            prior,
            "predicted_margin_bucket",
            "margin_error",
            margin_bucket(preliminary_margin),
        )

        adjusted_total = max(
            1.5,
            preliminary_total - TOTAL_BUCKET_ADJUSTMENT_MULTIPLIER * total_bias,
        )
        adjusted_margin = preliminary_margin - MARGIN_BUCKET_ADJUSTMENT_MULTIPLIER * margin_bias
        expected_home = max(0.25, (adjusted_total + adjusted_margin) / 2.0)
        expected_away = max(0.25, (adjusted_total - adjusted_margin) / 2.0)
        expected_home = soft_clip(expected_home, home_lower, home_upper)
        expected_away = soft_clip(expected_away, away_lower, away_upper)

        calibrated = row.to_dict()
        calibrated["raw_expected_home_goals"] = row["expected_home_goals"]
        calibrated["raw_expected_away_goals"] = row["expected_away_goals"]
        calibrated["raw_expected_total_goals"] = row["expected_total_goals"]
        calibrated["raw_expected_margin"] = row["expected_margin"]
        calibrated["expected_home_goals"] = expected_home
        calibrated["expected_away_goals"] = expected_away
        calibrated["expected_total_goals"] = expected_home + expected_away
        calibrated["expected_margin"] = expected_home - expected_away
        calibrated["expected_total_bucket"] = total_bucket(calibrated["expected_total_goals"])
        calibrated["predicted_margin_bucket"] = margin_bucket(calibrated["expected_margin"])
        evaluated = add_evaluation_columns(pd.DataFrame([calibrated])).iloc[0].to_dict()
        rows.append(evaluated)
        prior_calibrated.append(evaluated)
    return pd.DataFrame(rows)


def simulate_matchups(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    distribution_rows = []
    for _, row in predictions.iterrows():
        home = rng.poisson(max(float(row["expected_home_goals"]), 0.05), SIMULATIONS_PER_MATCHUP)
        away = rng.poisson(max(float(row["expected_away_goals"]), 0.05), SIMULATIONS_PER_MATCHUP)
        margins = home - away
        totals = home + away
        home_win_probability = float((home > away).mean() + 0.5 * (home == away).mean())
        favorite_home = row["expected_margin"] >= 0
        underdog_wins = (away > home) if favorite_home else (home > away)
        score_pairs, counts = np.unique(np.column_stack([home, away]), axis=0, return_counts=True)
        most_likely = score_pairs[int(np.argmax(counts))]
        summary_rows.append(
            {
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "expected_home_goals": row["expected_home_goals"],
                "expected_away_goals": row["expected_away_goals"],
                "expected_total_goals": row["expected_total_goals"],
                "expected_margin": row["expected_margin"],
                "home_win_probability": home_win_probability,
                "away_win_probability": 1.0 - home_win_probability,
                "predicted_winner": row["home_team"] if home_win_probability >= 0.5 else row["away_team"],
                "most_likely_score": f"{int(most_likely[0])}-{int(most_likely[1])}",
                "upset_probability": float(underdog_wins.mean()),
                "close_game_probability": float((np.abs(margins) <= 2).mean()),
                "simulations": SIMULATIONS_PER_MATCHUP,
            }
        )
        distribution_rows.append(
            {
                "game_id": row["game_id"],
                "home_goals_p10": percentile(home, 10),
                "home_goals_p50": percentile(home, 50),
                "home_goals_p90": percentile(home, 90),
                "away_goals_p10": percentile(away, 10),
                "away_goals_p50": percentile(away, 50),
                "away_goals_p90": percentile(away, 90),
                "total_goals_p10": percentile(totals, 10),
                "total_goals_p50": percentile(totals, 50),
                "total_goals_p90": percentile(totals, 90),
                "margin_p10": percentile(margins, 10),
                "margin_p50": percentile(margins, 50),
                "margin_p90": percentile(margins, 90),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(distribution_rows)


def percentile(values: np.ndarray, q: int) -> float:
    return float(np.percentile(values, q))


def add_evaluation_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    output["home_score_error"] = output["expected_home_goals"] - output["home_score"]
    output["away_score_error"] = output["expected_away_goals"] - output["away_score"]
    output["score_mae"] = (
        output["home_score_error"].abs() + output["away_score_error"].abs()
    ) / 2.0
    output["actual_total_goals"] = output["home_score"] + output["away_score"]
    output["actual_margin"] = output["home_score"] - output["away_score"]
    output["total_goals_error"] = output["expected_total_goals"] - output["actual_total_goals"]
    output["margin_error"] = output["expected_margin"] - output["actual_margin"]
    output["margin_mae"] = output["margin_error"].abs()
    output["total_goals_mae"] = output["total_goals_error"].abs()
    output["actual_winner"] = np.where(
        output["home_score"] >= output["away_score"],
        output["home_team"],
        output["away_team"],
    )
    return output


def add_winner_evaluation(predictions: pd.DataFrame, simulation: pd.DataFrame) -> pd.DataFrame:
    probability = simulation[["game_id", "home_win_probability", "predicted_winner"]]
    output = predictions.merge(probability, on="game_id", how="left")
    output["actual_home_win"] = (output["home_score"] > output["away_score"]).astype(float)
    output["winner_correct"] = output["predicted_winner"] == output["actual_winner"]
    output["brier_score"] = (output["home_win_probability"] - output["actual_home_win"]) ** 2
    return output


def summary_metrics(model: str, predictions: pd.DataFrame) -> dict[str, object]:
    return {
        "model": model,
        "calibration_type": "overall",
        "bucket": "all",
        "games": len(predictions),
        "score_mae": float(predictions["score_mae"].mean()),
        "margin_mae": float(predictions["margin_mae"].mean()),
        "total_goals_mae": float(predictions["total_goals_mae"].mean()),
        "winner_accuracy": float(predictions["winner_correct"].mean()),
        "brier_score": float(predictions["brier_score"].mean()),
        "average_expected_total": float(predictions["expected_total_goals"].mean()),
        "average_actual_total": float(predictions["actual_total_goals"].mean()),
        "average_expected_margin": float(predictions["expected_margin"].mean()),
        "average_actual_margin": float(predictions["actual_margin"].mean()),
        "calibration_gap": float(
            predictions["expected_total_goals"].mean() - predictions["actual_total_goals"].mean()
        ),
    }


def bucket_metric_rows(model: str, predictions: pd.DataFrame) -> list[dict[str, object]]:
    rows = [summary_metrics(model, predictions)]
    total_groups = predictions.assign(
        calibration_bucket=predictions["expected_total_goals"].map(total_bucket)
    ).groupby("calibration_bucket", observed=True)
    for bucket, group in total_groups:
        rows.append(
            {
                "model": model,
                "calibration_type": "expected_total_bucket",
                "bucket": str(bucket),
                "games": len(group),
                "score_mae": float(group["score_mae"].mean()),
                "margin_mae": float(group["margin_mae"].mean()),
                "total_goals_mae": float(group["total_goals_mae"].mean()),
                "winner_accuracy": float(group["winner_correct"].mean()),
                "brier_score": float(group["brier_score"].mean()),
                "average_expected_total": float(group["expected_total_goals"].mean()),
                "average_actual_total": float(group["actual_total_goals"].mean()),
                "average_expected_margin": float(group["expected_margin"].mean()),
                "average_actual_margin": float(group["actual_margin"].mean()),
                "calibration_gap": float(
                    group["expected_total_goals"].mean() - group["actual_total_goals"].mean()
                ),
            }
        )
    margin_groups = predictions.assign(
        calibration_bucket=predictions["expected_margin"].map(margin_bucket)
    ).groupby("calibration_bucket", observed=True)
    for bucket, group in margin_groups:
        rows.append(
            {
                "model": model,
                "calibration_type": "predicted_margin_bucket",
                "bucket": str(bucket),
                "games": len(group),
                "score_mae": float(group["score_mae"].mean()),
                "margin_mae": float(group["margin_mae"].mean()),
                "total_goals_mae": float(group["total_goals_mae"].mean()),
                "winner_accuracy": float(group["winner_correct"].mean()),
                "brier_score": float(group["brier_score"].mean()),
                "average_expected_total": float(group["expected_total_goals"].mean()),
                "average_actual_total": float(group["actual_total_goals"].mean()),
                "average_expected_margin": float(group["expected_margin"].mean()),
                "average_actual_margin": float(group["actual_margin"].mean()),
                "calibration_gap": float(
                    group["expected_margin"].mean() - group["actual_margin"].mean()
                ),
            }
        )
    return rows


def calibration_comparison(
    raw_predictions: pd.DataFrame,
    raw_simulation: pd.DataFrame,
    calibrated_predictions: pd.DataFrame,
    calibrated_simulation: pd.DataFrame,
) -> pd.DataFrame:
    raw_evaluated = add_winner_evaluation(raw_predictions, raw_simulation)
    calibrated_evaluated = add_winner_evaluation(calibrated_predictions, calibrated_simulation)
    rows = bucket_metric_rows("raw_poisson", raw_evaluated)
    rows.extend(bucket_metric_rows("calibrated_poisson", calibrated_evaluated))
    return pd.DataFrame(rows)


def score_calibration_rows(predictions: pd.DataFrame, model: str = "calibrated_poisson") -> list[dict[str, object]]:
    rows = []
    buckets = pd.cut(
        predictions["expected_total_goals"],
        bins=[0, 6, 9, 12, 15, 100],
        labels=["0-6", "6-9", "9-12", "12-15", "15+"],
        include_lowest=True,
    )
    grouped = predictions.assign(total_bucket=buckets).groupby("total_bucket", observed=True)
    for bucket, group in grouped:
        rows.append(
            {
                "analysis_type": "total_goals_calibration",
                "model": model,
                "game_id": "",
                "bucket": str(bucket),
                "games": len(group),
                "average_expected_total": float(group["expected_total_goals"].mean()),
                "average_actual_total": float(group["actual_total_goals"].mean()),
                "average_total_error": float(group["total_goals_error"].mean()),
                "score_mae": float(group["score_mae"].mean()),
                "margin_mae": float(group["margin_mae"].mean()),
                "total_goals_mae": float(group["total_goals_mae"].mean()),
            }
        )
    rows.append(
        {
            "analysis_type": "overall",
            "model": model,
            "game_id": "",
            "bucket": "all",
            "games": len(predictions),
            "average_expected_total": float(predictions["expected_total_goals"].mean()),
            "average_actual_total": float(predictions["actual_total_goals"].mean()),
            "average_total_error": float(predictions["total_goals_error"].mean()),
            "score_mae": float(predictions["score_mae"].mean()),
            "margin_mae": float(predictions["margin_mae"].mean()),
            "total_goals_mae": float(predictions["total_goals_mae"].mean()),
        }
    )
    return rows


def run_poisson_score_model() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_score_modeling_frame()
    raw_predictions = add_evaluation_columns(expanding_poisson_predictions(frame))
    calibrated_predictions = calibrate_poisson_predictions(raw_predictions, frame)
    raw_simulation, _raw_distribution = simulate_matchups(raw_predictions)
    calibrated_simulation, distribution = simulate_matchups(calibrated_predictions)
    comparison = calibration_comparison(
        raw_predictions,
        raw_simulation,
        calibrated_predictions,
        calibrated_simulation,
    )
    distribution["analysis_type"] = "per_game_score_distribution"
    distribution["model"] = "calibrated_poisson"
    distribution["bucket"] = ""
    distribution = pd.concat(
        [distribution, pd.DataFrame(score_calibration_rows(calibrated_predictions))],
        ignore_index=True,
        sort=False,
    )
    return raw_predictions, calibrated_predictions, calibrated_simulation, comparison, distribution


def main() -> None:
    raw_predictions, calibrated_predictions, calibrated_simulation, comparison, distribution = (
        run_poisson_score_model()
    )
    raw_predictions_path = write_output(raw_predictions, RAW_PREDICTIONS_OUTPUT)
    calibrated_predictions_path = write_output(calibrated_predictions, CALIBRATED_PREDICTIONS_OUTPUT)
    legacy_predictions_path = write_output(calibrated_predictions, LEGACY_PREDICTIONS_OUTPUT)
    calibrated_simulation_path = write_output(calibrated_simulation, CALIBRATED_SIMULATION_OUTPUT)
    legacy_simulation_path = write_output(calibrated_simulation, LEGACY_SIMULATION_OUTPUT)
    comparison_path = write_output(comparison, CALIBRATION_COMPARISON_OUTPUT)
    distribution_path = write_output(distribution, DISTRIBUTION_OUTPUT)
    print(f"Wrote {raw_predictions_path}")
    print(f"Wrote {calibrated_predictions_path}")
    print(f"Wrote {legacy_predictions_path}")
    print(f"Wrote {calibrated_simulation_path}")
    print(f"Wrote {legacy_simulation_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {distribution_path}")


if __name__ == "__main__":
    main()
