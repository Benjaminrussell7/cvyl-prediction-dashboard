from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUTPUTS, write_output


HYBRID_INPUT = "hybrid_predictions.csv"
DISAGREEMENT_INPUT = "disagreement_games.csv"
ARCHETYPES_INPUT = "game_archetypes.csv"

SUMMARY_OUTPUT = "rolling_backtest_summary.csv"
BY_WINDOW_OUTPUT = "rolling_backtest_by_window.csv"
BY_MODEL_OUTPUT = "rolling_backtest_by_model.csv"
BY_ARCHETYPE_OUTPUT = "rolling_backtest_by_archetype.csv"

TEST_WINDOW_DAYS = 7
EPSILON = 1e-6


MODELS = [
    {
        "model": "power_v3_calibrated",
        "probability_column": "power_v3_calibrated_home_probability",
        "score_model": False,
    },
    {
        "model": "calibrated_poisson",
        "probability_column": "poisson_home_win_probability",
        "score_model": True,
    },
    {
        "model": "hybrid",
        "probability_column": "hybrid_home_win_probability",
        "score_model": True,
    },
]


def read_lab_output(filename: str) -> pd.DataFrame:
    path = OUTPUTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing modeling lab input: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().removeprefix("\ufeff") for column in frame.columns]
    return frame


def actual_home_win(row: pd.Series) -> float:
    if row["actual_winner"] == row["home_team"]:
        return 1.0
    if row["actual_winner"] == row["away_team"]:
        return 0.0
    return 0.5


def predicted_winner(row: pd.Series, probability_column: str) -> str:
    return row["home_team"] if float(row[probability_column]) >= 0.5 else row["away_team"]


def brier_score(actual: pd.Series, probability: pd.Series) -> float:
    return float(((probability.astype(float) - actual.astype(float)) ** 2).mean())


def log_loss(actual: pd.Series, probability: pd.Series) -> float:
    clipped = probability.astype(float).clip(EPSILON, 1.0 - EPSILON)
    return float((-(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))).mean())


def calibration_gap(actual: pd.Series, probability: pd.Series) -> float:
    favorite_probability = np.where(probability.astype(float) >= 0.5, probability, 1.0 - probability)
    favorite_won = np.where(probability.astype(float) >= 0.5, actual, 1.0 - actual)
    return float(pd.Series(favorite_probability).mean() - pd.Series(favorite_won).mean())


def load_backtest_frame() -> pd.DataFrame:
    hybrid = read_lab_output(HYBRID_INPUT)
    disagreement = read_lab_output(DISAGREEMENT_INPUT)[
        [
            "game_id",
            "disagreement_tier",
            "directional_disagreement",
            "absolute_probability_gap",
            "power_correct",
            "poisson_correct",
        ]
    ]
    archetypes = read_lab_output(ARCHETYPES_INPUT)[
        [
            "game_id",
            "primary_archetype",
            "archetypes",
            "favorite_probability",
            "actual_upset",
            "actual_close_game",
        ]
    ]
    frame = hybrid.merge(disagreement, on="game_id", how="inner")
    frame = frame.merge(archetypes, on="game_id", how="inner", suffixes=("", "_archetype"))
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame = frame.sort_values(["game_date", "game_id"], ignore_index=True)
    frame["actual_home_win"] = frame.apply(actual_home_win, axis=1)
    frame["actual_total_goals"] = frame["home_score"].astype(float) + frame["away_score"].astype(float)
    frame["actual_margin"] = frame["home_score"].astype(float) - frame["away_score"].astype(float)
    for model in MODELS:
        column = model["probability_column"]
        frame[f"{model['model']}_predicted_winner"] = frame.apply(
            lambda row, probability_column=column: predicted_winner(row, probability_column),
            axis=1,
        )
    return frame


def assign_windows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    first_date = output["game_date"].min().normalize()
    day_offset = (output["game_date"].dt.normalize() - first_date).dt.days
    output["window_index"] = (day_offset // TEST_WINDOW_DAYS).astype(int) + 1
    output["window_start"] = first_date + pd.to_timedelta((output["window_index"] - 1) * TEST_WINDOW_DAYS, unit="D")
    output["window_end"] = output["window_start"] + pd.to_timedelta(TEST_WINDOW_DAYS - 1, unit="D")
    output["train_through_date"] = output["window_start"] - pd.to_timedelta(1, unit="D")
    output["train_games"] = output.groupby("window_index").cumcount()
    counts_before_window = (
        output.groupby("window_index")
        .size()
        .sort_index()
        .cumsum()
        .shift(fill_value=0)
        .rename("prior_games")
    )
    output = output.merge(counts_before_window, left_on="window_index", right_index=True, how="left")
    output["train_games"] = output["prior_games"].astype(int)
    return output.drop(columns=["prior_games"])


def metric_row(
    frame: pd.DataFrame,
    model: str,
    probability_column: str,
    predicted_winner_column: str,
    score_model: bool,
    extra: dict[str, object],
) -> dict[str, object]:
    if frame.empty:
        base = {
            "model": model,
            "games": 0,
            "winner_accuracy": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
            "margin_mae": np.nan,
            "total_goals_mae": np.nan,
            "calibration_gap": np.nan,
            "average_confidence_score": np.nan,
            "upset_rate": np.nan,
            "close_game_rate": np.nan,
        }
        return {**extra, **base}

    actual = frame["actual_home_win"]
    probability = frame[probability_column]
    predicted = frame[predicted_winner_column]
    base = {
        "model": model,
        "games": len(frame),
        "winner_accuracy": float((predicted == frame["actual_winner"]).mean()),
        "brier_score": brier_score(actual, probability),
        "log_loss": log_loss(actual, probability),
        "margin_mae": float(frame["margin_mae"].mean()) if score_model else np.nan,
        "total_goals_mae": float(frame["total_goals_mae"].mean()) if score_model else np.nan,
        "calibration_gap": calibration_gap(actual, probability),
        "average_confidence_score": float(frame["hybrid_confidence_score"].mean()),
        "upset_rate": float((predicted != frame["actual_winner"]).mean()),
        "close_game_rate": float(frame["actual_close_game"].mean()),
    }
    return {**extra, **base}


def by_window(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window_index, group in frame.groupby("window_index", observed=True):
        window_extra = {
            "window_index": int(window_index),
            "train_through_date": group["train_through_date"].iloc[0].date().isoformat(),
            "window_start": group["window_start"].iloc[0].date().isoformat(),
            "window_end": group["window_end"].iloc[0].date().isoformat(),
            "train_games": int(group["train_games"].iloc[0]),
            "test_games": len(group),
        }
        for model in MODELS:
            rows.append(
                metric_row(
                    group,
                    model["model"],
                    model["probability_column"],
                    f"{model['model']}_predicted_winner",
                    bool(model["score_model"]),
                    window_extra,
                )
            )
    return pd.DataFrame(rows)


def by_model(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        rows.append(
            metric_row(
                frame,
                model["model"],
                model["probability_column"],
                f"{model['model']}_predicted_winner",
                bool(model["score_model"]),
                {"analysis_type": "overall", "bucket": "all"},
            )
        )
        for tier, group in frame.groupby("hybrid_confidence_tier", observed=True):
            rows.append(
                metric_row(
                    group,
                    model["model"],
                    model["probability_column"],
                    f"{model['model']}_predicted_winner",
                    bool(model["score_model"]),
                    {"analysis_type": "confidence_tier", "bucket": str(tier)},
                )
            )
        for tier, group in frame.groupby("disagreement_tier", observed=True):
            rows.append(
                metric_row(
                    group,
                    model["model"],
                    model["probability_column"],
                    f"{model['model']}_predicted_winner",
                    bool(model["score_model"]),
                    {"analysis_type": "disagreement_tier", "bucket": str(tier)},
                )
            )
    return pd.DataFrame(rows)


def by_archetype(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for archetype, group in frame.groupby("primary_archetype", observed=True):
        for model in MODELS:
            rows.append(
                metric_row(
                    group,
                    model["model"],
                    model["probability_column"],
                    f"{model['model']}_predicted_winner",
                    bool(model["score_model"]),
                    {"archetype": str(archetype), "analysis_type": "primary_archetype"},
                )
            )
    expanded_rows: list[pd.DataFrame] = []
    for _, row in frame.iterrows():
        for archetype in str(row["archetypes"]).split("|"):
            item = row.copy()
            item["archetype"] = archetype
            expanded_rows.append(item.to_frame().T)
    if expanded_rows:
        expanded = pd.concat(expanded_rows, ignore_index=True, sort=False)
        for archetype, group in expanded.groupby("archetype", observed=True):
            for model in MODELS:
                rows.append(
                    metric_row(
                        group,
                        model["model"],
                        model["probability_column"],
                        f"{model['model']}_predicted_winner",
                        bool(model["score_model"]),
                        {"archetype": str(archetype), "analysis_type": "multi_label_archetype"},
                    )
                )
    return pd.DataFrame(rows)


def performance_drift(by_window_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in by_window_frame.groupby("model", observed=True):
        group = group.sort_values("window_index")
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "analysis_type": "performance_drift",
                "bucket": "first_to_last_window",
                "model": model,
                "games": int(group["test_games"].sum()),
                "winner_accuracy": float(last["winner_accuracy"] - first["winner_accuracy"]),
                "brier_score": float(last["brier_score"] - first["brier_score"]),
                "log_loss": float(last["log_loss"] - first["log_loss"]),
                "margin_mae": float(last["margin_mae"] - first["margin_mae"])
                if pd.notna(last["margin_mae"]) and pd.notna(first["margin_mae"])
                else np.nan,
                "total_goals_mae": float(last["total_goals_mae"] - first["total_goals_mae"])
                if pd.notna(last["total_goals_mae"]) and pd.notna(first["total_goals_mae"])
                else np.nan,
                "calibration_gap": float(last["calibration_gap"] - first["calibration_gap"]),
                "average_confidence_score": float(
                    last["average_confidence_score"] - first["average_confidence_score"]
                ),
                "upset_rate": float(last["upset_rate"] - first["upset_rate"]),
                "close_game_rate": float(last["close_game_rate"] - first["close_game_rate"]),
            }
        )
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame, by_window_frame: pd.DataFrame, by_model_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "analysis_type": "rolling_backtest_overview",
            "bucket": "all",
            "model": "all",
            "games": len(frame),
            "windows": int(frame["window_index"].nunique()),
            "first_game_date": frame["game_date"].min().date().isoformat(),
            "last_game_date": frame["game_date"].max().date().isoformat(),
            "winner_accuracy": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
            "margin_mae": np.nan,
            "total_goals_mae": np.nan,
            "calibration_gap": np.nan,
            "average_confidence_score": float(frame["hybrid_confidence_score"].mean()),
            "upset_rate": float((~frame["hybrid_winner_correct"].astype(bool)).mean()),
            "close_game_rate": float(frame["actual_close_game"].mean()),
        }
    )
    overall_models = by_model_frame.loc[by_model_frame["analysis_type"] == "overall"].copy()
    for _, row in overall_models.iterrows():
        rows.append(
            {
                "analysis_type": "overall_model",
                "bucket": "all",
                "model": row["model"],
                "games": row["games"],
                "windows": int(frame["window_index"].nunique()),
                "first_game_date": frame["game_date"].min().date().isoformat(),
                "last_game_date": frame["game_date"].max().date().isoformat(),
                "winner_accuracy": row["winner_accuracy"],
                "brier_score": row["brier_score"],
                "log_loss": row["log_loss"],
                "margin_mae": row["margin_mae"],
                "total_goals_mae": row["total_goals_mae"],
                "calibration_gap": row["calibration_gap"],
                "average_confidence_score": row["average_confidence_score"],
                "upset_rate": row["upset_rate"],
                "close_game_rate": row["close_game_rate"],
            }
        )
    confidence = by_model_frame.loc[
        (by_model_frame["analysis_type"] == "confidence_tier") & (by_model_frame["model"] == "hybrid")
    ].copy()
    for _, row in confidence.iterrows():
        rows.append(
            {
                "analysis_type": "calibration_by_confidence_tier",
                "bucket": row["bucket"],
                "model": "hybrid",
                "games": row["games"],
                "windows": int(frame["window_index"].nunique()),
                "first_game_date": frame["game_date"].min().date().isoformat(),
                "last_game_date": frame["game_date"].max().date().isoformat(),
                "winner_accuracy": row["winner_accuracy"],
                "brier_score": row["brier_score"],
                "log_loss": row["log_loss"],
                "margin_mae": row["margin_mae"],
                "total_goals_mae": row["total_goals_mae"],
                "calibration_gap": row["calibration_gap"],
                "average_confidence_score": row["average_confidence_score"],
                "upset_rate": row["upset_rate"],
                "close_game_rate": row["close_game_rate"],
            }
        )
    drift = performance_drift(by_window_frame)
    if not drift.empty:
        rows.extend(drift.to_dict("records"))
    return pd.DataFrame(rows)


def run_rolling_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = assign_windows(load_backtest_frame())
    window_results = by_window(frame)
    model_results = by_model(frame)
    archetype_results = by_archetype(frame)
    summary_results = summary(frame, window_results, model_results)
    return summary_results, window_results, model_results, archetype_results


def main() -> None:
    summary_results, window_results, model_results, archetype_results = run_rolling_backtest()
    summary_path = write_output(summary_results, SUMMARY_OUTPUT)
    window_path = write_output(window_results, BY_WINDOW_OUTPUT)
    model_path = write_output(model_results, BY_MODEL_OUTPUT)
    archetype_path = write_output(archetype_results, BY_ARCHETYPE_OUTPUT)
    print(f"Wrote {summary_path}")
    print(f"Wrote {window_path}")
    print(f"Wrote {model_path}")
    print(f"Wrote {archetype_path}")


if __name__ == "__main__":
    main()
