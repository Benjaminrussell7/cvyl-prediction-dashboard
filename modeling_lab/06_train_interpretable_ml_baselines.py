from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import OUTPUTS, write_output


FEATURES_INPUT = "leakage_safe_features.csv"
TARGETS_INPUT = "targets.csv"
MODEL_COMPARISON_OUTPUT = "model_comparison.csv"
FEATURE_IMPORTANCE_OUTPUT = "feature_importance.csv"
TOP_FEATURES_OUTPUT = "top_predictive_features.csv"
FEATURE_CORRELATIONS_OUTPUT = "feature_correlations.csv"
HIGHLY_CORRELATED_OUTPUT = "highly_correlated_features.csv"

RANDOM_SEED = 42
TEST_FRACTION = 0.30
L2_ALPHA = 1.0
LOGISTIC_LEARNING_RATE = 0.08
LOGISTIC_STEPS = 5000
FOREST_TREES = 75
FOREST_MAX_DEPTH = 3
FOREST_MIN_LEAF = 5


@dataclass
class PreparedData:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    means: pd.Series
    stds: pd.Series


@dataclass
class TreeNode:
    probability: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def load_modeling_frame() -> pd.DataFrame:
    features = pd.read_csv(OUTPUTS / FEATURES_INPUT)
    targets = pd.read_csv(OUTPUTS / TARGETS_INPUT)
    frame = features.merge(targets, on=["game_id", "game_date", "home_team", "away_team"], how="inner")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame.sort_values(["game_date", "game_id"], ignore_index=True)


def prepare_time_split(frame: pd.DataFrame) -> PreparedData:
    excluded = {
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_winner",
        "home_win",
        "home_margin",
        "total_goals",
    }
    feature_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    split_index = max(1, int(len(frame) * (1.0 - TEST_FRACTION)))
    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()
    means = train[feature_columns].astype(float).mean()
    stds = train[feature_columns].astype(float).std().replace(0.0, 1.0).fillna(1.0)
    return PreparedData(train=train, test=test, feature_columns=feature_columns, means=means, stds=stds)


def standardized_matrix(frame: pd.DataFrame, prepared: PreparedData) -> np.ndarray:
    values = frame[prepared.feature_columns].astype(float).fillna(prepared.means)
    return ((values - prepared.means) / prepared.stds).to_numpy(dtype=float)


def add_intercept(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(matrix)), matrix])


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float = L2_ALPHA) -> np.ndarray:
    x = add_intercept(x_train)
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y_train)


def predict_ridge(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ weights


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float = L2_ALPHA,
    learning_rate: float = LOGISTIC_LEARNING_RATE,
    steps: int = LOGISTIC_STEPS,
) -> np.ndarray:
    x = add_intercept(x_train)
    weights = np.zeros(x.shape[1], dtype=float)
    for _ in range(steps):
        probabilities = sigmoid(x @ weights)
        gradient = (x.T @ (probabilities - y_train)) / len(y_train)
        gradient[1:] += alpha * weights[1:] / len(y_train)
        weights -= learning_rate * gradient
    return weights


def predict_logistic(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return sigmoid(add_intercept(x) @ weights)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_trees: int = FOREST_TREES,
    max_depth: int = FOREST_MAX_DEPTH,
    min_leaf: int = FOREST_MIN_LEAF,
) -> tuple[list[TreeNode], np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    trees: list[TreeNode] = []
    importances = np.zeros(x_train.shape[1], dtype=float)
    for _ in range(n_trees):
        sample_idx = rng.integers(0, len(x_train), len(x_train))
        feature_count = max(1, int(math.sqrt(x_train.shape[1])))
        tree, tree_importance = build_tree(
            x_train[sample_idx],
            y_train[sample_idx],
            depth=0,
            max_depth=max_depth,
            min_leaf=min_leaf,
            rng=rng,
            feature_count=feature_count,
        )
        trees.append(tree)
        importances += tree_importance
    if importances.sum() > 0:
        importances = importances / importances.sum()
    return trees, importances


def build_tree(
    x: np.ndarray,
    y: np.ndarray,
    *,
    depth: int,
    max_depth: int,
    min_leaf: int,
    rng: np.random.Generator,
    feature_count: int,
) -> tuple[TreeNode, np.ndarray]:
    probability = float(y.mean()) if len(y) else 0.5
    importances = np.zeros(x.shape[1], dtype=float)
    if depth >= max_depth or len(y) < min_leaf * 2 or y.min() == y.max():
        return TreeNode(probability=probability), importances

    candidate_features = rng.choice(x.shape[1], size=feature_count, replace=False)
    parent_impurity = gini(y)
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    best_mask = None
    for feature_index in candidate_features:
        thresholds = np.unique(np.quantile(x[:, feature_index], [0.25, 0.5, 0.75]))
        for threshold in thresholds:
            mask = x[:, feature_index] <= threshold
            if mask.sum() < min_leaf or (~mask).sum() < min_leaf:
                continue
            gain = parent_impurity - (
                mask.mean() * gini(y[mask]) + (~mask).mean() * gini(y[~mask])
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = int(feature_index)
                best_threshold = float(threshold)
                best_mask = mask
    if best_feature is None or best_mask is None:
        return TreeNode(probability=probability), importances

    left, left_importance = build_tree(
        x[best_mask],
        y[best_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        rng=rng,
        feature_count=feature_count,
    )
    right, right_importance = build_tree(
        x[~best_mask],
        y[~best_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        rng=rng,
        feature_count=feature_count,
    )
    importances += left_importance + right_importance
    importances[best_feature] += best_gain * len(y)
    return (
        TreeNode(
            probability=probability,
            feature_index=best_feature,
            threshold=best_threshold,
            left=left,
            right=right,
        ),
        importances,
    )


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = float(y.mean())
    return 2.0 * p * (1.0 - p)


def predict_tree(node: TreeNode, row: np.ndarray) -> float:
    if node.feature_index is None or node.threshold is None:
        return node.probability
    if row[node.feature_index] <= node.threshold and node.left is not None:
        return predict_tree(node.left, row)
    if node.right is not None:
        return predict_tree(node.right, row)
    return node.probability


def predict_forest(trees: list[TreeNode], x: np.ndarray) -> np.ndarray:
    predictions = np.array([[predict_tree(tree, row) for tree in trees] for row in x])
    return predictions.mean(axis=1)


def evaluate_probability_model(probabilities: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(((probabilities >= 0.5) == (actual == 1.0)).mean()),
        "brier_score": float(((probabilities - actual) ** 2).mean()),
        "log_loss": float((-(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))).mean()),
        "calibration_gap": float(actual.mean() - probabilities.mean()),
    }


def evaluate_margin_model(predicted_margin: np.ndarray, actual_margin: np.ndarray) -> dict[str, float]:
    probabilities = sigmoid(predicted_margin / 4.0)
    binary_actual = (actual_margin > 0).astype(float)
    metrics = evaluate_probability_model(probabilities, binary_actual)
    metrics["mae_margin"] = float(np.abs(predicted_margin - actual_margin).mean())
    return metrics


def permutation_importance(
    x_test: np.ndarray,
    y_test: np.ndarray,
    baseline_score: float,
    predict_fn,
    *,
    metric: str,
) -> np.ndarray:
    importances = []
    for feature_index in range(x_test.shape[1]):
        permuted = x_test.copy()
        permuted[:, feature_index] = np.roll(permuted[:, feature_index], 1)
        predictions = predict_fn(permuted)
        if metric == "brier":
            score = float(((predictions - y_test) ** 2).mean())
        elif metric == "mae":
            score = float(np.abs(predictions - y_test).mean())
        else:
            raise ValueError(f"Unsupported permutation metric: {metric}")
        importances.append(score - baseline_score)
    return np.array(importances)


def run_ml_baselines() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_modeling_frame()
    prepared = prepare_time_split(frame)
    train_binary = prepared.train[prepared.train["home_win"].isin([0.0, 1.0])].copy()
    test_binary = prepared.test[prepared.test["home_win"].isin([0.0, 1.0])].copy()

    x_train = standardized_matrix(prepared.train, prepared)
    x_test = standardized_matrix(prepared.test, prepared)
    x_train_binary = standardized_matrix(train_binary, prepared)
    x_test_binary = standardized_matrix(test_binary, prepared)
    y_margin_train = prepared.train["home_margin"].astype(float).to_numpy()
    y_margin_test = prepared.test["home_margin"].astype(float).to_numpy()
    y_train_binary = train_binary["home_win"].astype(float).to_numpy()
    y_test_binary = test_binary["home_win"].astype(float).to_numpy()

    model_rows = []
    importance_rows = []
    baseline_columns = {
        "elo_baseline": "elo_home_probability",
        "power_v3_baseline": "power_v3_recency_home_probability",
        "power_v3_calibrated_baseline": "power_v3_calibrated_home_probability",
    }
    for model_name, column in baseline_columns.items():
        probabilities = test_binary[column].astype(float).to_numpy()
        metrics = evaluate_probability_model(probabilities, y_test_binary)
        metrics.update({"model": model_name, "model_type": "baseline", "test_games": len(test_binary), "mae_margin": np.nan})
        model_rows.append(metrics)

    ridge_weights = fit_ridge(x_train, y_margin_train)
    ridge_margin = predict_ridge(x_test, ridge_weights)
    ridge_metrics = evaluate_margin_model(ridge_margin, y_margin_test)
    ridge_metrics.update({"model": "ridge_margin", "model_type": "ridge_regression", "test_games": len(prepared.test)})
    model_rows.append(ridge_metrics)
    ridge_baseline_mae = float(np.abs(ridge_margin - y_margin_test).mean())
    ridge_perm = permutation_importance(
        x_test,
        y_margin_test,
        ridge_baseline_mae,
        lambda matrix: predict_ridge(matrix, ridge_weights),
        metric="mae",
    )
    add_linear_importance_rows(importance_rows, "ridge_margin", prepared.feature_columns, ridge_weights[1:], ridge_perm)

    logistic_weights = fit_logistic(x_train_binary, y_train_binary)
    logistic_probability = predict_logistic(x_test_binary, logistic_weights)
    logistic_metrics = evaluate_probability_model(logistic_probability, y_test_binary)
    logistic_metrics.update({"model": "logistic_win_probability", "model_type": "logistic_regression", "test_games": len(test_binary), "mae_margin": np.nan})
    model_rows.append(logistic_metrics)
    logistic_baseline_brier = float(((logistic_probability - y_test_binary) ** 2).mean())
    logistic_perm = permutation_importance(
        x_test_binary,
        y_test_binary,
        logistic_baseline_brier,
        lambda matrix: predict_logistic(matrix, logistic_weights),
        metric="brier",
    )
    add_linear_importance_rows(
        importance_rows,
        "logistic_win_probability",
        prepared.feature_columns,
        logistic_weights[1:],
        logistic_perm,
    )

    forest, forest_importance = fit_random_forest(x_train_binary, y_train_binary)
    forest_probability = predict_forest(forest, x_test_binary)
    forest_metrics = evaluate_probability_model(forest_probability, y_test_binary)
    forest_metrics.update({"model": "random_forest_win_probability", "model_type": "random_forest_classifier", "test_games": len(test_binary), "mae_margin": np.nan})
    model_rows.append(forest_metrics)
    forest_baseline_brier = float(((forest_probability - y_test_binary) ** 2).mean())
    forest_perm = permutation_importance(
        x_test_binary,
        y_test_binary,
        forest_baseline_brier,
        lambda matrix: predict_forest(forest, matrix),
        metric="brier",
    )
    for feature, impurity, permutation in zip(prepared.feature_columns, forest_importance, forest_perm, strict=True):
        importance_rows.append(
            {
                "model": "random_forest_win_probability",
                "feature": feature,
                "importance_type": "random_forest_impurity",
                "importance": float(impurity),
            }
        )
        importance_rows.append(
            {
                "model": "random_forest_win_probability",
                "feature": feature,
                "importance_type": "permutation_brier_increase",
                "importance": float(permutation),
            }
        )

    comparison = pd.DataFrame(model_rows)[
        ["model", "model_type", "test_games", "accuracy", "brier_score", "log_loss", "calibration_gap", "mae_margin"]
    ]
    importance = pd.DataFrame(importance_rows)
    top_features = (
        importance.assign(abs_importance=importance["importance"].abs())
        .sort_values(["abs_importance", "model", "feature"], ascending=[False, True, True])
        .groupby("model", as_index=False)
        .head(10)
        .drop(columns=["abs_importance"])
        .reset_index(drop=True)
    )
    correlations, highly_correlated = feature_correlation_analysis(frame, prepared.feature_columns)
    return comparison, importance, top_features, correlations, highly_correlated


def feature_correlation_analysis(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = frame[feature_columns].astype(float)
    corr = numeric.corr().fillna(0.0)
    rows = []
    for left_index, feature_a in enumerate(feature_columns):
        for feature_b in feature_columns[left_index + 1:]:
            value = float(corr.loc[feature_a, feature_b])
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    correlations = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False, ignore_index=True)
    highly_correlated = correlations[correlations["abs_correlation"] >= threshold].copy()
    return correlations, highly_correlated


def add_linear_importance_rows(
    rows: list[dict[str, object]],
    model_name: str,
    features: list[str],
    coefficients: np.ndarray,
    permutation: np.ndarray,
) -> None:
    for feature, coefficient, permuted in zip(features, coefficients, permutation, strict=True):
        rows.append(
            {
                "model": model_name,
                "feature": feature,
                "importance_type": "coefficient",
                "importance": float(coefficient),
            }
        )
        rows.append(
            {
                "model": model_name,
                "feature": feature,
                "importance_type": "permutation_mae_increase" if model_name == "ridge_margin" else "permutation_brier_increase",
                "importance": float(permuted),
            }
        )


def main() -> None:
    comparison, importance, top_features, correlations, highly_correlated = run_ml_baselines()
    comparison_path = write_output(comparison, MODEL_COMPARISON_OUTPUT)
    importance_path = write_output(importance, FEATURE_IMPORTANCE_OUTPUT)
    top_features_path = write_output(top_features, TOP_FEATURES_OUTPUT)
    correlations_path = write_output(correlations, FEATURE_CORRELATIONS_OUTPUT)
    highly_correlated_path = write_output(highly_correlated, HIGHLY_CORRELATED_OUTPUT)
    print(f"Wrote {comparison_path}")
    print(f"Wrote {importance_path}")
    print(f"Wrote {top_features_path}")
    print(f"Wrote {correlations_path}")
    print(f"Wrote {highly_correlated_path}")
    print("SHAP values skipped: shap is not installed and no new dependency was added.")


if __name__ == "__main__":
    main()
