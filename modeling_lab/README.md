# CVYL Modeling Lab

Standalone experimentation layer for future ML work.

Constraints:
- Reads existing files from `data/processed/`.
- Writes all generated artifacts to `modeling_lab/outputs/`.
- Does not modify production model formulas, prediction logic, dashboard code, or CSV schemas.
- No advanced ML models are included yet.

Run all lab steps:

```bash
python modeling_lab/run_all.py
```

Outputs:
- `outputs/historical_pregame_matchups.csv`
- `outputs/leakage_safe_features.csv`
- `outputs/targets.csv`
- `outputs/baseline_predictions.csv`
- `outputs/evaluation_summary.csv`
- `outputs/calibration_summary.csv`
- `outputs/model_comparison.csv`
- `outputs/feature_importance.csv`
- `outputs/top_predictive_features.csv`
- `outputs/feature_correlations.csv`
- `outputs/highly_correlated_features.csv`
- `outputs/confidence_predictions.csv`
- `outputs/upset_risk_analysis.csv`
- `outputs/prediction_agreement_analysis.csv`
- `outputs/poisson_predictions_raw.csv`
- `outputs/poisson_predictions_calibrated.csv`
- `outputs/poisson_calibration_comparison.csv`
- `outputs/poisson_simulation_summary_calibrated.csv`
- `outputs/poisson_predictions.csv`
- `outputs/poisson_simulation_summary.csv`
- `outputs/score_distribution_analysis.csv`
- `outputs/hybrid_predictions.csv`
- `outputs/hybrid_model_comparison.csv`
- `outputs/hybrid_explanation_components.csv`
- `outputs/disagreement_analysis.csv`
- `outputs/disagreement_games.csv`
- `outputs/model_disagreement_audit.csv`
- `outputs/game_archetypes.csv`
- `outputs/archetype_performance_summary.csv`
- `outputs/archetype_examples.csv`
- `outputs/modeling_lab_summary.csv`
- `outputs/production_readiness_recommendations.csv`
- `outputs/modeling_lab_key_findings.md`
- `outputs/rolling_backtest_summary.csv`
- `outputs/rolling_backtest_by_window.csv`
- `outputs/rolling_backtest_by_model.csv`
- `outputs/rolling_backtest_by_archetype.csv`

Interpretable ML baselines:
- Ridge regression predicts home scoring margin.
- Logistic regression predicts home win probability.
- A lightweight deterministic random forest classifier acts as a nonlinear challenger.
- All model comparisons use a chronological train/test split. No random train/test split is used.
- SHAP values are skipped unless the project later adds a SHAP dependency.
- Advanced engineered features include leakage-safe historical variance, momentum, matchup interaction, SOS decomposition, and uncertainty metrics.
- Confidence engine outputs are standalone research artifacts that combine stability, agreement, volatility, strength gap, and matchup consistency into a 0-100 confidence score and upset-risk tier.
- Poisson score modeling uses expanding-window pregame features to estimate expected home/away goals and simulation summaries for future simulator research.
- The Poisson calibration layer preserves raw predictions, shrinks expected goals toward prior league scoring averages, soft-clips extreme values, and compares raw vs calibrated score, margin, total-goals, winner, Brier, and bucket calibration metrics.
- The hybrid prediction layer keeps calibrated Power v3 as the winner-probability anchor, uses calibrated Poisson for expected scores and simulation context, and folds in confidence-engine reliability and upset-risk signals.
- The disagreement analysis layer audits when calibrated Power v3 and calibrated Poisson disagree, whether that disagreement predicts lower accuracy or confidence, and which games are most useful for review.
- The game archetype layer assigns interpretable matchup labels such as stable favorite, deceptive toss-up, shootout, grinder, upset-prone, high-variance, and consensus pick, then evaluates archetype reliability and example games.
- The executive summary layer condenses lab findings into production readiness recommendations, key decisions, and next experiments.
- The rolling backtest layer evaluates Power v3, calibrated Poisson, hybrid predictions, confidence tiers, disagreement tiers, and archetypes across expanding chronological test windows.
