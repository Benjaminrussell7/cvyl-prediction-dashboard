# Modeling Lab Production Integration Plan

## Executive Recommendation

Promote calibrated Power v3 as the production winner-probability engine. Use calibrated Poisson only for projected score, total goals, and distribution context. Use confidence tiers, disagreement warnings, and archetype labels as interpretation layers that help users understand reliability and risk.

Poisson win probability should not replace Power v3 calibrated probability. Current rolling backtests show Power v3 calibrated and the hybrid layer at 75.7% winner accuracy with a 0.188 Brier score, while calibrated Poisson winner probability is lower at 62.1% accuracy with a 0.241 Brier score. Poisson is valuable for score shape, not for choosing the favorite.

## Safe To Consider For Production

| Output | Production Role | Readiness | Notes |
| --- | --- | --- | --- |
| `cvyl_power_ratings_v3_recency.csv` | Rankings and team strength | Already production-facing | Keep as the primary ranking source. |
| `cvyl_model_comparison_v4_calibrated.csv` / summary | Validation metrics for calibrated Power v3 | Already production-facing | Continue using calibrated Power v3 metrics for dashboard validation. |
| `modeling_lab/outputs/hybrid_predictions.csv` | Candidate integration reference | Safe with guardrails | Use Power v3 calibrated probability from this layer; use Poisson fields only as score context. |
| `modeling_lab/outputs/production_readiness_recommendations.csv` | Planning input | Safe | Decision support only, not a user-facing dataset. |
| `modeling_lab/outputs/modeling_lab_summary.csv` | Planning input | Safe | Concise model-readiness table. |
| `modeling_lab/outputs/archetype_performance_summary.csv` | Archetype validation | Safe for limited use | Use only labels with enough samples and clear wording. |
| `modeling_lab/outputs/disagreement_analysis.csv` | Warning validation | Safe for limited use | Use as a guardrail for lower-confidence messaging. |
| `modeling_lab/outputs/rolling_backtest_summary.csv` | Ongoing monitoring | Safe | Good candidate for internal model health checks. |

## Research-Only For Now

| Output | Reason To Keep Research-Only |
| --- | --- |
| `poisson_predictions_raw.csv` | Raw Poisson produced unrealistic extremes and should remain audit-only. |
| `poisson_simulation_summary.csv` raw legacy output | Superseded by calibrated Poisson outputs. |
| `poisson_calibration_comparison.csv` | Useful for model development, not direct dashboard display. |
| `poisson_predictions_calibrated.csv` | Useful for score projection research; do not expose as winner-probability source. |
| `poisson_simulation_summary_calibrated.csv` | Candidate for score context, but needs presentation safeguards. |
| `model_comparison.csv` ML baseline rows | Ridge/logistic/random forest do not clearly beat calibrated Power v3. |
| `feature_importance.csv`, `top_predictive_features.csv`, correlation outputs | Research diagnostics only. |
| `confidence_predictions.csv` | Useful as input to a future production confidence layer, but direct scores need UX calibration. |
| `disagreement_games.csv`, `model_disagreement_audit.csv` | Audit/debug outputs, not user-facing. |
| `game_archetypes.csv`, `archetype_examples.csv` | Candidate source data, but labels should be curated before production display. |
| `rolling_backtest_by_window.csv`, `rolling_backtest_by_model.csv`, `rolling_backtest_by_archetype.csv` | Internal model monitoring and QA. |

## Phased Integration Order

### Phase 1: Production Validation And Monitoring

Add internal-only model monitoring from:
- `rolling_backtest_summary.csv`
- `modeling_lab_summary.csv`
- `production_readiness_recommendations.csv`

Goal: track whether calibrated Power v3 remains the best winner-probability engine as more games are added.

### Phase 2: Confidence Tier Display

Promote a simplified confidence label using the hybrid confidence tier concept:
- High
- Medium
- Low

Use the label as interpretive context only. Do not change the displayed win probability formula.

### Phase 3: Disagreement Warning

Add a lightweight warning when Power v3 calibrated and calibrated Poisson strongly disagree directionally or have a large probability gap.

Use this as a caution message, not as an override.

### Phase 4: Projected Score Context

Use calibrated Poisson expected goals for projected score and total-goals context:
- expected home goals
- expected away goals
- expected total goals
- close-game probability if framed carefully

Do not display Poisson win probability as the main probability.

### Phase 5: Curated Archetype Labels

Introduce a small set of user-facing archetypes after wording review:
- Stable Favorite
- Consensus Pick
- Toss-Up
- Upset-Prone
- High-Variance
- Projected Shootout
- Defensive Grinder

Avoid overloading users with multi-label archetypes. Pick one primary label and one optional caution note.

## Required Production Data Dependencies

Minimum production inputs:
- completed canonical games from `data/processed/cvyl_games.csv`
- team-game rows from `data/processed/cvyl_team_games.csv`
- Power v3 recency ratings from `data/processed/cvyl_power_ratings_v3_recency.csv`
- calibrated Power v3 comparison outputs from `data/processed/cvyl_model_comparison_v4_calibrated.csv`
- current matchup prediction helper inputs already used by the dashboard

For future score context:
- calibrated Poisson expected score outputs or equivalent productionized score model output
- stable team name matching between production prediction teams and score-model teams
- date-stamped backtest outputs for monitoring drift

For future confidence/archetype display:
- confidence tier output
- disagreement tier output
- primary archetype label
- validation summaries by confidence tier and archetype

## Risks And Safeguards

| Risk | Safeguard |
| --- | --- |
| Poisson winner probability underperforms Power v3 | Never use Poisson win probability as the primary displayed probability. |
| Users over-trust projected scores | Label projected scores as estimates, not forecasts of exact final scores. |
| Small samples make confidence tiers unstable | Display lower certainty language until each tier has more validation games. |
| Archetype labels sound too definitive | Use descriptive labels with cautious copy, especially for youth sports. |
| Model disagreement confuses users | Explain disagreement as a reason to treat the pick cautiously, not as a contradiction. |
| More recent CVYL results may be missing | Keep the existing data freshness disclaimer visible. |
| Team aliases or duplicate teams distort outputs | Continue team identity audits before promoting new data-dependent layers. |
| Rolling backtest windows are currently limited | Monitor over more weeks before using drift as a public-facing metric. |

## Proposed User-Facing Language

### Confidence Tier

High:
> Higher-confidence prediction based on current ratings, recent results, and model agreement.

Medium:
> Moderate confidence. The favorite is supported, but the matchup still has meaningful uncertainty.

Low:
> Lower-confidence prediction. Treat this matchup as more uncertain than the probability alone suggests.

### Archetype Label

Stable Favorite:
> The favorite is supported by multiple signals and has shown a more reliable profile.

Consensus Pick:
> Rating and score-context signals generally point in the same direction.

Toss-Up:
> This matchup profiles as close. Small changes in available scores could move the projection.

Deceptive Toss-Up:
> The win probability looks close, but supporting signals disagree. This is a higher-uncertainty matchup.

Upset-Prone:
> The favorite is still favored, but the matchup has enough risk indicators to watch for an upset.

High-Variance:
> The model sees more volatility than usual. Score and margin estimates may be less stable.

Projected Shootout:
> The scoring environment projects higher than usual.

Defensive Grinder:
> The scoring environment projects lower than usual, with fewer goals expected.

### Model Disagreement Warning

Standard warning:
> Model signals are split on this matchup, so confidence is lower than the win probability alone may imply.

Strong warning:
> Power Rating and score-distribution signals disagree sharply. Treat this projection cautiously.

Directional disagreement:
> The primary rating model and projected score model lean toward different teams. The Power Rating probability remains the official pick.

### Projected Score Context

Standard:
> Projected score: {team_a} {score_a}, {team_b} {score_b}. This is an estimated scoring range, not an exact final-score prediction.

Projected total:
> Projected total goals: {total}. Use this as scoring-environment context.

Close-game context:
> Simulations show an elevated chance of a close game, so the spread may be less reliable.

## Explicit Non-Promotion Decision

Poisson win probability should not replace Power v3 calibrated probability.

The production winner-probability hierarchy should be:
1. Primary: calibrated Power v3 probability.
2. Supporting: confidence tier, disagreement warning, and archetype label.
3. Context only: calibrated Poisson projected score, projected total, and distribution notes.
4. Research only: Poisson win probability, raw Poisson outputs, and current ML baseline outputs.

## Recommended Production Architecture

Use a layered presentation:

1. Prediction engine:
   - calibrated Power v3 win probability
   - Power v3 favorite

2. Score context:
   - calibrated Poisson expected score
   - expected total
   - optional close-game note

3. Reliability context:
   - confidence tier
   - disagreement warning
   - primary archetype label

4. Monitoring:
   - rolling backtest summary
   - model comparison summary
   - archetype performance summary

This gives users a clear favorite and probability while preserving the richer modeling lab signals as context rather than replacing the stable production prediction engine.
