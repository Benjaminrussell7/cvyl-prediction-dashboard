# Modeling Lab Key Findings

## Executive Decision

Use calibrated Power v3 as the production winner-probability engine. Use calibrated Poisson only for expected score and distribution context. Use confidence, disagreement, and archetype layers as interpretation and caution signals.

## Decision Summary

| decision_area | best_current_choice | status | recommendation |
| --- | --- | --- | --- |
| winner_prediction | power_v3_calibrated_baseline | production_ready | Keep calibrated Power v3 as the primary winner probability model. |
| probability_calibration | power_v3_calibrated_baseline | production_ready | Use calibrated Power v3 probabilities for displayed win probabilities. |
| score_projection | calibrated_poisson | experimental | Use calibrated Poisson as score context only, not as the winner engine. |
| confidence_uncertainty | hybrid confidence + disagreement analysis | experimental_but_useful | Use confidence and disagreement as warning labels, not as automatic model overrides. |
| reliable_archetypes | stable_favorite | production_candidate | Use reliable archetypes to explain when model confidence is more trustworthy. |
| volatile_archetypes | offensive_shootout | warning_signal | Show volatile archetypes as caution flags, not strong picks. |
| upset_archetypes | toss_up | warning_signal | Use upset-prone archetypes to lower narrative certainty. |
| recommended_architecture | Power v3 calibrated + Poisson score context + confidence warnings | recommended | Promote Power v3 calibrated for probabilities, calibrated Poisson for score projections, and confidence/disagreement/archetype layers for interpretation. |

## Production Readiness

| component | readiness | use_now | recommended_role |
| --- | --- | --- | --- |
| calibrated_power_v3_probability | production_ready | yes | Primary winner probability and displayed model confidence. |
| calibrated_poisson_scores | experimental | limited | Expected score, total, and simulation distribution context. |
| hybrid_prediction_layer | experimental_but_safe | yes_for_analysis | Decision layer that keeps Power v3 as anchor and attaches Poisson/confidence context. |
| confidence_engine | experimental_but_useful | yes_as_warning_label | Reliability tier and upset-risk language. |
| disagreement_analysis | production_candidate | yes_as_guardrail | Warn when Power and Poisson disagree or when confidence should be reduced. |
| archetype_labels | production_candidate | yes_for_explanations | Short interpretation labels for reliable, volatile, upset-prone, and toss-up games. |
| advanced_ml_baselines | not_recommended_yet | no | Research only. |
| raw_poisson_probabilities | not_recommended | no | Audit baseline only. |

## Reliable Archetypes

| archetype | games | winner_accuracy | brier_score | upset_rate | average_confidence |
| --- | --- | --- | --- | --- | --- |
| stable_favorite | 23 | 0.9565217391304348 | 0.102597042714405 | 0.0434782608695652 | 73.35521739130436 |
| consensus_pick | 19 | 0.9473684210526316 | 0.1059596849448434 | 0.0526315789473684 | 73.8736842105263 |
| volatile_favorite | 10 | 0.9 | 0.1229122271160301 | 0.1 | 65.166 |
| offensive_shootout | 16 | 0.875 | 0.1798055048208314 | 0.125 | 54.128125 |
| high_variance | 47 | 0.7446808510638298 | 0.203620515104053 | 0.2553191489361702 | 53.1427659574468 |

## Volatile Archetypes

| archetype | games | margin_mae | upset_rate | close_game_rate | average_confidence |
| --- | --- | --- | --- | --- | --- |
| offensive_shootout | 16 | 7.5529153726257 | 0.125 | 0.25 | 54.128125 |
| high_variance | 47 | 6.5759060996943 | 0.2553191489361702 | 0.2340425531914893 | 53.1427659574468 |
| deceptive_toss_up | 35 | 6.489216467952327 | 0.3714285714285714 | 0.2857142857142857 | 51.29685714285714 |
| volatile_favorite | 10 | 5.516689989051042 | 0.1 | 0.1 | 65.166 |
| upset_prone | 53 | 4.985008196645652 | 0.2830188679245283 | 0.3773584905660377 | 54.84056603773585 |

## Do Not Promote Yet

| component | readiness | rationale |
| --- | --- | --- |
| advanced_ml_baselines | not_recommended_yet | Current Ridge/logistic/random-forest baselines do not clearly beat calibrated Power v3. |
| raw_poisson_probabilities | not_recommended | Raw Poisson produced unrealistic extremes and worse score/winner calibration. |

## Next Experiments

- Tune calibrated Poisson score projections with more conservative total-goals and margin calibration.
- Test whether disagreement-aware confidence thresholds improve user-facing warning quality.
- Backtest archetype-specific probability calibration instead of one global calibration curve.
- Add more data before promoting ML baselines; current samples are too small for reliable nonlinear model selection.
- Evaluate whether Poisson score projections improve tournament simulation realism without affecting winner probabilities.
