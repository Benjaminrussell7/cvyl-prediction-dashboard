# Power V5 Candidate Summary

## Overview

All work described here is still experimental and remains inside `modeling_lab`. No production promotion has occurred yet, and no Streamlit or production prediction code has been modified.

The current modeling stack under evaluation has four pieces:

- Baseline Power v3: recency-weighted power ratings built from capped margins, opponent scoring profiles, shrinkage, and the existing logistic probability conversion.
- Opponent-adjusted performance: evaluates each prior team-game by capped actual margin versus expected margin based on opponent Power v3 strength, then averages performance above expectation with shrinkage.
- Calibration layer concepts: post-processing experiments that adjust only the translation from rating differential to win probability and implied spread.
- Rolling backtests: all major experiments use pregame-only rolling evaluation, so each prediction only sees games already completed before the target game.

## Key Metric Snapshot

| Model / Variant | Winner Accuracy | Brier | Log Loss | Margin MAE | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline Power v3 | 0.708 | 0.207 | 0.614 | 4.901 | Reference baseline |
| Hidden tiering connectivity | 0.701 | 0.211 | 0.624 | 4.945 | Rejected |
| Opponent-adjusted performance | 0.708 | 0.205 | 0.611 | 4.869 | Retained |
| Best blend, 60/40 | 0.715 | 0.206 | 0.613 | 4.888 | Not leading |
| Best recency variant, mild | 0.694 | 0.206 | 0.613 | 4.896 | Rejected |
| Power V5 calibrated candidate | 0.708 | 0.185 | 0.563 | 4.723 | Current leading candidate |

## Completed Experiments

### Hidden Tiering / Connectivity

Hypothesis: teams dominating weak or isolated schedules were overvalued, and teams in stronger ecosystems were undervalued. A graph exposure adjustment might improve ranking realism without using tournament data.

Result: connectivity penalties made some rankings feel more plausible, but worsened predictive performance. The connectivity-only variant fell to 0.701 accuracy with worse Brier, log loss, and margin MAE than baseline.

Decision: rejected. It was useful diagnostically, but it behaved too much like a manual confidence penalty and did not improve the backtest.

### Opponent-Adjusted Performance

Hypothesis: rankings should reward teams for outperforming expectation against opponent strength instead of manually penalizing weak schedule connectivity.

Result: improved Brier from 0.207 to 0.205, log loss from 0.614 to 0.611, and margin MAE from 4.901 to 4.869 while preserving 0.708 winner accuracy. It also improved ranking realism for teams such as Westfield, RHAM, Avon, Suffield, and Somers.

Decision: retained. This became the leading rating-engine candidate.

### Blended Power V4

Hypothesis: a weighted blend of baseline Power v3 and opponent-adjusted performance might outperform either model alone.

Result: blends improved over baseline, and the 60/40 blend reached 0.715 winner accuracy. However, pure opponent-adjusted remained better on Brier, log loss, and margin MAE.

Decision: not promoted as the leading candidate. Blending was informative but did not beat the opponent-adjusted model on probability quality.

### Recency Weighting

Hypothesis: youth teams evolve quickly, so stronger recency emphasis might improve prediction signal.

Result: mild, moderate, and aggressive recency variants all underperformed the existing opponent-adjusted model. Mild recency was the best recency variant but still dropped to 0.694 accuracy and worsened Brier, log loss, and margin MAE versus opponent-adjusted. Aggressive recency showed overreaction risk.

Decision: rejected. Recency is already present enough in the underlying pipeline; extra decay added instability.

### Matchup Calibration Analysis

Hypothesis: the opponent-adjusted rating engine is strong, but the probability and spread translation may be too conservative.

Result: confirmed. Favorite accuracy was 0.708, but mean favorite probability was only 0.569. Implied favorite spread averaged 1.14 goals while actual favorite margin averaged 3.05 goals. Favorites were generally underpredicted, not inflated.

Decision: retained as a diagnostic finding. It pointed to calibration as the next improvement area.

### Probability / Spread Calibration

Hypothesis: adjusting only the rating-differential-to-probability/spread layer can improve probability quality and spread realism without changing the rating engine.

Result: the best variant used logistic scale 2.0 and margin scale 1.50. It improved Brier to 0.185, log loss to 0.563, margin MAE to 4.723, and expected calibration error to 0.092 while preserving 0.708 winner accuracy.

Decision: retained as the current leading Power V5 candidate.

## Current Leading Candidate

CURRENT LEADING CANDIDATE: opponent-adjusted performance ratings with calibrated probability/spread translation.

Specifically:

- Rating engine: opponent_adjusted_performance
- Probability conversion: logistic scale 2.0
- Implied spread scaling: 1.50x rating differential

Why it leads:

- It keeps the strongest rating engine found so far.
- It fixes the main calibration issue: probabilities were too compressed.
- It improves Brier score and log loss by a meaningful amount.
- It improves margin MAE without reducing winner accuracy.
- It avoids manual tier penalties and avoids extra recency overreaction.

This is still a lab candidate only. No production promotion has occurred yet.

## Known Weaknesses

- Probabilities were historically compressed; the calibration layer improves this but should be validated on future data.
- Spreads remain somewhat conservative even after scaling.
- The youth sports schedule graph is sparse, which limits opponent-strength certainty.
- Inter-tier connectivity is limited, making hidden tier separation hard to infer from regular-season data alone.
- Tournament data has not been used, so cross-tier validation remains incomplete.
- Some teams remain volatile and hard to model because youth roster development can change quickly during the season.

## Future Roadmap

- Probability calibration layer: validate logistic scale 2.0 and margin scale 1.50 on future rolling windows before production promotion.
- Tournament simulation engine: use calibrated probabilities and spreads for bracket simulation once the production model is approved.
- Volatility analysis: add team-level uncertainty so volatile teams receive less extreme confidence.
- Matchup realism testing: continue checking large favorites, upset profiles, and spread errors by team and tier.
- Graph-based ratings: future research only. Explore graph methods for schedule structure, but avoid promoting graph penalties unless they improve rolling backtests.

