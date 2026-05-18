# Power v4 Promotion Implementation

## Rollout Approach

Power v4 is now the default production prediction engine for rankings, matchup probability context, and tournament simulation. The promoted engine is the opponent-adjusted performance model from `modeling_lab`, implemented as production code under `src/cvyl_scraper/` rather than imported from lab scripts.

Power v3 remains intact and continues to generate its existing files. The promotion changes only the default files and rating columns used by production flows.

## Files Changed

- `src/cvyl_scraper/power_v4_opponent_adjusted.py`
  - Added the production Power v4 rating builder.
  - Builds baseline Power v3 first, then rates teams by capped performance above expectation versus opponent strength.
  - Exports `cvyl_power_ratings_v4_opponent_adjusted.csv`.

- `src/cvyl_scraper/model_comparison_power_v4.py`
  - Added rolling backtest comparison for baseline Power v3 vs promoted Power v4.
  - Exports comparison rows, summary metrics, and calibration buckets.

- `src/cvyl_scraper/cli.py`
  - Added generation/export of Power v4 rankings and Power v4 comparison outputs.
  - Preserved all existing Power v3 generation.

- `app/streamlit_app.py`
  - Repointed default production rankings and matchup probability context to Power v4 output columns.
  - Kept Power v3/Power v2 fallback columns for rollback and older files.

- `src/cvyl_scraper/competition.py`
  - Tournament simulator now reads `power_rating_v4` first, then falls back to `power_rating_v3_recency`.
  - Uses the promoted Power v4 probability wrapper.

- `src/cvyl_scraper/probability_calibration.py`
  - Added `calibrated_power_v4_probability` as a semantic wrapper around the existing calibrated probability conversion.

- `src/cvyl_scraper/explanations.py`
  - Matchup explanation power-difference logic now reads Power v4 with Power v3 fallback.

- `pages/1_Team_Deep_Dive.py`
  - Team comparison and schedule difficulty helpers now use Power v4 with Power v3 fallback.

- `.github/workflows/refresh-data.yml`
  - Added the new Power v4 processed outputs to the refresh commit list.

- `tests/test_power_v4_opponent_adjusted.py`
  - Added focused tests for Power v4 schema, ranking output, v3 preservation, and opponent-adjusted behavior.

- `tests/test_model_comparison_power_v4.py`
  - Added rolling comparison tests for determinism, leakage prevention, summary metrics, calibration output, and v3 preservation.

- `tests/test_streamlit_app.py`
  - Updated dashboard processed-data contract expectations for the new Power v4 production files.

- `tests/test_competition.py`
  - Added coverage that tournament rating lookup prefers Power v4 and falls back to Power v3.

## New Production Outputs

- `data/processed/cvyl_power_ratings_v4_opponent_adjusted.csv`
- `data/processed/cvyl_model_comparison_power_v4.csv`
- `data/processed/cvyl_model_comparison_power_v4_summary.csv`
- `data/processed/cvyl_calibration_power_v4_opponent_adjusted.csv`

## Validation

Generated Power v4 outputs from the existing processed canonical data:

- Power v4 rating rows: `50`
- Rolling backtest games: `144`
- Winner accuracy: unchanged vs Power v3 at `0.7083`
- Brier score: improved from `0.2067` to `0.1968`
- Log loss: improved from `0.6144` to `0.5919`
- Margin MAE: improved from `4.9013` to `4.8687`

Sanity checks completed:

- Power v4 rankings generated successfully.
- Matchup probability context loads Power v4 columns successfully.
- Tournament simulation works with Power v4 ratings and preserves Power v3 fallback behavior.
- Changed Python files compile successfully with `python -m compileall`.
- CLI argument wiring loads successfully with the repo `.venv`.
- Full test suite passed: `265 passed`.

## Rollback Strategy

Rollback does not require deleting code.

1. Repoint `app/streamlit_app.py` constants back to:
   - `cvyl_power_ratings_v3_recency.csv`
   - `cvyl_model_comparison_v4_calibrated_summary.csv`
   - `cvyl_calibration_power_rating_v4.csv`
   - `power_rating_v3_recency`
   - `power_rank_v3_recency`
   - `power_v3_calibrated_accuracy`
   - `power_v3_calibrated_brier_score`
2. In `src/cvyl_scraper/competition.py`, put `power_rating_v3_recency` first in `POWER_RATING_COLUMNS`.
3. Leave Power v4 output generation in place or disable it later after confirming rollback stability.

Power v3 builders, comparison scripts, and processed outputs remain available.

## Known Remaining Limitations

- The probability layer is still the existing calibrated logistic conversion; Power v5 calibration work remains a separate candidate.
- Implied spreads are rating-difference based and may remain conservative for large favorites.
- Youth lacrosse schedule graphs are sparse, with limited inter-tier connectivity.
- No tournament data is used in this promotion.
- Historical snapshots and trend files remain based on the existing Power v3-era trend pipeline unless separately migrated.

No UI redesign or user-facing model toggle was introduced.
