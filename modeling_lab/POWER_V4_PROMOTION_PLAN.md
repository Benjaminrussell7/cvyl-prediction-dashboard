# Power V4 Promotion Implementation Plan

Branch: `promote-power-v4`

Scope: planning only. No production code has been changed yet.

## Current Production Flow

### 1. Current Power v3 Generation

Power v3 is generated in `src/cvyl_scraper/power_v3_recency.py` by `build_power_ratings_v3_recency()`.

The normal refresh pipeline calls it from `src/cvyl_scraper/cli.py` and exports:

- `data/processed/cvyl_power_ratings_v3_recency.csv`
- `data/processed/cvyl_model_comparison_v3.csv`
- `data/processed/cvyl_model_comparison_v3_summary.csv`
- `data/processed/cvyl_model_comparison_v4_calibrated.csv`
- `data/processed/cvyl_model_comparison_v4_calibrated_summary.csv`
- `data/processed/cvyl_calibration_power_rating_v4.csv`

The GitHub refresh workflow commits those same processed files in `.github/workflows/refresh-data.yml`.

### 2. Rankings Generation And Loading

Dashboard loading is centralized in `app/streamlit_app.py`:

- `PRIMARY_POWER_RATINGS_FILE = "cvyl_power_ratings_v3_recency.csv"`
- `POWER_RATING_COLUMN = "power_rating_v3_recency"`
- `POWER_RANK_COLUMN = "power_rank_v3_recency"`

Rankings are rendered through:

- `load_dashboard_data()`
- `render_power_rankings()`
- `rankings_display_data()`
- `rankings_display_columns()`
- `power_rating_value()`
- `power_rank_value()`

Secondary ranking consumers:

- `src/cvyl_scraper/historical_snapshots.py` rebuilds weekly historical `power_rank` / `power_rating` from Power v3.
- `src/cvyl_scraper/trends.py` computes current/prior power ranks from Power v3.
- `pages/2_Rankings_and_Ratings.py` delegates main ranking rendering to dashboard helpers.
- `pages/1_Team_Deep_Dive.py` and `pages/3_Model_Insights.py` consume loaded power ranking/history outputs.

### 3. Matchup Predictions

The production matchup screen has two layers:

- Supporting context comes from `cvyl_scraper.prediction.predict_matchup()`, which still uses ELO, Power v2 context, hybrid probability, and scoring-profile projected goals/spread.
- The primary displayed favorite and win probability come from `app/streamlit_app.py::matchup_power_context()`, using `POWER_RATING_COLUMN` and `calibrated_power_v3_probability()`.

Weekly matchup cards also call:

- `build_weekly_matchups()`
- `build_matchup_prediction()`
- `matchup_power_context()`

Important note: current displayed projected spread is not directly the Power v3 rating differential. It comes from scoring-profile projected goals in `cvyl_scraper.prediction`. The lab Power v5 spread calibration is not yet represented in production.

### 4. Tournament Simulator Probability Source

Tournament probabilities are generated in `src/cvyl_scraper/competition.py`:

- `build_competition_simulation()`
- `deterministic_competition_advancement()`
- `monte_carlo_competition_advancement()`
- `matchup_win_probability()`
- `team_power_rating()`

`team_power_rating()` currently hardcodes `power_rating_v3_recency`.

`matchup_win_probability()` applies:

- rating difference from `power_rating_v3_recency`
- `config.scoring_environment_multiplier`
- `calibrated_power_v3_probability()`

The Streamlit tournament page, `pages/4_Tournament_Simulator.py`, passes `data["power_ratings"]` into `build_competition_simulation()`.

## Files Likely To Change For Promotion

### New Or Updated Model Code

Add a production module for the opponent-adjusted model, likely:

- `src/cvyl_scraper/power_v4_opponent_adjusted.py`

It should include:

- `build_power_ratings_v4_opponent_adjusted(team_games)`
- deterministic output columns:
  - `team`
  - `games_played`
  - `avg_actual_margin`
  - `avg_capped_margin`
  - `avg_expected_margin_vs_opponent`
  - `avg_performance_above_expectation`
  - `power_rating_v4`
  - `power_rank_v4`
  - `confidence_tier`
  - `shrinkage_multiplier`
  - any retained offense/defense fields needed by UI cards

Keep `build_power_ratings_v3_recency()` unchanged.

### Pipeline / CLI

Update `src/cvyl_scraper/cli.py` to generate and export:

- `data/processed/cvyl_power_ratings_v4_opponent_adjusted.csv`
- a v4 model comparison/backtest summary file for app metrics
- optionally a calibration file if production UI should show calibration buckets

Keep existing Power v3 outputs exported for rollback and shadow comparison.

### Dashboard Constants And Helpers

Update `app/streamlit_app.py` to point the primary model at v4:

- `PRIMARY_POWER_RATINGS_FILE`
- `PRIMARY_MODEL_COMPARISON_SUMMARY_FILE`
- `PRIMARY_CALIBRATION_FILE`
- `POWER_ACCURACY_KEY`
- `POWER_BRIER_KEY`
- `POWER_RATING_COLUMN`
- `POWER_RANK_COLUMN`

Update fallback column handling in:

- `rankings_display_columns()`
- `render_rankings_table()`
- `power_rating_value()`
- `power_rank_value()`

Use compatibility aliases or fallback order so old v3 files can still render.

### Matchup Probability Calibration

If promoting only opponent-adjusted Power v4, use the current `calibrated_power_v3_probability()` conversion initially.

If promoting the lab Power V5 calibration at the same time, add a separate clearly named production calibration function rather than changing the existing function silently, for example:

- `calibrated_power_v4_probability(rating_difference, scale=2.0, min_probability=..., max_probability=...)`

Do not overload `calibrated_power_v3_probability()` with v4 behavior.

### Tournament Simulator

Update `src/cvyl_scraper/competition.py` so `team_power_rating()` does not hardcode `power_rating_v3_recency`.

Preferred approach:

- define a primary rating column constant, or
- use a fallback column list, e.g. `["power_rating_v4", "power_rating_v3_recency"]`.

This keeps the simulator usable if a rollback file is loaded.

### Historical Snapshots And Trends

Decide whether promotion includes historical/trend recalculation under v4.

If yes:

- update `src/cvyl_scraper/historical_snapshots.py` to build v4 snapshot ranks/ratings.
- update `src/cvyl_scraper/trends.py` to compute rank movement from v4.

If no:

- leave historical/trend outputs as Power v3-derived, but label them clearly in UI to avoid implying they are v4 rankings.

### Workflow And Committed Data

Update `.github/workflows/refresh-data.yml` to commit any new v4 processed outputs.

Likely additions:

- `data/processed/cvyl_power_ratings_v4_opponent_adjusted.csv`
- `data/processed/cvyl_model_comparison_power_v4.csv`
- `data/processed/cvyl_model_comparison_power_v4_summary.csv`
- optional calibration output

### Tests

Add focused tests before switching app defaults:

- model determinism and no future leakage for v4
- v4 output schema and ranking order
- CLI exports v4 files
- dashboard loads v4 columns
- matchup context uses v4 rating/rank columns
- tournament simulator uses v4 rating column
- fallback still works with v3-only data

Likely touched test files:

- `tests/test_power_v3_recency.py` should stay unchanged except shared expectations if needed.
- new `tests/test_power_v4_opponent_adjusted.py`
- `tests/test_streamlit_app.py`
- `tests/test_tournament_simulator_page.py`
- `tests/test_competition.py`
- possibly `tests/test_historical_snapshots.py` and `tests/test_trends.py`

## Suggested Implementation Sequence

1. Add production v4 model module copied from the lab-tested opponent-adjusted logic, with deterministic schema and tests.
2. Add CLI generation/export for v4 while keeping all v3 outputs.
3. Add v4 rolling comparison/summary output for dashboard cards.
4. Update app constants and helper fallback logic to use v4 as primary.
5. Update tournament simulator rating lookup to use v4 with v3 fallback.
6. Decide and implement historical/trend migration or labeling.
7. Update workflow committed files.
8. Run focused tests and then full test suite.
9. Refresh processed data locally and inspect key dashboard/tournament CSVs before merge.

## Rollback Strategy

Rollback should be low-risk if v3 outputs remain in the pipeline.

Keep these unchanged:

- `build_power_ratings_v3_recency()`
- `data/processed/cvyl_power_ratings_v3_recency.csv`
- v3 comparison outputs
- existing v3 tests

Fast rollback options:

1. Revert app constants in `app/streamlit_app.py`:
   - `PRIMARY_POWER_RATINGS_FILE` back to `cvyl_power_ratings_v3_recency.csv`
   - `POWER_RATING_COLUMN` back to `power_rating_v3_recency`
   - `POWER_RANK_COLUMN` back to `power_rank_v3_recency`
   - metric summary/calibration files back to current v3-calibrated files

2. Ensure `competition.team_power_rating()` fallback order includes v3, or switch the primary rating column back to v3.

3. Leave v4 files generated but unused. This allows shadow comparison to continue without affecting production.

4. If data refresh causes issues, remove v4 files from the workflow commit list while keeping v3 refresh intact.

Recommended rollback design during implementation: use centralized constants or a small model config module so rollback is a one-file switch instead of scattered edits.

## Open Decisions Before Coding

- Should this branch promote only the opponent-adjusted Power v4 rating engine, or also the Power V5 calibration layer from lab?
- Should production projected spread remain scoring-profile based, or move to calibrated rating-differential spread?
- Should historical snapshots and trends be recalculated with v4 immediately, or remain v3-derived until a separate migration?
- Should the dashboard expose model version labels so users can see when rankings switch from Power v3 to Power v4?

