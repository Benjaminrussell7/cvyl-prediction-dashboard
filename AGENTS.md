# AGENTS.md

Guidance for agents working in this repository.

## Project Context

This is a CVYL lacrosse analytics and prediction project. The current package scrapes and cleans CVYL/Crossbar schedule and score pages into a canonical one-row-per-game table; future modeling work should build on that canonical data without changing its meaning.

## Architecture

- Maintain a modular architecture under `src/cvyl_scraper/`.
- Prefer explicit, purpose-named modules over dumping unrelated helpers into a generic `utils.py`.
- Keep core logic in importable Python modules. Avoid notebooks for scraping, cleaning, feature generation, training, evaluation, or prediction logic.
- Use pandas for dataframe operations unless there is a clear project-level reason to add another dataframe dependency.
- Keep code readable and simple. Do not introduce premature optimization, broad abstractions, or framework machinery before the code needs it.

## Data Boundaries

- Preserve the canonical games table as one row per game.
- Keep the canonical games table separate from modeling tables, feature tables, predictions, and evaluation outputs.
- Do not add mirrored team-perspective rows to the canonical games table. If modeling needs team-game rows, create them in a separate modeling module/table.
- Preserve deterministic `game_id` generation. Any change to ID logic must be intentional, documented in code or tests, and covered by tests.
- Treat raw HTML, canonical cleaned data, modeling features, and predictions as distinct layers.

## Leakage Rules

- Never allow future leakage in features, labels, evaluation splits, or predictions.
- Prediction-time features may only use information that would have been available before the game being predicted.
- Rolling, aggregate, or team-history features must be computed using strictly prior games.
- Do not use final scores, game outcomes, post-game statuses, playoff results, or later-season aggregates as inputs for pre-game prediction rows.
- Tests for feature-generation or prediction code should include cases that catch accidental inclusion of the target game or future games.

## Parsing and Scraping

- Keep parsers resilient to CVYL/Crossbar HTML changes. Prefer semantic structure and tolerant extraction over brittle exact formatting assumptions.
- Parser changes should handle missing fields, extra whitespace, reordered classes, and scheduled games without scores.
- Add or update tests for every parser behavior change. Include representative HTML snippets in `tests/test_parsing.py` or a clearly named parser test file.
- Do not let scraping concerns leak into cleaning or modeling logic. Fetching, parsing, cleaning, exporting, feature building, and prediction should remain separate responsibilities.

## Testing

- Run focused tests for touched areas before finishing. For parser changes, run at least:

```bash
pytest tests/test_parsing.py
```

- For cleaning or canonical table changes, run relevant cleaning tests and any tests that assert output columns, row counts, or `game_id` behavior.
- Add tests before or alongside behavior changes when the expected behavior is subtle, especially for duplicate collapsing, date/team normalization, leakage prevention, and deterministic IDs.

## Style

- Keep functions small and explicit.
- Prefer clear names over cleverness.
- Make dataframe schemas obvious through constants, tests, or localized column selection.
- Avoid hidden mutation of shared dataframes; return new dataframes or document intentional in-place behavior clearly.
- Do not mix unrelated refactors into feature or bug-fix changes.
