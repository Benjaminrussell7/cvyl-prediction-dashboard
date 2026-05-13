# CVYL Schedule Scraper

Python project for scraping CVYL lacrosse schedules and scores into a cleaned, canonical one-row-per-game CSV.

This project intentionally stops at data collection and cleaning. It does not build prediction models, and core logic lives in importable Python modules rather than notebooks.

## Project Layout

```text
.
├── config/
│   └── sources.example.yml
├── data/
│   ├── processed/
│   └── raw/
├── src/
│   └── cvyl_scraper/
│       ├── __init__.py
│       ├── cli.py
│       ├── cleaning.py
│       ├── config.py
│       ├── export.py
│       ├── models.py
│       ├── parsing.py
│       └── scraping.py
├── tests/
│   └── test_cleaning.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configure Sources

Copy the example config and replace the placeholder URLs with real CVYL schedule pages.

```bash
cp config/sources.example.yml config/sources.yml
```

Example:

```yaml
sources:
  - name: cvyl_boys_14u
    url: "https://example.com/schedule"
    season: 2026
    division: "14U Boys"
```

## Run

```bash
python -m cvyl_scraper.cli \
  --config config/sources.yml \
  --output data/processed/cvyl_games.csv
```

The exported CSV contains one canonical row per game with normalized team names, dates, scores, game status, and source metadata.

## How to Run the Dashboard Locally

After generating the processed CSV outputs, run:

```bash
streamlit run app/streamlit_app.py
```

The dashboard reads existing files from `data/processed/` and shows model summary metrics, power rankings, matchup predictions, team details, and backtest results.

## Automated Data Refresh

GitHub Actions refreshes the processed CVYL data twice daily, once in the morning and once in the evening, and can also be run manually from the Actions tab.

The workflow is defined in `.github/workflows/refresh-data.yml`. It installs dependencies from `requirements.txt`, runs:

```bash
python -m cvyl_scraper.cli --config config/discovered_sources.yml --team-aliases config/team_aliases.yml
```

and commits updated processed CSV outputs back to `data/processed/` when the pipeline produces changes. The Streamlit dashboard reads those committed CSVs, so a successful refresh updates the deployed dashboard data after the repository changes are picked up by Streamlit Cloud.

## Known Limitations

- Predictions and rankings are based only on scores currently reported on CVYL.org.
- Some recent games may be missing if scores have not been posted yet.
- Team identity depends on the canonical names and any explicit aliases configured locally.
- The matchup projection uses simple ELO, scoring averages, SOS, and backtest outputs; it is not a machine learning model.

## Output Columns

```text
game_date, game_time, season, division, home_team, away_team,
home_score, away_score, status, source_name, source_url, game_id
```

`status` is either:

- `completed`: both scores are available
- `scheduled`: scores are missing

Mirrored duplicate rows are collapsed by a stable game key based on date, teams, division, season, and available score/status information.
