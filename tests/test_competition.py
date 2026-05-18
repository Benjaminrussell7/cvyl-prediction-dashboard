from __future__ import annotations

import pandas as pd
import pytest
import yaml

from cvyl_scraper.competition import (
    COMPETITION_SIMULATION_COLUMNS,
    CompetitionConfig,
    MatchupConfig,
    BracketRoundConfig,
    build_seeded_competition_config,
    build_competition_simulation,
    competition_config_from_dict,
    competition_config_to_dict,
    deterministic_competition_advancement,
    load_competition_config,
    matchup_win_probability,
    monte_carlo_competition_advancement,
    safe_competition_id,
    save_competition_config,
    team_power_rating,
    validate_builder_inputs,
    validate_competition_config,
)


def _config(*, completed_winner: str | None = None, multiplier: float = 1.0) -> CompetitionConfig:
    return CompetitionConfig(
        competition_name="CVYL Playoff Test",
        competition_type="playoffs",
        division_name="Division A",
        teams=["Avon", "Granby", "RHAM", "Simsbury"],
        seeds={"Avon": 1, "Granby": 4, "RHAM": 2, "Simsbury": 3},
        game_format="standard" if multiplier == 1.0 else "running_time",
        game_minutes=48 if multiplier == 1.0 else 32,
        scoring_environment_multiplier=multiplier,
        bracket_rounds=[
            BracketRoundConfig(
                name="Semifinals",
                matchups=[
                    MatchupConfig("sf1", "Avon", "Granby", completed_winner=completed_winner),
                    MatchupConfig("sf2", "RHAM", "Simsbury"),
                ],
            ),
            BracketRoundConfig(
                name="Final",
                matchups=[MatchupConfig("final", "Avon", "RHAM")],
            ),
        ],
    )


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "Avon", "power_rating_v3_recency": 3.0},
            {"team": "Granby", "power_rating_v3_recency": -1.0},
            {"team": "RHAM", "power_rating_v3_recency": 1.2},
            {"team": "Simsbury", "power_rating_v3_recency": 0.5},
        ]
    )


def test_team_power_rating_prefers_power_v4_and_falls_back_to_v3() -> None:
    ratings = pd.DataFrame(
        [
            {
                "team": "Avon",
                "power_rating_v4": 2.5,
                "power_rating_v3_recency": 1.0,
            },
            {"team": "Granby", "power_rating_v3_recency": -0.5},
        ]
    )

    assert team_power_rating("Avon", ratings) == 2.5
    assert team_power_rating("Granby", ratings) == -0.5


def test_load_competition_config_from_yaml(tmp_path) -> None:
    path = tmp_path / "competition.yml"
    payload = {
        "competition_name": "Weekend Tournament",
        "competition_type": "tournament",
        "division_name": "U12 Blue",
        "teams": ["Avon", "Granby"],
        "seeds": {"Avon": 1, "Granby": 2},
        "game_format": "running_time",
        "game_minutes": 32,
        "scoring_environment_multiplier": 0.75,
        "bracket_rounds": [
            {
                "name": "Final",
                "matchups": [{"matchup_id": "f1", "team_a": "Avon", "team_b": "Granby"}],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_competition_config(path)

    assert config.competition_name == "Weekend Tournament"
    assert config.competition_type == "tournament"
    assert config.scoring_environment_multiplier == 0.75
    assert config.bracket_rounds[0].matchups[0].matchup_id == "f1"


def test_validate_competition_config_errors_on_unknown_matchup_team() -> None:
    config = CompetitionConfig(
        competition_name="Bad",
        competition_type="playoffs",
        division_name="Division A",
        teams=["Avon"],
        seeds={"Avon": 1},
        bracket_rounds=[
            BracketRoundConfig(
                name="Final",
                matchups=[MatchupConfig("f1", "Avon", "Missing")],
            )
        ],
    )

    with pytest.raises(ValueError, match="unknown team"):
        validate_competition_config(config)


def test_competition_config_from_dict_validates_completed_winner() -> None:
    payload = {
        "competition_name": "Bad Winner",
        "competition_type": "playoffs",
        "division_name": "Division A",
        "teams": ["Avon", "Granby"],
        "seeds": {"Avon": 1, "Granby": 2},
        "bracket_rounds": [
            {
                "name": "Final",
                "matchups": [
                    {
                        "matchup_id": "f1",
                        "team_a": "Avon",
                        "team_b": "Granby",
                        "completed_winner": "RHAM",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="completed_winner"):
        validate_competition_config(competition_config_from_dict(payload))


def test_deterministic_advancement_uses_probabilities_and_columns() -> None:
    rows, winners = deterministic_competition_advancement(_config(), _ratings())

    assert list(rows.columns) == COMPETITION_SIMULATION_COLUMNS
    assert winners["sf1"] == "Avon"
    assert winners["sf2"] == "RHAM"
    assert rows.loc[rows["matchup_id"] == "sf1", "team_a_win_probability"].iloc[0] > 0.5


def test_persisted_winner_overrides_expected_advancement() -> None:
    rows, winners = deterministic_competition_advancement(_config(completed_winner="Granby"), _ratings())

    semifinal = rows[rows["matchup_id"] == "sf1"].iloc[0]

    assert winners["sf1"] == "Granby"
    assert semifinal["completed_winner"] == "Granby"
    assert semifinal["team_a_win_probability"] > 0.5


def test_monte_carlo_simulation_is_repeatable_with_fixed_seed() -> None:
    first = monte_carlo_competition_advancement(_config(), _ratings(), simulations=50, random_seed=7)
    second = monte_carlo_competition_advancement(_config(), _ratings(), simulations=50, random_seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["champion"].isin([True, False]).all()


def test_game_format_multiplier_reduces_probability_confidence() -> None:
    standard = matchup_win_probability(_config(multiplier=1.0), "Avon", "Granby", _ratings())
    shorter = matchup_win_probability(_config(multiplier=0.75), "Avon", "Granby", _ratings())

    assert standard > shorter > 0.5


def test_missing_team_rating_is_graceful() -> None:
    ratings = _ratings()[_ratings()["team"] != "Granby"]

    rows, _ = deterministic_competition_advancement(_config(), ratings)
    semifinal = rows[rows["matchup_id"] == "sf1"].iloc[0]

    assert semifinal["team_a_win_probability"] == 0.5
    assert "Granby" in semifinal["note"]


def test_build_competition_simulation_summary_is_deterministic() -> None:
    first_sim, first_summary = build_competition_simulation(_config(), _ratings(), simulations=50, random_seed=11)
    second_sim, second_summary = build_competition_simulation(_config(), _ratings(), simulations=50, random_seed=11)

    pd.testing.assert_frame_equal(first_sim, second_sim)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert first_summary["championship_probability"].between(0, 1).all()
    assert first_summary["round_advancement_probabilities"].astype(str).str.len().gt(0).any()
    assert "Avon" in set(first_summary["team"])


def test_save_competition_config_round_trips_yaml(tmp_path) -> None:
    path = tmp_path / "saved.yml"

    saved_path = save_competition_config(_config(), path)
    loaded = load_competition_config(saved_path)

    assert loaded == _config()
    assert competition_config_to_dict(loaded)["competition_name"] == "CVYL Playoff Test"


def test_build_seeded_competition_config_creates_simple_bracket() -> None:
    config = build_seeded_competition_config(
        competition_name="Builder Test",
        competition_type="Playoffs",
        division_name="Division A",
        seeded_teams=[("Avon", 1), ("Granby", 4), ("RHAM", 2), ("Simsbury", 3)],
    )

    assert config.teams == ["Avon", "RHAM", "Simsbury", "Granby"]
    assert config.bracket_rounds[0].name == "Semifinals"
    assert config.bracket_rounds[0].matchups[0].team_a == "Avon"
    assert config.bracket_rounds[0].matchups[0].team_b == "Granby"
    assert config.bracket_rounds[1].name == "Championship"


def test_validate_builder_inputs_rejects_bad_seed_and_team_data() -> None:
    with pytest.raises(ValueError, match="duplicate teams"):
        validate_builder_inputs(
            competition_name="Bad",
            competition_type="playoffs",
            division_name="Division A",
            seeded_teams=[("Avon", 1), ("Avon", 2)],
            game_minutes=48,
        )

    with pytest.raises(ValueError, match="seeds"):
        validate_builder_inputs(
            competition_name="Bad",
            competition_type="playoffs",
            division_name="Division A",
            seeded_teams=[("Avon", 1), ("Granby", 1)],
            game_minutes=48,
        )

    with pytest.raises(ValueError, match="game length"):
        validate_builder_inputs(
            competition_name="Bad",
            competition_type="playoffs",
            division_name="Division A",
            seeded_teams=[("Avon", 1), ("Granby", 2)],
            game_minutes=0,
        )


def test_safe_competition_id_is_filename_friendly() -> None:
    assert safe_competition_id("2026 CVYL Boys / Division A!") == "2026_cvyl_boys_division_a"
