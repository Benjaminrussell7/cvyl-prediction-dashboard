from __future__ import annotations

import importlib.util
from pathlib import Path

import altair as alt
import pandas as pd

from cvyl_scraper.competition import BracketRoundConfig, CompetitionConfig, MatchupConfig

PAGE_PATH = Path(__file__).resolve().parents[1] / "pages" / "4_Tournament_Simulator.py"
SPEC = importlib.util.spec_from_file_location("tournament_simulator_page", PAGE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
tournament_page = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tournament_page)


def _config() -> CompetitionConfig:
    return CompetitionConfig(
        competition_name="Playoff Preview",
        competition_type="playoffs",
        division_name="Division A",
        teams=["Avon", "Granby", "RHAM", "Simsbury"],
        seeds={"Avon": 1, "Granby": 4, "RHAM": 2, "Simsbury": 3},
        bracket_rounds=[
            BracketRoundConfig(
                "Semifinals",
                [
                    MatchupConfig("sf1", "Avon", "Granby"),
                    MatchupConfig("sf2", "RHAM", "Simsbury", completed_winner="Simsbury"),
                ],
            ),
            BracketRoundConfig("Final", [MatchupConfig("final", "Avon", "RHAM")]),
        ],
    )


def _simulation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "round": "Semifinals",
                "matchup_id": "sf1",
                "team_a": "Avon",
                "team_b": "Granby",
                "team_a_win_probability": 0.78,
                "team_b_win_probability": 0.22,
                "expected_winner": "Avon",
                "completed_winner": None,
                "upset_likelihood": 0.22,
                "note": "",
            },
            {
                "round": "Semifinals",
                "matchup_id": "sf2",
                "team_a": "RHAM",
                "team_b": "Simsbury",
                "team_a_win_probability": 0.54,
                "team_b_win_probability": 0.46,
                "expected_winner": "Simsbury",
                "completed_winner": "Simsbury",
                "upset_likelihood": 0.46,
                "note": "",
            },
            {
                "round": "Final",
                "matchup_id": "final",
                "team_a": "Avon",
                "team_b": "RHAM",
                "team_a_win_probability": 0.62,
                "team_b_win_probability": 0.38,
                "expected_winner": "Avon",
                "completed_winner": None,
                "upset_likelihood": 0.38,
                "note": "",
            },
        ]
    )


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Avon",
                "seed": 1,
                "championship_probability": 0.58,
                "round_advancement_probabilities": "Semifinals: 78.0%; Final: 62.0%",
                "expected_advancement_round": "Final",
                "most_likely_finalist": True,
            },
            {
                "team": "RHAM",
                "seed": 2,
                "championship_probability": 0.24,
                "round_advancement_probabilities": "Semifinals: 54.0%; Final: 38.0%",
                "expected_advancement_round": "Semifinals",
                "most_likely_finalist": True,
            },
            {
                "team": "Simsbury",
                "seed": 3,
                "championship_probability": 0.12,
                "round_advancement_probabilities": "Semifinals: 46.0%",
                "expected_advancement_round": "",
                "most_likely_finalist": False,
            },
        ]
    )


def test_available_competition_configs_filters_yaml_files(tmp_path) -> None:
    (tmp_path / "a.yml").write_text("competition_name: A", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("competition_name: B", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")

    configs = tournament_page.available_competition_configs(tmp_path)

    assert [path.name for path in configs] == ["a.yml", "b.yaml"]


def test_competition_config_status_is_visible() -> None:
    assert tournament_page.competition_config_status([]) == "No saved competition configs found."
    assert tournament_page.competition_config_status([Path("one.yml")]) == "1 saved competition config found."
    assert tournament_page.competition_config_status(
        [Path("one.yml"), Path("two.yml")]
    ) == "2 saved competition configs found."


def test_builder_input_helpers_create_valid_config() -> None:
    config = tournament_page.competition_config_from_builder_inputs(
        competition_name="Builder Cup",
        competition_type_label="Tournament",
        division_name="U12 Blue",
        selected_teams=["Avon", "Granby", "RHAM", "Simsbury"],
        seed_values={"Avon": 1, "Granby": 4, "RHAM": 2, "Simsbury": 3},
        game_format_label="Running-time game",
        game_minutes=32,
    )

    assert config.competition_type == "tournament"
    assert config.game_format == "running_time"
    assert config.scoring_environment_multiplier == 0.75
    assert config.bracket_rounds[0].name == "Semifinals"


def test_builder_initial_teams_and_seeds_use_loaded_config() -> None:
    config = _config()

    teams, seeds = tournament_page.builder_initial_teams_and_seeds(
        config,
        ["Avon", "Granby", "RHAM", "Simsbury", "Other"],
    )

    assert teams == ["Avon", "Granby", "RHAM", "Simsbury"]
    assert seeds == {"Avon": 1, "Granby": 4, "RHAM": 2, "Simsbury": 3}


def test_reconcile_seed_values_keeps_existing_and_assigns_new_teams() -> None:
    selected = ["Avon", "RHAM", "New Team"]
    existing = {"Avon": 1, "Granby": 4, "RHAM": 2}

    seeds = tournament_page.reconcile_seed_values(selected, existing)

    assert seeds == {"Avon": 1, "RHAM": 2, "New Team": 3}


def test_reconcile_seed_values_removes_unselected_and_repairs_duplicate_existing_seeds() -> None:
    selected = ["Avon", "Granby", "RHAM"]
    existing = {"Avon": 1, "Granby": 1, "RHAM": 5, "Removed": 2}

    seeds = tournament_page.reconcile_seed_values(selected, existing)

    assert seeds == {"Avon": 1, "Granby": 2, "RHAM": 3}


def test_validate_seed_values_reports_dynamic_builder_errors() -> None:
    assert tournament_page.validate_seed_values(["Avon"], {"Avon": 1}) == ["Select at least two teams."]
    duplicate_seed_errors = tournament_page.validate_seed_values(
        ["Avon", "Granby"],
        {"Avon": 1, "Granby": 1},
    )
    missing_seed_errors = tournament_page.validate_seed_values(
        ["Avon", "Granby"],
        {"Avon": 1},
    )

    assert "Duplicate seeds are not allowed." in duplicate_seed_errors
    assert "Missing seed values for: Granby." in missing_seed_errors


def test_builder_seed_keys_are_team_specific_and_stable() -> None:
    config = _config()

    assert tournament_page.builder_seed_key(config, "Avon 12U") == tournament_page.builder_seed_key(config, "Avon 12U")
    assert tournament_page.builder_seed_key(config, "Avon 12U") != tournament_page.builder_seed_key(config, "Granby 12U")


def test_reconcile_seed_state_preserves_and_cleans_session_state(monkeypatch) -> None:
    state = {
        tournament_page.builder_seed_key(None, "Avon"): 2,
        tournament_page.builder_seed_key(None, "Removed"): 1,
    }
    monkeypatch.setattr(tournament_page.st, "session_state", state)

    seeds = tournament_page.reconcile_seed_state(["Avon", "Granby"], {"Avon": 1}, None)

    assert seeds == {"Avon": 2, "Granby": 1}
    assert tournament_page.builder_seed_key(None, "Removed") not in state
    assert state[tournament_page.builder_seed_key(None, "Avon")] == 2
    assert state[tournament_page.builder_seed_key(None, "Granby")] == 1


def test_reconcile_seed_state_is_idempotent(monkeypatch) -> None:
    class TrackingState(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.writes = 0
            self.deletes = 0

        def __setitem__(self, key, value):
            self.writes += 1
            super().__setitem__(key, value)

        def __delitem__(self, key):
            self.deletes += 1
            super().__delitem__(key)

    state = TrackingState()
    monkeypatch.setattr(tournament_page.st, "session_state", state)

    first = tournament_page.reconcile_seed_state(["Avon", "Granby"], {"Avon": 1}, None)
    writes_after_first = state.writes
    deletes_after_first = state.deletes
    second = tournament_page.reconcile_seed_state(["Avon", "Granby"], {"Avon": 1}, None)

    assert first == second == {"Avon": 1, "Granby": 2}
    assert state.writes == writes_after_first
    assert state.deletes == deletes_after_first


def test_render_reactive_seed_controls_creates_control_for_each_selected_team(monkeypatch) -> None:
    calls: list[str] = []
    state: dict[str, int] = {}

    class FakeColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_columns(count):
        return [FakeColumn() for _ in range(count)]

    def fake_number_input(label, **kwargs):
        calls.append(label)
        return kwargs["value"]

    monkeypatch.setattr(tournament_page.st, "session_state", state)
    monkeypatch.setattr(tournament_page.st, "columns", fake_columns)
    monkeypatch.setattr(tournament_page.st, "number_input", fake_number_input)

    seeds = tournament_page.render_reactive_seed_controls(
        ["Avon", "Granby", "RHAM", "Simsbury", "Windsor"],
        {},
        None,
    )

    assert len(calls) == 5
    assert calls[-1] == "Seed: Windsor"
    assert seeds == {"Avon": 1, "Granby": 2, "RHAM": 3, "Simsbury": 4, "Windsor": 5}


def test_builder_helpers_fall_back_when_competition_module_lacks_new_helpers(monkeypatch) -> None:
    monkeypatch.delattr(tournament_page.competition, "build_seeded_competition_config", raising=False)
    monkeypatch.delattr(tournament_page.competition, "safe_competition_id", raising=False)
    monkeypatch.delattr(tournament_page.competition, "save_competition_config", raising=False)
    monkeypatch.delattr(tournament_page.competition, "competition_config_to_dict", raising=False)

    config = tournament_page.build_seeded_competition_config(
        competition_name="Fallback Cup",
        competition_type="playoffs",
        division_name="U12",
        seeded_teams=[("Avon", 1), ("Granby", 2)],
    )

    assert tournament_page.safe_competition_id("Fallback Cup!") == "fallback_cup"
    assert config.competition_name == "Fallback Cup"
    assert config.bracket_rounds[0].name == "Championship"
    assert tournament_page.competition_config_to_dict(config)["teams"] == ["Avon", "Granby"]


def test_builder_save_fallback_writes_valid_yaml(monkeypatch, tmp_path) -> None:
    monkeypatch.delattr(tournament_page.competition, "save_competition_config", raising=False)
    monkeypatch.delattr(tournament_page.competition, "competition_config_to_dict", raising=False)
    config = tournament_page.build_seeded_competition_config(
        competition_name="Fallback Save",
        competition_type="playoffs",
        division_name="U12",
        seeded_teams=[("Avon", 1), ("Granby", 2)],
    )

    path = tournament_page.save_competition_config(config, tmp_path / "fallback.yml")
    loaded = tournament_page.load_competition_config(path)

    assert loaded.competition_name == "Fallback Save"
    assert loaded.teams == ["Avon", "Granby"]


def test_builder_path_and_team_options_are_safe(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tournament_page, "COMPETITIONS_DIR", tmp_path)

    path = tournament_page.competition_config_path("../Bad Name")
    teams = tournament_page.league_team_options(
        {
            "power_ratings": pd.DataFrame([{"team": "Granby"}, {"team": "Avon"}]),
            "ratings": pd.DataFrame([{"team": "Fallback"}]),
        }
    )

    assert path == tmp_path / "Bad Name.yml"
    assert teams == ["Avon", "Granby"]


def test_game_format_and_type_mapping() -> None:
    assert tournament_page.competition_type_from_label("Tournament") == "tournament"
    assert tournament_page.competition_type_from_label("Playoffs") == "playoffs"
    assert tournament_page.game_format_from_label("Running-time game") == "running_time"
    assert tournament_page.game_format_from_label("Standard game") == "standard"
    assert tournament_page.scoring_multiplier_from_game_format("Running-time game") == 0.75
    assert tournament_page.scoring_multiplier_from_game_format("Standard game") == 1.0


def test_save_builder_submission_handles_overwrite_and_validation(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(tournament_page, "COMPETITIONS_DIR", tmp_path)
    monkeypatch.setattr(tournament_page.st, "success", lambda text: calls.append(("success", text)))
    monkeypatch.setattr(tournament_page.st, "error", lambda text: calls.append(("error", text)))

    saved = tournament_page.save_builder_submission(
        competition_name="Builder Cup",
        competition_type_label="Playoffs",
        division_name="U12",
        selected_teams=["Avon", "Granby"],
        seed_values={"Avon": 1, "Granby": 2},
        game_format_label="Standard game",
        game_minutes=48,
        filename="builder_cup.yml",
        overwrite=True,
        simulation_depth=1000,
        random_seed=42,
    )
    blocked = tournament_page.save_builder_submission(
        competition_name="Builder Cup",
        competition_type_label="Playoffs",
        division_name="U12",
        selected_teams=["Avon", "Granby"],
        seed_values={"Avon": 1, "Granby": 2},
        game_format_label="Standard game",
        game_minutes=48,
        filename="builder_cup.yml",
        overwrite=False,
        simulation_depth=1000,
        random_seed=42,
    )
    invalid = tournament_page.save_builder_submission(
        competition_name="Bad",
        competition_type_label="Playoffs",
        division_name="U12",
        selected_teams=["Avon"],
        seed_values={"Avon": 1},
        game_format_label="Standard game",
        game_minutes=48,
        filename="bad.yml",
        overwrite=True,
        simulation_depth=1000,
        random_seed=42,
    )

    assert saved == tmp_path / "builder_cup.yml"
    assert blocked is None
    assert invalid is None
    assert any(kind == "success" for kind, _text in calls)
    assert sum(1 for kind, _text in calls if kind == "error") == 2


def test_main_renders_basic_shell_when_no_configs(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(tournament_page, "COMPETITIONS_DIR", tmp_path)
    monkeypatch.setattr(tournament_page.dashboard, "render_data_freshness", lambda: calls.append(("freshness", "")))
    monkeypatch.setattr(tournament_page.st, "title", lambda text: calls.append(("title", text)))
    monkeypatch.setattr(tournament_page.st, "caption", lambda text: calls.append(("caption", text)))
    monkeypatch.setattr(tournament_page.st, "divider", lambda: calls.append(("divider", "")))
    monkeypatch.setattr(tournament_page, "render_no_configs_message", lambda: calls.append(("fallback", "")))
    monkeypatch.setattr(tournament_page.dashboard, "load_dashboard_data", lambda: {"power_ratings": pd.DataFrame()})
    monkeypatch.setattr(tournament_page, "render_competition_builder", lambda data, configs: calls.append(("builder", "")))

    tournament_page.main()

    assert ("title", "Tournament Simulator") in calls
    assert any(kind == "caption" and "Competition config status" in text for kind, text in calls)
    assert ("builder", "") in calls
    assert ("fallback", "") in calls


def test_competition_overview_helpers() -> None:
    config = _config()
    seeds = tournament_page.seed_table(config)

    assert tournament_page.persisted_winner_count(config) == 1
    assert seeds["team"].tolist() == ["Avon", "RHAM", "Simsbury", "Granby"]


def test_tournament_outlook_cards_include_core_storylines() -> None:
    cards = tournament_page.tournament_outlook_cards(_config(), _simulation(), _summary())
    labels = {card["label"] for card in cards}

    assert "Title Favorite" in labels
    assert "Most Likely Finalists" in labels
    assert "Upset Watch" in labels
    assert "Tightest Matchup" in labels
    assert cards == tournament_page.tournament_outlook_cards(_config(), _simulation(), _summary())


def test_matchup_selectors_identify_risk_favorite_and_tight_match() -> None:
    simulation = _simulation()

    assert tournament_page.highest_upset_risk_matchup(simulation)["matchup_id"] == "sf2"
    assert tournament_page.strongest_favorite_matchup(simulation)["matchup_id"] == "sf1"
    assert tournament_page.tightest_projected_matchup(simulation)["matchup_id"] == "sf2"


def test_first_unresolved_round_skips_completed_rounds_when_needed() -> None:
    config = CompetitionConfig(
        competition_name="Completed Semis",
        competition_type="playoffs",
        division_name="Division A",
        teams=["Avon", "Granby", "RHAM", "Simsbury"],
        seeds={},
        bracket_rounds=[
            BracketRoundConfig(
                "Semifinals",
                [
                    MatchupConfig("sf1", "Avon", "Granby", completed_winner="Avon"),
                    MatchupConfig("sf2", "RHAM", "Simsbury", completed_winner="RHAM"),
                ],
            ),
            BracketRoundConfig("Final", [MatchupConfig("final", "Avon", "RHAM")]),
        ],
    )

    assert tournament_page.first_unresolved_round(config).name == "Final"


def test_parse_round_advancement_and_charts_compile() -> None:
    parsed = tournament_page.parse_round_advancement("Semifinals: 78.0%; Final: 62.0%")

    assert parsed == {"Semifinals": 0.78, "Final": 0.62}
    assert isinstance(tournament_page.championship_probability_chart(_summary()), alt.Chart)
    assert isinstance(tournament_page.advancement_probability_chart(_summary()), alt.Chart)


def test_matchup_outlook_sentence_handles_completed_and_upset_watch() -> None:
    completed = _simulation().iloc[1]
    upset = _simulation().iloc[2]

    assert "already recorded" in tournament_page.matchup_outlook_sentence(completed, 0.54, False)
    assert "lower-seed push" in tournament_page.matchup_outlook_sentence(upset, 0.62, True)


def test_story_badge_falls_back_when_dashboard_helper_missing(monkeypatch) -> None:
    monkeypatch.delattr(tournament_page.dashboard, "story_badge", raising=False)

    first = tournament_page.story_badge("Upset Watch")
    second = tournament_page.story_badge("Upset Watch")

    assert first == second
    assert "Upset Watch" in first
    assert "#fef3c7" in first


def test_edge_helpers_fall_back_when_dashboard_helpers_missing(monkeypatch) -> None:
    monkeypatch.delattr(tournament_page.dashboard, "prediction_edge_label", raising=False)
    monkeypatch.delattr(tournament_page.dashboard, "edge_badge", raising=False)

    label = tournament_page.prediction_edge_label(0.72)
    badge = tournament_page.edge_badge(label)

    assert label == "Solid Favorite"
    assert "Solid Favorite" in badge


def test_matchup_preview_card_renders_without_shared_badge_helpers(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeColumn:
        def metric(self, label, value):
            calls.append((str(label), value))

    class FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.delattr(tournament_page.dashboard, "story_badge", raising=False)
    monkeypatch.delattr(tournament_page.dashboard, "edge_badge", raising=False)
    monkeypatch.delattr(tournament_page.dashboard, "prediction_edge_label", raising=False)
    monkeypatch.setattr(tournament_page.st, "container", lambda **kwargs: FakeContainer())
    monkeypatch.setattr(tournament_page.st, "markdown", lambda *args, **kwargs: calls.append(("markdown", args[0])))
    monkeypatch.setattr(tournament_page.st, "caption", lambda text: calls.append(("caption", text)))
    monkeypatch.setattr(tournament_page.st, "warning", lambda text: calls.append(("warning", text)))
    monkeypatch.setattr(tournament_page.st, "columns", lambda count: [FakeColumn() for _ in range(count)])

    tournament_page.render_matchup_preview_card(_simulation().iloc[0])

    assert ("Favorite", "Avon") in calls
    assert any(kind == "markdown" and "Playoff Edge" in str(value) for kind, value in calls)


def test_most_likely_path_story_and_toughest_path_are_deterministic() -> None:
    first = tournament_page.most_likely_path_story(_config(), _simulation(), _summary())
    second = tournament_page.most_likely_path_story(_config(), _simulation(), _summary())
    tough = tournament_page.toughest_projected_path_card(_config(), _simulation(), _summary())

    assert first == second
    assert "Avon" in first
    assert tough is not None
    assert tough["label"] == "Toughest Projected Path"


def test_roadblock_cards_include_upset_and_lower_seed() -> None:
    cards = tournament_page.roadblock_cards(_config(), _simulation(), _summary())
    labels = {card["label"] for card in cards}

    assert "Potential Upset Zone" in labels
    assert "Most Dangerous Lower Seed" in labels


def test_championship_favorite_cards_include_compact_bar_data() -> None:
    cards = tournament_page.championship_favorite_cards(_summary())

    assert cards[0]["team"] == "Avon"
    assert cards[0]["championship_probability"] == "58.0%"
    assert cards[0]["deep_run_probability"] == "78.0%"
    assert cards[0]["bar_width"] == "58%"


def test_bracket_team_line_highlights_favorite_and_probabilities() -> None:
    line = tournament_page.bracket_team_line("Avon", _config(), _summary(), True, 0.78)

    assert "#1 Avon" in line
    assert "Matchup 78%" in line
    assert "Title 58%" in line
    assert "#f0fdf4" in line


def test_highest_round_advancement_probability_parses_rounds() -> None:
    assert tournament_page.highest_round_advancement_probability("Semifinals: 78.0%; Final: 62.0%") == 0.78
    assert tournament_page.highest_round_advancement_probability("") == 0.0
