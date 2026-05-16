from __future__ import annotations

import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import altair as alt
import pandas as pd
import streamlit as st
import yaml

import cvyl_scraper.competition as competition
from cvyl_scraper.competition import (
    BracketRoundConfig,
    CompetitionConfig,
    MatchupConfig,
    build_competition_simulation,
    load_competition_config,
    matchup_win_probability,
)

import streamlit_app as dashboard


ROOT = Path(__file__).resolve().parents[1]
COMPETITIONS_DIR = ROOT / "config" / "competitions"


def main() -> None:
    st.title("Tournament Simulator")
    st.caption(
        "Preview saved playoff divisions or weekend tournament brackets using the current CVYL Power Rating model."
    )
    dashboard.render_data_freshness()

    st.divider()
    configs = available_competition_configs(COMPETITIONS_DIR)
    st.caption(f"Competition config status: {competition_config_status(configs)}")

    try:
        data = dashboard.load_dashboard_data()
    except Exception as exc:
        st.error(f"Dashboard data could not be loaded: {exc}")
        st.info("Run the data pipeline, then refresh this page.")
        data = {}

    render_competition_builder(data, configs)

    if not configs:
        render_no_configs_message()
        return

    selected = st.selectbox(
        "Competition config",
        configs,
        format_func=lambda path: path.stem.replace("_", " ").title(),
        key="competition_config",
    )
    try:
        config = load_competition_config(selected)
    except ValueError as exc:
        st.error(f"Competition config is invalid: {exc}")
        st.info("Fix the YAML config, then refresh this page to preview the tournament.")
        return
    if data.get("power_ratings", pd.DataFrame()).empty:
        st.warning("Power Rating data is not available yet, so the preview cannot run.")
        return

    simulations = st.slider("Monte Carlo runs", min_value=100, max_value=10000, value=1000, step=100)
    random_seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)
    st.button("Re-run simulation", key="rerun_competition_simulation")

    simulation, summary = build_competition_simulation(
        config,
        data["power_ratings"],
        simulations=int(simulations),
        random_seed=int(random_seed),
    )

    render_competition_overview(config)
    render_tournament_outlook(config, simulation, summary)
    render_road_to_championship(config, simulation, summary)
    render_competition_visuals(summary)
    render_current_round_matchups(config, simulation)


def available_competition_configs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}
        ],
        key=lambda path: path.name,
    )


def competition_config_status(configs: list[Path]) -> str:
    if not configs:
        return "No saved competition configs found."
    count = len(configs)
    return f"{count} saved competition config{'s' if count != 1 else ''} found."


def render_no_configs_message() -> None:
    with st.container(border=True):
        st.subheader("Simulator foundation ready")
        st.write(
            "No saved competition configs were found yet. Add YAML files to "
            "`config/competitions/` to preview playoff divisions or weekend tournaments."
        )
        st.caption(
            "Configs support playoffs, tournament divisions, seeds, matchup pairs, persisted winners, "
            "and configurable game formats."
        )


def render_competition_builder(data: dict[str, pd.DataFrame], configs: list[Path]) -> None:
    st.subheader("Competition Builder")
    with st.expander("Create or update a saved competition", expanded=not configs):
        team_options = league_team_options(data)
        if not team_options:
            st.info("Team list is unavailable until processed Power Rating or ELO data exists.")
            return
        initial_config = load_builder_initial_config(configs)
        initial_teams, initial_seeds = builder_initial_teams_and_seeds(initial_config, team_options)
        col1, col2 = st.columns(2)
        competition_name = col1.text_input(
            "Competition name",
            initial_config.competition_name if initial_config else "New CVYL Playoff Division",
            key=builder_field_key(initial_config, "competition_name"),
        )
        competition_type_options = ["Playoffs", "Tournament"]
        initial_type = "Tournament" if initial_config and initial_config.competition_type == "tournament" else "Playoffs"
        competition_type_label = col2.selectbox(
            "Competition type",
            competition_type_options,
            index=competition_type_options.index(initial_type),
            key=builder_field_key(initial_config, "competition_type"),
        )
        division_name = st.text_input(
            "Division name",
            initial_config.division_name if initial_config else "U12 Boys Division",
            key=builder_field_key(initial_config, "division_name"),
        )
        selected_teams = st.multiselect(
            "Teams",
            team_options,
            default=initial_teams,
            key=builder_multiselect_key(initial_config),
            help="Seed controls update immediately when teams are added or removed.",
        )
        st.caption("Assign seeds as 1 through the number of selected teams. Seed 1 is the top seed.")

        seed_values = render_reactive_seed_controls(selected_teams, initial_seeds, initial_config)
        seed_errors = validate_seed_values(selected_teams, seed_values)
        for error in seed_errors:
            st.error(error)

        col1, col2, col3 = st.columns(3)
        initial_game_format = (
            "Running-time game"
            if initial_config and initial_config.game_format == "running_time"
            else "Standard game"
        )
        game_format_label = col1.selectbox(
            "Game format",
            ["Standard game", "Running-time game"],
            index=0 if initial_game_format == "Standard game" else 1,
            key=builder_field_key(initial_config, "game_format"),
            help="Running-time games use a lower scoring environment multiplier for tournament previews.",
        )
        game_minutes = int(
            col2.number_input(
                "Game length",
                min_value=1,
                max_value=120,
                value=initial_config.game_minutes
                if initial_config
                else 48
                if game_format_label == "Standard game"
                else 32,
                step=1,
                key=builder_field_key(initial_config, "game_minutes"),
            )
        )
        simulation_depth = int(
            col3.number_input(
                "Simulation Depth",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key=builder_field_key(initial_config, "simulation_depth"),
                help="Number of Monte Carlo runs used when previewing this competition.",
            )
        )
        random_seed = int(
            st.number_input(
                "Random Seed",
                min_value=1,
                max_value=999999,
                value=42,
                step=1,
                key=builder_field_key(initial_config, "random_seed"),
                help="Using the same seed makes the Monte Carlo preview repeatable.",
            )
        )
        filename = st.text_input(
            "Config filename",
            f"{safe_competition_id(competition_name)}.yml"
            if initial_config is None
            else f"{safe_competition_id(initial_config.competition_name)}.yml",
            key=builder_field_key(initial_config, "filename"),
        )
        overwrite = st.checkbox(
            "Overwrite existing config if it already exists",
            value=True,
            key=builder_field_key(initial_config, "overwrite"),
        )
        if competition_type_label == "Tournament":
            st.info("Pool-stage tournament simulation coming soon. Builder currently creates a bracket-style preview config.")

        if st.button("Save competition config", key=builder_field_key(initial_config, "save")):
            if seed_errors:
                st.error("Fix seed/order errors before saving.")
            else:
                save_builder_submission(
                    competition_name=competition_name,
                    competition_type_label=competition_type_label,
                    division_name=division_name,
                    selected_teams=selected_teams,
                    seed_values=seed_values,
                    game_format_label=game_format_label,
                    game_minutes=game_minutes,
                    filename=filename,
                    overwrite=overwrite,
                    simulation_depth=simulation_depth,
                    random_seed=random_seed,
                )

    if configs:
        with st.expander("Manage saved competitions", expanded=False):
            selected = st.selectbox(
                "Saved competition",
                configs,
                format_func=lambda path: path.name,
                key="manage_competition_config",
            )
            st.caption("Existing competitions can be selected above for immediate simulation.")
            if st.checkbox("Enable delete for selected config"):
                if st.button("Delete selected competition config"):
                    try:
                        selected.unlink()
                        st.success(f"Deleted {selected.name}. Refresh the page to update the config list.")
                    except OSError as exc:
                        st.error(f"Could not delete config: {exc}")


def save_builder_submission(
    *,
    competition_name: str,
    competition_type_label: str,
    division_name: str,
    selected_teams: list[str],
    seed_values: dict[str, int],
    game_format_label: str,
    game_minutes: int,
    filename: str,
    overwrite: bool,
    simulation_depth: int,
    random_seed: int,
) -> Path | None:
    try:
        config = competition_config_from_builder_inputs(
            competition_name=competition_name,
            competition_type_label=competition_type_label,
            division_name=division_name,
            selected_teams=selected_teams,
            seed_values=seed_values,
            game_format_label=game_format_label,
            game_minutes=game_minutes,
        )
        output_path = competition_config_path(filename or f"{safe_competition_id(competition_name)}.yml")
        if output_path.exists() and not overwrite:
            st.error("That config already exists. Enable overwrite or choose another filename.")
            return None
        save_competition_config(config, output_path)
    except ValueError as exc:
        st.error(f"Competition config could not be saved: {exc}")
        return None
    st.success(
        f"Saved {output_path.name}. Select it from the dropdown to simulate with "
        f"{simulation_depth} runs and seed {random_seed}."
    )
    return output_path


def load_builder_initial_config(configs: list[Path]) -> CompetitionConfig | None:
    if not configs:
        return None
    try:
        return load_competition_config(configs[0])
    except ValueError:
        return None


def builder_initial_teams_and_seeds(
    config: CompetitionConfig | None,
    team_options: list[str],
) -> tuple[list[str], dict[str, int]]:
    if config is None:
        teams = team_options[: min(4, len(team_options))]
        return teams, {team: index + 1 for index, team in enumerate(teams)}
    available = set(team_options)
    teams = [team for team in config.teams if team in available]
    seeds = reconcile_seed_values(teams, config.seeds)
    return teams, seeds


def builder_multiselect_key(config: CompetitionConfig | None) -> str:
    suffix = safe_competition_id(config.competition_name) if config else "new"
    return f"builder_selected_teams_{suffix}"


def builder_seed_key(config: CompetitionConfig | None, team: str) -> str:
    suffix = safe_competition_id(config.competition_name) if config else "new"
    return f"builder_seed_{suffix}_{safe_competition_id(team)}"


def builder_field_key(config: CompetitionConfig | None, field: str) -> str:
    suffix = safe_competition_id(config.competition_name) if config else "new"
    return f"builder_{field}_{suffix}"


def render_reactive_seed_controls(
    selected_teams: list[str],
    initial_seeds: dict[str, int],
    initial_config: CompetitionConfig | None,
) -> dict[str, int]:
    seed_values = reconcile_seed_state(selected_teams, initial_seeds, initial_config)
    if not selected_teams:
        return seed_values
    seed_columns = st.columns(min(4, max(1, len(selected_teams))))
    for index, team in enumerate(selected_teams):
        key = builder_seed_key(initial_config, team)
        with seed_columns[index % len(seed_columns)]:
            seed_values[team] = int(
                st.number_input(
                    f"Seed: {team}",
                    min_value=1,
                    max_value=max(1, len(selected_teams)),
                    value=bounded_seed(seed_values.get(team, index + 1), len(selected_teams)),
                    step=1,
                    key=key,
                )
            )
    return {team: seed_values[team] for team in selected_teams}


def reconcile_seed_state(
    selected_teams: list[str],
    initial_seeds: dict[str, int],
    initial_config: CompetitionConfig | None,
) -> dict[str, int]:
    current = {}
    for team in selected_teams:
        key = builder_seed_key(initial_config, team)
        if key in st.session_state:
            current[team] = int(st.session_state[key])
        elif team in initial_seeds:
            current[team] = int(initial_seeds[team])
    reconciled = reconcile_seed_values(selected_teams, current)
    apply_reconciled_seed_state(selected_teams, reconciled, initial_config)
    return reconciled


def apply_reconciled_seed_state(
    selected_teams: list[str],
    reconciled: dict[str, int],
    initial_config: CompetitionConfig | None,
) -> None:
    selected_keys = {builder_seed_key(initial_config, team) for team in selected_teams}
    suffix = safe_competition_id(initial_config.competition_name) if initial_config else "new"
    prefix = f"builder_seed_{suffix}_"
    signature_key = f"builder_seed_signature_{suffix}"
    signature = seed_selection_signature(selected_teams)
    state_changed = st.session_state.get(signature_key) != signature

    for key in [key for key in list(st.session_state.keys()) if str(key).startswith(prefix)]:
        if key not in selected_keys and state_changed:
            del st.session_state[key]

    for team, seed in reconciled.items():
        key = builder_seed_key(initial_config, team)
        if st.session_state.get(key) != seed:
            st.session_state[key] = seed
            state_changed = True
    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature


def seed_selection_signature(selected_teams: list[str]) -> tuple[str, ...]:
    return tuple(selected_teams)


def reconciled_seed_values(selected_teams: list[str], seed_values: dict[str, int]) -> dict[str, int]:
    return reconcile_seed_values(selected_teams, seed_values)


def reconcile_seed_values(selected_teams: list[str], existing_seeds: dict[str, int]) -> dict[str, int]:
    reconciled: dict[str, int] = {}
    used: set[int] = set()
    for team in selected_teams:
        seed = existing_seeds.get(team)
        if seed is not None and 1 <= int(seed) <= len(selected_teams) and int(seed) not in used:
            reconciled[team] = int(seed)
            used.add(int(seed))
    next_seed = 1
    for team in selected_teams:
        if team in reconciled:
            continue
        while next_seed in used:
            next_seed += 1
        reconciled[team] = next_seed
        used.add(next_seed)
    return reconciled


def validate_seed_values(selected_teams: list[str], seed_values: dict[str, int]) -> list[str]:
    errors: list[str] = []
    if len(selected_teams) < 2:
        errors.append("Select at least two teams.")
    if len(set(selected_teams)) != len(selected_teams):
        errors.append("Duplicate teams are not allowed.")
    missing = [team for team in selected_teams if team not in seed_values or pd.isna(seed_values.get(team))]
    if missing:
        errors.append(f"Missing seed values for: {', '.join(missing)}.")
    seeds = [int(seed_values[team]) for team in selected_teams if team in seed_values and not pd.isna(seed_values[team])]
    if len(seeds) != len(set(seeds)):
        errors.append("Duplicate seeds are not allowed.")
    invalid = [seed for seed in seeds if seed < 1 or seed > max(1, len(selected_teams))]
    if invalid:
        errors.append("Seeds must be between 1 and the number of selected teams.")
    return errors


def bounded_seed(seed: object, team_count: int) -> int:
    try:
        value = int(seed)
    except (TypeError, ValueError):
        value = 1
    return max(1, min(max(1, team_count), value))


def competition_config_from_builder_inputs(
    *,
    competition_name: str,
    competition_type_label: str,
    division_name: str,
    selected_teams: list[str],
    seed_values: dict[str, int],
    game_format_label: str,
    game_minutes: int,
) -> CompetitionConfig:
    seeded_teams = [(team, int(seed_values.get(team, index + 1))) for index, team in enumerate(selected_teams)]
    return build_seeded_competition_config(
        competition_name=competition_name,
        competition_type=competition_type_from_label(competition_type_label),
        division_name=division_name,
        seeded_teams=seeded_teams,
        game_format=game_format_from_label(game_format_label),
        game_minutes=game_minutes,
        scoring_environment_multiplier=scoring_multiplier_from_game_format(game_format_label),
    )


def competition_config_path(filename: str) -> Path:
    safe_name = Path(filename).name
    if not safe_name.endswith((".yml", ".yaml")):
        safe_name = f"{safe_name}.yml"
    return COMPETITIONS_DIR / safe_name


def league_team_options(data: dict[str, pd.DataFrame]) -> list[str]:
    for key in ["power_ratings", "ratings"]:
        frame = data.get(key, pd.DataFrame())
        if not frame.empty and "team" in frame.columns:
            return sorted(frame["team"].dropna().astype(str).unique().tolist())
    return []


def competition_type_from_label(label: str) -> str:
    return "tournament" if str(label).strip().casefold().startswith("tournament") else "playoffs"


def game_format_from_label(label: str) -> str:
    return "running_time" if "running" in str(label).casefold() else "standard"


def scoring_multiplier_from_game_format(label: str) -> float:
    return 0.75 if game_format_from_label(label) == "running_time" else 1.0


def safe_competition_id(name: str) -> str:
    shared_helper = getattr(competition, "safe_competition_id", None)
    if callable(shared_helper):
        return str(shared_helper(name))
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().casefold()).strip("_")
    return slug or "competition"


def build_seeded_competition_config(
    *,
    competition_name: str,
    competition_type: str,
    division_name: str,
    seeded_teams: list[tuple[str, int]],
    game_format: str = "standard",
    game_minutes: int = 48,
    scoring_environment_multiplier: float = 1.0,
) -> CompetitionConfig:
    shared_helper = getattr(competition, "build_seeded_competition_config", None)
    if callable(shared_helper):
        return shared_helper(
            competition_name=competition_name,
            competition_type=competition_type,
            division_name=division_name,
            seeded_teams=seeded_teams,
            game_format=game_format,
            game_minutes=game_minutes,
            scoring_environment_multiplier=scoring_environment_multiplier,
        )
    validate_builder_inputs(
        competition_name=competition_name,
        competition_type=competition_type,
        division_name=division_name,
        seeded_teams=seeded_teams,
        game_minutes=game_minutes,
    )
    ordered = sorted(seeded_teams, key=lambda item: (item[1], item[0]))
    teams = [team for team, _seed in ordered]
    seeds = {team: int(seed) for team, seed in ordered}
    return CompetitionConfig(
        competition_name=competition_name.strip(),
        competition_type=competition_type.strip().casefold(),
        division_name=division_name.strip(),
        teams=teams,
        seeds=seeds,
        bracket_rounds=build_seeded_bracket_rounds(ordered),
        game_format=game_format,
        game_minutes=int(game_minutes),
        scoring_environment_multiplier=float(scoring_environment_multiplier),
    )


def validate_builder_inputs(
    *,
    competition_name: str,
    competition_type: str,
    division_name: str,
    seeded_teams: list[tuple[str, int]],
    game_minutes: int,
) -> None:
    shared_helper = getattr(competition, "validate_builder_inputs", None)
    if callable(shared_helper):
        shared_helper(
            competition_name=competition_name,
            competition_type=competition_type,
            division_name=division_name,
            seeded_teams=seeded_teams,
            game_minutes=game_minutes,
        )
        return
    if not competition_name.strip():
        raise ValueError("competition name is required.")
    if competition_type.strip().casefold() not in {"playoffs", "tournament"}:
        raise ValueError("competition type must be playoffs or tournament.")
    if not division_name.strip():
        raise ValueError("division name is required.")
    teams = [team.strip() for team, _seed in seeded_teams]
    if len(teams) < 2:
        raise ValueError("at least two teams are required.")
    if any(not team for team in teams):
        raise ValueError("all selected teams must have names.")
    if len(set(teams)) != len(teams):
        raise ValueError("duplicate teams are not allowed.")
    seeds = [int(seed) for _team, seed in seeded_teams]
    if sorted(seeds) != list(range(1, len(seeds) + 1)):
        raise ValueError("seeds must be unique and numbered from 1 through the number of teams.")
    if int(game_minutes) <= 0:
        raise ValueError("game length must be positive.")


def build_seeded_bracket_rounds(seeded_teams: list[tuple[str, int]]) -> list[BracketRoundConfig]:
    shared_helper = getattr(competition, "build_seeded_bracket_rounds", None)
    if callable(shared_helper):
        return shared_helper(seeded_teams)
    ordered = sorted(seeded_teams, key=lambda item: (item[1], item[0]))
    teams = [team for team, _seed in ordered]
    rounds: list[BracketRoundConfig] = []
    matchup_count = len(teams) // 2
    round_index = 1
    while matchup_count >= 1:
        representatives = teams[: max(2, matchup_count * 2)]
        matchups = [
            MatchupConfig(
                matchup_id=f"r{round_index}_m{index + 1}",
                team_a=representatives[index],
                team_b=representatives[-(index + 1)],
            )
            for index in range(matchup_count)
        ]
        rounds.append(BracketRoundConfig(name=round_name_for_matchups(matchup_count), matchups=matchups))
        matchup_count //= 2
        round_index += 1
    return rounds


def round_name_for_matchups(matchup_count: int) -> str:
    shared_helper = getattr(competition, "round_name_for_matchups", None)
    if callable(shared_helper):
        return str(shared_helper(matchup_count))
    if matchup_count >= 4:
        return "Quarterfinals"
    if matchup_count == 2:
        return "Semifinals"
    return "Championship"


def save_competition_config(config: CompetitionConfig, path: str | Path) -> Path:
    shared_helper = getattr(competition, "save_competition_config", None)
    if callable(shared_helper):
        return shared_helper(config, path)
    validate_helper = getattr(competition, "validate_competition_config", None)
    if callable(validate_helper):
        validate_helper(config)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(competition_config_to_dict(config), file, sort_keys=False)
    return output_path


def competition_config_to_dict(config: CompetitionConfig) -> dict[str, object]:
    shared_helper = getattr(competition, "competition_config_to_dict", None)
    if callable(shared_helper):
        return shared_helper(config)
    return {
        "competition_name": config.competition_name,
        "competition_type": config.competition_type,
        "division_name": config.division_name,
        "teams": config.teams,
        "seeds": config.seeds,
        "game_format": config.game_format,
        "game_minutes": config.game_minutes,
        "scoring_environment_multiplier": config.scoring_environment_multiplier,
        "bracket_rounds": [
            {
                "name": bracket_round.name,
                "matchups": [
                    {
                        key: value
                        for key, value in {
                            "matchup_id": matchup.matchup_id,
                            "team_a": matchup.team_a,
                            "team_b": matchup.team_b,
                            "completed_winner": matchup.completed_winner,
                        }.items()
                        if value is not None
                    }
                    for matchup in bracket_round.matchups
                ],
            }
            for bracket_round in config.bracket_rounds
        ],
    }


def render_competition_overview(config: CompetitionConfig) -> None:
    st.subheader("Competition Overview")
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Competition", config.competition_name)
        cols[1].metric("Type", config.competition_type.title())
        cols[2].metric("Division", config.division_name)
        cols[3].metric("Teams", len(config.teams))

        cols = st.columns(4)
        cols[0].metric("Game Format", config.game_format.replace("_", " ").title())
        cols[1].metric("Game Length", f"{config.game_minutes} min")
        cols[2].metric("Scoring Multiplier", f"{config.scoring_environment_multiplier:.2f}")
        cols[3].metric("Persisted Winners", persisted_winner_count(config))

        st.caption("Seeds")
        seeds = seed_table(config)
        st.dataframe(
            seeds,
            column_config={"seed": "Seed", "team": "Team"},
            use_container_width=True,
            hide_index=True,
        )


def seed_table(config: CompetitionConfig) -> pd.DataFrame:
    rows = [{"seed": config.seeds.get(team), "team": team} for team in config.teams]
    return pd.DataFrame(rows).sort_values(["seed", "team"], na_position="last", ignore_index=True)


def persisted_winner_count(config: CompetitionConfig) -> int:
    return sum(
        1
        for bracket_round in config.bracket_rounds
        for matchup in bracket_round.matchups
        if matchup.completed_winner
    )


def render_tournament_outlook(
    config: CompetitionConfig,
    simulation: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    st.subheader("Tournament Outlook")
    outlook = tournament_outlook_cards(config, simulation, summary)
    if not outlook:
        st.info("Tournament outlook is limited until matchup probabilities are available.")
        return
    columns = st.columns(min(3, len(outlook)))
    for index, card in enumerate(outlook):
        with columns[index % len(columns)]:
            render_outlook_card(card)


def tournament_outlook_cards(
    config: CompetitionConfig,
    simulation: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    if not summary.empty:
        favorite = summary.sort_values(["championship_probability", "seed"], ascending=[False, True]).iloc[0]
        cards.append(
            {
                "label": "Title Favorite",
                "headline": str(favorite["team"]),
                "body": f"Leads the field at {float(favorite['championship_probability']):.1%} championship probability.",
                "detail": f"Seed {format_seed(favorite.get('seed'))}",
            }
        )

        finalists = most_likely_finalists(summary)
        if finalists:
            cards.append(
                {
                    "label": "Most Likely Finalists",
                    "headline": " vs ".join(finalists),
                    "body": "The simulation most often points to this finals profile.",
                    "detail": "Based on fixed-seed Monte Carlo runs.",
                }
            )

        dark_horse = dark_horse_team(summary)
        if dark_horse is not None:
            cards.append(
                {
                    "label": "Dark Horse Team",
                    "headline": str(dark_horse["team"]),
                    "body": "A lower seed with enough title path to watch.",
                    "detail": f"Seed {format_seed(dark_horse.get('seed'))} | Title chance {float(dark_horse['championship_probability']):.1%}",
                }
            )

    if not simulation.empty:
        upset = highest_upset_risk_matchup(simulation)
        if upset is not None:
            cards.append(
                {
                    "label": "Upset Watch",
                    "headline": f"{upset['team_a']} vs {upset['team_b']}",
                    "body": "This matchup carries the highest lower-seed path to an upset.",
                    "detail": f"Upset likelihood {float(upset['upset_likelihood']):.1%}",
                }
            )

        favorite = strongest_favorite_matchup(simulation)
        if favorite is not None:
            cards.append(
                {
                    "label": "Strongest Favorite",
                    "headline": str(favorite["expected_winner"]),
                    "body": "The model sees the clearest first-look edge here.",
                    "detail": f"{favorite['team_a']} vs {favorite['team_b']}",
                }
            )

        tight = tightest_projected_matchup(simulation)
        if tight is not None:
            cards.append(
                {
                    "label": "Tightest Matchup",
                    "headline": f"{tight['team_a']} vs {tight['team_b']}",
                    "body": "This one profiles as the closest projected contest.",
                    "detail": f"Favorite edge {favorite_edge(tight):.1%}",
                }
            )
    return cards[:6]


def render_outlook_card(card: dict[str, str]) -> None:
    with st.container(border=True):
        st.markdown(f"**{card['label']}**")
        st.write(card["headline"])
        st.caption(card["body"])
        st.caption(card["detail"])


def render_road_to_championship(
    config: CompetitionConfig,
    simulation: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    st.subheader("Road to the Championship")
    if simulation.empty:
        st.info("Bracket path preview is not available for this competition yet.")
        return
    with st.container(border=True):
        st.caption(most_likely_path_story(config, simulation, summary))
        render_bracket_progression(config, simulation, summary)

    roadblocks = roadblock_cards(config, simulation, summary)
    if roadblocks:
        st.markdown("**Roadblocks & Danger Zones**")
        columns = st.columns(min(3, len(roadblocks)))
        for index, card in enumerate(roadblocks):
            with columns[index % len(columns)]:
                render_outlook_card(card)

    favorites = championship_favorite_cards(summary)
    if favorites:
        st.markdown("**Championship Favorites**")
        columns = st.columns(min(3, len(favorites)))
        for index, card in enumerate(favorites):
            with columns[index % len(columns)]:
                render_championship_favorite_card(card)


def render_bracket_progression(
    config: CompetitionConfig,
    simulation: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    columns = st.columns(max(1, len(config.bracket_rounds)))
    for column, bracket_round in zip(columns, config.bracket_rounds, strict=False):
        with column:
            st.markdown(f"**{bracket_round.name}**")
            round_rows = simulation[simulation["round"] == bracket_round.name]
            if round_rows.empty:
                st.caption("No matchups configured.")
                continue
            for _, matchup in round_rows.iterrows():
                render_bracket_matchup_card(matchup, config, summary)


def render_bracket_matchup_card(
    matchup: pd.Series,
    config: CompetitionConfig,
    summary: pd.DataFrame,
) -> None:
    favorite = str(matchup["expected_winner"])
    team_a = str(matchup["team_a"])
    team_b = str(matchup["team_b"])
    team_a_probability = float(matchup.get("team_a_win_probability", 0.5))
    upset_watch = float(matchup.get("upset_likelihood", 0.0)) >= 0.35
    with st.container(border=True):
        st.markdown(bracket_team_line(team_a, config, summary, team_a == favorite, team_a_probability), unsafe_allow_html=True)
        st.markdown(
            bracket_team_line(team_b, config, summary, team_b == favorite, 1.0 - team_a_probability),
            unsafe_allow_html=True,
        )
        badges = [story_badge("Upset Watch" if upset_watch else "Playoff Edge")]
        if str(matchup.get("completed_winner", "")) and str(matchup.get("completed_winner")) != "nan":
            badges.append(story_badge("Completed"))
        st.markdown(" ".join(badges), unsafe_allow_html=True)


def bracket_team_line(
    team: str,
    config: CompetitionConfig,
    summary: pd.DataFrame,
    favored: bool,
    probability: float,
) -> str:
    seed = format_seed(config.seeds.get(team))
    title_probability = team_summary_probability(summary, team, "championship_probability")
    border = "#16a34a" if favored else "#d1d5db"
    background = "#f0fdf4" if favored else "#ffffff"
    return (
        f"<div style='border-left:4px solid {border};background:{background};padding:0.35rem 0.45rem;"
        f"margin-bottom:0.3rem;border-radius:6px;'>"
        f"<div style='font-weight:800;color:#111827;overflow-wrap:anywhere;'>{seed} {team}</div>"
        f"<div style='font-size:0.78rem;color:#4b5563;'>"
        f"Matchup {probability:.0%} | Title {title_probability:.0%}</div></div>"
    )


def most_likely_path_story(
    config: CompetitionConfig,
    simulation: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    if summary.empty:
        return "The playoff path is still taking shape."
    favorite = summary.sort_values("championship_probability", ascending=False).iloc[0]
    team = str(favorite["team"])
    likely_round = str(favorite.get("expected_advancement_round") or config.bracket_rounds[-1].name)
    roadblock = toughest_projected_path_team(simulation, team)
    if roadblock:
        return f"{team} appears positioned for a deep run, with a projected {likely_round} roadblock against {roadblock}."
    if float(favorite["championship_probability"]) >= 0.45:
        return f"{team} appears positioned for a deep run if the bracket follows the ratings."
    return "The bracket looks volatile, with no single team separating clearly from the field."


def toughest_projected_path_team(simulation: pd.DataFrame, team: str) -> str:
    rows = simulation[(simulation["team_a"] == team) | (simulation["team_b"] == team)].copy()
    if rows.empty:
        return ""
    rows["edge"] = rows.apply(favorite_edge, axis=1)
    tight = rows.sort_values(["edge", "round"]).iloc[0]
    return str(tight["team_b"] if tight["team_a"] == team else tight["team_a"])


def roadblock_cards(
    config: CompetitionConfig,
    simulation: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[dict[str, str]]:
    cards = []
    upset = highest_upset_risk_matchup(simulation)
    if upset is not None:
        cards.append(
            {
                "label": "Potential Upset Zone",
                "headline": f"{upset['team_a']} vs {upset['team_b']}",
                "body": "Potential semifinal upset risk" if "semi" in str(upset["round"]).lower() else "Lower-seed danger is highest here.",
                "detail": f"Upset likelihood {float(upset['upset_likelihood']):.1%}",
            }
        )
    dark_horse = dark_horse_team(summary)
    if dark_horse is not None:
        cards.append(
            {
                "label": "Most Dangerous Lower Seed",
                "headline": str(dark_horse["team"]),
                "body": "This team has the best lower-seed championship path.",
                "detail": f"Seed {format_seed(dark_horse.get('seed'))} | Title chance {float(dark_horse['championship_probability']):.1%}",
            }
        )
    tough = toughest_projected_path_card(config, simulation, summary)
    if tough is not None:
        cards.append(tough)
    return cards[:3]


def toughest_projected_path_card(
    config: CompetitionConfig,
    simulation: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, str] | None:
    del config
    if summary.empty or simulation.empty:
        return None
    team_rows = []
    for team in summary["team"].astype(str):
        path_rows = simulation[(simulation["team_a"] == team) | (simulation["team_b"] == team)].copy()
        if path_rows.empty:
            continue
        path_rows["edge"] = path_rows.apply(favorite_edge, axis=1)
        team_rows.append({"team": team, "average_edge": float(path_rows["edge"].mean()), "games": len(path_rows)})
    if not team_rows:
        return None
    path = pd.DataFrame(team_rows).sort_values(["average_edge", "games", "team"], ascending=[True, False, True]).iloc[0]
    return {
        "label": "Toughest Projected Path",
        "headline": str(path["team"]),
        "body": "This path has the narrowest projected margins from the configured matchups.",
        "detail": f"Average favorite edge {float(path['average_edge']):.1%}",
    }


def championship_favorite_cards(summary: pd.DataFrame) -> list[dict[str, str]]:
    if summary.empty:
        return []
    ranked = summary.copy()
    ranked["championship_probability"] = pd.to_numeric(
        ranked["championship_probability"],
        errors="coerce",
    )
    ranked = ranked.dropna(subset=["championship_probability"]).sort_values(
        "championship_probability",
        ascending=False,
    )
    cards = []
    for _, row in ranked.head(3).iterrows():
        finalist_probability = highest_round_advancement_probability(row.get("round_advancement_probabilities"))
        cards.append(
            {
                "team": str(row["team"]),
                "seed": format_seed(row.get("seed")),
                "championship_probability": f"{float(row['championship_probability']):.1%}",
                "deep_run_probability": f"{finalist_probability:.1%}",
                "bar_width": f"{max(3, min(100, float(row['championship_probability']) * 100)):.0f}%",
            }
        )
    return cards


def render_championship_favorite_card(card: dict[str, str]) -> None:
    with st.container(border=True):
        st.markdown(f"**{card['seed']} {card['team']}**")
        st.caption(f"Championship odds: {card['championship_probability']}")
        st.caption(f"Deep-run probability: {card['deep_run_probability']}")
        st.markdown(
            f"""
            <div style="height:0.55rem;background:#e5e7eb;border-radius:999px;overflow:hidden;margin-top:0.35rem;">
              <div style="height:100%;width:{card['bar_width']};background:#2563eb;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def team_summary_probability(summary: pd.DataFrame, team: str, column: str) -> float:
    if summary.empty or column not in summary.columns:
        return 0.0
    matches = summary[summary["team"].astype(str) == team]
    if matches.empty:
        return 0.0
    value = pd.to_numeric(matches.iloc[0][column], errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def highest_round_advancement_probability(value: object) -> float:
    probabilities = parse_round_advancement(value)
    if not probabilities:
        return 0.0
    return max(probabilities.values())


def render_competition_visuals(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    st.subheader("Simulation Snapshot")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Championship Probability**")
        st.altair_chart(championship_probability_chart(summary), use_container_width=True)
    with col2:
        st.markdown("**Advancement Probability**")
        st.altair_chart(advancement_probability_chart(summary), use_container_width=True)


def championship_probability_chart(summary: pd.DataFrame) -> alt.Chart:
    chart_data = summary.copy()
    chart_data["championship_probability"] = pd.to_numeric(
        chart_data["championship_probability"],
        errors="coerce",
    )
    chart_data = chart_data.dropna(subset=["championship_probability"]).sort_values(
        "championship_probability",
        ascending=False,
    )
    return (
        alt.Chart(chart_data.head(12))
        .mark_bar(color="#2563eb", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("championship_probability:Q", title="Championship Probability", axis=alt.Axis(format="%")),
            y=alt.Y("team:N", sort=list(chart_data.head(12)["team"]), title=None),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("championship_probability:Q", title="Title Probability", format=".1%"),
            ],
        )
        .properties(height=280)
    )


def advancement_probability_chart(summary: pd.DataFrame) -> alt.Chart:
    rows = []
    for _, team in summary.iterrows():
        for round_name, probability in parse_round_advancement(team.get("round_advancement_probabilities")).items():
            rows.append({"team": team["team"], "round": round_name, "probability": probability})
    chart_data = pd.DataFrame(rows)
    if chart_data.empty:
        chart_data = pd.DataFrame([{"team": "Unavailable", "round": "Unavailable", "probability": 0.0}])
    return (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", title="Advancement Probability", axis=alt.Axis(format="%")),
            y=alt.Y("team:N", sort="-x", title=None),
            color=alt.Color("round:N", title="Round"),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("round:N", title="Round"),
                alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=280)
    )


def render_current_round_matchups(config: CompetitionConfig, simulation: pd.DataFrame) -> None:
    current_round = first_unresolved_round(config)
    rows = simulation[simulation["round"] == current_round.name] if current_round is not None else pd.DataFrame()
    st.subheader("Current Round Preview")
    if current_round is None or rows.empty:
        st.info("No unresolved matchup round is available in this config.")
        return

    columns = st.columns(min(2, len(rows)))
    for index, (_, matchup) in enumerate(rows.iterrows()):
        with columns[index % len(columns)]:
            render_matchup_preview_card(matchup)


def render_matchup_preview_card(matchup: pd.Series) -> None:
    favorite = str(matchup["expected_winner"])
    probability = favorite_probability(matchup)
    edge = prediction_edge_label(probability)
    upset_watch = float(matchup.get("upset_likelihood", 0.0)) >= 0.35
    with st.container(border=True):
        st.markdown(f"**{matchup['team_a']} vs {matchup['team_b']}**")
        st.caption(f"{matchup['round']} | {matchup['matchup_id']}")
        cols = st.columns(2)
        cols[0].metric("Favorite", favorite)
        cols[1].metric("Win Probability", f"{probability:.1%}")
        st.markdown(
            f"{edge_badge(edge)} "
            f"{story_badge('Upset Watch' if upset_watch else 'Playoff Edge')}",
            unsafe_allow_html=True,
        )
        st.caption(matchup_outlook_sentence(matchup, probability, upset_watch))
        if str(matchup.get("note", "")):
            st.warning(str(matchup["note"]))


def matchup_outlook_sentence(matchup: pd.Series, probability: float, upset_watch: bool) -> str:
    if str(matchup.get("completed_winner", "")) and str(matchup["completed_winner"]) != "nan":
        return f"{matchup['completed_winner']} is already recorded as the winner."
    if upset_watch:
        return "The favorite has the edge, but the matchup leaves room for a lower-seed push."
    if probability >= 0.7:
        return "The favorite has a clear path if the game follows the ratings."
    return "This profiles as a competitive playoff matchup."


def first_unresolved_round(config: CompetitionConfig):
    for bracket_round in config.bracket_rounds:
        if any(not matchup.completed_winner for matchup in bracket_round.matchups):
            return bracket_round
    return config.bracket_rounds[-1] if config.bracket_rounds else None


def highest_upset_risk_matchup(simulation: pd.DataFrame) -> pd.Series | None:
    if simulation.empty or "upset_likelihood" not in simulation:
        return None
    candidates = simulation.copy()
    candidates["upset_likelihood"] = pd.to_numeric(candidates["upset_likelihood"], errors="coerce")
    candidates = candidates.dropna(subset=["upset_likelihood"])
    if candidates.empty:
        return None
    return candidates.sort_values(["upset_likelihood", "round"], ascending=[False, True]).iloc[0]


def strongest_favorite_matchup(simulation: pd.DataFrame) -> pd.Series | None:
    if simulation.empty:
        return None
    candidates = simulation.copy()
    candidates["favorite_probability"] = candidates.apply(favorite_probability, axis=1)
    return candidates.sort_values(["favorite_probability", "round"], ascending=[False, True]).iloc[0]


def tightest_projected_matchup(simulation: pd.DataFrame) -> pd.Series | None:
    if simulation.empty:
        return None
    candidates = simulation.copy()
    candidates["favorite_edge"] = candidates.apply(favorite_edge, axis=1)
    return candidates.sort_values(["favorite_edge", "round"], ascending=[True, True]).iloc[0]


def favorite_probability(matchup: pd.Series) -> float:
    team_a_probability = float(matchup.get("team_a_win_probability", 0.5))
    return max(team_a_probability, 1.0 - team_a_probability)


def favorite_edge(matchup: pd.Series) -> float:
    return abs(float(matchup.get("team_a_win_probability", 0.5)) - 0.5)


def most_likely_finalists(summary: pd.DataFrame) -> list[str]:
    if summary.empty or "most_likely_finalist" not in summary:
        return []
    finalists = summary[summary["most_likely_finalist"].astype(bool)]["team"].astype(str).tolist()
    if finalists:
        return finalists[:2]
    return summary.sort_values("championship_probability", ascending=False)["team"].astype(str).head(2).tolist()


def dark_horse_team(summary: pd.DataFrame) -> pd.Series | None:
    if summary.empty:
        return None
    candidates = summary.copy()
    candidates["seed"] = pd.to_numeric(candidates["seed"], errors="coerce")
    candidates["championship_probability"] = pd.to_numeric(
        candidates["championship_probability"],
        errors="coerce",
    )
    candidates = candidates[(candidates["seed"] >= 3) & (candidates["championship_probability"] > 0)]
    if candidates.empty:
        return None
    return candidates.sort_values(["championship_probability", "seed"], ascending=[False, False]).iloc[0]


def parse_round_advancement(value: object) -> dict[str, float]:
    output: dict[str, float] = {}
    for part in str(value or "").split(";"):
        if ":" not in part:
            continue
        round_name, probability = part.split(":", 1)
        text = probability.strip().removesuffix("%")
        try:
            output[round_name.strip()] = float(text) / 100.0
        except ValueError:
            continue
    return output


def format_seed(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"#{int(value)}"


def prediction_edge_label(win_probability: float | None) -> str:
    shared_helper = getattr(dashboard, "prediction_edge_label", None)
    if callable(shared_helper):
        return str(shared_helper(win_probability))
    if win_probability is None or pd.isna(win_probability):
        return "Unavailable"
    probability = max(float(win_probability), 1.0 - float(win_probability))
    if probability < 0.55:
        return "Toss-up"
    if probability < 0.65:
        return "Slight Edge"
    if probability < 0.75:
        return "Solid Favorite"
    return "Strong Favorite"


def edge_badge(label: object) -> str:
    shared_helper = getattr(dashboard, "edge_badge", None)
    if callable(shared_helper):
        return str(shared_helper(label))
    value = str(label or "Unavailable")
    tones = {
        "Toss-up": ("#374151", "#f3f4f6"),
        "Slight Edge": ("#92400e", "#fef3c7"),
        "Solid Favorite": ("#075985", "#e0f2fe"),
        "Strong Favorite": ("#166534", "#dcfce7"),
        "Unavailable": ("#4b5563", "#f3f4f6"),
    }
    color, background = tones.get(value, tones["Unavailable"])
    return metric_badge(value, color=color, background=background)


def story_badge(label: object) -> str:
    shared_helper = getattr(dashboard, "story_badge", None)
    if callable(shared_helper):
        return str(shared_helper(label))
    value = str(label or "Unavailable")
    tones = {
        "Upset Watch": ("#92400e", "#fef3c7"),
        "Playoff Edge": ("#075985", "#e0f2fe"),
        "Strong Edge": ("#166534", "#dcfce7"),
        "Competitive Matchup": ("#075985", "#e0f2fe"),
        "Toss-Up": ("#374151", "#f3f4f6"),
        "Anything Can Happen": ("#92400e", "#fef3c7"),
        "Unavailable": ("#4b5563", "#f3f4f6"),
    }
    color, background = tones.get(value, ("#374151", "#f3f4f6"))
    return metric_badge(value, color=color, background=background)


def metric_badge(label: str, *, color: str, background: str) -> str:
    return (
        f"<span style='display:inline-block;padding:0.18rem 0.55rem;"
        f"border-radius:999px;font-size:0.82rem;font-weight:650;"
        f"color:{color};background:{background};margin-right:0.25rem;'>{label}</span>"
    )


if __name__ == "__main__":
    main()
