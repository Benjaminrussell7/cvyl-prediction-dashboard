from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cvyl_scraper.export import export_csv
from cvyl_scraper.probability_calibration import calibrated_power_v4_probability


DEFAULT_COMPETITION_SIMULATION_CSV = "data/processed/cvyl_competition_simulation.csv"
DEFAULT_COMPETITION_SUMMARY_CSV = "data/processed/cvyl_competition_summary.csv"
VALID_COMPETITION_TYPES = {"playoffs", "tournament"}
DEFAULT_GAME_FORMAT = "standard"
DEFAULT_GAME_MINUTES = 48
DEFAULT_SCORING_ENVIRONMENT_MULTIPLIER = 1.0
POWER_RATING_COLUMNS = ["power_rating_v4", "power_rating_v3_recency"]

COMPETITION_SIMULATION_COLUMNS = [
    "competition_name",
    "division_name",
    "round",
    "matchup_id",
    "team_a",
    "team_b",
    "team_a_win_probability",
    "team_b_win_probability",
    "expected_winner",
    "completed_winner",
    "upset_likelihood",
    "note",
]

COMPETITION_SUMMARY_COLUMNS = [
    "competition_name",
    "division_name",
    "team",
    "seed",
    "championship_probability",
    "round_advancement_probabilities",
    "expected_advancement_round",
    "most_likely_finalist",
]


@dataclass(frozen=True)
class MatchupConfig:
    matchup_id: str
    team_a: str
    team_b: str
    completed_winner: str | None = None


@dataclass(frozen=True)
class BracketRoundConfig:
    name: str
    matchups: list[MatchupConfig]


@dataclass(frozen=True)
class CompetitionConfig:
    competition_name: str
    competition_type: str
    division_name: str
    teams: list[str]
    seeds: dict[str, int]
    bracket_rounds: list[BracketRoundConfig]
    game_format: str = DEFAULT_GAME_FORMAT
    game_minutes: int = DEFAULT_GAME_MINUTES
    scoring_environment_multiplier: float = DEFAULT_SCORING_ENVIRONMENT_MULTIPLIER


def load_competition_config(path: str | Path) -> CompetitionConfig:
    with Path(path).open(encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    config = competition_config_from_dict(payload)
    validate_competition_config(config)
    return config


def save_competition_config(config: CompetitionConfig, path: str | Path) -> Path:
    validate_competition_config(config)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(competition_config_to_dict(config), file, sort_keys=False)
    return output_path


def competition_config_to_dict(config: CompetitionConfig) -> dict[str, Any]:
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


def safe_competition_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().casefold()).strip("_")
    return slug or "competition"


def build_seeded_competition_config(
    *,
    competition_name: str,
    competition_type: str,
    division_name: str,
    seeded_teams: list[tuple[str, int]],
    game_format: str = DEFAULT_GAME_FORMAT,
    game_minutes: int = DEFAULT_GAME_MINUTES,
    scoring_environment_multiplier: float = DEFAULT_SCORING_ENVIRONMENT_MULTIPLIER,
) -> CompetitionConfig:
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
    if not competition_name.strip():
        raise ValueError("competition name is required.")
    if competition_type.strip().casefold() not in VALID_COMPETITION_TYPES:
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
    ordered = sorted(seeded_teams, key=lambda item: (item[1], item[0]))
    teams = [team for team, _seed in ordered]
    rounds: list[BracketRoundConfig] = []
    first_round_matchups = [
        MatchupConfig(
            matchup_id=f"r1_m{index + 1}",
            team_a=teams[index],
            team_b=teams[-(index + 1)],
        )
        for index in range(len(teams) // 2)
    ]
    rounds.append(BracketRoundConfig(name=round_name_for_matchups(len(first_round_matchups)), matchups=first_round_matchups))

    matchup_count = len(first_round_matchups) // 2
    round_index = 2
    while matchup_count >= 1:
        representative_teams = teams[: max(2, matchup_count * 2)]
        matchups = [
            MatchupConfig(
                matchup_id=f"r{round_index}_m{index + 1}",
                team_a=representative_teams[index],
                team_b=representative_teams[-(index + 1)],
            )
            for index in range(matchup_count)
        ]
        rounds.append(BracketRoundConfig(name=round_name_for_matchups(matchup_count), matchups=matchups))
        matchup_count //= 2
        round_index += 1
    return rounds


def round_name_for_matchups(matchup_count: int) -> str:
    if matchup_count >= 4:
        return "Quarterfinals"
    if matchup_count == 2:
        return "Semifinals"
    return "Championship"


def competition_config_from_dict(payload: dict[str, Any]) -> CompetitionConfig:
    rounds = [
        BracketRoundConfig(
            name=str(round_payload.get("name", "")).strip(),
            matchups=[
                MatchupConfig(
                    matchup_id=str(matchup.get("matchup_id", "")).strip(),
                    team_a=str(matchup.get("team_a", "")).strip(),
                    team_b=str(matchup.get("team_b", "")).strip(),
                    completed_winner=_optional_text(matchup.get("completed_winner")),
                )
                for matchup in round_payload.get("matchups", []) or []
            ],
        )
        for round_payload in payload.get("bracket_rounds", []) or []
    ]
    seeds_payload = payload.get("seeds", {}) or {}
    return CompetitionConfig(
        competition_name=str(payload.get("competition_name", "")).strip(),
        competition_type=str(payload.get("competition_type", "")).strip(),
        division_name=str(payload.get("division_name", "")).strip(),
        teams=[str(team).strip() for team in payload.get("teams", []) or []],
        seeds={str(team).strip(): int(seed) for team, seed in seeds_payload.items()},
        bracket_rounds=rounds,
        game_format=str(payload.get("game_format", DEFAULT_GAME_FORMAT)).strip() or DEFAULT_GAME_FORMAT,
        game_minutes=int(payload.get("game_minutes", DEFAULT_GAME_MINUTES)),
        scoring_environment_multiplier=float(
            payload.get(
                "scoring_environment_multiplier",
                DEFAULT_SCORING_ENVIRONMENT_MULTIPLIER,
            )
        ),
    )


def validate_competition_config(config: CompetitionConfig) -> None:
    if not config.competition_name:
        raise ValueError("competition_name is required.")
    if config.competition_type not in VALID_COMPETITION_TYPES:
        raise ValueError("competition_type must be playoffs or tournament.")
    if not config.division_name:
        raise ValueError("division_name is required.")
    if len(set(config.teams)) != len(config.teams):
        raise ValueError("teams must be unique.")
    if not config.teams:
        raise ValueError("at least one team is required.")
    if config.game_minutes <= 0:
        raise ValueError("game_minutes must be positive.")
    if config.scoring_environment_multiplier <= 0:
        raise ValueError("scoring_environment_multiplier must be positive.")

    team_set = set(config.teams)
    unknown_seed_teams = set(config.seeds) - team_set
    if unknown_seed_teams:
        raise ValueError(f"seeds include unknown teams: {sorted(unknown_seed_teams)}")

    matchup_ids: set[str] = set()
    if not config.bracket_rounds:
        raise ValueError("at least one bracket_round is required.")
    for bracket_round in config.bracket_rounds:
        if not bracket_round.name:
            raise ValueError("each bracket round requires a name.")
        if not bracket_round.matchups:
            raise ValueError(f"round {bracket_round.name} requires at least one matchup.")
        for matchup in bracket_round.matchups:
            if not matchup.matchup_id:
                raise ValueError("each matchup requires matchup_id.")
            if matchup.matchup_id in matchup_ids:
                raise ValueError(f"duplicate matchup_id: {matchup.matchup_id}")
            matchup_ids.add(matchup.matchup_id)
            for team in [matchup.team_a, matchup.team_b]:
                if team not in team_set:
                    raise ValueError(f"matchup {matchup.matchup_id} includes unknown team: {team}")
            if matchup.team_a == matchup.team_b:
                raise ValueError(f"matchup {matchup.matchup_id} cannot contain the same team twice.")
            if matchup.completed_winner is not None and matchup.completed_winner not in {
                matchup.team_a,
                matchup.team_b,
            }:
                raise ValueError(
                    f"completed_winner for {matchup.matchup_id} must be one of the matchup teams."
                )


def build_competition_simulation(
    config: CompetitionConfig,
    power_ratings: pd.DataFrame,
    *,
    simulations: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_competition_config(config)
    deterministic_rows, expected_winners = deterministic_competition_advancement(config, power_ratings)
    monte_carlo = monte_carlo_competition_advancement(
        config,
        power_ratings,
        simulations=simulations,
        random_seed=random_seed,
    )
    summary = build_competition_summary(config, deterministic_rows, expected_winners, monte_carlo)
    return deterministic_rows, summary


def deterministic_competition_advancement(
    config: CompetitionConfig,
    power_ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, object]] = []
    expected_winners: dict[str, str] = {}
    for bracket_round in config.bracket_rounds:
        for matchup in bracket_round.matchups:
            probability = matchup_win_probability(config, matchup.team_a, matchup.team_b, power_ratings)
            expected_winner = matchup.completed_winner or (
                matchup.team_a if probability >= 0.5 else matchup.team_b
            )
            expected_winners[matchup.matchup_id] = expected_winner
            rows.append(
                {
                    "competition_name": config.competition_name,
                    "division_name": config.division_name,
                    "round": bracket_round.name,
                    "matchup_id": matchup.matchup_id,
                    "team_a": matchup.team_a,
                    "team_b": matchup.team_b,
                    "team_a_win_probability": probability,
                    "team_b_win_probability": 1.0 - probability,
                    "expected_winner": expected_winner,
                    "completed_winner": matchup.completed_winner,
                    "upset_likelihood": upset_likelihood(config, matchup, probability),
                    "note": missing_team_note(matchup.team_a, matchup.team_b, power_ratings),
                }
            )
    return pd.DataFrame(rows, columns=COMPETITION_SIMULATION_COLUMNS), expected_winners


def matchup_win_probability(
    config: CompetitionConfig,
    team_a: str,
    team_b: str,
    power_ratings: pd.DataFrame,
) -> float:
    team_a_rating = team_power_rating(team_a, power_ratings)
    team_b_rating = team_power_rating(team_b, power_ratings)
    if team_a_rating is None or team_b_rating is None:
        return 0.5
    difference = (team_a_rating - team_b_rating) * config.scoring_environment_multiplier
    return calibrated_power_v4_probability(difference)


def monte_carlo_competition_advancement(
    config: CompetitionConfig,
    power_ratings: pd.DataFrame,
    *,
    simulations: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    if simulations <= 0:
        raise ValueError("simulations must be positive.")
    generator = pd.Series(range(simulations)).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    # Use pandas' deterministic sampling only to derive a stable stream seed order, then simple LCG.
    state = int(generator.iloc[0]) + random_seed + 1
    rows: list[dict[str, object]] = []
    round_names = [round_config.name for round_config in config.bracket_rounds]
    for simulation_index in range(simulations):
        winners_by_round: dict[str, list[str]] = {name: [] for name in round_names}
        champion = ""
        for bracket_round in config.bracket_rounds:
            for matchup in bracket_round.matchups:
                probability = matchup_win_probability(config, matchup.team_a, matchup.team_b, power_ratings)
                if matchup.completed_winner:
                    winner = matchup.completed_winner
                else:
                    state = (1103515245 * state + 12345) % (2**31)
                    draw = state / float(2**31)
                    winner = matchup.team_a if draw <= probability else matchup.team_b
                winners_by_round[bracket_round.name].append(winner)
                champion = winner
        for round_name, winners in winners_by_round.items():
            for winner in winners:
                rows.append(
                    {
                        "simulation": simulation_index,
                        "round": round_name,
                        "team": winner,
                        "advanced": True,
                        "champion": winner == champion and round_name == round_names[-1],
                    }
                )
    return pd.DataFrame(rows)


def build_competition_summary(
    config: CompetitionConfig,
    deterministic_rows: pd.DataFrame,
    expected_winners: dict[str, str],
    monte_carlo: pd.DataFrame,
) -> pd.DataFrame:
    del expected_winners
    final_round = config.bracket_rounds[-1].name
    finalist_counts = (
        monte_carlo[monte_carlo["round"] == final_round]["team"].value_counts(normalize=True)
        if not monte_carlo.empty
        else pd.Series(dtype=float)
    )
    champion_counts = (
        monte_carlo[monte_carlo["champion"]]["team"].value_counts(normalize=True)
        if not monte_carlo.empty
        else pd.Series(dtype=float)
    )
    round_probabilities = round_advancement_probabilities(monte_carlo)
    deterministic_advancement = deterministic_advancement_round_by_team(deterministic_rows)
    rows = []
    for team in config.teams:
        rows.append(
            {
                "competition_name": config.competition_name,
                "division_name": config.division_name,
                "team": team,
                "seed": config.seeds.get(team),
                "championship_probability": float(champion_counts.get(team, 0.0)),
                "round_advancement_probabilities": round_probabilities.get(team, ""),
                "expected_advancement_round": deterministic_advancement.get(team, ""),
                "most_likely_finalist": bool(finalist_counts.get(team, 0.0) == finalist_counts.max())
                if not finalist_counts.empty
                else False,
            }
        )
    return pd.DataFrame(rows, columns=COMPETITION_SUMMARY_COLUMNS).sort_values(
        ["championship_probability", "seed", "team"],
        ascending=[False, True, True],
        na_position="last",
        ignore_index=True,
    )


def deterministic_advancement_round_by_team(deterministic_rows: pd.DataFrame) -> dict[str, str]:
    rounds: dict[str, str] = {}
    for _, row in deterministic_rows.iterrows():
        winner = str(row["expected_winner"])
        rounds[winner] = str(row["round"])
    return rounds


def round_advancement_probabilities(monte_carlo: pd.DataFrame) -> dict[str, str]:
    if monte_carlo.empty:
        return {}
    simulations = max(1, int(monte_carlo["simulation"].nunique()))
    counts = (
        monte_carlo.groupby(["team", "round"], as_index=False)
        .agg(advancements=("advanced", "sum"))
        .sort_values(["team", "round"])
    )
    counts["probability"] = counts["advancements"] / simulations
    output: dict[str, list[str]] = {}
    for _, row in counts.iterrows():
        output.setdefault(str(row["team"]), []).append(
            f"{row['round']}: {float(row['probability']):.1%}"
        )
    return {team: "; ".join(values) for team, values in output.items()}


def upset_likelihood(config: CompetitionConfig, matchup: MatchupConfig, team_a_probability: float) -> float:
    seed_a = config.seeds.get(matchup.team_a)
    seed_b = config.seeds.get(matchup.team_b)
    if seed_a is None or seed_b is None or seed_a == seed_b:
        return min(team_a_probability, 1.0 - team_a_probability)
    lower_seed_team_a = seed_a > seed_b
    return team_a_probability if lower_seed_team_a else 1.0 - team_a_probability


def missing_team_note(team_a: str, team_b: str, power_ratings: pd.DataFrame) -> str:
    missing = [team for team in [team_a, team_b] if team_power_rating(team, power_ratings) is None]
    if not missing:
        return ""
    return f"Missing Power Rating for: {', '.join(missing)}"


def team_power_rating(team: str, power_ratings: pd.DataFrame) -> float | None:
    if power_ratings.empty or "team" not in power_ratings.columns:
        return None
    matches = power_ratings[power_ratings["team"].astype(str) == team]
    if matches.empty:
        return None
    row = matches.iloc[0]
    for rating_column in POWER_RATING_COLUMNS:
        if rating_column not in power_ratings.columns:
            continue
        value = pd.to_numeric(row[rating_column], errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


def export_competition_simulation(
    config: CompetitionConfig,
    power_ratings: pd.DataFrame,
    simulation_output_path: str | Path = DEFAULT_COMPETITION_SIMULATION_CSV,
    summary_output_path: str | Path = DEFAULT_COMPETITION_SUMMARY_CSV,
    **kwargs: Any,
) -> tuple[Path, Path]:
    simulation, summary = build_competition_simulation(config, power_ratings, **kwargs)
    return export_csv(simulation, simulation_output_path), export_csv(summary, summary_output_path)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
