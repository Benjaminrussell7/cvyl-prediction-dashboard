from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import altair as alt
import pandas as pd
import streamlit as st

from cvyl_scraper.competition import (
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

    try:
        data = dashboard.load_dashboard_data()
    except Exception as exc:
        st.error(f"Dashboard data could not be loaded: {exc}")
        st.info("Run the data pipeline, then refresh this page.")
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
