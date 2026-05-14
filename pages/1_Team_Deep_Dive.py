from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

import streamlit_app as dashboard


def main() -> None:
    data = dashboard.load_dashboard_data()

    st.title("Team Deep Dive")
    dashboard.render_data_freshness()

    if not dashboard.has_core_data(data):
        dashboard.render_missing_core_data_warning()
        return

    teams = data["ratings"]["team"].dropna().sort_values().tolist()
    team = st.selectbox("Team", teams, key="team_deep_dive")
    render_team_header(team, data)
    render_team_trends(team, data)
    render_team_games(team, data)


def render_team_header(team: str, data: dict[str, pd.DataFrame]) -> None:
    rating_row = dashboard.find_team_row(data["ratings"], team)
    power_row = dashboard.find_team_row(data["power_ratings"], team)
    sos_row = dashboard.find_team_row(data["sos"], team)
    trend_row = dashboard.find_team_row(data["trends"], team)

    st.subheader(team)
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Power Rank", dashboard.format_optional_int(dashboard.power_rank_value(power_row)))
        col2.metric("Power Rating", dashboard.format_optional_float(dashboard.power_rating_value(power_row)))
        col3.metric("ELO", dashboard.format_row_float(rating_row, "elo"))
        col4.metric("SOS Rank", dashboard.format_row_int(sos_row, "sos_rank"))

        col5, col6, col7 = st.columns(3)
        col5.metric("Form", dashboard.team_recent_form(trend_row))
        col6.metric("Recent Rank Move", dashboard.team_recent_rank_move(trend_row))
        col7.metric("Games Played", dashboard.format_row_int(rating_row, "games_played"))

        st.caption(
            f"Offense Strength: {dashboard.format_row_float(power_row, 'adjusted_offense_rating')} | "
            f"Defense Strength: {dashboard.format_row_float(power_row, 'adjusted_defense_rating')}"
        )


def render_team_trends(team: str, data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Team Trends")
    power_row = dashboard.find_team_row(data["power_ratings"], team)
    dashboard.render_offense_defense_chart(power_row)
    dashboard.render_elo_trend(team, data["elo_history"])

    trend_row = dashboard.find_team_row(data["trends"], team)
    if trend_row is not None:
        st.caption(
            f"Recent form: {dashboard.team_recent_form(trend_row)}. "
            f"Rank movement: {dashboard.team_recent_rank_move(trend_row)}."
        )


def render_team_games(team: str, data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Recent Games")
    completed_log = data["team_games"][
        (data["team_games"]["team"] == team) & (data["team_games"]["status"] == "completed")
    ].copy()
    if completed_log.empty:
        st.info("No completed games found for this team.")
    else:
        st.dataframe(
            completed_log[
                [
                    column
                    for column in ["game_date", "opponent", "points_for", "points_against", "win"]
                    if column in completed_log.columns
                ]
            ].sort_values("game_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Upcoming Scheduled Games")
    upcoming = dashboard.upcoming_scheduled_games_for_team(data["games"], team)
    if upcoming.empty:
        st.info("No scheduled games found for this team.")
    else:
        st.dataframe(
            upcoming[
                [
                    column
                    for column in ["game_date", "game_time", "home_team", "away_team"]
                    if column in upcoming.columns
                ]
            ].sort_values(["game_date", "game_time"], na_position="last"),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
