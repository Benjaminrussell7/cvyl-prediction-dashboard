from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

import streamlit_app as dashboard


def main() -> None:
    data = dashboard.load_dashboard_data()

    st.title("Rankings & Ratings")
    dashboard.render_data_freshness()

    if not dashboard.has_core_data(data):
        dashboard.render_missing_core_data_warning()
        return

    dashboard.render_power_rankings(data["ratings"], data["sos"], data["power_ratings"])
    render_rating_leaderboards(data["power_ratings"])


def render_rating_leaderboards(power_ratings) -> None:
    st.divider()
    st.subheader("Rating Leaderboards")
    if power_ratings.empty:
        st.info("Power Rating data is not available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        render_leaderboard(power_ratings, "adjusted_offense_rating", "Offense Strength", ascending=False)
    with col2:
        render_leaderboard(power_ratings, "adjusted_defense_rating", "Defense Strength", ascending=False)


def render_leaderboard(power_ratings, column: str, title: str, *, ascending: bool) -> None:
    st.markdown(f"**{title}**")
    if column not in power_ratings.columns:
        st.info(f"{title} is not available.")
        return
    display = (
        power_ratings[["team", column, "games_played"]]
        .dropna(subset=[column])
        .sort_values([column, "team"], ascending=[ascending, True])
        .head(10)
    )
    st.dataframe(
        display,
        column_config={
            "team": "Team",
            column: st.column_config.NumberColumn(title, format="%.2f"),
            "games_played": "Games",
        },
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
