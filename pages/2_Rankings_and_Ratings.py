from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import altair as alt
import pandas as pd
import streamlit as st

from cvyl_scraper.historical_snapshots import longest_top_five_streaks

import streamlit_app as dashboard

HISTORICAL_SNAPSHOTS_FILE = "cvyl_historical_snapshots.csv"


def main() -> None:
    data = dashboard.load_dashboard_data()

    st.title("Rankings & Ratings")
    dashboard.render_data_freshness()

    if not dashboard.has_core_data(data):
        dashboard.render_missing_core_data_warning()
        return

    dashboard.render_power_rankings(data["ratings"], data["sos"], data["power_ratings"])
    render_historical_trends(load_historical_snapshots(data))
    render_rating_leaderboards(data["power_ratings"])


def load_historical_snapshots(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    snapshots = data.get("historical_snapshots", pd.DataFrame())
    if historical_snapshot_frame_has_required_columns(snapshots):
        return snapshots

    loaded = dashboard.load_csv(HISTORICAL_SNAPSHOTS_FILE)
    if historical_snapshot_frame_has_required_columns(loaded):
        return loaded

    direct_path = Path(__file__).resolve().parents[1] / "data" / "processed" / HISTORICAL_SNAPSHOTS_FILE
    if not direct_path.exists():
        return pd.DataFrame()
    try:
        direct = pd.read_csv(direct_path)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()
    direct.columns = [str(column).strip().removeprefix("\ufeff") for column in direct.columns]
    if historical_snapshot_frame_has_required_columns(direct):
        return direct
    return pd.DataFrame()


def historical_snapshot_frame_has_required_columns(snapshots: pd.DataFrame) -> bool:
    required = {"snapshot_date", "snapshot_label", "team", "power_rank"}
    return not snapshots.empty and required.issubset(snapshots.columns)


def render_historical_trends(snapshots: pd.DataFrame) -> None:
    history = historical_snapshot_display_data(snapshots)
    st.divider()
    st.subheader("Historical Trends")
    if history.empty or history["snapshot_date"].nunique() < 2:
        st.info("Historical trend snapshots are not available yet. Run the data refresh after more completed games.")
        return

    current_label = str(history.sort_values("snapshot_date")["snapshot_label"].iloc[-1])
    st.caption(f"Weekly snapshots show how the league picture has moved through {current_label}.")

    movement = latest_rank_movement(history)
    climbing = multi_snapshot_rank_movement(history, snapshots_back=3, direction="up")
    cooling = multi_snapshot_rank_movement(history, snapshots_back=3, direction="down")
    top_streaks = top_five_streak_display(history)

    col1, col2, col3 = st.columns(3)
    render_trend_card(col1, "Biggest mover this week", movement.head(1), value_column="rank_move")
    render_trend_card(col2, "Cooling off", movement.tail(1), value_column="rank_move")
    render_trend_card(col3, "Holding the top tier", top_streaks.head(1), value_column="top_5_streak")

    with st.expander("League movement details", expanded=False):
        tab1, tab2, tab3 = st.tabs(["This week", "Last 3 snapshots", "Top tier streaks"])
        with tab1:
            render_movement_tables(movement)
            if not movement.empty:
                st.altair_chart(rank_movement_chart(movement), use_container_width=True)
        with tab2:
            render_multi_snapshot_tables(climbing, cooling)
            movers = pd.concat([climbing.head(3), cooling.head(3)], ignore_index=True)
            if not movers.empty:
                st.altair_chart(rank_trajectory_chart(history, movers["team"].tolist()), use_container_width=True)
        with tab3:
            st.caption("Holding the top tier means a team has stayed inside the current top five across weekly snapshots.")
            st.altair_chart(top_five_streak_chart(top_streaks), use_container_width=True)
            st.dataframe(
                top_streaks.head(10),
                column_config={
                    "team": "Team",
                    "top_5_streak": "Current Top-5 Streak",
                },
                use_container_width=True,
                hide_index=True,
            )


def historical_snapshot_display_data(snapshots: pd.DataFrame) -> pd.DataFrame:
    if not historical_snapshot_frame_has_required_columns(snapshots):
        return pd.DataFrame()
    history = snapshots.copy()
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"], errors="coerce")
    numeric_columns = [
        "power_rank",
        "power_rating",
        "elo",
        "offense_strength",
        "defense_strength",
        "sos_rank",
        "momentum_score",
        "last3_win_pct",
        "last5_win_pct",
        "wins",
        "losses",
        "games_played",
    ]
    for column in numeric_columns:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    return history.dropna(subset=["snapshot_date", "team"]).sort_values(
        ["snapshot_date", "team"],
        ignore_index=True,
    )


def latest_rank_movement(history: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(history["snapshot_date"].dropna().unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["team", "previous_rank", "current_rank", "rank_move"])
    previous = history[history["snapshot_date"] == dates[-2]][["team", "power_rank"]].rename(
        columns={"power_rank": "previous_rank"}
    )
    current = history[history["snapshot_date"] == dates[-1]][["team", "power_rank"]].rename(
        columns={"power_rank": "current_rank"}
    )
    movement = current.merge(previous, on="team", how="inner").dropna(
        subset=["current_rank", "previous_rank"]
    )
    movement["rank_move"] = movement["previous_rank"] - movement["current_rank"]
    return movement.sort_values(["rank_move", "team"], ascending=[False, True], ignore_index=True)


def multi_snapshot_rank_movement(
    history: pd.DataFrame,
    *,
    snapshots_back: int,
    direction: str,
) -> pd.DataFrame:
    dates = sorted(history["snapshot_date"].dropna().unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["team", "starting_rank", "current_rank", "rank_move"])
    selected_dates = dates[-snapshots_back:] if len(dates) >= snapshots_back else dates
    start = history[history["snapshot_date"] == selected_dates[0]][["team", "power_rank"]].rename(
        columns={"power_rank": "starting_rank"}
    )
    current = history[history["snapshot_date"] == selected_dates[-1]][["team", "power_rank"]].rename(
        columns={"power_rank": "current_rank"}
    )
    movement = current.merge(start, on="team", how="inner").dropna(
        subset=["current_rank", "starting_rank"]
    )
    movement["rank_move"] = movement["starting_rank"] - movement["current_rank"]
    ascending = direction == "down"
    movement = movement.sort_values(["rank_move", "team"], ascending=[ascending, True], ignore_index=True)
    if direction == "up":
        return movement[movement["rank_move"] > 0].reset_index(drop=True)
    if direction == "down":
        return movement[movement["rank_move"] < 0].reset_index(drop=True)
    return movement


def top_five_streak_display(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["team", "top_5_streak"])
    streaks = longest_top_five_streaks(history)
    return streaks[streaks["top_5_streak"] > 0].sort_values(
        ["top_5_streak", "team"],
        ascending=[False, True],
        ignore_index=True,
    )


def render_trend_card(container, title: str, rows: pd.DataFrame, *, value_column: str) -> None:
    with container:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if rows.empty:
                st.caption("Not enough movement yet.")
                return
            row = rows.iloc[0]
            st.write(str(row["team"]))
            value = row.get(value_column)
            if value_column == "rank_move":
                st.caption(f"{format_rank_move(value)} since the prior snapshot")
            else:
                st.caption(f"{int(value)} straight weekly snapshot{'s' if int(value) != 1 else ''} in the top five")


def render_movement_tables(movement: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biggest risers since the previous snapshot**")
        st.caption("Biggest mover this week")
        st.dataframe(
            movement.head(8).assign(Move=lambda frame: frame["rank_move"].map(format_rank_move))[
                ["team", "previous_rank", "current_rank", "Move"]
            ],
            column_config={"team": "Team", "previous_rank": "Previous", "current_rank": "Current"},
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        st.markdown("**Biggest fallers since the previous snapshot**")
        st.caption("Cooling off")
        st.dataframe(
            movement.tail(8).sort_values(["rank_move", "team"]).assign(
                Move=lambda frame: frame["rank_move"].map(format_rank_move)
            )[["team", "previous_rank", "current_rank", "Move"]],
            column_config={"team": "Team", "previous_rank": "Previous", "current_rank": "Current"},
            use_container_width=True,
            hide_index=True,
        )


def render_multi_snapshot_tables(climbing: pd.DataFrame, cooling: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Climbing fast**")
        st.dataframe(
            climbing.head(8).assign(Move=lambda frame: frame["rank_move"].map(format_rank_move))[
                ["team", "starting_rank", "current_rank", "Move"]
            ],
            column_config={"team": "Team", "starting_rank": "Start", "current_rank": "Current"},
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        st.markdown("**Cooling off**")
        st.dataframe(
            cooling.head(8).assign(Move=lambda frame: frame["rank_move"].map(format_rank_move))[
                ["team", "starting_rank", "current_rank", "Move"]
            ],
            column_config={"team": "Team", "starting_rank": "Start", "current_rank": "Current"},
            use_container_width=True,
            hide_index=True,
        )


def format_rank_move(value: object) -> str:
    if pd.isna(value):
        return "No movement"
    move = int(value)
    if move > 0:
        return f"Up {move}"
    if move < 0:
        return f"Down {abs(move)}"
    return "No movement"


def rank_movement_chart(movement: pd.DataFrame) -> alt.Chart:
    chart_data = pd.concat([movement.head(5), movement.tail(5)], ignore_index=True).drop_duplicates("team")
    chart_data = chart_data.sort_values("rank_move", ascending=False)
    chart_data["direction"] = chart_data["rank_move"].map(
        lambda value: "Up" if value > 0 else "Down" if value < 0 else "Flat"
    )
    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("rank_move:Q", title="Rank Move"),
            y=alt.Y("team:N", sort=list(chart_data["team"]), title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Up", "Flat", "Down"], range=["#16a34a", "#9ca3af", "#dc2626"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("previous_rank:Q", title="Previous Rank", format=".0f"),
                alt.Tooltip("current_rank:Q", title="Current Rank", format=".0f"),
                alt.Tooltip("rank_move:Q", title="Rank Move", format="+.0f"),
            ],
        )
        .properties(height=280)
    )


def top_five_streak_chart(streaks: pd.DataFrame) -> alt.Chart:
    chart_data = streaks.head(10).sort_values("top_5_streak", ascending=False)
    return (
        alt.Chart(chart_data)
        .mark_bar(color="#2563eb", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("top_5_streak:Q", title="Current Top-5 Streak"),
            y=alt.Y("team:N", sort=list(chart_data["team"]), title=None),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("top_5_streak:Q", title="Top-5 Streak", format=".0f"),
            ],
        )
        .properties(height=260)
    )


def rank_trajectory_chart(history: pd.DataFrame, teams: list[str]) -> alt.Chart:
    chart_data = history[history["team"].isin(teams)].dropna(subset=["power_rank"]).copy()
    chart_data["snapshot_label"] = chart_data["snapshot_label"].astype(str)
    snapshot_order = chart_data.sort_values("snapshot_date")["snapshot_label"].drop_duplicates().tolist()
    return (
        alt.Chart(chart_data)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=55), strokeWidth=3)
        .encode(
            x=alt.X("snapshot_label:N", sort=snapshot_order, title=None),
            y=alt.Y("power_rank:Q", title="Power Rank", scale=alt.Scale(reverse=True, zero=False)),
            color=alt.Color("team:N", title="Team"),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("snapshot_label:N", title="Snapshot"),
                alt.Tooltip("power_rank:Q", title="Power Rank", format=".0f"),
            ],
        )
        .properties(height=300)
        .interactive()
    )


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
