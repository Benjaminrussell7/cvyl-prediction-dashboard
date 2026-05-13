from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from cvyl_scraper.prediction import format_matchup_prediction, predict_matchup


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = PROCESSED / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(page_title="CVYL U12 Boys Prediction Dashboard", layout="wide")

    games = load_csv("cvyl_games.csv")
    ratings = load_csv("cvyl_elo_ratings.csv")
    team_games = load_csv("cvyl_team_games.csv")
    elo_history = load_csv("cvyl_elo_history.csv")
    sos = load_csv("cvyl_sos.csv")
    backtest = load_csv("cvyl_backtest.csv")
    backtest_summary = load_csv("cvyl_backtest_summary.csv")

    st.title("CVYL U12 Boys Prediction Dashboard")
    render_data_freshness()

    if games.empty or ratings.empty or team_games.empty:
        st.warning("Missing core processed CSVs. Run the scraper/model pipeline before using the dashboard.")
        return

    render_summary_cards(games, ratings, backtest_summary)
    render_power_rankings(ratings, sos)
    render_matchup_predictor(ratings, team_games, sos)
    render_team_detail(games, ratings, team_games, elo_history, sos)
    render_backtest(backtest, backtest_summary)


def render_data_freshness() -> None:
    games_path = PROCESSED / "cvyl_games.csv"
    if games_path.exists():
        modified = datetime.fromtimestamp(games_path.stat().st_mtime).strftime("%Y-%m-%d %I:%M %p")
    else:
        modified = "Unavailable"

    st.subheader("Data Freshness")
    st.write(f"Latest `cvyl_games.csv` update: **{modified}**")
    st.caption(
        "Predictions and rankings are based only on scores currently reported on CVYL.org. "
        "Some recent games may not yet be reflected."
    )


def render_summary_cards(
    games: pd.DataFrame,
    ratings: pd.DataFrame,
    backtest_summary: pd.DataFrame,
) -> None:
    completed_games = int((games["status"] == "completed").sum()) if "status" in games else 0
    total_teams = len(ratings)
    accuracy = metric_value(backtest_summary, "accuracy", percentage=True)
    brier_score = metric_value(backtest_summary, "brier_score")

    st.subheader("Model Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completed Games", completed_games)
    col2.metric("Teams Rated", total_teams)
    col3.metric("Backtest Accuracy", accuracy)
    col4.metric("Brier Score", brier_score)


def render_power_rankings(ratings: pd.DataFrame, sos: pd.DataFrame) -> None:
    st.subheader("Power Rankings")
    rankings = ratings.copy()
    if not sos.empty:
        rankings = rankings.merge(
            sos[["team", "sos_rank", "average_opponent_elo"]],
            on="team",
            how="left",
        )

    team_filter = st.text_input("Search teams", "", key="power_filter")
    if team_filter:
        rankings = rankings[rankings["team"].str.contains(team_filter, case=False, na=False)]

    display_columns = [
        column
        for column in ["team", "elo", "games_played", "sos_rank", "average_opponent_elo"]
        if column in rankings.columns
    ]
    st.dataframe(
        rankings[display_columns].sort_values("elo", ascending=False),
        column_config={
            "team": "Team",
            "elo": st.column_config.NumberColumn("ELO", format="%.1f"),
            "games_played": "Games Played",
            "sos_rank": "SOS Rank",
            "average_opponent_elo": st.column_config.NumberColumn(
                "Avg. Opponent ELO",
                format="%.1f",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )


def render_matchup_predictor(
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
) -> None:
    st.subheader("Matchup Predictor")
    teams = ratings["team"].dropna().sort_values().tolist()
    if len(teams) < 2:
        st.info("At least two rated teams are required for matchup predictions.")
        return

    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A", teams, index=0)
    default_b_index = 1 if len(teams) > 1 else 0
    team_b = col2.selectbox("Team B", teams, index=default_b_index)

    if team_a == team_b:
        st.warning("Choose two different teams.")
        return

    prediction = predict_matchup(team_a, team_b, ratings, team_games, sos if not sos.empty else None)

    st.markdown(f"**Predicted winner:** {prediction.predicted_winner}")
    prob1, prob2 = st.columns(2)
    prob1.metric(f"{team_a} win probability", f"{prediction.team_a_win_probability:.1%}")
    prob2.metric(f"{team_b} win probability", f"{prediction.team_b_win_probability:.1%}")

    spread_col, total_col, score_col, confidence_col = st.columns(4)
    spread_col.metric("Projected Spread", prediction.projected_spread)
    total_col.metric("Projected Total", f"{prediction.projected_total_goals:.1f}")
    score_col.metric(
        "Projected Score",
        f"{prediction.projected_team_a_goals:.1f}-{prediction.projected_team_b_goals:.1f}",
    )
    confidence_col.metric("Confidence", prediction.confidence_level)

    st.caption(
        f"SOS: {team_a} {_format_sos_context(prediction.team_a_sos, prediction.team_a_sos_rank)} | "
        f"{team_b} {_format_sos_context(prediction.team_b_sos, prediction.team_b_sos_rank)}"
    )
    if prediction.confidence_warning:
        st.warning(prediction.confidence_warning)

    with st.expander("Prediction Details"):
        st.text(format_matchup_prediction(prediction))


def render_team_detail(
    games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    elo_history: pd.DataFrame,
    sos: pd.DataFrame,
) -> None:
    st.subheader("Team Detail")
    teams = ratings["team"].dropna().sort_values().tolist()
    team = st.selectbox("Team", teams, key="team_detail")

    rating_row = ratings[ratings["team"] == team].iloc[0]
    sos_row = sos[sos["team"] == team] if not sos.empty else pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    col1.metric("ELO", f"{float(rating_row['elo']):.1f}")
    col2.metric("Games Played", int(rating_row["games_played"]))
    if not sos_row.empty:
        col3.metric("SOS Rank", int(sos_row.iloc[0]["sos_rank"]))
    else:
        col3.metric("SOS Rank", "N/A")

    render_elo_trend(team, elo_history)

    completed_log = team_games[
        (team_games["team"] == team) & (team_games["status"] == "completed")
    ].copy()
    st.markdown("**Completed Game Log**")
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

    upcoming = games[
        (games["status"] == "scheduled")
        & ((games["home_team"] == team) | (games["away_team"] == team))
    ].copy()
    st.markdown("**Upcoming Scheduled Games**")
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


def render_elo_trend(team: str, elo_history: pd.DataFrame) -> None:
    st.markdown("**ELO Trend**")
    if elo_history.empty:
        st.info("ELO history is not available.")
        return

    team_history = elo_history[elo_history["team"] == team].copy()
    if team_history.empty:
        st.info("No completed ELO history found for this team.")
        return

    team_history["game_date"] = pd.to_datetime(team_history["game_date"], errors="coerce")
    team_history = team_history.dropna(subset=["game_date"]).sort_values(["game_date", "game_id"])
    trend = team_history[["game_date", "postgame_elo"]].rename(columns={"postgame_elo": "ELO"})
    st.line_chart(trend, x="game_date", y="ELO", height=260)


def render_backtest(backtest: pd.DataFrame, backtest_summary: pd.DataFrame) -> None:
    st.subheader("Backtest")
    if backtest_summary.empty:
        st.info("Backtest summary is not available.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", metric_value(backtest_summary, "accuracy", percentage=True))
    col2.metric("Average Confidence", metric_value(backtest_summary, "average_confidence", percentage=True))
    col3.metric("Brier Score", metric_value(backtest_summary, "brier_score"))

    if not backtest.empty:
        st.markdown("**Recent Backtest Rows**")
        st.dataframe(
            backtest.sort_values("game_date", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )


def metric_value(summary: pd.DataFrame, column: str, *, percentage: bool = False) -> str:
    if summary.empty or column not in summary.columns:
        return "N/A"
    value = float(summary.iloc[0][column])
    if percentage:
        return f"{value:.1%}"
    return f"{value:.3f}"


def _format_sos_context(average_opponent_elo: float | None, sos_rank: int | None) -> str:
    if average_opponent_elo is None or sos_rank is None:
        return "SOS unavailable"
    return f"SOS rank {sos_rank}, avg opponent ELO {average_opponent_elo:.1f}"


if __name__ == "__main__":
    main()
