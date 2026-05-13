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
    power_v2 = load_csv("cvyl_power_ratings_v2.csv")
    backtest = load_csv("cvyl_backtest.csv")
    model_comparison_summary = load_csv("cvyl_model_comparison_summary.csv")

    st.title("CVYL U12 Boys Prediction Dashboard")
    render_data_freshness()

    if games.empty or ratings.empty or team_games.empty:
        st.warning("Missing core processed CSVs. Run the scraper/model pipeline before using the dashboard.")
        return

    render_summary_cards(games, ratings, model_comparison_summary)
    render_power_rankings(ratings, sos, power_v2)
    render_matchup_predictor(ratings, team_games, sos, power_v2)
    render_team_detail(games, ratings, team_games, elo_history, sos, power_v2)
    render_model_comparison(model_comparison_summary)
    render_backtest(backtest)


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
    model_comparison_summary: pd.DataFrame,
) -> None:
    completed_games = int((games["status"] == "completed").sum()) if "status" in games else 0
    total_teams = len(ratings)
    accuracy = metric_value(model_comparison_summary, "power_v2_accuracy", percentage=True)
    brier_score = metric_value(model_comparison_summary, "power_v2_brier_score")

    st.subheader("Model Summary")
    st.caption("Power Rating is the primary prediction model. ELO is retained as supporting context.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completed Games", completed_games)
    col2.metric("Teams Rated", total_teams)
    col3.metric("Power Rating Accuracy", accuracy)
    col4.metric("Power Rating Brier", brier_score)


def render_power_rankings(ratings: pd.DataFrame, sos: pd.DataFrame, power_v2: pd.DataFrame) -> None:
    st.subheader("Power Rankings")
    rankings = power_v2.copy() if not power_v2.empty else pd.DataFrame()
    if rankings.empty:
        rankings = ratings.copy()
    elif not ratings.empty:
        rankings = rankings.merge(ratings[["team", "elo"]], on="team", how="left")

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
        for column in [
            "power_rank_v2",
            "team",
            "power_rating_v2",
            "confidence_tier",
            "elo",
            "games_played",
            "sos_rank",
            "average_opponent_elo",
        ]
        if column in rankings.columns
    ]
    sort_column = "power_rank_v2" if "power_rank_v2" in rankings.columns else "elo"
    ascending = sort_column == "power_rank_v2"
    st.dataframe(
        rankings[display_columns].sort_values(sort_column, ascending=ascending),
        column_config={
            "power_rank_v2": "Power Rank",
            "team": "Team",
            "power_rating_v2": st.column_config.NumberColumn("Power Rating", format="%.2f"),
            "confidence_tier": "Confidence",
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
    power_v2: pd.DataFrame,
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

    prediction = predict_matchup(
        team_a,
        team_b,
        ratings,
        team_games,
        sos if not sos.empty else None,
        power_v2 if not power_v2.empty else None,
    )

    st.markdown(f"**Power Rating favorite:** {prediction.power_v2_predicted_winner}")
    prob1, prob2 = st.columns(2)
    prob1.metric(f"{team_a} Power Rating win probability", f"{prediction.power_v2_win_probability_team_a:.1%}")
    prob2.metric(f"{team_b} Power Rating win probability", f"{prediction.power_v2_win_probability_team_b:.1%}")

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
    st.caption(
        "Supporting context: "
        f"ELO favors {prediction.predicted_winner} ({prediction.win_probability:.1%}); "
        f"Hybrid favors {prediction.hybrid_predicted_winner} "
        f"({max(prediction.hybrid_win_probability_team_a, prediction.hybrid_win_probability_team_b):.1%}); "
        f"{team_a} Power Rating {_format_power_context(prediction.team_a_power_v2, prediction.team_a_power_rank_v2)} | "
        f"{team_b} Power Rating {_format_power_context(prediction.team_b_power_v2, prediction.team_b_power_rank_v2)}"
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
    power_v2: pd.DataFrame,
) -> None:
    st.subheader("Team Detail")
    teams = ratings["team"].dropna().sort_values().tolist()
    team = st.selectbox("Team", teams, key="team_detail")

    rating_row = ratings[ratings["team"] == team].iloc[0]
    sos_row = sos[sos["team"] == team] if not sos.empty else pd.DataFrame()
    power_row = power_v2[power_v2["team"] == team] if not power_v2.empty else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    if not power_row.empty:
        col1.metric("Power Rank", int(power_row.iloc[0]["power_rank_v2"]))
        col2.metric("Power Rating", f"{float(power_row.iloc[0]['power_rating_v2']):.2f}")
        col3.metric("Confidence", str(power_row.iloc[0]["confidence_tier"]).title())
    else:
        col1.metric("Power Rank", "N/A")
        col2.metric("Power Rating", "N/A")
        col3.metric("Confidence", "N/A")
    if not sos_row.empty:
        col4.metric("SOS Rank", int(sos_row.iloc[0]["sos_rank"]))
    else:
        col4.metric("SOS Rank", "N/A")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("ELO", f"{float(rating_row['elo']):.1f}")
    metric2.metric("Games Played", int(rating_row["games_played"]))
    if not power_row.empty:
        metric3.metric(
            "Offense / Defense",
            f"{float(power_row.iloc[0]['adjusted_offense_rating']):.2f} / "
            f"{float(power_row.iloc[0]['adjusted_defense_rating']):.2f}",
        )
    else:
        metric3.metric("Offense / Defense", "N/A")

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


def render_model_comparison(model_comparison_summary: pd.DataFrame) -> None:
    st.subheader("Model Comparison")
    if model_comparison_summary.empty:
        st.info("Model comparison summary is not available.")
        return

    st.caption("Power Rating is primary; ELO and Hybrid are shown for comparison.")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Power Rating",
        metric_value(model_comparison_summary, "power_v2_accuracy", percentage=True),
        f"Brier {metric_value(model_comparison_summary, 'power_v2_brier_score')}",
    )
    col2.metric(
        "ELO",
        metric_value(model_comparison_summary, "elo_accuracy", percentage=True),
        f"Brier {metric_value(model_comparison_summary, 'elo_brier_score')}",
    )
    col3.metric(
        "Hybrid",
        metric_value(model_comparison_summary, "hybrid_accuracy", percentage=True),
        f"Brier {metric_value(model_comparison_summary, 'hybrid_brier_score')}",
    )


def render_backtest(backtest: pd.DataFrame) -> None:
    st.subheader("Recent ELO Backtest Rows")
    st.caption("Detailed ELO replay rows are retained for debugging and comparison context.")

    if not backtest.empty:
        st.dataframe(
            backtest.sort_values("game_date", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Backtest rows are not available.")


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


def _format_power_context(power_rating: float | None, power_rank: int | None) -> str:
    if power_rating is None or power_rank is None:
        return "unavailable"
    return f"{power_rating:.2f} (rank {power_rank})"


if __name__ == "__main__":
    main()
