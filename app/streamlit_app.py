from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from cvyl_scraper.hybrid import power_v2_win_probability
from cvyl_scraper.prediction import format_matchup_prediction, predict_matchup


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_COMPARISON_SUMMARY_FILE = "cvyl_model_comparison_summary.csv"
PRIMARY_POWER_RATINGS_FILE = "cvyl_power_ratings_v3_recency.csv"
PRIMARY_MODEL_COMPARISON_SUMMARY_FILE = "cvyl_model_comparison_v3_summary.csv"
POWER_ACCURACY_KEY = "power_v3_recency_accuracy"
POWER_BRIER_KEY = "power_v3_recency_brier_score"
POWER_RATING_COLUMN = "power_rating_v3_recency"
POWER_RANK_COLUMN = "power_rank_v3_recency"

LOGGER = logging.getLogger(__name__)


@st.cache_data
def _read_csv(filename: str, modified_ns: int) -> pd.DataFrame:
    path = PROCESSED / filename
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().removeprefix("\ufeff") for column in frame.columns]
    return frame


def load_csv(filename: str) -> pd.DataFrame:
    path = PROCESSED / filename
    modified_ns = path.stat().st_mtime_ns if path.exists() else 0
    return _read_csv(filename, modified_ns)


def main() -> None:
    st.set_page_config(page_title="CVYL U12 Boys Prediction Dashboard", layout="wide")

    games = load_csv("cvyl_games.csv")
    scheduled_games = load_csv("cvyl_scheduled_games.csv")
    ratings = load_csv("cvyl_elo_ratings.csv")
    team_games = load_csv("cvyl_team_games.csv")
    elo_history = load_csv("cvyl_elo_history.csv")
    sos = load_csv("cvyl_sos.csv")
    power_ratings = load_csv(PRIMARY_POWER_RATINGS_FILE)
    backtest = load_csv("cvyl_backtest.csv")
    model_comparison_summary = load_csv(PRIMARY_MODEL_COMPARISON_SUMMARY_FILE)
    log_model_comparison_summary(model_comparison_summary)

    st.title("CVYL U12 Boys Prediction Dashboard")
    render_data_freshness()

    if games.empty or ratings.empty or team_games.empty:
        st.warning("Missing core processed CSVs. Run the scraper/model pipeline before using the dashboard.")
        return

    render_summary_cards(games, ratings, model_comparison_summary)
    render_power_rankings(ratings, sos, power_ratings)
    render_matchup_predictor(ratings, team_games, sos, power_ratings)
    render_weekly_matchups(scheduled_games, ratings, team_games, sos, power_ratings)
    render_team_detail(games, ratings, team_games, elo_history, sos, power_ratings)
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
    accuracy = metric_value(model_comparison_summary, POWER_ACCURACY_KEY, percentage=True)
    brier_score = metric_value(model_comparison_summary, POWER_BRIER_KEY)

    st.subheader("Model Summary")
    st.caption("Power Rating is the primary prediction model. ELO is retained as supporting context.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completed Games", completed_games)
    col2.metric("Teams Rated", total_teams)
    col3.metric("Power Rating Accuracy", accuracy)
    col4.metric("Power Rating Brier", brier_score)


def render_power_rankings(ratings: pd.DataFrame, sos: pd.DataFrame, power_ratings: pd.DataFrame) -> None:
    st.subheader("Power Rankings")
    rankings = power_ratings.copy() if not power_ratings.empty else pd.DataFrame()
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
            "power_rank_v3_recency",
            "team",
            "power_rating_v2",
            "power_rating_v3_recency",
            "confidence_tier",
            "elo",
            "games_played",
            "sos_rank",
            "average_opponent_elo",
        ]
        if column in rankings.columns
    ]
    sort_column = POWER_RANK_COLUMN if POWER_RANK_COLUMN in rankings.columns else "elo"
    ascending = sort_column == POWER_RANK_COLUMN
    st.dataframe(
        rankings[display_columns].sort_values(sort_column, ascending=ascending),
        column_config={
            "power_rank_v2": "Power Rank",
            "power_rank_v3_recency": "Power Rank",
            "team": "Team",
            "power_rating_v2": st.column_config.NumberColumn("Power Rating", format="%.2f"),
            "power_rating_v3_recency": st.column_config.NumberColumn("Power Rating", format="%.2f"),
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
    power_ratings: pd.DataFrame,
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

    try:
        prediction = build_matchup_prediction(team_a, team_b, ratings, team_games, sos, power_ratings)
    except Exception as exc:
        LOGGER.exception("Matchup prediction failed for %s vs %s", team_a, team_b)
        st.warning(f"Unable to generate this matchup prediction right now: {exc}")
        return

    power_context = matchup_power_context(team_a, team_b, power_ratings)
    st.markdown(f"**Power Rating favorite:** {power_context['predicted_winner']}")
    prob1, prob2 = st.columns(2)
    prob1.metric(f"{team_a} Power Rating win probability", f"{power_context['team_a_probability']:.1%}")
    prob2.metric(f"{team_b} Power Rating win probability", f"{power_context['team_b_probability']:.1%}")

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
        format_supporting_prediction_context(prediction)
        + " "
        f"{team_a} Power Rating {_format_power_context(power_context['team_a_rating'], power_context['team_a_rank'])} | "
        f"{team_b} Power Rating {_format_power_context(power_context['team_b_rating'], power_context['team_b_rank'])}"
    )
    if prediction.confidence_warning:
        st.warning(prediction.confidence_warning)

    with st.expander("Prediction Details"):
        st.text(format_matchup_prediction(prediction))


def render_weekly_matchups(
    scheduled_games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
) -> None:
    st.subheader("This Week's Matchups")
    weekly_matchups = build_weekly_matchups(
        scheduled_games,
        ratings,
        team_games,
        sos,
        power_ratings,
    )
    if weekly_matchups.empty:
        st.info("No scheduled games found in the next 7 days.")
        return

    team_filter = st.text_input("Search weekly matchups", "", key="weekly_matchup_filter")
    weekly_matchups = filter_matchups_by_team(weekly_matchups, team_filter)
    if weekly_matchups.empty:
        st.info("No matchups match that team search.")
        return

    st.dataframe(
        weekly_matchups,
        use_container_width=True,
        hide_index=True,
    )


def build_weekly_matchups(
    scheduled_games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
    *,
    today: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    columns = [
        "Date",
        "Time",
        "Home",
        "Away",
        "Projected Winner",
        "Win Probability",
        "Projected Spread",
        "Projected Total",
        "Confidence",
        "Note",
    ]
    if scheduled_games.empty:
        return pd.DataFrame(columns=columns)

    games = scheduled_games.copy()
    if "status" in games.columns:
        games = games[games["status"] == "scheduled"].copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    current_day = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    window_end = current_day + pd.Timedelta(days=7)
    games = games[
        (games["game_date"] >= current_day)
        & (games["game_date"] <= window_end)
    ].copy()
    if games.empty:
        return pd.DataFrame(columns=columns)

    sort_columns = [column for column in ["game_date", "game_time", "game_id"] if column in games.columns]
    games = games.sort_values(sort_columns, na_position="last")

    rows = []
    for _, game in games.iterrows():
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        row = {
            "Date": game["game_date"].date().isoformat(),
            "Time": game.get("game_time", ""),
            "Home": home_team,
            "Away": away_team,
            "Projected Winner": "",
            "Win Probability": "",
            "Projected Spread": "",
            "Projected Total": "",
            "Confidence": "",
            "Note": "",
        }
        try:
            prediction = build_matchup_prediction(
                home_team,
                away_team,
                ratings,
                team_games,
                sos,
                power_ratings,
            )
            power_context = matchup_power_context(home_team, away_team, power_ratings)
            row.update(
                {
                    "Projected Winner": power_context["predicted_winner"],
                    "Win Probability": f"{max(power_context['team_a_probability'], power_context['team_b_probability']):.1%}",
                    "Projected Spread": prediction.projected_spread,
                    "Projected Total": f"{prediction.projected_total_goals:.1f}",
                    "Confidence": matchup_confidence_tier(home_team, away_team, power_ratings),
                }
            )
        except Exception as exc:
            LOGGER.warning(
                "Weekly matchup prediction failed for %s vs %s: %s",
                home_team,
                away_team,
                exc,
            )
            row["Note"] = f"Prediction unavailable: {exc}"
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def filter_matchups_by_team(matchups: pd.DataFrame, team_filter: str) -> pd.DataFrame:
    if matchups.empty or not team_filter.strip():
        return matchups
    pattern = team_filter.strip()
    return matchups[
        matchups["Home"].str.contains(pattern, case=False, na=False, regex=False)
        | matchups["Away"].str.contains(pattern, case=False, na=False, regex=False)
    ].copy()


def build_matchup_prediction(
    team_a: str,
    team_b: str,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
):
    del power_ratings
    return predict_matchup(
        team_a,
        team_b,
        ratings,
        team_games,
        sos if not sos.empty else None,
    )


def matchup_power_context(team_a: str, team_b: str, power_ratings: pd.DataFrame) -> dict[str, object]:
    team_a_row = find_team_row(power_ratings, team_a)
    team_b_row = find_team_row(power_ratings, team_b)
    team_a_rating = power_rating_value(team_a_row)
    team_b_rating = power_rating_value(team_b_row)

    if team_a_rating is None or team_b_rating is None:
        team_a_probability = 0.5
        team_b_probability = 0.5
    else:
        team_a_probability = power_v2_win_probability(team_a_rating - team_b_rating)
        team_b_probability = 1.0 - team_a_probability

    return {
        "team_a_rating": team_a_rating,
        "team_b_rating": team_b_rating,
        "team_a_rank": power_rank_value(team_a_row),
        "team_b_rank": power_rank_value(team_b_row),
        "team_a_probability": team_a_probability,
        "team_b_probability": team_b_probability,
        "predicted_winner": team_a if team_a_probability >= team_b_probability else team_b,
    }


def format_supporting_prediction_context(prediction) -> str:
    context = (
        "Supporting context: "
        f"ELO favors {prediction.predicted_winner} ({prediction.win_probability:.1%}); "
    )
    hybrid_winner = getattr(prediction, "hybrid_predicted_winner", None)
    hybrid_team_a = getattr(prediction, "hybrid_win_probability_team_a", None)
    hybrid_team_b = getattr(prediction, "hybrid_win_probability_team_b", None)
    if hybrid_winner is not None and hybrid_team_a is not None and hybrid_team_b is not None:
        context += f"Hybrid favors {hybrid_winner} ({max(hybrid_team_a, hybrid_team_b):.1%}); "
    return context


def find_team_row(frame: pd.DataFrame, team: str) -> pd.Series | None:
    if frame.empty or "team" not in frame.columns:
        return None
    normalized_team = normalize_team_name(team)
    matches = frame[frame["team"].map(normalize_team_name) == normalized_team]
    if matches.empty:
        return None
    return matches.iloc[0]


def normalize_team_name(team: object) -> str:
    return " ".join(str(team).strip().casefold().split())


def power_rating_value(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    for column in [POWER_RATING_COLUMN, "power_rating_v2"]:
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return None


def power_rank_value(row: pd.Series | None) -> int | None:
    if row is None:
        return None
    for column in [POWER_RANK_COLUMN, "power_rank_v2"]:
        if column in row and pd.notna(row[column]):
            return int(row[column])
    return None


def matchup_confidence_tier(team_a: str, team_b: str, power_ratings: pd.DataFrame) -> str:
    tiers = [
        power_confidence_tier(find_team_row(power_ratings, team_a)),
        power_confidence_tier(find_team_row(power_ratings, team_b)),
    ]
    available_tiers = [tier for tier in tiers if tier is not None]
    if not available_tiers:
        return "Unavailable"
    order = {"very low": 0, "low": 1, "medium": 2, "high": 3}
    return min(available_tiers, key=lambda tier: order.get(tier, -1)).title()


def power_confidence_tier(row: pd.Series | None) -> str | None:
    if row is None or "confidence_tier" not in row or pd.isna(row["confidence_tier"]):
        return None
    return str(row["confidence_tier"]).strip().casefold()


def render_team_detail(
    games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    elo_history: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
) -> None:
    st.subheader("Team Detail")
    teams = ratings["team"].dropna().sort_values().tolist()
    team = st.selectbox("Team", teams, key="team_detail")

    rating_row = ratings[ratings["team"] == team].iloc[0]
    sos_row = sos[sos["team"] == team] if not sos.empty else pd.DataFrame()
    power_row = find_team_row(power_ratings, team)

    col1, col2, col3, col4 = st.columns(4)
    if power_row is not None:
        col1.metric("Power Rank", power_rank_value(power_row))
        col2.metric("Power Rating", f"{power_rating_value(power_row):.2f}")
        col3.metric("Confidence", str(power_row["confidence_tier"]).title())
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
    if power_row is not None:
        recency_weight = (
            f" / {float(power_row['average_recency_weight']):.2f}"
            if "average_recency_weight" in power_row and pd.notna(power_row["average_recency_weight"])
            else ""
        )
        metric3.metric(
            "Offense / Defense / Recency",
            f"{float(power_row['adjusted_offense_rating']):.2f} / "
            f"{float(power_row['adjusted_defense_rating']):.2f}"
            f"{recency_weight}",
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

    upcoming = upcoming_scheduled_games_for_team(games, team)
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


def upcoming_scheduled_games_for_team(
    games: pd.DataFrame,
    team: str,
    *,
    today: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    current_day = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    upcoming = games[
        (games["status"] == "scheduled")
        & ((games["home_team"] == team) | (games["away_team"] == team))
    ].copy()
    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"], errors="coerce")
    return upcoming[upcoming["game_date"] >= current_day].copy()


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

    st.caption("Power Rating is primary; ELO and Power v2 are shown for comparison.")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Power Rating",
        metric_value(model_comparison_summary, POWER_ACCURACY_KEY, percentage=True),
        f"Brier {metric_value(model_comparison_summary, POWER_BRIER_KEY)}",
    )
    col2.metric(
        "ELO",
        metric_value(model_comparison_summary, "elo_accuracy", percentage=True),
        f"Brier {metric_value(model_comparison_summary, 'elo_brier_score')}",
    )
    col3.metric(
        "Power v2",
        metric_value(model_comparison_summary, "power_v2_accuracy", percentage=True),
        f"Brier {metric_value(model_comparison_summary, 'power_v2_brier_score')}",
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
    if summary.empty:
        raise ValueError(f"Cannot read metric '{column}' because the comparison summary is empty.")
    normalized = summary.copy()
    normalized.columns = [str(name).strip().removeprefix("\ufeff") for name in normalized.columns]
    if column not in normalized.columns:
        raise KeyError(
            f"Missing metric column '{column}'. Available columns: {normalized.columns.tolist()}"
        )
    if pd.isna(normalized.iloc[0][column]):
        raise ValueError(f"Metric column '{column}' is present but the first row value is missing.")
    value = float(normalized.iloc[0][column])
    if percentage:
        return f"{value:.1%}"
    return f"{value:.3f}"


def log_model_comparison_summary(summary: pd.DataFrame) -> None:
    path = PROCESSED / PRIMARY_MODEL_COMPARISON_SUMMARY_FILE
    raw_headers = read_csv_header(path)
    row_values = summary.iloc[0].to_dict() if not summary.empty else {}
    LOGGER.warning("Loaded comparison summary path: %s", path)
    LOGGER.warning("Comparison summary raw CSV headers: %s", raw_headers)
    LOGGER.warning("Comparison summary dataframe columns: %s", summary.columns.tolist())
    LOGGER.warning("Comparison summary first row values: %s", row_values)
    LOGGER.warning("Power Rating metric lookup keys: %s, %s", POWER_ACCURACY_KEY, POWER_BRIER_KEY)


def read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
    return [header.strip() for header in first_line.split(",")]


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
