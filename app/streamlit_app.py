from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from cvyl_scraper.explanations import generate_matchup_explanation
from cvyl_scraper.prediction import format_matchup_prediction, predict_matchup
from cvyl_scraper.probability_calibration import calibrated_power_v3_probability


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_COMPARISON_SUMMARY_FILE = "cvyl_model_comparison_summary.csv"
PRIMARY_POWER_RATINGS_FILE = "cvyl_power_ratings_v3_recency.csv"
PRIMARY_MODEL_COMPARISON_SUMMARY_FILE = "cvyl_model_comparison_v4_calibrated_summary.csv"
PRIMARY_CALIBRATION_FILE = "cvyl_calibration_power_rating_v4.csv"
POWER_ACCURACY_KEY = "power_v3_calibrated_accuracy"
POWER_BRIER_KEY = "power_v3_calibrated_brier_score"
POWER_RATING_COLUMN = "power_rating_v3_recency"
POWER_RANK_COLUMN = "power_rank_v3_recency"
MATCHUP_TEAM_COLORS = ["#2563eb", "#16a34a"]

LOGGER = logging.getLogger(__name__)
EASTERN_TIME = ZoneInfo("America/New_York")


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


def load_dashboard_data() -> dict[str, pd.DataFrame]:
    return {
        "games": load_csv("cvyl_games.csv"),
        "scheduled_games": load_csv("cvyl_scheduled_games.csv"),
        "ratings": load_csv("cvyl_elo_ratings.csv"),
        "team_games": load_csv("cvyl_team_games.csv"),
        "elo_history": load_csv("cvyl_elo_history.csv"),
        "sos": load_csv("cvyl_sos.csv"),
        "trends": load_csv("cvyl_trends.csv"),
        "power_ratings": load_csv(PRIMARY_POWER_RATINGS_FILE),
        "backtest": load_csv("cvyl_backtest.csv"),
        "model_comparison_summary": load_csv(PRIMARY_MODEL_COMPARISON_SUMMARY_FILE),
        "calibration": load_csv(PRIMARY_CALIBRATION_FILE),
    }


def configure_page(title: str = "CVYL U12 Boys Prediction Dashboard") -> None:
    st.set_page_config(page_title=title, layout="wide")


def has_core_data(data: dict[str, pd.DataFrame]) -> bool:
    return not (
        data["games"].empty
        or data["ratings"].empty
        or data["team_games"].empty
    )


def render_missing_core_data_warning() -> None:
    st.warning("Missing core processed CSVs. Run the scraper/model pipeline before using the dashboard.")


def main() -> None:
    configure_page()
    if hasattr(st, "navigation") and hasattr(st, "Page"):
        page = st.navigation(
            [
                st.Page(render_home_page, title="Home", default=True),
                st.Page(ROOT / "pages" / "1_Team_Deep_Dive.py", title="Team Deep Dive"),
                st.Page(ROOT / "pages" / "2_Rankings_and_Ratings.py", title="Rankings & Ratings"),
                st.Page(ROOT / "pages" / "3_Model_Insights.py", title="Model Insights"),
                st.Page(ROOT / "pages" / "4_Tournament_Simulator.py", title="Tournament Simulator"),
            ],
            position="sidebar",
            expanded=True,
        )
        page.run()
        return

    render_home_page()


def render_home_page() -> None:
    data = load_dashboard_data()
    log_model_comparison_summary(data["model_comparison_summary"])

    st.title("CVYL U12 Boys Prediction Dashboard")
    render_data_freshness()

    if not has_core_data(data):
        render_missing_core_data_warning()
        return

    render_summary_cards(data["games"], data["ratings"], data["model_comparison_summary"])
    render_featured_insights(
        data["scheduled_games"],
        data["ratings"],
        data["team_games"],
        data["sos"],
        data["power_ratings"],
        data["trends"],
    )
    render_matchup_predictor(
        data["ratings"],
        data["team_games"],
        data["sos"],
        data["power_ratings"],
        data["trends"],
    )
    render_weekly_matchups(
        data["scheduled_games"],
        data["ratings"],
        data["team_games"],
        data["sos"],
        data["power_ratings"],
        data["trends"],
    )
    render_trending_teams(data["trends"])
    render_rankings_preview(data["ratings"], data["sos"], data["power_ratings"])


def render_data_freshness() -> None:
    games_path = PROCESSED / "cvyl_games.csv"
    if games_path.exists():
        modified = format_eastern_timestamp(datetime.fromtimestamp(games_path.stat().st_mtime))
    else:
        modified = "Unavailable"

    st.caption(f"Latest `cvyl_games.csv` update: **{modified}**")
    st.caption(
        "Predictions and rankings are based only on scores currently reported on CVYL.org. "
        "Some recent games may not yet be reflected."
    )


def format_eastern_timestamp(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.astimezone()
    eastern_timestamp = timestamp.astimezone(EASTERN_TIME)
    return f"{eastern_timestamp:%Y-%m-%d %I:%M %p} ET"


def render_summary_cards(
    games: pd.DataFrame,
    ratings: pd.DataFrame,
    model_comparison_summary: pd.DataFrame,
) -> None:
    completed_games = int((games["status"] == "completed").sum()) if "status" in games else 0
    total_teams = len(ratings)
    accuracy = metric_value(model_comparison_summary, POWER_ACCURACY_KEY, percentage=True)
    brier_score = metric_value(model_comparison_summary, POWER_BRIER_KEY)

    st.divider()
    st.subheader("At a Glance")
    st.caption("Power Rating is the primary prediction model. ELO is retained as supporting context.")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Completed Games", completed_games)
        col2.metric("Teams Rated", total_teams)
        col3.metric("Power Rating Accuracy", accuracy)
        col4.metric("Power Rating Brier", brier_score)


def render_featured_insights(
    scheduled_games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
) -> None:
    cards = featured_insight_cards(scheduled_games, ratings, team_games, sos, power_ratings, trends)
    if not cards:
        return
    st.subheader("Featured Insights")
    first_row = cards[:2]
    second_row = cards[2:]
    for row_cards in [first_row, second_row]:
        if not row_cards:
            continue
        columns = st.columns(len(row_cards))
        for column, card in zip(columns, row_cards, strict=True):
            with column:
                render_featured_insight_card(card)


def featured_insight_cards(
    scheduled_games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    weekly = build_weekly_matchups(scheduled_games, ratings, team_games, sos, power_ratings, trends)
    game_of_week = select_game_of_week(weekly, power_ratings)
    if game_of_week is not None:
        cards.append(featured_game_card("Game of the Week", game_of_week))

    upset = select_upset_watch(weekly)
    if upset is not None:
        cards.append(featured_game_card("Upset Watch", upset))

    rising = fastest_rising_team_card(trends, power_ratings)
    if rising is not None:
        cards.append(rising)

    defense = defense_to_watch_card(power_ratings)
    if defense is not None:
        cards.append(defense)

    contender = quiet_contender_card(power_ratings, trends)
    if contender is not None:
        cards.append(contender)

    return cards[:5]


def render_featured_insight_card(card: dict[str, str]) -> None:
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="min-height:9.5rem;">
              <div style="font-size:0.76rem;font-weight:800;color:#4b5563;text-transform:uppercase;
                          letter-spacing:0.04rem;margin-bottom:0.2rem;">{card['label']}</div>
              <div style="font-size:1.05rem;font-weight:800;color:#111827;line-height:1.25;
                          overflow-wrap:anywhere;margin-bottom:0.35rem;">{card['headline']}</div>
              <div style="margin-bottom:0.35rem;">{story_badge(card['tag'])}</div>
              <div style="font-size:0.9rem;color:#374151;line-height:1.35;">{card['body']}</div>
              <div style="font-size:0.78rem;color:#6b7280;margin-top:0.45rem;">{card['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def select_game_of_week(weekly: pd.DataFrame, power_ratings: pd.DataFrame) -> pd.Series | None:
    if weekly.empty:
        return None
    candidates = weekly[weekly["Projected Winner"].astype(str).str.len() > 0].copy()
    if candidates.empty:
        return None
    candidates["favorite_probability"] = candidates["Win Probability"].map(parse_percentage)
    candidates["closeness_score"] = (candidates["favorite_probability"] - 0.5).abs()
    candidates["team_quality"] = candidates.apply(
        lambda row: matchup_quality_score(str(row["Home"]), str(row["Away"]), power_ratings),
        axis=1,
    )
    candidates["interest_score"] = (
        (1.0 - candidates["closeness_score"].fillna(0.5))
        + candidates["team_quality"].fillna(0.0)
        + candidates["Matchup Type"].isin(["Upset Watch", "Closer Than It Looks", "Tight Contest"]).astype(float) * 0.35
    )
    return candidates.sort_values(["interest_score", "Date", "Time"], ascending=[False, True, True]).iloc[0]


def select_upset_watch(weekly: pd.DataFrame) -> pd.Series | None:
    if weekly.empty or "Matchup Type" not in weekly:
        return None
    candidates = weekly[
        weekly["Matchup Type"].isin(["Upset Watch", "Closer Than It Looks", "Tight Contest"])
        | weekly["Edge"].isin(["Toss-up", "Slight Edge"])
    ].copy()
    if candidates.empty:
        return None
    candidates["favorite_probability"] = candidates["Win Probability"].map(parse_percentage)
    candidates["upset_watch_score"] = (0.70 - candidates["favorite_probability"].fillna(0.70)).clip(lower=0)
    return candidates.sort_values(["upset_watch_score", "Date", "Time"], ascending=[False, True, True]).iloc[0]


def featured_game_card(label: str, matchup: pd.Series) -> dict[str, str]:
    teams = f"{matchup['Home']} vs {matchup['Away']}"
    favorite = str(matchup["Projected Winner"] or "No clear favorite")
    tag = str(matchup.get("Matchup Type") or matchup.get("Edge") or label)
    detail = f"{matchup['Date']} {matchup['Time']}".strip()
    if label == "Upset Watch":
        body = f"{favorite} is favored, but the matchup profile says this could stay tighter than expected."
    else:
        body = str(matchup.get("What Model Sees") or matchup.get("Explanation") or "")
        if not body:
            body = f"The model leans toward {favorite}, with enough intrigue to make this the lead matchup."
    return {
        "label": label,
        "headline": teams,
        "tag": tag,
        "body": body,
        "detail": detail,
    }


def fastest_rising_team_card(trends: pd.DataFrame, power_ratings: pd.DataFrame) -> dict[str, str] | None:
    if trends.empty or "team" not in trends:
        return None
    candidates = trends.copy()
    for column in ["power_rank_movement", "momentum_score", "games_played"]:
        if column in candidates:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates = candidates[candidates["games_played"].fillna(0) >= 2] if "games_played" in candidates else candidates
    if candidates.empty:
        return None
    sort_columns = [column for column in ["power_rank_movement", "momentum_score"] if column in candidates]
    if not sort_columns:
        return None
    team_row = candidates.sort_values(sort_columns, ascending=[False] * len(sort_columns)).iloc[0]
    team = str(team_row["team"])
    form = team_recent_form(team_row)
    power_row = find_team_row(power_ratings, team)
    rank = power_rank_value(power_row)
    move = team_recent_rank_move(team_row)
    return {
        "label": "Fastest Rising Team",
        "headline": team,
        "tag": "Momentum Clash" if form in {"Surging", "Improving"} else "Emerging Contender Matchup",
        "body": notification_phrase_for_team(team, form),
        "detail": f"Current rank {format_optional_int(rank)} | Rank move {move}",
    }


def defense_to_watch_card(power_ratings: pd.DataFrame) -> dict[str, str] | None:
    if power_ratings.empty or "adjusted_defense_rating" not in power_ratings:
        return None
    candidates = power_ratings.copy()
    candidates["adjusted_defense_rating"] = pd.to_numeric(candidates["adjusted_defense_rating"], errors="coerce")
    candidates["games_played"] = pd.to_numeric(candidates.get("games_played"), errors="coerce")
    candidates = candidates.dropna(subset=["adjusted_defense_rating"])
    candidates = candidates[candidates["games_played"].fillna(0) >= 3]
    if candidates.empty:
        return None
    row = candidates.sort_values(["adjusted_defense_rating", "team"], ascending=[False, True]).iloc[0]
    return {
        "label": "Defense to Watch",
        "headline": str(row["team"]),
        "tag": "Physical Defensive Battle",
        "body": "Defense continues to lead the way, with one of the strongest recent profiles in the division.",
        "detail": f"Power rank {format_optional_int(power_rank_value(row))} | Goals allowed {format_row_float(row, 'avg_points_against')}",
    }


def quiet_contender_card(power_ratings: pd.DataFrame, trends: pd.DataFrame) -> dict[str, str] | None:
    if power_ratings.empty:
        return None
    candidates = power_ratings.copy()
    if not trends.empty and "team" in trends:
        candidates = candidates.merge(
            trends[["team", "momentum_score", "momentum_label", "power_rank_movement"]],
            on="team",
            how="left",
        )
    for column in [POWER_RANK_COLUMN, "games_played", "momentum_score"]:
        if column in candidates:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates = candidates[
        (candidates[POWER_RANK_COLUMN] <= 10)
        & (candidates["games_played"].fillna(0) >= 3)
        & (candidates[POWER_RANK_COLUMN] > 3)
    ].copy()
    if candidates.empty:
        return None
    candidates["contender_score"] = (
        (11 - candidates[POWER_RANK_COLUMN].astype(float))
        + candidates.get("momentum_score", pd.Series(0, index=candidates.index)).fillna(0) / 5.0
    )
    row = candidates.sort_values(["contender_score", POWER_RANK_COLUMN], ascending=[False, True]).iloc[0]
    team = str(row["team"])
    form = team_recent_form(row)
    return {
        "label": "Quiet Contender",
        "headline": team,
        "tag": "Emerging Contender Matchup",
        "body": f"{team} is quietly sitting in contender territory, with a profile that may be stronger than the headline record suggests.",
        "detail": f"Power rank {format_optional_int(power_rank_value(row))} | {notification_phrase_for_team(team, form)}",
    }


def matchup_quality_score(team_a: str, team_b: str, power_ratings: pd.DataFrame) -> float:
    ranks = [
        power_rank_value(find_team_row(power_ratings, team_a)),
        power_rank_value(find_team_row(power_ratings, team_b)),
    ]
    available = [rank for rank in ranks if rank is not None]
    if not available:
        return 0.0
    return sum(max(0.0, (16.0 - float(rank)) / 15.0) for rank in available) / len(available)


def parse_percentage(value: object) -> float | None:
    text = str(value or "").strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text) / 100.0
    except ValueError:
        return None


def render_power_rankings(ratings: pd.DataFrame, sos: pd.DataFrame, power_ratings: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Rankings")
    rankings = rankings_display_data(ratings, sos, power_ratings)

    team_filter = st.text_input("Search teams", "", key="power_filter")
    if team_filter:
        rankings = rankings[rankings["team"].str.contains(team_filter, case=False, na=False)]

    display_columns = rankings_display_columns(rankings)
    sort_column = POWER_RANK_COLUMN if POWER_RANK_COLUMN in rankings.columns else "elo"
    ascending = sort_column == POWER_RANK_COLUMN
    render_power_rating_chart(rankings, sort_column, ascending)
    render_rankings_table(rankings, display_columns, sort_column, ascending)
    with st.expander("Compact rankings", expanded=False):
        render_compact_power_rankings(rankings, sort_column, ascending)


def rankings_display_data(ratings: pd.DataFrame, sos: pd.DataFrame, power_ratings: pd.DataFrame) -> pd.DataFrame:
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
    return rankings


def rankings_display_columns(rankings: pd.DataFrame) -> list[str]:
    return [
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


def render_rankings_table(
    rankings: pd.DataFrame,
    display_columns: list[str],
    sort_column: str,
    ascending: bool,
) -> None:
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


def render_rankings_preview(ratings: pd.DataFrame, sos: pd.DataFrame, power_ratings: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Rankings Preview")
    rankings = rankings_display_data(ratings, sos, power_ratings)
    if rankings.empty:
        st.info("Rankings are not available.")
        return
    sort_column = POWER_RANK_COLUMN if POWER_RANK_COLUMN in rankings.columns else "elo"
    ascending = sort_column == POWER_RANK_COLUMN
    display_columns = [
        column
        for column in [POWER_RANK_COLUMN, "team", POWER_RATING_COLUMN, "confidence_tier", "games_played"]
        if column in rankings.columns
    ]
    st.dataframe(
        rankings[display_columns].sort_values(sort_column, ascending=ascending).head(10),
        column_config={
            POWER_RANK_COLUMN: "Rank",
            "team": "Team",
            POWER_RATING_COLUMN: st.column_config.NumberColumn("Power Rating", format="%.2f"),
            "confidence_tier": "Confidence",
            "games_played": "Games",
        },
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Open Rankings & Ratings from the sidebar for the full table and rating charts.")


def render_power_rating_chart(rankings: pd.DataFrame, sort_column: str, ascending: bool) -> None:
    if POWER_RATING_COLUMN not in rankings.columns or "team" not in rankings.columns:
        return
    top_power = (
        rankings.dropna(subset=[POWER_RATING_COLUMN])
        .sort_values(POWER_RATING_COLUMN, ascending=False)
        .head(10)
        [["team", POWER_RATING_COLUMN]]
    )
    if top_power.empty:
        return
    st.markdown("**Top 10 Power Ratings**")
    st.altair_chart(
        ordered_bar_chart(
            top_power,
            x_column="team",
            y_column=POWER_RATING_COLUMN,
            y_title="Power Rating",
            color="#1d4ed8",
        ),
        use_container_width=True,
    )


def render_compact_power_rankings(rankings: pd.DataFrame, sort_column: str, ascending: bool) -> None:
    compact_columns = [
        column
        for column in [
            POWER_RANK_COLUMN,
            "team",
            POWER_RATING_COLUMN,
            "confidence_tier",
            "games_played",
        ]
        if column in rankings.columns
    ]
    if not compact_columns:
        return

    compact = rankings[compact_columns].sort_values(sort_column, ascending=ascending).head(25)
    st.dataframe(
        compact,
        column_config={
            POWER_RANK_COLUMN: "Rank",
            "team": "Team",
            POWER_RATING_COLUMN: st.column_config.NumberColumn("Power", format="%.2f"),
            "confidence_tier": "Confidence",
            "games_played": "Games",
        },
        use_container_width=True,
        hide_index=True,
    )


def render_trending_teams(trends: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Trending Teams")
    st.caption(
        "Recent form uses completed games only. Momentum blends recent wins, scoring margin, "
        "offense, defense, and Power Rating rank movement."
    )
    st.caption(
        "Rank Move compares current Power Rank to the prior snapshot from recent completed games. "
        "Positive means the team moved up."
    )
    if trends.empty:
        st.info("Trend data is not available yet.")
        return

    render_momentum_chart(trends)
    render_form_distribution(trends)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Hottest Teams**")
        render_trend_table(
            trends.sort_values(["momentum_score", "team"], ascending=[False, True]).head(8)
        )
    with col2:
        st.markdown("**Biggest Risers**")
        render_trend_table(
            trends.sort_values(["power_rank_movement", "momentum_score"], ascending=[False, False]).head(8)
        )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Strongest Recent Defense**")
        render_trend_table(
            trends.sort_values(["recent_defense_rating", "team"], ascending=[True, True]).head(8)
        )
    with col4:
        st.markdown("**Strongest Recent Offense**")
        render_trend_table(
            trends.sort_values(["recent_offense_rating", "team"], ascending=[False, True]).head(8)
        )


def render_momentum_chart(trends: pd.DataFrame) -> None:
    if "momentum_score" not in trends.columns or "team" not in trends.columns:
        return
    momentum = (
        trends[["team", "momentum_score"]]
        .dropna()
        .sort_values(["momentum_score", "team"], ascending=[False, True])
        .head(10)
    )
    if momentum.empty:
        return
    with st.expander("Momentum chart", expanded=True):
        st.altair_chart(
            ordered_bar_chart(
                momentum,
                x_column="team",
                y_column="momentum_score",
                y_title="Momentum Score",
                color="#0f766e",
            ),
            use_container_width=True,
        )


def render_form_distribution(trends: pd.DataFrame) -> None:
    form = form_distribution_data(trends)
    if form.empty:
        return
    with st.expander("Recent form mix", expanded=False):
        st.altair_chart(form_distribution_chart(form), use_container_width=True)


def form_distribution_data(trends: pd.DataFrame) -> pd.DataFrame:
    required = {"momentum_label", "power_rank_movement"}
    if trends.empty or not required.issubset(trends.columns):
        return pd.DataFrame(columns=["Form", "Teams"])
    form_order = ["Surging", "Improving", "Recovering", "Steady", "Cooling"]
    display = trends.copy()
    display["Form"] = display.apply(
        lambda row: display_form_state(row["momentum_label"], row["power_rank_movement"]),
        axis=1,
    )
    counts = display["Form"].value_counts().reindex(form_order, fill_value=0)
    return pd.DataFrame({"Form": counts.index, "Teams": counts.values})


def form_distribution_chart(form: pd.DataFrame) -> alt.Chart:
    colors = ["#16a34a", "#0ea5e9", "#f59e0b", "#6b7280", "#dc2626"]
    return (
        alt.Chart(form)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Form:N", sort=form["Form"].tolist(), title=None),
            x=alt.X("Teams:Q", title="Teams"),
            color=alt.Color(
                "Form:N",
                scale=alt.Scale(domain=form["Form"].tolist(), range=colors),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Form:N", title="Form"),
                alt.Tooltip("Teams:Q", title="Teams"),
            ],
        )
        .properties(height=190)
    )


def ordered_bar_chart(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    y_title: str,
    color: str = "#2563eb",
) -> alt.Chart:
    ordered_categories = frame[x_column].astype(str).tolist()
    return (
        alt.Chart(frame)
        .mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                f"{x_column}:N",
                sort=ordered_categories,
                title=None,
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y(f"{y_column}:Q", title=y_title),
            tooltip=[
                alt.Tooltip(f"{x_column}:N", title="Team"),
                alt.Tooltip(f"{y_column}:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=260)
    )


def render_trend_table(trends: pd.DataFrame) -> None:
    display = trend_display(trends)
    st.dataframe(
        style_trend_display(display),
        use_container_width=True,
        hide_index=True,
    )


def trend_display(trends: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "team",
        "momentum_label",
        "last_3_win_pct",
        "last_5_win_pct",
        "recent_avg_margin",
        "recent_offense_rating",
        "recent_defense_rating",
        "current_power_rank",
        "prior_power_rank",
        "power_rank_movement",
    ]
    display = trends[[column for column in columns if column in trends.columns]].copy()
    if {"momentum_label", "power_rank_movement"}.issubset(display.columns):
        display["momentum_label"] = display.apply(
            lambda row: display_form_state(row["momentum_label"], row["power_rank_movement"]),
            axis=1,
        )
    for column in ["last_3_win_pct", "last_5_win_pct"]:
        if column in display.columns:
            display[column] = (display[column] * 100).round(0).astype(int).astype(str) + "%"
    for column in ["recent_avg_margin", "recent_offense_rating", "recent_defense_rating"]:
        if column in display.columns:
            display[column] = display[column].round(1)
    for column in ["current_power_rank", "prior_power_rank"]:
        if column in display.columns:
            display[column] = display[column].round(0).astype("Int64")
    if "power_rank_movement" in display.columns:
        display["power_rank_movement"] = display["power_rank_movement"].map(format_rank_movement)
    rename = {
        "team": "Team",
        "momentum_label": "Form",
        "last_3_win_pct": "Last 3",
        "last_5_win_pct": "Last 5",
        "recent_avg_margin": "Margin",
        "recent_offense_rating": "Offense",
        "recent_defense_rating": "Defense",
        "current_power_rank": "Current Rank",
        "prior_power_rank": "Prior Rank",
        "power_rank_movement": "Rank Move",
    }
    return display.rename(columns=rename)


def style_trend_display(display: pd.DataFrame) -> pd.io.formats.style.Styler:
    styled_columns = [
        column
        for column in ["Form", "Rank Move"]
        if column in display.columns
    ]
    if not styled_columns:
        return display.style
    return display.style.map(trend_cell_style, subset=styled_columns)


def trend_cell_style(value: object) -> str:
    text = str(value)
    if text in {"<NA>", "nan", "None"}:
        return ""
    if text == "Surging":
        return "background-color: #dcfce7; color: #166534;"
    if text == "Improving":
        return "background-color: #e0f2fe; color: #075985;"
    if text == "Recovering":
        return "background-color: #fef3c7; color: #92400e;"
    if text == "Steady":
        return "background-color: #f3f4f6; color: #374151;"
    if text == "Cooling":
        return "background-color: #fecaca; color: #7f1d1d;"
    if text.startswith("↑"):
        return "background-color: #dcfce7; color: #166534;"
    if text.startswith("↓"):
        return "background-color: #fee2e2; color: #991b1b;"
    if text.startswith("→"):
        return "background-color: #f3f4f6; color: #374151;"
    return ""


def format_rank_movement(value: object) -> str:
    if pd.isna(value):
        return "→ 0"
    movement = int(round(float(value)))
    if movement > 0:
        return f"↑ +{movement}"
    if movement < 0:
        return f"↓ {movement}"
    return "→ 0"


def display_form_state(label: object, rank_movement: object) -> str:
    base = str(label or "Steady").strip().title()
    try:
        movement = float(rank_movement)
    except (TypeError, ValueError):
        movement = 0.0

    if base == "Surging":
        return "Surging" if movement >= 0 else "Cooling"
    if base == "Steady":
        return "Improving" if movement > 0 else "Steady"
    if base == "Cooling":
        return "Recovering" if movement > 0 else "Cooling"
    if base == "Improving":
        return "Improving"
    if base == "Recovering":
        return "Recovering"
    return base


def momentum_indicator(label: object, rank_movement: object) -> str:
    value = str(label or "Steady")
    try:
        movement = float(rank_movement)
    except (TypeError, ValueError):
        movement = 0.0
    if movement > 0:
        return f"{value} ↑"
    if movement < 0:
        return f"{value} ↓"
    return value


def render_matchup_predictor(
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
) -> None:
    st.divider()
    st.subheader("Matchup Predictor")
    teams = ratings["team"].dropna().sort_values().tolist()
    if len(teams) < 2:
        st.info("At least two rated teams are required for matchup predictions.")
        return

    with st.container(border=True):
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
    favorite_probability = max(
        power_context["team_a_probability"],
        power_context["team_b_probability"],
    )
    edge_label = prediction_edge_label(favorite_probability)
    explanation = generate_matchup_explanation(
        team_a,
        team_b,
        predicted_winner=str(power_context["predicted_winner"]),
        win_probability=favorite_probability,
        confidence_tier=prediction.confidence_level,
        power_ratings=power_ratings,
        trends=trends,
        sos=sos,
    )
    team_colors = matchup_team_colors(team_a, team_b)
    render_matchup_summary_cards(
        team_a,
        team_b,
        prediction,
        power_context,
        power_ratings,
        trends,
        edge_label,
        team_colors,
    )
    render_projected_score_summary(team_a, team_b, prediction, team_colors)
    render_win_probability_bar(team_a, team_b, power_context, team_colors)
    render_projected_score_comparison(team_a, team_b, prediction, team_colors)
    render_what_model_sees(
        matchup_story_observations(
            team_a,
            team_b,
            prediction,
            power_context,
            power_ratings,
            trends,
            sos,
        ),
        title="What the Model Sees",
    )
    render_team_profile_comparison(team_a, team_b, power_ratings, trends, sos)
    render_matchup_strength_comparison(team_a, team_b, power_ratings, team_colors)

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


def matchup_team_colors(team_a: str, team_b: str) -> dict[str, str]:
    return {team_a: MATCHUP_TEAM_COLORS[0], team_b: MATCHUP_TEAM_COLORS[1]}


def render_matchup_summary_cards(
    team_a: str,
    team_b: str,
    prediction,
    power_context: dict[str, object],
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
    edge_label: str,
    team_colors: dict[str, str],
) -> None:
    favorite = str(power_context["predicted_winner"])
    archetype = matchup_archetype_label(
        team_a,
        team_b,
        prediction,
        power_context,
        power_ratings,
        trends,
    )
    fan_confidence = fan_confidence_label(prediction.confidence_level, max(
        float(power_context["team_a_probability"]),
        float(power_context["team_b_probability"]),
    ))
    st.markdown(
        f"{edge_badge(edge_label)} {story_badge(archetype)} {story_badge(fan_confidence)}",
        unsafe_allow_html=True,
    )
    st.caption(matchup_preview_blurb(team_a, team_b, favorite, archetype, power_ratings, trends))
    col1, col2 = st.columns(2)
    cards = matchup_summary_card_data(team_a, team_b, prediction, power_context, power_ratings, trends)
    for column, card in zip([col1, col2], cards, strict=True):
        with column:
            render_team_summary_card(card, favored=card["Team"] == favorite, color=team_colors[card["Team"]])


def matchup_summary_card_data(
    team_a: str,
    team_b: str,
    prediction,
    power_context: dict[str, object],
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
) -> list[dict[str, str]]:
    rows = []
    for team, probability, projected_goals in [
        (team_a, power_context["team_a_probability"], prediction.projected_team_a_goals),
        (team_b, power_context["team_b_probability"], prediction.projected_team_b_goals),
    ]:
        power_row = find_team_row(power_ratings, team)
        trend_row = find_team_row(trends, team)
        rows.append(
            {
                "Team": team,
                "Win Probability": f"{float(probability):.1%}",
                "Projected Goals": f"{float(projected_goals):.1f}",
                "Power Rank": format_optional_int(power_rank_value(power_row)),
                "Recent Form": team_recent_form(trend_row),
            }
        )
    return rows


def render_team_summary_card(card: dict[str, str], *, favored: bool, color: str) -> None:
    border = color if favored else "#e5e7eb"
    label = "Favored" if favored else "Underdog"
    st.markdown(
        f"""
        <div style="border:1px solid {border};border-left:6px solid {color};
                    border-radius:8px;padding:0.85rem 0.9rem;margin-bottom:0.45rem;
                    background:#ffffff;">
          <div style="font-size:0.78rem;font-weight:700;color:{color};text-transform:uppercase;
                      letter-spacing:0.03rem;">{label}</div>
          <div style="font-size:1rem;font-weight:750;color:#111827;line-height:1.25;
                      overflow-wrap:anywhere;margin:0.15rem 0 0.6rem;">
            {card["Team"]}
          </div>
          <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.45rem;">
            <div><span style="font-size:0.72rem;color:#6b7280;">Win Prob</span><br>
              <strong>{card["Win Probability"]}</strong></div>
            <div><span style="font-size:0.72rem;color:#6b7280;">Proj Goals</span><br>
              <strong>{card["Projected Goals"]}</strong></div>
            <div><span style="font-size:0.72rem;color:#6b7280;">Power Rank</span><br>
              <strong>{card["Power Rank"]}</strong></div>
            <div><span style="font-size:0.72rem;color:#6b7280;">Recent Form</span><br>
              <strong>{card["Recent Form"]}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_projected_score_summary(
    team_a: str,
    team_b: str,
    prediction,
    team_colors: dict[str, str],
) -> None:
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:0.85rem 1rem;
                    background:#f9fafb;margin:0.5rem 0 0.8rem;">
          <div style="font-size:0.78rem;font-weight:700;color:#4b5563;text-transform:uppercase;
                      letter-spacing:0.03rem;">Projected Score</div>
          <div style="display:flex;flex-wrap:wrap;gap:0.75rem;align-items:baseline;margin-top:0.25rem;">
            <span style="color:{team_colors[team_a]};font-weight:750;overflow-wrap:anywhere;">
              {team_a}: {prediction.projected_team_a_goals:.1f}
            </span>
            <span style="color:#6b7280;font-weight:650;">vs</span>
            <span style="color:{team_colors[team_b]};font-weight:750;overflow-wrap:anywhere;">
              {team_b}: {prediction.projected_team_b_goals:.1f}
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_win_probability_bar(
    team_a: str,
    team_b: str,
    power_context: dict[str, object],
    team_colors: dict[str, str],
) -> None:
    probabilities = matchup_probability_data(team_a, team_b, power_context)
    if probabilities.empty:
        return
    st.markdown("**Win Probability**")
    st.altair_chart(matchup_probability_chart(probabilities, team_colors), use_container_width=True)


def matchup_probability_data(
    team_a: str,
    team_b: str,
    power_context: dict[str, object],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Team": team_a, "Probability": float(power_context["team_a_probability"])},
            {"Team": team_b, "Probability": float(power_context["team_b_probability"])},
        ]
    )


def matchup_probability_chart(probabilities: pd.DataFrame, team_colors: dict[str, str]) -> alt.Chart:
    ordered = probabilities.sort_values("Probability", ascending=False)
    return (
        alt.Chart(ordered)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Team:N", sort=ordered["Team"].tolist(), title=None),
            x=alt.X("Probability:Q", title="Win Probability", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Team:N",
                scale=alt.Scale(domain=list(team_colors.keys()), range=list(team_colors.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("Probability:Q", title="Win Probability", format=".1%"),
            ],
        )
        .properties(height=120)
    )


def render_projected_score_comparison(
    team_a: str,
    team_b: str,
    prediction,
    team_colors: dict[str, str],
) -> None:
    scores = projected_score_data(team_a, team_b, prediction)
    st.markdown("**Projected Goals**")
    st.altair_chart(projected_score_chart(scores, team_colors), use_container_width=True)


def projected_score_data(team_a: str, team_b: str, prediction) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Team": team_a, "Projected Goals": float(prediction.projected_team_a_goals)},
            {"Team": team_b, "Projected Goals": float(prediction.projected_team_b_goals)},
        ]
    )


def projected_score_chart(scores: pd.DataFrame, team_colors: dict[str, str]) -> alt.Chart:
    ordered = scores.sort_values("Projected Goals", ascending=False)
    return (
        alt.Chart(ordered)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Team:N", sort=ordered["Team"].tolist(), title=None),
            x=alt.X("Projected Goals:Q", title="Projected Goals"),
            color=alt.Color(
                "Team:N",
                scale=alt.Scale(domain=list(team_colors.keys()), range=list(team_colors.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("Projected Goals:Q", title="Projected Goals", format=".1f"),
            ],
        )
        .properties(height=120)
    )


def render_team_profile_comparison(
    team_a: str,
    team_b: str,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
    sos: pd.DataFrame,
) -> None:
    profile = team_profile_comparison_data(team_a, team_b, power_ratings, trends, sos)
    if profile.empty:
        return
    st.markdown("**Team Profiles**")
    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True,
    )


def team_profile_comparison_data(
    team_a: str,
    team_b: str,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
    sos: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for team in [team_a, team_b]:
        power_row = find_team_row(power_ratings, team)
        trend_row = find_team_row(trends, team)
        sos_row = find_team_row(sos, team)
        rows.append(
            {
                "Team": team,
                "Power Rank": format_optional_int(power_rank_value(power_row)),
                "Power Rating": format_optional_float(power_rating_value(power_row)),
                "Recent Form": team_recent_form(trend_row),
                "Recent Rank Move": team_recent_rank_move(trend_row),
                "Offense Strength": format_row_float(power_row, "adjusted_offense_rating"),
                "Defense Strength": format_row_float(power_row, "adjusted_defense_rating"),
                "SOS Rank": format_row_int(sos_row, "sos_rank"),
            }
        )
    return pd.DataFrame(rows)


def team_recent_form(row: pd.Series | None) -> str:
    if row is None:
        return "N/A"
    if "momentum_label" not in row or "power_rank_movement" not in row:
        return "N/A"
    return display_form_state(row["momentum_label"], row["power_rank_movement"])


def team_recent_rank_move(row: pd.Series | None) -> str:
    if row is None or "power_rank_movement" not in row:
        return "N/A"
    return format_rank_movement(row["power_rank_movement"])


def format_optional_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"


def format_optional_int(value: int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return str(int(value))


def format_row_float(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return "N/A"
    return f"{float(row[column]):.2f}"


def format_row_int(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return "N/A"
    return str(int(row[column]))


def render_matchup_strength_comparison(
    team_a: str,
    team_b: str,
    power_ratings: pd.DataFrame,
    team_colors: dict[str, str],
) -> None:
    strengths = matchup_strength_data(team_a, team_b, power_ratings)
    if strengths.empty:
        return
    st.markdown("**Offense vs Defense**")
    st.caption(
        "Offense Strength: higher is better. Defense Strength: higher is better here because it "
        "rewards allowing fewer goals than opponents usually score."
    )
    st.altair_chart(matchup_strength_chart(strengths, team_colors), use_container_width=True)


def matchup_strength_data(team_a: str, team_b: str, power_ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team in [team_a, team_b]:
        row = find_team_row(power_ratings, team)
        if row is None:
            continue
        for source_column, label in [
            ("adjusted_offense_rating", "Offense Strength"),
            ("adjusted_defense_rating", "Defense Strength"),
        ]:
            if source_column in row and pd.notna(row[source_column]):
                rows.append({"Team": team, "Metric": label, "Rating": float(row[source_column])})
    return pd.DataFrame(rows, columns=["Team", "Metric", "Rating"])


def matchup_strength_chart(strengths: pd.DataFrame, team_colors: dict[str, str]) -> alt.Chart:
    return (
        alt.Chart(strengths)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Metric:N", title=None),
            y=alt.Y("Rating:Q", title="Rating"),
            color=alt.Color(
                "Team:N",
                scale=alt.Scale(domain=list(team_colors.keys()), range=list(team_colors.values())),
            ),
            xOffset="Team:N",
            tooltip=[
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Rating:Q", title="Rating", format=".2f"),
            ],
        )
        .properties(height=240)
    )


def render_weekly_matchups(
    scheduled_games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
) -> None:
    st.divider()
    st.subheader("This Week's Matchups")
    weekly_matchups = build_weekly_matchups(
        scheduled_games,
        ratings,
        team_games,
        sos,
        power_ratings,
        trends,
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
        column_config={
            "What Model Sees": st.column_config.TextColumn("What the Model Sees", width="large"),
            "Explanation": st.column_config.TextColumn("Preview", width="large"),
        },
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Compact matchup cards", expanded=False):
        render_weekly_matchup_cards(weekly_matchups)


def build_weekly_matchups(
    scheduled_games: pd.DataFrame,
    ratings: pd.DataFrame,
    team_games: pd.DataFrame,
    sos: pd.DataFrame,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
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
        "Edge",
        "Projected Spread",
        "Projected Total",
        "Confidence",
        "Matchup Type",
        "What Model Sees",
        "Explanation",
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
            "Edge": "Unavailable",
            "Projected Spread": "",
            "Projected Total": "",
            "Confidence": "",
            "Matchup Type": "",
            "What Model Sees": "",
            "Explanation": "Unavailable",
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
            favorite_probability = max(
                power_context["team_a_probability"],
                power_context["team_b_probability"],
            )
            row.update(
                {
                    "Projected Winner": power_context["predicted_winner"],
                    "Win Probability": f"{favorite_probability:.1%}",
                    "Edge": prediction_edge_label(favorite_probability),
                    "Projected Spread": prediction.projected_spread,
                    "Projected Total": f"{prediction.projected_total_goals:.1f}",
                    "Confidence": matchup_confidence_tier(home_team, away_team, power_ratings),
                    "Matchup Type": matchup_archetype_label(
                        home_team,
                        away_team,
                        prediction,
                        power_context,
                        power_ratings,
                        trends,
                    ),
                    "What Model Sees": " ".join(
                        matchup_story_observations(
                            home_team,
                            away_team,
                            prediction,
                            power_context,
                            power_ratings,
                            trends,
                            sos,
                        )[:2]
                    ),
                    "Explanation": generate_matchup_explanation(
                        home_team,
                        away_team,
                        predicted_winner=str(power_context["predicted_winner"]),
                        win_probability=favorite_probability,
                        confidence_tier=matchup_confidence_tier(home_team, away_team, power_ratings),
                        power_ratings=power_ratings,
                        trends=trends,
                        sos=sos,
                    ),
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


def render_weekly_matchup_cards(weekly_matchups: pd.DataFrame) -> None:
    for _, matchup in weekly_matchups.iterrows():
        with st.container(border=True):
            st.markdown(
                f"**{matchup['Home']} vs {matchup['Away']}**  \n"
                f"{matchup['Date']} {matchup['Time']}"
            )
            st.markdown(
                f"{edge_badge(matchup['Edge'])} {story_badge(matchup.get('Matchup Type', ''))} "
                f"{story_badge(fan_confidence_label(matchup['Confidence']))}",
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            cols[0].metric("Projected Winner", matchup["Projected Winner"] or "Unavailable")
            cols[1].metric("Win Probability", matchup["Win Probability"] or "N/A")
            cols = st.columns(3)
            cols[0].metric("Edge", matchup["Edge"] or "Unavailable")
            cols[1].metric("Spread", matchup["Projected Spread"] or "N/A")
            cols[2].metric("Total", matchup["Projected Total"] or "N/A")
            if matchup.get("What Model Sees"):
                st.markdown("**What the Model Sees**")
                st.caption(matchup["What Model Sees"])
            st.caption(f"Read: {fan_confidence_label(matchup['Confidence'])}")
            st.caption(matchup["Explanation"] or "Explanation unavailable.")
            if matchup["Note"]:
                st.caption(matchup["Note"])


def filter_matchups_by_team(matchups: pd.DataFrame, team_filter: str) -> pd.DataFrame:
    if matchups.empty or not team_filter.strip():
        return matchups
    pattern = team_filter.strip()
    return matchups[
        matchups["Home"].str.contains(pattern, case=False, na=False, regex=False)
        | matchups["Away"].str.contains(pattern, case=False, na=False, regex=False)
    ].copy()


def prediction_edge_label(win_probability: float | None) -> str:
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
    value = str(label or "Unavailable")
    tones = {
        "Strong Edge": ("#166534", "#dcfce7"),
        "Competitive Matchup": ("#075985", "#e0f2fe"),
        "Toss-Up": ("#374151", "#f3f4f6"),
        "Anything Can Happen": ("#92400e", "#fef3c7"),
        "Defensive Grinder": ("#075985", "#e0f2fe"),
        "Track Meet": ("#9a3412", "#ffedd5"),
        "Heavyweight Battle": ("#166534", "#dcfce7"),
        "Momentum Clash": ("#7c3aed", "#ede9fe"),
        "Defense vs Firepower": ("#0f766e", "#ccfbf1"),
        "Tight Contest": ("#374151", "#f3f4f6"),
        "Upset Watch": ("#92400e", "#fef3c7"),
        "Emerging Contender Matchup": ("#075985", "#e0f2fe"),
        "Fast-Paced Battle": ("#9a3412", "#ffedd5"),
        "Physical Defensive Battle": ("#075985", "#dbeafe"),
        "Closer Than It Looks": ("#92400e", "#fef3c7"),
        "Unavailable": ("#4b5563", "#f3f4f6"),
    }
    color, background = tones.get(value, ("#374151", "#f3f4f6"))
    return metric_badge(value, color=color, background=background)


def fan_confidence_label(confidence: object, favorite_probability: float | None = None) -> str:
    if favorite_probability is not None and not pd.isna(favorite_probability):
        if max(float(favorite_probability), 1.0 - float(favorite_probability)) < 0.55:
            return "Toss-Up"
    value = str(confidence or "").strip().casefold()
    if value == "high":
        return "Strong Edge"
    if value == "medium":
        return "Competitive Matchup"
    if value in {"low", "very low"}:
        return "Anything Can Happen"
    return "Toss-Up" if value == "toss-up" else "Unavailable"


def confidence_badge(label: object) -> str:
    value = str(label or "Unavailable").title()
    tones = {
        "High": ("#166534", "#dcfce7"),
        "Medium": ("#92400e", "#fef3c7"),
        "Low": ("#991b1b", "#fee2e2"),
        "Very Low": ("#991b1b", "#fee2e2"),
        "Unavailable": ("#4b5563", "#f3f4f6"),
    }
    color, background = tones.get(value, tones["Unavailable"])
    return metric_badge(f"Confidence: {value}", color=color, background=background)


def metric_badge(label: str, *, color: str, background: str) -> str:
    return (
        f"<span style='display:inline-block;padding:0.18rem 0.55rem;"
        f"border-radius:999px;font-size:0.82rem;font-weight:650;"
        f"color:{color};background:{background};margin-right:0.25rem;'>{label}</span>"
    )


def render_what_model_sees(observations: list[str], *, title: str = "What the Model Sees") -> None:
    if not observations:
        return
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for observation in observations[:4]:
            st.caption(observation)


def matchup_archetype_label(
    team_a: str,
    team_b: str,
    prediction,
    power_context: dict[str, object],
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
) -> str:
    del team_a, team_b
    favorite_probability = max(
        float(power_context["team_a_probability"]),
        float(power_context["team_b_probability"]),
    )
    expected_total = float(getattr(prediction, "projected_total_goals", 0.0) or 0.0)
    spread = abs(float(getattr(prediction, "projected_margin", 0.0) or 0.0))
    team_a_row = find_team_row(power_ratings, str(power_context.get("predicted_winner", "")))
    favorite_rank = power_rank_value(team_a_row)
    forms = [
        team_recent_form(find_team_row(trends, str(power_context.get("predicted_winner", "")))),
    ]

    if favorite_probability < 0.55:
        return "Tight Contest"
    if favorite_probability < 0.62 and spread >= 3.0:
        return "Closer Than It Looks"
    if expected_total >= 14.0 and spread <= 4.0:
        return "Track Meet"
    if expected_total >= 14.0:
        return "Fast-Paced Battle"
    if expected_total <= 9.0 and spread <= 3.0:
        return "Defensive Grinder"
    if expected_total <= 9.0:
        return "Physical Defensive Battle"
    if favorite_probability < 0.65 and spread >= 4.5:
        return "Upset Watch"
    if favorite_rank is not None and favorite_rank <= 5 and favorite_probability >= 0.68:
        return "Heavyweight Battle"
    if any(form in {"Surging", "Improving"} for form in forms) and favorite_probability < 0.70:
        return "Momentum Clash"
    return "Defense vs Firepower" if spread >= 3.0 else "Emerging Contender Matchup"


def matchup_story_observations(
    team_a: str,
    team_b: str,
    prediction,
    power_context: dict[str, object],
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
    sos: pd.DataFrame,
) -> list[str]:
    observations: list[str] = []
    favorite = str(power_context["predicted_winner"])
    underdog = team_b if favorite == team_a else team_a
    favorite_probability = max(
        float(power_context["team_a_probability"]),
        float(power_context["team_b_probability"]),
    )
    expected_total = float(getattr(prediction, "projected_total_goals", 0.0) or 0.0)
    spread = abs(float(getattr(prediction, "projected_margin", 0.0) or 0.0))

    favorite_row = find_team_row(power_ratings, favorite)
    underdog_row = find_team_row(power_ratings, underdog)
    favorite_trend = find_team_row(trends, favorite)
    underdog_trend = find_team_row(trends, underdog)
    favorite_sos = find_team_row(sos, favorite)

    favorite_rank = power_rank_value(favorite_row)
    if favorite_rank is not None and favorite_rank <= 5:
        observations.append(f"{favorite} brings one of the division's strongest overall profiles into this game.")

    defense_rank = metric_rank_for_dashboard(power_ratings, "adjusted_defense_rating", favorite, ascending=False)
    offense_rank = metric_rank_for_dashboard(power_ratings, "adjusted_offense_rating", favorite, ascending=False)
    underdog_offense_rank = metric_rank_for_dashboard(power_ratings, "adjusted_offense_rating", underdog, ascending=False)
    if defense_rank is not None and defense_rank <= 5 and underdog_offense_rank is not None and underdog_offense_rank <= 8:
        observations.append("This has a Defense vs Firepower feel: a strong defensive profile meets a team that can score.")
    elif defense_rank is not None and defense_rank <= 5:
        observations.append(f"{favorite}'s defense continues to be one of the matchup's clearest strengths.")
    elif offense_rank is not None and offense_rank <= 5:
        observations.append(f"{favorite} has the firepower to put pressure on the scoreboard early.")

    favorite_form = team_recent_form(favorite_trend)
    underdog_form = team_recent_form(underdog_trend)
    if favorite_form in {"Surging", "Improving"} and underdog_form in {"Surging", "Improving"}:
        observations.append("Both teams come in with positive momentum, giving this a momentum-clash feel.")
    elif favorite_form in {"Surging", "Improving"}:
        observations.append(notification_phrase_for_team(favorite, favorite_form))
    elif underdog_form in {"Surging", "Improving"}:
        observations.append(f"{underdog} has enough recent momentum to make this more interesting than the ranking gap suggests.")

    sos_rank = row_int_for_dashboard(favorite_sos, "sos_rank")
    if sos_rank is not None and sos_rank <= 8:
        observations.append(f"{favorite} looks battle-tested against a tougher schedule.")

    if favorite_probability < 0.55:
        observations.append("The model sees a tight contest where small swings could decide it.")
    elif favorite_probability < 0.62 or (spread >= 4.0 and favorite_probability < 0.68):
        observations.append("This one is closer than it looks on paper, so the favorite should not feel automatic.")

    if expected_total >= 14.0:
        observations.append("The scoring environment points toward a faster-paced game.")
    elif expected_total <= 9.0:
        observations.append("Goals may be harder to come by, with defense likely shaping the game.")

    return dedupe_text(observations)[:4]


def matchup_preview_blurb(
    team_a: str,
    team_b: str,
    favorite: str,
    archetype: str,
    power_ratings: pd.DataFrame,
    trends: pd.DataFrame,
) -> str:
    favorite_trend = team_recent_form(find_team_row(trends, favorite))
    favorite_row = find_team_row(power_ratings, favorite)
    rank = power_rank_value(favorite_row)
    if favorite_trend in {"Surging", "Improving"}:
        return f"{favorite} enters with momentum, but the {archetype.lower()} profile keeps the matchup worth watching."
    if rank is not None and rank <= 5:
        return f"{favorite} brings a top-tier profile into a {archetype.lower()} against {team_b if favorite == team_a else team_a}."
    return f"{team_a} and {team_b} profile as a {archetype.lower()} with the model leaning toward {favorite}."


def notification_phrase_for_team(team: str, form: str) -> str:
    phrases = {
        "Surging": f"{team} is one of the hotter teams in the division right now.",
        "Improving": f"{team} has momentum trending upward.",
        "Recovering": f"{team} is starting to steady itself after a tougher stretch.",
        "Steady": f"{team} has been consistent in recent results.",
        "Cooling": f"{team} is looking to reverse a cooling stretch.",
    }
    return phrases.get(form, f"{team} has a balanced recent profile.")


def metric_rank_for_dashboard(frame: pd.DataFrame, column: str, team: str, *, ascending: bool) -> int | None:
    if frame.empty or column not in frame.columns or "team" not in frame.columns:
        return None
    ranked = frame.dropna(subset=[column]).sort_values([column, "team"], ascending=[ascending, True]).reset_index(drop=True)
    matches = ranked[ranked["team"].map(normalize_team_name) == normalize_team_name(team)]
    if matches.empty:
        return None
    return int(matches.index[0]) + 1


def row_int_for_dashboard(row: pd.Series | None, column: str) -> int | None:
    if row is None or column not in row or pd.isna(row[column]):
        return None
    return int(row[column])


def dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


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
        team_a_probability = calibrated_power_v3_probability(team_a_rating - team_b_rating)
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
    st.divider()
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
        st.markdown(confidence_badge(power_row["confidence_tier"]), unsafe_allow_html=True)
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

    render_offense_defense_chart(power_row)
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


def render_offense_defense_chart(power_row: pd.Series | None) -> None:
    if power_row is None:
        return
    required = ["adjusted_offense_rating", "adjusted_defense_rating"]
    if any(column not in power_row or pd.isna(power_row[column]) for column in required):
        return
    chart = pd.DataFrame(
        [
            {"Metric": "Offense", "Rating": float(power_row["adjusted_offense_rating"])},
            {"Metric": "Defense", "Rating": float(power_row["adjusted_defense_rating"])},
        ]
    )
    with st.expander("Offense vs Defense", expanded=False):
        st.bar_chart(chart, x="Metric", y="Rating", height=220)


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
    render_model_comparison_content(model_comparison_summary)


def render_model_comparison_content(model_comparison_summary: pd.DataFrame) -> None:
    if model_comparison_summary.empty:
        st.info("Model comparison summary is not available.")
        return

    st.caption("Calibrated Power Rating probabilities are primary; Power v3 baseline is shown for comparison.")
    col1, col2 = st.columns(2)
    col1.metric(
        "Calibrated Power Rating",
        metric_value(model_comparison_summary, POWER_ACCURACY_KEY, percentage=True),
        f"Brier {metric_value(model_comparison_summary, POWER_BRIER_KEY)}",
    )
    col2.metric(
        "Power v3 Baseline",
        metric_value(model_comparison_summary, "power_v3_recency_accuracy", percentage=True),
        f"Brier {metric_value(model_comparison_summary, 'power_v3_recency_brier_score')}",
    )


def render_model_calibration(calibration: pd.DataFrame) -> None:
    st.subheader("Model Calibration")
    render_model_calibration_content(calibration)


def render_model_calibration_content(calibration: pd.DataFrame) -> None:
    st.caption(
        "Calibration shows whether teams predicted around 60% actually win around 60% of the time."
    )
    if calibration.empty:
        st.info("Power Rating calibration is not available.")
        return

    render_calibration_chart(calibration)
    display = calibration.copy()
    percentage_columns = [
        "average_predicted_probability",
        "actual_win_rate",
        "calibration_gap",
    ]
    for column in percentage_columns:
        if column in display.columns:
            display[column] = display[column] * 100.0

    st.dataframe(
        display,
        column_config={
            "bucket": "Bucket",
            "games": "Games",
            "average_predicted_probability": st.column_config.NumberColumn(
                "Avg. Predicted",
                format="%.1f%%",
            ),
            "actual_win_rate": st.column_config.NumberColumn("Actual Win Rate", format="%.1f%%"),
            "calibration_gap": st.column_config.NumberColumn("Calibration Gap", format="%.1f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )


def render_calibration_chart(calibration: pd.DataFrame) -> None:
    required = {"bucket", "average_predicted_probability", "actual_win_rate"}
    if calibration.empty or not required.issubset(calibration.columns):
        return
    chart = calibration[list(required)].copy()
    chart["average_predicted_probability"] = chart["average_predicted_probability"] * 100.0
    chart["actual_win_rate"] = chart["actual_win_rate"] * 100.0
    chart = chart.rename(
        columns={
            "bucket": "Bucket",
            "average_predicted_probability": "Predicted",
            "actual_win_rate": "Actual",
        }
    )
    melted = chart.melt("Bucket", var_name="Series", value_name="Rate")
    st.altair_chart(
        calibration_line_chart(melted),
        use_container_width=True,
    )


def calibration_line_chart(calibration: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(calibration)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=3)
        .encode(
            x=alt.X("Bucket:N", sort=calibration["Bucket"].drop_duplicates().tolist(), title=None),
            y=alt.Y("Rate:Q", title="Win Rate", axis=alt.Axis(format=".0f"), scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Predicted", "Actual"], range=["#2563eb", "#16a34a"]),
            ),
            tooltip=[
                alt.Tooltip("Bucket:N", title="Bucket"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Rate:Q", title="Rate", format=".1f"),
            ],
        )
        .properties(height=260)
    )


def render_backtest(backtest: pd.DataFrame) -> None:
    st.subheader("Recent ELO Backtest Rows")
    render_backtest_content(backtest)


def render_backtest_content(backtest: pd.DataFrame) -> None:
    st.caption("Detailed ELO replay rows are retained for debugging and comparison context.")

    if not backtest.empty:
        st.dataframe(
            backtest.sort_values("game_date", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Backtest rows are not available.")


def render_model_insights(
    model_comparison_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    backtest: pd.DataFrame,
) -> None:
    st.divider()
    st.subheader("Model Insights")
    st.caption("Validation details are available here without taking over the main coaching workflow.")
    with st.expander("Model Comparison", expanded=True):
        render_model_comparison_content(model_comparison_summary)
    with st.expander("Model Calibration", expanded=False):
        render_model_calibration_content(calibration)
    with st.expander("Recent ELO Backtest Rows", expanded=False):
        render_backtest_content(backtest)


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
