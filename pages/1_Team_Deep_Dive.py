from __future__ import annotations

import math
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import altair as alt
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

    render_team_headquarters(team, data)


def render_team_headquarters(team: str, data: dict[str, pd.DataFrame]) -> None:
    st.divider()
    render_team_header(team, data)
    render_snapshot_cards(team, data)
    render_team_story_sections(team, data)
    render_team_strengths(team, data)
    render_team_timeline(team, data)
    render_recent_results_cards(team, data)
    render_highlight_cards(team, data)
    render_remaining_schedule_difficulty(team, data)
    render_team_games_table(team, data)


def render_team_header(team: str, data: dict[str, pd.DataFrame]) -> None:
    power_row = dashboard.find_team_row(data["power_ratings"], team)
    trend_row = dashboard.find_team_row(data["trends"], team)
    with st.container(border=True):
        st.subheader(team)
        badges = [
            dashboard.confidence_badge(format_row_text(power_row, "confidence_tier")),
            dashboard.metric_badge(
                dashboard.team_recent_form(trend_row),
                color=form_text_color(dashboard.team_recent_form(trend_row)),
                background=form_background_color(dashboard.team_recent_form(trend_row)),
            ),
        ]
        st.markdown(" ".join(badges), unsafe_allow_html=True)
        st.caption(build_team_narrative(team, data))


def render_snapshot_cards(team: str, data: dict[str, pd.DataFrame]) -> None:
    snapshot = team_snapshot(team, data)
    st.subheader("Team Snapshot")
    first_row = st.columns(5)
    for column, item in zip(first_row, snapshot[:5], strict=True):
        column.metric(item["label"], item["value"])
    second_row = st.columns(4)
    for column, item in zip(second_row, snapshot[5:], strict=True):
        column.metric(item["label"], item["value"])


def team_snapshot(team: str, data: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    rating_row = dashboard.find_team_row(data["ratings"], team)
    power_row = dashboard.find_team_row(data["power_ratings"], team)
    sos_row = dashboard.find_team_row(data["sos"], team)
    trend_row = dashboard.find_team_row(data["trends"], team)
    completed = completed_games_for_team(team, data["team_games"])
    wins = int(completed["win"].sum()) if not completed.empty and "win" in completed else 0
    losses = int((~completed["win"].astype(bool)).sum()) if not completed.empty and "win" in completed else 0

    return [
        {"label": "Current Rank", "value": dashboard.format_optional_int(dashboard.power_rank_value(power_row))},
        {"label": "Record", "value": f"{wins}-{losses}"},
        {"label": "Power Rating", "value": dashboard.format_optional_float(dashboard.power_rating_value(power_row))},
        {"label": "ELO", "value": dashboard.format_row_float(rating_row, "elo")},
        {"label": "SOS Rank", "value": dashboard.format_row_int(sos_row, "sos_rank")},
        {"label": "Avg Goals Scored", "value": dashboard.format_row_float(power_row, "avg_points_for")},
        {"label": "Avg Goals Allowed", "value": dashboard.format_row_float(power_row, "avg_points_against")},
        {"label": "Avg Margin", "value": dashboard.format_row_float(power_row, "avg_margin")},
        {"label": "Momentum", "value": dashboard.team_recent_form(trend_row)},
    ]


def render_team_strengths(team: str, data: dict[str, pd.DataFrame]) -> None:
    insights = team_strength_insights(team, data)
    if not insights:
        return
    st.subheader("Model Notes")
    columns = st.columns(min(3, len(insights)))
    for index, insight in enumerate(insights):
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(f"**{insight['title']}**")
                st.caption(insight["detail"])


def render_team_story_sections(team: str, data: dict[str, pd.DataFrame]) -> None:
    story = team_storytelling(team, data)
    st.subheader("Team Identity")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Identity**")
            st.caption(story["identity"])
            st.markdown("**Recent Storyline**")
            st.caption(story["storyline"])
        with col2:
            st.markdown("**What the Model Sees**")
            for observation in story["model_sees"][:3]:
                st.caption(observation)

    st.subheader("Strengths & Watchouts")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Strengths**")
            for strength in story["strengths"]:
                st.caption(strength)
    with col2:
        with st.container(border=True):
            st.markdown("**Watchouts**")
            for watchout in story["watchouts"]:
                st.caption(watchout)


def team_storytelling(team: str, data: dict[str, pd.DataFrame]) -> dict[str, list[str] | str]:
    power = data["power_ratings"]
    power_row = dashboard.find_team_row(power, team)
    trend_row = dashboard.find_team_row(data["trends"], team)
    sos_row = dashboard.find_team_row(data["sos"], team)
    completed = completed_games_for_team(team, data["team_games"])

    offense_rank = metric_rank(power, "adjusted_offense_rating", team, ascending=False)
    defense_rank = metric_rank(power, "adjusted_defense_rating", team, ascending=False)
    rank = dashboard.power_rank_value(power_row)
    form = dashboard.team_recent_form(trend_row)
    rank_move = row_float(trend_row, "power_rank_movement")
    avg_for = row_float(power_row, "avg_points_for")
    avg_against = row_float(power_row, "avg_points_against")
    avg_margin = row_float(power_row, "avg_margin")
    sos_rank = row_float(sos_row, "sos_rank")

    identity = team_identity_sentence(team, rank, offense_rank, defense_rank, avg_for, avg_against)
    storyline = team_storyline_sentence(team, form, rank_move, completed)
    strengths = team_story_strengths(team, rank, offense_rank, defense_rank, form, sos_rank, avg_margin)
    watchouts = team_story_watchouts(team, avg_against, avg_for, avg_margin, completed, sos_rank)
    model_sees = dashboard.dedupe_text([
        identity,
        storyline,
        *strengths[:2],
        *watchouts[:1],
    ])
    return {
        "identity": identity,
        "storyline": storyline,
        "model_sees": model_sees,
        "strengths": strengths or ["The profile is balanced without one obvious headline strength yet."],
        "watchouts": watchouts or ["No major red flag stands out from currently reported results."],
    }


def team_identity_sentence(
    team: str,
    rank: int | None,
    offense_rank: int | None,
    defense_rank: int | None,
    avg_for: float | None,
    avg_against: float | None,
) -> str:
    if defense_rank is not None and defense_rank <= 5 and offense_rank is not None and offense_rank <= 8:
        return f"{team} looks like a complete contender, pairing strong defense with enough scoring punch."
    if defense_rank is not None and defense_rank <= 5:
        return f"{team} wins with defense and consistency."
    if offense_rank is not None and offense_rank <= 5:
        return f"{team} is built around offensive pressure and scoring depth."
    if rank is not None and rank <= 6:
        return f"{team} has the profile of a division contender."
    if avg_for is not None and avg_against is not None and avg_for > avg_against:
        return f"{team} has been on the right side of the scoreboard more often than not."
    return f"{team} has a developing profile as more scores are reported."


def team_storyline_sentence(
    team: str,
    form: str,
    rank_move: float | None,
    completed: pd.DataFrame,
) -> str:
    if form in {"Surging", "Improving", "Recovering", "Steady", "Cooling"}:
        phrase = dashboard.notification_phrase_for_team(team, form)
        if rank_move is not None and rank_move >= 3:
            return f"{phrase} Recent results have pushed them up the board."
        if rank_move is not None and rank_move <= -3:
            return f"{phrase} The recent rank movement still shows some turbulence."
        return phrase
    if completed.empty:
        return "The current storyline is limited because few completed scores are available."
    latest = completed.sort_values("game_date", ascending=False).iloc[0]
    result = "a win" if bool(latest["win"]) else "a loss"
    return f"{team} is coming off {result} against {latest['opponent']}."


def team_story_strengths(
    team: str,
    rank: int | None,
    offense_rank: int | None,
    defense_rank: int | None,
    form: str,
    sos_rank: float | None,
    avg_margin: float | None,
) -> list[str]:
    strengths: list[str] = []
    if rank is not None and rank <= 5:
        strengths.append(f"Top-tier overall profile: {team} sits inside the top five overall.")
    if defense_rank is not None and defense_rank <= 5:
        strengths.append("Defense continues to lead the way.")
    if offense_rank is not None and offense_rank <= 5:
        strengths.append("The offense has shown one of the stronger scoring profiles in the division.")
    if form in {"Surging", "Improving"}:
        strengths.append("Momentum is trending upward.")
    if sos_rank is not None and sos_rank <= 8:
        strengths.append("The schedule has been battle-tested.")
    if avg_margin is not None and avg_margin >= 4:
        strengths.append("Recent score margins suggest they can create separation.")
    return strengths[:4]


def team_story_watchouts(
    team: str,
    avg_against: float | None,
    avg_for: float | None,
    avg_margin: float | None,
    completed: pd.DataFrame,
    sos_rank: float | None,
) -> list[str]:
    del team
    watchouts: list[str] = []
    if avg_against is not None and avg_against >= 8:
        watchouts.append("Goals allowed remain the main thing to tighten against stronger opponents.")
    if avg_for is not None and avg_for <= 5:
        watchouts.append("Scoring depth remains a question if the game turns into a track meet.")
    if avg_margin is not None and abs(avg_margin) <= 1.5:
        watchouts.append("Many games profile as close, so small swings can matter.")
    if sos_rank is not None and sos_rank > 20:
        watchouts.append("The schedule has not been as demanding as some top-ranked teams.")
    if not completed.empty:
        recent = completed.sort_values("game_date", ascending=False).head(3)
        if len(recent) >= 3 and int(recent["win"].sum()) <= 1:
            watchouts.append("Recent form has been uneven over the last few reported games.")
    return watchouts[:4]


def team_strength_insights(team: str, data: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    power = data["power_ratings"]
    power_row = dashboard.find_team_row(power, team)
    trend_row = dashboard.find_team_row(data["trends"], team)
    sos_row = dashboard.find_team_row(data["sos"], team)
    if power_row is None:
        return []

    insights: list[dict[str, str]] = []
    rank = dashboard.power_rank_value(power_row)
    offense_rank = metric_rank(power, "adjusted_offense_rating", team, ascending=False)
    defense_rank = metric_rank(power, "adjusted_defense_rating", team, ascending=False)
    avg_against = row_float(power_row, "avg_points_against")
    form = dashboard.team_recent_form(trend_row)
    rank_move = row_float(trend_row, "power_rank_movement")
    sos_rank = row_float(sos_row, "sos_rank")

    if rank is not None and rank <= 5:
        insights.append({"title": "Elite Overall Profile", "detail": f"Current Power Rank is #{rank}."})
    if offense_rank is not None and offense_rank <= 5:
        insights.append({"title": "Top 5 Offense", "detail": f"Offense Strength ranks #{offense_rank} in the field."})
    if defense_rank is not None and defense_rank <= 5:
        insights.append({"title": "Elite Recent Defense", "detail": f"Defense Strength ranks #{defense_rank}; higher is better in this metric."})
    if form in {"Surging", "Improving", "Recovering"}:
        insights.append({"title": f"{form} Form", "detail": f"Recent trend reads as {form.lower()}."})
    if rank_move is not None and rank_move >= 3:
        insights.append({"title": "Improving Rapidly", "detail": f"Power Rank movement is {dashboard.format_rank_movement(rank_move)}."})
    if sos_rank is not None and sos_rank <= 8:
        insights.append({"title": "Battle-Tested Schedule", "detail": f"SOS Rank is #{int(sos_rank)}."})
    if avg_against is not None and avg_against >= 8:
        insights.append({"title": "Defensive Watchout", "detail": f"Average goals allowed is {avg_against:.1f}."})
    if not insights:
        insights.append({"title": "Balanced Profile", "detail": "No single metric dominates the current team profile."})
    return insights[:6]


def render_team_timeline(team: str, data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Trend Timeline")
    render_elo_timeline(team, data["elo_history"])
    render_league_comparison(team, data)


def render_elo_timeline(team: str, elo_history: pd.DataFrame) -> None:
    timeline = elo_timeline_data(team, elo_history)
    st.markdown("**ELO Over Time**")
    if timeline.empty:
        st.info("ELO history is not available for this team.")
        return
    st.altair_chart(line_chart(timeline, "Date", "ELO", "ELO", "#2563eb"), use_container_width=True)


def render_league_comparison(team: str, data: dict[str, pd.DataFrame]) -> None:
    comparison = league_comparison_data(data)
    metrics = available_comparison_metrics(comparison)
    st.subheader("League Comparison")
    st.caption(
        "Use this chart to compare the selected team against the rest of the league. "
        "Hover over nearby teams to see who profiles similarly."
    )
    st.caption(
        "Higher Power Rating is better. Lower Power Rank is better. Higher offense strength is better. "
        "Higher defense strength is better here because it rewards allowing fewer goals than opponents usually score. "
        "Lower SOS Rank means a tougher schedule."
    )
    if comparison.empty or len(metrics) < 2:
        st.info("League comparison data is not available.")
        return

    default_x = "Offensive Strength" if "Offensive Strength" in metrics else metrics[0]
    default_y = "Defensive Strength" if "Defensive Strength" in metrics else metrics[min(1, len(metrics) - 1)]
    col1, col2 = st.columns(2)
    x_metric = col1.selectbox("X-axis metric", metrics, index=metrics.index(default_x), key="league_compare_x")
    y_metric = col2.selectbox("Y-axis metric", metrics, index=metrics.index(default_y), key="league_compare_y")
    comparison_options = sorted(
        value for value in comparison["Team"].dropna().astype(str).unique().tolist()
        if value != team
    )
    comparison_teams = st.multiselect(
        "Highlight comparison teams",
        comparison_options,
        default=[],
        key="league_compare_highlights",
    )
    st.altair_chart(
        league_comparison_chart(comparison, team, comparison_teams, x_metric, y_metric),
        use_container_width=True,
    )


def elo_timeline_data(team: str, elo_history: pd.DataFrame) -> pd.DataFrame:
    if elo_history.empty:
        return pd.DataFrame(columns=["Date", "ELO"])
    rows = elo_history[elo_history["team"] == team].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Date", "ELO"])
    rows["Date"] = pd.to_datetime(rows["game_date"], errors="coerce")
    rows["ELO"] = pd.to_numeric(rows["postgame_elo"], errors="coerce")
    return rows.dropna(subset=["Date", "ELO"]).sort_values(["Date", "game_id"])[["Date", "ELO"]]


def league_comparison_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    power = data["power_ratings"].copy()
    if power.empty or "team" not in power.columns:
        return pd.DataFrame()
    comparison = power.rename(
        columns={
            dashboard.POWER_RATING_COLUMN: "Power Rating",
            dashboard.POWER_RANK_COLUMN: "Power Rank",
            "games_played": "Games Played",
            "adjusted_offense_rating": "Offensive Strength",
            "adjusted_defense_rating": "Defensive Strength",
            "avg_margin": "Recent Margin",
        }
    )
    if not data["ratings"].empty:
        comparison = comparison.merge(
            data["ratings"][["team", "elo"]].rename(columns={"elo": "ELO"}),
            on="team",
            how="left",
        )
    if not data["sos"].empty:
        comparison = comparison.merge(
            data["sos"][["team", "sos_rank"]].rename(columns={"sos_rank": "SOS Rank"}),
            on="team",
            how="left",
        )
    if not data["trends"].empty:
        trends = data["trends"].copy()
        trends["Momentum/Form"] = trends.apply(
            lambda row: dashboard.display_form_state(row["momentum_label"], row["power_rank_movement"]),
            axis=1,
        )
        comparison = comparison.merge(
            trends[
                [
                    "team",
                    "last_3_win_pct",
                    "last_5_win_pct",
                    "momentum_score",
                    "Momentum/Form",
                ]
            ].rename(
                columns={
                    "last_3_win_pct": "Last 3 Win %",
                    "last_5_win_pct": "Last 5 Win %",
                    "momentum_score": "Momentum Score",
                }
            ),
            on="team",
            how="left",
        )
    return comparison.rename(columns={"team": "Team"})


def available_comparison_metrics(comparison: pd.DataFrame) -> list[str]:
    candidates = [
        "Power Rating",
        "Power Rank",
        "ELO",
        "SOS Rank",
        "Games Played",
        "Offensive Strength",
        "Defensive Strength",
        "Recent Margin",
        "Last 3 Win %",
        "Last 5 Win %",
        "Momentum Score",
    ]
    metrics = []
    for metric in candidates:
        if metric in comparison.columns and pd.to_numeric(comparison[metric], errors="coerce").notna().any():
            metrics.append(metric)
    return metrics


def league_comparison_chart(
    comparison: pd.DataFrame,
    selected_team: str,
    comparison_teams: list[str],
    x_metric: str,
    y_metric: str,
) -> alt.Chart:
    chart_data = comparison.copy()
    chart_data[x_metric] = pd.to_numeric(chart_data[x_metric], errors="coerce")
    chart_data[y_metric] = pd.to_numeric(chart_data[y_metric], errors="coerce")
    chart_data["Point Type"] = chart_data["Team"].map(
        lambda team: comparison_point_type(str(team), selected_team, comparison_teams)
    )
    chart_data = chart_data.dropna(subset=[x_metric, y_metric])
    tooltip_columns = [
        "Team",
        x_metric,
        y_metric,
        "Power Rank",
        "Power Rating",
        "ELO",
        "Momentum/Form",
        "Games Played",
        "Point Type",
    ]
    tooltips = []
    for column in tooltip_columns:
        if column not in chart_data.columns:
            continue
        if column in {"Team", "Momentum/Form"}:
            tooltips.append(alt.Tooltip(f"{column}:N", title=column))
        else:
            tooltips.append(alt.Tooltip(f"{column}:Q", title=column, format=".2f"))
    return (
        alt.Chart(chart_data)
        .mark_circle()
        .encode(
            x=alt.X(f"{x_metric}:Q", title=x_metric, scale=axis_scale_for_metric(x_metric)),
            y=alt.Y(f"{y_metric}:Q", title=y_metric, scale=axis_scale_for_metric(y_metric)),
            color=alt.Color(
                "Point Type:N",
                scale=alt.Scale(
                    domain=["Selected Team", "Comparison Team", "League Team"],
                    range=["#dc2626", "#2563eb", "#9ca3af"],
                ),
                legend=alt.Legend(title=None),
            ),
            size=alt.Size(
                "Point Type:N",
                scale=alt.Scale(
                    domain=["Selected Team", "Comparison Team", "League Team"],
                    range=[190, 125, 55],
                ),
                legend=None,
            ),
            opacity=alt.Opacity(
                "Point Type:N",
                scale=alt.Scale(
                    domain=["Selected Team", "Comparison Team", "League Team"],
                    range=[1.0, 0.9, 0.45],
                ),
                legend=None,
            ),
            tooltip=tooltips,
        )
        .properties(height=360)
        .interactive()
    )


def axis_scale_for_metric(metric: str) -> alt.Scale:
    if metric in {"Power Rank", "SOS Rank"}:
        return alt.Scale(reverse=True, zero=False)
    return alt.Scale(zero=False)


def comparison_point_type(team: str, selected_team: str, comparison_teams: list[str]) -> str:
    if team == selected_team:
        return "Selected Team"
    if team in set(comparison_teams):
        return "Comparison Team"
    return "League Team"


def line_chart(frame: pd.DataFrame, x: str, y: str, title: str, color: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=3, color=color)
        .encode(
            x=alt.X(f"{x}:T", title=None),
            y=alt.Y(f"{y}:Q", title=title),
            tooltip=[alt.Tooltip(f"{x}:T", title="Date"), alt.Tooltip(f"{y}:Q", title=title, format=".2f")],
        )
        .properties(height=220)
    )


def render_recent_results_cards(team: str, data: dict[str, pd.DataFrame]) -> None:
    games = recent_result_cards(team, data)
    st.subheader("Recent Results")
    if games.empty:
        st.info("No completed games found for this team.")
        return
    for _, game in games.head(6).iterrows():
        render_game_card(game)


def recent_result_cards(team: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    games = completed_games_for_team(team, data["team_games"])
    if games.empty:
        return pd.DataFrame()
    games = games.sort_values("game_date", ascending=False).copy()
    games["margin"] = games["points_for"].astype(float) - games["points_against"].astype(float)
    games["opponent_rank"] = games["opponent"].map(lambda value: dashboard.power_rank_value(dashboard.find_team_row(data["power_ratings"], value)))
    games["expected_win_probability"] = games.apply(expected_win_probability_from_elo, axis=1)
    games["upset"] = games.apply(lambda row: bool(row["win"]) and row["expected_win_probability"] < 0.5, axis=1)
    return games


def expected_win_probability_from_elo(row: pd.Series) -> float:
    team_elo = row.get("pregame_elo")
    opponent_elo = row.get("opponent_pregame_elo")
    if pd.isna(team_elo) or pd.isna(opponent_elo):
        return 0.5
    return 1.0 / (1.0 + math.pow(10.0, (float(opponent_elo) - float(team_elo)) / 400.0))


def render_game_card(game: pd.Series) -> None:
    won = bool(game["win"])
    color = "#16a34a" if won else "#dc2626"
    result = "W" if won else "L"
    upset = "Upset win" if bool(game.get("upset", False)) else ""
    opponent_rank = dashboard.format_optional_int(game.get("opponent_rank"))
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="border-left:5px solid {color};padding-left:0.75rem;">
              <div style="display:flex;flex-wrap:wrap;gap:0.65rem;align-items:center;">
                <span style="font-weight:800;color:{color};">{result}</span>
                <strong>{game['points_for']}-{game['points_against']} vs {game['opponent']}</strong>
                <span style="color:#6b7280;">{pd.to_datetime(game['game_date']).date().isoformat()}</span>
              </div>
              <div style="color:#4b5563;font-size:0.9rem;margin-top:0.25rem;">
                Margin {int(game['margin']):+d} | Opponent rank {opponent_rank} |
                Expected win {float(game['expected_win_probability']):.1%}
                {" | " + upset if upset else ""}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_highlight_cards(team: str, data: dict[str, pd.DataFrame]) -> None:
    highlights = team_highlights(team, data)
    if not highlights:
        return
    st.subheader("Best Wins & Toughest Losses")
    columns = st.columns(min(3, len(highlights)))
    for index, highlight in enumerate(highlights):
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(f"**{highlight['title']}**")
                st.write(highlight["headline"])
                st.caption(highlight["detail"])


def team_highlights(team: str, data: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    games = recent_result_cards(team, data)
    if games.empty:
        return []
    wins = games[games["win"].astype(bool)].copy()
    losses = games[~games["win"].astype(bool)].copy()
    highlights: list[dict[str, str]] = []
    if not wins.empty:
        best_win = wins.sort_values(["opponent_rank", "margin"], ascending=[True, False], na_position="last").iloc[0]
        highlights.append(game_highlight("Highest-Rated Win", best_win))
        largest_win = wins.sort_values("margin", ascending=False).iloc[0]
        highlights.append(game_highlight("Largest Margin Win", largest_win))
        closest_win = wins.sort_values("margin", ascending=True).iloc[0]
        highlights.append(game_highlight("Closest Win", closest_win))
        upset_wins = wins[wins["upset"]]
        if not upset_wins.empty:
            biggest_upset = upset_wins.sort_values("expected_win_probability").iloc[0]
            highlights.append(game_highlight("Biggest Upset", biggest_upset))
    if not losses.empty:
        toughest_loss = losses.sort_values(["opponent_rank", "margin"], ascending=[True, False], na_position="last").iloc[0]
        highlights.append(game_highlight("Toughest Loss", toughest_loss))
    return highlights[:6]


def game_highlight(title: str, game: pd.Series) -> dict[str, str]:
    margin = int(game["margin"])
    return {
        "title": title,
        "headline": f"{int(game['points_for'])}-{int(game['points_against'])} vs {game['opponent']}",
        "detail": f"{pd.to_datetime(game['game_date']).date().isoformat()} | Margin {margin:+d} | Opponent rank {dashboard.format_optional_int(game.get('opponent_rank'))}",
    }


def render_remaining_schedule_difficulty(team: str, data: dict[str, pd.DataFrame]) -> None:
    schedule = remaining_schedule(team, data)
    if schedule.empty:
        return
    average_strength = float(schedule["opponent_power_rating"].mean())
    difficulty = schedule_difficulty_label(average_strength, data["power_ratings"])
    st.subheader("Remaining Schedule Difficulty")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining Games", len(schedule))
        col2.metric("Avg Opponent Power", f"{average_strength:.2f}")
        col3.metric("Difficulty", difficulty)
        st.dataframe(
            schedule[["game_date", "game_time", "opponent", "opponent_power_rank", "opponent_power_rating"]],
            column_config={
                "game_date": "Date",
                "game_time": "Time",
                "opponent": "Opponent",
                "opponent_power_rank": "Opponent Rank",
                "opponent_power_rating": st.column_config.NumberColumn("Opponent Power", format="%.2f"),
            },
            use_container_width=True,
            hide_index=True,
        )


def remaining_schedule(team: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    upcoming = dashboard.upcoming_scheduled_games_for_team(data["games"], team)
    if upcoming.empty:
        return pd.DataFrame()
    rows = []
    for _, game in upcoming.iterrows():
        opponent = game["away_team"] if game["home_team"] == team else game["home_team"]
        power_row = dashboard.find_team_row(data["power_ratings"], opponent)
        rows.append(
            {
                "game_date": game.get("game_date"),
                "game_time": game.get("game_time", ""),
                "opponent": opponent,
                "opponent_power_rank": dashboard.power_rank_value(power_row),
                "opponent_power_rating": dashboard.power_rating_value(power_row),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["opponent_power_rating"])


def schedule_difficulty_label(average_strength: float, power_ratings: pd.DataFrame) -> str:
    if power_ratings.empty or POWER_COLUMN not in power_ratings.columns:
        return "Moderate"
    upper = float(power_ratings[POWER_COLUMN].quantile(0.67))
    lower = float(power_ratings[POWER_COLUMN].quantile(0.33))
    if average_strength >= upper:
        return "Difficult"
    if average_strength <= lower:
        return "Easy"
    return "Moderate"


POWER_COLUMN = dashboard.POWER_RATING_COLUMN


def render_team_games_table(team: str, data: dict[str, pd.DataFrame]) -> None:
    with st.expander("Completed Game Log", expanded=False):
        completed_log = completed_games_for_team(team, data["team_games"])
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


def completed_games_for_team(team: str, team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame()
    games = team_games[
        (team_games["team"] == team) & (team_games["status"] == "completed")
    ].copy()
    for column in ["points_for", "points_against"]:
        if column in games.columns:
            games[column] = pd.to_numeric(games[column], errors="coerce")
    return games.dropna(subset=["points_for", "points_against"])


def build_team_narrative(team: str, data: dict[str, pd.DataFrame]) -> str:
    story = team_storytelling(team, data)
    model_sees = story.get("model_sees", [])
    if not model_sees:
        return "Current team profile is limited by available reported results."
    return " ".join(str(item) for item in model_sees[:2])


def metric_rank(frame: pd.DataFrame, column: str, team: str, *, ascending: bool) -> int | None:
    if frame.empty or column not in frame.columns:
        return None
    ranked = frame.dropna(subset=[column]).sort_values([column, "team"], ascending=[ascending, True]).reset_index(drop=True)
    matches = ranked[ranked["team"] == team]
    if matches.empty:
        return None
    return int(matches.index[0]) + 1


def row_float(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def format_row_text(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return "Unavailable"
    return str(row[column])


def form_text_color(form: str) -> str:
    return {
        "Surging": "#166534",
        "Improving": "#075985",
        "Recovering": "#92400e",
        "Steady": "#374151",
        "Cooling": "#7f1d1d",
    }.get(form, "#374151")


def form_background_color(form: str) -> str:
    return {
        "Surging": "#dcfce7",
        "Improving": "#e0f2fe",
        "Recovering": "#fef3c7",
        "Steady": "#f3f4f6",
        "Cooling": "#fecaca",
    }.get(form, "#f3f4f6")


if __name__ == "__main__":
    main()
