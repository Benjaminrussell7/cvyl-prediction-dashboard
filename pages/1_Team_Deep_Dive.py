from __future__ import annotations

import html
import logging
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
from cvyl_scraper import club_branding
from cvyl_scraper.club_branding import ClubBranding


ROOT_DIR = Path(__file__).resolve().parents[1]
CLUB_BRANDING_CSV = ROOT_DIR / "data" / "processed" / "cvyl_club_branding.csv"
BRANDING_RESOLVER_PATH = "cvyl_scraper.club_branding.resolve_club_for_team"
LOGGER = logging.getLogger(__name__)


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
    branding = resolve_team_branding(team)
    accent = safe_branding_accent(
        str(branding.primary_color if branding else ""),
        str(branding.secondary_color if branding else ""),
    )
    logo_source = branding_logo_source(branding)
    with st.container(border=True):
        if accent:
            st.markdown(
                f"<div style='height:4px;border-radius:999px;background:{accent};margin-bottom:0.85rem;'></div>",
                unsafe_allow_html=True,
            )
        if logo_source:
            logo_column, text_column = st.columns([0.14, 0.86])
            with logo_column:
                st.image(logo_source, width=58)
            with text_column:
                st.subheader(team)
                render_branding_caption(branding, accent)
        else:
            st.subheader(team)
            render_branding_caption(branding, accent)
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


def render_branding_caption(branding: ClubBranding | None, accent: str) -> None:
    if branding is None:
        return
    club_name = html.escape(branding.club_name)
    if accent:
        text_color = contrast_text_color(accent)
        st.markdown(
            (
                "<span style='display:inline-block;padding:0.18rem 0.5rem;border-radius:999px;"
                f"background:{hex_to_rgba(accent, 0.12)};color:{text_color};border:1px solid {hex_to_rgba(accent, 0.32)};"
                "font-size:0.82rem;font-weight:600;'>"
                f"{club_name}</span>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.caption(club_name)


def resolve_team_branding(team: str) -> ClubBranding | None:
    clubs = load_team_deep_dive_club_branding()
    if not clubs:
        return None
    return club_branding.resolve_club_for_team(team, clubs).club


def team_branding_debug(team: str) -> dict[str, object]:
    clubs = load_team_deep_dive_club_branding()
    if not clubs:
        LOGGER.debug("Branding debug for %s: no clubs loaded", team)
        return {
            "team": team,
            "resolver_path": BRANDING_RESOLVER_PATH,
            "branding_csv_path": str(CLUB_BRANDING_CSV),
            "branding_rows_loaded": 0,
            "loaded_club_names": [],
            "branding": None,
            "resolved_club_name": "",
            "resolved_logo_path": "",
            "logo_path_exists": False,
            "primary_color": "",
            "secondary_color": "",
            "branding_applied": False,
            "notes": "No club branding records were loaded.",
        }
    resolution = club_branding.resolve_club_for_team(team, clubs)
    branding = resolution.club
    logo_path = ""
    logo_path_exists = False
    if branding is not None:
        logo_path = branding.logo_path
        logo_path_exists = resolved_logo_path_exists(branding)
    LOGGER.debug(
        "Branding debug for %s: resolved=%s logo_path=%s exists=%s notes=%s",
        team,
        bool(branding),
        logo_path,
        logo_path_exists,
        resolution.notes,
    )
    return {
        "team": team,
        "resolver_path": BRANDING_RESOLVER_PATH,
        "branding_csv_path": str(CLUB_BRANDING_CSV),
        "branding_rows_loaded": len(clubs),
        "loaded_club_names": [club.club_name for club in clubs[:5]],
        "branding": branding,
        "resolved_club_name": branding.club_name if branding else "",
        "resolved_logo_path": logo_path,
        "logo_path_exists": logo_path_exists,
        "primary_color": branding.primary_color if branding else "",
        "secondary_color": branding.secondary_color if branding else "",
        "branding_applied": bool(branding and (logo_path_exists or branding.logo_url)),
        "notes": resolution.notes or ("Branding resolved." if branding else "Branding not resolved."),
    }


def load_team_deep_dive_club_branding() -> list[ClubBranding]:
    return club_branding.load_club_branding_registry(csv_path=CLUB_BRANDING_CSV)


def branding_logo_source(branding: ClubBranding | None) -> str:
    if branding is None:
        return ""
    if branding.logo_path:
        logo_path = Path(branding.logo_path)
        if not logo_path.is_absolute():
            logo_path = ROOT_DIR / logo_path
        if logo_path.exists():
            return str(logo_path)
    if branding.logo_url.startswith(("http://", "https://")):
        return branding.logo_url
    return ""


def resolved_logo_path_exists(branding: ClubBranding) -> bool:
    if not branding.logo_path:
        return False
    logo_path = Path(branding.logo_path)
    if not logo_path.is_absolute():
        logo_path = ROOT_DIR / logo_path
    return logo_path.exists()


def safe_branding_accent(primary_color: str, secondary_color: str = "") -> str:
    for color in (primary_color, secondary_color):
        normalized = normalize_hex_color(color)
        if normalized and is_safe_accent_color(normalized):
            return normalized
    for color in (primary_color, secondary_color):
        normalized = normalize_hex_color(color)
        if normalized and not is_near_white(normalized):
            return normalized
    return ""


def normalize_hex_color(value: str) -> str:
    color = str(value or "").strip()
    if not color:
        return ""
    if not color.startswith("#"):
        color = f"#{color}"
    if len(color) == 4:
        color = "#" + "".join(character * 2 for character in color[1:])
    if len(color) != 7:
        return ""
    try:
        int(color[1:], 16)
    except ValueError:
        return ""
    return color.lower()


def is_safe_accent_color(color: str) -> bool:
    red, green, blue = hex_to_rgb(color)
    if is_near_white(color):
        return False
    if max(red, green, blue) - min(red, green, blue) < 18 and relative_luminance(color) > 0.62:
        return False
    return True


def is_near_white(color: str) -> bool:
    red, green, blue = hex_to_rgb(color)
    return red >= 235 and green >= 235 and blue >= 235


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    normalized = normalize_hex_color(color)
    if not normalized:
        return (0, 0, 0)
    return (int(normalized[1:3], 16), int(normalized[3:5], 16), int(normalized[5:7], 16))


def relative_luminance(color: str) -> float:
    red, green, blue = [channel / 255 for channel in hex_to_rgb(color)]
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_text_color(color: str) -> str:
    return "#111827" if relative_luminance(color) >= 0.55 else "#ffffff"


def hex_to_rgba(color: str, alpha: float) -> str:
    red, green, blue = hex_to_rgb(color)
    return f"rgba({red}, {green}, {blue}, {max(0.0, min(1.0, alpha)):.2f})"


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
    headline = team_headline_sentence(team, rank, form, rank_move, avg_margin)
    model_sees = team_model_observations(
        rank=rank,
        offense_rank=offense_rank,
        defense_rank=defense_rank,
        form=form,
        sos_rank=sos_rank,
        avg_margin=avg_margin,
        avg_for=avg_for,
        avg_against=avg_against,
        excluded=[headline, identity, storyline],
    )
    return {
        "headline": headline,
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
        return "Profiles as a balanced contender with strength on both ends."
    if defense_rank is not None and defense_rank <= 5:
        return "Wins with defense and control."
    if offense_rank is not None and offense_rank <= 5:
        return "Leans on scoring strength."
    if rank is not None and rank <= 6:
        return "Profiles as an upper-tier contender."
    if avg_for is not None and avg_against is not None and avg_for > avg_against:
        return "Profiles as a balanced, steady team."
    return "Still searching for consistency."


def team_storyline_sentence(
    team: str,
    form: str,
    rank_move: float | None,
    completed: pd.DataFrame,
) -> str:
    if form in {"Surging", "Improving", "Recovering", "Steady", "Cooling"}:
        if rank_move is not None and rank_move >= 3:
            return "Trending upward after recent strong results."
        if rank_move is not None and rank_move <= -3:
            return "Trying to reverse a cooling stretch."
        return {
            "Surging": "Riding one of the stronger recent runs in the division.",
            "Improving": "Momentum is trending upward.",
            "Recovering": "Starting to steady after a tougher stretch.",
            "Steady": "Holding steady in the middle of the table.",
            "Cooling": "Trying to reverse a cooling stretch.",
        }.get(form, "Recent form is still taking shape.")
    if completed.empty:
        return "The current storyline is limited because few completed scores are available."
    latest = completed.sort_values("game_date", ascending=False).iloc[0]
    result = "a win" if bool(latest["win"]) else "a loss"
    return f"{team} is coming off {result} against {latest['opponent']}."


def notification_phrase_for_team(team: str, form: str) -> str:
    shared_helper = getattr(dashboard, "notification_phrase_for_team", None)
    if callable(shared_helper):
        return str(shared_helper(team, form))
    phrases = {
        "Surging": f"{team} is one of the hotter teams in the division right now.",
        "Improving": f"{team} has momentum trending upward.",
        "Recovering": f"{team} is starting to steady itself after a tougher stretch.",
        "Steady": f"{team} has been consistent in recent results.",
        "Cooling": f"{team} is looking to reverse a cooling stretch.",
    }
    return phrases.get(form, f"{team} has a balanced recent profile.")


def team_headline_sentence(
    team: str,
    rank: int | None,
    form: str,
    rank_move: float | None,
    avg_margin: float | None,
) -> str:
    if rank is not None and rank <= 3:
        return f"{team} sits firmly in the contender tier."
    if rank_move is not None and rank_move >= 3:
        return f"{team} is climbing quickly."
    if form in {"Surging", "Improving"}:
        return f"{team} is building momentum."
    if avg_margin is not None and avg_margin >= 4:
        return f"{team} has been creating separation on the scoreboard."
    if form == "Cooling":
        return f"{team} is looking for a response."
    return f"{team} has a profile worth tracking."


def team_model_observations(
    *,
    rank: int | None,
    offense_rank: int | None,
    defense_rank: int | None,
    form: str,
    sos_rank: float | None,
    avg_margin: float | None,
    avg_for: float | None,
    avg_against: float | None,
    excluded: list[str],
) -> list[str]:
    observations: list[str] = []
    if defense_rank is not None and defense_rank <= 5:
        observations.append("Strong recent defensive profile.")
    if offense_rank is not None and offense_rank <= 5:
        observations.append("Scoring profile sits among the stronger groups.")
    if rank is not None and rank <= 6:
        observations.append("Power Rating still places them in the upper tier.")
    if sos_rank is not None and sos_rank <= 8:
        observations.append("Tougher schedule than most teams around them.")
    elif sos_rank is not None and sos_rank > 20:
        observations.append("Lower schedule difficulty than other top teams.")
    if form in {"Cooling", "Recovering"}:
        observations.append("Recent results show some volatility.")
    if avg_margin is not None and avg_margin >= 4:
        observations.append("Recent margins suggest they can pull away.")
    elif avg_margin is not None and abs(avg_margin) <= 1.5:
        observations.append("Several results point toward close-game margins.")
    if avg_against is not None and avg_against >= 8:
        observations.append("Goals allowed remain a watchout.")
    if avg_for is not None and avg_for <= 5:
        observations.append("Scoring consistency is still a question.")
    return dedupe_text(observations, excluded=excluded)[:4]


def dedupe_text(items: list[str], *, excluded: list[str] | None = None) -> list[str]:
    seen: set[str] = set()
    for item in excluded or []:
        seen.add(normalize_sentence(item))
    output: list[str] = []
    for item in items:
        key = normalize_sentence(item)
        if key and key not in seen:
            output.append(item)
            seen.add(key)
    return output


def normalize_sentence(value: object) -> str:
    return " ".join(str(value).strip().casefold().rstrip(".").split())


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
    render_season_trajectory(team, data)
    render_elo_timeline(team, data["elo_history"])
    render_league_comparison(team, data)


def render_season_trajectory(team: str, data: dict[str, pd.DataFrame]) -> None:
    source = historical_snapshot_source(data)
    snapshots = team_historical_snapshots(team, source)
    st.subheader("Season Trajectory")
    if snapshots.empty or snapshots["Snapshot"].nunique() < 2:
        st.info("Season trajectory is limited until this team has multiple weekly snapshots.")
        return

    summary = season_trajectory_summary(snapshots)
    with st.container(border=True):
        st.caption(season_trajectory_narrative(summary))
        columns = st.columns(5)
        for column, item in zip(columns, season_trajectory_callouts(summary), strict=True):
            column.metric(item["label"], item["value"])

        options = historical_comparison_options(source, team)
        comparison_teams = st.multiselect(
            "Compare trajectory with up to 3 teams",
            options,
            default=[],
            key="season_trajectory_comparison",
        )
        comparison_teams = comparison_teams[:3]
        trajectory = historical_trajectory_data(source, team, comparison_teams)
        st.altair_chart(historical_snapshot_chart(trajectory), use_container_width=True)


def historical_snapshot_source(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    snapshots = data.get("historical_snapshots", pd.DataFrame())
    if not snapshots.empty:
        return snapshots
    return dashboard.load_csv("cvyl_historical_snapshots.csv")


def team_historical_snapshots(team: str, snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or "team" not in snapshots.columns:
        return pd.DataFrame(columns=["Team", "Snapshot Date", "Snapshot", "Power Rank", "Power Rating"])
    rows = snapshots[snapshots["team"] == team].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Team", "Snapshot Date", "Snapshot", "Power Rank", "Power Rating"])
    rows["Team"] = rows["team"].astype(str)
    rows["Snapshot Date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    rows["Snapshot"] = rows["snapshot_label"].astype(str) if "snapshot_label" in rows.columns else ""
    rows["Power Rank"] = (
        pd.to_numeric(rows["power_rank"], errors="coerce") if "power_rank" in rows.columns else pd.NA
    )
    rows["Power Rating"] = (
        pd.to_numeric(rows["power_rating"], errors="coerce") if "power_rating" in rows.columns else pd.NA
    )
    rows = rows.dropna(subset=["Snapshot Date"]).sort_values("Snapshot Date")
    return rows[["Team", "Snapshot Date", "Snapshot", "Power Rank", "Power Rating"]]


def historical_comparison_options(snapshots: pd.DataFrame, selected_team: str) -> list[str]:
    if snapshots.empty or "team" not in snapshots.columns:
        return []
    return sorted(
        team
        for team in snapshots["team"].dropna().astype(str).unique().tolist()
        if team != selected_team
    )


def historical_trajectory_data(
    snapshots: pd.DataFrame,
    selected_team: str,
    comparison_teams: list[str],
) -> pd.DataFrame:
    teams = [selected_team, *comparison_teams[:3]]
    rows = pd.concat(
        [team_historical_snapshots(team, snapshots) for team in teams],
        ignore_index=True,
    )
    if rows.empty:
        return rows
    rows["Point Type"] = rows["Team"].map(lambda value: "Selected Team" if value == selected_team else "Comparison Team")
    return rows


def season_trajectory_summary(snapshots: pd.DataFrame) -> dict[str, object]:
    ranked = snapshots.dropna(subset=["Power Rank"]).sort_values("Snapshot Date")
    if ranked.empty:
        return {
            "start_label": "N/A",
            "start_rank": None,
            "current_rank": None,
            "net_movement": None,
            "best_rank": None,
            "worst_rank": None,
            "recent_movement": None,
        }
    start = ranked.iloc[0]
    current = ranked.iloc[-1]
    recent_start = ranked.iloc[-min(3, len(ranked))]
    return {
        "start_label": str(start["Snapshot"]),
        "start_rank": int(start["Power Rank"]),
        "current_rank": int(current["Power Rank"]),
        "net_movement": int(start["Power Rank"] - current["Power Rank"]),
        "best_rank": int(ranked["Power Rank"].min()),
        "worst_rank": int(ranked["Power Rank"].max()),
        "recent_movement": int(recent_start["Power Rank"] - current["Power Rank"]),
    }


def season_trajectory_callouts(summary: dict[str, object]) -> list[dict[str, str]]:
    start_label = str(summary.get("start_label") or "Week 1")
    return [
        {"label": f"Started {start_label}", "value": format_rank(summary.get("start_rank"))},
        {"label": "Current Rank", "value": format_rank(summary.get("current_rank"))},
        {"label": "Net Movement", "value": format_net_movement(summary.get("net_movement"))},
        {"label": "Best Rank", "value": format_rank(summary.get("best_rank"))},
        {"label": "Worst Rank", "value": format_rank(summary.get("worst_rank"))},
    ]


def season_trajectory_narrative(summary: dict[str, object]) -> str:
    current_rank = summary.get("current_rank")
    net_movement = summary.get("net_movement")
    recent_movement = summary.get("recent_movement")
    start_rank = summary.get("start_rank")
    if current_rank is None or net_movement is None:
        return "Season trajectory is still taking shape."
    current_rank = int(current_rank)
    net_movement = int(net_movement)
    recent_movement = int(recent_movement or 0)
    if current_rank <= 5 and abs(net_movement) <= 2:
        return "Holding near the top."
    if net_movement >= 4:
        return "Climbing steadily."
    if recent_movement >= 3:
        return "Finding momentum late."
    if net_movement <= -4 and start_rank is not None and int(start_rank) <= 8:
        return "Cooling after a strong start."
    if net_movement > 0:
        return "Moving in the right direction."
    if net_movement < 0:
        return "Trying to regain earlier form."
    return "Holding steady."


def format_rank(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"#{int(value)}"


def format_net_movement(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    movement = int(value)
    if movement > 0:
        return f"↑ {movement}"
    if movement < 0:
        return f"↓ {abs(movement)}"
    return "→ 0"


def historical_snapshot_chart(snapshots: pd.DataFrame) -> alt.Chart:
    if "Point Type" not in snapshots.columns:
        snapshots = snapshots.copy()
        snapshots["Point Type"] = "Selected Team"
    snapshot_order = snapshots.sort_values("Snapshot Date")["Snapshot"].drop_duplicates().tolist()
    rank_chart = (
        alt.Chart(snapshots)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=65), strokeWidth=3)
        .encode(
            x=alt.X("Snapshot:N", sort=snapshot_order, title=None),
            y=alt.Y("Power Rank:Q", title="Power Rank", scale=alt.Scale(reverse=True, zero=False)),
            color=historical_team_color(),
            strokeDash=historical_team_stroke_dash(),
            opacity=historical_team_opacity(),
            tooltip=[
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("Snapshot:N", title="Snapshot"),
                alt.Tooltip("Snapshot Date:T", title="Date"),
                alt.Tooltip("Power Rank:Q", title="Power Rank", format=".0f"),
                alt.Tooltip("Power Rating:Q", title="Power Rating", format=".2f"),
            ],
        )
        .properties(height=220)
    )
    rating_chart = (
        alt.Chart(snapshots)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=65), strokeWidth=3)
        .encode(
            x=alt.X("Snapshot:N", sort=snapshot_order, title=None),
            y=alt.Y("Power Rating:Q", title="Power Rating", scale=alt.Scale(zero=False)),
            color=historical_team_color(),
            strokeDash=historical_team_stroke_dash(),
            opacity=historical_team_opacity(),
            tooltip=[
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("Snapshot:N", title="Snapshot"),
                alt.Tooltip("Snapshot Date:T", title="Date"),
                alt.Tooltip("Power Rank:Q", title="Power Rank", format=".0f"),
                alt.Tooltip("Power Rating:Q", title="Power Rating", format=".2f"),
            ],
        )
        .properties(height=220)
    )
    return alt.vconcat(rank_chart, rating_chart).resolve_scale(y="independent")


def historical_team_color() -> alt.Color:
    return alt.Color(
        "Point Type:N",
        scale=alt.Scale(
            domain=["Selected Team", "Comparison Team"],
            range=["#dc2626", "#64748b"],
        ),
        legend=alt.Legend(title=None),
    )


def historical_team_stroke_dash() -> alt.StrokeDash:
    return alt.StrokeDash(
        "Point Type:N",
        scale=alt.Scale(
            domain=["Selected Team", "Comparison Team"],
            range=[[1, 0], [5, 4]],
        ),
        legend=None,
    )


def historical_team_opacity() -> alt.Opacity:
    return alt.Opacity(
        "Point Type:N",
        scale=alt.Scale(
            domain=["Selected Team", "Comparison Team"],
            range=[1.0, 0.65],
        ),
        legend=None,
    )


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
    headline = story.get("headline")
    if not headline:
        return "Current team profile is limited by available reported results."
    return str(headline)


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
