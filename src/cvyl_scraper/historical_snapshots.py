from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvyl_scraper.elo import build_elo_outputs
from cvyl_scraper.export import export_csv
from cvyl_scraper.power_v3_recency import build_power_ratings_v3_recency
from cvyl_scraper.sos import build_sos
from cvyl_scraper.trends import build_trends


DEFAULT_HISTORICAL_SNAPSHOTS_CSV = "data/processed/cvyl_historical_snapshots.csv"

HISTORICAL_SNAPSHOT_COLUMNS = [
    "snapshot_date",
    "snapshot_label",
    "team",
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

STORYLINE_COLUMNS = [
    "snapshot_date",
    "snapshot_label",
    "storyline_type",
    "team",
    "value",
    "description",
]


def build_historical_snapshots(team_games: pd.DataFrame) -> pd.DataFrame:
    completed = _completed_team_games(team_games)
    if completed.empty:
        return pd.DataFrame(columns=HISTORICAL_SNAPSHOT_COLUMNS)

    teams = sorted(completed["team"].dropna().astype(str).unique())
    snapshot_dates = weekly_snapshot_dates(completed)
    snapshots = []
    for index, snapshot_date in enumerate(snapshot_dates, start=1):
        snapshot_rows = completed[completed["game_date"] <= snapshot_date].copy()
        snapshots.append(
            _snapshot_for_date(
                snapshot_rows,
                teams,
                snapshot_date=snapshot_date,
                snapshot_label=f"Week {index}",
            )
        )
    if not snapshots:
        return pd.DataFrame(columns=HISTORICAL_SNAPSHOT_COLUMNS)
    return pd.concat(snapshots, ignore_index=True, sort=False)[HISTORICAL_SNAPSHOT_COLUMNS]


def weekly_snapshot_dates(completed_team_games: pd.DataFrame) -> list[pd.Timestamp]:
    completed = _completed_team_games(completed_team_games)
    if completed.empty:
        return []
    first_date = completed["game_date"].min().normalize()
    last_date = completed["game_date"].max().normalize()
    dates: list[pd.Timestamp] = []
    current = first_date + pd.Timedelta(days=6)
    while current < last_date:
        dates.append(current)
        current += pd.Timedelta(days=7)
    if not dates or dates[-1] != last_date:
        dates.append(last_date)
    return dates


def build_historical_storylines(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=STORYLINE_COLUMNS)
    rows: list[dict[str, object]] = []
    ordered_dates = sorted(pd.to_datetime(snapshots["snapshot_date"]).dropna().unique())
    for snapshot_date in ordered_dates:
        current = snapshots[pd.to_datetime(snapshots["snapshot_date"]) == snapshot_date].copy()
        current_label = str(current["snapshot_label"].iloc[0])
        previous_dates = [date for date in ordered_dates if date < snapshot_date]
        previous = (
            snapshots[pd.to_datetime(snapshots["snapshot_date"]) == previous_dates[-1]].copy()
            if previous_dates
            else pd.DataFrame()
        )
        history = snapshots[pd.to_datetime(snapshots["snapshot_date"]) <= snapshot_date].copy()
        rows.extend(_storyline_rows_for_snapshot(current, previous, history, current_label))
    return pd.DataFrame(rows, columns=STORYLINE_COLUMNS)


def export_historical_snapshots(
    team_games: pd.DataFrame,
    output_path: str | Path = DEFAULT_HISTORICAL_SNAPSHOTS_CSV,
) -> Path:
    return export_csv(build_historical_snapshots(team_games), output_path)


def _snapshot_for_date(
    snapshot_rows: pd.DataFrame,
    teams: list[str],
    *,
    snapshot_date: pd.Timestamp,
    snapshot_label: str,
) -> pd.DataFrame:
    power = build_power_ratings_v3_recency(snapshot_rows)
    elo_ratings, _elo_history = build_elo_outputs(snapshot_rows)
    sos = build_sos(snapshot_rows, elo_ratings)
    trends = build_trends(snapshot_rows)
    records = _records(snapshot_rows)

    base = pd.DataFrame({"team": teams})
    base = base.merge(
        power[
            [
                "team",
                "power_rank_v3_recency",
                "power_rating_v3_recency",
                "adjusted_offense_rating",
                "adjusted_defense_rating",
                "games_played",
            ]
        ].rename(
            columns={
                "power_rank_v3_recency": "power_rank",
                "power_rating_v3_recency": "power_rating",
                "adjusted_offense_rating": "offense_strength",
                "adjusted_defense_rating": "defense_strength",
                "games_played": "power_games_played",
            }
        ),
        on="team",
        how="left",
    )
    base = base.merge(elo_ratings[["team", "elo"]], on="team", how="left")
    base = base.merge(sos[["team", "sos_rank"]], on="team", how="left")
    base = base.merge(
        trends[
            [
                "team",
                "momentum_score",
                "last_3_win_pct",
                "last_5_win_pct",
            ]
        ].rename(columns={"last_3_win_pct": "last3_win_pct", "last_5_win_pct": "last5_win_pct"}),
        on="team",
        how="left",
    )
    base = base.merge(records, on="team", how="left")
    base["snapshot_date"] = snapshot_date.date().isoformat()
    base["snapshot_label"] = snapshot_label
    base["wins"] = base["wins"].fillna(0).astype(int)
    base["losses"] = base["losses"].fillna(0).astype(int)
    base["games_played"] = base["games_played"].fillna(0).astype(int)
    base = base.sort_values(["snapshot_date", "team"], ignore_index=True)
    return base[HISTORICAL_SNAPSHOT_COLUMNS]


def _completed_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame()
    completed = team_games[team_games["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()
    completed["game_date"] = pd.to_datetime(completed["game_date"], errors="coerce")
    completed["points_for"] = pd.to_numeric(completed["points_for"], errors="coerce")
    completed["points_against"] = pd.to_numeric(completed["points_against"], errors="coerce")
    completed["win"] = completed["win"].astype(bool)
    return completed.dropna(subset=["game_id", "team", "opponent", "game_date", "points_for", "points_against"])


def _records(completed: pd.DataFrame) -> pd.DataFrame:
    if completed.empty:
        return pd.DataFrame(columns=["team", "wins", "losses", "games_played"])
    records = (
        completed.groupby("team", as_index=False)
        .agg(
            wins=("win", lambda values: int(values.astype(bool).sum())),
            games_played=("win", "size"),
        )
    )
    records["losses"] = records["games_played"] - records["wins"]
    return records[["team", "wins", "losses", "games_played"]]


def _storyline_rows_for_snapshot(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    history: pd.DataFrame,
    snapshot_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    snapshot_date = str(current["snapshot_date"].iloc[0])

    if not previous.empty:
        movement = current[["team", "power_rank"]].merge(
            previous[["team", "power_rank"]].rename(columns={"power_rank": "previous_power_rank"}),
            on="team",
            how="inner",
        ).dropna(subset=["power_rank", "previous_power_rank"])
        movement["rank_move"] = movement["previous_power_rank"] - movement["power_rank"]
        rows.extend(_best_and_worst_movement_rows(movement, snapshot_date, snapshot_label))

        offense_movement = current[["team", "offense_strength"]].merge(
            previous[["team", "offense_strength"]].rename(
                columns={"offense_strength": "previous_offense_strength"}
            ),
            on="team",
            how="inner",
        ).dropna(subset=["offense_strength", "previous_offense_strength"])
        offense_movement["offense_change"] = (
            offense_movement["offense_strength"] - offense_movement["previous_offense_strength"]
        )
        offense_candidate = _top_candidate(offense_movement, "offense_change", ascending=False)
        if offense_candidate is not None:
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "snapshot_label": snapshot_label,
                    "storyline_type": "fastest_improving_offense",
                    "team": offense_candidate["team"],
                    "value": offense_candidate["offense_change"],
                    "description": "Fastest improving offense since the prior snapshot.",
                }
            )

    top_streak = _top_candidate(
        longest_top_five_streaks(history, current_snapshot_date=snapshot_date),
        "top_5_streak",
        ascending=False,
    )
    if top_streak is not None and float(top_streak["top_5_streak"]) > 0:
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "snapshot_label": snapshot_label,
                "storyline_type": "longest_top_5_streak",
                "team": top_streak["team"],
                "value": top_streak["top_5_streak"],
                "description": "Longest current top-five streak.",
            }
        )

    for storyline_type, sort_column, ascending, description in [
        ("surging_contender", "momentum_score", False, "Surging contender by momentum score."),
        ("cooling_team", "momentum_score", True, "Cooling team by current momentum score."),
    ]:
        candidate = _top_candidate(current, sort_column, ascending=ascending)
        if candidate is not None:
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "snapshot_label": snapshot_label,
                    "storyline_type": storyline_type,
                    "team": candidate["team"],
                    "value": candidate[sort_column],
                    "description": description,
                }
            )
    return rows


def _best_and_worst_movement_rows(
    movement: pd.DataFrame,
    snapshot_date: str,
    snapshot_label: str,
) -> list[dict[str, object]]:
    rows = []
    riser = _top_candidate(movement, "rank_move", ascending=False)
    faller = _top_candidate(movement, "rank_move", ascending=True)
    if riser is not None:
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "snapshot_label": snapshot_label,
                "storyline_type": "biggest_riser",
                "team": riser["team"],
                "value": riser["rank_move"],
                "description": "Biggest rise since the prior snapshot.",
            }
        )
    if faller is not None:
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "snapshot_label": snapshot_label,
                "storyline_type": "biggest_faller",
                "team": faller["team"],
                "value": faller["rank_move"],
                "description": "Biggest fall since the prior snapshot.",
            }
        )
    return rows


def _top_candidate(frame: pd.DataFrame, column: str, *, ascending: bool) -> pd.Series | None:
    if frame.empty or column not in frame.columns:
        return None
    candidates = frame.dropna(subset=[column]).copy()
    if candidates.empty:
        return None
    return candidates.sort_values([column, "team"], ascending=[ascending, True]).iloc[0]


def longest_top_five_streaks(
    snapshots: pd.DataFrame,
    *,
    current_snapshot_date: str | None = None,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["team", "top_5_streak"])
    history = snapshots.copy()
    if current_snapshot_date is not None:
        history = history[pd.to_datetime(history["snapshot_date"]) <= pd.Timestamp(current_snapshot_date)]
    history = history.sort_values(["team", "snapshot_date"])
    rows = []
    for team, group in history.groupby("team", sort=True):
        streak = 0
        for _, row in group.sort_values("snapshot_date", ascending=False).iterrows():
            rank = row.get("power_rank")
            if pd.notna(rank) and int(rank) <= 5:
                streak += 1
            else:
                break
        rows.append({"team": team, "top_5_streak": streak})
    return pd.DataFrame(rows).sort_values(["top_5_streak", "team"], ascending=[False, True], ignore_index=True)
