from __future__ import annotations

import pandas as pd

from cvyl_scraper.team_identity import (
    build_team_identity_audit,
    export_team_identity_audit,
    possible_duplicate_group,
)


def test_build_team_identity_audit_counts_games_sources_and_opponents() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "home_team": "West Hartford 12U Green",
                "away_team": "Farmington 12U Red",
                "status": "completed",
                "source_name": "west_hartford_12u_green",
            },
            {
                "game_id": "game-2",
                "home_team": "Avon 12U B",
                "away_team": "West Hartford 12U Green",
                "status": "scheduled",
                "source_name": "avon_12u_b",
            },
        ]
    )

    audit = build_team_identity_audit(games).set_index("team_name")
    west_hartford = audit.loc["West Hartford 12U Green"]

    assert west_hartford["games_played"] == 2
    assert west_hartford["completed_games"] == 1
    assert west_hartford["scheduled_games"] == 1
    assert west_hartford["source_count"] == 2
    assert west_hartford["source_names"] == "avon_12u_b; west_hartford_12u_green"
    assert west_hartford["opponent_count"] == 2
    assert west_hartford["opponents"] == "Avon 12U B; Farmington 12U Red"


def test_build_team_identity_audit_sets_flags_and_duplicate_group() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "home_team": "Granby",
                "away_team": "Granby 12U",
                "status": "completed",
                "source_name": "granby_12u",
            }
        ]
    )

    audit = build_team_identity_audit(games).set_index("team_name")

    assert audit.loc["Granby", "appears_as_source_team"].item() is False
    assert audit.loc["Granby", "has_12u_suffix"].item() is False
    assert audit.loc["Granby", "possible_duplicate_group"] == "granby"
    assert audit.loc["Granby 12U", "appears_as_source_team"].item() is True
    assert audit.loc["Granby 12U", "has_12u_suffix"].item() is True
    assert audit.loc["Granby 12U", "possible_duplicate_group"] == "granby"


def test_possible_duplicate_group_removes_common_u12_variant_terms() -> None:
    assert possible_duplicate_group("West Hartford 12U Green") == "west_hartford"
    assert possible_duplicate_group("West Hartford 12U Gold") == "west_hartford"
    assert possible_duplicate_group("Somers Juniors") == "somers"
    assert possible_duplicate_group("Avon 12U B") == "avon"


def test_export_team_identity_audit_writes_report(tmp_path) -> None:
    input_path = tmp_path / "cvyl_games.csv"
    output_path = tmp_path / "team_identity_audit.csv"
    pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "home_team": "West Hartford 12U Green",
                "away_team": "Farmington 12U Red",
                "status": "completed",
                "source_name": "west_hartford_12u_green",
            }
        ]
    ).to_csv(input_path, index=False)

    generated_path = export_team_identity_audit(input_path, output_path)
    audit = pd.read_csv(output_path)

    assert generated_path == output_path
    assert audit["team_name"].tolist() == ["Farmington 12U Red", "West Hartford 12U Green"]
