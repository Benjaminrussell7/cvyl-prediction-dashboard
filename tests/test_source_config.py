from __future__ import annotations

import pandas as pd
import yaml

from cvyl_scraper.source_config import (
    discovered_sources_to_config,
    generate_discovered_sources_config,
    safe_source_name,
)


def test_discovered_sources_to_config_builds_valid_yaml_structure() -> None:
    discovered = pd.DataFrame(
        [
            {
                "team_name": "West Hartford 12U Green",
                "division": "Boys Junior A",
                "team_games_url": "https://www.cvyl.org/team/101/west-hartford/games",
            }
        ]
    )

    payload = discovered_sources_to_config(discovered)

    assert payload == {
        "sources": [
            {
                "name": "west_hartford_12u_green",
                "url": "https://www.cvyl.org/team/101/west-hartford/games",
                "season": 2026,
                "division": "U12 Boys",
            }
        ]
    }


def test_safe_source_name_normalizes_team_names_to_ids() -> None:
    assert safe_source_name("West Hartford 12U Green") == "west_hartford_12u_green"
    assert safe_source_name("  Farmington 12U-Red! ") == "farmington_12u_red"
    assert safe_source_name("Canton/U12 Blue") == "canton_u12_blue"


def test_generate_discovered_sources_config_writes_valid_output(tmp_path) -> None:
    input_path = tmp_path / "discovered_sources.csv"
    output_path = tmp_path / "discovered_sources.yml"
    pd.DataFrame(
        [
            {
                "team_name": "West Hartford 12U Green",
                "division": "Boys Junior A",
                "team_games_url": "https://www.cvyl.org/team/101/west-hartford/games",
            },
            {
                "team_name": "Farmington 12U Red",
                "division": "Boys Junior B",
                "team_games_url": "https://www.cvyl.org/team/102/farmington/games",
            },
        ]
    ).to_csv(input_path, index=False)

    generated_path = generate_discovered_sources_config(input_path, output_path)

    assert generated_path == output_path
    with output_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    assert payload["sources"] == [
        {
            "name": "west_hartford_12u_green",
            "url": "https://www.cvyl.org/team/101/west-hartford/games",
            "season": 2026,
            "division": "U12 Boys",
        },
        {
            "name": "farmington_12u_red",
            "url": "https://www.cvyl.org/team/102/farmington/games",
            "season": 2026,
            "division": "U12 Boys",
        },
    ]
