from __future__ import annotations

from cvyl_scraper.discovery import discover_team_sources, is_boys_junior_division


def test_is_boys_junior_division_handles_naming_variations() -> None:
    assert is_boys_junior_division("Boys Junior A")
    assert is_boys_junior_division("Boys Junior B")
    assert is_boys_junior_division("12U Boys")
    assert is_boys_junior_division("U12 Boys")
    assert is_boys_junior_division("12U")
    assert is_boys_junior_division("U12")
    assert is_boys_junior_division("Boys 12U A")


def test_is_boys_junior_division_excludes_girls_and_older_groups() -> None:
    assert not is_boys_junior_division("Girls Junior A")
    assert not is_boys_junior_division("Girls 12U")
    assert not is_boys_junior_division("Boys Senior A")
    assert not is_boys_junior_division("Boys 14U")
    assert not is_boys_junior_division("U14 Boys")


def test_discover_team_sources_filters_target_divisions_and_extracts_urls() -> None:
    html = """
    <html>
      <body>
        <section>
          <h2>Boys Junior A</h2>
          <a href="/team/101/west-hartford-green">West Hartford Green</a>
          <a href="/team/102/farmington/games">Farmington Red</a>
        </section>
        <section>
          <h2>Girls Junior A</h2>
          <a href="/team/201/girls-team">Girls Team</a>
        </section>
        <section>
          <h2>Boys 14U</h2>
          <a href="/team/301/older-team">Older Team</a>
        </section>
      </body>
    </html>
    """

    sources = discover_team_sources(html, "https://www.cvyl.org/club/teams")

    assert sources["team_name"].tolist() == ["Farmington Red", "West Hartford Green"]
    assert sources["division"].tolist() == ["Boys Junior A", "Boys Junior A"]
    assert sources["team_games_url"].tolist() == [
        "https://www.cvyl.org/team/102/farmington/games",
        "https://www.cvyl.org/team/101/west-hartford-green/games",
    ]


def test_discover_team_sources_uses_table_division_context() -> None:
    html = """
    <html>
      <body>
        <table>
          <tr><th>Division</th><th>Team</th></tr>
          <tr>
            <td>U12 Boys</td>
            <td><a href="https://www.cvyl.org/team/401/avon">Avon</a></td>
          </tr>
          <tr>
            <td>Boys Junior B</td>
            <td><a href="/team/402/canton">Canton</a></td>
          </tr>
          <tr>
            <td>U14 Boys</td>
            <td><a href="/team/403/simsbury">Simsbury</a></td>
          </tr>
        </table>
      </body>
    </html>
    """

    sources = discover_team_sources(html, "https://www.cvyl.org/teams")

    assert sources["team_name"].tolist() == ["Canton", "Avon"]
    assert sources["division"].tolist() == ["Boys Junior B", "U12 Boys"]
    assert sources["team_games_url"].tolist() == [
        "https://www.cvyl.org/team/402/canton/games",
        "https://www.cvyl.org/team/401/avon/games",
    ]
