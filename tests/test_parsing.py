from __future__ import annotations

import pandas as pd

from cvyl_scraper.models import Source
from cvyl_scraper.parsing import parse_schedule_page


def test_parse_crossbar_team_games_with_scores_and_scheduled_times() -> None:
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Connecticut Valley Youth Lacrosse | West Hartford 12U Green | Boys Division | 2026 BOYS</title>
      </head>
      <body>
        <h3><i class="fas fa-calendar-alt"></i> Game Schedule</h3>
        <div class="col-xs-12">
          <div class="box" style="margin-top:10px;">
            <div class="row">
              <div class="col-xs-2" style="text-align:center;">
                <h1 style="margin:0px; font-size:25px; text-transform: uppercase; line-height:90%;">Apr</h1>
                <h1 style="margin:0px; font-size:45px; line-height:90%;">22</h1>
              </div>
              <div class="col-xs-5">
                <h2 class="nomargin">
                  <span class="small">vs.  </span>
                  Farmington 12U Red
                </h2>
                <p><a target="_blank" href="https://www.google.com/maps/search/?api=1">King Phillip</a></p>
              </div>
              <div class="col-xs-3 text-center">
                <h3 class="mb10" style="line-height:100%;">
                  7 - 0

                  (W)
                </h3>
              </div>
            </div>
          </div>
          <div class="box" style="margin-top:10px;">
            <div class="row">
              <div class="col-xs-2" style="text-align:center;">
                <h1 style="margin:0px; font-size:25px; text-transform: uppercase; line-height:90%;">Apr</h1>
                <h1 style="margin:0px; font-size:45px; line-height:90%;">26</h1>
              </div>
              <div class="col-xs-5">
                <h2 class="nomargin">
                  <span class="small">@  </span>
                  Avon 12U B
                </h2>
                <p><a target="_blank" href="https://www.google.com/maps/search/?api=1">Avon HS Turf</a></p>
              </div>
              <div class="col-xs-3 text-center">
                <h3 class="mb10" style="line-height:100%;">
                  5 - 4

                  (W)
                </h3>
              </div>
            </div>
          </div>
          <div class="box" style="margin-top:10px;">
            <div class="row">
              <div class="col-xs-2" style="text-align:center;">
                <h1 style="margin:0px; font-size:25px; text-transform: uppercase; line-height:90%;">May</h1>
                <h1 style="margin:0px; font-size:45px; line-height:90%;">11</h1>
              </div>
              <div class="col-xs-5">
                <h2 class="nomargin">
                  <span class="small">@  </span>
                  Tolland 12U
                </h2>
                <p><a target="_blank" href="https://www.google.com/maps/search/?api=1">Tolland MS</a></p>
              </div>
              <div class="col-xs-3 text-center">
                <h3 class="mb10" style="line-height:100%;">
                  6:30 PM
                </h3>
              </div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
    source = Source(
        name="west_hartford_green_u12",
        url="https://www.cvyl.org/team/225939/games",
        season=2026,
        division="U12 Boys",
    )

    games = parse_schedule_page(html, source)

    assert len(games) == 3
    assert games.iloc[0]["game_date"] == "2026-04-22"
    assert pd.isna(games.iloc[0]["game_time"])
    assert games.iloc[0]["home_team"] == "West Hartford 12U Green"
    assert games.iloc[0]["away_team"] == "Farmington 12U Red"
    assert games.iloc[0]["home_score"] == 7
    assert games.iloc[0]["away_score"] == 0
    assert games.iloc[0]["source_name"] == "west_hartford_green_u12"
    assert games.iloc[0]["source_url"] == "https://www.cvyl.org/team/225939/games"
    assert games.iloc[0]["season"] == 2026
    assert games.iloc[0]["division"] == "U12 Boys"
    assert games.iloc[1]["home_team"] == "Avon 12U B"
    assert games.iloc[1]["away_team"] == "West Hartford 12U Green"
    assert games.iloc[1]["home_score"] == 4
    assert games.iloc[1]["away_score"] == 5
    assert games.iloc[2]["game_date"] == "2026-05-11"
    assert games.iloc[2]["game_time"] == "6:30 PM"
    assert pd.isna(games.iloc[2]["home_score"])
    assert pd.isna(games.iloc[2]["away_score"])
