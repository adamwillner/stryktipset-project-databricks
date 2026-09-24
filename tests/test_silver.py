"""Tests for transforms/silver.py (parse_swedish_decimal, extract_match_row).

TODO: write tests here. Some starting points:
- parse_swedish_decimal: "1,50" -> 1.5, does it handle already-dotted
  strings / nulls?
- extract_match_row: needs a small fake `event` struct (built with
  F.struct/F.lit or spark.createDataFrame + to_json/from_json) to run
  against -- check the derived `result` column ('1'/'X'/'2') lands right
  for a home win / draw / away win.
"""

from pyspark.sql import functions as F, Row

from transforms.silver import parse_swedish_decimal, extract_match_row


def test_parse_swedish_decimal(spark):
    # 1. fake input -- a tiny 2-row DataFrame with one string column
    df = spark.createDataFrame([("1,50",), ("0,45",)], ["raw"])

    # 2. run the real function on it
    result = df.select(parse_swedish_decimal(F.col("raw")).alias("parsed"))

    # 3. pull the actual values back into plain Python
    values = [row["parsed"] for row in result.collect()]

    # 4. check they're what we expect
    assert values == [1.50, 0.45]


def test_extract_match_row(spark):
    def event_row(number, match_id, start, home_goals, away_goals):
        """Nested fake data must be built from Row(...), not dicts -- a dict
        infers as MapType, whose values must all share one type."""
        return Row(
            eventNumber=number,
            match=Row(
                matchId=match_id,
                matchStart=start,
                sportEventStatus="Ended",
                participants=[
                    Row(name="Home Team", result=home_goals, id=69, countryName="England"),
                    Row(name="Away Team", result=away_goals, id=70, countryName="England"),
                ],
                league=Row(name="Premier League", id=1, country=Row(name="England")),
            ),
            svenskaFolket=Row(one="45,00", x="30,00", two="25,00"),
            odds=Row(one="1,80", x="3,40", two="4,20"),
            startOdds=Row(one="1,90", x="3,30", two="4,00"),
            favouriteOdds=Row(one="58.52", x="24.59", two="16.89"),
            # BetRadar deliberately NOT first -- it must be picked by name
            providerIds=[
                Row(provider="Kambi", type="Normal", id="1028072794"),
                Row(provider="BetRadar", type="Normal", id="72221288"),
            ],
        )

    df = spark.createDataFrame([
        Row(drawNumber=1, event=event_row(1, 123, "2024-01-01T15:00:00", "2", "1")),
        # not played yet -- Svenska Spel has no score for it
        Row(drawNumber=1, event=event_row(2, 456, "2024-01-02T15:00:00", None, None)),
    ])

    result = df.select(extract_match_row(F.col("drawNumber"), F.col("event")))

    rows = {row["match_id"]: row for row in result.collect()}
    row = rows[123]

    assert row["home_team"] == "Home Team"
    assert row["away_team"] == "Away Team"
    assert row["home_goals"] == 2
    assert row["away_goals"] == 1
    assert row["result"] == "1"

    # an unplayed fixture must come back as null, not as an away win --
    # there is no .otherwise() branch for exactly this reason
    assert rows[456]["home_goals"] is None
    assert rows[456]["away_goals"] is None
    assert rows[456]["result"] is None

    # fields carried since 2026-09-24
    assert row["home_team_id"] == 69
    assert row["away_team_id"] == 70
    assert row["home_team_country"] == "England"
    assert row["league_id"] == 1
    assert row["league_country"] == "England"
    assert row["status"] == "Ended"

    # favouriteOdds is a percentage at source; stored as a 0-1 fraction so
    # it sits on the same scale as streck_*, and the three sum to 1
    assert abs(row["favourite_odds_1"] - 0.5852) < 1e-9
    assert abs(sum(row[f"favourite_odds_{o}"] for o in ("1", "x", "2")) - 1.0) < 1e-6

    # chosen by provider name, not by position in the array
    assert row["betradar_id"] == "72221288"
