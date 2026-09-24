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
    event = Row(
        eventNumber=1,
        match=Row(
            matchId=123,
            matchStart="2024-01-01T15:00:00",
            participants=[
                Row(name="Home Team", result="2"),
                Row(name="Away Team", result="1"),
            ],
            league=Row(name="Premier League"),
        ),
        svenskaFolket=Row(one="45,00", x="30,00", two="25,00"),
        odds=Row(one="1,80", x="3,40", two="4,20"),
        startOdds=Row(one="1,90", x="3,30", two="4,00"),
    )

    # same shape, but not played yet -- Svenska Spel has no score for it
    unplayed = Row(
        eventNumber=2,
        match=Row(
            matchId=456,
            matchStart="2024-01-02T15:00:00",
            participants=[
                Row(name="Home Team", result=None),
                Row(name="Away Team", result=None),
            ],
            league=Row(name="Premier League"),
        ),
        svenskaFolket=Row(one="45,00", x="30,00", two="25,00"),
        odds=Row(one="1,80", x="3,40", two="4,20"),
        startOdds=Row(one="1,90", x="3,30", two="4,00"),
    )

    df = spark.createDataFrame(
        [Row(drawNumber=1, event=event), Row(drawNumber=1, event=unplayed)]
    )

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