from datetime import date

import pytest
from pyspark.sql import functions as F

from transforms.dimensions import (
    LEAGUE_COUNTRY,
    build_dim_team,
    build_dim_date,
    build_dim_league,
    build_dim_season,
    build_dim_team_scd2,
    league_surrogate_key,
    team_surrogate_key,
)


def _key(fn, value):
  """Evaluate a surrogate-key expression on a one-row DataFrame, so the
  test checks the real function rather than restating the hash."""
  from pyspark.sql import SparkSession
  spark = SparkSession.builder.getOrCreate()
  return spark.range(1).select(fn(F.lit(value)).alias("k")).first()["k"]


def team_surrogate_key_value(name):
  return _key(team_surrogate_key, name)


def league_surrogate_key_value(name):
  return _key(league_surrogate_key, name)

def test_build_dim_team(spark):
  df = spark.createDataFrame(
      [
          # older draw, back when Svenska Spel used the 1000xxx id range
          ("Arsenal", "Chelsea", 1000041, 1000489, "England", "England", "2023-01-01T15:00:00"),
          # newer draw, after they renumbered -- this one must win
          ("Chelsea", "Arsenal", 70, 69, "England", "England", "2026-01-01T15:00:00"),
      ],
      "home_team string, away_team string, home_team_id int, away_team_id int, "
      "home_team_country string, away_team_country string, match_start string",
  )

  result = build_dim_team(df)
  rows = {row["team_key"]: row for row in result.collect()}

  assert rows['ARSENAL']['team_name'] == 'Arsenal'
  assert rows['CHELSEA']['team_name'] == 'Chelsea'

  # the surrogate key is a hash of the business key, so it is the same on
  # every rebuild and different per team
  assert rows['ARSENAL']['team_sk'] == team_surrogate_key_value('Arsenal')
  assert rows['ARSENAL']['team_sk'] != rows['CHELSEA']['team_sk']

  # ids are not stable across the history, so the newest row wins
  assert rows['ARSENAL']['team_id'] == 69
  assert rows['CHELSEA']['team_id'] == 70

  assert rows['ARSENAL']['country'] == 'England'

def test_build_dim_date(spark):
  df = spark.createDataFrame(
      [("2024-01-01T15:00:00",), ("2024-06-15T12:00:00",)],
      ["match_start"],
  )

  result = build_dim_date(df)
  rows = {row['date_key']: row for row in result.collect()}

  assert rows[20240101]['day_of_week'] == 1
  assert rows[20240101]['is_weekend'] == False
  
  assert rows[20240106]["day_of_week"] == 6
  assert rows[20240106]["is_weekend"] == True

  assert 20240101 in rows
  assert 20231231 not in rows

def test_build_dim_league(spark):
  df = spark.createDataFrame(
      [("Premier League", "England"), ("League Two", "England")],
      ["league_name", "country"],
  )

  result = build_dim_league(df)
  rows = {row['league_key']: row for row in result.collect()}

  assert rows['PREMIER LEAGUE']['country'] == 'England'
  assert rows['PREMIER LEAGUE']['league_name'] == 'Premier League'
  assert rows['LEAGUE TWO']['league_name'] == 'League Two'

  # an Unknown member is always present, so a coupon match from a league
  # outside gold's scope still gets a real key instead of a null one
  assert 'UNKNOWN' in rows
  assert rows['UNKNOWN']['country'] is None

  # surrogate keys are derived from the business key, so they are stable
  # across rebuilds and distinct per league
  assert rows['PREMIER LEAGUE']['league_sk'] == league_surrogate_key_value('Premier League')
  assert rows['PREMIER LEAGUE']['league_sk'] != rows['LEAGUE TWO']['league_sk']

def test_build_dim_season(spark):
  df = spark.createDataFrame(
      [
          ("2024-08-15T15:00:00",),  # Aug -> season "2024/2025"
          ("2024-03-10T15:00:00",),  # Mar -> still "2023/2024"
      ],
      ["match_start"],
  )

  result = build_dim_season(df)
  rows = {row["season_key"]: row for row in result.collect()}

  # July is the cutoff that decides which year pair a match belongs to
  assert rows["2024/2025"]["start_date"] == date(2024, 7, 1)
  assert rows["2024/2025"]["end_date"] == date(2025, 6, 30)

  assert rows["2023/2024"]["start_date"] == date(2023, 7, 1)
  assert rows["2023/2024"]["end_date"] == date(2024, 6, 30)

  assert rows["2024/2025"]["is_current"] == False
  assert rows["2023/2024"]["is_current"] == False







def test_league_country_is_english_only():
  """Gold dropped Sweden on 2026-09-29. This pins that decision: adding a
  non-English league here also needs a per-country rule in build_dim_season,
  which was deliberately collapsed to the England shape."""
  assert set(LEAGUE_COUNTRY.values()) == {"England"}
  assert len(LEAGUE_COUNTRY) == 5


@pytest.mark.skip(reason="SCD Type 2 not implemented -- see build_dim_team_scd2")
def test_build_dim_team_scd2(spark):
  """Three cases to assert once it exists:

  1. attributes unchanged -> one row, still is_current, valid_to still null
  2. team_name changed    -> two rows for that team: the old one closed
                             (is_current False, valid_to set) and a new
                             current one; a fact row dated before the change
                             must still join to the old version
  3. brand new team       -> one row, is_current True, valid_to null

  And the trap: team_sk must differ between the two versions, or they
  collide on the same key and the history is unreachable.
  """
  build_dim_team_scd2(None, None)
