from datetime import date

from transforms.dimensions import (
    LEAGUE_COUNTRY,
    build_dim_team,
    build_dim_date,
    build_dim_league,
    build_dim_season,
)

def test_build_dim_team(spark):
  df = spark.createDataFrame(
      [("Arsenal", "Chelsea"), ("Chelsea", "Arsenal")],
      ["home_team", "away_team"],
  )

  result = build_dim_team(df)
  rows = {row["team_key"]: row for row in result.collect()}

  assert rows['ARSENAL']['team_name'] == 'Arsenal'
  assert rows['CHELSEA']['team_name'] == 'Chelsea'

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
      [("Premier League", "England"), ("Allsvenskan", "Sweden")],
      ["league_name", "country"],
  )

  result = build_dim_league(df)
  rows = {row['league_key']: row for row in result.collect()}

  assert rows['PREMIER LEAGUE']['country'] == 'England'
  assert rows['ALLSVENSKAN']['country'] == 'Sweden'

  assert rows['PREMIER LEAGUE']['league_name'] == 'Premier League'
  assert rows['ALLSVENSKAN']['league_name'] == 'Allsvenskan' 

def test_build_dim_season(spark):
  df = spark.createDataFrame(
      [
          ("2024-08-15T15:00:00", "England"),  # -> season "2024/2025"
          ("2024-03-10T15:00:00", "Sweden"),   # -> season "2024"
      ],
      ["match_start", "country"],
  )

  result = build_dim_season(df)
  rows = {row["season_key"]: row for row in result.collect()}

  assert rows["2024/2025"]["start_date"] == date(2024, 7, 1)
  assert rows["2024/2025"]["end_date"] == date(2025, 6, 30)

  assert rows["2024"]["start_date"] == date(2024, 1, 1)
  assert rows["2024"]["end_date"] == date(2024, 12, 31)

  assert rows["2024/2025"]["is_current"] == False
  assert rows["2024"]["is_current"] == False





