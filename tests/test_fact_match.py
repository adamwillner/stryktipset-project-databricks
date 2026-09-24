from transforms.fact_match import build_fact_match

MATCHES_SCHEMA = (
    'match_id string, league string, home_team string, away_team string, '
    'match_start string, home_goals int, away_goals int, result string, '
    'streck_1 double, streck_x double, streck_2 double, '
    'odds_1 double, odds_x double, odds_2 double, '
    'start_odds_1 double, start_odds_x double, start_odds_2 double, '
    'favourite_odds_1 double, favourite_odds_x double, favourite_odds_2 double, '
    'status string, betradar_id string'
)

def test_build_fact_match(spark):
  matches = spark.createDataFrame(
      [
          # England, Aug -> season 2024/2025
          ("m1", "Premier League", "Arsenal", "Chelsea", "2024-08-15T15:00:00",
           2, 1, "1", 45.0, 30.0, 25.0, 1.8, 3.5, 4.2, 1.9, 3.4, 4.0,
           0.5852, 0.2459, 0.1689, "Ended", "72221288"),
          # England, Mar -> season 2023/2024
          ("m2", "Premier League", "Arsenal", "Chelsea", "2024-03-10T15:00:00",
           2, 1, "1", 45.0, 30.0, 25.0, 1.8, 3.5, 4.2, 1.9, 3.4, 4.0,
           0.5852, 0.2459, 0.1689, "Ended", "72221288"),
          # Sweden, Mar -> season 2024
          ("m3", "Allsvenskan", "AIK", "Djurgarden", "2024-03-10T15:00:00",
           2, 1, "1", 45.0, 30.0, 25.0, 1.8, 3.5, 4.2, 1.9, 3.4, 4.0,
           0.5852, 0.2459, 0.1689, "Ended", "72221288"),
          # out of scope -> dropped by the inner join
          ("m4", "Champions League", "Arsenal", "Chelsea", "2024-08-15T15:00:00",
           2, 1, "1", 45.0, 30.0, 25.0, 1.8, 3.5, 4.2, 1.9, 3.4, 4.0,
           0.5852, 0.2459, 0.1689, "Ended", "72221288"),
      ],
      MATCHES_SCHEMA,
  )
  dim_league = spark.createDataFrame(
      [
          ("PREMIER LEAGUE", "Premier League", "England"),
          ("ALLSVENSKAN", "Allsvenskan", "Sweden"),
      ],
      'league_key string, league_name string, country string',
  )

  result = build_fact_match(matches, dim_league)
  rows = {row["match_key"]: row for row in result.collect()}

  assert "m4" not in rows
  assert result.count() == 3

  assert rows["m1"]["home_team_key"] == "ARSENAL"
  assert rows["m1"]["away_team_key"] == "CHELSEA"
  assert rows["m1"]["date_key"] == 20240815
  assert rows["m1"]["league_key"] == "PREMIER LEAGUE"

  assert rows["m1"]["season_key"] == "2024/2025"
  assert rows["m2"]["season_key"] == "2023/2024"
  assert rows["m3"]["season_key"] == "2024"

  assert rows["m1"]["calibrated_1"] is None
  assert rows["m1"]["calibrated_x"] is None
  assert rows["m1"]["calibrated_2"] is None
  assert result.schema["calibrated_1"].dataType.simpleString() == "double"

  assert rows["m1"]["elo_home"] is None
  assert rows["m1"]["elo_away"] is None
  assert result.schema["elo_home"].dataType.simpleString() == "double"

  assert rows["m1"]["favourite_odds_1"] == 0.5852
  assert rows["m1"]["status"] == "Ended"
  assert rows["m1"]["betradar_id"] == "72221288"
