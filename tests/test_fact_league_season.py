from transforms.fact_league_season import build_fact_league_season

FACT_MATCH_SCHEMA = (
    'league_key string, season_key string, home_goals int, away_goals int, '
    'result string'
)

def test_build_fact_league_season(spark):
  fact_match = spark.createDataFrame(
      [
          ("L1", "2024/2025", 2, 1, "1"),   # home win, 3 goals
          ("L1", "2024/2025", 1, 1, "X"),   # draw, 2 goals
          ("L1", "2024/2025", 0, 2, "2"),   # away win, 2 goals
          ("L1", "2023/2024", 3, 0, "1"),   # different season
          ("L2", "2024/2025", 1, 0, "1"),   # different league
          # not played yet -- must not be counted, or every rate is wrong
          ("L1", "2024/2025", None, None, None),
      ],
      FACT_MATCH_SCHEMA,
  )

  result = build_fact_league_season(fact_match)
  rows = {(row["league_key"], row["season_key"]): row for row in result.collect()}

  # one row per league per season, and the unplayed fixture is excluded
  assert len(rows) == 3

  row = rows[("L1", "2024/2025")]
  assert row["match_count"] == 3
  assert row["total_goals"] == 7
  assert row["avg_goals"] == 2.333  # rounded to 3 dp by the transform
  assert row["home_wins"] == 1
  assert row["draws"] == 1
  assert row["away_wins"] == 1
  assert row["home_win_rate"] == 0.333

  assert rows[("L1", "2023/2024")]["match_count"] == 1
  assert rows[("L2", "2024/2025")]["home_win_rate"] == 1.0
