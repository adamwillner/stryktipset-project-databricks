from pyspark.sql import functions as F

from transforms.elo import expected_home_score
from transforms.dimensions import build_dim_league
from transforms.scoring import build_mart_coupon, expected_home_score_column

MATCHES_SCHEMA = (
    'draw_number int, event_number int, match_id string, match_start string, '
    'league string, home_team string, away_team string, status string, '
    'streck_1 double, streck_x double, streck_2 double, '
    'calibrated_1 double, calibrated_x double, calibrated_2 double'
)

def test_expected_home_score_column(spark):
  df = spark.createDataFrame(
      [(1500.0, 1500.0), (1900.0, 1500.0), (1100.0, 1500.0)],
      "home double, away double",
  )

  rows = df.select(
      "home", "away", expected_home_score_column(F.col("home"), F.col("away")).alias("p")
  ).collect()

  # this column expression duplicates expected_home_score's formula, so pin
  # the two together -- tune one without the other and this fails
  for row in rows:
    assert abs(row["p"] - expected_home_score(row["home"], row["away"])) < 1e-9

def test_build_mart_coupon(spark):
  matches = spark.createDataFrame(
      [
          (4971, 2, "m2", "2026-09-19T18:30:00", "Premier League", "Arsenal",
           "Chelsea", "NotStarted", 0.45, 0.30, 0.25, 0.44, 0.31, 0.25),
          (4971, 1, "m1", "2026-09-19T16:00:00", "Eliteserien", "Bodo",
           "Molde", "NotStarted", 0.50, 0.28, 0.22, 0.49, 0.29, 0.22),
          # previous draw -- must not appear
          (4970, 1, "m0", "2026-09-12T16:00:00", "Premier League", "Spurs",
           "Everton", "Ended", 0.60, 0.25, 0.15, 0.58, 0.26, 0.16),
      ],
      MATCHES_SCHEMA,
  )
  elo = spark.createDataFrame(
      [("m2", 1700.0, 1500.0), ("m0", 1600.0, 1500.0)],
      "match_id string, elo_home double, elo_away double",
  )

  dim_league = build_dim_league(
      spark.createDataFrame(
          [("Premier League", "England")], 'league_name string, country string'
      )
  )

  result = build_mart_coupon(matches, elo, dim_league, 4971)
  rows = result.collect()

  assert [row["match_key"] for row in rows] == ["m1", "m2"]  # ordered by event_number

  keyed = {row["match_key"]: row for row in rows}

  # readable by design: this is the table a person opens on a Thursday
  assert keyed["m2"]["home_team"] == "Arsenal"
  assert keyed["m2"]["league"] == "Premier League"

  # and still joinable: an out-of-scope league resolves to the Unknown
  # member rather than to a null foreign key
  unknown_sk = {r["league_key"]: r["league_sk"] for r in dim_league.collect()}["UNKNOWN"]
  assert keyed["m1"]["league"] == "Eliteserien"
  assert keyed["m1"]["league_sk"] == unknown_sk
  assert keyed["m2"]["league_sk"] != unknown_sk
  assert abs(keyed["m2"]["elo_expected_score"] - expected_home_score(1700.0, 1500.0)) < 1e-9

  # a match with no Elo still appears rather than being dropped -- an
  # out-of-scope league or a team with no history must stay visible
  assert keyed["m1"]["elo_home"] is None
  assert keyed["m1"]["elo_expected_score"] is None
  assert keyed["m1"]["streck_1"] == 0.50
