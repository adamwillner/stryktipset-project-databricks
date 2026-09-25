from pyspark.sql import functions as F

from transforms.elo import expected_home_score
from transforms.scoring import build_coupon_predictions, elo_probability

MATCHES_SCHEMA = (
    'draw_number int, event_number int, match_id string, match_start string, '
    'league string, home_team string, away_team string, status string, '
    'streck_1 double, streck_x double, streck_2 double, '
    'calibrated_1 double, calibrated_x double, calibrated_2 double'
)

def test_elo_probability(spark):
  df = spark.createDataFrame(
      [(1500.0, 1500.0), (1900.0, 1500.0), (1100.0, 1500.0)],
      "home double, away double",
  )

  rows = df.select(
      "home", "away", elo_probability(F.col("home"), F.col("away")).alias("p")
  ).collect()

  # this column expression duplicates expected_home_score's formula, so pin
  # the two together -- tune one without the other and this fails
  for row in rows:
    assert abs(row["p"] - expected_home_score(row["home"], row["away"])) < 1e-9

def test_build_coupon_predictions(spark):
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

  result = build_coupon_predictions(matches, elo, 4971)
  rows = result.collect()

  assert [row["match_id"] for row in rows] == ["m1", "m2"]  # ordered by event_number

  keyed = {row["match_id"]: row for row in rows}
  assert abs(keyed["m2"]["elo_prob_1"] - expected_home_score(1700.0, 1500.0)) < 1e-9

  # a match with no Elo still appears rather than being dropped -- an
  # out-of-scope league or a team with no history must stay visible
  assert keyed["m1"]["elo_home"] is None
  assert keyed["m1"]["elo_prob_1"] is None
  assert keyed["m1"]["streck_1"] == 0.50
