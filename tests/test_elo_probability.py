from transforms.elo_probability import elo_predictor, to_elo_long_format

MATCHES_SCHEMA = 'match_id string, result string, elo_expected_score double'

def test_elo_predictor(spark):
  df = spark.createDataFrame(
      [("even", 0.5), ("strong_home", 0.9), ("strong_away", 0.1)],
      "label string, elo_expected_score double",
  )
  rows = {
      row["label"]: row
      for row in df.select(
          "label",
          elo_predictor("1").alias("p1"),
          elo_predictor("X").alias("px"),
          elo_predictor("2").alias("p2"),
      ).collect()
  }

  # home and away are straight slopes in opposite directions
  assert rows["strong_home"]["p1"] > rows["even"]["p1"] > rows["strong_away"]["p1"]
  assert rows["strong_away"]["p2"] > rows["even"]["p2"] > rows["strong_home"]["p2"]

  # the draw predictor is "evenness": highest for a level match, and equal
  # for mismatches in either direction -- which is what lets isotonic fit
  # a hump-shaped relationship with a monotone curve
  assert rows["even"]["px"] == 0.5
  assert rows["strong_home"]["px"] == rows["strong_away"]["px"]
  assert rows["even"]["px"] > rows["strong_home"]["px"]

def test_to_elo_long_format(spark):
  df = spark.createDataFrame(
      [
          ("m1", "1", 0.7),
          ("m2", None, 0.6),  # unplayed -- nothing to learn from
      ],
      MATCHES_SCHEMA,
  )

  result = to_elo_long_format(df)
  rows = {row["outcome"]: row for row in result.collect()}

  assert result.count() == 3  # one played match, three outcomes
  assert rows["1"]["actual_occurred"] is True
  assert rows["X"]["actual_occurred"] is False
  assert rows["2"]["actual_occurred"] is False
  assert abs(rows["1"]["predicted_prob"] - 0.7) < 1e-9
