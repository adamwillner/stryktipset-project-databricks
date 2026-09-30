from transforms.backtest import (
    blend,
    fit_blend_weight,
    paired_brier_difference,
    scope_to_leagues,
    split_by_date,
)

LEAGUE_SCHEMA = 'match_id string, league string'

def test_scope_to_leagues(spark):
  matches = spark.createDataFrame(
      [
          ("m1", "Premier League"),
          ("m2", "Allsvenskan"),
          ("m3", "National League"),
          ("m4", "FA Cup"),          # English, but not one of the five tiers
          ("m5", "premier league"),  # free-text match is case sensitive
          ("m6", None),
      ],
      LEAGUE_SCHEMA,
  )

  scoped = scope_to_leagues(matches, ("Premier League", "National League"))

  assert sorted(row["match_id"] for row in scoped.collect()) == ["m1", "m3"]

MATCHES_SCHEMA = 'match_id string, match_start string, result string'

def test_split_by_date(spark):
  matches = spark.createDataFrame(
      [
          ("m1", "2022-12-31T15:00:00", "1"),   # before  -> train
          ("m2", "2023-01-01T15:00:00", "X"),   # the cutoff day itself -> test
          ("m3", "2023-06-01T15:00:00", "2"),   # after   -> test
          ("m4", "2021-05-01T15:00:00", "1"),   # before  -> train
          ("m5", "2023-09-01T15:00:00", None),  # unplayed -> dropped entirely
      ],
      MATCHES_SCHEMA,
  )

  train, test = split_by_date(matches, "2023-01-01")

  assert sorted(row["match_id"] for row in train.collect()) == ["m1", "m4"]
  assert sorted(row["match_id"] for row in test.collect()) == ["m2", "m3"]

  # an unplayed fixture has no result to score against, so it belongs in
  # neither half rather than quietly padding the test set
  assert train.count() + test.count() == 4


SCORED_SCHEMA = (
    'result string, streck_1 double, streck_x double, streck_2 double, '
    'calibrated_1 double, calibrated_x double, calibrated_2 double'
)

def test_paired_brier_difference(spark):
  df = spark.createDataFrame(
      [
          # home win. raw brier = ((0.5-1)^2 + 0.3^2 + 0.2^2)/3 = 0.126667
          #       calibrated    = ((0.6-1)^2 + 0.25^2 + 0.15^2)/3 = 0.081667
          #       difference    = +0.045  (calibrated better)
          ("1", 0.5, 0.3, 0.2, 0.6, 0.25, 0.15),
          # draw. raw brier = (0.4^2 + (0.4-1)^2 + 0.2^2)/3 = 0.186667
          #       calibrated   = (0.5^2 + (0.3-1)^2 + 0.2^2)/3 = 0.260000
          #       difference   = -0.073333  (calibrated worse)
          ("X", 0.4, 0.4, 0.2, 0.5, 0.3, 0.2),
          # unplayed -- no result to score against, must not be counted
          (None, 0.4, 0.4, 0.2, 0.5, 0.3, 0.2),
      ],
      SCORED_SCHEMA,
  )

  stats = paired_brier_difference(df)

  assert stats["n"] == 2  # the unplayed row is excluded
  assert abs(stats["mean_difference"] - ((0.045 + -0.073333333) / 2)) < 1e-6

  # the interval is centred on the mean and has real width
  assert stats["ci_low"] < stats["mean_difference"] < stats["ci_high"]


BLEND_SCHEMA = (
    'result string, good_1 double, good_x double, good_2 double, '
    'bad_1 double, bad_x double, bad_2 double'
)

def test_blend(spark):
  df = spark.createDataFrame(
      [("1", 0.6, 0.25, 0.15, 0.4, 0.35, 0.25)], BLEND_SCHEMA
  )

  row = blend(df, "good", "bad", 0.5).first()
  assert abs(row["blended_1"] - 0.5) < 1e-9
  assert abs(sum(row[f"blended_{s}"] for s in ("1", "x", "2")) - 1.0) < 1e-9

  # weight 1 ignores the right-hand side entirely
  row = blend(df, "good", "bad", 1.0).first()
  assert abs(row["blended_1"] - 0.6) < 1e-9

def test_fit_blend_weight(spark):
  # `good` names the actual outcome every time, `bad` names the opposite.
  # The only sensible answer is to put all the weight on `good`.
  df = spark.createDataFrame(
      [
          ("1", 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
          ("2", 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
          ("X", 0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
      ],
      BLEND_SCHEMA,
  )

  weight, brier = fit_blend_weight(df, "good", "bad")

  assert weight == 1.0
  assert brier == 0.0  # a perfect forecast scores zero
