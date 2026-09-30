from transforms.backtest import split_by_date

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
