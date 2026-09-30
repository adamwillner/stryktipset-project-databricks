from transforms.pooling import apply_logistic_pool, coefficient_summary, fit_logistic_pool

SCHEMA = (
    'match_id string, result string, '
    'good_1 double, good_x double, good_2 double, '
    'noise_1 double, noise_x double, noise_2 double'
)

def _rows():
  """`good` points at the real outcome; `noise` says the same thing every
  time and so carries no information at all."""
  pattern = ["1", "X", "2"]
  rows = []
  for i in range(30):
    outcome = pattern[i % 3]
    good = {
        "1": (0.8, 0.1, 0.1),
        "X": (0.1, 0.8, 0.1),
        "2": (0.1, 0.1, 0.8),
    }[outcome]
    rows.append((f"m{i}", outcome, *good, 0.34, 0.33, 0.33))
  return rows

def test_fit_logistic_pool(spark):
  df = spark.createDataFrame(_rows(), SCHEMA)

  model = fit_logistic_pool(df, sources=("good", "noise"))

  # it should lean on the informative source and ignore the constant one
  summary = coefficient_summary(model, sources=("good", "noise"))
  weights = dict(part.split("=") for part in summary.split())
  assert float(weights["good"]) > float(weights["noise"])

def test_apply_logistic_pool(spark):
  df = spark.createDataFrame(_rows(), SCHEMA)
  model = fit_logistic_pool(df, sources=("good", "noise"))

  result = apply_logistic_pool(df, model, sources=("good", "noise"))
  rows = {row["match_id"]: row for row in result.collect()}

  assert result.count() == 30
  for row in rows.values():
    probabilities = [row[f"pooled_{s}"] for s in ("1", "x", "2")]
    assert abs(sum(probabilities) - 1.0) < 1e-9          # a real distribution
    assert max(probabilities) > 0.5                       # and a confident one

  # the most likely outcome should be the one that actually happened
  best = {"1": "pooled_1", "X": "pooled_x", "2": "pooled_2"}
  for row in rows.values():
    picked = max(("1", "x", "2"), key=lambda s: row[f"pooled_{s}"])
    assert best[row["result"]] == f"pooled_{picked}"
