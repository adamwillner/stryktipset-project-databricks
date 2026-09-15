"""Tests for transforms/calibration.py.

TODO: write tests here. Some starting points:
- to_long_format: 3 wide streck columns become 3 long rows per match,
  actual_occurred is True on exactly one of them (the one matching
  `result`).
- fit_calibration_curve / fit_all_calibration_curves: on a tiny synthetic
  df_long, does the fitted IsotonicRegression at least produce values in
  [0, 1]? (Exact calibration curve isn't really unit-testable -- it's a
  statistical fit, not a pure function -- so keep these to shape/sanity
  checks, not exact-value assertions.)
- calibrate_matches: calibrated_1 + calibrated_x + calibrated_2 sums to 1
  per row after renormalizing.
- evaluate_calibration: Brier score is 0 for a perfect prediction, 1 for
  the worst possible one.
"""
import pytest
from sklearn.isotonic import IsotonicRegression

from transforms.calibration import (
    OUTCOMES,
    to_long_format,
    fit_calibration_curve,
    fit_all_calibration_curves,
    calibrate_matches,
    evaluate_calibration,
)

def test_to_long_format(spark):
  df = spark.createDataFrame(
      [(1, 1, 123, "1", 0.45, 0.30, 0.25)], 
      ["draw_number", "event_number", "match_id", "result", "streck_1", "streck_x", "streck_2"]
    )

  result = to_long_format(df)

  rows = {row["outcome"]: row for row in result.collect()}

  assert rows["1"]["predicted_prob"] == 0.45
  assert rows["X"]["predicted_prob"] == 0.30
  assert rows["2"]["predicted_prob"] == 0.25

  assert rows['1']['actual_occurred'] == True
  assert rows["X"]["actual_occurred"] == False 
  assert rows["2"]["actual_occurred"] == False

def test_fit_calibration_curve(spark):
    # made-up but deliberately monotonic: as predicted_prob goes up,
    # actual_occurred flips from False to True
    df_long = spark.createDataFrame(
        [
            ("1", 0.1, False),
            ("1", 0.3, False),
            ("1", 0.5, True),
            ("1", 0.7, True),
            ("1", 0.9, True),
        ],
        ["outcome", "predicted_prob", "actual_occurred"],
    )

    model = fit_calibration_curve(df_long, "1")
    predictions = model.predict([0.1, 0.5, 0.9])

    # isotonic regression is non-decreasing by definition -- can't check
    # exact values (it's a statistical fit), but this must always hold
    assert predictions[0] <= predictions[1] <= predictions[2]
    assert all(0 <= p <= 1 for p in predictions)


def test_calibrate_matches(spark):
    df = spark.createDataFrame([(0.50, 0.30, 0.20)], ["streck_1", "streck_x", "streck_2"])

    # an "identity" model -- fit on (0,0) and (1,1) so predict(x) == x,
    # which makes the expected output easy to know in advance
    identity_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    identity_model.fit([0, 1], [0, 1])
    models = {"1": identity_model, "X": identity_model, "2": identity_model}

    result = calibrate_matches(df, models)
    row = result.collect()[0]

    # streck_1/x/2 already sum to 1, and the models don't change anything,
    # so renormalizing should leave them unchanged
    assert row["calibrated_1"] == pytest.approx(0.50)
    assert row["calibrated_x"] == pytest.approx(0.30)
    assert row["calibrated_2"] == pytest.approx(0.20)


def test_evaluate_calibration(spark):
    # result "1" actually happened; calibrated columns predict it perfectly
    # (1.0/0.0/0.0), raw streck columns don't -- lets us check both brier
    # numbers land where they should
    df_calibrated = spark.createDataFrame(
        [("1", 0.50, 0.30, 0.20, 1.0, 0.0, 0.0)],
        ["result", "streck_1", "streck_x", "streck_2", "calibrated_1", "calibrated_x", "calibrated_2"],
    )

    metrics = evaluate_calibration(df_calibrated)

    assert metrics["brier_raw_1"] == pytest.approx(0.25)       # (0.50 - 1)^2
    assert metrics["brier_calibrated_1"] == pytest.approx(0.0)  # (1.0 - 1)^2
    assert metrics["brier_calibrated_pooled"] == pytest.approx(0.0)
