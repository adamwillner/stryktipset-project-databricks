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

from transforms.calibration import (
    OUTCOMES,
    to_long_format,
    fit_calibration_curve,
    fit_all_calibration_curves,
    calibrate_matches,
    evaluate_calibration,
)
