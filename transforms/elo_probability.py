"""Pure transform functions used by backtest.

Turns Elo's single number into three probabilities, so it can be scored
against 1/X/2 like any other prediction.
"""

from functools import reduce

import pandas as pd
from pyspark.sql import Column, DataFrame, functions as F
from sklearn.isotonic import IsotonicRegression

OUTCOMES = ("1", "X", "2")


def elo_predictor(outcome: str) -> Column:
    """What to feed the isotonic fit for each outcome.

    Isotonic regression can only fit a curve that goes one way. P(home
    win) rises steadily with the Elo expected score, so that one is
    straightforward -- but P(draw) does not: draws peak when teams are
    evenly matched and fall away in both directions. A hump, not a slope,
    and isotonic would flatten it into nonsense.

    Feeding it *evenness* instead -- 0.5 minus the distance from an even
    match -- turns the hump into a clean rising slope: the more evenly
    matched, the likelier the draw. Same tool, different input.
    """
    expected = F.col("elo_expected_score")
    if outcome == "1":
        return expected
    if outcome == "2":
        return F.lit(1.0) - expected
    return F.lit(0.5) - F.abs(expected - F.lit(0.5))


def to_elo_long_format(df: DataFrame) -> DataFrame:
    """One row per (match, outcome), shaped exactly like
    calibration.to_long_format so fit_all_calibration_curves can be reused
    unchanged: predicted_prob, outcome, actual_occurred.

    Played matches only -- an unplayed fixture has no result to learn from.
    """
    played = df.filter(F.col("result").isNotNull())

    parts = [
        played.select(
            F.lit(outcome).alias("outcome"),
            elo_predictor(outcome).alias("predicted_prob"),
            (F.col("result") == F.lit(outcome)).alias("actual_occurred"),
        )
        for outcome in OUTCOMES
    ]
    return reduce(DataFrame.unionByName, parts)


def apply_elo_models(df: DataFrame, models: dict) -> DataFrame:
    """Turn elo_expected_score into elo_1/elo_x/elo_2 using the fitted
    curves, then renormalise so the three sum to 1.

    Three independent fits will not sum to 1 on their own, exactly as with
    the streck calibration. Same closure-over-the-model trick too: Spark
    Connect has no sparkContext to broadcast through, so the pandas_udf
    captures the fitted model directly.
    """
    def predict_with(model: IsotonicRegression):
        @F.pandas_udf("double")
        def _predict(raw: pd.Series) -> pd.Series:
            return pd.Series(model.predict(raw.to_numpy()))
        return _predict

    out = df
    for outcome in OUTCOMES:
        column = f"elo_{outcome.lower()}"
        out = out.withColumn(column, predict_with(models[outcome])(elo_predictor(outcome)))

    row_sum = F.col("elo_1") + F.col("elo_x") + F.col("elo_2")
    for outcome in OUTCOMES:
        column = f"elo_{outcome.lower()}"
        out = out.withColumn(column, F.col(column) / row_sum)
    return out


def combine(df: DataFrame, left: str, right: str, prefix: str = "combined") -> DataFrame:
    """Average two sets of probabilities, outcome by outcome.

    The simplest way to ask whether one prediction knows anything the
    other doesn't. If the average beats both, they carry independent
    information; if it lands between them, one is just noise on the other.
    """
    out = df
    for suffix in ("1", "x", "2"):
        out = out.withColumn(
            f"{prefix}_{suffix}",
            (F.col(f"{left}_{suffix}") + F.col(f"{right}_{suffix}")) / F.lit(2.0),
        )
    return out
