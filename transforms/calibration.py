"""Pure transform functions used by 05_gold_calibrate."""

from pyspark.sql import DataFrame, functions as F
import pandas as pd
from sklearn.isotonic import IsotonicRegression

OUTCOMES = ("1", "X", "2")


def to_long_format(df: DataFrame) -> DataFrame:
    """One row per (match, outcome): predicted_prob is the streck value,
    outcome is '1'/'X'/'2', actual_occurred is True when that outcome is
    what actually happened.

    Unplayed fixtures are dropped: they have no result to learn from, and
    including them would train the calibration curves on outcomes that
    never occurred. Only the fitting path filters -- calibrate_matches
    still scores every match, which is the point for an upcoming coupon.
    """
    return (
        df.filter(F.col("result").isNotNull())
        .unpivot(
            ids=["draw_number", "event_number", "match_id", "result"],
            values=["streck_1", "streck_x", "streck_2"],
            variableColumnName="streck_col",
            valueColumnName="predicted_prob",
        )
        .withColumn(
            "outcome",
            F.upper(F.regexp_replace("streck_col", "^streck_", "")),
        )
        .withColumn("actual_occurred", F.col("outcome") == F.col("result"))
        .orderBy("draw_number", "event_number")
    )


def fit_calibration_curve(df_long: DataFrame, outcome: str) -> IsotonicRegression:
    """Fit one outcome's calibration curve. Done separately per outcome
    ('1'/'X'/'2') because the crowd's over/under-confidence isn't the same
    for home/away wins as it is for draws."""
    subset = (
        df_long.filter(F.col("outcome") == outcome)
        .select("predicted_prob", "actual_occurred")
        .toPandas()
    )

    x = subset["predicted_prob"].to_numpy()
    y = subset["actual_occurred"].to_numpy().astype(float)

    model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    model.fit(x, y)

    return model


def fit_all_calibration_curves(df_long: DataFrame) -> dict[str, IsotonicRegression]:
    """Returns {'1': model, 'X': model, '2': model}."""
    return {outcome: fit_calibration_curve(df_long, outcome) for outcome in OUTCOMES}


def calibrate_matches(df: DataFrame, models: dict[str, IsotonicRegression]) -> DataFrame:
    """Apply the fitted per-outcome models to streck_1/streck_x/streck_2
    for every match, adding calibrated_1/calibrated_x/calibrated_2
    columns. Renormalize each row so the three calibrated probabilities
    sum to 1 -- independent per-outcome fits won't sum to 1 on their own.

    Databricks serverless runs on Spark Connect, which doesn't expose
    sparkContext to the client -- so this closes over each fitted model
    directly in the pandas_udf instead of using sc.broadcast(); Spark
    Connect ships the closure to the workers on its own.
    """
    def make_predict_udf(model: IsotonicRegression):
        @F.pandas_udf("double")
        def _predict(raw: pd.Series) -> pd.Series:
            return pd.Series(model.predict(raw.to_numpy()))
        return _predict

    raw_calibrated = (
        df.withColumn("calibrated_1", make_predict_udf(models["1"])(F.col("streck_1")))
        .withColumn("calibrated_x", make_predict_udf(models["X"])(F.col("streck_x")))
        .withColumn("calibrated_2", make_predict_udf(models["2"])(F.col("streck_2")))
    )

    row_sum = F.col("calibrated_1") + F.col("calibrated_x") + F.col("calibrated_2")

    return (
        raw_calibrated
        .withColumn("calibrated_1", F.col("calibrated_1") / row_sum)
        .withColumn("calibrated_x", F.col("calibrated_x") / row_sum)
        .withColumn("calibrated_2", F.col("calibrated_2") / row_sum)
    )


def evaluate_calibration(df_calibrated: DataFrame) -> dict[str, float]:
    """Prints raw vs. calibrated Brier score per outcome + pooled, and
    returns the same numbers so main() can log them to MLflow."""
    column_pairs = {
        "1": ("streck_1", "calibrated_1"),
        "X": ("streck_x", "calibrated_x"),
        "2": ("streck_2", "calibrated_2"),
    }

    metrics: dict[str, float] = {}

    print(f"{'outcome':<8}{'raw brier':>12}{'calibrated brier':>20}")
    for outcome, (raw_col, calibrated_col) in column_pairs.items():
        actual = (F.col("result") == outcome).cast("double")
        row = df_calibrated.select(
            F.avg(F.pow(F.col(raw_col) - actual, 2)).alias("raw_brier"),
            F.avg(F.pow(F.col(calibrated_col) - actual, 2)).alias("calibrated_brier"),
        ).first()

        print(f"{outcome:<8}{row['raw_brier']:>12.4f}{row['calibrated_brier']:>20.4f}")
        metrics[f"brier_raw_{outcome}"] = row["raw_brier"]
        metrics[f"brier_calibrated_{outcome}"] = row["calibrated_brier"]

    pooled_raw = sum(metrics[f"brier_raw_{o}"] for o in OUTCOMES) / len(OUTCOMES)
    pooled_calibrated = sum(metrics[f"brier_calibrated_{o}"] for o in OUTCOMES) / len(OUTCOMES)
    print(f"{'pooled':<8}{pooled_raw:>12.4f}{pooled_calibrated:>20.4f}")
    metrics["brier_raw_pooled"] = pooled_raw
    metrics["brier_calibrated_pooled"] = pooled_calibrated

    return metrics
