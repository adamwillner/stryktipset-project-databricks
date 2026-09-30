"""Pure transform functions used by backtest_calibration."""

from pyspark.sql import DataFrame, functions as F


def split_by_date(matches: DataFrame, cutoff: str) -> tuple[DataFrame, DataFrame]:
    """Split matches into (train, test) at a date. Everything before the
    cutoff trains; the cutoff day itself and everything after is test.

    **Chronological, never random.** A random split would put 2025 matches
    in the training set and 2020 matches in the test set, so the model
    would learn from the future to score the past -- and the resulting
    score would flatter it. It would also cut a single draw across both
    sides, putting near-identical rows in train and test.

    A date rather than a percentage, so the boundary lands somewhere
    meaningful (a season edge) instead of slicing a draw in half.

    Only played matches can be scored, so unplayed fixtures are dropped
    here rather than silently skewing the test set: they carry a null
    result, which no Brier score can use.
    """
    played = matches.filter(F.col('result').isNotNull())
    boundary = F.to_date(F.lit(cutoff))
    match_date = F.to_date('match_start')

    return (
        played.filter(match_date < boundary),
        played.filter(match_date >= boundary),
    )


OUTCOMES = (("1", "1"), ("x", "X"), ("2", "2"))  # column suffix -> result value


def _per_match_brier(prefix: str):
    """Brier score for one match from one set of probability columns:
    mean over the three outcomes of (predicted - actual) squared."""
    total = None
    for suffix, result_value in OUTCOMES:
        actual = (F.col("result") == F.lit(result_value)).cast("double")
        term = F.pow(F.col(f"{prefix}_{suffix}") - actual, 2)
        total = term if total is None else total + term
    return total / F.lit(float(len(OUTCOMES)))


def paired_brier_difference(df_calibrated: DataFrame) -> dict:
    """Is calibrated actually better than raw, or is the gap noise?

    Scores each match twice -- once from streck_*, once from calibrated_*
    -- and looks at the *difference per match*. Positive means calibrated
    did better on that match.

    Paired on purpose. The two predictions describe the same matches, so
    comparing them match by match cancels out the thing that dominates the
    variance: some matches are simply easier to call than others. An
    unpaired comparison would drown a real effect in that noise.

    Returns the mean difference, its standard error, a t statistic and a
    95% interval. If the interval excludes zero, the improvement is real
    at that sample size; if it straddles zero, the data cannot tell.
    """
    scored = df_calibrated.filter(F.col("result").isNotNull()).withColumn(
        "brier_difference", _per_match_brier("streck") - _per_match_brier("calibrated")
    )

    row = scored.agg(
        F.avg("brier_difference").alias("mean"),
        F.stddev("brier_difference").alias("sd"),
        F.count("*").alias("n"),
    ).first()

    mean, sd, n = row["mean"], row["sd"], row["n"]
    standard_error = (sd / (n ** 0.5)) if sd and n else None

    return {
        "mean_difference": mean,
        "sd": sd,
        "n": n,
        "standard_error": standard_error,
        "t_statistic": (mean / standard_error) if standard_error else None,
        "ci_low": (mean - 1.96 * standard_error) if standard_error else None,
        "ci_high": (mean + 1.96 * standard_error) if standard_error else None,
    }
