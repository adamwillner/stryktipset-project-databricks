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


def paired_brier_difference(
    df_calibrated: DataFrame, baseline: str = "streck", candidate: str = "calibrated"
) -> dict:
    """Is calibrated actually better than raw, or is the gap noise?

    Scores each match twice -- once from <baseline>_1/_x/_2, once from
    <candidate>_1/_x/_2 -- and looks at the *difference per match*.
    Positive means the candidate did better on that match. Defaults compare
    raw streck against calibrated streck; pass other prefixes to compare
    anything else on the same footing.

    Paired on purpose. The two predictions describe the same matches, so
    comparing them match by match cancels out the thing that dominates the
    variance: some matches are simply easier to call than others. An
    unpaired comparison would drown a real effect in that noise.

    Returns the mean difference, its standard error, a t statistic and a
    95% interval. If the interval excludes zero, the improvement is real
    at that sample size; if it straddles zero, the data cannot tell.
    """
    scored = df_calibrated.filter(F.col("result").isNotNull()).withColumn(
        "brier_difference",
        _per_match_brier(baseline) - _per_match_brier(candidate),
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

def pooled_brier(df: DataFrame, prefix: str) -> float:
    """Pooled Brier score for one set of probability columns
    (<prefix>_1/_x/_2). Lower is better; 0.2222 is what predicting
    33/33/33 every time scores, whatever the real base rates are."""
    scored = df.filter(F.col("result").isNotNull())
    return scored.select(F.avg(_per_match_brier(prefix)).alias("brier")).first()["brier"]

SUFFIXES = ("1", "x", "2")


def blend(
    df: DataFrame, left: str, right: str, weight: float, prefix: str = "blended"
) -> DataFrame:
    """weight * left + (1 - weight) * right, outcome by outcome.

    weight = 1 means ignore `right` entirely; 0.5 is a plain average.
    """
    out = df
    for suffix in SUFFIXES:
        out = out.withColumn(
            f"{prefix}_{suffix}",
            F.lit(weight) * F.col(f"{left}_{suffix}")
            + F.lit(1.0 - weight) * F.col(f"{right}_{suffix}"),
        )
    return out


def fit_blend_weight(
    df: DataFrame, left: str, right: str, steps: int = 21
) -> tuple[float, float]:
    """Search weights from 0 to 1 and return (best weight, its Brier).

    **Call this on the training half only.** Searching for the weight that
    looks best on the test set would be choosing the answer to the exam
    you are about to sit -- the same leakage that makes 05's in-sample
    scores meaningless, just smaller.

    A fixed 0.5 average is a blunt test: blending a clearly weaker
    forecast at half weight will hurt almost regardless, so it measures
    the choice of weight as much as the forecast. Fitting the weight asks
    the better question -- is there *any* amount of `right` that helps? A
    best weight of 1.0 means no, and that is a far stronger statement than
    "my arbitrary average did not work".

    Searched in pandas rather than Spark: the training half is a few
    thousand rows, and 21 candidate weights would otherwise be 21 passes.

    Note the curves producing these probabilities were themselves fitted
    on this same data, so the Brier values here are in-sample and the
    chosen weight is mildly optimistic. It is one parameter, and the
    judgement that matters still happens on the untouched test half.
    """
    columns = [f"{side}_{suffix}" for side in (left, right) for suffix in SUFFIXES]
    pdf = df.filter(F.col("result").isNotNull()).select("result", *columns).toPandas()

    actual = {
        suffix: (pdf["result"] == value).to_numpy(dtype=float)
        for suffix, value in zip(SUFFIXES, ("1", "X", "2"))
    }

    best_weight, best_brier = None, None
    for step in range(steps):
        weight = step / (steps - 1)
        total = 0.0
        for suffix in SUFFIXES:
            blended = (
                weight * pdf[f"{left}_{suffix}"].to_numpy()
                + (1.0 - weight) * pdf[f"{right}_{suffix}"].to_numpy()
            )
            total += ((blended - actual[suffix]) ** 2).mean()
        brier = total / len(SUFFIXES)
        if best_brier is None or brier < best_brier:
            best_weight, best_brier = weight, brier

    return best_weight, best_brier
