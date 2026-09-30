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
