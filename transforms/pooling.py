"""Pure transform functions used by backtest.

Combines two sets of 1X2 probabilities with a multinomial logistic
regression, in log-odds space.

Why not just average them: a weighted average always lands *between* its
inputs. If the crowd says 70% and Elo says 75%, no weight produces 78% --
yet two independent sources agreeing should arguably be more confident
than either alone. Averaging probabilities is "linear pooling", and it is
known to produce systematically under-confident forecasts for exactly
that reason.

Log-odds is the scale on which combining evidence becomes addition, which
is what Bayes' rule does with independent observations. And a logistic
regression's coefficients are not forced to sum to 1, so it can learn 0.7
and 0.6 and land outside the range of both inputs -- the freedom a blend
does not have.
"""

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from sklearn.linear_model import LogisticRegression

SUFFIXES = ("1", "x", "2")
CLASSES = ("1", "X", "2")
CLIP = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    """Clipped, because a probability of exactly 0 or 1 has infinite odds."""
    p = np.clip(p, CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p))


def _features(pdf: pd.DataFrame, sources: tuple, interaction: bool) -> np.ndarray:
    """Log-odds of every outcome from every source, optionally times how
    uncertain the first source is.

    The interaction is the one place a genuinely different answer could
    hide: it asks whether the second source earns its keep only where the
    first has no strong opinion. An additive fit cannot discover that on
    its own -- it has to be asked explicitly.
    """
    blocks = [
        _logit(pdf[f"{source}_{suffix}"].to_numpy())
        for source in sources
        for suffix in SUFFIXES
    ]
    features = np.column_stack(blocks)

    if interaction and len(sources) > 1:
        first = np.column_stack(
            [pdf[f"{sources[0]}_{suffix}"].to_numpy() for suffix in SUFFIXES]
        )
        # entropy of the first source, scaled to 0..1: 1 = no opinion at all
        safe = np.clip(first, CLIP, 1.0)
        uncertainty = -(safe * np.log(safe)).sum(axis=1) / np.log(len(SUFFIXES))
        second = np.column_stack(
            [_logit(pdf[f"{sources[1]}_{suffix}"].to_numpy()) for suffix in SUFFIXES]
        )
        features = np.column_stack([features, second * uncertainty[:, None]])

    return features


def _frame(df: DataFrame, sources: tuple) -> pd.DataFrame:
    columns = [f"{source}_{suffix}" for source in sources for suffix in SUFFIXES]
    return (
        df.filter(F.col("result").isNotNull())
        .select("match_id", "result", *columns)
        .toPandas()
    )


def fit_logistic_pool(
    train: DataFrame, sources: tuple = ("calibrated", "elo"), interaction: bool = False
) -> LogisticRegression:
    """Fit on the training half only -- the usual rule.

    Fitted in pandas rather than through a pandas_udf: the training half
    is a few thousand rows, sklearn needs numpy anyway, and predict_proba
    returns all three classes at once, which a scalar UDF cannot express
    without predicting three times over.
    """
    pdf = _frame(train, sources)
    features = _features(pdf, sources, interaction)

    model = LogisticRegression(max_iter=1000)
    model.fit(features, pdf["result"].to_numpy())

    # Stashed so coefficients can be reported per standard deviation. Raw
    # coefficients are not comparable across sources: the crowd's log-odds
    # swing much further than Elo's, so a smaller coefficient on the crowd
    # can still carry more influence.
    model.feature_sd_ = features.std(axis=0)
    return model


def apply_logistic_pool(
    df: DataFrame,
    model: LogisticRegression,
    sources: tuple = ("calibrated", "elo"),
    interaction: bool = False,
    prefix: str = "pooled",
) -> DataFrame:
    """Add <prefix>_1/_x/_2 from the fitted model, joined back on match_id."""
    pdf = _frame(df, sources)
    probabilities = model.predict_proba(_features(pdf, sources, interaction))

    order = {label: index for index, label in enumerate(model.classes_)}
    out = pd.DataFrame({"match_id": pdf["match_id"]})
    for suffix, label in zip(SUFFIXES, CLASSES):
        out[f"{prefix}_{suffix}"] = probabilities[:, order[label]]

    return df.join(df.sparkSession.createDataFrame(out), on="match_id", how="inner")


def coefficient_summary(
    model: LogisticRegression, sources: tuple = ("calibrated", "elo"), interaction: bool = False
) -> str:
    """Mean absolute coefficient per source, **standardised** -- the effect
    of moving that feature by one standard deviation.

    Raw coefficients would be misleading here. The crowd's log-odds have a
    far wider spread than Elo's, so the same raw coefficient means much
    more influence on the crowd side. Scaling by each feature's standard
    deviation makes the two comparable; a near-zero figure then really does
    mean the fit ignores that source.
    """
    weights = np.abs(model.coef_).mean(axis=0) * getattr(
        model, "feature_sd_", np.ones(model.coef_.shape[1])
    )
    labels = list(sources) + (["interaction"] if interaction else [])
    parts = []
    for index, label in enumerate(labels):
        block = weights[index * len(SUFFIXES) : (index + 1) * len(SUFFIXES)]
        parts.append(f"{label}={block.mean():.3f}")
    return "  ".join(parts)
