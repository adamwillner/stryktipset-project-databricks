"""Pure transform functions used by 07_score_coupon."""

from pyspark.sql import Column, DataFrame, functions as F

from transforms.elo import HOME_ADVANTAGE


def elo_probability(home_rating: Column, away_rating: Column) -> Column:
    """Spark-column twin of transforms.elo.expected_home_score.

    The formula is duplicated rather than imported because
    expected_home_score works on plain floats inside build_elo's loop,
    while this one has to be a column expression over a DataFrame.
    tests/test_scoring.py asserts the two agree for the same ratings --
    the same guard test_build_fact_match applies to season_key, which is
    duplicated between build_dim_season and build_fact_match.

    Note this is P(home win) on Elo's chess-shaped scale, where a draw
    counts as half a win. It is not a 1X2 probability and must not be
    read as one: splitting it into three outcomes is a separate problem.
    """
    gap = away_rating - (home_rating + F.lit(HOME_ADVANTAGE))
    return F.lit(1.0) / (F.lit(1.0) + F.pow(F.lit(10.0), gap / F.lit(400.0)))


def build_coupon_predictions(
    matches: DataFrame, elo: DataFrame, draw_number: int
) -> DataFrame:
    """One row per match on a given draw, carrying every opinion available
    about it: the crowd (`streck_*`), the crowd corrected for its known
    biases (`calibrated_*`), and Elo.

    `matches` must already be calibrated -- 07 runs calibrate_matches over
    silver before calling this, so the calibrated_* columns are present.

    Reads silver rather than fact_match on purpose. gold is scoped to
    England/Sweden, but roughly 16% of coupon matches are not (Eliteserien,
    Scottish Premiership, La Liga...), and a scorer that silently returned
    11 of 13 rows would be worse than useless.

    Left join on Elo so a match with no rating still appears, with nulls.
    A missing rating means a team with no prior match anywhere in the data,
    which is worth seeing rather than quietly dropping.
    """
    coupon = matches.filter(F.col("draw_number") == draw_number)

    return (
        coupon.join(elo, on="match_id", how="left")
        .withColumn(
            "elo_prob_1", elo_probability(F.col("elo_home"), F.col("elo_away"))
        )
        .select(
            "draw_number",
            "event_number",
            "match_id",
            "match_start",
            "league",
            "home_team",
            "away_team",
            "status",
            "streck_1",
            "streck_x",
            "streck_2",
            "calibrated_1",
            "calibrated_x",
            "calibrated_2",
            "elo_home",
            "elo_away",
            "elo_prob_1",
        )
        .orderBy("event_number")
    )
