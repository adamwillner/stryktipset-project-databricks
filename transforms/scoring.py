"""Pure transform functions used by 07_score_coupon."""

from pyspark.sql import Column, DataFrame, functions as F

from transforms.dimensions import (
    UNKNOWN_LEAGUE_NAME,
    date_key,
    league_surrogate_key,
    team_surrogate_key,
)
from transforms.elo import HOME_ADVANTAGE


def expected_home_score_column(home_rating: Column, away_rating: Column) -> Column:
    """Spark-column twin of transforms.elo.expected_home_score.

    The formula is duplicated rather than imported because
    expected_home_score works on plain floats inside build_elo's loop,
    while this one has to be a column expression over a DataFrame.
    tests/test_scoring.py asserts the two agree for the same ratings.

    The result is an expected *score*, not a probability: on Elo's chess
    scale a win is 1, a draw 0.5 and a loss 0, so it is roughly
    `P(home win) + P(draw)/2`. Hence the column name `elo_expected_score`
    -- it was briefly called `elo_prob_1`, which invited exactly the
    misreading that broke 07's first gap column. Splitting it into a real
    1X2 probability needs a draw model and is a separate problem.
    """
    gap = away_rating - (home_rating + F.lit(HOME_ADVANTAGE))
    return F.lit(1.0) / (F.lit(1.0) + F.pow(F.lit(10.0), gap / F.lit(400.0)))


def build_mart_coupon(
    matches: DataFrame, elo: DataFrame, dim_league: DataFrame, draw_number: int
) -> DataFrame:
    """One row per match on a given draw, carrying every opinion available
    about it: the crowd (`streck_*`), the crowd corrected for its known
    biases (`calibrated_*`), and Elo.

    This is a **mart** table, deliberately not part of the star. A star
    records what happened; these are model outputs, read by a person on a
    Thursday. Being denormalised is the point of a mart, so team and league
    names are carried as plain text -- SELECT * reads like a coupon with no
    join. The surrogate keys are carried too, so it can still be joined
    back to the dimensions when you want to aggregate.

    Reads silver rather than fact_match on purpose. Gold covers the English
    leagues only, but a coupon regularly carries Norwegian, Scottish or
    Spanish fixtures, and a scorer that silently returned 11 of 13 rows
    would be worse than useless. Those matches resolve to dim_league's
    Unknown member rather than to a null key.

    `matches` must already be calibrated -- 07 runs calibrate_matches over
    silver before calling this.

    Left join on Elo so a match with no rating still appears, with nulls.
    A missing rating means a team with no prior match anywhere in the data,
    which is worth seeing rather than quietly dropping.
    """
    coupon = matches.filter(F.col("draw_number") == draw_number)

    scoped_league = dim_league.select(
        F.col("league_name").alias("_league_name"), F.col("league_sk").alias("_league_sk")
    )

    return (
        coupon.join(elo, on="match_id", how="left")
        .join(scoped_league, on=coupon["league"] == scoped_league["_league_name"], how="left")
        .withColumn("match_key", F.col("match_id"))
        .withColumn("home_team_sk", team_surrogate_key(F.col("home_team")))
        .withColumn("away_team_sk", team_surrogate_key(F.col("away_team")))
        .withColumn("date_key", date_key(F.col("match_start")))
        .withColumn(
            "league_sk",
            F.coalesce(
                F.col("_league_sk"),
                league_surrogate_key(F.lit(UNKNOWN_LEAGUE_NAME)),
            ),
        )
        .withColumn(
            "elo_expected_score",
            expected_home_score_column(F.col("elo_home"), F.col("elo_away")),
        )
        .withColumn("scored_at", F.current_timestamp())
        .select(
            "draw_number",
            "event_number",
            "match_key",
            "match_start",
            "home_team",
            "away_team",
            "league",
            "status",
            "home_team_sk",
            "away_team_sk",
            "date_key",
            "league_sk",
            "streck_1",
            "streck_x",
            "streck_2",
            "calibrated_1",
            "calibrated_x",
            "calibrated_2",
            "elo_home",
            "elo_away",
            "elo_expected_score",
            "scored_at",
        )
        .orderBy("event_number")
    )
