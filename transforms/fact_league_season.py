"""Pure transform function used by 04_gold_fact_match."""

from pyspark.sql import DataFrame, functions as F


def build_fact_league_season(fact_match: DataFrame) -> DataFrame:
    """One row per league per season, aggregated straight from fact_match.

    This is what makes gold a fact *constellation* rather than a single
    star: two fact tables at different grains sharing the same conformed
    dimensions (dim_league, dim_season). It needs no new data -- it is a
    summary of what fact_match already holds.

    Played matches only. An unplayed fixture has a null result and null
    goals, and counting it would quietly understate every rate below.

    Grain: league_sk + season_key.
    """
    played = fact_match.filter(F.col('result').isNotNull())
    goals = F.col('home_goals') + F.col('away_goals')

    return (
        played.groupBy('league_sk', 'season_key')
        .agg(
            F.count('*').alias('match_count'),
            F.sum(goals).alias('total_goals'),
            F.round(F.avg(goals), 3).alias('avg_goals'),
            F.sum((F.col('result') == '1').cast('int')).alias('home_wins'),
            F.sum((F.col('result') == 'X').cast('int')).alias('draws'),
            F.sum((F.col('result') == '2').cast('int')).alias('away_wins'),
        )
        .withColumn(
            'home_win_rate', F.round(F.col('home_wins') / F.col('match_count'), 3)
        )
    )
