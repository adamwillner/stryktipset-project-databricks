"""Pure transform function used by 04_gold_fact_match."""

from pyspark.sql import DataFrame, functions as F

from transforms.dimensions import date_key


def build_fact_match(matches: DataFrame, dim_league: DataFrame) -> DataFrame:
    """Join silver matches to dim_league -- an inner join, which is also
    what scopes gold to the English leagues, since dim_league holds only
    those. Derive the surrogate keys, bring across goals and odds.

    Keys are the natural ones: home_team_key/away_team_key are the
    uppercased team names, league_key the uppercased league name. Surrogate
    keys were tried on 2026-09-29 and removed the same day -- see
    project-context for the evidence. Short version: they only pay off with
    SCD Type 2 versioning, and team names in this data have never changed.

    calibrated_1/x/2 and elo_home/elo_away are created here as null
    placeholders and filled in later by 05 and 06 -- Delta resolves a named
    MERGE assignment before schema evolution runs, so the columns have to
    exist first.
    """
    joined_df = matches.join(
        dim_league, on=matches['league'] == dim_league['league_name'], how='inner'
    )

    # Seasons run Aug-May, so July decides which year pair a match belongs
    # to. This used to branch per country; Sweden left gold on 2026-09-29.
    # Same rule as build_dim_season -- kept in step by test_build_fact_match.
    joined_df = joined_df.withColumn(
        'season_start_year',
        F.when(F.month('match_start') >= 7, F.year('match_start'))
        .otherwise(F.year('match_start') - 1),
    )

    return (
        joined_df
        .withColumn('match_key', F.col('match_id'))
        .withColumn('home_team_key', F.upper(F.col('home_team')))
        .withColumn('away_team_key', F.upper(F.col('away_team')))
        .withColumn('date_key', date_key(F.col('match_start')))
        .withColumn(
            'season_key',
            F.concat(
                F.col('season_start_year').cast('string'),
                F.lit('/'),
                (F.col('season_start_year') + 1).cast('string'),
            ),
        )
        .select(
            F.col('match_key'),
            F.col('home_team_key'),
            F.col('away_team_key'),
            F.col('date_key'),
            F.col('league_key'),
            F.col('season_key'),
            F.col('home_goals'),
            F.col('away_goals'),
            F.col('result'),
            F.col('streck_1'),
            F.col('streck_x'),
            F.col('streck_2'),
            F.col('odds_1'),
            F.col('odds_x'),
            F.col('odds_2'),
            F.col('start_odds_1'),
            F.col('start_odds_x'),
            F.col('start_odds_2'),
            F.col('favourite_odds_1'),
            F.col('favourite_odds_x'),
            F.col('favourite_odds_2'),
            F.col('status'),
            F.col('betradar_id'),
        )
        .withColumn('calibrated_1', F.lit(None).cast('double'))
        .withColumn('calibrated_x', F.lit(None).cast('double'))
        .withColumn('calibrated_2', F.lit(None).cast('double'))
        .withColumn('elo_home', F.lit(None).cast('double'))
        .withColumn('elo_away', F.lit(None).cast('double'))
    )
