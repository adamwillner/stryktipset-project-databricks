"""Pure transform function used by 04_gold_fact_match."""

from pyspark.sql import DataFrame, functions as F


def build_fact_match(matches: DataFrame, dim_league: DataFrame) -> DataFrame:
    """Join silver matches to dim_league (this also scopes to
    England/Sweden — non-matching leagues drop out via the inner join).
    Derive team_key/date_key directly (pure functions of existing
    columns); get league_key + country from the join; derive season_key
    from country + match_start using the same per-country logic as
    build_dim_season. Bring across home_goals/away_goals and odds.
    Elo, calibrated_prob, xG: not yet -- added by later notebooks/passes.
    """
    joined_df = (
        matches
        .join(dim_league, on=matches['league'] == dim_league['league_name'], how='inner')
    )

    is_england = F.col('country') == 'England'

    joined_df = joined_df.withColumn(
        'season_start_year',
        F.when(
            is_england & (F.month('match_start') >= 7),
            F.year('match_start'))
        .when(
            is_england,
            F.year('match_start') - 1)
        .otherwise(
            F.year('match_start'))
    )

    return (
        joined_df
        .withColumn('match_key', F.col('match_id'))
        .withColumn('home_team_key', F.upper(F.col('home_team')))
        .withColumn('away_team_key', F.upper(F.col('away_team')))
        .withColumn('date_key', F.date_format(F.to_date("match_start"), "yyyyMMdd").cast("int"))
        .withColumn(
            'season_key',
            F.when(
                is_england,
                F.concat(F.col('season_start_year').cast('string'), F.lit('/'), (F.col('season_start_year') + 1).cast('string')))
            .otherwise(F.col('season_start_year').cast('string'))
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
        )
        .withColumn('calibrated_1', F.lit(None).cast('double'))
        .withColumn('calibrated_x', F.lit(None).cast('double'))
        .withColumn('calibrated_2', F.lit(None).cast('double'))
    )
