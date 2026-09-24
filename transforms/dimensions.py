"""Pure transform functions used by 03_gold_dimensions.

LEAGUE_COUNTRY lives here too since build_dim_league is built directly
from it, and 03's notebook imports it from here rather than redefining
it.

Note on `spark`: 03's notebook has a `spark` global injected by
Databricks, but a function moved into an imported module does NOT see
that -- a function's globals are its *defining* module's globals, not
the caller's. build_dim_date sidesteps this by using `df.sparkSession`
(derived from the DataFrame it's already given) instead of a bare
`spark` reference. build_dim_league briefly took `spark` as an explicit
parameter for the same reason, but since LEAGUE_COUNTRY is a static dict
03's main() now builds the DataFrame itself and passes it in, so every
build_dim_* function has the same plain DataFrame shape.
"""

import datetime

from pyspark.sql import DataFrame, Window, functions as F

LEAGUE_COUNTRY = {
    "Premier League": "England",
    "Championship": "England",
    "League One": "England",
    "League Two": "England",
    "National League": "England",
    "Allsvenskan": "Sweden",
    "Superettan": "Sweden",
    "Ettan Norra": "Sweden",
    "Ettan Södra": "Sweden",
    "Div 2, Norra Götaland": "Sweden",
    "Div 2, Norra Svealand": "Sweden",
    "Div 2, Södra Götaland": "Sweden",
    "Div 2, Södra Svealand": "Sweden",
    "Div 2, Västra Götaland": "Sweden",
}


def build_dim_team(df: DataFrame) -> DataFrame:
    """One row per team: team_key, team_name, team_id, country.

    team_key stays the uppercased name -- it is what fact_match joins on.
    team_id is Svenska Spel's own id, which is NOT stable across the full
    history: they have renumbered at least once (1000041 in 2013, 448 by
    2023, 69 by 2026). So the dedupe takes the newest row per team via a
    window rather than dropDuplicates, which picks arbitrarily and could
    hand a team a decade-old id -- the same reason 02 dedupes match_id
    with row_number rather than dropDuplicates.
    """
    columns = lambda side: df.select(
        F.col(f'{side}_team').alias('team_name'),
        F.col(f'{side}_team_id').alias('team_id'),
        F.col(f'{side}_team_country').alias('country'),
        F.col('match_start'),
    )

    newest_per_team = Window.partitionBy('team_key').orderBy(F.col('match_start').desc())

    return (
        columns('home')
        .union(columns('away'))
        .withColumn('team_key', F.upper('team_name'))
        .withColumn('row_num', F.row_number().over(newest_per_team))
        .filter(F.col('row_num') == 1)
        .select('team_key', 'team_name', 'team_id', 'country')
    )


def build_dim_date(df: DataFrame) -> DataFrame:
    """Full calendar, one row per day -- not just dates that had matches.
    Spans from the earliest match in your data to two years past today,
    so upcoming draws are already covered without ever needing to extend
    this table by hand."""
    min_date = df.agg(F.min(F.to_date("match_start"))).first()[0]
    max_date = datetime.date.today() + datetime.timedelta(days=365 * 2)

    return (
        df.sparkSession.createDataFrame([(min_date, max_date)], ["start", "end"])
        .select(F.explode(F.sequence("start", "end", F.expr("interval 1 day"))).alias("full_date"))
        .withColumn("date_key", F.date_format("full_date", "yyyyMMdd").cast("int"))
        .withColumn("year", F.year("full_date"))
        .withColumn("quarter", F.quarter("full_date"))
        .withColumn("month", F.month("full_date"))
        .withColumn("month_name", F.date_format("full_date", "MMMM"))
        .withColumn("day", F.dayofmonth("full_date"))
        .withColumn("day_of_week", F.dayofweek("full_date"))
        .withColumn("day_of_week", F.when(F.col("day_of_week") == 1, 7).otherwise(F.col("day_of_week") - 1))  # ISO: 1=Monday..7=Sunday
        .withColumn("day_name", F.date_format("full_date", "EEEE"))
        .withColumn("week_of_year", F.weekofyear("full_date"))
        .withColumn("is_weekend", F.dayofweek("full_date").isin(1, 7))  # Spark: 1=Sunday, 7=Saturday
    )


def build_dim_league(df: DataFrame) -> DataFrame:
    """df: two columns, league_name/country -- the raw LEAGUE_COUNTRY dict
    as a DataFrame.

    Columns: league_key, league_name, country.
    """
    return (
        df
        .withColumn('league_key', F.upper('league_name'))
        .dropDuplicates(['league_key'])
    )


def build_dim_season(df: DataFrame) -> DataFrame:
    """One row per football season. England seasons span two calendar
    years (Aug-May, so a July cutoff decides which year pair); Sweden
    seasons sit inside one calendar year, so the key is just that year.

    Requires df to already have a `country` column (join against
    LEAGUE_COUNTRY before calling this).

    Columns: season_key, start_date, end_date, is_current.
    """
    df = df.select(F.to_date('match_start').alias('match_start'), 'country')

    is_england = F.col('country') == 'England'

    df = df.withColumn(
        'season_start_year',
        F.when(
            is_england & (F.month('match_start') >= 7),
            F.year('match_start'))
        .when(
            is_england,
            F.year('match_start') - 1)
        .otherwise(
            F.year('match_start'))  # Sweden: season = the calendar year itself
    )

    return (
        df
        .withColumn(
            'season_key',
            F.when(
                is_england,
                F.concat(F.col('season_start_year').cast('string'), F.lit('/'), (F.col('season_start_year') + 1).cast('string')))
            .otherwise(F.col('season_start_year').cast('string'))
        )
        .withColumn(
            'start_date',
            F.when(is_england, F.make_date(F.col('season_start_year'), F.lit(7), F.lit(1)))
            .otherwise(F.make_date(F.col('season_start_year'), F.lit(1), F.lit(1)))
        )
        .withColumn(
            'end_date',
            F.when(is_england, F.make_date(F.col('season_start_year') + 1, F.lit(6), F.lit(30)))
            .otherwise(F.make_date(F.col('season_start_year'), F.lit(12), F.lit(31)))
        )
        .withColumn(
            'is_current',
            F.current_date().between(F.col('start_date'), F.col('end_date'))
        )
        .drop('match_start', 'country', 'season_start_year')
        .dropDuplicates(['season_key'])
    )
