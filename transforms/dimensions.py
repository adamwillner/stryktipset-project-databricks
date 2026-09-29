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

from pyspark.sql import Column, DataFrame, Window, functions as F

# Gold is scoped to English domestic football. Sweden was dropped on
# 2026-09-29 -- deliberate, and it cost build_dim_season its second branch.
LEAGUE_COUNTRY = {
    "Premier League": "England",
    "Championship": "England",
    "League One": "England",
    "League Two": "England",
    "National League": "England",
}

def date_key(match_start: Column) -> Column:
    """The single place date_key is derived.

    A smart key (yyyyMMdd) rather than a hash, deliberately: date is the
    one dimension where a readable key is the convention, because it makes
    partitioning and range filters obvious.
    """
    return F.date_format(F.to_date(match_start), 'yyyyMMdd').cast('int')



def build_dim_team(df: DataFrame) -> DataFrame:
    """One row per team: team_key, team_name, team_id, country.

    team_key is the uppercased team_name, and it is what fact_match joins
    on. `team_id` is Svenska Spel's own id, carried as an attribute rather
    than a key: it was renumbered between 2021 and 2023, so the same team
    has two ids across the history (Newcastle is both 88 and 1000041) and
    keying on it would split every long-lived team in two. The name is the
    stable identifier in this data -- no team has been renamed in thirteen
    years.

    Deliberately NOT scoped to English teams even though gold is: a coupon
    regularly carries Norwegian, Scottish or Spanish fixtures and
    mart.coupon needs a real team key for them. A dimension holding rows
    no fact references is normal.

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
    """df: two columns, league_name/country -- the LEAGUE_COUNTRY dict as a
    DataFrame.

    Columns: league_key, league_name, country.

    No Unknown member. One was added on 2026-09-29 so mart.coupon could
    give out-of-scope leagues a real key rather than a null one, and
    removed the same day once the mart dropped its surrogate keys -- with
    no fact referencing it, it was machinery with no consumer. Add it back
    if a fact ever needs a non-null league for something outside gold.
    """
    return (
        df.select('league_name', 'country')
        .withColumn('league_key', F.upper('league_name'))
        .dropDuplicates(['league_key'])
        .select('league_key', 'league_name', 'country')
    )


def build_dim_season(df: DataFrame) -> DataFrame:
    """One row per English season. Seasons run Aug-May, so a July cutoff
    decides which year pair a match belongs to, and the key looks like
    "2024/2025".

    This used to branch per country, because Swedish seasons sit inside a
    single calendar year and English ones do not. Sweden left gold on
    2026-09-29 and the branch went with it, so `country` is no longer
    required on the input. Re-adding any country means restoring a
    per-country rule here -- seasons genuinely do not share a shape across
    countries.

    Columns: season_key, start_date, end_date, is_current.
    """
    df = df.select(F.to_date('match_start').alias('match_start'))

    df = df.withColumn(
        'season_start_year',
        F.when(F.month('match_start') >= 7, F.year('match_start'))
        .otherwise(F.year('match_start') - 1),
    )

    return (
        df
        .withColumn(
            'season_key',
            F.concat(
                F.col('season_start_year').cast('string'),
                F.lit('/'),
                (F.col('season_start_year') + 1).cast('string'),
            ),
        )
        .withColumn('start_date', F.make_date(F.col('season_start_year'), F.lit(7), F.lit(1)))
        .withColumn('end_date', F.make_date(F.col('season_start_year') + 1, F.lit(6), F.lit(30)))
        .withColumn(
            'is_current',
            F.current_date().between(F.col('start_date'), F.col('end_date')),
        )
        .drop('match_start', 'season_start_year')
        .dropDuplicates(['season_key'])
    )
