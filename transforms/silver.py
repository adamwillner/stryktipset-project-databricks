"""Pure transform functions used by 02_silver_transform.

Kept separate from the notebook so they're importable by pytest without
a Databricks connection -- neither function touches spark.table(...) or
any catalog/table name, they only take/return Columns.
"""

from pyspark.sql import functions as F
from pyspark.sql import Column


def parse_swedish_decimal(value: Column) -> Column:
    return F.translate(value, ",", ".").cast("double")


def favourite_odds(value: Column) -> Column:
    """Svenska Spel's own implied probability, with the bookmaker's margin
    already removed -- the raw odds imply ~103%, these sum to exactly 100.

    Divided by 100 so it lands on the same 0-1 scale as streck_*, and cast
    through string first because the field is quoted in some draws and
    bare in others; translating ',' to '.' is a no-op on a dotted value.
    """
    return parse_swedish_decimal(value.cast("string")) / 100


def extract_match_row(draw_number: Column, event: Column) -> list:

    home_goals = event["match"]["participants"][0]["result"].try_cast("int")
    away_goals = event["match"]["participants"][1]["result"].try_cast("int")

    # providerIds is an array and BetRadar isn't always first -- and it's
    # empty altogether in the oldest draws, so F.get (which returns null
    # out of range) rather than indexing, which can raise under ANSI mode.
    betradar = F.filter(
        event["providerIds"], lambda provider: provider["provider"] == "BetRadar"
    )

    # Three explicit branches and deliberately no .otherwise(): when a match
    # hasn't been played both goal columns are null, every comparison below is
    # null rather than true, and `result` comes back null. An .otherwise('2')
    # here would silently label every unplayed fixture an away win.
    result = (
        F.when(home_goals > away_goals, '1')
        .when(home_goals == away_goals, 'X')
        .when(home_goals < away_goals, '2')
    )

    return [
        draw_number.alias('draw_number'),
        event["eventNumber"].alias('event_number'),
        event["match"]["matchId"].alias('match_id'),
        event["match"]["matchStart"].alias('match_start'),
        event["match"]["participants"][0]["name"].alias('home_team'),
        event["match"]["participants"][1]["name"].alias('away_team'),
        event["match"]["league"]["name"].alias('league'),
        (parse_swedish_decimal(event["svenskaFolket"]["one"]) / 100).alias('streck_1'),
        (parse_swedish_decimal(event["svenskaFolket"]["x"]) / 100).alias('streck_x'),
        (parse_swedish_decimal(event["svenskaFolket"]["two"]) / 100).alias('streck_2'),
        home_goals.alias('home_goals'),
        away_goals.alias('away_goals'),
        result.alias('result'),
        parse_swedish_decimal(event["odds"]["one"]).alias('odds_1'),
        parse_swedish_decimal(event["odds"]["x"]).alias('odds_x'),
        parse_swedish_decimal(event["odds"]["two"]).alias('odds_2'),
        parse_swedish_decimal(event["startOdds"]["one"]).alias('start_odds_1'),
        parse_swedish_decimal(event["startOdds"]["x"]).alias('start_odds_x'),
        parse_swedish_decimal(event["startOdds"]["two"]).alias('start_odds_2'),

        # --- fields carried since 2026-09-24; see project-context for coverage ---
        event["match"]["participants"][0]["id"].try_cast('int').alias('home_team_id'),
        event["match"]["participants"][1]["id"].try_cast('int').alias('away_team_id'),
        event["match"]["participants"][0]["countryName"].alias('home_team_country'),
        event["match"]["participants"][1]["countryName"].alias('away_team_country'),
        event["match"]["league"]["id"].try_cast('int').alias('league_id'),
        event["match"]["league"]["country"]["name"].alias('league_country'),
        event["match"]["sportEventStatus"].alias('status'),
        favourite_odds(event["favouriteOdds"]["one"]).alias('favourite_odds_1'),
        favourite_odds(event["favouriteOdds"]["x"]).alias('favourite_odds_x'),
        favourite_odds(event["favouriteOdds"]["two"]).alias('favourite_odds_2'),
        F.get(betradar, F.lit(0))["id"].alias('betradar_id'),
    ]
