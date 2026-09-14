"""Pure transform functions used by 02_silver_transform.

Kept separate from the notebook so they're importable by pytest without
a Databricks connection -- neither function touches spark.table(...) or
any catalog/table name, they only take/return Columns.
"""

from pyspark.sql import functions as F
from pyspark.sql import Column


def parse_swedish_decimal(value: Column) -> Column:
    return F.translate(value, ",", ".").cast("double")


def extract_match_row(draw_number: Column, event: Column) -> list:

    home_goals = event["match"]["participants"][0]["result"].try_cast("int")
    away_goals = event["match"]["participants"][1]["result"].try_cast("int")

    result = F.when(home_goals > away_goals, '1').when(home_goals == away_goals, 'X').otherwise('2')

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
        parse_swedish_decimal(event["startOdds"]["two"]).alias('start_odds_2')
    ]
