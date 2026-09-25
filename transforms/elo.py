"""Pure transform functions used by 06_gold_elo."""

from pyspark.sql import DataFrame

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 65.0

# what the home team actually scored, in Elo terms
ACTUAL_SCORE = {'1': 1.0, 'X': 0.5, '2': 0.0}

ELO_SCHEMA = 'match_id string, elo_home double, elo_away double'


def expected_home_score(home_rating: float, away_rating: float) -> float:
    """The score the home team is expected to take (1 = win, 0.5 = draw),
    derived from the rating gap alone.

    HOME_ADVANTAGE is added to the home rating before comparing -- a team
    at home plays stronger than its number, and folding that in here
    keeps the rating itself venue-independent.
    """
    gap = away_rating - (home_rating + HOME_ADVANTAGE)
    return 1.0 / (1.0 + 10 ** (gap / 400.0))


def build_elo(matches: DataFrame) -> DataFrame:
    """One row per match: match_id plus both teams' ratings as they stood
    before kick-off.

    Every match in the input feeds the ratings -- all tiers, all countries,
    one pool. That is what keeps divisions comparable (a relegated team
    carries its rating down with it) and what lets a coupon containing
    Eliteserien or La Liga be scored at all.

    Played and unplayed matches are both returned. A played match records
    the pre-match ratings and then applies its result; an unplayed one
    records the ratings as they currently stand and changes nothing,
    because there is no result to learn from. That is what makes a rating
    usable as a predictor for next week's coupon, and it also handles a
    postponed fixture sitting in the middle of history.

    Pre-match, never post-match: storing the rating after the result had
    been applied would leak the outcome into the feature, and a model
    trained on it would look excellent and predict nothing.

    Sequential by nature -- each match depends on every match before it --
    so this is a plain loop over a pandas frame, not a column expression.
    """
    pdf = (
        matches
        .select('match_id', 'match_start', 'home_team', 'away_team', 'result')
        .toPandas()
        .sort_values(['match_start', 'match_id'], kind='mergesort')
    )

    ratings: dict[str, float] = {}
    rows = []

    for match in pdf.itertuples(index=False):
        home_before = ratings.get(match.home_team, INITIAL_RATING)
        away_before = ratings.get(match.away_team, INITIAL_RATING)

        # recorded BEFORE any update -- this is the whole point
        rows.append((match.match_id, home_before, away_before))

        if match.result not in ACTUAL_SCORE:
            continue  # not played (or postponed): rate it, learn nothing

        change = K_FACTOR * (
            ACTUAL_SCORE[match.result] - expected_home_score(home_before, away_before)
        )
        ratings[match.home_team] = home_before + change
        ratings[match.away_team] = away_before - change

    return matches.sparkSession.createDataFrame(rows, ELO_SCHEMA)
