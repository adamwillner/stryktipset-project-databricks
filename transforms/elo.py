"""Pure transform functions used by 06_gold_elo."""

from pyspark.sql import DataFrame, functions as F

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
    """One row per played match: match_id plus both teams' ratings *as
    they stood before kick-off*.

    `matches` must already be scoped to the gold leagues -- 06 inner-joins
    dim_league first, the same way build_fact_match does, so the
    England/Sweden league list stays in dimensions.py and isn't
    duplicated here.

    Pre-match, not post-match. The stored rating is what was knowable on
    the morning of the match; storing the rating after the result has
    been applied would leak the outcome into the feature, and a model
    trained on it would look excellent and predict nothing.

    Unplayed fixtures are dropped on home_goals/away_goals rather than
    on `result`. Silver nulls `result` for them now, so either would
    work, but the goal columns are the direct evidence that a match was
    played and don't depend on a derived column staying correct.

    Sequential by nature -- each match depends on both teams' entire
    history in order -- so this is a plain loop over a pandas frame
    rather than a column expression. All tiers are rated in one pool,
    which is what keeps divisions on a comparable scale: a relegated team
    carries its rating down with it.
    """
    played = matches.filter(
        F.col('home_goals').isNotNull() & F.col('away_goals').isNotNull()
    )

    pdf = (
        played
        .select('match_id', 'match_start', 'home_team', 'away_team', 'result')
        .toPandas()
        .sort_values(['match_start', 'match_id'], kind='mergesort')
    )

    ratings: dict[str, float] = {}
    rows = []

    for match in pdf.itertuples(index=False):
        home_before = ratings.get(match.home_team, INITIAL_RATING)
        away_before = ratings.get(match.away_team, INITIAL_RATING)

        # recorded BEFORE the update -- this is the whole point
        rows.append((match.match_id, home_before, away_before))

        change = K_FACTOR * (
            ACTUAL_SCORE[match.result] - expected_home_score(home_before, away_before)
        )

        ratings[match.home_team] = home_before + change
        ratings[match.away_team] = away_before - change

    return matches.sparkSession.createDataFrame(rows, ELO_SCHEMA)
