from transforms.elo import (
    HOME_ADVANTAGE,
    INITIAL_RATING,
    build_elo,
    expected_home_score,
)

MATCHES_SCHEMA = (
    'match_id string, match_start string, home_team string, away_team string, '
    'home_goals int, away_goals int, result string'
)

def test_expected_home_score():
  # equal ratings still favour the home side -- that's HOME_ADVANTAGE
  assert expected_home_score(1500.0, 1500.0) > 0.5

  # give the away team exactly the home bonus and it's a coin flip again
  assert abs(expected_home_score(1500.0, 1500.0 + HOME_ADVANTAGE) - 0.5) < 1e-9

  # a 400-point lead is ~91% before home advantage, ~94% with it
  assert 0.93 < expected_home_score(1900.0, 1500.0) < 0.94

  # a 400-point deficit at home is ~13% -- home advantage softens it
  # (it would be ~9% on neutral ground) but nowhere near cancels it
  assert 0.12 < expected_home_score(1100.0, 1500.0) < 0.13

def test_build_elo(spark):
  matches = spark.createDataFrame(
      [
          # A beats B at home
          ("m1", "2024-01-01T15:00:00", "A", "B", 2, 0, "1"),
          # A draws with C -- A's rating here reflects m1 only
          ("m2", "2024-01-08T15:00:00", "A", "C", 1, 1, "X"),
          # B hosts A -- B's rating here reflects m1 only
          ("m3", "2024-01-15T15:00:00", "B", "A", 0, 1, "2"),
          # not played yet: silver labels this '2' even though nothing happened
          ("m4", "2024-02-01T15:00:00", "D", "E", None, None, "2"),
      ],
      MATCHES_SCHEMA,
  )

  result = build_elo(matches)
  rows = {row["match_id"]: row for row in result.collect()}

  # the unplayed fixture must not reach the ratings
  assert "m4" not in rows
  assert result.count() == 3

  # every team starts level, and m1 is both teams' first match
  assert rows["m1"]["elo_home"] == INITIAL_RATING
  assert rows["m1"]["elo_away"] == INITIAL_RATING

  # A won m1, so A is above starting by the time it plays m2
  assert rows["m2"]["elo_home"] > INITIAL_RATING
  # B lost m1, so B is below starting by the time it plays m3
  assert rows["m3"]["elo_home"] < INITIAL_RATING

  # Elo is zero-sum: what A gained in m1, B lost
  a_gain = rows["m2"]["elo_home"] - INITIAL_RATING
  b_loss = INITIAL_RATING - rows["m3"]["elo_home"]
  assert abs(a_gain - b_loss) < 1e-9

  # C has not played before m2, so it is still exactly at the start
  assert rows["m2"]["elo_away"] == INITIAL_RATING
