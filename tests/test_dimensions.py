"""Tests for transforms/dimensions.py.

TODO: write tests here. Some starting points:
- build_dim_team: dedupes a team appearing as both home and away into one
  row; team_key is upper(team_name).
- build_dim_date: date_key format, is_weekend on a known Saturday/Sunday,
  ISO day_of_week (1=Monday..7=Sunday).
- build_dim_league: one row per LEAGUE_COUNTRY entry, league_key is
  upper(league_name); needs `spark` fixture as the new required arg.
- build_dim_season: England season spans two years with a July cutoff,
  Sweden season is just the calendar year; is_current for a season
  containing today's date.
"""

from transforms.dimensions import (
    LEAGUE_COUNTRY,
    build_dim_team,
    build_dim_date,
    build_dim_league,
    build_dim_season,
)
