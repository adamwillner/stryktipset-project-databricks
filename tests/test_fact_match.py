"""Tests for transforms/fact_match.py (build_fact_match).

TODO: write tests here. Some starting points:
- a league not present in dim_league drops out (inner join) -- confirm a
  fake row with an unknown league name doesn't appear in the result.
- season_key matches build_dim_season's logic for the same match_start +
  country (worth asserting they agree, since the logic is duplicated).
- calibrated_1/calibrated_x/calibrated_2 come back as null doubles, not
  missing columns.
"""

from transforms.fact_match import build_fact_match
