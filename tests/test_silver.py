"""Tests for transforms/silver.py (parse_swedish_decimal, extract_match_row).

TODO: write tests here. Some starting points:
- parse_swedish_decimal: "1,50" -> 1.5, does it handle already-dotted
  strings / nulls?
- extract_match_row: needs a small fake `event` struct (built with
  F.struct/F.lit or spark.createDataFrame + to_json/from_json) to run
  against -- check the derived `result` column ('1'/'X'/'2') lands right
  for a home win / draw / away win.
"""

from transforms.silver import parse_swedish_decimal, extract_match_row
