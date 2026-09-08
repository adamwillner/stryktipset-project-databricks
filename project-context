# Stryktipset — Databricks Rebuild: Project Context

## What this is
A PySpark + Delta Lake rebuild of an existing local Stryktipset (Swedish football pool) pipeline, running natively on Databricks Free Edition. This is a new, separate build — not a port of the old repo's files.

## Original project (reference only, not reused)
`stryktipset-projekt`: local Python pipeline, medallion architecture (bronze/silver/gold), Docker + Airflow, sklearn isotonic-regression calibration. Left untouched as its own working project.

## Architecture decisions
- Unity Catalog: catalog `stryktipset`, schemas `bronze` / `silver` / `gold`
- Bronze: plain Python (`requests`) ingestion from Svenska Spel's API, raw JSON landed as files in a Volume (`stryktipset.bronze.raw_draws`) — deliberately kept as inspectable files, not a Delta table
- Silver: PySpark — flattens/cleans bronze JSON into a `matches` Delta table
- Gold: PySpark — a proper star schema, replacing the original's flat CSV
- Calibrated probability is a column on `fact_match`, not a separate table (decided explicitly)
- Compute: Free Edition is serverless-only, capped at 5 concurrent job tasks account-wide — irrelevant here since the pipeline runs as one linear chain
- Scheduling: one Databricks Job chaining notebooks `00`–`05`, cron set on the Job itself — replaces Airflow entirely for this version

## Star schema (gold layer)
Two fact tables sharing dimensions:
- `fact_match` — one row per match. Dimensions: `dim_team` (role-playing: home + away), `dim_date`, `dim_league`, `dim_season`. Measures: goals, Elo (home/away), odds, `calibrated_prob`, xG (nullable — only some leagues have it).
- `fact_player_season` — one row per player per season (stretch goal, notebook `06`). Dimensions: `dim_player`, `dim_team`, `dim_season`. Measures: appearances, goals, assists, minutes, cards.
- `dim_player` holds only slowly-changing bio fields (name, birthdate, nationality, position) — stats belong in the fact table, not the dimension.

## Notebooks (chain into one Databricks Job)
| # | Notebook | Purpose | Status |
|---|---|---|---|
| 00 | setup_catalog_schemas | one-time SQL: catalog, schemas, volume | done |
| 01 | bronze_ingest | fetch new draws from Svenska Spel's API | done |
| 02 | silver_transform | PySpark flatten into `matches` | not started |
| 03 | gold_dimensions | build dim_team/date/league/season | not started |
| 04 | gold_fact_match | Elo, form, rest days, odds → fact_match | not started |
| 05 | gold_add_calibration | isotonic fit, MLflow log, merge into fact_match | not started |
| 06 | gold_fact_player_season | player stats (stretch) | not started |

## Data sourcing notes
- Team-level: football-data.org — free, no meaningful cap, fixtures/results/standings.
- Player/team stats: API-Football via RapidAPI — free tier capped at 100 requests/day + 10/minute, resets 00:00 UTC. Budget per team, not per player; only need the ~26 teams on a given week's coupon.
- xG: no free source covers Stryktipset's actual league mix (Understat only covers the Big 5). A custom xG model trained on StatsBomb's free open shot data is viable as a standalone exercise, not for live scoring of most matches on the coupon.

## Gotchas already worked out
- `spark.sql()` runs exactly one statement per call — no `;`-separated multi-statement strings like a SQL notebook cell allows.
- Unity Catalog Volumes support plain Python file I/O directly (`pathlib.Path`, `os`) — `dbutils.fs` isn't required.
- Notebooks don't need `if __name__ == "__main__":` — Databricks doesn't execute files like `python script.py` does, and the numbered filenames (`01_...`) can't be imported as modules anyway. Just call `main()` directly.
- A failure inside a loop (e.g. one draw that didn't fetch) won't fail the notebook/Job unless you explicitly `raise` — otherwise the run reports Success even when something silently broke. Job failure notifications (email/Slack) are a separate setup step on top of the `raise`.

## How Adam wants to work
- Writes the code himself — wants review and explanation, not finished files handed over.
- Prefers short, visual docs; dislikes redundant status markers.
- Wants diagrams only when genuinely load-bearing — plain text/tables are often enough.
