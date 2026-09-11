# Stryktipset — Databricks Rebuild: Project Context

## What this is
A PySpark + Delta Lake rebuild of an existing local Stryktipset (Swedish football pool) pipeline, running natively on Databricks Free Edition. This is a new, separate build — not a port of the old repo's files.

## Original project (reference only, not reused)
`stryktipset-projekt`: local Python pipeline, medallion architecture (bronze/silver/gold), Docker + Airflow, sklearn isotonic-regression calibration. Left untouched as its own working project.

## Architecture decisions
- Unity Catalog: catalog `stryktipset`, schemas `bronze` / `silver` / `gold`
- Bronze: plain Python (`requests`) ingestion from Svenska Spel's API, raw JSON landed as files in a Volume (`stryktipset.bronze.raw_draws`) — deliberately kept as inspectable files, not a Delta table
- Silver: PySpark — flattens/cleans bronze JSON into a `matches` Delta table. Stays complete and unfiltered (every league, every country, cups, internationals, national teams) — this is what lets calibration (`05`) train on the full historical dataset even though gold is scoped narrower. One row per `match_id` is enforced here (see gotchas below).
- Gold: PySpark — a proper star schema, replacing the original's flat CSV
- Calibrated probability is a column on `fact_match`, not a separate table (decided explicitly)
- **Scope: `fact_match` only covers English and Swedish domestic leagues.** `dim_league` is built from a curated `LEAGUE_COUNTRY` name→country lookup (only the leagues you explicitly list); `04_gold_fact_match` filters `matches` down to that same set before building fact rows. Everything else — cups, Champions League, national teams, other countries — is simply excluded, not nulled out. Silver is untouched by this; only gold is scoped.
- `dim_league.country` comes from the curated lookup, not derived from team data — an earlier idea (derive league country from the countries of teams that played in it) was considered and dropped as unnecessary complexity once the direct lookup proved simpler.
- `dim_team` deliberately has **no** `league` or `country` column. League isn't a stable team attribute — teams get promoted/relegated between tiers, so it's a property of a match (`fact_match.league_key`), not the team. Country was considered but dropped along with the league-derivation idea above. `dim_team` is just `team_key` (uppercased `team_name`, for case-safety) and `team_name` — thin by design, kept mainly as a clean deduplicated team list and a place to add things like external team IDs (football-data.org/API-Football) later, if that integration happens.
- `dim_date` is a full calendar — every calendar day, not just dates that had matches — spanning from the earliest match in the data to two years past today, so future draws are already covered. `day_of_week` uses the ISO convention (Monday=1..Sunday=7), computed via `F.dayofweek()` (Spark's Sunday-first result) remapped with a `when/otherwise`, not `date_format(..., "u")` — that pattern letter isn't recognized by Spark 3.0+'s `DateTimeFormatter` and throws `INCONSISTENT_BEHAVIOR_CROSS_VERSION_DATETIME_PATTERN_RECOGNITION`. No `matchday` column — that's `fact_match.draw_number`, not a calendar attribute.
- `dim_season` covers both England and Sweden, but the two countries' seasons don't share a shape, so `season_key`/`start_date`/`end_date` are computed differently per country rather than with one global rule: England seasons span two calendar years (Aug–May, July cutoff decides the year pair, key like `"2024/2025"`); Sweden seasons sit inside one calendar year (key is just `"2024"`, Jan 1–Dec 31). `build_dim_season` requires its input already joined against `LEAGUE_COUNTRY` (adds a `country` column) and filtered to non-null `country` first — matches from leagues outside the England/Sweden scope must not reach this function, or they'd fall into the Sweden-shaped branch by default. `season_label` was dropped as redundant with `season_key`; `start_date`/`end_date`/`is_current` were added instead, since those carry actual information `season_key` alone doesn't.
- **`04_gold_fact_match` design:** joins `matches` to the already-built `dim_league` table (not a re-imported `LEAGUE_COUNTRY` dict — avoids having the England/Sweden league list live in two places) via `matches['league'] == dim_league['league_name']`; that inner join both scopes to England/Sweden and supplies `league_key`/`country`. `team_key`/`date_key` are derived directly from existing columns (pure functions — `upper(team_name)`, formatted `match_start`), no join needed for those. `season_key` reuses the same per-country logic as `build_dim_season`, computed inline (not by calling `build_dim_season` itself, since that function collapses to one row per season — here every match row needs to keep its own `season_key`).
- **`fact_match`'s real primary key is `match_key`, copied directly from silver's `match_id`** (`withColumn('match_key', F.col('match_id'))`) — resolves the gap flagged when `05` was first written. `match_id` itself is *not* carried onto `fact_match` (not in `04`'s final `.select(...)`), so anything joining against `fact_match` on the match grain must use `match_key`.
- `fact_match` carries odds and calibration as separate per-outcome columns (`odds_1`/`odds_x`/`odds_2`, `start_odds_1`/`x`/`2`, `calibrated_1`/`x`/`2`), not single combined fields — matches the shape `streck_1`/`streck_x`/`streck_2` already uses in silver.
- **`calibrated_1`/`calibrated_x`/`calibrated_2` are created as null placeholder columns by `04` itself**, not left absent for `05` to add. Reason: Delta's `MERGE` schema evolution (`WITH SCHEMA EVOLUTION` or the classic autoMerge config) only auto-adds new columns for the wildcard forms (`UPDATE SET *` / `INSERT *`) — an explicit assignment like `target.calibrated_1 = source.calibrated_1` fails with `DELTA_MERGE_UNRESOLVED_EXPRESSION` if that column doesn't already exist on target, since Spark tries to resolve it before any evolution logic runs. Pre-creating the columns as null in `04` sidesteps this; `05`'s merge stays a plain, explicit-column `MERGE INTO` (schema evolution not actually needed once the columns already exist).
- Elo and xG will follow the same pattern as calibration eventually (computed elsewhere, merged onto existing `fact_match` rows) — but neither is built yet. Elo specifically needs its own dedicated notebook/pass: it's sequential (each match's rating depends on both teams' full match history in order), not a plain column expression, so it doesn't fit into `04`'s per-row transformations.
- Compute: Free Edition is serverless-only, capped at 5 concurrent job tasks account-wide — irrelevant here since the pipeline runs as one linear chain
- Scheduling: one Databricks Job chaining notebooks `00`–`05`, cron set on the Job itself — replaces Airflow entirely for this version

## Star schema (gold layer)
Two fact tables sharing dimensions:
- `fact_match` — one row per match, **English/Swedish domestic leagues only** (see scope decision above). Dimensions: `dim_team` (role-playing: home + away), `dim_date`, `dim_league`, `dim_season`. Measures: goals, odds (`odds_1/x/2`, `start_odds_1/x/2`), `calibrated_1/x/2`, Elo (home/away, not built), xG (nullable, not sourced).
- `fact_player_season` — one row per player per season (stretch goal, notebook `06`). Dimensions: `dim_player`, `dim_team`, `dim_season`. Measures: appearances, goals, assists, minutes, cards.
- `dim_player` holds only slowly-changing bio fields (name, birthdate, nationality, position) — stats belong in the fact table, not the dimension.
- Possible future addition (not started): a second small fact table, `fact_league_season` (one row per league+season, holding aggregated `match_count`/`total_goals`), sitting alongside `fact_match` and sharing `dim_league`/`dim_season` — a fact constellation, not a change to either dimension. Considered if per-league season stats become useful; `dim_league`/`dim_season` themselves would stay flat, unchanged.

## Notebooks (chain into one Databricks Job)
| # | Notebook | Purpose | Status |
|---|---|---|---|
| 00 | setup_catalog_schemas | one-time SQL: catalog, schemas, volume | done |
| 01 | bronze_ingest | fetch new draws from Svenska Spel's API | done |
| 02 | silver_transform | PySpark flatten into `matches` | done |
| 03 | gold_dimensions | build dim_team/date/league/season | done |
| 04 | gold_fact_match | join dims, derive keys, goals & odds → fact_match | done |
| 05 | gold_add_calibration | isotonic fit, MLflow log, merge into fact_match | done |
| 06 | gold_fact_player_season | player stats (stretch) | not started |

## Known gaps
- No automated tests yet anywhere in the pipeline — flagged as a near-term thing to add, not started.
- Elo not built — needs its own dedicated notebook/pass (see architecture decisions above), no notebook number assigned yet.
- xG has no confirmed free data source covering these leagues (see Data sourcing notes) — column exists on `fact_match`, always null for now.

## Data sourcing notes
- Team-level: football-data.org — free, no meaningful cap, fixtures/results/standings.
- Player/team stats: API-Football via RapidAPI — free tier capped at 100 requests/day + 10/minute, resets 00:00 UTC. Budget per team, not per player; only need the ~26 teams on a given week's coupon.
- xG: no free source covers Stryktipset's actual league mix (Understat only covers the Big 5). A custom xG model trained on StatsBomb's free open shot data is viable as a standalone exercise, not for live scoring of most matches on the coupon.

## Gotchas already worked out
- `spark.sql()` runs exactly one statement per call — no `;`-separated multi-statement strings like a SQL notebook cell allows.
- Unity Catalog Volumes support plain Python file I/O directly (`pathlib.Path`, `os`) — `dbutils.fs` isn't required.
- Notebooks don't need `if __name__ == "__main__":` — Databricks doesn't execute files like `python script.py` does, and the numbered filenames (`01_...`) can't be imported as modules anyway. Just call `main()` directly.
- A failure inside a loop (e.g. one draw that didn't fetch) won't fail the notebook/Job unless you explicitly `raise` — otherwise the run reports Success even when something silently broke. Job failure notifications (email/Slack) are a separate setup step on top of the `raise`.
- The `league` column from Svenska Spel's API is much messier than it looks — around 140 distinct values, including casing duplicates (`Damallsvenskan`/`DamAllsvenskan`), spacing duplicates (`La Liga`/`LaLiga`), and inconsistent comma usage for competition groups/qualifiers (some use `"X, Grupp A"`, others `"X Grupp A"` with no comma at all). Don't trust a quick `.show()` of distinct leagues — it truncates by default and hides most of this; use `.collect()` + plain `print()` instead.
- `spark.createDataFrame(some_list_of_tuples, [...])` can fail with `CANNOT_INFER_TYPE_FOR_FIELD` if Spark can't pin down a column's type from the first row (e.g. a `None` value). Passing an explicit schema string (`'col_name string, other_col string'`) instead of a bare column-name list sidesteps the inference entirely.
- `dict.items()` on its own is a single dict-items object, not a list of rows — wrapping it as `[some_dict.items()]` makes a one-row DataFrame with one giant field, not one row per key. `list(some_dict.items())` is what actually gives one `(key, value)` tuple per row.
- `matches.match_id` is not guaranteed unique in silver: a handful of real matches (COVID-era Championship fixtures rescheduled in the June 2020 catch-up period) legitimately appear on two different `draw_number`s with the identical `match_id`/`match_start`. `02_silver_transform` dedupes deterministically with a window function (`row_number()` partitioned by `match_id`, ordered by `draw_number` descending, keep rank 1) rather than plain `dropDuplicates`, which would pick an arbitrary (not necessarily newest) row.
- **Databricks Free Edition serverless runs on Spark Connect, which doesn't support several classic-Spark/session-level APIs:**
  - `df.sparkContext` / `sc.broadcast(...)` → `ATTRIBUTE_NOT_SUPPORTED`. Fix: don't broadcast manually — a `pandas_udf` can just close over a Python object (e.g. a fitted sklearn model) directly; Spark Connect ships the closure to the workers itself.
  - `.cache()` / `.persist()` / `.unpersist()` → `NOT_SUPPORTED_WITH_SERVERLESS` ("PERSIST TABLE is not supported"). Fix: just don't cache — for small-to-medium data, recomputing a DataFrame a second time is a fine trade-off.
  - `spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")` → `CONFIG_NOT_AVAILABLE`. Fix: use `MERGE WITH SCHEMA EVOLUTION INTO ...` SQL syntax on the `MERGE` statement itself instead of the session-level config — though note this only evolves schema for `UPDATE SET *`/`INSERT *`, not named-column assignments (see the `calibrated_1/x/2` placeholder-column decision above).
  - Plain `.write.mode('overwrite')` on an existing Delta table enforces the existing schema by default — add `.option('overwriteSchema', 'true')` when the new DataFrame's columns differ from what's already there. Fine to leave on permanently for a table that's genuinely fully rebuilt every run (like `fact_match`, the `dim_*` tables); avoid it for anything incrementally merged/updated, where schema drift should still error loudly.

## How Adam wants to work
- Writes the code himself — wants review and explanation, not finished files handed over.
- Prefers short, visual docs; dislikes redundant status markers.
- Wants diagrams only when genuinely load-bearing — plain text/tables are often enough.
