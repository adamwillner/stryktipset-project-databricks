# 🥉🥈🥇 Stryktipset — Databricks Edition

> PySpark + Delta Lake rebuild of the Stryktipset prediction pipeline, running natively on Databricks Free Edition (Unity Catalog + Jobs).

## Pipeline

```mermaid
flowchart LR
    A["🥉 Bronze<br/>raw JSON<br/>Volume"] --> B["🥈 Silver<br/>PySpark<br/>matches table"]
    B --> C["🥇 Gold<br/>star schema"]
```

## Gold layer — star schema

`fact_match` is scoped to English and Swedish domestic leagues only (see project-context.md) — everything else in the raw data (cups, European competitions, national teams, other countries) stays in silver but never reaches gold.

```mermaid
erDiagram
    DIM_TEAM ||--o{ FACT_MATCH : "home_team"
    DIM_TEAM ||--o{ FACT_MATCH : "away_team"
    DIM_DATE ||--o{ FACT_MATCH : "match_date"
    DIM_LEAGUE ||--o{ FACT_MATCH : "league"
    DIM_SEASON ||--o{ FACT_MATCH : "season"

    DIM_PLAYER ||--o{ FACT_PLAYER_SEASON : "player"
    DIM_TEAM ||--o{ FACT_PLAYER_SEASON : "team"
    DIM_SEASON ||--o{ FACT_PLAYER_SEASON : "season"

    DIM_TEAM {
        string team_key PK
        string team_name
    }
    DIM_DATE {
        int date_key PK
        date full_date
        int year
        int quarter
        int month
        string month_name
        int day
        int day_of_week
        string day_name
        int week_of_year
        boolean is_weekend
    }
    DIM_LEAGUE {
        string league_key PK
        string league_name
        string country
    }
    DIM_SEASON {
        string season_key PK
        date start_date
        date end_date
        boolean is_current
    }
    DIM_PLAYER {
        string player_key PK
        string name
        date birthdate
        string nationality
        string position
    }
    FACT_MATCH {
        string match_key PK
        string home_team_key FK
        string away_team_key FK
        int date_key FK
        string league_key FK
        string season_key FK
        int home_goals
        int away_goals
        float elo_home
        float elo_away
        float odds_home_draw_away
        float calibrated_prob
        float xg_home_away
    }
    FACT_PLAYER_SEASON {
        string player_key FK
        string team_key FK
        string season_key FK
        int appearances
        int goals
        int assists
        int minutes
        int cards
    }
```

## Notebooks

| # | Notebook | Layer | Purpose | Status |
|---|---|---|---|---|
| 00 | `setup_catalog_schemas` | — | one-time: catalog, schemas, volume | ✅ |
| 01 | `bronze_ingest` | 🥉 | fetch new draws from Svenska Spel's API | ✅ |
| 02 | `silver_transform` | 🥈 | flatten & clean into `matches` | ✅ |
| 03 | `gold_dimensions` | 🥇 | build `dim_team` / `dim_date` / `dim_league` / `dim_season` | ✅ |
| 04 | `gold_fact_match` | 🥇 | Elo, form, rest days, odds → `fact_match` | ⬜ |
| 05 | `gold_add_calibration` | 🥇 | isotonic fit, MLflow log, merged into `fact_match` | ✅ |
| 06 | `gold_fact_player_season` | 🥇 | player stats (stretch goal) | ⬜ |

`00`–`05` chain into one Databricks Job; `06` waits until player-stats sourcing is worked out.

## Stack

Databricks Free Edition · PySpark · Delta Lake · Unity Catalog · Databricks Jobs · MLflow

## Why this exists

Replaces the local Docker + Airflow version of this pipeline — same domain, same medallion shape, rebuilt cloud-native: Delta tables instead of local files, Databricks Jobs instead of Airflow, a proper star schema instead of a flat CSV.
