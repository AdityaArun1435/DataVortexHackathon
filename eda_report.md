# Data Vortex — Round 1 Phase 1: EDA Report

## 1. Dataset Overview
- **Posts**: 12,360 raw rows → 12,000 after dropping 360 exact duplicates.
- **Users**: 1,500 rows, verified clean on load (no nulls, no duplicates, referential integrity with posts holds — every `user_id` in posts exists in users).

## 2. Cleaning Summary
| Issue | Column | Count | Fix |
|---|---|---|---|
| Exact duplicate rows | post_id | 360 | Dropped, kept first occurrence |
| Mixed timestamp formats (epoch / DD-MM-YYYY / ISO8601) | timestamp | 12,360 | Parsed to single datetime dtype |
| Missing platform | platform | 1,784 | Filled `"Unknown"` (no basis to infer) |
| Literal `"NULL"` strings | text_content | 23 | Converted to true NaN |
| HTML-escaped text | text_content | 328 | Unescaped (`&amp;` → `&`, etc.) |
| Genuinely missing text | text_content | 1,711 | Left as NaN, not fabricated |
| Negative likes (impossible) | likes | 509 | Converted to NaN before imputation |
| Missing/negative likes | likes | 2,323 | Imputed with platform-median, flagged in `likes_imputed` |

Full transformation log: `cleaning_log.txt`.

## 3. Key Findings

**Platform distribution** is roughly even across the 5 real platforms (~2,000–2,074 posts each), with 1,784 posts of unknown platform. No single platform dominates the corrupted dataset, suggesting the corruption was applied close to uniformly at random rather than targeting one platform.

**Posting volume over time** is stable, ~975–1,040 posts/month from May 2024 to April 2025, no visible growth or decay trend and no seasonal spike. See `eda_charts.png`.

**Engagement metrics show no meaningful correlation with each other or with follower count** (|r| ≤ 0.02 across likes, shares, comments, follower_count). This indicates the engagement numbers in this dataset are synthetic/randomly generated rather than reflecting real social dynamics — a more popular account (higher followers) does not get more likes here. This is an important caveat for Phase 2: any SQL "correlation analysis" task on this data will correctly find near-zero correlation, and that's the right answer, not a bug in the query.

**User base** spans 10 languages fairly evenly (1,119–1,369 users each) and is geographically diverse — no single city dominates (top locations sit at 420–460 posts each out of 12,000).

## 4. Files
- `cleaned_posts.csv` — 12,000 rows, 9 columns (adds `likes_imputed` flag)
- `cleaned_users.csv` — 1,500 rows, unchanged from source (verified clean)
- `cleaning_log.txt` — plain-text log of every transformation applied
- `eda_charts.png` — platform volume + posts-per-month charts
- `clean_pipeline.py` — full reproducible cleaning script
