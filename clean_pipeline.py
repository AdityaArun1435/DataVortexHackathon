"""
Data Vortex Round 1 - Phase 1 cleaning pipeline

Takes the two corrupted CSVs recovered from node_07 and produces
cleaned_posts.csv, cleaned_users.csv, and a log of every change made.

Run with: python3 clean_pipeline.py
"""

import pandas as pd
import numpy as np
import html
import re

pd.set_option('display.max_columns', None)

posts = pd.read_csv('Social_Engine_Posts_Corrupted.csv')
users = pd.read_csv('Social_Engine_Users.csv')

# keeping a running log so I can explain every change later in the report
# instead of just doing stuff silently
log = []

def note(msg):
    log.append(msg)
    print(msg)

note(f"Loaded posts: {posts.shape[0]} rows, {posts.shape[1]} cols")
note(f"Loaded users: {users.shape[0]} rows, {users.shape[1]} cols")

# Checked the users table by hand first - no nulls, no duplicate user_id,
# no duplicate rows, dates are all already YYYY-MM-DD. Nothing to fix here,
# just converting account_created to an actual datetime so it's usable later.
assert users['user_id'].is_unique, "users.user_id not unique"
assert users.isna().sum().sum() == 0, "unexpected nulls in users"
users['account_created'] = pd.to_datetime(users['account_created'])
note("users table needed no cleaning, verified clean on load.")

# posts had 360 rows that were 100% identical duplicates (same post_id,
# same everything). Just keeping the first copy of each.
before = len(posts)
posts = posts.drop_duplicates(subset=['post_id'], keep='first').reset_index(drop=True)
note(f"Dropped {before - len(posts)} duplicate post_id rows (kept first occurrence).")

# timestamps came in three different formats mixed together in the same
# column: unix epoch seconds, DD-MM-YYYY, and ISO datetimes. Converting
# everything to one real datetime column.
def parse_ts(v):
    v = str(v).strip()
    if re.fullmatch(r'\d{10}', v):
        return pd.to_datetime(int(v), unit='s')
    if re.fullmatch(r'\d{2}-\d{2}-\d{4}', v):
        return pd.to_datetime(v, format='%d-%m-%Y')
    return pd.to_datetime(v, errors='coerce')

posts['timestamp'] = posts['timestamp'].apply(parse_ts)
unparsed = posts['timestamp'].isna().sum()
note(f"Normalized timestamp to one datetime format. Rows that still failed to parse: {unparsed}.")

# platform was missing on a chunk of rows. No way to guess what platform
# a post actually came from, so labeling it Unknown instead of dropping
# the row or making something up.
n_platform_null = posts['platform'].isna().sum()
posts['platform'] = posts['platform'].fillna('Unknown')
note(f"platform: filled {n_platform_null} missing values as 'Unknown'.")

# text_content had a mix of real nulls and the literal string "NULL"
# (and a few other null-ish strings), which pandas doesn't catch on its
# own since they're just text. Normalizing those to real NaN first, then
# unescaping HTML entities like &amp; so the text is actually readable.
literal_null_mask = posts['text_content'].astype(str).str.strip().str.lower().isin(
    ['null', 'nan', 'none', 'n/a', '']
)
n_literal_null = literal_null_mask.sum()
posts.loc[literal_null_mask, 'text_content'] = np.nan

n_html = posts['text_content'].dropna().astype(str).str.contains(
    '&amp;|&quot;|&#39;|&lt;|&gt;'
).sum()
posts['text_content'] = posts['text_content'].apply(
    lambda x: html.unescape(x) if isinstance(x, str) else x
)
note(f"text_content: converted {n_literal_null} literal 'NULL'-style strings to real NaN.")
note(f"text_content: unescaped HTML entities in {n_html} rows.")
note(f"text_content: {posts['text_content'].isna().sum()} rows are still genuinely missing. "
     f"Leaving them as NaN since fabricating post text isn't allowed.")

# likes had negative numbers, which can't happen in real life (min was -4987),
# so those are corruption, not real data. Setting them to NaN along with the
# genuinely missing values, then filling with the median for that platform
# rather than a single global number, since engagement differs by platform.
# The likes_imputed column flags exactly which rows got a filled-in value,
# so nothing here is silently pretending to be real data.
n_negative = (posts['likes'] < 0).sum()
posts.loc[posts['likes'] < 0, 'likes'] = np.nan
note(f"likes: {n_negative} negative values (impossible) converted to NaN before imputing.")

posts['likes_imputed'] = posts['likes'].isna()
n_missing_total = posts['likes_imputed'].sum()
posts['likes'] = posts.groupby('platform')['likes'].transform(lambda s: s.fillna(s.median()))
posts['likes'] = posts['likes'].fillna(posts['likes'].median())  # just in case a whole group was empty
posts['likes'] = posts['likes'].round().astype(int)
note(f"likes: imputed {n_missing_total} missing/negative values using the platform median "
     f"(flagged in the likes_imputed column).")

posts.to_csv('cleaned_posts.csv', index=False)
users.to_csv('cleaned_users.csv', index=False)
note(f"\nSaved cleaned_posts.csv ({posts.shape[0]} rows) and cleaned_users.csv ({users.shape[0]} rows).")

with open('cleaning_log.txt', 'w') as f:
    f.write('\n'.join(log))
