"""Check for duplicate matches."""
import polars as pl
from pathlib import Path

df = pl.read_parquet(Path("data/future/upcoming_matches_latest.parquet"))

print(f"Total matches: {len(df)}")
print(f"Unique event_ids: {df['event_id'].n_unique()}")

# Find duplicates
dupes = df.group_by("event_id").agg(pl.len().alias("count")).filter(pl.col("count") > 1)
print(f"\nDuplicate event_ids: {len(dupes)}")

if len(dupes) > 0:
    print("\nSample duplicates:")
    dupe_ids = dupes["event_id"].head(3).to_list()
    for eid in dupe_ids:
        matches = df.filter(pl.col("event_id") == eid)
        print(f"\nEvent {eid} ({len(matches)} rows):")
        for row in matches.iter_rows(named=True):
            print(f"  - {row['player_name']} vs {row['opponent_name']} | {row.get('tournament_type', 'N/A')}")
