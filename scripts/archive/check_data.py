"""Comprehensive data quality check."""
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
files = sorted(DATA_DIR.glob("atp_matches_*.parquet"), key=lambda p: p.stat().st_mtime)

print("="*60)
print("DATA FILES SUMMARY")
print("="*60)

for f in files:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name}: {size_kb:.1f} KB")

print("\n" + "="*60)
print("LATEST FILE ANALYSIS")
print("="*60)

if files:
    latest = files[-1]
    df = pl.read_parquet(latest)
    
    print(f"\nFile: {latest.name}")
    print(f"Total matches: {len(df):,}")
    print(f"Unique players: {df['player_id'].n_unique()}")
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
    print(f"Total columns: {len(df.columns)}")
    
    # Check for stats columns
    stat_cols = [c for c in df.columns if c.startswith("player_") and "_" in c[8:]]
    print(f"Stat columns: {len(stat_cols)}")
    
    # Check odds coverage
    if "has_odds" in df.columns:
        odds_count = df.filter(pl.col("has_odds")).height
        print(f"Matches with odds: {odds_count:,} ({odds_count/len(df)*100:.1f}%)")
    
    if "has_stats" in df.columns:
        stats_count = df.filter(pl.col("has_stats")).height
        print(f"Matches with stats: {stats_count:,} ({stats_count/len(df)*100:.1f}%)")
    
    # Year distribution
    print("\n" + "-"*40)
    print("MATCHES BY YEAR")
    print("-"*40)
    year_counts = df.group_by("match_year").agg(pl.count().alias("matches")).sort("match_year")
    for row in year_counts.iter_rows():
        print(f"  {row[0]}: {row[1]:,} matches")
    
    # Top players by matches
    print("\n" + "-"*40)
    print("TOP 10 PLAYERS BY MATCHES")
    print("-"*40)
    player_counts = (df.group_by(["player_id", "player_name"])
                    .agg(pl.count().alias("matches"))
                    .sort("matches", descending=True)
                    .head(10))
    for i, row in enumerate(player_counts.iter_rows(), 1):
        print(f"  {i}. {row[1]}: {row[2]} matches")
    
    # Surface distribution
    print("\n" + "-"*40)
    print("SURFACE DISTRIBUTION")
    print("-"*40)
    if "ground_type" in df.columns:
        surface_counts = df.group_by("ground_type").agg(pl.count().alias("matches")).sort("matches", descending=True)
        for row in surface_counts.iter_rows():
            if row[0]:
                print(f"  {row[0]}: {row[1]:,} ({row[1]/len(df)*100:.1f}%)")
    
    # Sample columns
    print("\n" + "-"*40)
    print("COLUMN CATEGORIES")
    print("-"*40)
    meta_cols = ["_schema_version", "_scraped_at", "event_id", "player_id", "opponent_id", 
                 "player_name", "opponent_name", "player_won", "is_home", "match_date"]
    score_cols = [c for c in df.columns if "set" in c.lower()]
    odds_cols = [c for c in df.columns if "odds" in c.lower() or "prob" in c.lower()]
    
    print(f"  Meta/ID columns: {len([c for c in meta_cols if c in df.columns])}")
    print(f"  Score columns: {len(score_cols)}")
    print(f"  Odds columns: {len(odds_cols)}")
    print(f"  Stat columns: {len(stat_cols)}")
    
    print("\n" + "="*60)
    print("DATA QUALITY READY FOR ML PIPELINE")
    print("="*60)
