"""Check 2026 data discrepancy"""
import polars as pl
from datetime import date

# Check features_dataset (used by backtest - historical data with outcomes)
df = pl.read_parquet('data/processed/features_dataset.parquet')
df_2026 = df.filter(pl.col('match_date') >= date(2026, 1, 1))

print("=" * 60)
print("FEATURES_DATASET (Historical - used by backtest)")
print("=" * 60)
print(f"Total 2026 matches: {len(df_2026)}")
print(f"With odds: {len(df_2026.filter(pl.col('odds_player').is_not_null()))}")
if len(df_2026) > 0:
    print(f"Date range: {df_2026['match_date'].min()} to {df_2026['match_date'].max()}")
print(f"Has player_won: {'player_won' in df.columns}")

# Check upcoming matches (used by dashboard - future matches)
print("\n" + "=" * 60)
print("UPCOMING_MATCHES (Future - used by dashboard)")
print("=" * 60)
df_upcoming = pl.read_parquet('data/future/upcoming_matches_latest.parquet')
print(f"Total upcoming: {len(df_upcoming)}")
print(f"With odds: {len(df_upcoming.filter(pl.col('has_odds')))}")
if len(df_upcoming) > 0:
    print(f"Date range: {df_upcoming['match_date'].min()} to {df_upcoming['match_date'].max()}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("Backtest uses HISTORICAL data with known outcomes (player_won)")
print("Dashboard uses FUTURE data with no outcomes yet")
print(f"Historical 2026 with odds: {len(df_2026.filter(pl.col('odds_player').is_not_null()))}")
print(f"Future matches with odds: {len(df_upcoming.filter(pl.col('has_odds')))}")
