"""Analyze WTA vs ATP in training data."""
import polars as pl
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pl.read_parquet(ROOT / "data" / "processed" / "features_dataset.parquet")

print("="*70)
print("WTA vs ATP ANALYSIS IN TRAINING DATA")
print("="*70)

# Identify tour based on tournament name patterns
wta_keywords = ["women", "wta", "w15", "w25", "w35", "w50", "w75", "w100", "w125", "girls"]
atp_keywords = ["men", "atp", "m15", "m25", "m35", "m50", "m75", "m100", "m125", "boys"]

df = df.with_columns([
    pl.when(
        pl.col("tournament_name").str.to_lowercase().str.contains("|".join(wta_keywords))
    ).then(pl.lit("WTA/Women"))
    .when(
        pl.col("tournament_name").str.to_lowercase().str.contains("|".join(atp_keywords))
    ).then(pl.lit("ATP/Men"))
    .otherwise(pl.lit("Mixed/Unknown"))
    .alias("detected_tour")
])

# Stats by tour
print("\n" + "-"*50)
print("TRAINING DATA BREAKDOWN BY TOUR")
print("-"*50)

tour_stats = df.group_by("detected_tour").agg([
    pl.len().alias("matches"),
    pl.col("player_won").mean().alias("win_rate"),
    pl.col("has_odds").sum().alias("with_odds"),
]).sort("matches", descending=True)

for row in tour_stats.iter_rows(named=True):
    pct = row["matches"] / len(df) * 100
    print(f"\n{row['detected_tour']}:")
    print(f"  Matches: {row['matches']:,} ({pct:.1f}%)")
    print(f"  Win Rate: {row['win_rate']:.3f}")
    print(f"  With Odds: {row['with_odds']:,}")

# Sample WTA tournament names
print("\n" + "-"*50)
print("SAMPLE WTA TOURNAMENTS IN DATA")
print("-"*50)
wta_df = df.filter(pl.col("detected_tour") == "WTA/Women")
if len(wta_df) > 0:
    for t in wta_df["tournament_name"].unique().head(15).to_list():
        print(f"  - {t}")
else:
    print("  No WTA tournaments detected!")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
