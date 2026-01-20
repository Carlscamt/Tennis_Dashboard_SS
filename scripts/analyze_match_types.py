"""Analyze model performance by tournament type."""
import polars as pl
from pathlib import Path

FUTURE_DIR = Path(__file__).parent.parent / "data" / "future"
df = pl.read_parquet(FUTURE_DIR / "upcoming_matches_latest.parquet")

print("="*70)
print("MATCH TYPE ANALYSIS")
print("="*70)
print(f"\nTotal future matches: {len(df)}")

# Group by tournament name patterns
print("\n" + "-"*50)
print("MATCHES BY TOURNAMENT TYPE")
print("-"*50)

# Identify tournament types
df = df.with_columns([
    pl.when(pl.col("tournament_name").str.contains("(?i)doubles"))
        .then(pl.lit("Doubles"))
    .when(pl.col("tournament_name").str.contains("(?i)itf|challenger"))
        .then(pl.lit("ITF/Challenger"))
    .when(pl.col("tournament_name").str.contains("(?i)utr"))
        .then(pl.lit("UTR"))
    .when(pl.col("tournament_name").str.contains("(?i)atp|wta|grand slam|australian|french|wimbledon|us open"))
        .then(pl.lit("ATP/WTA Tour"))
    .otherwise(pl.lit("Other"))
    .alias("tournament_type")
])

type_counts = df.group_by("tournament_type").agg([
    pl.len().alias("count"),
    pl.col("has_odds").sum().alias("with_odds"),
]).sort("count", descending=True)

for row in type_counts.iter_rows(named=True):
    pct = row["count"] / len(df) * 100
    print(f"{row['tournament_type']:<20} {row['count']:>5} ({pct:>5.1f}%) | {row['with_odds']} with odds")

# Unique tournament names
print("\n" + "-"*50)
print("SAMPLE TOURNAMENT NAMES BY TYPE")
print("-"*50)

for t_type in ["ITF/Challenger", "UTR", "Doubles", "Other"][:10]:
    subset = df.filter(pl.col("tournament_type") == t_type)
    if len(subset) > 0:
        print(f"\n{t_type}:")
        for name in subset["tournament_name"].unique().head(5).to_list():
            print(f"  - {name}")

print("\n" + "="*70)
print("MODEL TRAINING DATA COVERAGE")
print("="*70)

# Check historical data
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
if (PROCESSED_DIR / "features_dataset.parquet").exists():
    hist = pl.read_parquet(PROCESSED_DIR / "features_dataset.parquet")
    
    # The model was trained on ATP rankings data
    print(f"\nHistorical matches: {len(hist):,}")
    print("The model was trained on TOP 30 ATP SINGLES players only.")
    print("\n⚠️  MODEL COVERAGE ISSUES:")
    print("  - Doubles: NOT covered (different team dynamics)")
    print("  - ITF/Challenger: PARTIALLY covered (some players may have ATP history)")
    print("  - UTR: NOT covered (different rating system, amateur leagues)")
    print("  - WTA: NOT covered if only ATP was scraped")

print("\n" + "="*70)
print("RECOMMENDATION: Filter predictions to ATP/WTA tour matches only")
print("="*70)
