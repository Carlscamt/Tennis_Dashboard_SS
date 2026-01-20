"""Check if any WTA players remain in predictions."""
import polars as pl
from pathlib import Path

ROOT = Path(__file__).parent.parent

pred_df = pl.read_parquet(ROOT / "data/predictions/predictions_latest.parquet")

# Check tournament types
print("="*70)
print("TOURNAMENT TYPES IN PREDICTIONS")
print("="*70)

if "tournament_type" in pred_df.columns:
    types = pred_df.group_by("tournament_type").agg([
        pl.len().alias("count")
    ]).sort("count", descending=True)
    
    for row in types.iter_rows(named=True):
        print(f"  {row['tournament_type']}: {row['count']}")
    
    # Check if WTA exists
    has_wta = pred_df.filter(pl.col("tournament_type") == "WTA")
    if len(has_wta) > 0:
        print(f"\n❌ WTA MATCHES FOUND: {len(has_wta)}")
        print("\nSample WTA matches:")
        for row in has_wta.head(5).iter_rows(named=True):
            print(f"  - {row.get('player_name')} vs {row.get('opponent_name')}")
    else:
        print("\n✅ NO WTA MATCHES IN PREDICTIONS!")
else:
    print("No tournament_type column found")

print("="*70)
