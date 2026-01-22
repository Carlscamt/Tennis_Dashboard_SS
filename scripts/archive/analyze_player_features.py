"""Analyze feature quality for players in predictions.

Compares original Top 30 players vs new players discovered in upcoming matches.
"""
import polars as pl
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent

print("="*70)
print("FEATURE QUALITY ANALYSIS")
print("="*70)

# Load original historical data (Top 30 ATP scrape)
historical_df = pl.read_parquet(ROOT / "data" / "processed" / "features_dataset.parquet")
original_players = set(historical_df["player_id"].unique().to_list())

print(f"\nOriginal training data: {len(original_players)} unique players")
print(f"Total matches: {len(historical_df):,}")

# Load updated historical data (after update_active_players.py)
updated_hist_path = ROOT / "data" / "raw" / "atp_matches"
if updated_hist_path.exists():
    # Check all parquet files for player coverage
    all_players = set()
    total_updated_matches = 0
    
    for year_dir in updated_hist_path.glob("year=*"):
        for player_file in year_dir.glob("*.parquet"):
            try:
                df_temp = pl.read_parquet(player_file)
                total_updated_matches += len(df_temp)
                all_players.update(df_temp["player_id"].unique().to_list())
            except:
                pass
    
    print(f"\nUpdated raw data: {len(all_players)} unique players")
    print(f"Total matches: {total_updated_matches:,}")
    
    new_players = all_players - original_players
    print(f"NEW players added: {len(new_players)}")
else:
    print("\n⚠️  No updated raw data found")
    new_players = set()

# Load current predictions
predictions_file = ROOT / "data" / "predictions" / "predictions_latest.parquet"
if predictions_file.exists():
    predictions_df = pl.read_parquet(predictions_file)
    
    # Get all players in predictions
    pred_players =set(predictions_df["player_id"].unique().to_list())
    pred_opponents = set(predictions_df["opponent_id"].unique().to_list())
    all_pred_players = pred_players.union(pred_opponents)
    
    print(f"\nPlayers in predictions: {len(all_pred_players)}")
    
    # Analyze feature quality
    print(f"\n{'='*70}")
    print("FEATURE QUALITY BY PLAYER TYPE")
    print('='*70)
    
    # Check if predictions have feature columns
    feature_cols = [c for c in predictions_df.columns if "win_rate" in c or "h2h" in c]
    
    if feature_cols:
        print(f"\nFeature columns available: {len(feature_cols)}")
        
        # Sample analysis
        tier_1 = []  # In training, good features
        tier_2 = []  # Not in training, good features
        tier_3 = []  # Missing features
        
        for row in predictions_df.head(50).iter_rows(named=True):
            player_id = row.get("player_id")
            player_name = row.get("player_name", "Unknown")
            
            # Check if in training
            in_training = player_id in original_players
            
            # Check feature quality (use win_rate_5 as proxy)
            win_rate_5 = row.get("player_win_rate_5")
            has_features = win_rate_5 is not None and win_rate_5 != -999
            
            if in_training and has_features:
                tier_1.append(player_name)
            elif has_features:
                tier_2.append(player_name)
            else:
                tier_3.append(player_name)
        
        print(f"\nTier 1 (In training + features): {len(tier_1)}")
        print(f"Tier 2 (New player + features): {len(tier_2)}")
        print(f"Tier 3 (Missing features): {len(tier_3)}")
        
        if tier_3:
            print(f"\n⚠️  Players with missing features:")
            for name in tier_3[:10]:
                print(f"    - {name}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print('='*70)
    
    new_in_predictions = all_pred_players - original_players
    coverage = len(original_players.intersection(all_pred_players)) / len(all_pred_players) * 100 if all_pred_players else 0
    
    print(f"\nTraining Coverage: {coverage:.1f}%")
    print(f"New players in predictions: {len(new_in_predictions)}")
    
    if coverage < 50:
        print("\n⚠️  WARNING: Less than 50% of prediction players were in training!")
        print("   Consider:")
        print("   - Re-scraping with wider player base (e.g., Top 100)")
        print("   - Flagging low-confidence predictions")
        print("   - Only betting on high-confidence matches")
    elif coverage < 75:
        print("\n⚡ MODERATE: 50-75% training coverage")
        print("   Consider adding confidence scores to predictions")
    else:
        print("\n✓ GOOD: >75% training coverage")

else:
    print("\n⚠️  No predictions file found")

print("\n" + "="*70)
