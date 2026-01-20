"""Data quality analysis for new players in predictions."""
import polars as pl
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("NEW PLAYER DATA QUALITY ANALYSIS")
print("="*70)

# Load training players
hist_df = pl.read_parquet(ROOT / "data/processed/features_dataset.parquet")
trained_players = set(hist_df["player_id"].unique().to_list())
print(f"\nOriginal training players: {len(trained_players)}")

# Load current predictions
pred_df = pl.read_parquet(ROOT / "data/predictions/predictions_latest.parquet")
pred_players = set(pred_df["player_id"].unique().to_list())
pred_opponents = set(pred_df["opponent_id"].unique().to_list())
all_pred_players = pred_players.union(pred_opponents)

new_players = all_pred_players - trained_players
print(f"Players in predictions: {len(all_pred_players)}")
print(f"NEW players: {len(new_players)} ({len(new_players)/len(all_pred_players)*100:.1f}%)")

# Analyze feature quality for new players
print(f"\n{'='*70}")
print("FEATURE QUALITY CHECK")
print('='*70)

# Check key features
feature_checks = {
    "player_win_rate_5": "Recent form (5 matches)",
    "player_win_rate_10": "Short-term form (10 matches)",
    "player_win_rate_20": "Mid-term form (20 matches)",
    "h2h_matches": "Head-to-head history",
    "player_surface_win_rate_10": "Surface-specific form"
}

new_player_rows = pred_df.filter(
    pl.col("player_id").is_in(list(new_players)) | 
    pl.col("opponent_id").is_in(list(new_players))
)

print(f"\nPredictions involving new players: {len(new_player_rows)}")

# Check feature completeness
for feature, description in feature_checks.items():
    if feature in new_player_rows.columns:
        # Count non-null, non-default values
        valid = new_player_rows.filter(
            (pl.col(feature).is_not_null()) & 
            (pl.col(feature) != -999)
        )
        pct = len(valid) / len(new_player_rows) * 100 if len(new_player_rows) > 0 else 0
        
        status = "✓" if pct >= 70 else "⚠" if pct >= 40 else "✗"
        print(f"  {status} {description:.<40} {pct:.1f}%")

# Check for players with NO historical data
print(f"\n{'='*70}")
print("PLAYERS WITH MISSING DATA")
print('='*70)

missing_data_count = 0
for row in new_player_rows.head(30).iter_rows(named=True):
    player_id = row.get("player_id")
    player_name = row.get("player_name", "Unknown")
    
    # Check if this is a new player
    if player_id in new_players:
        win_rate = row.get("player_win_rate_5")
        
        if win_rate is None or win_rate == -999:
            missing_data_count += 1
            opponent = row.get("opponent_name", "Unknown")
            conf_score = row.get("confidence_score", 0)
            
            if missing_data_count <= 10:
                print(f"  - {player_name} vs {opponent} | Confidence: {conf_score}")

if missing_data_count > 10:
    print(f"  ... and {missing_data_count - 10} more")

# Check updated data directory
print(f"\n{'='*70}")
print("CHECKING UPDATED PLAYER DATA")
print('='*70)

updated_dir = ROOT / "data/raw/atp_matches"
if updated_dir.exists():
    # Count matches per new player
    player_match_counts = {}
    
    for year_dir in updated_dir.glob("year=*"):
        for player_file in year_dir.glob("*.parquet"):
            try:
                player_id = int(player_file.stem)
                if player_id in new_players:
                    df_temp = pl.read_parquet(player_file)
                    player_match_counts[player_id] = len(df_temp)
            except:
                pass
    
    print(f"\nNew players with updated data: {len(player_match_counts)}")
    
    # Categorize by match count
    categories = {
        "Excellent (50+ matches)": [],
        "Good (20-49 matches)": [],
        "Moderate (10-19 matches)": [],
        "Poor (5-9 matches)": [],
        "Very Poor (<5 matches)": []
    }
    
    for pid, count in player_match_counts.items():
        if count >= 50:
            categories["Excellent (50+ matches)"].append((pid, count))
        elif count >= 20:
            categories["Good (20-49 matches)"].append((pid, count))
        elif count >= 10:
            categories["Moderate (10-19 matches)"].append((pid, count))
        elif count >= 5:
            categories["Poor (5-9 matches)"].append((pid, count))
        else:
            categories["Very Poor (<5 matches)"].append((pid, count))
    
    for category, players in categories.items():
        count = len(players)
        pct = count / len(player_match_counts) * 100 if player_match_counts else 0
        print(f"  {category}: {count} players ({pct:.1f}%)")
        
        # Show examples
        for pid, mcount in sorted(players, key=lambda x: -x[1])[:2]:
            # Find player name
            name = "Unknown"
            for row in new_player_rows.filter(pl.col("player_id") == pid).head(1).iter_rows(named=True):
                name = row.get("player_name", "Unknown")
                break
            print(f"    - {name}: {mcount} matches")
else:
    print("\n⚠️  No updated player data directory found!")
    print("   Run: python scripts/update_active_players.py")

print(f"\n{'='*70}")
print("RECOMMENDATIONS")
print('='*70)

# Calculate recommendations
if len(player_match_counts) < len(new_players) * 0.5:
    print("\n⚠️  ACTION REQUIRED: update_active_players.py needs to run")
    print("   Command: python scripts/update_active_players.py")

poor_data = len(categories.get("Poor (5-9 matches)", [])) + len(categories.get("Very Poor (<5 matches)", []))
if poor_data > len(player_match_counts) * 0.3:
    print("\n⚠️  WARNING: 30%+ of new players have <10 matches")
    print("   Consider filtering predictions to require minimum match history")

print("\n✓ Add minimum confidence filter in predictions (e.g., score >= 40)")
print("✓ Consider re-training model with wider player base (Top 100 instead of Top 30)")

print("\n" + "="*70)
