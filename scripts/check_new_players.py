"""Check for new players in predictions that weren't in original Top 30 scrape."""
import polars as pl
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent

# Load original historical data (Top 30 ATP scrape)
historical_df = pl.read_parquet(ROOT / "data" / "processed" / "features_dataset.parquet")

# Get unique player IDs from historical data
original_players = set(historical_df["player_id"].unique().to_list())
print(f"Original historical data: {len(original_players):,} unique players")

# Load current predictions
predictions_file = ROOT / "data" / "predictions" / "predictions_latest.parquet"
if predictions_file.exists():
    predictions_df = pl.read_parquet(predictions_file)
    
    # Get unique player IDs from predictions
    prediction_players = set(predictions_df["player_id"].unique().to_list())
    prediction_opponents = set(predictions_df["opponent_id"].unique().to_list())
    all_prediction_players = prediction_players.union(prediction_opponents)
    
    print(f"Players in current predictions: {len(all_prediction_players):,}")
    
    # Find new players (not in original scrape)
    new_players = all_prediction_players - original_players
    
    print(f"\n{'='*70}")
    print(f"NEW PLAYERS (not in original Top 30 scrape): {len(new_players)}")
    print('='*70)
    
    if new_players:
        # Get details about these new players
        new_player_matches = predictions_df.filter(
            pl.col("player_id").is_in(list(new_players)) | 
            pl.col("opponent_id").is_in(list(new_players))
        )
        
        print("\nNEW PLAYERS IN PREDICTIONS:")
        for pid in sorted(new_players)[:20]:  # Show first 20
            matches = new_player_matches.filter(
                (pl.col("player_id") == pid) | (pl.col("opponent_id") == pid)
            )
            
            if len(matches) > 0:
                first_match = matches.row(0, named=True)
                player_name = first_match.get("player_name") if first_match.get("player_id") == pid else first_match.get("opponent_name")
                tournament = first_match.get("tournament_name", "Unknown")
                
                print(f"  - {player_name} (ID: {pid}) | {tournament}")
        
        if len(new_players) > 20:
            print(f"  ... and {len(new_players) - 20} more")
        
        # Check if they have features
        print(f"\n{'='*70}")
        print("FEATURE AVAILABILITY CHECK")
        print('='*70)
        
        # Sample a few predictions with new players
        sample = new_player_matches.head(5)
        
        for row in sample.iter_rows(named=True):
            player = row.get("player_name", "Unknown")
            win_rate = row.get("player_win_rate_5", None)
            
            if win_rate is not None and win_rate != -999:
                status = f"✓ Has features (win_rate_5={win_rate:.3f})"
            else:
                status = "✗ Missing features (likely brand new player)"
            
            print(f"  {player}: {status}")
    else:
        print("\n✓ All players in predictions were in original Top 30 scrape!")
        
else:
    print("No predictions file found!")

# Load value bets to see if new players have value bets
value_bets_file = ROOT / "data" / "predictions" / "value_bets_latest.json"
if value_bets_file.exists() and new_players:
    with open(value_bets_file) as f:
        vb_data = json.load(f)
    
    value_bets = vb_data.get("value_bets", [])
    
    print(f"\n{'='*70}")
    print(f"VALUE BETS WITH NEW PLAYERS")
    print('='*70)
    
    # Check which value bets include new players (need to cross-reference IDs)
    # This is tricky without player IDs in the JSON, so we'll note this limitation
    print("Note: Check manually in dashboard if any value bets feature new Challenger players")
