"""Quick summary - check if update_active_players was run."""
import polars as pl
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Check updated data
updated_dir = ROOT / "data/raw/atp_matches"
if updated_dir.exists():
    all_files = list(updated_dir.glob("year=*/**.parquet"))
    print(f"Updated player files found: {len(all_files)}")
    
    # Check if recently modified
    from datetime import datetime, timedelta
    recent = sum(1 for f in all_files if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)) < timedelta(hours=24))
    print(f"Files updated in last 24h: {recent}")
    
    if recent == 0:
        print("\nACTION REQUIRED: Run update_active_players.py to fetch new player data")
else:
    print("No updated player data found!")
    print("\nACTION REQUIRED: Run update_active_players.py")
