"""Quick verification of latest predictions confidence scores."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
vb_path = ROOT / "data" / "predictions" / "value_bets_latest.json"

with open(vb_path) as f:
    data = json.load(f)

value_bets = data.get("value_bets", [])

print("="*70)
print(f"VALUE BETS WITH CONFIDENCE SCORES ({len(value_bets)} total)")
print("="*70)

# Group by confidence tier
by_tier = {"High": [], "Medium": [], "Low": [], "Unknown": []}

for bet in value_bets:
    tier = bet.get("confidence_tier", "Unknown")
    by_tier[tier].append(bet)

for tier in ["High", "Medium", "Low", "Unknown"]:
    count = len(by_tier[tier])
    print(f"\n{tier} Confidence: {count} bets")
    
    if count > 0:
        for bet in by_tier[tier][:3]:  # Show first 3
            player = bet.get("player", "Unknown")
            opponent = bet.get("opponent", "Unknown")
            score = bet.get("confidence_score", 0)
            edge = bet.get("edge", 0)
            print(f"  - {player} vs {opponent} | Score: {score} | Edge: +{edge:.1f}%")
        
        if count > 3:
            print(f"  ... and {count - 3} more")

print("\n" + "="*70)
avg_confidence = sum(b.get("confidence_score", 0) for b in value_bets) / len(value_bets) if value_bets else 0
print(f"Average Confidence Score: {avg_confidence:.1f}")
print("="*70)
