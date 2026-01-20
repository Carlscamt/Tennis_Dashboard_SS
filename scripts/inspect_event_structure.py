"""Check SofaScore event structure for WTA detection."""
import sys
from pathlib import Path
import time
import random
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from tls_client import Session
    session = Session(client_identifier="firefox_120")
    HAS_TLS = True
except:
    import httpx
    session = httpx.Client(headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/120.0",
        "Accept": "application/json"
    })
    HAS_TLS = False

BASE_URL = "https://www.sofascore.com/api/v1"

# Fetch today's events
date_str = datetime.now().date().isoformat()
url = f"{BASE_URL}/sport/tennis/scheduled-events/{date_str}"

time.sleep(0.5)
response = session.get(url)
data = response.json()

events = data.get("events", [])[:10]  # First 10 events

print("="*70)
print("SOFASCORE EVENT STRUCTURE ANALYSIS")
print("="*70)

for i, event in enumerate(events, 1):
    home = event.get("homeTeam", {}).get("name", "Unknown")
    away = event.get("awayTeam", {}).get("name", "Unknown")
    
    tournament = event.get("tournament", {})
    unique_tournament = tournament.get("uniqueTournament", {})
    category = tournament.get("category", {})
    
    print(f"\n[{i}] {home} vs {away}")
    print(f"  Tournament: {tournament.get('name', 'N/A')}")
    print(f"  Tournament slug: {tournament.get('slug', 'N/A')}")
    print(f"  Unique Tournament: {unique_tournament.get('name', 'N/A')}")
    print(f"  Unique Tournament slug: {unique_tournament.get('slug', 'N/A')}")
    print(f"  Category: {category.get('name', 'N/A')}")
    print(f"  Category slug: {category.get('slug', 'N/A')}")
    print(f"  Category ID: {category.get('id', 'N/A')}")
    
    # Check for WTA indicators
    has_wta = any([
        "wta" in str(tournament.get('slug', '')).lower(),
        "wta" in str(unique_tournament.get('slug', '')).lower(),
        "wta" in str(category.get('slug', '')).lower(),
        "women" in str(category.get('name', '')).lower(),
    ])
    
    if has_wta:
        print(f"  >>> WTA DETECTED <<<")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("Check 'category.slug' or 'category.name' for 'wta' prefix")
print("="*70)
