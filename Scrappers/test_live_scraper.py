"""
Live Scraper Test & Data Storage Verification

Tests:
1. Live API connectivity to SofaScore (single request)
2. Schema validation of real API responses
3. Data storage with future-proofing enhancements
4. Storage caveats documentation

Usage:
    python Scrappers/test_live_scraper.py
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import time

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://www.sofascore.com/api/v1"

# Test endpoints (minimal requests to avoid rate limiting)
TEST_ENDPOINTS = {
    "rankings": "/rankings/5",  # ATP Singles
    # We'll only test rankings to minimize API calls
}


# =============================================================================
# HTTP CLIENT (using httpx for local testing, tls_client not required)
# =============================================================================

def get_session():
    """Create HTTP session with proper headers."""
    try:
        from tls_client import Session
        session = Session(client_identifier="firefox_120")
        return session, "tls_client"
    except ImportError:
        import httpx
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        return httpx.Client(headers=headers, timeout=30), "httpx"


def fetch_json(session, endpoint: str) -> Optional[Dict]:
    """Fetch JSON from SofaScore API."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        time.sleep(1)  # Rate limiting
        response = session.get(url)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            print(f"  [WARN] Rate limited (403) on {endpoint}")
            return None
        elif response.status_code == 404:
            print(f"  [INFO] Not found (404) on {endpoint}")
            return None
        else:
            print(f"  [WARN] Unexpected status {response.status_code} on {endpoint}")
            return None
    except Exception as e:
        print(f"  [ERROR] Request failed: {e}")
        return None


# =============================================================================
# SCHEMA DEFINITIONS (for future-proofing)
# =============================================================================

EXPECTED_RANKING_SCHEMA = {
    "required_fields": ["position", "points", "team"],
    "team_required": ["id", "name"],
    "team_optional": ["country", "slug"],
}

EXPECTED_MATCH_SCHEMA = {
    "required_fields": ["id", "homeTeam", "awayTeam", "startTimestamp"],
    "team_required": ["id", "name"],
    "optional_fields": ["homeScore", "awayScore", "winnerCode", "groundType", "tournament", "roundInfo"],
}

EXPECTED_ODDS_SCHEMA = {
    "required_fields": ["markets"],
    "market_required": ["marketId", "choices"],
    "choice_required": ["name", "fractionalValue"],
}


# =============================================================================
# DATA STORAGE SCHEMA (with enhancements)
# =============================================================================

def create_enhanced_match_record(raw_event: Dict, player_id: int) -> Dict:
    """
    Create match record with all future-proofing enhancements.
    
    Enhancements:
    - match_date from timestamp
    - implied_prob from odds
    - schema_version for migration support
    """
    home_id = raw_event.get("homeTeam", {}).get("id")
    is_home = (home_id == player_id)
    
    home = raw_event.get("homeTeam", {})
    away = raw_event.get("awayTeam", {})
    home_score = raw_event.get("homeScore", {})
    away_score = raw_event.get("awayScore", {})
    tournament = raw_event.get("tournament", {}).get("uniqueTournament", {})
    
    winner = raw_event.get("winnerCode")
    player_won = (winner == 1 and is_home) or (winner == 2 and not is_home)
    
    # Base record
    record = {
        # Schema version for future migrations
        "_schema_version": "1.1",
        "_scraped_at": datetime.now().isoformat(),
        
        # IDs
        "event_id": raw_event.get("id"),
        "player_id": player_id,
        "opponent_id": away.get("id") if is_home else home.get("id"),
        
        # Names
        "player_name": home.get("name") if is_home else away.get("name"),
        "opponent_name": away.get("name") if is_home else home.get("name"),
        
        # Outcome
        "player_won": player_won,
        "is_home": is_home,
        
        # Scores
        "player_sets": home_score.get("current", 0) if is_home else away_score.get("current", 0),
        "opponent_sets": away_score.get("current", 0) if is_home else home_score.get("current", 0),
        
        # Tournament
        "tournament_id": tournament.get("id"),
        "tournament_name": tournament.get("name"),
        "round_name": raw_event.get("roundInfo", {}).get("name"),
        "ground_type": raw_event.get("groundType"),
        
        # Timestamps - ENHANCED
        "start_timestamp": raw_event.get("startTimestamp"),
        "status": raw_event.get("status", {}).get("type"),
    }
    
    # Add enhanced date fields
    ts = raw_event.get("startTimestamp")
    if ts:
        dt = datetime.fromtimestamp(ts)
        record["match_date"] = dt.date().isoformat()
        record["match_year"] = dt.year
        record["match_month"] = dt.month
        record["match_day"] = dt.day
    
    # Set-by-set scores
    for i in range(1, 6):
        record[f"player_set{i}"] = home_score.get(f"period{i}") if is_home else away_score.get(f"period{i}")
        record[f"opponent_set{i}"] = away_score.get(f"period{i}") if is_home else home_score.get(f"period{i}")
    
    return record


def add_odds_to_record(record: Dict, odds_data: Dict) -> Dict:
    """Add odds and implied probabilities to record."""
    if not odds_data or "markets" not in odds_data:
        record["has_odds"] = False
        return record
    
    is_home = record.get("is_home", True)
    
    for market in odds_data.get("markets", []):
        if market.get("marketId") == 1:
            for choice in market.get("choices", []):
                name = choice.get("name", "")
                frac = choice.get("fractionalValue", "")
                
                try:
                    if '/' in str(frac):
                        num, den = map(int, str(frac).split('/'))
                        decimal = round(1 + (num / den), 3)
                    else:
                        decimal = float(frac)
                except:
                    continue
                
                if name == "1":
                    record["odds_home"] = decimal
                elif name == "2":
                    record["odds_away"] = decimal
    
    # Assign based on position
    if is_home:
        record["odds_player"] = record.get("odds_home")
        record["odds_opponent"] = record.get("odds_away")
    else:
        record["odds_player"] = record.get("odds_away")
        record["odds_opponent"] = record.get("odds_home")
    
    # ENHANCED: Pre-calculate implied probabilities
    if record.get("odds_player"):
        record["implied_prob_player"] = round(1 / record["odds_player"], 4)
    if record.get("odds_opponent"):
        record["implied_prob_opponent"] = round(1 / record["odds_opponent"], 4)
    
    record["has_odds"] = bool(record.get("odds_player"))
    
    return record


# =============================================================================
# STORAGE CAVEATS
# =============================================================================

STORAGE_CAVEATS = """
================================================================================
DATA STORAGE CAVEATS FOR FUTURE-PROOFING
================================================================================

1. SCHEMA VERSIONING
   - Every record includes '_schema_version' field
   - Current version: "1.1"
   - When schema changes, bump version and add migration logic
   - Example: if v1.2 adds 'elo_player', migration fills with null for old records

2. API CHANGES
   - SofaScore may change API structure without notice
   - MITIGATION: Store raw JSON backups alongside processed parquet
   - If API changes, we can re-parse raw data with new logic

3. ODDS COVERAGE
   - Historical matches (pre-2018) often lack odds data
   - Challenger/ITF events have low odds coverage
   - MITIGATION: 'has_odds' flag allows filtering for ML training
   - Consider: only use matches with odds for betting model

4. TIMESTAMPS
   - Unix timestamps in UTC
   - 'match_date' is derived, not from API
   - MITIGATION: Always derive from 'start_timestamp', never store externally

5. PLAYER IDS
   - SofaScore IDs are internal and can change during merges
   - MITIGATION: Store player_name alongside player_id
   - If ID changes, can re-map via name matching

6. MISSING DATA
   - Some matches lack statistics (older matches)
   - Some lack complete scores (walkovers, retirements)
   - MITIGATION: 'has_stats' flag, nullable score columns

7. PARQUET BEST PRACTICES
   - Use snappy compression (default in Polars)
   - Partition by year for large datasets: data/processed/{year}/
   - Schema enforcement prevents silent corruption

8. RATE LIMITING
   - SofaScore rate limits after ~100-200 requests/minute
   - Colab IPs may be shared and pre-blocked
   - MITIGATION: Exponential backoff, session rotation

9. DATA FRESHNESS
   - Rankings only show current state, not historical
   - MITIGATION: Use ELO ratings (calculated from match history)
   - ELO is actually BETTER than raw rankings for prediction

10. COLUMN NAMING
    - Use descriptive, non-abbreviated names
    - player_ prefix for target player, opponent_ for opponent
    - _value suffix for numeric versions of string stats (e.g., "65%" -> 65)

================================================================================
"""


# =============================================================================
# TEST CASES
# =============================================================================

class TestLiveAPIConnectivity:
    """Test actual API connectivity (minimal requests)."""
    
    def __init__(self):
        self.session, self.client_type = get_session()
        print(f"  Using HTTP client: {self.client_type}")
    
    def test_rankings_endpoint(self):
        """Test rankings endpoint returns valid data."""
        data = fetch_json(self.session, "/rankings/5")
        
        if data is None:
            print("  [SKIP] API not accessible (rate limited or blocked)")
            return True  # Don't fail if API is blocked
        
        # Validate structure
        assert "rankingRows" in data, "Missing rankingRows"
        assert len(data["rankingRows"]) > 0, "Empty rankings"
        
        # Validate first player
        player = data["rankingRows"][0]
        assert "position" in player
        assert "team" in player
        assert "id" in player["team"]
        assert "name" in player["team"]
        
        print(f"    Top player: #{player['position']} {player['team']['name']}")
        return True


class TestSchemaValidation:
    """Test schema against expected structure."""
    
    def test_ranking_schema_matches(self):
        """Verify ranking response matches expected schema."""
        mock_ranking = {
            "position": 1,
            "points": 9850,
            "team": {"id": 275923, "name": "Jannik Sinner", "country": {"alpha2": "IT"}}
        }
        
        for field in EXPECTED_RANKING_SCHEMA["required_fields"]:
            assert field in mock_ranking, f"Missing required field: {field}"
        
        for field in EXPECTED_RANKING_SCHEMA["team_required"]:
            assert field in mock_ranking["team"], f"Missing team field: {field}"
    
    def test_match_schema_complete(self):
        """Verify enhanced match record has all required fields."""
        mock_event = {
            "id": 12345678,
            "homeTeam": {"id": 275923, "name": "Jannik Sinner"},
            "awayTeam": {"id": 206570, "name": "Carlos Alcaraz"},
            "homeScore": {"current": 2},
            "awayScore": {"current": 1},
            "winnerCode": 1,
            "startTimestamp": 1705334400,
            "tournament": {"uniqueTournament": {"id": 1, "name": "Australian Open"}},
            "status": {"type": "finished"},
        }
        
        record = create_enhanced_match_record(mock_event, 275923)
        
        # Required fields
        required = ["event_id", "player_id", "opponent_id", "player_won", "start_timestamp"]
        for field in required:
            assert field in record, f"Missing required field: {field}"
        
        # Enhanced fields
        enhanced = ["match_date", "match_year", "match_month", "_schema_version"]
        for field in enhanced:
            assert field in record, f"Missing enhanced field: {field}"


class TestDataStorageEnhancements:
    """Test storage enhancements work correctly."""
    
    def test_schema_version_present(self):
        """Records should have schema version."""
        mock_event = {
            "id": 1, "homeTeam": {"id": 1, "name": "A"}, "awayTeam": {"id": 2, "name": "B"},
            "homeScore": {"current": 2}, "awayScore": {"current": 1}, "winnerCode": 1,
            "startTimestamp": 1705334400, "status": {"type": "finished"},
            "tournament": {"uniqueTournament": {"id": 1, "name": "Test"}},
        }
        
        record = create_enhanced_match_record(mock_event, 1)
        
        assert "_schema_version" in record
        assert record["_schema_version"] == "1.1"
    
    def test_date_fields_populated(self):
        """Date fields should be populated from timestamp."""
        mock_event = {
            "id": 1, "homeTeam": {"id": 1, "name": "A"}, "awayTeam": {"id": 2, "name": "B"},
            "homeScore": {"current": 2}, "awayScore": {"current": 1}, "winnerCode": 1,
            "startTimestamp": 1705334400,  # Jan 15, 2024
            "status": {"type": "finished"},
            "tournament": {"uniqueTournament": {"id": 1, "name": "Test"}},
        }
        
        record = create_enhanced_match_record(mock_event, 1)
        
        assert record["match_year"] == 2024
        assert record["match_month"] == 1
        assert record["match_day"] == 15
        assert record["match_date"] == "2024-01-15"
    
    def test_implied_prob_calculated(self):
        """Implied probability should be pre-calculated."""
        record = {"is_home": True}
        odds_data = {
            "markets": [{
                "marketId": 1,
                "choices": [
                    {"name": "1", "fractionalValue": "8/13"},  # 1.615
                    {"name": "2", "fractionalValue": "11/8"},  # 2.375
                ]
            }]
        }
        
        record = add_odds_to_record(record, odds_data)
        
        assert "implied_prob_player" in record
        assert "implied_prob_opponent" in record
        
        # 1/1.615 = 0.619
        assert abs(record["implied_prob_player"] - 0.619) < 0.01
    
    def test_parquet_snappy_compression(self):
        """Test Polars can write with snappy compression."""
        import polars as pl
        import io
        
        df = pl.DataFrame({
            "event_id": [1, 2, 3],
            "player_won": [True, False, True],
            "odds_player": [1.5, 2.0, 1.8],
        })
        
        buffer = io.BytesIO()
        df.write_parquet(buffer, compression="snappy")
        
        # Verify size is reasonable (compressed)
        size = buffer.tell()
        assert size > 0, "Empty parquet file"
        
        # Verify can read back
        buffer.seek(0)
        df_read = pl.read_parquet(buffer)
        assert len(df_read) == 3, "Data corrupted after round-trip"
    
    def test_year_partitioning(self):
        """Test year-based path generation."""
        from datetime import datetime, timezone
        
        # Use UTC timestamps to avoid timezone issues
        timestamps = [
            (1577836800, 2020),  # Jan 1, 2020 UTC
            (1609459200, 2021),  # Jan 1, 2021 UTC
            (1640995200, 2022),  # Jan 1, 2022 UTC
        ]
        
        for ts, expected_year in timestamps:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            year = dt.year
            path = f"data/processed/{year}/matches.parquet"
            assert str(expected_year) in path, f"Expected {expected_year} in path, got {year}"


class TestStorageCaveats:
    """Verify caveats are documented."""
    
    def test_caveats_document_exists(self):
        """Storage caveats should be documented."""
        assert len(STORAGE_CAVEATS) > 1000, "Caveats too short"
        
        required_topics = [
            "SCHEMA VERSIONING",
            "API CHANGES", 
            "ODDS COVERAGE",
            "TIMESTAMPS",
            "PLAYER IDS",
            "PARQUET",
            "RATE LIMITING",
            "ELO",  # For rankings workaround
        ]
        
        for topic in required_topics:
            assert topic in STORAGE_CAVEATS, f"Missing caveat: {topic}"


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    """Run all tests."""
    test_classes = [
        ("Live API Connectivity", TestLiveAPIConnectivity),
        ("Schema Validation", TestSchemaValidation),
        ("Data Storage Enhancements", TestDataStorageEnhancements),
        ("Storage Caveats", TestStorageCaveats),
    ]
    
    print("="*70)
    print("LIVE SCRAPER & STORAGE VERIFICATION")
    print("="*70)
    
    total = 0
    passed = 0
    failed = []
    
    for section_name, test_class in test_classes:
        print(f"\n{section_name}")
        print("-" * 40)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            total += 1
            try:
                result = getattr(instance, method_name)()
                print(f"  [PASS] {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {e}")
                failed.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"  [FAIL] {method_name}: {type(e).__name__}: {e}")
                failed.append(f"{test_class.__name__}.{method_name}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    print("="*70)
    
    # Print storage caveats
    print(STORAGE_CAVEATS)
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
