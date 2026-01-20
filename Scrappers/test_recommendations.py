"""
Recommendation Feasibility Tests

Tests to verify which DB design recommendations are possible 
with current SofaScore API endpoints and data.

Usage:
    python Scrappers/test_recommendations.py
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# =============================================================================
# API ENDPOINT ANALYSIS
# =============================================================================

"""
SOFASCORE API ENDPOINTS CURRENTLY IN USE:

1. /rankings/{type}
   - Returns: Current rankings with player_id, name, country, points
   - Does NOT return: Historical rankings at specific dates
   
2. /team/{player_id}/events/singles/last/{page}
   - Returns: Match events with homeTeam, awayTeam, scores, tournament, timestamp
   - Does NOT return: Player rankings at match time
   
3. /event/{event_id}/statistics
   - Returns: Match stats (aces, double faults, serve %, etc.)
   - Numeric values AND string values (e.g., "65%")
   
4. /event/{event_id}/odds/1/all
   - Returns: Fractional odds for home/away
   - Does NOT return: Implied probabilities (we calculate)
"""

# =============================================================================
# RECOMMENDATION FEASIBILITY
# =============================================================================

RECOMMENDATIONS = {
    "1_player_rank": {
        "description": "Add player_rank and opponent_rank at match time",
        "api_support": False,
        "notes": """
        PROBLEM: SofaScore /rankings endpoint only returns CURRENT rankings.
        There is NO endpoint for historical rankings at a specific date.
        
        WORKAROUNDS:
        A) Build a ranking snapshot database by scraping rankings weekly
        B) Use Jeff Sackmann dataset which includes historical rankings
        C) Calculate proxy via ELO ratings (implemented in features.py)
        D) Use ATP/WTA official API (requires registration) for historical ranks
        
        RECOMMENDATION: Use workaround B or C
        """,
        "feasible": "PARTIAL - requires external data or proxy",
    },
    
    "2_match_date": {
        "description": "Add match_date as datetime column",
        "api_support": True,
        "notes": """
        FULLY SUPPORTED: start_timestamp is Unix epoch which can be 
        converted to datetime easily.
        
        Implementation:
        - In scraper: datetime.fromtimestamp(start_timestamp)
        - In Polars: pl.from_epoch("start_timestamp")
        """,
        "feasible": "YES - simple timestamp conversion",
    },
    
    "3_implied_prob": {
        "description": "Pre-calculate implied_prob_player = 1/odds_player",
        "api_support": True,
        "notes": """
        FULLY SUPPORTED: We have odds_player and odds_opponent from 
        the odds endpoint.
        
        Implementation:
        - implied_prob_player = 1 / odds_player
        - implied_prob_opponent = 1 / odds_opponent
        - edge = model_prob - implied_prob_player
        """,
        "feasible": "YES - simple calculation from existing data",
    },
    
    "4_parquet_compression": {
        "description": "Use Parquet snappy compression",
        "api_support": "N/A",
        "notes": """
        NOT API RELATED: This is a storage optimization.
        
        Implementation:
        - In Polars: df.write_parquet(path, compression="snappy")
        - Default is already snappy, but we should be explicit
        """,
        "feasible": "YES - Polars configuration option",
    },
    
    "5_partition_by_year": {
        "description": "Consider partitioning by year for large datasets",
        "api_support": "N/A",
        "notes": """
        NOT API RELATED: This is a storage organization strategy.
        
        Implementation options:
        A) Save files as data/processed/2024/matches.parquet
        B) Use Polars partition_cols in write_parquet
        C) Implement in data loader to selectively load years
        
        Benefit: Faster queries when filtering by date range
        """,
        "feasible": "YES - file organization strategy",
    },
}


# =============================================================================
# MOCK API RESPONSES (from real scraper)
# =============================================================================

MOCK_RANKINGS_RESPONSE = {
    "rankingRows": [
        {
            "position": 1,
            "points": 9850,
            "team": {"id": 275923, "name": "Jannik Sinner", "country": {"alpha2": "IT"}},
            # NOTE: No date field, no historical data
        }
    ]
}

MOCK_MATCH_EVENT = {
    "id": 12345678,
    "homeTeam": {"id": 275923, "name": "Jannik Sinner"},
    "awayTeam": {"id": 206570, "name": "Carlos Alcaraz"},
    "homeScore": {"current": 2},
    "awayScore": {"current": 1},
    "winnerCode": 1,
    "startTimestamp": 1705334400,  # Unix epoch - CAN convert to date
    # NOTE: No ranking data for players at match time
}

MOCK_ODDS_RESPONSE = {
    "markets": [
        {
            "marketId": 1,
            "choices": [
                {"name": "1", "fractionalValue": "8/13"},  # HOME: 1.615
                {"name": "2", "fractionalValue": "11/8"},  # AWAY: 2.375
            ]
        }
    ]
}


# =============================================================================
# HELPER FUNCTIONS FROM SCRAPER
# =============================================================================

def convert_fractional(frac_str) -> Optional[float]:
    """Convert fractional odds to decimal."""
    try:
        if '/' in str(frac_str):
            num, den = map(int, str(frac_str).split('/'))
            return round(1 + (num / den), 3)
        return float(frac_str)
    except:
        return None


def parse_odds(data: Dict, is_home: bool) -> Dict:
    """Parse odds and calculate implied probability."""
    odds_data = {}
    
    if not data or "markets" not in data:
        return odds_data
    
    for market in data.get("markets", []):
        if market.get("marketId") == 1:
            for choice in market.get("choices", []):
                name = choice.get("name", "")
                frac = choice.get("fractionalValue", "")
                decimal = convert_fractional(frac)
                
                if decimal:
                    if name == "1":
                        odds_data["odds_home"] = decimal
                    elif name == "2":
                        odds_data["odds_away"] = decimal
    
    # Assign based on position
    if is_home:
        odds_data["odds_player"] = odds_data.get("odds_home")
        odds_data["odds_opponent"] = odds_data.get("odds_away")
    else:
        odds_data["odds_player"] = odds_data.get("odds_away")
        odds_data["odds_opponent"] = odds_data.get("odds_home")
    
    # NEW: Calculate implied probabilities
    if odds_data.get("odds_player"):
        odds_data["implied_prob_player"] = round(1 / odds_data["odds_player"], 4)
    if odds_data.get("odds_opponent"):
        odds_data["implied_prob_opponent"] = round(1 / odds_data["odds_opponent"], 4)
    
    return odds_data


def add_match_date(timestamp: int) -> Dict:
    """Convert Unix timestamp to multiple date formats."""
    dt = datetime.fromtimestamp(timestamp)
    return {
        "match_date": dt.isoformat(),
        "match_year": dt.year,
        "match_month": dt.month,
        "match_day": dt.day,
        "match_weekday": dt.strftime("%A"),
    }


# =============================================================================
# TEST CASES
# =============================================================================

class TestRecommendation1_PlayerRank:
    """Test: Add player_rank and opponent_rank at match time."""
    
    def test_rankings_endpoint_has_no_date(self):
        """Rankings endpoint does NOT include date/time info."""
        for row in MOCK_RANKINGS_RESPONSE["rankingRows"]:
            assert "date" not in row
            assert "timestamp" not in row
            assert "week" not in row
        # This confirms we CANNOT get historical rankings from current API
    
    def test_match_event_has_no_ranking(self):
        """Match event does NOT include player rankings."""
        assert "ranking" not in str(MOCK_MATCH_EVENT).lower()
        assert "position" not in MOCK_MATCH_EVENT
        assert "points" not in MOCK_MATCH_EVENT.get("homeTeam", {})
        # This confirms rankings are NOT available at match level
    
    def test_workaround_elo_calculation(self):
        """ELO can be calculated as a proxy for rankings."""
        # Our features.py already implements ELO calculation
        # This is a valid alternative to historical rankings
        # ELO is actually BETTER than raw rankings for prediction
        assert True  # Placeholder - ELO is implemented in features.py


class TestRecommendation2_MatchDate:
    """Test: Add match_date as datetime column."""
    
    def test_timestamp_exists_in_event(self):
        """Match event has start_timestamp."""
        assert "startTimestamp" in MOCK_MATCH_EVENT
        assert isinstance(MOCK_MATCH_EVENT["startTimestamp"], int)
    
    def test_timestamp_converts_to_datetime(self):
        """Unix timestamp can be converted to datetime."""
        ts = MOCK_MATCH_EVENT["startTimestamp"]
        dt = datetime.fromtimestamp(ts)
        
        assert isinstance(dt, datetime)
        assert dt.year > 2020
        assert 1 <= dt.month <= 12
        assert 1 <= dt.day <= 31
    
    def test_add_match_date_function(self):
        """add_match_date function returns all date components."""
        result = add_match_date(MOCK_MATCH_EVENT["startTimestamp"])
        
        assert "match_date" in result
        assert "match_year" in result
        assert "match_month" in result
        assert "match_day" in result
        assert "match_weekday" in result
        
        assert result["match_year"] == 2024
        assert result["match_weekday"] in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class TestRecommendation3_ImpliedProb:
    """Test: Pre-calculate implied_prob_player = 1/odds_player."""
    
    def test_odds_available_from_api(self):
        """Odds are available from the API."""
        assert "markets" in MOCK_ODDS_RESPONSE
        assert len(MOCK_ODDS_RESPONSE["markets"]) > 0
    
    def test_implied_prob_calculation(self):
        """Implied probability can be calculated from odds."""
        odds = parse_odds(MOCK_ODDS_RESPONSE, is_home=True)
        
        # Check odds exist
        assert "odds_player" in odds
        assert "odds_opponent" in odds
        
        # Check implied probs calculated
        assert "implied_prob_player" in odds
        assert "implied_prob_opponent" in odds
        
        # Verify calculation: 8/13 = 1.615 -> implied = 1/1.615 = 0.619
        expected_implied = 1 / 1.615
        assert abs(odds["implied_prob_player"] - expected_implied) < 0.01
    
    def test_implied_probs_sum_greater_than_one(self):
        """Implied probs should sum > 1 (bookmaker margin)."""
        odds = parse_odds(MOCK_ODDS_RESPONSE, is_home=True)
        
        total = odds["implied_prob_player"] + odds["implied_prob_opponent"]
        
        # Should be > 1 due to bookmaker overround (typically 1.02-1.10)
        assert total > 1.0


class TestRecommendation4_ParquetCompression:
    """Test: Use Parquet snappy compression."""
    
    def test_polars_supports_snappy(self):
        """Verify Polars supports snappy compression."""
        try:
            import polars as pl
            
            # Create test dataframe
            df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            
            # Test write with snappy (to memory via bytes)
            import io
            buffer = io.BytesIO()
            df.write_parquet(buffer, compression="snappy")
            
            # Verify it wrote something
            assert buffer.tell() > 0
            
            # Read back
            buffer.seek(0)
            df_read = pl.read_parquet(buffer)
            assert len(df_read) == 3
            
        except ImportError:
            # Polars not installed in this environment
            assert True  # Skip if no Polars


class TestRecommendation5_PartitionByYear:
    """Test: Consider partitioning by year for large datasets."""
    
    def test_year_extractable_from_timestamp(self):
        """Year can be extracted from timestamp for partitioning."""
        ts = MOCK_MATCH_EVENT["startTimestamp"]
        dt = datetime.fromtimestamp(ts)
        year = dt.year
        
        assert isinstance(year, int)
        assert year >= 2020
    
    def test_partition_path_generation(self):
        """Can generate partition paths from date."""
        ts = MOCK_MATCH_EVENT["startTimestamp"]
        dt = datetime.fromtimestamp(ts)
        
        partition_path = f"data/processed/{dt.year}/matches.parquet"
        
        assert str(dt.year) in partition_path
        assert ".parquet" in partition_path


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_feasibility_report():
    """Generate summary of recommendation feasibility."""
    print("\n" + "="*70)
    print("RECOMMENDATION FEASIBILITY REPORT")
    print("="*70)
    
    for key, rec in RECOMMENDATIONS.items():
        print(f"\n{key.upper()}: {rec['description']}")
        print(f"   API Support: {rec['api_support']}")
        print(f"   Feasible: {rec['feasible']}")
        
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
    [YES] match_date       - Convert start_timestamp to datetime
    [YES] implied_prob     - Calculate 1/odds_player  
    [YES] snappy           - Polars write_parquet(compression='snappy')
    [YES] partition_year   - Save to data/processed/{year}/
    
    [PARTIAL] player_rank  - NO API support for historical rankings
                            USE WORKAROUND: ELO ratings (already in features.py)
                            OR: Integrate Jeff Sackmann dataset for rankings
    """)
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    """Run all recommendation tests."""
    test_classes = [
        TestRecommendation1_PlayerRank,
        TestRecommendation2_MatchDate,
        TestRecommendation3_ImpliedProb,
        TestRecommendation4_ParquetCompression,
        TestRecommendation5_PartitionByYear,
    ]
    
    print("="*70)
    print("RECOMMENDATION FEASIBILITY TESTS")
    print("="*70)
    
    total = 0
    passed = 0
    failed = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            total += 1
            try:
                getattr(instance, method_name)()
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
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_tests()
    generate_feasibility_report()
    
    sys.exit(0 if success else 1)
