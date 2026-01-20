"""
Test cases for Colab Scraper Functions.

This file tests the scraper logic WITHOUT making actual API calls.
Run these tests locally before running the full scraper in Colab.

Usage:
    python Scrappers/test_scraper_functions.py
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

# Mock data for testing (simulates API responses)
# =============================================================================

MOCK_RANKINGS_RESPONSE = {
    "rankingRows": [
        {
            "position": 1,
            "points": 9850,
            "team": {
                "id": 275923,
                "name": "Jannik Sinner",
                "country": {"alpha2": "IT"}
            }
        },
        {
            "position": 2,
            "points": 8400,
            "team": {
                "id": 206570,
                "name": "Carlos Alcaraz",
                "country": {"alpha2": "ES"}
            }
        },
    ]
}

MOCK_MATCH_EVENT = {
    "id": 12345678,
    "homeTeam": {"id": 275923, "name": "Jannik Sinner"},
    "awayTeam": {"id": 206570, "name": "Carlos Alcaraz"},
    "homeScore": {"current": 2, "period1": 6, "period2": 7, "period3": None},
    "awayScore": {"current": 1, "period1": 4, "period2": 5, "period3": None},
    "winnerCode": 1,  # 1=home wins
    "groundType": "Hardcourt outdoor",
    "startTimestamp": 1705334400,
    "status": {"type": "finished"},
    "tournament": {
        "uniqueTournament": {
            "id": 2671,
            "name": "Australian Open"
        }
    },
    "roundInfo": {"name": "Final"}
}

MOCK_STATS_RESPONSE = {
    "statistics": [
        {
            "period": "ALL",
            "groups": [
                {
                    "groupName": "Service",
                    "statisticsItems": [
                        {
                            "key": "aces",
                            "home": "12",
                            "away": "8",
                            "homeValue": 12,
                            "awayValue": 8
                        },
                        {
                            "key": "doubleFaults",
                            "home": "2",
                            "away": "4",
                            "homeValue": 2,
                            "awayValue": 4
                        },
                        {
                            "key": "firstServeAccuracy",
                            "home": "65%",
                            "away": "62%",
                            "homeValue": 65,
                            "awayValue": 62
                        },
                    ]
                },
                {
                    "groupName": "Points",
                    "statisticsItems": [
                        {
                            "key": "pointsTotal",
                            "home": "120",
                            "away": "105",
                            "homeValue": 120,
                            "awayValue": 105
                        }
                    ]
                }
            ]
        }
    ]
}

MOCK_ODDS_RESPONSE = {
    "markets": [
        {
            "marketId": 1,
            "choices": [
                {"name": "1", "fractionalValue": "8/13"},  # Home
                {"name": "2", "fractionalValue": "11/8"},  # Away
            ]
        }
    ]
}


# =============================================================================
# FUNCTIONS UNDER TEST (copied from notebook for local testing)
# =============================================================================

def convert_fractional(frac_str) -> Optional[float]:
    """
    Convert fractional odds to decimal.
    Example: '8/13' -> 1.615
    """
    try:
        if '/' in str(frac_str):
            num, den = map(int, str(frac_str).split('/'))
            return round(1 + (num / den), 3)
        return float(frac_str)
    except:
        return None


def parse_rankings(data: Dict) -> List[Dict]:
    """Parse rankings response."""
    if not data or "rankingRows" not in data:
        return []
    
    players = []
    for row in data["rankingRows"]:
        team = row.get("team", {})
        players.append({
            "position": row.get("position"),
            "player_id": team.get("id"),
            "name": team.get("name"),
            "country": team.get("country", {}).get("alpha2"),
            "points": row.get("points"),
        })
    return players


def process_match(event: Dict, player_id: int) -> tuple:
    """
    Convert raw match event to player-centric format.
    """
    home_id = event.get("homeTeam", {}).get("id")
    is_home = (home_id == player_id)
    
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    home_score = event.get("homeScore", {})
    away_score = event.get("awayScore", {})
    tournament = event.get("tournament", {}).get("uniqueTournament", {})
    
    winner = event.get("winnerCode")  # 1=home, 2=away, 3=draw
    player_won = (winner == 1 and is_home) or (winner == 2 and not is_home)
    
    match_data = {
        "event_id": event.get("id"),
        "player_id": player_id,
        "player_name": home.get("name") if is_home else away.get("name"),
        "opponent_id": away.get("id") if is_home else home.get("id"),
        "opponent_name": away.get("name") if is_home else home.get("name"),
        "player_won": player_won,
        "is_home": is_home,
        "player_sets": home_score.get("current", 0) if is_home else away_score.get("current", 0),
        "opponent_sets": away_score.get("current", 0) if is_home else home_score.get("current", 0),
        "tournament_id": tournament.get("id"),
        "tournament_name": tournament.get("name"),
        "round_name": event.get("roundInfo", {}).get("name"),
        "ground_type": event.get("groundType"),
        "start_timestamp": event.get("startTimestamp"),
        "status": event.get("status", {}).get("type"),
    }
    
    # Set scores (up to 5 sets)
    for i in range(1, 6):
        match_data[f"player_set{i}"] = home_score.get(f"period{i}") if is_home else away_score.get(f"period{i}")
        match_data[f"opponent_set{i}"] = away_score.get(f"period{i}") if is_home else home_score.get(f"period{i}")
    
    return match_data, "home" if is_home else "away", "away" if is_home else "home"


def flatten_stats(stats_data: Dict, player_prefix: str, opponent_prefix: str) -> Dict:
    """
    Flatten nested statistics into player-centric columns.
    """
    if not stats_data or "statistics" not in stats_data:
        return {}
    
    result = {}
    for period_data in stats_data["statistics"]:
        period = period_data.get("period", "ALL")
        prefix = "" if period == "ALL" else f"{period.lower()}_"
        
        for group in period_data.get("groups", []):
            group_name = group.get("groupName", "").lower().replace(" ", "_")
            
            for item in group.get("statisticsItems", []):
                key = item.get("key", "").lower()
                stat_name = f"{prefix}{group_name}_{key}"
                
                # Raw values (can be strings like "65%")
                result[f"player_{stat_name}"] = item.get(player_prefix)
                result[f"opponent_{stat_name}"] = item.get(opponent_prefix)
                
                # Numeric values
                if f"{player_prefix}Value" in item:
                    result[f"player_{stat_name}_value"] = item.get(f"{player_prefix}Value")
                    result[f"opponent_{stat_name}_value"] = item.get(f"{opponent_prefix}Value")
    
    return result


def parse_odds(data: Dict, is_home: bool) -> Dict:
    """Parse odds response and assign to player/opponent."""
    odds_data = {}
    
    if not data or "markets" not in data:
        return odds_data
    
    for market in data.get("markets", []):
        market_id = market.get("marketId")
        
        # Market 1 = Match Winner
        if market_id == 1:
            for choice in market.get("choices", []):
                name = choice.get("name", "")
                frac = choice.get("fractionalValue", "")
                decimal = convert_fractional(frac)
                
                if decimal:
                    if name == "1":  # Home
                        odds_data["odds_home"] = decimal
                    elif name == "2":  # Away
                        odds_data["odds_away"] = decimal
    
    # Assign to player/opponent based on position
    if is_home:
        odds_data["odds_player"] = odds_data.get("odds_home")
        odds_data["odds_opponent"] = odds_data.get("odds_away")
    else:
        odds_data["odds_player"] = odds_data.get("odds_away")
        odds_data["odds_opponent"] = odds_data.get("odds_home")
    
    return odds_data


# =============================================================================
# TEST CASES
# =============================================================================

class TestConvertFractional:
    """Test fractional to decimal odds conversion."""
    
    def test_simple_fraction(self):
        """8/13 should give about 1.615"""
        result = convert_fractional("8/13")
        assert result is not None
        assert abs(result - 1.615) < 0.01
    
    def test_evens(self):
        """1/1 should give 2.0"""
        result = convert_fractional("1/1")
        assert result == 2.0
    
    def test_heavy_favorite(self):
        """1/10 should give 1.1"""
        result = convert_fractional("1/10")
        assert result == 1.1
    
    def test_underdog(self):
        """11/8 should give about 2.375"""
        result = convert_fractional("11/8")
        assert result is not None
        assert abs(result - 2.375) < 0.01
    
    def test_already_decimal(self):
        """Decimal input should return same value"""
        result = convert_fractional("2.5")
        assert result == 2.5
    
    def test_invalid_input(self):
        """Invalid input should return None"""
        assert convert_fractional("invalid") is None
        assert convert_fractional("") is None
        assert convert_fractional(None) is None
    
    def test_zero_denominator(self):
        """Zero denominator should return None (no crash)"""
        result = convert_fractional("1/0")
        assert result is None


class TestParseRankings:
    """Test rankings parsing."""
    
    def test_parse_valid_rankings(self):
        """Should parse rankings correctly."""
        players = parse_rankings(MOCK_RANKINGS_RESPONSE)
        
        assert len(players) == 2
        assert players[0]["position"] == 1
        assert players[0]["player_id"] == 275923
        assert players[0]["name"] == "Jannik Sinner"
        assert players[0]["country"] == "IT"
        assert players[0]["points"] == 9850
    
    def test_empty_response(self):
        """Should handle empty response."""
        assert parse_rankings({}) == []
        assert parse_rankings(None) == []
    
    def test_missing_fields(self):
        """Should handle missing fields gracefully."""
        data = {"rankingRows": [{"position": 1, "team": {}}]}
        players = parse_rankings(data)
        assert len(players) == 1
        assert players[0]["name"] is None


class TestProcessMatch:
    """Test match processing to player-centric format."""
    
    def test_home_player_wins(self):
        """Home player winning should be processed correctly."""
        match, player_pref, opp_pref = process_match(MOCK_MATCH_EVENT, 275923)
        
        assert match["event_id"] == 12345678
        assert match["player_id"] == 275923
        assert match["player_name"] == "Jannik Sinner"
        assert match["opponent_name"] == "Carlos Alcaraz"
        assert match["player_won"] == True
        assert match["is_home"] == True
        assert match["player_sets"] == 2
        assert match["opponent_sets"] == 1
        assert player_pref == "home"
        assert opp_pref == "away"
    
    def test_away_player_loses(self):
        """Away player (loser) should be processed correctly."""
        match, player_pref, opp_pref = process_match(MOCK_MATCH_EVENT, 206570)
        
        assert match["player_id"] == 206570
        assert match["player_name"] == "Carlos Alcaraz"
        assert match["opponent_name"] == "Jannik Sinner"
        assert match["player_won"] == False
        assert match["is_home"] == False
        assert match["player_sets"] == 1
        assert match["opponent_sets"] == 2
        assert player_pref == "away"
        assert opp_pref == "home"
    
    def test_set_scores(self):
        """Set-by-set scores should be extracted."""
        match, _, _ = process_match(MOCK_MATCH_EVENT, 275923)
        
        assert match["player_set1"] == 6
        assert match["player_set2"] == 7
        assert match["opponent_set1"] == 4
        assert match["opponent_set2"] == 5
    
    def test_tournament_info(self):
        """Tournament info should be extracted."""
        match, _, _ = process_match(MOCK_MATCH_EVENT, 275923)
        
        assert match["tournament_name"] == "Australian Open"
        assert match["round_name"] == "Final"
        assert match["ground_type"] == "Hardcourt outdoor"


class TestFlattenStats:
    """Test statistics flattening."""
    
    def test_flatten_home_player(self):
        """Stats should be flattened with player/opponent perspective."""
        stats = flatten_stats(MOCK_STATS_RESPONSE, "home", "away")
        
        # Check service stats
        assert stats["player_service_aces"] == "12"
        assert stats["opponent_service_aces"] == "8"
        assert stats["player_service_aces_value"] == 12
        assert stats["opponent_service_aces_value"] == 8
        
        # Check double faults
        assert stats["player_service_doublefaults_value"] == 2
        assert stats["opponent_service_doublefaults_value"] == 4
    
    def test_flatten_away_player(self):
        """Away player should get flipped perspective."""
        stats = flatten_stats(MOCK_STATS_RESPONSE, "away", "home")
        
        assert stats["player_service_aces_value"] == 8  # Was "away"
        assert stats["opponent_service_aces_value"] == 12  # Was "home"
    
    def test_empty_stats(self):
        """Empty stats should return empty dict."""
        assert flatten_stats({}, "home", "away") == {}
        assert flatten_stats(None, "home", "away") == {}


class TestParseOdds:
    """Test odds parsing."""
    
    def test_parse_home_player(self):
        """Home player should get correct odds assignment."""
        odds = parse_odds(MOCK_ODDS_RESPONSE, is_home=True)
        
        # 8/13 = 1.615 for home
        assert odds["odds_home"] is not None
        assert abs(odds["odds_home"] - 1.615) < 0.01
        
        # 11/8 = 2.375 for away
        assert abs(odds["odds_away"] - 2.375) < 0.01
        
        # Player is home
        assert odds["odds_player"] == odds["odds_home"]
        assert odds["odds_opponent"] == odds["odds_away"]
    
    def test_parse_away_player(self):
        """Away player should get flipped odds."""
        odds = parse_odds(MOCK_ODDS_RESPONSE, is_home=False)
        
        # Player is away
        assert odds["odds_player"] == odds["odds_away"]
        assert odds["odds_opponent"] == odds["odds_home"]
    
    def test_empty_odds(self):
        """Empty odds should return empty dict."""
        assert parse_odds({}, True) == {}
        assert parse_odds(None, True) == {}


class TestDataIntegrity:
    """Test data integrity and schema."""
    
    def test_required_columns_present(self):
        """All required columns for ML should be present."""
        match, _, _ = process_match(MOCK_MATCH_EVENT, 275923)
        
        required = [
            "event_id", "player_id", "opponent_id",
            "player_won", "start_timestamp"
        ]
        
        for col in required:
            assert col in match, f"Missing required column: {col}"
    
    def test_no_null_ids(self):
        """IDs should never be null."""
        match, _, _ = process_match(MOCK_MATCH_EVENT, 275923)
        
        assert match["event_id"] is not None
        assert match["player_id"] is not None
        assert match["opponent_id"] is not None
    
    def test_timestamp_is_integer(self):
        """Timestamp should be integer (Unix epoch)."""
        match, _, _ = process_match(MOCK_MATCH_EVENT, 275923)
        
        assert isinstance(match["start_timestamp"], int)


# =============================================================================
# DB DESIGN EFFICIENCY ANALYSIS
# =============================================================================

def analyze_db_design():
    """
    Analyze the database (parquet) design for ML efficiency.
    """
    print("\n" + "="*60)
    print("DATABASE DESIGN ANALYSIS")
    print("="*60)
    
    # Simulate a full match record
    match, player_pref, opp_pref = process_match(MOCK_MATCH_EVENT, 275923)
    stats = flatten_stats(MOCK_STATS_RESPONSE, player_pref, opp_pref)
    odds = parse_odds(MOCK_ODDS_RESPONSE, is_home=True)
    
    full_record = {**match, **stats, **odds}
    
    print(f"\n[RECORD STRUCTURE]")
    print(f"   Total columns: {len(full_record)}")
    
    # Categorize columns
    id_cols = [k for k in full_record if 'id' in k.lower()]
    stat_cols = [k for k in full_record if 'value' in k.lower() or 'service' in k.lower()]
    meta_cols = [k for k in full_record if k not in id_cols + stat_cols]
    
    print(f"   ID columns: {len(id_cols)}")
    print(f"   Stat columns: {len(stat_cols)}")
    print(f"   Meta columns: {len(meta_cols)}")
    
    # Check for potential issues
    print("\n[POTENTIAL ISSUES]")
    
    issues = []
    
    # Issue 1: Missing historical rank
    if "player_rank" not in full_record:
        issues.append("[X] No historical player rank at match time")
    
    # Issue 2: No implied probability pre-calculated
    if "implied_prob_player" not in full_record:
        issues.append("[!] Consider adding pre-calculated implied_prob columns")
    
    # Issue 3: Timestamps are Unix (need conversion for analysis)
    if isinstance(full_record.get("start_timestamp"), int):
        issues.append("[i] Timestamps are Unix epoch - consider adding date column")
    
    # Issue 4: String stats instead of numeric
    for k, v in full_record.items():
        if isinstance(v, str) and '%' in str(v):
            issues.append(f"[!] {k} is string with % - strip and convert to float")
            break
    
    for issue in issues:
        print(f"   {issue}")
    
    # Recommendations
    print("\n[RECOMMENDATIONS]")
    recommendations = [
        "1. Add 'player_rank' and 'opponent_rank' at match time",
        "2. Add 'match_date' as proper datetime column",
        "3. Add 'implied_prob_player' = 1/odds_player",
        "4. Use Parquet compression (snappy) for storage efficiency",
        "5. Consider partitioning by year for large datasets",
    ]
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "="*60)


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    """Run all tests."""
    test_classes = [
        TestConvertFractional,
        TestParseRankings,
        TestProcessMatch,
        TestFlattenStats,
        TestParseOdds,
        TestDataIntegrity,
    ]
    
    print("="*60)
    print("SCRAPER FUNCTION TESTS")
    print("="*60)
    
    total = 0
    passed = 0
    failed = []
    
    for test_class in test_classes:
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            total += 1
            try:
                getattr(instance, method_name)()
                print(f"[PASS] {test_class.__name__}.{method_name}")
                passed += 1
            except AssertionError as e:
                print(f"[FAIL] {test_class.__name__}.{method_name}: {e}")
                failed.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"[FAIL] {test_class.__name__}.{method_name}: {type(e).__name__}: {e}")
                failed.append(f"{test_class.__name__}.{method_name}")
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} tests passed")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    print("="*60)
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_tests()
    analyze_db_design()
    
    sys.exit(0 if success else 1)
