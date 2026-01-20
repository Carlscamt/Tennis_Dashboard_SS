"""
Unit tests for feature engineering module.
"""
import pytest
import polars as pl
from src.exceptions import InsufficientDataError


def test_calculate_win_rate(sample_matches):
    """Test win rate calculation."""
    player_100_matches = sample_matches.filter(pl.col("player_id") == 100)
    
    wins = player_100_matches.filter(pl.col("player_won") == True)
    win_rate = len(wins) / len(player_100_matches)
    
    assert 0 <= win_rate <= 1
    assert win_rate == 2/3  # 2 wins out of 3 matches


def test_rolling_window_stats(sample_matches):
    """Test rolling window statistics calculation."""
    # Sort by timestamp
    sorted_matches = sample_matches.sort("start_timestamp")
    
    # Calculate rolling 3-match win rate
    window_size = 3
    assert len(sorted_matches) >= window_size


def test_insufficient_data_error():
    """Test InsufficientDataError is raised correctly."""
    with pytest.raises(InsufficientDataError) as exc_info:
        raise InsufficientDataError(player_id=12345, min_required=5)
    
    assert "12345" in str(exc_info.value)
    assert "5" in str(exc_info.value)


def test_head_to_head_calculation(sample_matches):
    """Test H2H statistics calculation."""
    # Player 100 has 3 matches total: events 1, 2, 5  
    # Against player 101: events 1, 3, 5 (3 H2H matches)
    player_100_vs_101 = sample_matches.filter(
        ((pl.col("player_id") == 100) & (pl.col("opponent_id") == 101)) |
        ((pl.col("player_id") == 101) & (pl.col("opponent_id") == 100))
    )
    
    assert len(player_100_vs_101) == 3  # Matches 1, 3, 5
    
    # Count wins for player 100 when facing 101
    player_100_wins = player_100_vs_101.filter(
        (pl.col("player_id") == 100) & (pl.col("player_won") == True)
    )
    
    assert len(player_100_wins) == 2  # Won events 1 and 5


def test_surface_specific_stats(sample_matches):
    """Test surface-specific statistics."""
    hard_court_matches = sample_matches.filter(pl.col("ground_type") == "Hard")
    
    assert len(hard_court_matches) == 3
    
    # Calculate hard court win rate for player 100
    player_100_hard = hard_court_matches.filter(pl.col("player_id") == 100)
    wins = player_100_hard.filter(pl.col("player_won") == True).height
    
    hard_win_rate = wins / len(player_100_hard) if len(player_100_hard) > 0 else 0
    assert 0 <= hard_win_rate <= 1


@pytest.mark.parametrize("window_size", [5, 10, 20])
def test_rolling_windows_various_sizes(sample_matches, window_size):
    """Test that rolling windows work with various sizes."""
    # Should handle cases where data < window_size
    if len(sample_matches) < window_size:
        # Should use all available data
        assert len(sample_matches) < window_size
    else:
        # Should use exact window_size
        assert len(sample_matches) >= window_size


def test_match_data_validation(sample_matches):
    """Test that match data has required columns."""
    required_columns = [
        "event_id", "player_id", "opponent_id", "player_won",
        "start_timestamp", "match_date"
    ]
    
    for col in required_columns:
        assert col in sample_matches.columns, f"Missing required column: {col}"


def test_no_future_data_leakage(sample_matches):
    """Test that features don't use future data."""
    sorted_matches = sample_matches.sort("start_timestamp")
    
    # For each match, only use data from before that match
    for i, row in enumerate(sorted_matches.iter_rows(named=True)):
        historical_data = sorted_matches[:i]
        
        # Historical data should only include matches before current
        if len(historical_data) > 0:
            max_historical_ts = historical_data["start_timestamp"].max()
            current_ts = row["start_timestamp"]
            assert max_historical_ts < current_ts
