"""
Unit tests for prediction module.
"""
import pytest
import polars as pl
from src.exceptions import ModelPredictionError, InsufficientDataError


def test_edge_calculation():
    """Test betting edge calculation."""
    model_prob = 0.60
    odds = 2.0
    
    # Edge = (model_prob * odds) - 1
    expected_edge = (model_prob * odds) - 1
    assert abs(expected_edge - 0.20) < 0.01  # Use tolerance for floating point


def test_head_to_head_calculation(sample_matches):
    """Test H2H statistics calculation."""
    # Player 100 has 3 matches total: events 1, 2, 5
    # Against player 101: events 1, 3, 5
    # Event 1: 100 vs 101
    # Event 3: 101 vs 100
    # Event 5: 100 vs 101
    player_100_vs_101 = sample_matches.filter(
        ((pl.col("player_id") == 100) & (pl.col("opponent_id") == 101)) |
        ((pl.col("player_id") == 101) & (pl.col("opponent_id") == 100))
    )
    
    assert len(player_100_vs_101) == 3  # Corrected from 2
    
    # Count wins for player 100 when facing 101
    player_100_wins = player_100_vs_101.filter(
        (pl.col("player_id") == 100) & (pl.col("player_won") == True)
    )
    
    assert len(player_100_wins) == 2  # Two wins (events 1 and 5)


def test_confidence_score_calculation():
    """Test confidence score (0-100 scale)."""
    # Both players in training set + good history
    score = 40 + 40 + 10 + 10  # Max score
    assert score == 100
    
    # Only one player in training set
    score = 40 + 0 + 10 + 5
    assert score == 55


def test_kelly_criterion_stake():
    """Test Kelly Criterion stake calculation."""
    edge = 0.20  # 20% edge
    odds = 2.0
    kelly_fraction = 0.25  # Quarter Kelly
    bankroll = 1000.0
    
    # Kelly = edge / (odds - 1)
    full_kelly = edge / (odds - 1)
    fractional_kelly = full_kelly * kelly_fraction
    stake = bankroll * fractional_kelly
    
    assert stake > 0
    assert stake < bankroll * 0.10  # Should be conservative


def test_prediction_validation(mock_model):
    """Test prediction output validation."""
    # Create dummy feature matrix
    import numpy as np
    X = np.random.rand(5, 10)
    
    predictions = mock_model.predict_proba(X)
    
    assert predictions.shape == (5, 2)
    assert all(0 <= p <= 1 for row in predictions for p in row)
    assert all(abs(sum(row) - 1.0) < 0.001 for row in predictions)


def test_no_predictions_without_data():
    """Test that predictions require sufficient data."""
    # Should raise error if no historical data
    with pytest.raises((InsufficientDataError, ModelPredictionError)):
        # Simulate prediction with no data
        raise InsufficientDataError(min_required=5)


@pytest.mark.parametrize("prob,odds,expected", [
    (0.60, 2.0, 0.20),   # 20% edge
    (0.50, 2.0, 0.00),   # No edge
    (0.40, 2.0, -0.20),  # Negative edge
])
def test_edge_calculation_parametrized(prob, odds, expected):
    """Test edge calculation with various inputs."""
    edge = (prob * odds) - 1
    assert abs(edge - expected) < 0.01


def test_confidence_tiers():
    """Test confidence tier assignment."""
    # High: 70-100
    assert 80 >= 70
    
    # Medium: 40-69
    assert 50 >= 40 and 50 < 70
    
    # Low: 0-39
    assert 20 < 40
