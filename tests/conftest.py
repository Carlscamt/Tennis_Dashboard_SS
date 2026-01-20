"""
Pytest configuration and fixtures for Tennis Betting ML Pipeline.
"""
import pytest
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta


@pytest.fixture
def sample_matches():
    """Sample match data for testing."""
    return pl.DataFrame({
        "event_id": [1, 2, 3, 4, 5],
        "player_id": [100, 100, 101, 101, 100],
        "opponent_id": [101, 102, 100, 102, 101],
        "player_won": [True, False, False, True, True],
        "player_sets": [2, 1, 0, 2, 2],
        "opponent_sets": [0, 2, 2, 1, 1],
        "start_timestamp": [
            int((datetime.now() - timedelta(days=i)).timestamp())
            for i in range(5, 0, -1)
        ],
        "match_date": [
            (datetime.now() - timedelta(days=i)).date().isoformat()
            for i in range(5, 0, -1)
        ],
        "ground_type": ["Hard", "Clay", "Hard", "Grass", "Hard"],
    })


@pytest.fixture
def sample_player_stats():
    """Sample player statistics."""
    return {
        "player_id": 100,
        "total_matches": 50,
        "wins": 35,
        "win_rate": 0.70,
        "avg_sets_won": 1.8,
        "hard_court_win_rate": 0.75,
        "clay_court_win_rate": 0.60,
    }


@pytest.fixture
def sample_odds():
    """Sample betting odds data."""
    return pl.DataFrame({
        "event_id": [1, 2, 3],
        "odds": [2.10, 1.50, 2.50],
        "bookmaker": ["Bet365", "Bet365", "Bet365"],
    })


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "predictions").mkdir(parents=True)
    return data_dir


@pytest.fixture
def mock_model():
    """Mock XGBoost model for testing."""
    class MockModel:
        def predict_proba(self, X):
            # Return dummy probabilities
            import numpy as np
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.array([[0.4, 0.6]] * n_samples)
        
        def predict(self, X):
            return self.predict_proba(X)[:, 1] > 0.5
    
    return MockModel()
