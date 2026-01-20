"""
Tests for data leakage prevention.
Critical tests to ensure ML validity.
"""
import pytest
import polars as pl
from datetime import date, datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transform.leakage_guard import (
    validate_temporal_order,
    create_train_test_split,
    assert_no_leakage,
    LeakageError,
)


def create_sample_data(n: int = 100) -> pl.LazyFrame:
    """Create sample data for testing."""
    import random
    
    timestamps = sorted([
        int(datetime(2024, 1, 1).timestamp()) + i * 86400
        for i in range(n)
    ])
    
    return pl.LazyFrame({
        "event_id": list(range(n)),
        "player_id": [random.randint(1, 10) for _ in range(n)],
        "opponent_id": [random.randint(1, 10) for _ in range(n)],
        "player_won": [random.choice([True, False]) for _ in range(n)],
        "start_timestamp": timestamps,
    })


class TestTemporalOrder:
    """Test temporal ordering validation."""
    
    def test_sorted_data_passes(self):
        """Sorted data should pass validation."""
        df = create_sample_data()
        assert validate_temporal_order(df) is True
    
    def test_unsorted_data_fails(self):
        """Unsorted data should raise LeakageError."""
        df = pl.LazyFrame({
            "start_timestamp": [3, 1, 2],
            "event_id": [1, 2, 3],
        })
        
        with pytest.raises(LeakageError):
            validate_temporal_order(df)


class TestTrainTestSplit:
    """Test temporal train/test splitting."""
    
    def test_split_creates_two_sets(self):
        """Split should create non-empty train and test sets."""
        df = create_sample_data(100)
        cutoff = date(2024, 2, 1)
        
        train, test = create_train_test_split(df, cutoff)
        
        train_count = train.select(pl.len()).collect().item()
        test_count = test.select(pl.len()).collect().item()
        
        assert train_count > 0
        assert test_count > 0
        assert train_count + test_count == 100
    
    def test_train_before_test(self):
        """All train data should be before all test data."""
        df = create_sample_data(100)
        cutoff = date(2024, 2, 1)
        
        train, test = create_train_test_split(df, cutoff)
        
        train_max = train.select(pl.col("start_timestamp").max()).collect().item()
        test_min = test.select(pl.col("start_timestamp").min()).collect().item()
        
        assert train_max < test_min


class TestNoLeakage:
    """Test leakage assertion."""
    
    def test_valid_split_passes(self):
        """Valid split should pass assertion."""
        df = create_sample_data(100)
        cutoff = date(2024, 2, 1)
        
        train, test = create_train_test_split(df, cutoff)
        
        # Should not raise
        assert_no_leakage(train, test)
    
    def test_overlapping_data_fails(self):
        """Overlapping train/test should fail."""
        # Create overlapping data
        train = pl.LazyFrame({
            "start_timestamp": [1, 2, 3, 4, 5],
        })
        test = pl.LazyFrame({
            "start_timestamp": [3, 4, 5, 6, 7],  # Overlaps at 3, 4, 5
        })
        
        with pytest.raises(LeakageError):
            assert_no_leakage(train, test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
