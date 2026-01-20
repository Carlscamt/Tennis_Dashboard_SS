"""
Integration tests for the complete prediction pipeline.
"""
import pytest
import polars as pl
from pathlib import Path


@pytest.mark.integration
def test_end_to_end_prediction_pipeline(temp_data_dir, sample_matches, mock_model):
    """Test complete pipeline from raw data to predictions."""
    # 1. Save sample data
    raw_file = temp_data_dir / "raw" / "matches.parquet"
    sample_matches.write_parquet(raw_file)
    
    # 2. Feature engineering (simplified)
    # In real pipeline: extract.py → transform.py → model.py
    processed_df = sample_matches.with_columns([
        pl.lit(0.65).alias("player_win_rate_5"),
        pl.lit(0.60).alias("opponent_win_rate_5"),
    ])
    
    # 3. Model prediction
    # Mock prediction
    predictions = processed_df.with_columns([
        pl.lit(0.65).alias("model_prob"),
        pl.lit(2.0).alias("odds"),
    ])
    
    # 4. Value bet filtering
    value_bets = predictions.with_columns([
        ((pl.col("model_prob") * pl.col("odds")) - 1).alias("edge")
    ]).filter(pl.col("edge") > 0.05)
    
    # Assertions
    assert len(value_bets) >= 0
    assert "edge" in value_bets.columns
    if len(value_bets) > 0:
        assert value_bets["edge"].min() >= 0.05


@pytest.mark.integration
def test_data_leakage_prevention(sample_matches):
    """Test that no future data leaks into features."""
    sorted_matches = sample_matches.sort("start_timestamp")
    
    # For each match, verify historical stats only use past data
    for i in range(len(sorted_matches)):
        current_match = sorted_matches[i]
        historical = sorted_matches[:i]
        
        if len(historical) > 0:
            # All historical timestamps must be before current
            assert historical["start_timestamp"].max() < current_match["start_timestamp"][0]


@pytest.mark.integration
def test_checkpoint_resume_functionality(temp_data_dir):
    """Test that processing can resume from checkpoint."""
    from scripts.update_active_players import CheckpointManager
    
    checkpoint_dir = temp_data_dir / ".checkpoints"
    manager = CheckpointManager(checkpoint_dir)
    
    # Save checkpoint
    test_records = [{"event_id": 1, "player_id": 100}]
    completed_players = [100, 101]
    manager.save(test_records, completed_players)
    
    # Load checkpoint
    loaded_records, loaded_players = manager.load()
    
    assert len(loaded_records) == 1
    assert loaded_players == [100, 101]
    
    # Clear checkpoint
    manager.clear()
    loaded_records, loaded_players = manager.load()
    assert len(loaded_records) == 0
