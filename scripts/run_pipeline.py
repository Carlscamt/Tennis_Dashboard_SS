"""
Full ETL + Training Pipeline

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --data-only
    python scripts/run_pipeline.py --train-only
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import polars as pl
from datetime import date

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    FEATURES, MODEL, BETTING
)
from src.extract import load_all_parquet_files
from src.extract.data_loader import prepare_base_dataset, get_dataset_stats
from src.transform import FeatureEngineer, create_train_test_split, DataValidator
from src.transform.leakage_guard import validate_temporal_order, assert_no_leakage
from src.model import ModelTrainer, ModelRegistry
from src.utils import setup_logging

logger = setup_logging()


def run_data_pipeline(raw_dir: Path, output_dir: Path) -> pl.LazyFrame:
    """
    Run data extraction and feature engineering.
    
    Args:
        raw_dir: Directory with raw parquet files
        output_dir: Directory to save processed data
        
    Returns:
        Processed LazyFrame
    """
    logger.info("=" * 60)
    logger.info("DATA PIPELINE")
    logger.info("=" * 60)
    
    # Load raw data
    logger.info(f"Loading data from {raw_dir}")
    df = load_all_parquet_files(raw_dir)
    
    # Prepare base dataset
    df = prepare_base_dataset(df)
    
    # Validate temporal order
    validate_temporal_order(df)
    
    # Get stats
    stats = get_dataset_stats(df)
    logger.info(f"Loaded {stats['total_matches']:,} matches")
    logger.info(f"Date range: {stats['earliest_match']} to {stats['latest_match']}")
    if "odds_coverage" in stats:
        logger.info(f"Odds coverage: {stats['odds_coverage']:.1%}")
    
    # Feature engineering
    logger.info("Engineering features...")
    fe = FeatureEngineer(
        rolling_windows=FEATURES.rolling_windows,
        min_matches=FEATURES.min_matches_for_stats,
        elo_k=FEATURES.elo_k_factor,
    )
    df = fe.add_all_features(df)
    
    # Validate data quality
    validator = DataValidator(min_odds_coverage=MODEL.min_odds_coverage)
    if not validator.validate_all(df):
        logger.warning("Data validation failed - check warnings above")
    
    # Save processed data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features_dataset.parquet"
    df.collect().write_parquet(output_path)
    logger.info(f"Saved processed data to {output_path}")
    
    return df


def run_training_pipeline(
    data_path: Path,
    models_dir: Path,
    cutoff_date: date = None
) -> None:
    """
    Run model training pipeline.
    
    Args:
        data_path: Path to processed dataset
        models_dir: Directory to save models
        cutoff_date: Train/test split date
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)
    
    cutoff_date = cutoff_date or MODEL.train_cutoff_date
    
    # Load processed data
    logger.info(f"Loading data from {data_path}")
    df = pl.scan_parquet(data_path)
    
    # Train/test split
    train_df, test_df = create_train_test_split(df, cutoff_date)
    
    # Assert no leakage
    assert_no_leakage(train_df, test_df)
    
    # Collect to memory
    train_data = train_df.collect()
    test_data = test_df.collect()
    
    logger.info(f"Train: {len(train_data):,} samples")
    logger.info(f"Test: {len(test_data):,} samples")
    
    # Get feature columns
    fe = FeatureEngineer()
    feature_cols = fe.get_feature_columns(train_df)
    logger.info(f"Features: {len(feature_cols)}")
    
    # Filter to features that exist AND are numeric
    existing_cols = [c for c in feature_cols if c in train_data.columns]
    
    # Filter to numeric types only
    numeric_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64, pl.Boolean]
    numeric_cols = [
        c for c in existing_cols 
        if train_data[c].dtype in numeric_types
    ]
    logger.info(f"Using {len(numeric_cols)} numeric features")
    
    # Train model
    trainer = ModelTrainer(params=MODEL.xgb_params, calibrate=True)
    result = trainer.train(
        train_data,
        feature_cols=numeric_cols,
        eval_df=test_data
    )
    
    # Log metrics
    logger.info("Metrics:")
    for key, value in result.metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Top features
    logger.info("Top 10 Features:")
    for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10]):
        logger.info(f"  {i+1}. {feat}: {imp:.4f}")
    
    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgboost_model"
    trainer.save(model_path)
    
    # Register model
    registry = ModelRegistry(models_dir)
    version = registry.register(
        model_path,
        metrics=result.metrics,
        description=f"Trained on data up to {cutoff_date}",
        set_active=True
    )
    
    logger.info(f"Model registered: {version}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Tennis Betting ML Pipeline")
    parser.add_argument("--data-only", action="store_true", help="Run data pipeline only")
    parser.add_argument("--train-only", action="store_true", help="Run training only")
    parser.add_argument("--cutoff", type=str, help="Train/test cutoff date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    cutoff = None
    if args.cutoff:
        cutoff = date.fromisoformat(args.cutoff)
    
    if args.train_only:
        data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
        if not data_path.exists():
            logger.error(f"Processed data not found: {data_path}")
            logger.error("Run with --data-only first")
            sys.exit(1)
        run_training_pipeline(data_path, MODELS_DIR, cutoff)
    elif args.data_only:
        run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    else:
        # Full pipeline
        run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)
        data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
        run_training_pipeline(data_path, MODELS_DIR, cutoff)
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
