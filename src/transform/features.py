"""
Feature engineering using Polars with strict temporal ordering.
All rolling features use ONLY past data to prevent leakage.
"""
import polars as pl
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineer:
    """
    Feature engineering pipeline using Polars.
    All features respect temporal ordering to prevent data leakage.
    """
    
    rolling_windows: tuple = (5, 10, 20, 50)
    elo_k: float = 32.0
    elo_initial: float = 1500.0
    min_matches: int = 5
    
    def add_all_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add all features to the dataset.
        
        Args:
            df: LazyFrame sorted by start_timestamp
            
        Returns:
            LazyFrame with all features added
        """
        # Ensure sorted by time
        df = df.sort("start_timestamp")
        
        # Add features in order
        df = self.add_rolling_win_rate(df)
        df = self.add_rolling_serve_stats(df)
        df = self.add_days_since_last_match(df)
        df = self.add_h2h_features(df)
        df = self.add_surface_features(df)
        df = self.add_round_features(df)
        df = self.add_odds_features(df)
        
        return df
    
    def add_rolling_win_rate(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add rolling win rate features per player.
        Uses shift(1) to exclude current match.
        """
        for window in self.rolling_windows:
            df = df.with_columns([
                pl.col("player_won")
                .cast(pl.Float64)
                .shift(1)  # Exclude current match
                .rolling_mean(window_size=window, min_periods=self.min_matches)
                .over("player_id")
                .alias(f"player_win_rate_{window}")
            ])
        
        return df
    
    def add_rolling_serve_stats(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add rolling serve statistics per player.
        """
        serve_cols = [
            ("player_service_aces_value", "aces"),
            ("player_service_doublefaults_value", "double_faults"),
            ("player_service_firstserveaccuracy_value", "first_serve_pct"),
            ("player_service_firstservepointsaccuracy_value", "first_serve_won"),
        ]
        
        for window in [10, 20]:
            for col_name, short_name in serve_cols:
                if col_name in df.collect_schema().names():
                    df = df.with_columns([
                        pl.col(col_name)
                        .cast(pl.Float64)
                        .shift(1)
                        .rolling_mean(window_size=window, min_periods=self.min_matches)
                        .over("player_id")
                        .alias(f"player_{short_name}_avg_{window}")
                    ])
        
        return df
    
    def add_days_since_last_match(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add days since player's last match (fatigue/rust indicator).
        """
        return df.with_columns([
            (
                pl.col("match_date") - 
                pl.col("match_date").shift(1).over("player_id")
            ).dt.total_days().alias("days_since_last_match")
        ])
    
    def add_h2h_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add head-to-head record between players.
        Calculated using only past encounters.
        """
        # Create matchup key (sorted player IDs for consistency)
        df = df.with_columns([
            pl.when(pl.col("player_id") < pl.col("opponent_id"))
            .then(pl.concat_str([pl.col("player_id"), pl.lit("_"), pl.col("opponent_id")]))
            .otherwise(pl.concat_str([pl.col("opponent_id"), pl.lit("_"), pl.col("player_id")]))
            .alias("matchup_key")
        ])
        
        # H2H wins for this player in this matchup
        df = df.with_columns([
            pl.col("player_won")
            .cast(pl.Float64)
            .shift(1)
            .cum_sum()
            .over(["player_id", "matchup_key"])
            .alias("h2h_wins"),
            
            pl.lit(1)
            .shift(1)
            .cum_sum()
            .over(["player_id", "matchup_key"])
            .alias("h2h_matches")
        ])
        
        df = df.with_columns([
            (pl.col("h2h_wins") / pl.col("h2h_matches").clip(1))
            .alias("h2h_win_rate")
        ])
        
        return df
    
    def add_surface_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add surface-specific win rate.
        """
        # Normalize surface names
        df = df.with_columns([
            pl.col("ground_type")
            .str.to_lowercase()
            .str.replace_all(r".*clay.*", "clay")
            .str.replace_all(r".*grass.*", "grass")
            .str.replace_all(r".*hard.*", "hard")
            .alias("surface_normalized")
        ])
        
        # Surface-specific rolling win rate
        for window in [10, 20]:
            df = df.with_columns([
                pl.col("player_won")
                .cast(pl.Float64)
                .shift(1)
                .rolling_mean(window_size=window, min_periods=3)
                .over(["player_id", "surface_normalized"])
                .alias(f"player_surface_win_rate_{window}")
            ])
        
        return df
    
    def add_round_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Encode tournament round as numeric.
        """
        round_map = {
            "Final": 7,
            "Semifinal": 6,
            "Quarterfinal": 5,
            "Round of 16": 4,
            "Round of 32": 3,
            "Round of 64": 2,
            "Round of 128": 1,
            "Qualification": 0,
        }
        
        df = df.with_columns([
            pl.col("round_name")
            .replace(round_map, default=2)
            .alias("round_num")
        ])
        
        return df
    
    def add_odds_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add odds-derived features.
        """
        schema = df.collect_schema().names()
        
        if "odds_player" in schema and "odds_opponent" in schema:
            df = df.with_columns([
                # Implied probabilities
                (1 / pl.col("odds_player")).alias("implied_prob_player"),
                (1 / pl.col("odds_opponent")).alias("implied_prob_opponent"),
                
                # Odds ratio
                (pl.col("odds_opponent") / pl.col("odds_player")).alias("odds_ratio"),
                
                # Is underdog (odds > 2.0)
                (pl.col("odds_player") > 2.0).alias("is_underdog"),
            ])
        
        return df
    
    def get_feature_columns(self, df: pl.LazyFrame) -> List[str]:
        """
        Return list of feature columns for ML training.
        Excludes identifiers, target, metadata, and POST-MATCH outcome data.
        """
        # Columns to completely exclude
        exclude_exact = {
            "event_id", "player_id", "opponent_id", "player_name", "opponent_name",
            "player_won", "start_timestamp", "match_date", "match_year", "match_month",
            "match_day", "status", "matchup_key", "tournament_id", "tournament_name",
            "round_name", "ground_type", "surface_normalized",
            # POST-MATCH OUTCOME DATA (causes leakage!)
            "player_sets", "opponent_sets", 
            "player_set1", "player_set2", "player_set3", "player_set4", "player_set5",
            "opponent_set1", "opponent_set2", "opponent_set3", "opponent_set4", "opponent_set5",
        }
        
        # Patterns to exclude
        exclude_patterns = [
            "_scraped", "has_stats", "has_odds", "_schema_version", "is_home",
        ]
        
        schema = df.collect_schema().names()
        
        feature_cols = []
        for col in schema:
            # Skip exact matches
            if col in exclude_exact:
                continue
            # Skip pattern matches
            if any(excl in col for excl in exclude_patterns):
                continue
            # Skip columns starting with _ (metadata)
            if col.startswith("_"):
                continue
            
            feature_cols.append(col)
        
        return feature_cols
