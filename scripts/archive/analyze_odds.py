"""Analyze model accuracy by odds ranges."""
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
df = pl.read_parquet(DATA_DIR / "features_dataset.parquet")

# Filter to test set (2024+) with odds
df = df.filter(
    (pl.col("match_year") >= 2024) &
    (pl.col("odds_player").is_not_null())
)

print(f"Test matches with odds: {len(df):,}")
print()

# Create granular odds bins
df = df.with_columns([
    pl.when(pl.col("odds_player") < 1.20).then(pl.lit("1.01-1.20"))
    .when(pl.col("odds_player") < 1.40).then(pl.lit("1.20-1.40"))
    .when(pl.col("odds_player") < 1.60).then(pl.lit("1.40-1.60"))
    .when(pl.col("odds_player") < 1.80).then(pl.lit("1.80-1.80"))
    .when(pl.col("odds_player") < 2.00).then(pl.lit("1.80-2.00"))
    .when(pl.col("odds_player") < 2.50).then(pl.lit("2.00-2.50"))
    .when(pl.col("odds_player") < 3.00).then(pl.lit("2.50-3.00"))
    .when(pl.col("odds_player") < 4.00).then(pl.lit("3.00-4.00"))
    .when(pl.col("odds_player") < 5.00).then(pl.lit("4.00-5.00"))
    .otherwise(pl.lit("5.00+"))
    .alias("odds_bin")
])

# Also add implied probability bins
df = df.with_columns([
    (1 / pl.col("odds_player")).alias("implied_prob")
])

# Calculate stats per bin
print("="*70)
print("MODEL ACCURACY BY ODDS RANGE (Test Set 2024+)")
print("="*70)
print()
print(f"{'Odds Range':<12} | {'Matches':>8} | {'Actual Win%':>11} | {'Implied%':>10} | {'Edge':>8}")
print("-"*70)

bins_order = [
    "1.01-1.20", "1.20-1.40", "1.40-1.60", "1.80-1.80", "1.80-2.00",
    "2.00-2.50", "2.50-3.00", "3.00-4.00", "4.00-5.00", "5.00+"
]

for bin_name in bins_order:
    subset = df.filter(pl.col("odds_bin") == bin_name)
    if len(subset) == 0:
        continue
    
    matches = len(subset)
    actual_win = subset["player_won"].mean()
    implied = subset["implied_prob"].mean()
    edge = actual_win - implied
    
    print(f"{bin_name:<12} | {matches:>8,} | {actual_win*100:>10.1f}% | {implied*100:>9.1f}% | {edge*100:>+7.1f}%")

print("-"*70)

# Overall
total = len(df)
overall_win = df["player_won"].mean()
overall_implied = df["implied_prob"].mean()
overall_edge = overall_win - overall_implied

print(f"{'TOTAL':<12} | {total:>8,} | {overall_win*100:>10.1f}% | {overall_implied*100:>9.1f}% | {overall_edge*100:>+7.1f}%")
print()

# More granular analysis with model predictions
print("="*70)
print("MODEL PREDICTION ACCURACY BY CONFIDENCE LEVEL")
print("="*70)

# Check if model_prob column exists
if "model_prob" not in df.columns:
    print("\nNote: Run predictions first to get model confidence analysis.")
else:
    df = df.with_columns([
        pl.when(pl.col("model_prob") < 0.40).then(pl.lit("0-40%"))
        .when(pl.col("model_prob") < 0.50).then(pl.lit("40-50%"))
        .when(pl.col("model_prob") < 0.55).then(pl.lit("50-55%"))
        .when(pl.col("model_prob") < 0.60).then(pl.lit("55-60%"))
        .when(pl.col("model_prob") < 0.65).then(pl.lit("60-65%"))
        .when(pl.col("model_prob") < 0.70).then(pl.lit("65-70%"))
        .when(pl.col("model_prob") < 0.80).then(pl.lit("70-80%"))
        .otherwise(pl.lit("80%+"))
        .alias("conf_bin")
    ])
    
    print(f"\n{'Confidence':<12} | {'Matches':>8} | {'Actual Win%':>11} | {'Accuracy':>10}")
    print("-"*50)
    
    conf_order = ["0-40%", "40-50%", "50-55%", "55-60%", "60-65%", "65-70%", "70-80%", "80%+"]
    
    for conf_bin in conf_order:
        subset = df.filter(pl.col("conf_bin") == conf_bin)
        if len(subset) == 0:
            continue
        
        matches = len(subset)
        actual = subset["player_won"].mean()
        
        # For confidence above 50%, we predict win. Below 50%, we predict loss.
        # So accuracy is different
        if "50" in conf_bin or "55" in conf_bin or "60" in conf_bin or "65" in conf_bin or "70" in conf_bin or "80" in conf_bin:
            accuracy = actual  # Predicting win
        else:
            accuracy = 1 - actual  # Predicting loss
        
        print(f"{conf_bin:<12} | {matches:>8,} | {actual*100:>10.1f}% | {accuracy*100:>9.1f}%")
