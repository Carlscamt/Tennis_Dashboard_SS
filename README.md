# Tennis Betting ML Pipeline

Professional tennis betting prediction system using machine learning with real-time dashboard.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape upcoming matches
python tennis.py scrape upcoming --days 7

# Get betting predictions
python tennis.py predict --days 7 --min-odds 1.5 --max-odds 3.0

# Launch dashboard
streamlit run dashboard/app.py
```

## CLI Commands

All operations are now unified through `tennis.py`:

```bash
# Scraping
python tennis.py scrape historical --top 50      # Top 50 ATP players
python tennis.py scrape upcoming --days 7        # Next 7 days
python tennis.py scrape players --ids 12345      # Specific players

# Predictions
python tennis.py predict --days 7 --min-odds 1.5 --max-odds 3.0

# Model Management
python tennis.py train                           # Retrain model
python tennis.py audit                           # Run model audit
python tennis.py backtest                        # Historical backtest
```

## Project Structure

```
TENNIS 4.0/
├── tennis.py                 # Unified CLI entry point
├── src/
│   ├── scraper.py           # Consolidated SofaScore scraper
│   ├── pipeline.py          # Prediction workflows
│   ├── schema.py            # Data validation & deduplication
│   ├── model/               # XGBoost model & registry
│   ├── transform/           # Feature engineering
│   ├── betting/             # Kelly criterion & value finder
│   └── extract/             # Data loaders
├── scripts/
│   ├── run_pipeline.py      # Training pipeline
│   ├── model_audit.py       # Comprehensive model audit
│   ├── backtest.py          # Backtesting framework
│   └── archive/             # Deprecated scripts (14 files)
├── dashboard/               # Streamlit UI
├── data/
│   ├── tennis.parquet       # Historical match data
│   └── upcoming.parquet     # Cached upcoming matches
├── models/                  # Trained model artifacts
├── config/                  # Settings & configuration
└── tests/                   # Test suite
```

## Features

- **Unified CLI** - Single `tennis.py` for all operations
- **Polars-based** - Fast columnar data processing
- **XGBoost** - Calibrated probability predictions
- **Data Leakage Prevention** - Strict temporal ordering with shift(1)
- **ATP/Challenger Filter** - Strictly singles matches only
- **Smart Caching** - Avoids re-scraping recent data (< 1 hour)
- **Value Bet Detection** - Edge calculation vs bookmaker odds
- **Kelly Criterion** - Optimal stake sizing (1/4 Kelly)
- **Real-time Dashboard** - Streamlit UI for betting signals

## Example Output

```
#1 >>> BET ON: Mats Rosenkranz
    vs Chris Rodesch
    Win Prob: 44.2% | Odds: 3.00 | Edge: +10.9%
    Tournament: Oeiras, Portugal

#2 >>> BET ON: Lukas Neumayer
    vs Borna Gojo
    Win Prob: 49.4% | Odds: 2.50 | Edge: +9.4%
    Tournament: ATP Challenger Soma Bay, Egypt
```

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 67.6% |
| AUC-ROC | 0.692 |
| ECE (Calibration) | 0.042 |
| Profitable Odds Range | 1.5-3.0 |

## Data Sources

- **SofaScore API** - Match data, statistics, and odds
- **Historical Data** - 96,000+ ATP matches

## License

MIT
