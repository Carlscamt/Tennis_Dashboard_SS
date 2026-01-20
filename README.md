# Tennis Betting ML Pipeline

Professional tennis betting prediction system with real-time dashboard.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python scripts/run_pipeline.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
├── config/          # Configuration files
├── data/            # Raw, processed, predictions
├── src/             # Core modules (extract, transform, model, betting)
├── dashboard/       # Streamlit UI
├── scripts/         # Pipeline scripts
├── tests/           # Test suite
└── models/          # Trained model artifacts
```

## Features

- **Polars-based** data processing for speed
- **XGBoost** with probability calibration
- **Kelly Criterion** stake sizing (1/4 Kelly)
- **Data leakage prevention** built-in
- **Real-time dashboard** for betting signals
