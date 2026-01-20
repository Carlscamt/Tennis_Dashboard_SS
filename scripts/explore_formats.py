"""Explore data for weird stat formats like '50% (5/10)'."""
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
files = sorted(DATA_DIR.glob("atp_matches_*.parquet"), key=lambda p: p.stat().st_mtime)

print("="*70)
print("EXPLORING DATA FOR WEIRD FORMATS")
print("="*70)

if files:
    # Use the latest file with stats
    latest = files[-1]
    df = pl.read_parquet(latest)
    
    print(f"\nFile: {latest.name}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Find string columns
    print("\n" + "-"*50)
    print("STRING COLUMNS (potential format issues)")
    print("-"*50)
    
    string_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    print(f"Found {len(string_cols)} string columns")
    
    # Check each string column for weird formats
    print("\n" + "-"*50)
    print("COLUMNS WITH PERCENTAGE/BRACKET PATTERNS")
    print("-"*50)
    
    weird_patterns = []
    
    for col in string_cols:
        # Get sample values
        sample = df[col].drop_nulls().head(100)
        
        has_percent = any("%" in str(v) for v in sample if v is not None)
        has_bracket = any("(" in str(v) for v in sample if v is not None)
        has_slash = any("/" in str(v) for v in sample if v is not None)
        
        if has_percent or has_bracket or has_slash:
            # Get unique samples
            unique_vals = df[col].drop_nulls().unique().head(5).to_list()
            weird_patterns.append({
                "column": col,
                "has_%": has_percent,
                "has_()": has_bracket,
                "has_/": has_slash,
                "samples": unique_vals[:5]
            })
            
            print(f"\n{col}:")
            print(f"  Patterns: {'%' if has_percent else ''} {'()' if has_bracket else ''} {'/' if has_slash else ''}")
            print(f"  Samples: {unique_vals[:5]}")
    
    # Summary of columns to clean
    print("\n" + "="*70)
    print("SUMMARY: COLUMNS NEEDING CLEANING")
    print("="*70)
    
    # Group by pattern type
    percent_cols = [p["column"] for p in weird_patterns if p["has_%"]]
    bracket_cols = [p["column"] for p in weird_patterns if p["has_()"]]
    
    print(f"\nColumns with % patterns: {len(percent_cols)}")
    for col in percent_cols[:10]:
        print(f"  - {col}")
    if len(percent_cols) > 10:
        print(f"  ... and {len(percent_cols) - 10} more")
    
    print(f"\nColumns with () bracket patterns: {len(bracket_cols)}")
    for col in bracket_cols[:10]:
        print(f"  - {col}")
    if len(bracket_cols) > 10:
        print(f"  ... and {len(bracket_cols) - 10} more")
    
    # Show detailed samples for a few key columns
    print("\n" + "-"*50)
    print("DETAILED SAMPLES")
    print("-"*50)
    
    for p in weird_patterns[:5]:
        col = p["column"]
        print(f"\n{col}:")
        vals = df[col].drop_nulls().unique().head(10).to_list()
        for v in vals:
            print(f"  '{v}'")
    
    print("\n" + "="*70)
    print("RECOMMENDATION: Parse these columns to extract numeric values")
    print("="*70)

else:
    print("No data files found")
