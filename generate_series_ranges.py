#!/usr/bin/env python3
"""
Script to generate series_range.json by scanning all feather files.
Run this script whenever you add new experiments or want to update the ranges.

Usage: python3 generate_series_ranges.py
"""

import os
import json
import pandas as pd
from pathlib import Path

def get_unit(col):
    """Extract unit from column name"""
    if isinstance(col, tuple):
        measurement_with_unit = col[1] if len(col) > 1 else col[0]
    else:
        measurement_with_unit = col
    
    if '-' in str(measurement_with_unit):
        unit = str(measurement_with_unit).split('-')[-1].strip()
        return unit
    return 'Unknown'

def load_df(filename):
    """Load and process feather file"""
    THIS_FOLDER = Path(__file__).parent.resolve()
    DATA_DIR = os.path.join(THIS_FOLDER, 'data')
    
    df = pd.read_feather(os.path.join(DATA_DIR, filename))
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df = df[numeric_cols]
    df = df.interpolate().resample('1min').mean()
    return df

def build_series_ranges():
    """
    Scan all feather files and build a dictionary of min/max ranges for each series.
    """
    THIS_FOLDER = Path(__file__).parent.resolve()
    DATA_DIR = os.path.join(THIS_FOLDER, 'data')
    
    print("Scanning all feather files to build series ranges...")
    
    series_ranges = {}
    feather_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.feather')]
    processed_files = 0
    
    for feather_file in feather_files:
        try:
            df = load_df(feather_file)
            processed_files += 1
            
            for col in df.columns:
                col_str = str(col)
                
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Get min and max values, ignoring NaN
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                
                col_min = float(col_data.min())
                col_max = float(col_data.max())
                
                # Update global ranges
                if col_str not in series_ranges:
                    series_ranges[col_str] = {
                        'min': col_min,
                        'max': col_max,
                        'unit': get_unit(col)
                    }
                else:
                    series_ranges[col_str]['min'] = min(series_ranges[col_str]['min'], col_min)
                    series_ranges[col_str]['max'] = max(series_ranges[col_str]['max'], col_max)
            
            if processed_files % 10 == 0:
                print(f"  Processed {processed_files}/{len(feather_files)} files...")
                
        except Exception as e:
            print(f"Error processing {feather_file}: {e}")
            continue
    
    # Add padding to ranges (5% on each side)
    for series_name, range_data in series_ranges.items():
        range_span = range_data['max'] - range_data['min']
        padding = range_span * 0.05 if range_span > 0 else 0.1
        
        series_ranges[series_name]['padded_min'] = range_data['min'] - padding
        series_ranges[series_name]['padded_max'] = range_data['max'] + padding
    
    print(f"Built ranges for {len(series_ranges)} series from {processed_files} files")
    return series_ranges

def main():
    """Generate and save series ranges to JSON file"""
    THIS_FOLDER = Path(__file__).parent.resolve()
    DATA_DIR = os.path.join(THIS_FOLDER, 'data')
    output_file = os.path.join(DATA_DIR, 'series_range.json')
    
    # Build ranges
    series_ranges = build_series_ranges()
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(series_ranges, f, indent=2)
    
    print(f"\nSaved series ranges to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    # Show some sample ranges
    print("\nSample ranges:")
    for i, (series_name, range_data) in enumerate(list(series_ranges.items())[:5]):
        print(f"  {series_name}:")
        print(f"    Range: [{range_data['padded_min']:.2f}, {range_data['padded_max']:.2f}] {range_data['unit']}")
    
    print(f"\n✅ Ready to use! Restart your Flask app to load the new ranges.")

if __name__ == "__main__":
    main()
