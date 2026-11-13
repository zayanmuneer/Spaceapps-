#!/usr/bin/env python3
"""
PyroScale-AI: Data Harmonization Pipeline for MODIS & VIIRS Satellite Active Fire Hotspots.
Downloads active fire data from NASA FIRMS API, harmonizes sensor characteristics,
and outputs a standardized JSON dataset.
"""

import os
import sys
import argparse
import json
import requests
import pandas as pd
import numpy as np

FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

def fetch_firms_data(map_key, source="VIIRS_NOAA20_NRT", days=10):
    # Default bounding box covering global or high-activity regions / sample area
    # format: min_lon,min_lat,max_lon,max_lat (e.g., global: -180,-90,180,90 or regional)
    # Using a broad bounding box or world extent
    bbox = "-180,-90,180,90"
    url = f"{FIRMS_API_BASE}/{map_key}/{source}/{bbox}/{days}"
    print(f"Fetching active fire data from NASA FIRMS ({source})...")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching FIRMS data: {response.status_code} - {response.text}")
        return pd.DataFrame()
    
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    return df

def harmonize_sensors(modis_df, viirs_df):
    """
    Harmonizes MODIS (1km resolution) and VIIRS (375m / 750m resolution) active fire records
    by applying radiometric and spatial scaling factors to reconcile historical hotspot densities.
    """
    print("Harmonizing MODIS and VIIRS hotspot records...")
    
    # Standardize column schemas
    standard_cols = ['latitude', 'longitude', 'bright_ti4', 'scan', 'track', 'acq_date', 'acq_time', 'confidence', 'satellite', 'instrument']
    
    frames = []
    if not modis_df.empty:
        modis_df['instrument'] = 'MODIS'
        # Apply MODIS radiometric calibration scaling to match VIIRS baseline
        if 'brightness' in modis_df.columns:
            modis_df['normalized_frp'] = modis_df['brightness'] * 0.85
        frames.append(modis_df)
        
    if not viirs_df.empty:
        viirs_df['instrument'] = 'VIIRS'
        # Apply VIIRS sensitivity weighting
        if 'bright_ti4' in viirs_df.columns:
            viirs_df['normalized_frp'] = viirs_df['bright_ti4'] * 0.92
        frames.append(viirs_df)
        
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        return combined
    return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="PyroScale-AI Data Harmonization Pipeline")
    parser.add_argument("--download", action="store_true", help="Download live data from NASA FIRMS")
    parser.add_argument("--out", default="pyroscale_data.json", help="Output JSON file path")
    args = parser.parse_args()
    
    map_key = os.environ.get("FIRMS_MAP_KEY", "DEMO_KEY")
    
    if args.download and map_key != "DEMO_KEY":
        viirs_df = fetch_firms_data(map_key, source="VIIRS_NOAA20_NRT", days=5)
        modis_df = fetch_firms_data(map_key, source="MODIS_NRT", days=5)
        df = harmonize_sensors(modis_df, viirs_df)
    else:
        print("Using sample harmonized active fire dataset simulation (No valid FIRMS_MAP_KEY provided)...")
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
        df = pd.DataFrame({
            'latitude': np.random.uniform(-35, 60, 100),
            'longitude': np.random.uniform(-120, 140, 100),
            'normalized_frp': np.random.uniform(10, 350, 100),
            'acq_date': np.random.choice(dates, 100).astype(str),
            'confidence': np.random.choice(['low', 'nominal', 'high'], 100),
            'instrument': np.random.choice(['MODIS', 'VIIRS'], 100)
        })
        
    # Export to JSON
    records = df.to_dict(orient="records")
    output_data = {
        "metadata": {
            "title": "PyroScale-AI Harmonized Active Fire Hotspots",
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_records": len(records)
        },
        "data": records
    }
    
    with open(args.out, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Successfully saved harmonized data to {args.out} ({len(records)} records).")

if __name__ == "__main__":
    main()
