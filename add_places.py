#!/usr/bin/env python3
"""
PyroScale-AI: Reverse Geocoding & Country/City Share Mapping Script.
Maps active fire hotspot coordinates to countries, administrative regions, and cities.
"""

import argparse
import json
import reverse_geocoder as rg
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="PyroScale-AI Place Matching")
    parser.add_argument("--json", default="pyroscale_data.json", help="Input harmonized JSON data")
    parser.add_argument("--places", default="pyroscale_places.json", help="Output places mapping JSON")
    args = parser.parse_args()
    
    print(f"Loading data from {args.json}...")
    with open(args.json, "r") as f:
        content = json.load(f)
        
    df = pd.DataFrame(content.get("data", []))
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("No valid spatial coordinates found in dataset.")
        return
        
    print("Performing offline reverse geocoding...")
    coords = list(zip(df['latitude'], df['longitude']))
    results = rg.search(coords) # returns list of dicts with cc, name, admin1, etc.
    
    df['country_code'] = [r.get('cc', 'Unknown') for r in results]
    df['admin_region'] = [r.get('admin1', 'Unknown') for r in results]
    df['nearest_city'] = [r.get('name', 'Unknown') for r in results]
    
    # Calculate country share summaries
    country_counts = df['country_code'].value_counts().reset_index()
    country_counts.columns = ['country_code', 'hotspot_count']
    country_counts['percentage'] = (country_counts['hotspot_count'] / len(df)) * 100
    
    output_places = {
        "country_shares": country_counts.to_dict(orient="records"),
        "city_hotspots": df[['latitude', 'longitude', 'country_code', 'admin_region', 'nearest_city', 'normalized_frp', 'acq_date']].to_dict(orient="records")
    }
    
    with open(args.places, "w") as f:
        json.dump(output_places, f, indent=2)
    print(f"Successfully generated regional place mappings in {args.places}.")

if __name__ == "__main__":
    main()
