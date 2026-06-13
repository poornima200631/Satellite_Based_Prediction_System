import pandas as pd
import pickle
from geopy.geocoders import Nominatim
import time

def main():
    print("Loading data and encoders...")
    df = pd.read_csv("data/processed/model_ready_data.csv")
    with open("models/encoder.pkl", "rb") as f:
        encoders = pickle.load(f)
    
    loc_encoder = encoders['location']
    
    # Reverse the mapping: {0: 'Delhi', 1: 'Mumbai', ...}
    inv_loc = {v: k for k, v in loc_encoder.items()}
    
    # Get unique encoded locations in the dataset
    unique_locs = df['location'].unique()
    print(f"Found {len(unique_locs)} unique locations to geocode.")
    
    geolocator = Nominatim(user_agent="satellite_air_quality_app")
    
    coords = {}
    for code in unique_locs:
        city_name = inv_loc.get(code)
        if not city_name:
            continue
        
        # Try to geocode with "India" appended for better accuracy
        query = f"{city_name}, India"
        try:
            location = geolocator.geocode(query, timeout=10)
            if location:
                # Convert numpy int64 key to regular int so it is JSON serializable
                coords[int(code)] = {'lat': location.latitude, 'lon': location.longitude, 'name': city_name}
                print(f"Success: {city_name} -> {location.latitude}, {location.longitude}")
            else:
                print(f"Failed to geocode: {city_name}")
            time.sleep(1) # Be polite to the API
        except Exception as e:
            print(f"Error for {city_name}: {e}")
            time.sleep(2)
            
    # Save the coordinates to a file
    import json
    with open("data/processed/location_coords.json", "w") as f:
        json.dump(coords, f)
    
    print("\nGeocoding complete! Saved to data/processed/location_coords.json")

if __name__ == "__main__":
    main()
