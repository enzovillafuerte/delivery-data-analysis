import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time


df = pd.read_csv('train.csv')


# Total-time in minutes only
# df['time (minutes)'] = df['Time_taken(min)'].split()[1]

# Total-time in minutes only
# df['time (minutes)'] = df['Time_taken(min)'].apply(lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) > 0 else None)
df['time (minutes)'] = df['Time_taken(min)'].apply(lambda x: x.split()[1])


# Harversine distances (an idea)
# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate distances and create 'Province' column
df['distance (km)'] = df.apply(lambda row: haversine(row['Restaurant_latitude'], row['Restaurant_longitude'], 
                                                       row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)
     

# City origin

'''
TAKES TOO LONG TO PROCESS
# Function to get city name from latitude and longitude using Nominatim
def get_city(lat, lon):
    geolocator = Nominatim(user_agent="my_geocoding_app")  # Set a unique user agent
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location and 'address' in location.raw:
            return location.raw['address'].get('city', location.raw['address'].get('town', 'Unknown'))
    except GeocoderTimedOut:
        return 'Unknown'
    except Exception as e:
        print(f"Error: {e}")
        return 'Unknown'
    return 'Unknown'

# Get city names and create 'Province' column
df['Province'] = df.apply(lambda row: get_city(row['Restaurant_latitude'], row['Restaurant_longitude']), axis=1)
time.sleep(1)  # Sleep for 1 second between requests

'''

# Consolidated dataset for jupyter analysis

df.to_csv('final_dataset.csv', index=False)

print(df.head())

# python pre-processing.py