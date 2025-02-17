from geopy.geocoders import Nominatim

def get_coordinates(country, city, postcode=None):
    try:
        query = f"{postcode}, {city}, {country}" if postcode else f"{city}, {country}"
        location = Nominatim(user_agent="birds_eye_view_downloader").geocode(query)
        if not location: raise ValueError(f"Could not find location: {query}")
        return location.latitude, location.longitude, location.address
    except Exception as e: raise Exception(f"Location lookup failed: {e}")

def get_location_suggestions(query, limit=5):
    geolocator = Nominatim(user_agent="location_suggestions")
    locations = geolocator.geocode(query, exactly_one=False, limit=limit)
    return [location.address for location in locations] if locations else []