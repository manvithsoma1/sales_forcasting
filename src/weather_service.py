import requests

def get_current_weather(api_key="7f8ffc7f3a4e580bb947dc11bc4f6eb9"):
    """
    Fetches real-time weather for Quito, Ecuador.
    Returns: dictionary with 'main' (Rain/Clear) and 'temp' (Celsius).
    """
    # Coordinates for Quito, Ecuador (Store 1 location)
    LAT = "-0.1807"
    LON = "-78.4678"
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            return {
                "condition": data['weather'][0]['main'], # e.g., "Rain", "Clouds"
                "description": data['weather'][0]['description'],
                "temp": data['main']['temp']
            }
        else:
            return {"condition": "Unknown", "temp": 20} # Fallback
    except:
        return {"condition": "Unknown", "temp": 20} # Fallback