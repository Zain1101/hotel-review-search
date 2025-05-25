import requests

def get_hotel_image_url(hotel_name, city, api_key, cse_id):
    query = f"{hotel_name} {city} hotel"
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id,
        'searchType': 'image',
        'num': 1
    }
    response = requests.get(url, params=params).json()
    if 'items' in response:
        return response['items'][0]['link']
    return None

