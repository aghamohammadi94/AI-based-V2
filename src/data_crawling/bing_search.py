
# load used library
import requests

BING_API_KEY = 'PUT_YOUR_API_KEY_HERE'
BING_ENDPOINT = 'https://api.bing.microsoft.com/v7.0/images/search'


def search_images(query, num_images=2000):
    headers = {'Ocp-Apim-Subscription-key': BING_API_KEY}
    params = {
        'q': query,
        'count': num_images,
        'imageType': 'Photo',
        'safeSearch': 'Moderate'
    }
    
    response = requests.get(BING_ENDPOINT, headers=headers, params=params)
    response.raise_for_status()
    
    results = response.json()
    return [img['contentUrl'] for img in results['value']]