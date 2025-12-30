
# load used library
import requests

def download_image(url, save_path):
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
