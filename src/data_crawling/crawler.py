
# load used library
import os

from data_crawling.downloader import download_image
from data_crawling.bing_search import search_images


def crawl(urls, label, output_dir):
    """
    urls: list of image urls
    label: 'yes' or 'no'
    output_dir: where images are saved
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, url in enumerate(urls):
        filename = f'{label}_{i+1}.jpg'
        save_path = os.path.join(output_dir, filename)
        
        download_image(url, save_path)


def crawl_google_images(query, label, output_dir, num_images=2000):
    """
    query: search text (hijab / without hijab)
    label: yes / no
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    urls = search_images(query, num_images=num_images)
    
    for i, url in enumerate(urls):
        filename = f'{label}_{i+1}.jpg'
        save_path = os.path.join(output_dir, filename)
        
        try:
            download_image(url, save_path)
        except Exception as e:
            print(f'Failed: {url} -> {e}')