
# crawl data from Google for without hijab images

# load used libraries
from icrawler.builtin import GoogleImageCrawler

import configs.config as config

google_crawler = GoogleImageCrawler(
    parser_threads=2,
    downloader_threads=2,
    storage={'root_dir': config.RAW_IMAGES_DIR}
)

keywords = ['without hijab'] # your keywords here
num_images = 2000 # number of images per keyword

for keyword in keywords:
    google_crawler.crawl(keyword=keyword, max_num=num_images)
    
