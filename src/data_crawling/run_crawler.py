
'''
# load used library
from data_crawling.crawler import crawl
import configs.config as config


# -----------------------------
# URLs of images WITH hijab
# -----------------------------
yes_urls = [
    "https://example.com/hijab1.jpg",
    "https://example.com/hijab2.jpg",
    "https://example.com/hijab3.jpg",
]

# -----------------------------
# URLs of images WITHOUT hijab
# -----------------------------
no_urls = [
    "https://example.com/no_hijab1.jpg",
    "https://example.com/no_hijab2.jpg",
    "https://example.com/no_hijab3.jpg",
]

def main():
    crawl(yes_urls, label='yes', output_dir=config.RAW_IMAGES_DIR)
    crawl(yes_urls, label='no', output_dir=config.RAW_IMAGES_DIR)
    
if __name__ == '__main__':
    main()
'''  


from data_crawling.crawler import crawl_google_images
import configs.config as config


def main():
    # WITH hijab
    crawl_google_images(
        query="woman face with hijab",
        label="yes",
        output_dir=config.RAW_IMAGES_DIR,
        num_images=2000
    )

    # WITHOUT hijab
    crawl_google_images(
        query="woman face without hijab",
        label="no",
        output_dir=config.RAW_IMAGES_DIR,
        num_images=2000
    )


if __name__ == "__main__":
    main()