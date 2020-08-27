import pandas as pd
import os 
import time
import requests
import sys
import argparse

from SquidleAPI import SquidleAPI

SQUIDLE_URL = "https://soi.squidle.org"

def squidle_extract_prefix(media_collection_id, api_token):
    # Setup initial params
    squid_api = SquidleAPI(api_token, SQUIDLE_URL)
    media_list = squid_api.get_media_list(media_collection_id, limit=2)

    m = media_list[0] 
    ann = squid_api.get_media_annotation_for_id(m['media_id'])
    url = ann['path_best'].strip()

    prefix = url.split('/')[0:-1]
    prefix_url = '/'.join(prefix)
    print(prefix_url)


    return prefix_url

def squidle_extract_urls(media_collection_id, api_token, limit = 1000):
    # Setup initial params
    squid_api = SquidleAPI(api_token, SQUIDLE_URL)
    media_list = squid_api.get_media_list(media_collection_id, limit=limit)

    image_urls = []

    for m in media_list:
        ann = squid_api.get_media_annotation_for_id(m['media_id'])
        url = ann['path_best'].strip()
        image_urls.append(url)
    
    # prefix = url.split('/')[0:-1]
    # prefix_url = '/'.join(prefix)
    # print(prefix_url)
    return image_urls

def get_urls(filename, col_name, prefix=None):
    df = pd.read_csv(filename)
    urls = df[col_name]
    if prefix is not None:
        for i in range(0, len(urls)):
            urls[i] = prefix + urls[i]
    return urls

def save_images(img_urls, dest, limit=1500):
    if len(img_urls) > limit:
        img_urls = img_urls[0:limit]
    filenames = []
    for url in img_urls:
        url = url.strip(' /')
        filename = url.split('/')[-1]
        if filename in filenames:
            continue
        print("Copying " + url)
        filenames.append(filename)
        img_data = requests.get(url).content
        with open(data_dest + filename, 'wb') as handler:
            handler.write(img_data)

if __name__ == "__main__":    

    parser = argparse.ArgumentParser(description='Specify where to get data from')
    parser.add_argument('-s', '--squidle', required=False, 
                        help='Get from squidle', action='store_true')
    parser.add_argument('-f', '--folder', required=False,
                        help="Folder to store data in")

    args = vars(parser.parse_args())

    
    if args['folder']:
        data_dest = args['folder']
    else:
        data_dest = "./images/"

    try:
        os.mkdir(data_dest)
    except FileExistsError:
        print("Directory exists, continuing ...")

    file_path = 'sesoko_crest.csv'
    col_name = 'media_path'

    print("Starting data + image extraction ...")
    api_token = None # add your own here  
    media_id = 76 # ID 80 for AE2000_3000
                  # ID 76 for sesoko crest
    if args['squidle']:
        # Set this to your API token for your account
        url_prefix = squidle_extract_prefix(media_id, api_token)
        img_urls = get_urls(file_path, col_name, url_prefix)
    else: # read URLs from file
        img_urls = get_urls(file_path, col_name)
    
    save_images(img_urls, data_dest)


    print("Finished copying images.")
