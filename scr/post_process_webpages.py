import json
from collections import Counter
import pandas as pd
from bs4 import BeautifulSoup

path_web_jsonl = '/Users/celistittse/Documents/CultTech/Data/raw_data/gt_webtexts.jsonl'
path_gt_df = '/Users/celistittse/Documents/CultTech/Data/raw_data/gt_contacts.xlsx'

gt_df = pd.read_excel(path_gt_df)
gt_id_name_d = dict(zip(gt_df['id'], gt_df['source_name']))


social_medias = ['marktplaats', 'spotify', 'tiktok', 'linkedin', 'facebook',
                 'music.apple', 'youtube', 'soundcloud', 'play.google', 'imdb', 'deezer']

def find_word_in_url (url: str, search_list: list) -> bool:
    for search_str in search_list:
        if search_str in url:
            return True
    return False

# remove social media and music sites
def filter_websites (data: list, socialMedia: list) -> list:
    filt_data = []
    for webpage in data:
        url = webpage['url']

        # if social media website not in url add to filtered results
        if not find_word_in_url(url=url, search_list=socialMedia):
            filt_data.append(webpage)
    return filt_data

def filter_on_name(data: list, nameDict: dict) -> list:
    filt_data = []
    for website in data:
        # isolate e.g. "12.1" -> int(12)
        id_name = int(website['key'].split('.')[0])

        # retrieve search name
        name = nameDict[id_name]

        # if name in website text add to filtered results
        if name.lower() in website['full_text'].lower():
            website.update({'name': name})
            filt_data.append(website)

    return filt_data

def post_process_names_d (raw_json_path:str=None, results:list=False) -> list:
    if not results and raw_json_path:
        with open(raw_json_path, encoding='utf-8-sig') as json_file:
            # open json file
            results = [json.loads(json_str) for json_str in list(json_file)]
    else:
        # remove social medias
        results = filter_websites(data=results,
                                  socialMedia=social_medias)

        # remove text that don't include name from searched person
        results = filter_on_name (data=results,
                                  nameDict=gt_id_name_d)

    return results

