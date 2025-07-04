import re
import os
import json

path_webtxts = '/Users/celistittse/Documents/CultTech/Data/raw_data/gt_ddg_txts/'
save_path = '/Users/celistittse/Documents/CultTech/Data/raw_data/'

def read_web_txts (f) -> list:
    lines = f.readlines()

    # replace DDG annotation of websites with searchable syntax
    lines = [line.replace(' › ', '/') for line in lines]

    # isolate websites
    web_lines = [line.strip() for line in lines if 'https' in line]

    return web_lines

def filter_double_sites (site_dict: dict[list]) -> dict[list]:
    # Generalize filtering for keys ending in '.1'
    for key in list(site_dict.keys()):
        if key.endswith(".1"):
            base_key = key[:-1] + "0"  # e.g. "26.1" → "26.0"
            if base_key in site_dict:
                site_dict[key] = [
                    site for site in site_dict[key] if site not in site_dict[base_key]
                ]

    return site_dict

def read_txt_gt_files (path: str) -> dict[list]:
    webstrings_d = {}

    for file_id in os.listdir(path):

        with open(path + file_id, encoding='utf-8-sig') as f_txt:
            # open txt and isolate http strings
            web_strs = read_web_txts(f_txt)

            # add to dict with id_0 (prompt: "name" affiliation) or id_1 (prompt: "name")
            webstrings_d[file_id.replace('.txt', '')] = web_strs

    # filter double sites in X.1 if already in X.0
    webStrFilter = filter_double_sites(site_dict=webstrings_d)

    return webStrFilter

