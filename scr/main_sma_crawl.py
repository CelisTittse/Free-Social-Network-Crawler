from extract_webpages import read_txt_gt_files
from post_process_webpages import post_process_names_d
from NER_persons import load_huggingface_ners, sort_ner_results_to_df
import json
import pandas as pd

path_webtxts = '/Users/celistittse/Documents/CultTech/Data/raw_data/gt_ddg_txts/'
path_web_jsonl = '/Users/celistittse/Documents/CultTech/Data/raw_data/gtWebText2.jsonl'
path_save_ner_df = '/Users/celistittse/Documents/CultTech/Data/derived_data/'

# read copy paste txts from people -> {id.0/1 : [urls]}
raw_pages_l = read_txt_gt_files(path=path_webtxts)

# transform json to jsonl
# >> crawl with scrape_fulltext_websites scrapy (seperate script)


# open scraped websites texts
with open(path_web_jsonl, encoding='utf-8-sig') as json_file:
    # open json file
    results = [json.loads(json_str) for json_str in list(json_file)]

# filter SM, music, and movies websites
filtered_pages_l = post_process_names_d(raw_json_path=None,
                                        results=results)

# NER models
# install models first "python -m spacy download {model_name}
wiki_ner_str = "Babelscape/wikineural-multilingual-ner"
news_ner_str = "dslim/distilbert-NER"

ner_dfs = []
for model_name in [wiki_ner_str, news_ner_str]:
    nlp = load_huggingface_ners(model_s=model_name)

    ner_df = sort_ner_results_to_df(raw_data=filtered_pages_l, nlp=nlp)
    ner_df['model_name'] = model_name
    ner_dfs.append(ner_df)

ner_df_merged = pd.concat(ner_dfs, ignore_index=True, sort=False)

ner_df_merged.to_excel(path_save_ner_df+'gt_web_ners_2.xlsx', index=False)

#