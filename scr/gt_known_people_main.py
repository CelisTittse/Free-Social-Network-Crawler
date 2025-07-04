import pandas as pd
import json
import numpy as np
from collections import Counter

# TABLE OF CONTENT
# Datasets:
# - gt annotated people -> same last name (col: family), first and last name in target (col: self_auto)
# - NER results WikiNER (redone) -> count, search type [0, 1]
# - Websites json -> co-occur max distance, language, (one-shot web type)

# ======================== DATASETS ==========================
wd_path = '/Users/celistittse/Documents/CultTech/Data'
ner_df_path = f'{wd_path}/derived_data/gt_web_ners_2.xlsx'
gt_anno_path = f'{wd_path}/derived_data/GtContactsAnno.xlsx'
path_web_jsonl = f'{wd_path}/raw_data/gt_webtexts.jsonl'
gt_contacts_path = f'{wd_path}/raw_data/gt_contacts.xlsx'
ner_df_path = f'{wd_path}/result_data/WikiNerNames.xlsx'

nerDf = pd.read_excel(ner_df_path)

gtAnnoDf = pd.read_excel(gt_anno_path)

gtContactsDf = pd.read_excel(gt_contacts_path)

nerScrapedTargetNames = set(pd.read_excel(ner_df_path)['target_name'])

# add native language on source name in annotated source target
gtAnnoDf = pd.merge(nerDf, gtContactsDf[['source_name', 'native_language']],
                    how='left', on='source_name')



# ============= COUNT, SEARCH TYPE DICT WEBNERS ==============
def aggr_source_target_names (source_row: pd.Series, target_row: pd.Series) -> list:
    # source target list and counter dict
    s_t_l = list(zip(source_row, target_row))

    # merge source_name target_name
    s_t_l_agg = ['_'.join([s.strip(), t.strip()]) for s, t in s_t_l]

    return s_t_l_agg

def web_s_t_count (df: pd.DataFrame) -> dict:
    # dict: (source_name, target_name) : count
    s_t_l = aggr_source_target_names(df['source_name'], df['target_name'])
    return dict(Counter(s_t_l))

def web_s_t_search_type (df: pd.DataFrame) -> dict:
    # filter on search_type = 0 and make list (s_name, t_name)
    direct_df = df[df['search_type'] == 0]
    direct_s_t_l = set(aggr_source_target_names(direct_df['source_name'], direct_df['target_name']))

    s_t_set = set(aggr_source_target_names(df['source_name'], df['target_name']))

    # if source_target in direct search df than 0 else 1
    s_t_type = {s_t: 0 if s_t in direct_s_t_l else 1 for s_t in s_t_set}

    return s_t_type

def web_s_t_min_dist (df: pd.DataFrame) -> dict:
    s_t_l = aggr_source_target_names(df['source_name'], df['target_name'])

    df['source_target'] = s_t_l

    min_dis_df = df[['source_target', 'min_distance']].groupby(by=['source_target'], as_index=False).min()

    return dict(zip(min_dis_df['source_target'], min_dis_df['min_distance']))

# make per target name : "lang1, lang2 etc."
def web_s_t_languages (df: pd.DataFrame) -> dict:
    s_t_l = aggr_source_target_names(df['source_name'], df['target_name'])

    df['source_target'] = s_t_l

    lang_df = df.groupby('source_target')['language'].apply(
        lambda x: ', '.join(sorted(set(x)))
    ).reset_index()
    return dict(zip(lang_df['source_target'], lang_df['language']))

# filter on wikiNER
wikiNerDf = nerDf[nerDf['model_name'] == 'Babelscape/wikineural-multilingual-ner']

# select X.1 -> 1 for search type
wikiNerDf['search_type'] = wikiNerDf['source_key'].apply(lambda x: int(str(x).split('.')[-1]))

# source_target : count
STCountDict = web_s_t_count(df=wikiNerDf)

# source_target : direct (0) or indirect (1) if 0 somewhere than not 1
STTypeDict = web_s_t_search_type(df=wikiNerDf)

# minimal distance
STMinDistDict = web_s_t_min_dist(df=wikiNerDf)

# languages found
STLangDict = web_s_t_languages(df=wikiNerDf)

# =============== IDENTIFY SIMILAR LAST NAME =================
def similar_last_name(row: pd.Series) -> int:
    source_last_name = row['source_name'].split(' ')[-1]
    return int(source_last_name in row['target_name'])

def self_first_last_name (row: pd.Series) -> int:
    source_last_name = row['source_name'].split(' ')[-1].lower()
    source_first_name = row['source_name'].split(' ')[0].lower()

    if (source_last_name in row['target_name'].lower()
            and source_first_name in row['target_name'].lower()):
        return 1
    return 0


gtAnnoDf['family'] = gtAnnoDf.apply(similar_last_name, axis=1)

# self is included manually, but for other purposes it needs to be included
gtAnnoDf['self_auto'] = gtAnnoDf.apply(self_first_last_name, axis=1)

# ==================== CO-OCCURRENCE COUNT ====================
def webNer_dict_translation (row: pd.Series, relev_dict: dict) -> int:
    gt_s_t = '_'.join([str(row['source_name']).strip(), str(row['target_name']).strip()])

    # return (count, or type) values from webNer counter
    if gt_s_t in relev_dict:
        value = relev_dict[gt_s_t]
        return value

# not native or english = 0, native = 1, english = 2
def language_match (row: pd.Series) -> int:
    nat_lang = row['native_language']
    web_langs = row['language_l']

    if not isinstance(web_langs, str) or not isinstance(nat_lang, str):
        return None
    if nat_lang in web_langs:
        return 'native'
    if 'en-US' in web_langs:
        return 'english'
    return 'not native'

# find source_target in webpage count, type dict and return values
# if type = False -> count, if True -> type (X.[0,1])
gtAnnoDf['count'] = gtAnnoDf.apply(webNer_dict_translation, axis=1, args=(STCountDict,))

gtAnnoDf['type'] = gtAnnoDf.apply(webNer_dict_translation, axis=1, args=(STTypeDict,))
gtAnnoDf['type'] = gtAnnoDf['type'].replace({1: 'indirect', 0: 'direct'})

gtAnnoDf['min_distance'] = gtAnnoDf.apply(webNer_dict_translation, axis=1, args=(STMinDistDict,))

gtAnnoDf['language_l'] = gtAnnoDf.apply(webNer_dict_translation, axis=1, args=(STLangDict,))

gtAnnoDf['lang_match'] = gtAnnoDf.apply(language_match, axis=1)

gtAnnoDf['auto_identified'] = [1 if t_name in nerScrapedTargetNames else 0 for t_name in gtAnnoDf['target_name']]

# drop na because it is not found within scraping
gtAnnoDfClean = gtAnnoDf.dropna()

gtAnnoDfClean.to_excel(f'{wd_path}/result_data/WebNerAgr.xlsx')

