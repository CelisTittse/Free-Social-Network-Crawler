import pandas as pd
import json
from itertools import product
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm

# FIND MINIMAL SENTENCE DISTANCE BETWEEN SOURCE AND TARGET NAMES

# ================== DATASETS ===================
path_web_jsonl = '/Users/celistittse/Documents/CultTech/Data/raw_data/gtWebText2.jsonl'
ner_df_path = '/Users/celistittse/Documents/CultTech/Data/derived_data/gt_web_ners_2.xlsx'

wikiNerDf = pd.read_excel(ner_df_path)

def load_url_sentences(path_web_jsonl):
    """Load and preprocess URL sentences more efficiently"""
    url_sentences = {}
    with open(path_web_jsonl, encoding='utf-8-sig') as f:
        for line in f:
            result = json.loads(line)

            url = result['url']

            if url in set(wikiNerDf['url']):
                # Split once and store as list for later use
                url_sentences[url] = [
                    elem.strip() for elem in result['full_text'].split('.') if elem.strip()
                ]
    return url_sentences


def find_min_distance(row: pd.Series, sent_l: list) -> int:
    s_name = str(row['source_name']).lower()
    t_name = str(row['target_name']).lower()

    s_indices = []
    t_indices = []

    # Pre-compile lower case versions if many sentences
    for i, sent in enumerate(sent_l):
        lower_sent = sent.lower()
        if s_name in lower_sent:
            s_indices.append(i)
        if t_name in lower_sent:
            t_indices.append(i)

    if not s_indices or not t_indices:
        return -1  # Indicate no matches found

    # Calculate minimal distance between all pairs
    min_distance = float('inf')
    for s_pos, t_pos in product(s_indices, t_indices):
        distance = abs(s_pos - t_pos)
        if distance < min_distance:
            min_distance = distance
            if min_distance == 0:  # Can't get better than this
                break

    return min_distance if min_distance != float('inf') else -1


def process_wiki_ner(wikiNerDf, url_sentences):
    """Process the DataFrame in chunks for better memory management"""
    results = []

    for url, subDf in wikiNerDf.groupby('url'):
        sentences = url_sentences.get(url)
        if not sentences:
            continue

        # Vectorize the operation within each group
        sub_results = subDf.apply(
            lambda row: find_min_distance(row, sentences),
            axis=1
        )
        results.append(sub_results)

    return pd.concat(results)

def url_languages (url_sents):
    url_languages = {}
    for url, sents in tqdm(url_sents.items()):
        # only use first 3 sentence
        sent_str = '. '.join(sents[:3]) if len(sents) > 3 else ' '.join(sents)

        # keep merged sentence below 514 characters, because expanded size of the tensor
        sent_str = sent_str if len(sent_str) < 514 else sent_str[:514]

        language_l = classifier(sent_str)

        language = language_l[0]['label']

        language = 'nl-NL' if language == 'af-ZA' else language

        url_languages[url] = language

    return url_languages

def insert_languages (url: pd.Series) -> str:
    # return language from url lanuage dict
    return urlLanuagesD.get(url)


model_name = 'qanastek/51-languages-classifier'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)


# Main execution
urlSentencesD = load_url_sentences(path_web_jsonl)

urlLanuagesD = url_languages(urlSentencesD)

wikiNerDf['language'] = wikiNerDf['url'].apply(lambda x: insert_languages(x))

# wikiNerDf['min_distance'] = process_wiki_ner(wikiNerDf, urlSentencesD)


wikiNerDf.to_excel(ner_df_path)