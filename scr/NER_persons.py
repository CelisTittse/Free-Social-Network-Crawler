from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd
from tqdm import tqdm


# ================ INSTALLING NER MODELS ===================
def load_huggingface_ners(model_s: str):
    tokenizer = AutoTokenizer.from_pretrained(model_s)
    model = AutoModelForTokenClassification.from_pretrained(model_s)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="average")

    return nlp


# ============== PARSE NER TEXTS ====================
def parse_fulltext_names(text: str, nlp) -> list[dict]:
    # parse per sentence
    ner_results = [ner_result for sent in text.split('.')
                   for ner_result in nlp(sent.strip())
                   if ner_result['entity_group'] == 'PER']

    return ner_results


result = parse_fulltext_names(text='Hallo ik ben Celis Tittse',
                              nlp=load_huggingface_ners('dslim/distilbert-NER'))
print(result)


def sort_ner_results_to_df(raw_data: list, nlp) -> pd.DataFrame:
    df_d = {'source_key': [],
            'source_name': [],
            'target_name': [],
            'score': [],
            'url': [],
            }

    for web_data in tqdm(raw_data):
        url = web_data['url']
        name = web_data['name']
        full_text = web_data['full_text']
        source_key = web_data['key']

        ner_results = parse_fulltext_names(text=full_text, nlp=nlp)

        for ner_result in ner_results:
            # dont incorporate substrings of the source name and
            # use full names (surname, last name)
            if ner_result['word'] not in name and ' ' in ner_result['word']:
                df_d['source_key'].append(source_key)
                df_d['url'].append(url)
                df_d['source_name'].append(name)
                df_d['target_name'].append(ner_result['word'])
                df_d['score'].append(ner_result['score'])

    return pd.DataFrame(df_d)
