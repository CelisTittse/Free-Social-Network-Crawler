import pandas as pd
from collections import Counter

save_dir = '/Users/celistittse/Documents/CultTech/Data/derived_data/testperson_contacts/'
web_ners_path = '/Users/celistittse/Documents/CultTech/Data/derived_data/gt_web_ners.xlsx'

ners_df = pd.read_excel(web_ners_path)

# filter distil bert because the results suck
ners_df = ners_df[ners_df['model_name'] == 'Babelscape/wikineural-multilingual-ner']

def differenate_source_names_dfs (df: pd.DataFrame)-> None:
    for source_name, subdf in df.groupby('source_name'):
        save_s_name = source_name.replace(' ', '_')

        target_df = subdf[['target_name', 'url']]

        # filter on names that are too long <40 chr
        target_df = target_df[target_df['target_name'].str.len() < 40]

        # only yield website
        target_df['url'] = target_df['url'].apply(lambda x: x.replace('https://', '').split('/')[0])

        target_df = target_df.drop_duplicates(subset='target_name')

        target_df['known [1 (yes), 0 (no)]'] = 0
        target_df['wrong identified [1 (yes), 0 (no)]'] = 0

        target_df.to_excel(f'{save_dir}{save_s_name}.xlsx', index=False)

differenate_source_names_dfs(df=ners_df)