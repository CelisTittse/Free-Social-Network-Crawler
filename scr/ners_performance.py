import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import TwoSlopeNorm

# 100 random website checks
# 3 models, wikibased, distilner, spacy multiling

check_path = '/Users/celistittse/Documents/CultTech/Data/derived_data/gt_check_ners_2.xlsx'
true_websites_path = '/Users/celistittse/Documents/CultTech/Data/derived_data/trueTotalPersWebsites.xlsx'

# per website per model:
# TP: n TP [1]
# FP: n TP [0]
# FN: n total manual - n TP [1]

# recall = TP / TP + FN
# precision = TP / TP + FP
# F1 = (2 * precision * recall) / precision + recall

# recall = TP / TP + FN
def recall (tp: int, fn: int) -> float:
    if any([True for elem in [tp, fn] if not isinstance(elem, int)]):
        print("recall elements are not ints")
        return None

    return tp / (tp + fn)

# precision = TP / TP + FP
def precision (tp: int, fp: int) -> float:
    if any([True for elem in [tp, fp] if not isinstance(elem, int)]):
        print("precision elements are not ints")
        return None

    return tp / (tp + fp)


# dict for url : how many people should have been identified
def url_true_persons_dict (true_df: pd.DataFrame):
    return dict(zip(true_df['url'], true_df['total_names']))


def retrieve_tp_fp_fn(true_df: pd.DataFrame, subdf: pd.DataFrame) -> list:
    # All real persons on website
    url_true_n_d = url_true_persons_dict(true_df=true_df)

    TP = 0
    FP = 0
    FN = 0

    for url, website_subdf in subdf.groupby('url'):
        true_n = url_true_n_d[url]

        pred_n = len(website_subdf[website_subdf['TP'] == 1])
        false_pred_n = len(website_subdf[website_subdf['TP'] == 0])

        fn_url = true_n - pred_n

        # if more real names are identified than written accept the max true names
        if abs(fn_url) != fn_url:
            pred_n = true_n
            fn_url = 0

        TP += pred_n
        FP += false_pred_n
        FN += fn_url

    return TP, FP, FN

# create confusion matrix
def confusion_matrix_plot (tp: int, fp: int, fn: int, tn: int = 0) -> plt.plot:
    # Create 2×2 matrix
    matrix = np.array([[tp, fn],
                       [fp, tn]])

    print(matrix)

    labels = [['TP', 'FN'],
              ['FP', 'TN']]

    fig, ax = plt.subplots()

    norm = TwoSlopeNorm(vmin=0, vcenter=150, vmax=300)
    cax = ax.matshow(matrix, cmap='Reds') #, norm=norm)
    fig.colorbar(cax, label="Identified Names")

    # Annotate with values and labels
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{labels[i][j]}\n{matrix[i, j]}',
                    va='center', ha='center', color='black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels(['Actual Positive', 'Actual Negative'])

    ax.set_title(f'Precision: {precision(tp=tp, fp=fp):.3f}, '
                 f'Recall: {recall(tp=tp, fn=fn):.3f}')


    return fig

# confusion matrix performance scores and print them (with plot)
def calc_performances (tp: int, fp: int, fn:int, tn: int = 0, plot=False) -> plt.plot:
    print(f'Precision: {precision(tp=tp, fp=fp)}')

    print(f'Recall: {recall(tp=tp, fn=fn)}')

    if plot:
        # create confusion matrix
        conf_mtx_plt = confusion_matrix_plot(tp=tp, fp=fp, fn=fn, tn=tn)
        return conf_mtx_plt

def calc_length_names (name: str) -> int:
    return len(name.split(' ')) if isinstance(name, str) else None

def plot_n_words_tp (df: pd.DataFrame) -> None:
    save_dir = '/Users/celistittse/Documents/CultTech/plots/'
    df = df.dropna(subset=['TP', 'words_n'])

    # Step 1: Categorize number of words
    df['words_group'] = df['words_n'].apply(
        lambda x: str(x) if x <= 5 else '6+'
    )

    # Step 3: Group and count occurrences
    matrix_df = df.groupby(['TP', 'words_group']).size().unstack(fill_value=0).reindex([0, 1])

    print(matrix_df.to_string())

    # Step 4: Plot matrix
    fig, ax = plt.subplots(figsize=(8, 3))
    cax = ax.matshow(matrix_df.values, cmap='Blues')
    fig.colorbar(cax)

    # Annotate matrix with counts
    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            ax.text(j, i, str(matrix_df.iloc[i, j]), va='center', ha='center', color='black')

    # Axes setup
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Not Real Names', 'Real Names'])

    plt.xlabel('Words in Target Name')
    plt.title('Words Count vs True Positives')
    plt.tight_layout()
    plt.savefig(f'{save_dir}word_count_filter.png', dpi=600)
    plt.show()

# regex filter on string
def re_filter (name: str) -> bool:
    if not isinstance(name, str):
        return None

    # digit in name
    if re.search(pattern='[0-9]', string=name):
        return 'Digit'

    # first letter name is not capitalised
    if name[0] == name[0].lower():
        return 'Not Capitalised'

    # last word is one character
    if len(name.split(' ')[-1]) == 1:
        return 'No lastname'

    return 'Name'

def plot_re_filter (df: pd.DataFrame) -> None:
    save_dir = '/Users/celistittse/Documents/CultTech/plots/'

    # Step 1: Group and count occurrences
    matrix_df = df.groupby(['TP', 're_filt']).size().unstack(fill_value=0).reindex([0, 1])

    print(matrix_df.to_string())

    # Step 4: Plot matrix
    fig, ax = plt.subplots(figsize=(8, 3))
    cax = ax.matshow(matrix_df.values, cmap='Reds')
    fig.colorbar(cax)

    # Annotate matrix with counts
    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            ax.text(j, i, str(matrix_df.iloc[i, j]), va='center', ha='center', color='black')

    # Axes setup
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Not Real Names', 'Real Names'])

    plt.xlabel('Type filter')
    plt.title('Regex filter vs True Positives')
    plt.tight_layout()
    plt.savefig(f'{save_dir}re_filter.png', dpi=600)
    plt.show()


def plot_model_performance (ner_df: pd.DataFrame, true_df: pd.DataFrame, filtered=False) -> None:
    save_dir = '/Users/celistittse/Documents/CultTech/plots/'
    for model, subdf in ner_df.groupby('model_name'):
        tp, fp, fn = retrieve_tp_fp_fn(subdf=subdf, true_df=true_df)

        print(model)
        print(f'tp: {tp}, fp: {fp}, fn: {fn}')
        fig_conf = calc_performances(tp=tp, fp=fp, fn=fn, plot=True)
        fig_conf.suptitle(f'{model}', fontsize=16)

        model = model.replace('/', '_')

        save_path = f'{save_dir}conf_{model}.png' if not filtered else f'{save_dir}conf_filt_{model}.png'

        plt.tight_layout()
        # plt.savefig(save_path, dpi=600)
        # plt.show()
        plt.close(fig_conf)
        print()

# read dfs
trueDf = pd.read_excel(true_websites_path)
nersDf = pd.read_excel(check_path)

# plot_model_performance(true_df=trueDf, ner_df=nersDf)

nersDf['words_n'] = nersDf['target_name'].apply(lambda x: calc_length_names(x))

# plot number of words vs TP
plot_n_words_tp(nersDf)

# filter by regex on names:
# 1. digit in names, 2. last name is one character, 3. not capitalised first name
nersDf['re_filt'] = nersDf['target_name'].apply(lambda x: re_filter(x))

# print(len(nersDf['target_name'].unique()))
# print(len(nersDf['url'].unique()))
# print(Counter(nersDf['re_filt'].to_list()))
# print(np.mean(nersDf['words_n']), np.std(nersDf['words_n']))

# plot regex filters vs TP
plot_re_filter(nersDf)

nerFilt = nersDf[(nersDf['words_n'] <= 4) & (nersDf['re_filt'] == 'Name')]



# plot_model_performance(true_df=trueDf, ner_df=nerFilt, filtered=True)
p=1
p=2
# # filter on used model
# nerFilt = nerFilt[nerFilt['model_name'] == 'Babelscape/wikineural-multilingual-ner']
#
# nerFilt.drop(columns=['re_filt'])

# END NETWORK PERFORMANCE
def plot_network_performance (tp: int, fp: int, fn:int, tn:int, model:str, plot=False) -> None:
    save_dir = '/Users/celistittse/Documents/CultTech/plots/'

    fig_conf = calc_performances(tp=tp, fp=fp, fn=fn, tn=tn, plot=True)
    fig_conf.suptitle(f'{model}', fontsize=16)

    model = model.replace(' ', '_')

    save_path = f'{save_dir}conf_{model}.png'

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()
    plt.close(fig_conf)
    print()

# plot_network_performance(tp=65, fp=29, fn=287, tn=1617-65-29-287,
#                          model='Network Performance',
#                          plot=True)
