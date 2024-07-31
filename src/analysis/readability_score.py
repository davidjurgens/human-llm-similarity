from readability import Readability
from os.path import join as opj
import pandas as pd
from readability.exceptions import ReadabilityException
from tqdm import tqdm
import multiprocessing as mp
import textstat
import glob
import os

debug_mode = False
n_cpus = 1 if debug_mode else 100
package_to_use = 'textstat'


# OLD CODE - not used
# def get_readability(idx, text, package='textstat'):
#     if package == 'textstat':
#         if len(text.split()) < 10:
#             return {'flesch_kincaid': -1, 'flesch': -1}
#         else:
#             return {'flesch_kincaid': textstat.flesch_kincaid_grade(text), 'flesch': textstat.flesch_reading_ease(text)}
#     elif package == 'readability':
#         try:
#             r = Readability(text)
#             return {'flesch_kincaid': r.flesch_kincaid().score, 'flesch': r.flesch().score}
#         # Handle cases with less than 100 words
#         except ReadabilityException:
#             return {'flesch_kincaid': -1, 'flesch': -1}
#     else:
#         raise IOError("Invalid input package name provided.")


def get_flesch_readability(idx, text, normalize=True):
    if len(text.split()) < 10:
        return {'flesch': -1}
    else:
        readability_score = textstat.flesch_reading_ease(text)
        if readability_score < 0:
            return {'flesch': -1}
        if normalize:
            normalized_score = readability_score / 110.0
            return {'flesch': normalized_score if normalized_score < 1 else -1}
        else:
            return {'flesch': readability_score}


def get_dataframe_readability(df, package='textstat', n_cpus=100):
    # looping over each column in the dataframe
    df_combined = df.copy()
    columns = df_combined.columns
    for column_name in tqdm(columns):
        column_data = df[column_name]
        # getting the score in a mp way
        input_for_pool = [(idx, cd, package) for idx, cd in enumerate(column_data)]
        pool = mp.Pool(processes=n_cpus)
        with pool as pool:
            #cur_column_score = pool.starmap(get_readability, input_for_pool)
            cur_column_score = pool.starmap(get_flesch_readability, input_for_pool)
        new_column_names = [n + '_' + column_name for n in cur_column_score[0].keys()]
        cur_column_score_df = pd.DataFrame(cur_column_score)
        cur_column_score_df.columns = new_column_names
        df_combined = pd.concat([df_combined, cur_column_score_df], axis=1)
    return df_combined


if __name__ == '__main__':
    data_path = '/shared/0/projects/research-jam-summer-2024/data/english_only/100k_results'
    jsonl_files_found = glob.glob(opj(data_path, '*.jsonl'))
    file_to_analyze = jsonl_files_found[0]
    f_name_to_save = os.path.basename(file_to_analyze).split('.jsonl')[0] + '_readability_normalized.csv'
    required_columns = ['human_turn_3', 'Prompt_11', 'Prompt_15', 'Prompt_19']
    data = pd.read_json(file_to_analyze, orient='records', lines=True)
    data_subset = data[required_columns].copy()
    data_with_readability_scores = get_dataframe_readability(df=data_subset, package=package_to_use, n_cpus=n_cpus)
    #data_with_readability_scores.to_csv(opj('/shared/0/projects/research-jam-summer-2024/results', f_name_to_save), index=False)
    print(data_with_readability_scores.shape)
    print(data_with_readability_scores.iloc[0])
