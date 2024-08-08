#from readability import Readability
from os.path import join as opj
import pandas as pd
#from readability.exceptions import ReadabilityException
from tqdm import tqdm
import multiprocessing as mp
import textstat
import glob
import os
import numpy as np

debug_mode = False
n_cpus = 1 if debug_mode else 100
package_to_use = 'textstat'


def get_flesch_readability(idx, text, normalize=True):
    # Note! in some cases we return None!
    if len(text.split()) < 10:
        return {'flesch': None}
    else:
        readability_score = textstat.flesch_reading_ease(text)
        if readability_score < 0 or readability_score > 110:
            return {'flesch': None}
        if normalize:
            normalized_score = readability_score / 110.0
            return {'flesch': normalized_score if normalized_score < 1 else None}
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


def readability_single_column(data, package='textstat', n_cpus=100):
    input_for_pool = [(idx, cd, package) for idx, cd in enumerate(data)]
    pool = mp.Pool(processes=n_cpus)
    with pool as pool:
        cur_column_score = pool.starmap(get_flesch_readability, input_for_pool)
    return np.array(pd.DataFrame(cur_column_score))


if __name__ == '__main__':
    data_path = '/shared/0/projects/research-jam-summer-2024/data/english_only/100k_results'
    jsonl_files_found = glob.glob(opj(data_path, '*.jsonl'))
    file_to_analyze = jsonl_files_found[0]
    f_name_to_save = os.path.basename(file_to_analyze).split('.jsonl')[0] + '_readability_normalized.csv'
    required_columns = ['human_turn_3', 'Prompt_11', 'Prompt_15', 'Prompt_19']
    data = pd.read_json(file_to_analyze, orient='records', lines=True)
    # there are two ways to get the value.
    # OPTION A: pass ALL columns required for prediction
    data_subset = data[required_columns].copy()
    data_with_readability_scores = get_dataframe_readability(df=data_subset, package=package_to_use, n_cpus=n_cpus)
    # data_with_readability_scores.to_csv(opj('/shared/0/projects/research-jam-summer-2024/results', f_name_to_save), index=False)
    # print(data_with_readability_scores.shape)
    # print(data_with_readability_scores.iloc[0])

    # OPTION B: get prediction for a single column only
    # readability_scores = readability_single_column(data['Prompt_11'])

