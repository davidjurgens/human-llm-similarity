import pandas as pd
import os
import numpy as np
import nltk
from tqdm import tqdm
#import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math

def preprocess_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

## return a dict with the number of pos tags normalised by length of the turn (a probability distrubution)
def POS_tags(input_list):
    output_list = []
    for text in tqdm(input_list):
        pos_count = {
            "ADJ": 0,
            "ADP": 0,
            "ADV": 0,
            "CONJ": 0,
            "DET": 0,
            "NOUN": 0,
            "NUM": 0,
            "PRT": 0,
            "PRON": 0,
            "VERB": 0,
            ".": 0,
            "X": 0
        }
        tokens = nltk.word_tokenize(text)
        tag = nltk.pos_tag(tokens, tagset='universal')
        for t in tag:
            pos_count[t[1]] += 1/len(tokens)
        output_list.append(pos_count)
    return output_list

def pos_tag_metric(human_list, llm_list=None):
    human_pos = POS_tags(human_list)
    if llm_list is None:
        return human_pos
    llm_pos = POS_tags(llm_list)
    pos_jsd = []

    for human, llm in tqdm(zip(human_pos, llm_pos), total = len(human_pos)):
        human = list(human.values())
        llm = list(llm.values())

        sim = 1 - jensenshannon(human, llm)
        if math.isnan(sim):
            sim = 1
        pos_jsd.append(sim)

    return human_pos, llm_pos, pos_jsd


if __name__ == "__main__":
    mod_dir = '/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/'
    all_files = os.listdir(mod_dir)

    for fname in all_files:
        print("Working for the file: ", mod_dir + fname)

        # to get each dataframe
        data = pd.read_json(mod_dir + fname, orient='records', lines=True)
        vals = [c for c in data.columns if c.startswith('Prompt_')]
        ids = [c for c in data.columns if c not in vals]
        data = pd.melt(data, id_vars = ids, value_vars = vals, var_name = 'prompt', value_name = 'llm_turn_3')
        data = data[data.llm_turn_3 != '[INVALID_DO_NOT_USE]']

        # cross tab on no response
        pd.crosstab((data['human_turn_3'] == '[no response]'), (data['llm_turn_3'] == '[no response]'))

        print("Length of file: ", len(data))

        pos_jsd = pos_tag_metric(data['human_turn_3'], data['llm_turn_3'])
        data['POS_JSD'] = pos_jsd

        data.to_csv("pos_results_"+fname.split(".")[0] + ".csv")
        print("POS results saved at: ", "pos_results_"+fname)