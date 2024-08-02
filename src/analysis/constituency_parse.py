import pandas as pd
import os
import numpy as np
import benepar
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
from textwrap3 import wrap

def preprocess_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

## return a dict with the number of pos tags normalised by length of the turn (a probability distrubution)
def parse_tree(input_list):
    output_list = []
    parser = benepar.Parser("benepar_en2")
    for text in tqdm(input_list):
        const_count = {
            "ADJP":0, "-ADV":0, "ADVP":0, "-BNF":0, "CC": 0, "CD": 0, "-CLF": 0, "-CLR": 0,
            "CONJP": 0, "-DIR": 0, "DT": 0, "-DTV": 0, "EX":0, "-EXT":0, "FRAG": 0, "FW": 0,
            "-HLN": 0, "IN":0, "INTJ":0, "JJ":0, "JJR":0, "JJS":0, "-LGS":0, "-LOC":0, "-LRB-": 0, "LS":0,
            "LST": 0, "MD": 0, "-MNR":0, "NAC": 0, "NN":0, "NNS":0, "NNP":0, "NNPS":0, "-NOM":0,
            "NP":0, "NX":0, "PDT":0, "POS":0, "PP":0, "-PRD":0, "PRN":0, "PRP":0, "-PRP":0,
            "PRP$":0, "PRP-S":0, "PRT":0, "-PUT":0, "QP": 0, "RB":0, "RBR":0, "RBS":0, "RP":0, "-RRB-":0,
            "RRC":0, "S": 0, "SBAR": 0, "SBARQ":0, "-SBJ":0, "SINV":0, "SQ":0, "SYM":0, "-TMP":0,
            "TO":0, "-TPC":0, "-TTL":0, "UCP":0, "UH":0, "VB":0, "VBD":0, "VBG":0, "VBN":0,
            "VBP":0, "VBZ":0, "-VOC":0, "VP":0, "WDT":0, "WHADJP":0, "WHADVP":0, "WHNP":0,
            "WHPP":0, "WP":0, "WP$":0, "WP-S":0, "WRB":0, "X":0
        }
        complete_parse_string = ""
        text_chunks = wrap(text, 356)
        for chunk in text_chunks:
            trees = parser.parse_sents(chunk)
            for item in trees:
                parse_string = ' '.join(str(item).split())
                complete_parse_string += " " + parse_string
        consts = [x.strip() for x in re.findall("\(([A-Z-]+)", complete_parse_string)]
        for x in consts:
            if x not in const_count.keys():
                print("x: ", x)
            const_count[x] += 1/len(consts)
        output_list.append(const_count)
    return output_list

def const_parse_metric(human_list, llm_list):
    human_pos = parse_tree(human_list)
    llm_pos = parse_tree(llm_list)
    pos_jsd = []

    for human, llm in tqdm(zip(human_pos, llm_pos), total = len(human_pos)):
        human = list(human.values())
        llm = list(llm.values())

        dist = 1-jensenshannon(human, llm)
        if math.isnan(dist):
            dist = 1
        pos_jsd.append(dist)

    return human_pos, llm_pos, pos_jsd

if __name__ == "__main__":
    mod_dir = '/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/'
    # all_files = os.listdir(mod_dir)

    # for fname in all_files:
    # print("Working for the file: ", mod_dir + fname)

    # to get each dataframe
    fname = "llama-3.1-8B-smaller.jsonl"
    data = pd.read_json(mod_dir + fname, orient='records', lines=True)
    vals = [c for c in data.columns if c.startswith('Prompt_')]
    ids = [c for c in data.columns if c not in vals]
    data = pd.melt(data, id_vars = ids, value_vars = vals, var_name = 'prompt', value_name = 'llm_turn_3')
    data = data[data.llm_turn_3 != '[INVALID_DO_NOT_USE]']

    # cross tab on no response
    pd.crosstab((data['human_turn_3'] == '[no response]'), (data['llm_turn_3'] == '[no response]'))

    print("Length of file: ", len(data))

    const_parse_human, const_parse_llm, const_parse_jsd = const_parse_metric(data['human_turn_3'], data['llm_turn_3'])
    #print(const_parse_jsd)
    # data['CONST_PARSE_JSD'] = pos_jsd

    # data.to_csv("pos_results_"+fname.split(".")[0] + ".csv")
    # print("POS results saved at: ", "pos_results_"+fname)

    # print(const_parse_metric(['I am a girl. She is also a girl', 'She is a dancer. I am a singer.'], ['He is a boy. You are a boy. So many boys!', 'I am a researcher. I do research. I may suck at it, but I am at it.']))
