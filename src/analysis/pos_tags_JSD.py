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
import spacy

def preprocess_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

## return a dict with the number of pos tags normalised by length of the turn (a probability distrubution)
def POS_tags_NLTK(input_list):
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

def avg_dep_dist(nlp_op):
    total_distance = 0
    for token in nlp_op:
        if token.dep_ != "punct":  # Exclude punctuation
            total_distance += abs(token.i - token.head.i)

    num_tokens = len(nlp_op)
    try:
        return total_distance / num_tokens      ## returning average distance
    except:
        return 0

def dep_depth(nlp_op):
    max_depth = 0
    for token in nlp_op:
        depth = 0
        current_token = token
        while current_token.head != current_token:
            depth += 1
            current_token = current_token.head
        max_depth = max(max_depth, depth)
    try:
        return max_depth
    except:
        return 0

def dep_breadth(nlp_op):
    children_count = {}
    for token in nlp_op:
        children_count[token] = 0
        for _ in token.children:
            children_count[token] += 1
    try:
        return max(children_count.values()), sum(children_count.values())/len(children_count.values())
    except:
        return 0, 0
    ## Returning maximum and average number of dependents
    return max(children_count.values()), sum(children_count.values())/len(children_count.values())

def nested_noun_chunks(nlp_op):
    noun_depth = {}
    for chunk in nlp_op.noun_chunks:
        noun_depth[chunk] = 0
        current_token = chunk.root
        while current_token.head != current_token:
            noun_depth[chunk] += 1
            current_token = current_token.head

    try:
        ## Returning the maximum and average of the nested noun chunks distance
        return max(noun_depth.values()), sum(noun_depth.values())/len(noun_depth.values())
    except:
        return 0, 0
    
def POS_tags_Dep_Spacy(input_list):
    nlp = spacy.load("en_core_web_sm")
    output_list_pos = []
    output_list_dep = []
    output_list_dep_dpth = []
    output_list_dep_brth = []
    output_list_dep_avg_brth = []
    output_list_dep_dep_dist = []
    output_list_dep_max_noun_chunks = []
    output_list_dep_avg_noun_chunks = []
    for text in tqdm(input_list):
        pos_count = {"ADJ": 0, "ADP": 0, "ADV": 0, "AUX": 0, "CCONJ": 0, "DET": 0, "INTJ": 0, "NOUN": 0, "NUM": 0, "PART": 0, "PRON": 0, "PROPN": 0, "PUNCT": 0, "SCONJ": 0, "SYM": 0, "VERB": 0, "X": 0, "SPACE": 0}
        dep_count = {"ROOT": 0, "acl": 0, "acomp": 0, "advcl": 0, "advmod": 0, "agent": 0, "amod": 0, "appos": 0, "attr": 0, "aux": 0, "auxpass": 0, "case": 0, "cc": 0, "ccomp": 0, "compound": 0, "conj": 0, "csubj": 0, "csubjpass": 0, "dative": 0, "dep": 0, "det": 0, "dobj": 0, "expl": 0, "intj": 0, "mark": 0, "meta": 0, "neg": 0, "nmod": 0, "npadvmod": 0, "nsubj": 0, "nsubjpass": 0, "nummod": 0, "oprd": 0, "parataxis": 0, "pcomp": 0, "pobj": 0, "poss": 0, "preconj": 0, "predet": 0, "prep": 0, "prt": 0, "punct": 0, "quantmod": 0, "relcl": 0, "xcomp": 0}
        nlp_op = nlp(text)
        for t in nlp_op:
            pos_count[t.pos_] += 1/len(nlp_op)
            dep_count[t.dep_] += 1/len(nlp_op)
        output_list_pos.append(pos_count)

        output_list_dep.append(dep_count)
        output_list_dep_dpth.append(dep_depth(nlp_op))
        max_breadth, avg_breadth = dep_breadth(nlp_op)
        output_list_dep_brth.append(max_breadth)
        output_list_dep_avg_brth.append(avg_breadth)
        output_list_dep_dep_dist.append(avg_dep_dist(nlp_op))
        max_noun_chunks, avg_noun_chunks = nested_noun_chunks(nlp_op)
        output_list_dep_max_noun_chunks.append(max_noun_chunks)
        output_list_dep_avg_noun_chunks.append(avg_noun_chunks)

    return output_list_pos, output_list_dep, output_list_dep_dpth, output_list_dep_brth, output_list_dep_avg_brth, output_list_dep_dep_dist, output_list_dep_max_noun_chunks, output_list_dep_avg_noun_chunks

def pos_tag_metric(human_list, llm_list=None):
    human_pos = POS_tags_NLTK(human_list)
    if llm_list is None:
        return human_pos
    llm_pos = POS_tags_NLTK(llm_list)
    pos_jsd = []

    for human, llm in tqdm(zip(human_pos, llm_pos), total = len(human_pos)):
        human = list(human.values())
        llm = list(llm.values())

        dist = 1-jensenshannon(human, llm)
        if math.isnan(dist):
            dist = 1
        pos_jsd.append(dist)

    return human_pos, llm_pos, pos_jsd

def pos_tag_dep_parse_metric(human_list, llm_list=None):
    human_pos, human_dep, human_dep_dpth, human_dep_brth, human_dep_avg_brth, human_dep_dep_dist, human_dep_max_noun_chunks, human_dep_avg_noun_chunks = POS_tags_Dep_Spacy(human_list)
    if llm_list is None:
        return human_pos, human_dep, human_dep_dpth, human_dep_brth, human_dep_avg_brth, human_dep_dep_dist, human_dep_max_noun_chunks, human_dep_avg_noun_chunks
    
    llm_pos, llm_dep, llm_dep_dpth, llm_dep_brth, llm_dep_avg_brth, llm_dep_dep_dist, llm_dep_max_noun_chunks, llm_dep_avg_noun_chunks = POS_tags_Dep_Spacy(llm_list)
    
    pos_jsd = []
    for human, llm in tqdm(zip(human_pos, llm_pos), total = len(human_pos)):
        human = list(human.values())
        llm = list(llm.values())

        dist = 1-jensenshannon(human, llm)
        if math.isnan(dist):
            dist = 1
        pos_jsd.append(dist)

    dep_jsd = []
    for human, llm in tqdm(zip(human_dep, llm_dep), total = len(human_dep)):
        human = list(human.values())
        llm = list(llm.values())

        dist = 1-jensenshannon(human, llm)
        if math.isnan(dist):
            dist = 1
        dep_jsd.append(dist)

    return human_pos, llm_pos, pos_jsd, human_dep, llm_dep, dep_jsd, human_dep_dpth, human_dep_brth, human_dep_avg_brth, human_dep_dep_dist, human_dep_max_noun_chunks, human_dep_avg_noun_chunks, llm_dep_dpth, llm_dep_brth, llm_dep_avg_brth, llm_dep_dep_dist, llm_dep_max_noun_chunks, llm_dep_avg_noun_chunks

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the WildChat data", required=True)
    parser.add_argument("--output_path", type=str, help="Name of the file to save the generated text", required=True)
    
    # mod_dir = '/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/'
    # op_dir = "/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/with_metrics/"
    # op_dir = "./"

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    
    # all_files = os.listdir(mod_dir)
    # for fname in all_files:
    #     if "wildchat" in fname:
    #         continue

    print("Working for the file: ", input_path)

    # to get each dataframe
    data = pd.read_json(input_path, orient='records', lines=True)
    vals = [c for c in data.columns if c.startswith('Prompt_')]
    ids = [c for c in data.columns if c not in vals]
    data = pd.melt(data, id_vars = ids, value_vars = vals, var_name = 'prompt', value_name = 'llm_turn_3')
    data = data[data.llm_turn_3 != '[INVALID_DO_NOT_USE]']
    # cross tab on no response
    pd.crosstab((data['human_turn_3'] == '[no response]'), (data['llm_turn_3'] == '[no response]'))

    print("Length of file: ", len(data))

    human_pos, llm_pos, pos_jsd, human_dep, llm_dep, dep_jsd, human_dep_dpth, human_dep_brth, human_dep_avg_brth, human_dep_dep_dist, human_dep_max_noun_chunks, human_dep_avg_noun_chunks, llm_dep_dpth, llm_dep_brth, llm_dep_avg_brth, llm_dep_dep_dist, llm_dep_max_noun_chunks, llm_dep_avg_noun_chunks = pos_tag_dep_parse_metric(data['human_turn_3'], data['llm_turn_3'])

    data.insert(len(data.columns), "human_pos", human_pos)
    data.insert(len(data.columns), "llm_pos", llm_pos)
    data.insert(len(data.columns), "spacy_pos", pos_jsd)

    data.insert(len(data.columns), "human_dep", human_dep)
    data.insert(len(data.columns), "llm_dep", llm_dep)
    data.insert(len(data.columns), "spacy_dep", dep_jsd)

    data.insert(len(data.columns), "human_dep_dpth", human_dep_dpth)
    data.insert(len(data.columns), "human_dep_brth", human_dep_brth)
    data.insert(len(data.columns), "human_dep_avg_brth", human_dep_avg_brth)
    data.insert(len(data.columns), "human_dep_dep_dist", human_dep_dep_dist)
    data.insert(len(data.columns), "human_dep_max_noun_chunks", human_dep_max_noun_chunks)
    data.insert(len(data.columns), "human_dep_avg_noun_chunks", human_dep_avg_noun_chunks)

    data.insert(len(data.columns), "llm_dep_dpth", llm_dep_dpth)
    data.insert(len(data.columns), "llm_dep_brth", llm_dep_brth)
    data.insert(len(data.columns), "llm_dep_avg_brth", llm_dep_avg_brth)
    data.insert(len(data.columns), "llm_dep_dep_dist", llm_dep_dep_dist)
    data.insert(len(data.columns), "llm_dep_max_noun_chunks", llm_dep_max_noun_chunks)
    data.insert(len(data.columns), "llm_dep_avg_noun_chunks", llm_dep_avg_noun_chunks)

    data.to_json(output_path, orient='records', lines=True)

    print("POS results saved at: ", output_path)