# Author: Abraham I, UMSI, July 25, 2024
# This code is used to extract the LIWC distribution of text chunks. It also allows to compare texts using
# simple heuristics
# NOTE: you must have the LIWC dictionary in the environment of the code! We cannot share this dictionary unfortunately

from collections import Counter, defaultdict
import re
from scipy.spatial.distance import cosine, jensenshannon
import os
import pandas as pd
from tqdm import tqdm
import numpy as np


class LiwcDistExtractor:
    def __init__(self, agg_results=False, normalize=True):
        self.agg_results = agg_results
        self.normalize = normalize

    @staticmethod
    def clean_text(local_text):
        if type(local_text) is not str:
            text = ''
        local_text = local_text.lower()
        local_text = re.sub(r"what's", "what is ", local_text)
        local_text = re.sub(r"\'s", " ", local_text)
        local_text = re.sub(r"\'ve", " have ", local_text)
        local_text = re.sub(r"can't", "can not ", local_text)
        local_text = re.sub(r"n't", " not ", local_text)
        local_text = re.sub(r"i'm", "i am ", local_text)
        local_text = re.sub(r"\'re", " are ", local_text)
        local_text = re.sub(r"\'d", " would ", local_text)
        local_text = re.sub(r"\'ll", " will ", local_text)
        local_text = re.sub(r"\'scuse", " excuse ", local_text)
        local_text = re.sub('\W', ' ', local_text)
        local_text = re.sub('\s+', ' ', local_text)
        local_text = local_text.strip(' ')
        return local_text

    def extract_liwc_occurrences(self, text_list):
        """
        Pull out a dictionary with the occurrences of the LIWC categories in a given text.
        It is based on a LIWC categories mapping given in the 'liwc_file_path
        :param text_list: list
            list of strings. Each item is a sentence. For each sentence we calculate the LIWC categories occurrences
        :param liwc_file_path: str
            the path to the LIWC text file. Default: location in yalla: '/data/work/data/LIWC_Features.txt'
        :param agg_results: boolean
            whether to aggregate the results of the LIWC categories occurrences
        :return: list
            list of size len(text_list). Each item is the LIWC categories occurrences in this piece of text.

        Examples
        --------
        >>> text_to_analyse = ['I myself am here', 'Great work. Good job', 'how ARE you?']
        >>> results = extract_liwc_occurrences(text_to_analyse, agg_results=False, normalize=True)
        >>> print(results)
        """
        # reading the LIWC dictionary
        liwc_dict_path = open('LIWC_Features.txt')
        lines = liwc_dict_path.readlines()
        liwc_dict_path.close()

        liwc_cat_dict = {}  # {cat: (w1,w2,w3,...)}
        for line in lines[1:]:  # first line is a comment about the use of *
            tokens = line.strip().lower().split(', ')
            liwc_cat_dict[tokens[0]] = tokens[1:]

        # creating a LIWC regex dict
        liwc_regex_dict = {}
        for k, v in liwc_cat_dict.items():
            s = '|'.join(v)
            s = re.sub(r'\*', r'\\w*', s)
            liwc_regex_dict[k] = re.compile(r'\b(' + s + r')\b')

        # handling the input text_list (text_list is a list of sentences
        cleaned_corpus = [self.clean_text(local_text=t) for t in text_list]

        # creating a counter of the LIWC categories to the given list of texts
        liwc_categ_in_text = list()
        for cur_cc in cleaned_corpus:
            cur_liwc_distrib = dict(Counter({cat: len(regex.findall(cur_cc))
                                             for cat, regex in liwc_regex_dict.items()}))
            # sort the liwc categories (keys)
            cur_liwc_distrib = {k : cur_liwc_distrib[k] for k in sorted(cur_liwc_distrib)}# if regex.findall(cur_cc)})))
            liwc_categ_in_text.append(cur_liwc_distrib)
        if self.agg_results:
            tot_words = 0
            aggregated_res = defaultdict(int)
            # looping over each Counter object in the list
            for cur_res in liwc_categ_in_text:
                # looping over each category in the Counter (each is a LIWC category)
                for cur_categ, cnt in cur_res.items():
                    aggregated_res[cur_categ] += cnt
                    tot_words += cnt
            # returning the distribution, if needed - normalizing it
            return dict(aggregated_res) if not normalize else {word: cnt/tot_words for word, cnt in aggregated_res.items()}
        else:
            if self.normalize:
                liwc_categ_in_text_normalized = list()
                for cur_distrib in liwc_categ_in_text:
                    tot_words = max(sum(cur_distrib.values()), 1)
                    liwc_categ_in_text_normalized.append({word: cnt/tot_words for word, cnt in cur_distrib.items()})
                return liwc_categ_in_text_normalized
            # if no need to normalize
            else:
                return liwc_categ_in_text
    @staticmethod
    def liwc_similarity(text1_liwc_dict, text2_liwc_dict, method='jsd'):
        if text1_liwc_dict.keys() != text2_liwc_dict.keys():
            raise IOError("Invalid input for the similarity function. Keys must be the same name and order.")
        if method == 'cosine':
            dicts_similarity = cosine(list(text1_liwc_dict.values()), list(text2_liwc_dict.values()))
        elif method == 'manhattan':
            dicts_similarity = 1 - sum(abs(text1_liwc_dict[word] - text2_liwc_dict[word])/2.0 for word in text1_liwc_dict)
        elif method == 'jsd':
            dicts_similarity = 1 - jensenshannon(list(text1_liwc_dict.values()), list(text2_liwc_dict.values()))
        else:
            raise IOError("Invalid method value for the similarity function")
        return dicts_similarity

if __name__ == '__main__':
    mod_dir = '/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/'
    fname = os.listdir(mod_dir)[0] #there are 4 files here

    # to get each dataframe
    data = pd.read_json(mod_dir + fname, orient='records', lines=True)
    vals = [c for c in data.columns if c.startswith('Prompt_')]
    ids = [c for c in data.columns if c not in vals]
    data = pd.melt(data, id_vars = ids, value_vars = vals, var_name = 'prompt', value_name = 'llm_turn_3')
    data = data[data.llm_turn_3 != '[INVALID_DO_NOT_USE]']

    # cross tab on no response
    pd.crosstab((data['human_turn_3'] == '[no response]'), (data['llm_turn_3'] == '[no response]'))

    # produce a score comparing human vs. llm text
    liwc_extractor_obj = LiwcDistExtractor(agg_results=False, normalize=True)
    liwc_similarities = list()
    for row_idx, row_content in tqdm(data.iterrows()):
        human_liwc_distrib = liwc_extractor_obj.extract_liwc_occurrences([row_content['human_turn_3']])[0]
        llm_liwc_distrib = liwc_extractor_obj.extract_liwc_occurrences([row_content['llm_turn_3']])[0]
        similarity = liwc_extractor_obj.liwc_similarity(human_liwc_distrib, llm_liwc_distrib, method="jsd")
        liwc_similarities.append(similarity)
    data['liwc_similarities'] = liwc_similarities
    print(f"Mean and STD similarity: {np.nanmean(liwc_similarities)}, {np.nanstd(liwc_similarities)}")
    # now results can be saved using data.to_csv()


