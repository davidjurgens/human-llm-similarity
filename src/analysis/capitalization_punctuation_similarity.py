import pandas as pd
import numpy as np
import string
from scipy.spatial.distance import jensenshannon


def capitalization(df, human_col, ai_col):
    def capital_rate(text):
        # count number of capital letters
        capital_count = sum(1 for char in text if char.isupper())

        # count number of aplhabet letters (i.e., letters that can be capitalized)
        total_letter_count = sum(1 for char in text if char.isalpha())

        if total_letter_count > 0:
            return capital_count / total_letter_count
        else:
            return np.nan
    
    # calculate capital rate for human and ai
    human_rates = df[human_col].apply(capital_rate)
    ai_rates = df[ai_col].apply(capital_rate)

    # calculate similarity LLM - human
    return human_rates, ai_rates, ai_rates - human_rates


def punctuation(df, human_col, ai_col=None, lang="en"):

    def punctuation_rate(text, lang="en"):
        # Translate Chinese punctuations into English
        table = {ord(f):ord(t) for f,t in zip(
            u'，。！？【】（）％＃＠＆１２３４５６７８９０“”‘’：；',
            u',.!?[]()%#@&1234567890""'':;')}
        if lang == "cn":
            text = text.translate(table)

        # create a dictionary of all punctuation marks to store counts
        punctuation_counts = {punct: 0 for punct in string.punctuation}
        
        if len(text) < 1:
            return punctuation_counts

        # iterate over each character
        for char in text:
            # if character is punctuation, increment count in dict
            if char in punctuation_counts:
                punctuation_counts[char] += 1

        # divide counts by length of string to get rate
        punctuation_counts = {punct: count / len(text) for punct, count in punctuation_counts.items()}

        return punctuation_counts
    
    # calculate punctuation distributions for human and ai
    human_distributions = df[human_col].apply(punctuation_rate, lang=lang)
    
    if ai_col is None:
        return human_distributions
        
    ai_distributions = df[ai_col].apply(punctuation_rate, lang=lang)

    outputs = []
    
    # calculate MSE for all examples
    for i in df.index:
        outputs.append(np.sqrt(np.mean((np.array(list(ai_distributions[i].values())) -
                                        np.array(list(human_distributions[i].values()))) ** 2)))
    return human_distributions, ai_distributions, outputs
