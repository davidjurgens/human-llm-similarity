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

    # standardize human and ai rates based on human rates
    human_mean = human_rates.mean()
    human_std = human_rates.std()

    standard_human = (human_rates - human_mean) / human_std
    standard_ai = (ai_rates - human_mean) / human_std

    # calculate similarity LLM - human
    return standard_ai - standard_human


def punctuation(df, human_col, ai_col):

    def punctuation_rate(text):

        if len(text) < 1:
            return np.nan

        # create a dictionary of all punctuation marks to store counts
        punctuation_counts = {punct: 0 for punct in string.punctuation}

        # iterate over each character
        for char in text:
            # if character is punctuation, increment count in dict
            if char in punctuation_counts:
                punctuation_counts[char] += 1

        # divide counts by length of string to get rate
        punctuation_counts = {punct: count / len(text) for punct, count in punctuation_counts.items()}

        return np.array(list(punctuation_counts.values()))
    
    # calculate punctuation distributions for human and ai
    human_distributions = df[human_col].apply(punctuation_rate)
    ai_distributions = df[ai_col].apply(punctuation_rate)

    outputs = []
    
    # calculate 1 - JSD for all examples
    for i in df.index:
        outputs.append(1 - jensenshannon(human_distributions[i], ai_distributions[i]))
    return outputs