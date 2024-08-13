import os
import argparse
import torch
import random
import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from numpy import log

from analysis.pos_tags_JSD import pos_tag_metric
from analysis.liwc_dist_extractor import LiwcDistExtractor
from analysis.embedding_similarity import EmbeddingSimilarity
from analysis.capitalization_punctuation_similarity import capitalization, punctuation
from analysis.syntactic_metrics import BasicSyntacticStatistics


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


def load_hf_model(model_name):
    cache_dir = None
    if 'HF_MODEL_CACHE' in os.environ:
        cache_dir = os.environ['HF_MODEL_CACHE']
    pipe = pipeline("text-classification", model=model_name, model_kwargs={"cache_dir": cache_dir},
                    device_map='cuda', max_length=512, truncation=True, return_all_scores=True)
    return pipe


def hf_data(data, max_char_len=2000):
    for row in data:
        yield row[:max_char_len]


def run_hf_model(data, model_name, column_name):
    pipe = load_hf_model(model_name)
    # First, human 3rd turn
    new_data = []
    for out in tqdm(pipe(hf_data(data), batch_size=4), total=len(data)):
        new_data.append([s['score'] for s in out])
    return new_data


def sentiment(human, llm):
    human_sentiment = run_hf_model(human, "lxyuan/distilbert-base-multilingual-cased-sentiments-student", 'sentiment')
    llm_sentiment = run_hf_model(llm, "lxyuan/distilbert-base-multilingual-cased-sentiments-student", 'sentiment')

    return 1 - jensenshannon(human_sentiment, llm_sentiment, axis=1)


def formality(human, llm):
    human_formality = run_hf_model(human, "s-nlp/mdeberta-base-formality-ranker", 'formality')
    llm_formality = run_hf_model(llm, "s-nlp/mdeberta-base-formality-ranker", 'formality')

    return 1 - jensenshannon(human_formality, llm_formality, axis=1)


def politeness(human, llm):
    human_politeness = run_hf_model(human, "Genius1237/xlm-roberta-large-tydip", 'politeness')#[politenessr.predict([text])[0] for text in human]
    llm_politeness = run_hf_model(llm, "Genius1237/xlm-roberta-large-tydip", 'politeness')#[politenessr.predict([text])[0] for text in llm]

    # Now all of the prompts
    return 1 - jensenshannon(human_politeness, llm_politeness, axis=1)#(np.array(human_politeness) - np.array(llm_politeness)).abs()


def toxicity(human, llm):
    human_toxicity = run_hf_model(human, "s-nlp/roberta_toxicity_classifier", 'toxicity')
    llm_toxicity = run_hf_model(llm, "s-nlp/roberta_toxicity_classifier", 'toxicity')
    
    return 1 - jensenshannon(human_toxicity, llm_toxicity, axis=1)

def is_no_response(col, no_response_indicator = '[no response]'): 
    return (col == no_response_indicator).apply(lambda x: 1 if x else 0)

def jitter_prob(p, eps = 1./1000.):
    if p == 0: p = p + eps
    elif p == 1: p = p - eps
    return p
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the WildChat data",
                        required=True)
    parser.add_argument("--output_path", type=str, help="Name of the file to save the generated text",
                        required=True)
    parser.add_argument("--metrics", type=str, default='all', help="Comma separated list of the metrics you want to run, or 'all' to run all")
    parser.add_argument("--seed", type=int, help="Random seed",
                        default=1000)

    args = parser.parse_args()

    enforce_reproducibility(args.seed)

    input_path = args.input_path
    output_path = args.output_path
    metrics = args.metrics.split(",")

    # to get each dataframe
    data = pd.read_json(input_path, orient='records', lines=True)

    if 'prompt' not in data.columns:
        vals = [c for c in data.columns if c.startswith('Prompt_')]
        ids = [c for c in data.columns if c not in vals]
        data = pd.melt(data, id_vars=ids, value_vars=vals, var_name='prompt', value_name='llm_turn_3')
        data = data[data.llm_turn_3 != '[INVALID_DO_NOT_USE]']

    # compare whether the llm and human produces a response at the same time
    if 'all' in metrics or 'end' in metrics:
        human_end = is_no_response(data['human_turn_3']) 
        llm_end = is_no_response(data['llm_turn_3'])
        data.insert(len(data.columns), "human_end", human_end)
        data.insert(len(data.columns), "llm_end", llm_end)
        data.insert(len(data.columns), "metric_end", (human_end == llm_end).astype(int))

    #subset to cases where llm and human produce a response for the remaining metrics
    data_full = data
    data = data_full[(is_no_response(data_full['human_turn_3']) == 0) & 
                     (is_no_response(data_full['llm_turn_3']) == 0) & 
                     (data_full['human_turn_3'].apply(len) > 0)]
    
    # produce a score comparing human vs. llm text
    if 'all' in metrics or 'lexical' in metrics:
        args.no_response_indicators = '[no response]'
        args.metrics = 'char_count,word_count,typo_count'
        bss = BasicSyntacticStatistics(args)
        #human metrics
        human = bss.get_counts(data['human_turn_3'])
        human['p_typo'] = human['typo_count'] / human['word_count']
        #llm metrics
        llm = bss.get_counts(data['llm_turn_3'])
        llm['p_typo'] = llm['typo_count'] / llm['word_count']
        #comparison
        nchar = (human['char_count'] - llm['char_count']) / human['char_count']
        typo = 1 - jensenshannon(np.array([(p, 1-p) for p in human['p_typo'].apply(jitter_prob)]), 
                                 np.array([(p, 1-p) for p in llm['p_typo'].apply(jitter_prob)]), axis=1)
        data.insert(len(data.columns), "metric_typo", typo)
        
    if 'all' in metrics or 'capitalization' in metrics:
        cap = capitalization(data, 'human_turn_3', 'llm_turn_3')
        data.insert(len(data.columns), "metric_capitalization", cap)

    if 'all' in metrics or 'grammar' in metrics:
        args.no_response_indicators = '[no response]'
        args.metrics = 'grammar_error_count'
        bss = BasicSyntacticStatistics(args)
        #human metrics
        human = bss.get_counts(data['human_turn_3'])
        human['p_grammar'] = human['grammar_error_count'] / human['word_count']
        #llm metrics
        llm = bss.get_counts(data['llm_turn_3'])
        llm['p_grammar'] = llm['grammar_error_count'] / llm['word_count']
        #comparison
        grammar = 1 - jensenshannon(np.array([(p, 1-p) for p in human['p_grammar'].apply(jitter_prob)]),
                                    np.array([(p, 1-p) for p in llm['p_grammar'].apply(jitter_prob)]), axis=1)
        data.insert(len(data.columns), "metric_grammar", grammar)
        
    if 'all' in metrics or 'punctuation' in metrics:
        cap = punctuation(data, 'human_turn_3', 'llm_turn_3')
        data.insert(len(data.columns), "metric_punctuation", cap)

    if 'all' in metrics or 'pos' in metrics:
        pos = pos_tag_metric(data['human_turn_3'], data['llm_turn_3'])
        data.insert(len(data.columns), "metric_pos", pos)

    if 'all' in metrics or 'sbert' in metrics:
        embeddings = EmbeddingSimilarity()
        embeddings_1 = embeddings.get_embeddings(data['human_turn_3'])
        embeddings_2 = embeddings.get_embeddings(data['llm_turn_3'])

        similarity = embeddings.cosine_similarity(embeddings_1, embeddings_2)
        data.insert(len(data.columns), "metric_sbert", similarity)

    if 'all' in metrics or 'sentiment' in metrics:
        sentiment = sentiment(data['human_turn_3'], data['llm_turn_3'])
        data.insert(len(data.columns), "metric_sentiment", sentiment)

    if 'all' in metrics or 'semantic' in metrics:
        args.no_response_indicators = '[no response]'
        args.metrics = 'bleu,rogue,luar_similarity' 
        bss = BasicSyntacticStatistics(args)
        df_metrics = bss.get_metrics(data['human_turn_3'], data['llm_turn_3'])
        rouge = 1 - jensenshannon(np.array([(p, 1-p) for p in human['p_grammar'].apply(jitter_prob)]),
                                  np.array([(p, 1-p) for p in llm['p_grammar'].apply(jitter_prob)]), axis=1)
        data = pd.concat([data, df_metrics], axis=1)
    
    if 'all' in metrics or 'formality' in metrics:
        formality = formality(data['human_turn_3'], data['llm_turn_3'])
        data.insert(len(data.columns), "metric_formality", formality)

    if 'all' in metrics or 'politeness' in metrics:
        politeness = politeness(data['human_turn_3'], data['llm_turn_3'])
        data.insert(len(data.columns), "metric_politeness", politeness)
        
    if 'all' in metrics or 'toxicity' in metrics:
        toxicity = toxicity(data['human_turn_3'], data['llm_turn_3'])
        data.insert(len(data.columns), "metric_toxicity", toxicity)

    if 'all' in metrics or 'liwc' in metrics:
        liwc_extractor_obj = LiwcDistExtractor(agg_results=False, normalize=True)
        human_liwc = liwc_extractor_obj.extract_liwc_occurrences(data['human_turn_3'].to_list())
        llm_liwc = liwc_extractor_obj.extract_liwc_occurrences(data['llm_turn_3'].to_list())
        liwc = [liwc_extractor_obj.liwc_similarity(human, llm, method="jsd") for human, llm in zip(human_liwc, llm_liwc)]
        data.insert(len(data.columns), "metric_liwc", liwc)

    data_full.to_json(re.sub('.json','_end.json',output_path), orient='records', lines=True)
    data.to_json(output_path, orient='records', lines=True)