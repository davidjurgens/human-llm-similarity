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
import re
import lmppl
from argparse import Namespace

from analysis.pos_tags_JSD import pos_tag_metric
from analysis.liwc_dist_extractor import LiwcDistExtractor
from analysis.embedding_similarity import EmbeddingSimilarity
from analysis.capitalization_punctuation_similarity import capitalization, punctuation
from analysis.syntactic_metrics import BasicSyntacticStatistics
# from analysis.subjectivity import SubjectivityAnalyzer
# from analysis.factuality_eval import get_align_score
# from analysis.constituency_parse import const_parse_metric
from analysis.readability_score import get_flesch_readability, readability_single_column


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
        new_data.append([s for s in out])
    return new_data


def sentiment(human, llm=None):
    def convert_to_scalar(data):
        score_map = {
            'negative': 0,
            'neutral': 0.5,
            'positive': 1,
        }
        out_data = []
        for d in data:
            out_data.append(sum([score_map[s['label']]*s['score'] for s in d]))
        return out_data

    human_sentiment = convert_to_scalar(run_hf_model(human, "lxyuan/distilbert-base-multilingual-cased-sentiments-student", 'sentiment'))
    if llm is None:
        return human_sentiment
    llm_sentiment = convert_to_scalar(run_hf_model(llm, "lxyuan/distilbert-base-multilingual-cased-sentiments-student", 'sentiment'))

    return human_sentiment, llm_sentiment, np.array(llm_sentiment) - np.array(human_sentiment)

def combine_precomputed(col_1_data, col_2_data, metric_name):
    '''
        col_1_data: By default should be human
        col_2_data: By default should be llm
    '''
    if metric_name in ["sentiment", "formality", "politeness", "perplexity", "toxicity", "readability"]:
        return col_1_data, col_2_data, np.array(col_2_data) - np.array(col_1_data)
    elif metric_name in ["subjectivity", "topic"]:
        return col_1_data, col_2_data, 1 - jensenshannon(col_1_data, col_2_data, axis=1)
    else:
        raise KeyError("Metric Name not found!")

def formality(human, llm=None):
    def convert_to_scalar(data):
        # They use label 0 for formal
        score_map = {
            'LABEL_0': 1,
            'LABEL_1': 0
        }
        out_data = []
        for d in data:
            out_data.append(sum([score_map[s['label']]*s['score'] for s in d]))
        return out_data

    human_formality = convert_to_scalar(run_hf_model(human, "s-nlp/mdeberta-base-formality-ranker", 'formality'))
    if llm is None:
        return human_formality
    llm_formality = convert_to_scalar(run_hf_model(llm, "s-nlp/mdeberta-base-formality-ranker", 'formality'))

    return human_formality, llm_formality, np.array(llm_formality) - np.array(human_formality)


def politeness(human, llm=None):
    def convert_to_scalar(data):
        score_map = {
            'polite': 1,
            'impolite': 0
        }
        out_data = []
        for d in data:
            out_data.append(sum([score_map[s['label']]*s['score'] for s in d]))
        return out_data

    human_politeness = convert_to_scalar(run_hf_model(human, "Genius1237/xlm-roberta-large-tydip", 'politeness'))#[politenessr.predict([text])[0] for text in human]
    if llm is None:
        return human_politeness
    llm_politeness = convert_to_scalar(run_hf_model(llm, "Genius1237/xlm-roberta-large-tydip", 'politeness'))#[politenessr.predict([text])[0] for text in llm]

    # Now all of the prompts
    return human_politeness, llm_politeness, np.array(llm_politeness) - np.array(human_politeness)#1 - jensenshannon(human_politeness, llm_politeness, axis=1)#(np.array(human_politeness) - np.array(llm_politeness)).abs()

def perplexity(human, llm=None):
    scorer = lmppl.LM('gpt2')
    human_perplexity = scorer.get_perplexity(human, batch=4)
    if llm is None:
        return human_perplexity
    llm_perplexity = scorer.get_perplexity(llm, batch=4)
    
    # Now all of the prompts
    return human_perplexity, llm_perplexity, np.array(llm_perplexity) - np.array(human_perplexity)


def toxicity(human, llm=None):
    def convert_to_scalar(data):
        score_map = {
            'toxic': 1,
            'neutral': 0
        }
        out_data = []
        for d in data:
            out_data.append(sum([score_map[s['label']]*s['score'] for s in d]))
        return out_data

    human_toxicity = convert_to_scalar(run_hf_model(human, "s-nlp/roberta_toxicity_classifier", 'toxicity'))
    if llm is None:
        return human_toxicity
    llm_toxicity = convert_to_scalar(run_hf_model(llm, "s-nlp/roberta_toxicity_classifier", 'toxicity'))
    
    return human_toxicity, llm_toxicity, np.array(llm_toxicity) - np.array(human_toxicity)


def subjectivity(human, llm=None):
    human_subjectivity = SubjectivityAnalyzer().get_subjectivity_scores(human)
    if llm is None:
        return human_subjectivity
    llm_subjectivity = SubjectivityAnalyzer().get_subjectivity_scores(llm)
    
    return human_subjectivity, llm_subjectivity, np.array(llm_subjectivity) - np.array(human_subjectivity)#1 - jensenshannon(human_subjectivity, llm_subjectivity, axis=1)

def topic(human, llm=None):
    human_topic = [[s['score'] for s in d] for d in run_hf_model(human, "valpy/prompt-classification", 'topic')]
    if llm is None:
        return human_topic
    llm_topic = [[s['score'] for s in d] for d in run_hf_model(llm, "valpy/prompt-classification", 'topic')]

    return human_topic, llm_topic, 1 - jensenshannon(human_topic, llm_topic, axis=1)


def jitter_prob(p, eps = 1./1000.):
    if p == 0: p = p + eps
    elif p == 1: p = p - eps
    return p


def is_no_response(col, no_response_indicator = '[no response]'): 
    cond = (col == no_response_indicator) | (col.apply(len) == 0)
    return (cond).apply(lambda x: 1 if x else 0)


def compute_single_col_metric_list(data, turn_name, metrics, output_folder, data_file_name):
    if 'all' in metrics:
        metrics = [
            'lexical', 'perplexity',
            'punctuation', 'pos', 'constituency',
            'sbert', 'liwc',
            'topic', 'sentiment', 'politeness', 'formality',
            'toxicity', 'readability', 'subjectivity', 'luar'
        ]
    
    os.makedirs(output_folder, exist_ok=True)
    
    for metric in metrics:
        output_path = os.path.join(output_folder, f'{data_file_name}_{turn_name}_{metric}.json')
        compute_single_col_metric(data, turn_name, metric, output_path)

def compute_single_col_metric(data, turn_name, metric, output_path):
    if metric == 'lexical':
        # count utterance length (log word count), avg. word length
        print("Metric: lexical - log word count, avg. word length, capitalization, contraction, typo") # 5 cols
        args = Namespace()
        args.contraction_file_path = '/shared/0/projects/research-jam-summer-2024/data/contractions_dict.json'
        args.no_response_indicators = '[no response]'
        args.metrics = 'char_count,word_count,upper_count,lower_count,contract_count,typo_count'

        bss = BasicSyntacticStatistics(args)
        df_metric = bss.get_counts(data[turn_name])

        df_metric['log_word_count'] = log(df_metric['word_count'])
        df_metric['avg_word_length'] = df_metric['char_count'] / df_metric['word_count']
        df_metric['proportion_capital_over_alpha'] = df_metric['upper_count'] / (df_metric['upper_count'] + df_metric['lower_count'])
        df_metric['proportion_contraction_over_word'] = df_metric['contract_count'] / df_metric['word_count']
        df_metric['proportion_typo_over_word'] = df_metric['typo_count'] / df_metric['word_count']
        df_metric.fillna(
            value={
                'avg_word_length': 0, 'proportion_capital_over_alpha': 0,
                'proportion_contraction_over_word': 0,
                'proportion_typo_over_word': 0
            },
            inplace=True
        )
        df_metric = df_metric[[
            'log_word_count', 'avg_word_length',
            'proportion_capital_over_alpha', 'proportion_contraction_over_word',
            'proportion_typo_over_word'
        ]]
    elif metric == 'perplexity':
        print("Metric: perplexity") # 1 col, gpu
        df_metric = perplexity(list(data[turn_name]))
        df_metric = pd.DataFrame(df_metric, columns=['perplexity'], index=data.index)
    elif metric == 'punctuation':
        print("Metric: punctuation") # 32 cols
        df_metric = punctuation(data, turn_name)
        df_metric = pd.DataFrame(list(df_metric), index=df_metric.index)
        df_metric.fillna(0, inplace=True)
    elif metric == 'pos':
        print("Metric: pos") # 12 cols, gpu
        df_metric = pos_tag_metric(data[turn_name])
        df_metric = pd.DataFrame(df_metric, index=data.index)
    elif metric == 'constituency':
        print("Metric: constituency") # not checked yet, gpu
        df_metric = const_parse_metric(data[turn_name])
        df_metric = pd.DataFrame(df_metric, index=data.index)
    elif metric == 'sbert':
        print("Metric: sbert") # 384 cols, gpu
        embedding_similarity = EmbeddingSimilarity()
        embeddings = embedding_similarity.get_embeddings(list(data[turn_name])).detach().cpu().numpy()
        df_metric = pd.DataFrame(embeddings, index=data.index)
    elif metric == 'liwc':
        print("Metric: liwc") # 69 cols
        liwc_extractor_obj = LiwcDistExtractor(agg_results=False, normalize=True)
        liwc_dist = liwc_extractor_obj.extract_liwc_occurrences(data[turn_name].to_list())
        df_metric = pd.DataFrame(liwc_dist, index=data.index)
    elif metric == 'topic':
        print("Metric: topic") # 15 cols, gpu
        df_metric = topic(data[turn_name])
        topic_labels = [
            "analysis or decision explanation",
            "assisting or creative writing",
            "classification",
            "coding",
            "editing or rewriting",
            "factual information (general or professional), history or common practices",
            "information extraction or summarization",
            "linguistics",
            "logical reasoning",
            "mathematical reasoning or calculation",
            "multilinguality or translation",
            "other",
            "recommendation",
            "roleplay",
            "tips, opinions or advice"
        ]
        df_metric = pd.DataFrame(df_metric, columns=topic_labels, index=data.index)
    elif metric == 'sentiment':
        print("Metric: sentiment") # 1 col, gpu
        df_metric = sentiment(data[turn_name])
        df_metric = pd.DataFrame(df_metric, columns=['sentiment'], index=data.index)
    elif metric == 'politeness':
        print("Metric: politeness") # 1 col, gpu
        df_metric = politeness(data[turn_name])
        df_metric = pd.DataFrame(df_metric, columns=['politeness'], index=data.index)
    elif metric == 'formality':
        print("Metric: formality") # 1 col, gpu
        df_metric = formality(data[turn_name])
        df_metric = pd.DataFrame(df_metric, columns=['formality'], index=data.index)
    elif metric == 'toxicity':
        print("Metric: toxicity") # 1 col, gpu
        df_metric = toxicity(data[turn_name])
        df_metric = pd.DataFrame(df_metric, columns=['toxicity'], index=data.index)
    elif metric == 'readability':
        print("Metric: readability") # 1 col
        # note that the function returns also nan values!
        df_metric = readability_single_column(data[turn_name])
        df_metric = pd.DataFrame(df_metric, columns=['readability'], index=data.index)
    elif metric == 'subjectivity':
        print("Metric: subjectivity") # 1 col
        df_metric = subjectivity(data[turn_name])
        df_metric = pd.DataFrame(df_metric, columns=['subjectivity'], index=data.index)
    elif metric == 'luar':
        print("Metric: luar") # 512 cols, gpu
        args = Namespace()
        args.no_response_indicators = '[no response]'
        args.metrics = 'luar_similarity'

        bss = BasicSyntacticStatistics(args)
        embeddings = bss.get_luar_embeddings(data[turn_name])
        df_metric = pd.DataFrame(embeddings.numpy(), index=data.index)
    
    df_metric.to_json(output_path, orient='records', lines=True)

def compute_pairwise_metric_list(data, turn_name_1, turn_name_2, metrics, output_folder, data_file_name):
    if 'all' in metrics:
        metrics = [
            'semantic'
        ]
    
    os.makedirs(output_folder, exist_ok=True)
    
    for metric in metrics:
        output_path = os.path.join(output_folder, f'{data_file_name}_{turn_name_1}_{turn_name_2}_{metric}.json')
        compute_pairwise_metric(data, turn_name_1, turn_name_2, metric, output_path)

def compute_pairwise_metric(data, turn_name_1, turn_name_2, metric, output_path):
    if metric == 'semantic':
        args = Namespace()
        args.no_response_indicators = '[no response]'
        args.metrics = 'bleu,rouge'
        
        bss = BasicSyntacticStatistics(args)
        df_metric = bss.get_metrics(data[turn_name_1], data[turn_name_2])
    
    df_metric.to_json(output_path, orient='records', lines=True)

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

    #subset to cases where llm and human produce a response for the remaining metrics
    # produce a score comparing human vs. llm text
    data = data[(is_no_response(data['human_turn_3']) == 0) &
                (is_no_response(data['llm_turn_3']) == 0)]

    if 'all' in metrics or 'lexical' in metrics:
        print("Metric: lexical")
        args.no_response_indicators = '[no response]'
        args.metrics = 'char_count,word_count,contract_count,typo_count'
        args.contraction_file_path = "/shared/0/projects/research-jam-summer-2024/data/contractions_dict.json"
        bss = BasicSyntacticStatistics(args)
        # human metrics
        human = bss.get_counts(data['human_turn_1'])
        human['p_typo'] = human['typo_count'] / human['word_count']
        human['p_contract'] = human['contract_count'] / human['word_count']
        # llm metrics
        llm = bss.get_counts(data['ai_turn_2'])
        llm['p_typo'] = llm['typo_count'] / llm['word_count']
        llm['p_contract'] = llm['contract_count'] / llm['word_count']

        # comparison num words
        words = log(llm['word_count']) - log(human['word_count'])
        data.insert(len(data.columns), "human_turn_1_log_word_count", log(human['word_count']))
        data.insert(len(data.columns), "ai_turn_2_log_word_count", log(llm['word_count']))

        # comparison num words
        human['word_length'] = human['char_count'] / human['word_count'] - 1
        llm['word_length'] = llm['char_count'] / llm['word_count'] - 1
        word_length = llm['word_length'] - human['word_length']
        data.insert(len(data.columns), "human_turn_1_word_length", log(human['word_length']))
        data.insert(len(data.columns), "ai_turn_2_word_length", log(llm['word_length']))

        # comparison contractions
        contract_count = llm['p_contract'] - human['p_contract']
        data.insert(len(data.columns), "human_turn_1_contract_count", human['p_contract'])
        data.insert(len(data.columns), "ai_turn_2_contract_count", llm['p_contract'])

        # comparison typo
        typo = llm['p_typo'] - human['p_typo']
        data.insert(len(data.columns), "human_turn_1_typo", human['p_typo'])
        data.insert(len(data.columns), "ai_turn_2_typo", llm['p_typo'])

    if 'all' in metrics or 'capitalization' in metrics:
        print("Metric: capitalization")
        hum_cap, llm_cap, cap = capitalization(data, 'human_turn_1', 'ai_turn_2')
        data.insert(len(data.columns), "human_turn_1_capitalization", hum_cap)
        data.insert(len(data.columns), "ai_turn_2_capitalization", llm_cap)

    if 'all' in metrics or 'grammar' in metrics:
        print("Metric: grammar")
        args.no_response_indicators = '[no response]'
        args.metrics = 'grammar_error_count,word_count'
        bss = BasicSyntacticStatistics(args)
        #human metrics
        human = bss.get_counts(data['human_turn_1'])
        human['p_grammar'] = human['grammar_error_count'] / human['word_count']
        #llm metrics
        llm = bss.get_counts(data['ai_turn_2'])
        llm['p_grammar'] = llm['grammar_error_count'] / llm['word_count']
        #comparison
        grammar = 1 - jensenshannon(np.array([(p, 1-p) for p in human['p_grammar'].apply(jitter_prob)]),
                                    np.array([(p, 1-p) for p in llm['p_grammar'].apply(jitter_prob)]), axis=1)
        data.insert(len(data.columns), "human_turn_1_grammar", human['p_grammar'].apply(jitter_prob))
        data.insert(len(data.columns), "ai_turn_2_grammar", llm['p_grammar'].apply(jitter_prob))

    if 'all' in metrics or 'punctuation' in metrics:
        print("Metric: punctuation")
        hum_pun, llm_pun, pun = punctuation(data, 'human_turn_1', 'ai_turn_2')
        data.insert(len(data.columns), "human_turn_1_punctuation", hum_pun)
        data.insert(len(data.columns), "ai_turn_2_punctuation", llm_pun)

    if 'all' in metrics or 'pos' in metrics:
        print("Metric: pos")
        hum_pos, llm_pos, pos = pos_tag_metric(data['human_turn_1'], data['ai_turn_2'])
        data.insert(len(data.columns), "human_turn_1_pos", hum_pos)
        data.insert(len(data.columns), "ai_turn_2_pos", llm_pos)

    if 'all' in metrics or 'liwc' in metrics:
        print("Metric: liwc")
        liwc_extractor_obj = LiwcDistExtractor(agg_results=False, normalize=True)
        human_liwc = liwc_extractor_obj.extract_liwc_occurrences(data['human_turn_1'].to_list())
        llm_liwc = liwc_extractor_obj.extract_liwc_occurrences(data['ai_turn_2'].to_list())
        liwc = [liwc_extractor_obj.liwc_similarity(human, llm, method="jsd") for human, llm in zip(human_liwc, llm_liwc)]
        data.insert(len(data.columns), "human_turn_1_liwc", human_liwc)
        data.insert(len(data.columns), "ai_turn_2_liwc", llm_liwc)

    if 'all' in metrics or 'topic' in metrics:
        print("Metric: topic")
        human_topic, llm_topic, topic_data = topic(data['human_turn_1'], data['ai_turn_2'])
        data.insert(len(data.columns), "human_turn_1_topic", human_topic)
        data.insert(len(data.columns), "ai_turn_2_topic", llm_topic)

    if 'all' in metrics or 'sentiment' in metrics:
        print("Metric: sentiment")
        human_sent, llm_sent, sentiment_data = sentiment(data['human_turn_1'], data['ai_turn_2'])
        data.insert(len(data.columns), "human_turn_1_sentiment", human_sent)
        data.insert(len(data.columns), "ai_turn_2_sentiment", llm_sent)

    if 'all' in metrics or 'formality' in metrics:
        print("Metric: formality")
        human_form, llm_form, formality_data = formality(data['human_turn_1'], data['ai_turn_2'])
        data.insert(len(data.columns), "human_turn_1_formality", human_form)
        data.insert(len(data.columns), "ai_turn_2_formality", llm_form)

    if 'all' in metrics or 'politeness' in metrics:
        print("Metric: politeness")
        human_polite, llm_polite, politeness_data = politeness(data['human_turn_1'], data['ai_turn_2'])
        data.insert(len(data.columns), "human_turn_1_politeness", human_polite)
        data.insert(len(data.columns), "ai_turn_2_politeness", llm_polite)

    if 'all' in metrics or 'toxicity' in metrics:
        print("Metric: toxicity")
        human_toxic, llm_toxic, toxicity_data = toxicity(data['human_turn_1'], data['ai_turn_2'])
        data.insert(len(data.columns), "human_turn_1_toxicity", human_toxic)
        data.insert(len(data.columns), "ai_turn_2_toxicity", llm_toxic)

    if 'all' in metrics or 'subjectivity' in metrics:
        print("Metric: subjectivity")
        from analysis.subjectivity import SubjectivityAnalyzer
        human_subject, llm_subject, subjectivity_data = subjectivity(list(data['human_turn_1']), list(data['ai_turn_2']))
        data.insert(len(data.columns), "human_turn_1_subjectivity", human_subject)
        data.insert(len(data.columns), "ai_turn_2_subjectivity", llm_subject)

    if 'all' in metrics or 'perplexity' in metrics:
        print("Metric: perplexity")
        human_perplexity, llm_perplexity, perplexity = perplexity(list(data['human_turn_1']), list(data['ai_turn_2']))
        data.insert(len(data.columns), "human_turn_1_perplexity", human_perplexity)
        data.insert(len(data.columns), "ai_turn_2_perplexity", llm_perplexity)


    if 'all' in metrics or 'constituency' in metrics:
        print("Metric: constituency")
        from analysis.constituency_parse import const_parse_metric
        human_constituency, llm_constituency, constituency = const_parse_metric(list(data['human_turn_1']), list(data['ai_turn_2']))
        data.insert(len(data.columns), "human_turn_1_constituency_parse", human_constituency)
        data.insert(len(data.columns), "ai_turn_2_constituency_parse", llm_constituency)

    if 'all' in metrics or 'readability' in metrics:
        print("Metric: readability")
        # note that the function returns also nan values!
        human = readability_single_column(list(data['human_turn_1']))
        llm = readability_single_column(list(data['ai_turn_2']))
        comp = llm - human
        data.insert(len(data.columns), "human_turn_1_readability", human)
        data.insert(len(data.columns), "ai_turn_2_readability", llm)

    data.to_json(output_path, orient='records', lines=True)
