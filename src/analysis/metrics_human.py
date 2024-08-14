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

from analysis.pos_tags_JSD import POS_tags_Dep_Spacy
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the human baseline",
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
    original_data = pd.read_csv(input_path, sep='\t')

    all_ids = set(original_data['instance_id'])

    # Get the human turn 3 original data
    sample_2k = pd.read_json("/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/wildchat_subset_en_2k_prompting_Mistral-Large-Instruct.jsonl", orient='records', lines=True)

    sample_2k['instance_id'] = [x['hashed_ip'] + x['conversation_hash'] for i,x in sample_2k.iterrows()]

    sample_2k = sample_2k.set_index('instance_id').loc[original_data['instance_id']].reset_index(inplace=False)
    assert len(sample_2k) == len(original_data)

    original_data['human_turn_3'] = sample_2k['human_turn_3']

    #subset to cases where llm and human produce a response for the remaining metrics
    # produce a score comparing human vs. llm text
    locs = (original_data['HasResponse:::[1] Yes'] == 1) & (~original_data['ResponseText:::text_box'].isna()) & (is_no_response(original_data['human_turn_3']) == 0)
    data = original_data[locs]

    print(sum(locs))
    print(len(original_data))

    if 'all' in metrics or 'lexical' in metrics:
        print("Metric: lexical")
        args.no_response_indicators = '[no response]'
        args.metrics = 'char_count,word_count,contract_count,typo_count'
        args.contraction_file_path = "/shared/0/projects/research-jam-summer-2024/data/contractions_dict.json"
        bss = BasicSyntacticStatistics(args)
        # human metrics
        annotated = bss.get_counts(data['ResponseText:::text_box'])
        annotated['p_typo'] = annotated['typo_count'] / annotated['word_count']
        annotated['p_contract'] = annotated['contract_count'] / annotated['word_count']

        human = bss.get_counts(data['human_turn_3'])
        human['p_typo'] = human['typo_count'] / human['word_count']
        human['p_contract'] = human['contract_count'] / human['word_count']

        # comparison num words
        original_data["annotated_log_word_count"] = np.nan
        original_data.loc[locs, "annotated_log_word_count"] = log(annotated['word_count'])

        # comparison num words
        annotated['word_length'] = annotated['char_count'] / annotated['word_count'] - 1
        original_data["annotated_word_length"] = np.nan
        original_data.loc[locs, "annotated_word_length"] = log(annotated['word_length'])

        # comparison contractions
        original_data["annotated_contract_count"] = np.nan
        original_data.loc[locs, "annotated_contract_count"] = annotated['p_contract']

        # comparison typo
        original_data["annotated_typo"] = np.nan
        original_data.loc[locs, "annotated_typo"] =  annotated['p_typo']

        original_data["human_log_word_count"] = np.nan
        original_data.loc[locs, "human_log_word_count"] = log(human['word_count'])

        # comparison num words
        human['word_length'] = human['char_count'] / human['word_count'] - 1
        original_data["human_word_length"] = np.nan
        original_data.loc[locs, "human_word_length"] = log(human['word_length'])

        # comparison contractions
        original_data["human_contract_count"] = np.nan
        original_data.loc[locs, "human_contract_count"] = human['p_contract']

        # comparison typo
        original_data["human_typo"] = np.nan
        original_data.loc[locs, "human_typo"] = human['p_typo']

        
    if 'all' in metrics or 'capitalization' in metrics:
        print("Metric: capitalization")
        hum_cap, llm_cap, cap = capitalization(data, 'human_turn_3', 'ResponseText:::text_box')
        original_data["human_capitalization"] = np.nan
        original_data.loc[locs, "human_capitalization"] = hum_cap
        original_data["annotated_capitalization"] = np.nan
        original_data.loc[locs, "annotated_capitalization"] = llm_cap


    if 'all' in metrics or 'grammar' in metrics:
        print("Metric: grammar")
        args.no_response_indicators = '[no response]'
        args.metrics = 'grammar_error_count,word_count'
        bss = BasicSyntacticStatistics(args)
        #annotated metrics
        annotated = bss.get_counts(data['ResponseText:::text_box'])
        annotated['p_grammar'] = annotated['grammar_error_count'] / annotated['word_count']

        original_data["annotated_grammar"] = np.nan
        original_data.loc[locs, "annotated_grammar"] = annotated['p_grammar'].apply(jitter_prob)

        human = bss.get_counts(data['human_turn_3'])
        human['p_grammar'] = human['grammar_error_count'] / human['word_count']

        original_data["human_grammar"] = np.nan
        original_data.loc[locs, "human_grammar"] = human['p_grammar'].apply(jitter_prob)

        
    if 'all' in metrics or 'punctuation' in metrics:
        print("Metric: punctuation")
        hum_pun, llm_pun, pun = punctuation(data, 'human_turn_3', 'ResponseText:::text_box')
        original_data["human_punctuation"] = np.nan
        original_data.loc[locs, "human_punctuation"] = hum_pun
        original_data["annotated_punctuation"] = np.nan
        original_data.loc[locs, "annotated_punctuation"] = llm_pun


    if 'all' in metrics or 'pos' in metrics:
        print("Metric: pos")
        output_list_pos, output_list_dep, output_list_dep_dpth, output_list_dep_brth, output_list_dep_avg_brth, output_list_dep_dep_dist, output_list_dep_max_noun_chunks, output_list_dep_avg_noun_chunks = POS_tags_Dep_Spacy(data['ResponseText:::text_box'])
        original_data["annotated_pos"] = np.nan
        original_data.loc[locs, "annotated_pos"] = output_list_pos
        original_data["annotated_dep"] = np.nan
        original_data.loc[locs, "annotated_dep"] = output_list_dep
        original_data["annotated_dep_dpth"] = np.nan
        original_data.loc[locs, "annotated_dep_dpth"] = output_list_dep_dpth
        original_data["annotated_dep_brth"] = np.nan
        original_data.loc[locs, "annotated_dep_brth"] = output_list_dep_brth
        original_data["annotated_dep_avg_brth"] = np.nan
        original_data.loc[locs, "annotated_dep_avg_brth"] = output_list_dep_avg_brth
        original_data["annotated_dep_dep_dist"] = np.nan
        original_data.loc[locs, "annotated_dep_dep_dist"] = output_list_dep_dep_dist
        original_data["annotated_dep_max_noun_chunks"] = np.nan
        original_data.loc[locs, "annotated_dep_max_noun_chunks"] = output_list_dep_max_noun_chunks
        original_data["annotated_dep_avg_noun_chunks"] = np.nan
        original_data.loc[locs, "annotated_dep_avg_noun_chunks"] = output_list_dep_avg_noun_chunks

        output_list_pos, output_list_dep, output_list_dep_dpth, output_list_dep_brth, output_list_dep_avg_brth, output_list_dep_dep_dist, output_list_dep_max_noun_chunks, output_list_dep_avg_noun_chunks = POS_tags_Dep_Spacy(data['human_turn_3'])
        original_data["human_pos"] = np.nan
        original_data.loc[locs, "human_pos"] = output_list_pos
        original_data["human_dep"] = np.nan
        original_data.loc[locs, "human_dep"] = output_list_dep
        original_data["human_dep_dpth"] = np.nan
        original_data.loc[locs, "human_dep_dpth"] = output_list_dep_dpth
        original_data["human_dep_brth"] = np.nan
        original_data.loc[locs, "human_dep_brth"] = output_list_dep_brth
        original_data["human_dep_avg_brth"] = np.nan
        original_data.loc[locs, "human_dep_avg_brth"] = output_list_dep_avg_brth
        original_data["human_dep_dep_dist"] = np.nan
        original_data.loc[locs, "human_dep_dep_dist"] = output_list_dep_dep_dist
        original_data["human_dep_max_noun_chunks"] = np.nan
        original_data.loc[locs, "human_dep_max_noun_chunks"] = output_list_dep_max_noun_chunks
        original_data["human_dep_avg_noun_chunks"] = np.nan
        original_data.loc[locs, "human_dep_avg_noun_chunks"] = output_list_dep_avg_noun_chunks

    if 'all' in metrics or 'sbert' in metrics:
        print("Metric: sbert")
        embeddings = EmbeddingSimilarity()

        text_sbert = embeddings.get_embeddings(list(data['ResponseText:::text_box']))
        sbert_embeddings = np.zeros((len(original_data), text_sbert.shape[1]))
        sbert_embeddings[np.array(locs)] = text_sbert.detach().cpu().numpy()

        human_sbert = embeddings.get_embeddings(list(data['human_turn_3']))
        human_sbert_embeddings = np.zeros((len(original_data), human_sbert.shape[1]))
        human_sbert_embeddings[np.array(locs)] = human_sbert.detach().cpu().numpy()
        # np.savez(
        #     f"{output_path}_sbert_embeddings.npz",
        #     annotated_sbert=sbert_embeddings,
        #     human_sbert=human_sbert_embeddings
        #     # luar=luar_embeddings
        # )


    if 'all' in metrics or 'semantic' in metrics:
        print("Metric: semantic")
        args.no_response_indicators = '[no response]'
        args.metrics = 'luar_similarity'
        #args.metrics = 'bleu,rouge'
        bss = BasicSyntacticStatistics(args)
        text_luar = bss.get_luar_embeddings(data['ResponseText:::text_box'])
        luar_embeddings = np.zeros((len(original_data), text_luar.shape[1]))
        luar_embeddings[np.array(locs)] = text_luar.detach().cpu().numpy()

        human_luar = bss.get_luar_embeddings(data['human_turn_3'])
        human_luar_embeddings = np.zeros((len(original_data), human_luar.shape[1]))
        human_luar_embeddings[np.array(locs)] = human_luar.detach().cpu().numpy()
        # np.savez(
        #     f"{output_path}_luar_embeddings.npz",
        #     annotated_luar=luar_embeddings,
        #     human_luar=human_luar_embeddings
        #     # luar=luar_embeddings
        # )

        
    if 'all' in metrics or 'liwc' in metrics:
        print("Metric: liwc")
        liwc_extractor_obj = LiwcDistExtractor(agg_results=False, normalize=True)
        human_liwc = liwc_extractor_obj.extract_liwc_occurrences(data['human_turn_3'].to_list())
        original_data["human_liwc"] = np.nan
        original_data.loc[locs, "human_liwc"] = human_liwc

        annotated_liwc = liwc_extractor_obj.extract_liwc_occurrences(data['ResponseText:::text_box'].to_list())
        original_data["annotated_liwc"] = np.nan
        original_data.loc[locs, "annotated_liwc"] = annotated_liwc
        
    if 'all' in metrics or 'topic' in metrics:
        print("Metric: topic")
        annotated_topic, human_topic, topic_data = topic(data['ResponseText:::text_box'], data['human_turn_3'])
        ann = []
        j = 0
        for val in locs:
            if val:
                ann.append(annotated_topic[j])
                j += 1
            else:
                ann.append([np.nan]*len(annotated_topic[0]))
        # ann = [[np.nan]*len(annotated_topic[0])]*len(original_data)
        # ann[np.array(original_data.index[locs], dtype=np.int32)] = annotated_topic
        original_data["annotated_topic"] = ann
        #original_data.loc[locs, "annotated_topic"] = annotated_topic

        ann = []
        j = 0
        for val in locs:
            if val:
                ann.append(human_topic[j])
                j += 1
            else:
                ann.append([np.nan] * len(human_topic[0]))
        # ann = [[np.nan]*len(annotated_topic[0])]*len(original_data)
        # ann[np.array(original_data.index[locs], dtype=np.int32)] = annotated_topic
        original_data["annotated_topic"] = ann

        # hum = [[np.nan]*len(annotated_topic[0])]*len(original_data)
        # hum[np.array(original_data.index[locs], dtype=np.int32)] = human_topic
        #original_data["human_topic"] = hum
        #original_data.loc[locs, "human_topic"] = human_topic

    if 'all' in metrics or 'sentiment' in metrics:
        print("Metric: sentiment")
        annotated_sent, human_sent, sentiment_data = sentiment(data['ResponseText:::text_box'], data['human_turn_3'])
        original_data["annotated_sentiment"] = np.nan
        original_data.loc[locs, "annotated_sentiment"] = annotated_sent

        original_data["human_sentiment"] = np.nan
        original_data.loc[locs, "human_sentiment"] = human_sent

    if 'all' in metrics or 'formality' in metrics:
        print("Metric: formality")
        annotated_form, human_form, formality_data = formality(data['ResponseText:::text_box'], data['human_turn_3'])
        original_data["annotated_formality"] = np.nan
        original_data.loc[locs, "annotated_formality"] = annotated_form

        original_data["human_formality"] = np.nan
        original_data.loc[locs, "human_formality"] = human_form


    if 'all' in metrics or 'politeness' in metrics:
        print("Metric: politeness")
        annotated_polite, human_polite, politeness_data = politeness(data['ResponseText:::text_box'], data['human_turn_3'])
        original_data["annotated_politeness"] = np.nan
        original_data.loc[locs, "annotated_politeness"] = annotated_polite

        original_data["human_politeness"] = np.nan
        original_data.loc[locs, "human_politeness"] = human_polite

        
    if 'all' in metrics or 'toxicity' in metrics:
        print("Metric: toxicity")
        annotated_toxic, human_toxic, toxicity_data = toxicity(data['ResponseText:::text_box'], data['human_turn_3'])
        original_data["annotated_toxicity"] = np.nan
        original_data.loc[locs, "annotated_toxicity"] = annotated_toxic

        original_data["human_toxicity"] = np.nan
        original_data.loc[locs, "human_toxicity"] = human_toxic

    if 'all' in metrics or 'subjectivity' in metrics:
        print("Metric: subjectivity")
        from analysis.subjectivity import SubjectivityAnalyzer
        annotated_subject, human_subject, subjectivity_data = subjectivity(list(data['ResponseText:::text_box']), list(data['human_turn_3']))
        original_data["annotated_subjectivity"] = np.nan
        original_data.loc[locs, "annotated_subjectivity"] = annotated_subject

        original_data["human_subjectivity"] = np.nan
        original_data.loc[locs, "human_subjectivity"] = human_subject

    if 'all' in metrics or 'perplexity' in metrics:
        print("Metric: perplexity")
        annotated_perplexity, human_perplexity, perplexity = perplexity(list(data['ResponseText:::text_box']), list(data['human_turn_3']))
        original_data["annotated_perplexity"] = np.nan
        original_data.loc[locs, "annotated_perplexity"] = annotated_perplexity

        original_data["human_perplexity"] = np.nan
        original_data.loc[locs, "human_perplexity"] = human_perplexity


    if 'all' in metrics or 'readability' in metrics:
        print("Metric: readability")
        # note that the function returns also nan values!
        annotated = readability_single_column(list(data['ResponseText:::text_box']))
        original_data["annotated_readability"] = np.nan
        original_data.loc[locs, "annotated_readability"] = annotated

        human = readability_single_column(list(data['human_turn_3']))
        original_data["human_readability"] = np.nan
        original_data.loc[locs, "human_readability"] = human


    #original_data.to_csv(f"{output_path}.tsv", sep='\t', index=False)
    original_data.to_json(f"{output_path}.jsonl", orient='records', lines=True)
    if 'sbert' or 'semantic' in metrics:
        np.savez(
            f"{output_path}_embeddings.npz",
            annotated_sbert=sbert_embeddings,
            human_sbert=human_sbert_embeddings,
            annotated_luar=luar_embeddings,
            human_luar=human_luar_embeddings
            #luar=luar_embeddings
        )
