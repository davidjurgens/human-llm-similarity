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
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from syntactic_metrics_chinese import BasicSyntacticStatisticsChinese
from embedding_similarity import EmbeddingSimilarity
from capitalization_punctuation_similarity import punctuation

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


def load_hf_model(model_name, use_pipeline=True):
    cache_dir = None
    if 'HF_MODEL_CACHE' in os.environ:
        cache_dir = os.environ['HF_MODEL_CACHE']
    if use_pipeline:
        pipe = pipeline("text-classification", model=model_name, model_kwargs={"cache_dir": cache_dir},
                        device_map='cuda', max_length=512, truncation=True, return_all_scores=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to("mps")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        pipe = model, tokenizer
    return pipe



def hf_data(data, max_char_len=2000):
    for row in data:
        yield row[:max_char_len]


def run_hf_model(data, model_name, column_name, use_pipeline=True):
    pipe = load_hf_model(model_name, use_pipeline)

    def predict(text):
        model, tokenizer = pipe
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to("mps")
        outputs = model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        # Convert to format
        labels = ['neutral', 'positive', 'negative']
        return [{'label': labels[i], 'score': predicted[0][i].item()} for i in range(3)]
    
    new_data = []
    if use_pipeline:
        for out in tqdm(pipe(hf_data(data), batch_size=4), total=len(data)):
            new_data.append([s for s in out])
    else:
        for out in tqdm(hf_data(data), total=len(data)):
            new_data.append(predict(out))
    return new_data

def sentiment(human, llm):
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

    human_sentiment = convert_to_scalar(run_hf_model(human, "lxyuan/distilbert-base-multilingual-cased-sentiments-student", 'sentiment', use_pipeline=True))
    llm_sentiment = convert_to_scalar(run_hf_model(llm, "lxyuan/distilbert-base-multilingual-cased-sentiments-student", 'sentiment', use_pipeline=True))

    return human_sentiment, llm_sentiment, np.array(llm_sentiment) - np.array(human_sentiment)

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

    human_toxicity = convert_to_scalar(run_hf_model(human, "textdetox/xlmr-large-toxicity-classifier", 'toxicity'))
    if llm is None:
        return human_toxicity
    llm_toxicity = convert_to_scalar(run_hf_model(llm, "textdetox/xlmr-large-toxicity-classifier", 'toxicity'))
    
    return human_toxicity, llm_toxicity, np.array(llm_toxicity) - np.array(human_toxicity)

def perplexity(human, llm=None):
    scorer = lmppl.LM('uer/gpt2-chinese-cluecorpussmall')
    human_perplexity = scorer.get_perplexity(human, batch=8)
    if llm is None:
        return human_perplexity
    llm_perplexity = scorer.get_perplexity(llm, batch=8)
    
    # Now all of the prompts
    return human_perplexity, llm_perplexity, np.array(llm_perplexity) - np.array(human_perplexity)

def is_no_response(col, no_response_indicator = '[no response]'): 
    cond = (col == no_response_indicator) | (col.apply(len) == 0)
    return (cond).apply(lambda x: 1 if x else 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the WildChat data",
                        required=True)
    parser.add_argument("--output_path", type=str, help="Name of the file to save the generated text",
                        required=True)
    parser.add_argument("--metrics", type=str, default='perplexity', help="Comma separated list of the metrics you want to run, or 'all' to run all")
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
        print("Metric: end")
        human_end = is_no_response(data['human_turn_3']) 
        llm_end = is_no_response(data['llm_turn_3'])
        data.insert(len(data.columns), "human_end", human_end)
        data.insert(len(data.columns), "llm_end", llm_end)
        data.insert(len(data.columns), "metric_end", (human_end == llm_end).astype(int))
        data.to_json(re.sub('.json','_end.json',output_path), orient='records', lines=True)

    #subset to cases where llm and human produce a response for the remaining metrics
    # produce a score comparing human vs. llm text
    data = data[(is_no_response(data['human_turn_3']) == 0) &
                (is_no_response(data['llm_turn_3']) == 0)]
    
    if 'all' in metrics or 'lexical' in metrics:
        print("Metric: lexical")
        args.no_response_indicators = '[no response]'
        args.metrics = 'word_count,alnum_count'
        bss = BasicSyntacticStatisticsChinese(args)

        human = bss.get_counts(data['human_turn_3'])
        llm = bss.get_counts(data['llm_turn_3'])
        # comparison num words
        words = log(llm['word_count']) - log(human['word_count'])
        data.insert(len(data.columns), "human_turn_3_log_word_count", log(human['word_count']))
        data.insert(len(data.columns), "llm_turn_3_log_word_count", log(llm['word_count']))
        data.insert(len(data.columns), "metric_log_word_count", words)
        # alpha_num ratio
        data.insert(len(data.columns), "human_turn_3_alnum_ratio", human['alnum_count'] / human['word_count'])
        data.insert(len(data.columns), "llm_turn_3_alnum_ratio", llm['alnum_count'] / llm['word_count'])

        # Individual metrics
        human = bss.get_counts(data['human_turn_1'])
        llm = bss.get_counts(data['ai_turn_2'])
        # comparison num words
        data.insert(len(data.columns), "human_turn_1_log_word_count", log(human['word_count']))
        data.insert(len(data.columns), "ai_turn_2_log_word_count", log(llm['word_count']))
        # alpha_num ratio
        data.insert(len(data.columns), "human_turn_1_alnum_ratio", human['alnum_count'] / human['word_count'])
        data.insert(len(data.columns), "ai_turn_2_alnum_ratio", llm['alnum_count'] / llm['word_count'])

    
    if 'all' in metrics or 'punctuation' in metrics:
        print("Metric: punctuation")
        hum_pun, llm_pun, pun = punctuation(data, 'human_turn_3', 'llm_turn_3', lang="cn")
        data.insert(len(data.columns), "human_turn_3_punctuation", hum_pun)
        data.insert(len(data.columns), "llm_turn_3_punctuation", llm_pun)
        data.insert(len(data.columns), "metric_punctuation", pun)

        # Individual metrics
        hum_pun, llm_pun, pun = punctuation(data, 'human_turn_1', 'ai_turn_2', lang="cn")
        data.insert(len(data.columns), "human_turn_1_punctuation", hum_pun)
        data.insert(len(data.columns), "ai_turn_2_punctuation", llm_pun)


    if 'all' in metrics or 'sentiment' in metrics:
        print("Metric: sentiment")
        human_sent, llm_sent, sentiment_data = sentiment(data['human_turn_3'], data['llm_turn_3'])
        data.insert(len(data.columns), "human_turn_3_sentiment", human_sent)
        data.insert(len(data.columns), "llm_turn_3_sentiment", llm_sent)
        data.insert(len(data.columns), "metric_sentiment", sentiment_data)

        # Individual metrics
        human_sent, llm_sent, sentiment_data = sentiment(data['human_turn_1'], data['ai_turn_2'])
        data.insert(len(data.columns), "human_turn_1_sentiment", human_sent)
        data.insert(len(data.columns), "ai_turn_2_sentiment", llm_sent)


    if 'all' in metrics or 'toxicity' in metrics:
        print("Metric: toxicity")
        human_toxic, llm_toxic, toxicity_data = toxicity(data['human_turn_3'], data['llm_turn_3'])
        data.insert(len(data.columns), "human_turn_3_toxicity", human_toxic)
        data.insert(len(data.columns), "llm_turn_3_toxicity", llm_toxic)
        data.insert(len(data.columns), "metric_toxicity", toxicity_data)

        # Individual metrics
        human_toxic, llm_toxic, toxicity_data = toxicity(data['human_turn_1'], data['ai_turn_2'])
        data.insert(len(data.columns), "human_turn_1_toxicity", human_toxic)
        data.insert(len(data.columns), "ai_turn_2_toxicity", llm_toxic)


    if 'all' in metrics or 'perplexity' in metrics:
        print("Metric: perplexity")
        human_perplexity, llm_perplexity, perplexity_data = perplexity(list(data['human_turn_3']), list(data['llm_turn_3']))
        data.insert(len(data.columns), "human_turn_3_perplexity", human_perplexity)
        data.insert(len(data.columns), "llm_turn_3_perplexity", llm_perplexity)
        data.insert(len(data.columns), "metric_perplexity", perplexity_data)

        # Individual metrics
        human_perplexity, llm_perplexity, perplexity_data = perplexity(list(data['human_turn_1']), list(data['ai_turn_2']))
        data.insert(len(data.columns), "human_turn_1_perplexity", human_perplexity)
        data.insert(len(data.columns), "ai_turn_2_perplexity", llm_perplexity)
    

    if 'all' in metrics or 'sbert' in metrics:
        print("Metric: sbert")
        embeddings = EmbeddingSimilarity(model_name="Alibaba-NLP/gte-Qwen2-7B-instruct")
        embeddings_1 = embeddings.get_embeddings(list(data['human_turn_3']), batch_size=4).cpu()
        embeddings_2 = embeddings.get_embeddings(list(data['llm_turn_3']), batch_size=4).cpu()
        similarity = embeddings.cosine_similarity(embeddings_1, embeddings_2)
        data.insert(len(data.columns), "human_turn_3_sbert_embedding", [embeddings_1[i] for i in range(embeddings_1.shape[0])])
        data.insert(len(data.columns), "llm_turn_3_sbert_embedding", [embeddings_2[i] for i in range(embeddings_2.shape[0])])
        data.insert(len(data.columns), "metric_sbert", similarity)

        # Individual metrics
        embeddings_3 = embeddings.get_embeddings(list(data['human_turn_1']), batch_size=4).cpu()
        embeddings_4 = embeddings.get_embeddings(list(data['ai_turn_2']), batch_size=4).cpu()
        data.insert(len(data.columns), "human_turn_1_sbert_embedding", [embeddings_3[i] for i in range(embeddings_3.shape[0])])
        data.insert(len(data.columns), "ai_turn_2_sbert_embedding", [embeddings_4[i] for i in range(embeddings_4.shape[0])])

    data.to_json(output_path, orient='records', lines=True)