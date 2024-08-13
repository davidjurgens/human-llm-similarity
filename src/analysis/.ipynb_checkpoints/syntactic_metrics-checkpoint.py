import argparse
import os
import pandas as pd
import logging
import time

from argparse import Namespace
from pandas import DataFrame, Series

import string
import ujson as json
from spellchecker import SpellChecker
import nltk
from rouge_metric import PyRouge
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import contractions


class BasicSyntacticStatistics:
    def __init__(self, args: Namespace):
        if args.metrics == 'all':
            self.metric_list = [
                'no_response_exact', 'no_response_include', 'word_count', 'char_count',
                'upper_count', 'lower_count', 'space_count', 'space_count', 'alnum_count', 'punct_count', 'contract_count',
                # 'typo_count', 'grammar_error_count', 'bleu', 'rouge', 'luar_similarity'
                'typo_count', 'bleu', 'rouge', 'luar_similarity'
            ]
        else:
            self.metric_list = args.metrics.split(',')
        self.no_response_indicator_list = args.no_response_indicators.split(',')

        if 'contract_count' in self.metric_list:
            with open(args.contraction_file_path) as input_file:
                self.contractions_dict = contractions.contractions_dict #json.load(input_file)
        if 'typo_count' in self.metric_list:
            self.spell_checker = SpellChecker()
        if 'grammar_error_count' in self.metric_list:
            self.grammar_checker = language_tool_python.LanguageTool('en-US')
        if 'rouge' in self.metric_list:
            self.rouge_evaluator = PyRouge(
                rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4
            )
        if 'luar_similarity' in self.metric_list:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(2)
            self.luar_tokenizer = AutoTokenizer.from_pretrained('rrivera1849/LUAR-CRUD')
            self.luar_embedder = AutoModel.from_pretrained('rrivera1849/LUAR-CRUD', trust_remote_code=True).to(self.device)
            
    def is_no_response(self, text: str, is_exact_match: bool = True) -> bool:
        if is_exact_match:
            return any(
                text.strip() == no_response_indicator
                for no_response_indicator in self.no_response_indicator_list
            )
        else:
            return any(
                text.find(no_response_indicator) >= 0
                for no_response_indicator in self.no_response_indicator_list
            )

    @staticmethod
    def get_length(text: str, is_word_counts: bool = True) -> int:
        if is_word_counts:
            return len(text.split())
        else:
            return len(text)

    @staticmethod
    def get_upper_count(text: str) -> int:
        return sum(1 for char in text if char.isupper())

    @staticmethod
    def get_lower_count(text: str) -> int:
        return sum(1 for char in text if char.islower())

    @staticmethod
    def get_space_count(text: str) -> int:
        return sum(1 for char in text if char.isspace())

    @staticmethod
    def get_numeric_count(text: str) -> int:
        return sum(1 for char in text if char.isdigit())

    @staticmethod
    def get_alnum_count(text: str) -> int:
        return sum(1 for char in text if char.isalnum())

    @staticmethod
    def get_punctuation_count(text: str) -> int:
        return sum(1 for char in text if string.punctuation.find(char) >= 0)

    @staticmethod
    def get_contraction_count(text: str) -> int:
        try:
            return sum(1 for word in text.split() if word in self.contractions_dict)
        except Exception as e:
            return 0
    
    def get_typo_count(self, text: str) -> int:
        misspelled = self.spell_checker.unknown(text.split())
        return len(misspelled)

    def get_grammar_error_count(self, text: str) -> int:
        matches = self.grammar_checker.check(text)
        return len(matches)

    def get_bleu_score(self, text_pred: str, text_ref: str) -> float:
        if len(text_pred) == 0 or len(text_ref) == 0:
            return 0
        try:
            results = nltk.translate.bleu_score.sentence_bleu([text_ref.split()], text_pred.split())
            return results
        except:
            return 0

    def get_rouge_score(self, text_pred: str, text_ref: str) -> float:
        if len(text_pred) == 0 or len(text_ref) == 0:
            return 0
        try:
            results = self.rouge_evaluator.evaluate(text_pred, text_ref)
            return {key: value['f'] for key, value in results.items()}
        except Exception as e:
            print(e)
            return 0

    def get_luar_embeddings(self, text_series: Series) -> torch.Tensor:
        batch_size = 32
        num_batches = (len(text_series) - 1) // batch_size + 1

        embeddings = []
        for i in range(num_batches):
            texts = list(text_series.iloc[i*batch_size: (i+1)*batch_size])
            actual_batch_size = len(texts)
            tokenized_texts = self.luar_tokenizer(texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            tokenized_texts['input_ids'] = tokenized_texts['input_ids'].reshape(actual_batch_size, 1, -1)
            tokenized_texts['attention_mask'] = tokenized_texts['attention_mask'].reshape(actual_batch_size, 1, -1)

            out = self.luar_embedder(**tokenized_texts)
            embeddings.append(out.detach().cpu())
        return torch.concatenate(embeddings, dim=0)

    def get_cosine_similarity(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor) -> torch.Tensor:
        embeddings_1 = embeddings_1.to(self.device)
        embeddings_2 = embeddings_2.to(self.device)

        embeddings_1 = embeddings_1 / embeddings_1.norm(dim=1, keepdim=True)
        embeddings_2 = embeddings_2 / embeddings_2.norm(dim=1, keepdim=True)

        similarities = torch.sum(embeddings_1 * embeddings_2, dim=1)  # A matrix of similarities

        return similarities.detach().cpu()
        
    def get_counts(self, df_input: Series):
        df_output = {}
        
        if 'no_response_exact' in self.metric_list:
            df_output['no_response_exact'] = df_input.transform(lambda x: self.is_no_response(x, is_exact_match=True))
            
        if 'no_response_include' in self.metric_list:
            df_output['no_response_include'] = df_input.transform(lambda x: self.is_no_response(x, is_exact_match=False))
            
        if 'word_count' in self.metric_list:
            df_output['word_count'] = df_input.transform(lambda x: BasicSyntacticStatistics.get_length(x, is_word_counts=True))
            
        if 'char_count' in self.metric_list:
            df_output['char_count'] = df_input.transform(lambda x: BasicSyntacticStatistics.get_length(x, is_word_counts=False))
            
        if 'upper_count' in self.metric_list:
            df_output['upper_count'] = df_input.transform(BasicSyntacticStatistics.get_upper_count)
            
        if 'lower_count' in self.metric_list:
            df_output['lower_count'] = df_input.transform(BasicSyntacticStatistics.get_lower_count)
            
        if 'space_count' in self.metric_list:
            df_output['space_count'] = df_input.transform(BasicSyntacticStatistics.get_space_count)
            
        if 'numeric_count' in self.metric_list:
            df_output['numeric_count'] = df_input.transform(BasicSyntacticStatistics.get_numeric_count)
            
        if 'alnum_count' in self.metric_list:
            df_output['alnum_count'] = df_input.transform(BasicSyntacticStatistics.get_alnum_count)
            
        if 'punct_count' in self.metric_list:
            df_output['punct_count'] = df_input.transform(BasicSyntacticStatistics.get_punctuation_count)
            
        if 'contract_count' in self.metric_list:
            df_output['contract_count'] = df_input.transform(BasicSyntacticStatistics.get_contraction_count)
            
        if 'typo_count' in self.metric_list:
            df_output['typo_count'] = df_input.transform(self.get_typo_count)
            
        if 'grammar_error_count' in self.metric_list:
            df_output['grammar_error_count'] = df_input.transform(self.get_grammar_error_count)

        return pd.DataFrame(df_output)

    def get_metrics(self, df_input1: Series, df_input2: Series):
        df_output = {}
        if 'bleu' in self.metric_list:
            start = time.time()
            df_output['bleu'] = df_input1.combine(df_input2, self.get_bleu_score)
            end = time.time()
            print(f"Completed bleu in {end-start}s")
            
        if 'rouge' in self.metric_list:
            start = time.time()
            df_temp = df_input1.combine(df_input2, self.get_rouge_score)
            df_temp = pd.DataFrame(list(df_temp), index=df_temp.index)

            for col in df_temp.columns:
                df_output[col] = df_temp[col]
            end = time.time()
            print(f"Completed rouge in {end-start}s")

        if 'luar_similarity' in self.metric_list:
            start = time.time()
            embeddings_1 = self.get_luar_embeddings(df_input1)
            end = time.time()
            print(f"Completed embeddings_1 in {end-start}s")
            
            start = time.time()
            embeddings_2 = self.get_luar_embeddings(df_input2)
            end = time.time()
            print(f"Completed embeddings_2 in {end-start}s")
            
            start = time.time()
            df_output['luar_similarity'] = pd.Series(self.get_cosine_similarity(embeddings_1, embeddings_2), index=df_input1.index)
            end = time.time()
            print(f"Completed luar_similarity in {end-start}s")
            
        return pd.DataFrame(df_output)
    

def get_dataframe(input_path: str) -> DataFrame:
    data = pd.read_json(input_path, orient='records', lines=True)
    vals = [c for c in data.columns if c.startswith('Prompt_')]
    ids = [c for c in data.columns if c not in vals]
    data = pd.melt(data, id_vars=ids, value_vars=vals, var_name='prompt', value_name='llm_turn_3')
    data = data[data.llm_turn_3 != '[INVALID_DO_NOT_USE]']
    return data

def main(args):
    input_folder = args.input_folder
    output_folder = args.output_folder
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_folder, 'syntactic_metrics.log'), mode='a'),  # Log file in the current directory
            logging.StreamHandler()  # Log to stdout
        ]
    )
    logger = logging.getLogger(__name__)
    
    print("Beginning setting up class")
    start = time.time()
    bss = BasicSyntacticStatistics(args)
    end = time.time()
    print(f"Completed setting up class in {end-start}s")

    try:
        for input_file_name in os.listdir(input_folder):
            print(input_file_name)
            start = time.time()
            input_path = os.path.join(input_folder, input_file_name)
            df_input = get_dataframe(input_path)
            end = time.time()
            print(f"Completed getting data in {end-start}s")
            
            start = time.time()
            df_human_turn_counts = bss.get_counts(df_input['human_turn_3'])
            df_human_turn_counts.rename(columns={col: f'human_turn_{col}' for col in df_human_turn_counts.columns}, inplace=True)
            end = time.time()
            print(f"Completed getting counts for human turns in {end-start}s")
            start = time.time()
            df_llm_turn_counts = bss.get_counts(df_input['llm_turn_3'])
            df_llm_turn_counts.rename(columns={col: f'llm_turn_{col}' for col in df_llm_turn_counts.columns}, inplace=True)
            end = time.time()
            print(f"Completed getting counts for llm turns in {end-start}s")
            start = time.time()
            df_metrics = bss.get_metrics(df_input['human_turn_3'], df_input['llm_turn_3'])
            end = time.time()
            print(f"Completed getting pairwise metrics in {end-start}s")
    
            df_output = pd.concat([df_input, df_human_turn_counts, df_llm_turn_counts, df_metrics], axis=1)
            
            output_file_path = os.path.join(output_folder, input_file_name)
            os.makedirs(output_folder, exist_ok=True)
            df_output.to_json(output_file_path, orient='records', lines=True)
            print(f"Completed metric computations for {input_file_name}")
    except Exception as e:
        print(e)
        # bss.grammar_checker.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI for syntactic metrics - get metrics for sampled conversation dataset')
    parser.add_argument(
        '--input_folder',
        help='Path to input conversations.',
        default="/shared/0/projects/research-jam-summer-2024/data/english_only/prompting_results_clean/",
        type=str
    )
    parser.add_argument(
        '--output_folder',
        help='Path to output computed metrics.',
        default="/home/kimjhj/projects/research-jam-summer-2024/human-llm-similarity/metrics/english_only/prompting_results_clean/",
        type=str
    )
    parser.add_argument(
        '--contraction_file_path',
        help='Path to the file with the list of contractions.',
        default="/home/kimjhj/projects/research-jam-summer-2024/human-llm-similarity/src/analysis/punctuation_list.json",
        type=str
    )
    parser.add_argument(
        '--log_folder',
        help='Path to store log.',
        default="/home/kimjhj/projects/research-jam-summer-2024/",
        type=str
    )
    parser.add_argument(
        '--metrics',
        help='Metrics to get, separated by commas. If "all", extract *every* available metrics.',
        default="all",
        type=str
    )
    parser.add_argument(
        '--no_response_indicators',
        help='Tokens indicating no response, separated by commas.',
        default='[no response],[No Response],<CONV_STOP>,[SILENT]',
        type=str
    )
    args = parser.parse_args()
    main(args)
 