import argparse
import os
import pandas as pd
import logging
import time
import json
from argparse import Namespace
from pandas import DataFrame, Series

import string
import torch.nn.functional as F

table = {ord(f):ord(t) for f,t in zip(
     u'，。！？【】（）％＃＠＆１２３４５６７８９０',
     u',.!?[]()%#@&1234567890')}

class BasicSyntacticStatisticsChinese:
    def __init__(self, args: Namespace):
        if args.metrics == 'all':
            self.metric_list = [
                'no_response_exact', 'no_response_include', 'word_count', 
                'space_count', 'alnum_count', 'punct_count', 'numeric_count',
            ]
        else:
            self.metric_list = args.metrics.split(',')
        self.no_response_indicator_list = args.no_response_indicators.split(',')

        
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
    def get_length(text: str, is_word_counts: bool = False) -> int:
        if is_word_counts:
            return len(text.split())
        else:
            return len(text)

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
        text = text.translate(table)
        return sum(1 for char in text if string.punctuation.find(char) >= 0)

    def get_counts(self, df_input: Series):
        df_output = {}
        
        if 'no_response_exact' in self.metric_list:
            df_output['no_response_exact'] = df_input.transform(lambda x: self.is_no_response(x, is_exact_match=True))
            
        if 'no_response_include' in self.metric_list:
            df_output['no_response_include'] = df_input.transform(lambda x: self.is_no_response(x, is_exact_match=False))
            
        if 'word_count' in self.metric_list:
            df_output['word_count'] = df_input.transform(lambda x: BasicSyntacticStatisticsChinese.get_length(x, is_word_counts=True))
              
        if 'upper_count' in self.metric_list:
            df_output['upper_count'] = df_input.transform(BasicSyntacticStatisticsChinese.get_upper_count)
            
        if 'lower_count' in self.metric_list:
            df_output['lower_count'] = df_input.transform(BasicSyntacticStatisticsChinese.get_lower_count)
            
        if 'space_count' in self.metric_list:
            df_output['space_count'] = df_input.transform(BasicSyntacticStatisticsChinese.get_space_count)
            
        if 'numeric_count' in self.metric_list:
            df_output['numeric_count'] = df_input.transform(BasicSyntacticStatisticsChinese.get_numeric_count)
            
        if 'alnum_count' in self.metric_list:
            df_output['alnum_count'] = df_input.transform(BasicSyntacticStatisticsChinese.get_alnum_count)
            
        if 'punct_count' in self.metric_list:
            df_output['punct_count'] = df_input.transform(BasicSyntacticStatisticsChinese.get_punctuation_count)

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
            logging.FileHandler(os.path.join(args.log_folder, 'syntactic_metrics_chinese.log'), mode='a'),  # Log file in the current directory
            logging.StreamHandler()  # Log to stdout
        ]
    )
    logger = logging.getLogger(__name__)
    
    print("Beginning setting up class")
    start = time.time()
    bss = BasicSyntacticStatisticsChinese(args)
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
        default="/shared/0/projects/research-jam-summer-2024/data/chinese_only/",
        type=str
    )
    parser.add_argument(
        '--output_folder',
        help='Path to output computed metrics.',
        default="/shared/3/projects/bowenyi/research jam 2024/output_folder",
        type=str
    )
    parser.add_argument(
        '--log_folder',
        help='Path to store log.',
        default="/shared/3/projects/bowenyi/research jam 2024/log_folder",
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
 