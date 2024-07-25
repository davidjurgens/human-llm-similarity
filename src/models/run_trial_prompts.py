import pandas as pd
from vllm import LLM, SamplingParams
import ujson as json
from transformers import AutoTokenizer
import os
import argparse
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the WildChat data",
                        required=True)
    parser.add_argument("--prompt_path", type=str, help="Name of the file with the prompts",
                        required=True)
    parser.add_argument("--output_path", type=str, help="Name of the file to save the generated text",
                        required=True)
    parser.add_argument("--model_path", type=str, help="Name of the model to run",
                        required=True)
    parser.add_argument("--seed", type=int, help="Random seed",
                        default=1000)

    args = parser.parse_args()

    enforce_reproducibility(args.seed)

    input_path = args.input_path #'./wildchat_subset_20.jsonl'
    prompt_path = args.prompt_path
    model_path = args.model_path #'mistralai/Mistral-7B-Instruct-v0.3'
    output_path = args.output_path #'./wildchat_subset_Mistral-7B-v0.3.txt'

    cache_dir = None
    if 'HF_MODEL_CACHE' in os.environ:
        cache_dir = os.environ['HF_MODEL_CACHE']

    print("Reading data...")
    data = pd.read_json(input_path, orient='records', lines=True)
    prompts = pd.read_csv(prompt_path).dropna(subset=["Prompt Design"])

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

    print("Loading LLM...")
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192, download_dir=cache_dir, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)

    batch = [
        tokenizer.apply_chat_template([
            {"role": "user", "content": prompt.replace('[TURN1]', row['human_turn_1']).replace('[TURN2]', row['ai_turn_2'])}
        ], tokenize=False, add_special_tokens=False)
        for index, row in data.iterrows()
        for j, prompt in enumerate(prompts['Prompt Design'])
    ]
    print("Generating...")
    output_batch = llm.generate(batch, sampling_params)

    results = defaultdict(list)
    k = 0
    for index, row in data.iterrows():
        for j, prompt in enumerate(prompts['Prompt Design']):
            results[j].append(output_batch[k].outputs[0].text)
            k += 1

    #results = {i: output.outputs[0].text for i, output in enumerate(output_batch)}

    with open(output_path, 'wt') as output_file:
        output_file.write(json.dumps(results))