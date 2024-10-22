import pandas as pd
from vllm import LLM, SamplingParams
#import ujson as json
from transformers import AutoTokenizer
import os
import argparse
import torch
import random
import numpy as np
import pandas as pd
#from collections import defaultdict
from tqdm import tqdm


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
    parser.add_argument("--num_gpus", type=int, help="Number of gpus for parallel inference",
                        default=1)
                        
    args = parser.parse_args()

    enforce_reproducibility(args.seed)

    input_path = args.input_path #'./wildchat_subset_20.jsonl'
    prompt_path = args.prompt_path
    model_path = args.model_path #'mistralai/Mistral-7B-Instruct-v0.3'
    output_path = args.output_path #'./wildchat_subset_Mistral-7B-v0.3.jsonl'

    cache_dir = None
    if 'HF_MODEL_CACHE' in os.environ:
        cache_dir = os.environ['HF_MODEL_CACHE']

    print("Reading data...")
    data = pd.read_json(input_path, orient='records', lines=True)
    prompts = pd.read_csv(prompt_path).dropna(subset=["Prompt Design"])

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

    print("Loading LLM...")
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192, download_dir=cache_dir, tensor_parallel_size=args.num_gpus)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

    print("Generating...")
    for j, prompt in tqdm(prompts.iterrows(), total=len(prompts)):
        batch = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": prompt['Prompt Design'].replace('[TURN1]', row['human_turn_1'][:5000]).replace('[TURN2]', row['ai_turn_2'][:5000])}
            ], tokenize=False, add_special_tokens=False, add_generation_prompt=True)
            for index, row in data.iterrows()
        ]
    
        output_batch = llm.generate(batch, sampling_params)
        data[f"Prompt_{prompt['Prompt ID']}"] = [output.outputs[0].text.strip() for output in output_batch]

        data.to_json(output_path, orient='records', lines=True)


    # results = defaultdict(list)
    # k = 0
    # for index, row in data.iterrows():
    #     for j, prompt in prompts.iterrows():
    #         results[prompt['Prompt ID']].append(output_batch[k].outputs[0].text)
    #         k += 1

    #results = {i: output.outputs[0].text for i, output in enumerate(output_batch)}

    # with open(output_path, 'wt') as output_file:
    #     output_file.write(json.dumps(results))