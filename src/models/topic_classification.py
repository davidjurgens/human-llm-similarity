from transformers import pipeline
import pandas as pd
import os
import argparse
import torch
import random
import numpy as np
import json
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

def data(dataset, max_char_len=2000):

    for index, row in dataset.iterrows():
        yield row['human_turn_1'][:max_char_len]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the prompts",
                        required=True)
    parser.add_argument("--output_path", type=str, help="Name of the file to save the generated text",
                        required=True)
    parser.add_argument("--seed", type=int, help="Random seed",
                        default=1000)

    args = parser.parse_args()

    enforce_reproducibility(args.seed)

    input_path = args.input_path  # './wildchat_subset_20.jsonl'
    output_path = args.output_path

    dataset = pd.read_json(input_path, orient='records', lines=True)

    cache_dir = None
    if 'HF_MODEL_CACHE' in os.environ:
        cache_dir = os.environ['HF_MODEL_CACHE']
    pipe = pipeline("text-classification", model="valpy/prompt-classification", model_kwargs={"cache_dir": cache_dir}, device_map='cuda')

    topic_classes = []

    for out, (i,row) in tqdm(zip(pipe(data(dataset), batch_size=1), dataset.iterrows()), total=len(dataset)):
        out['text'] = row['human_turn_1']
        out['conversation_hash'] = row['conversation_hash']
        topic_classes.append(out)

    with open(output_path, 'wt') as f:
        for sample in topic_classes:
            f.write(json.dumps(sample) + '\n')
