#import matplotlib.pyplot as plt
import little_mallet_wrapper as lmw
import pandas as pd
import pdb
import argparse


PATH_TO_MALLET = "/shared/0/resources/mallet/mallet-2.0.8/bin/mallet"

def main(args):

    dataset = pd.read_json(args.input_path, orient='records', lines=True)

    training_data = [lmw.process_string(t) for t in dataset['human_turn_1'].tolist()]
    training_data = [d for d in training_data if d.strip()]
    lmw.print_dataset_stats(training_data)

    topic_keys, topic_distributions = lmw.quick_train_topic_model(PATH_TO_MALLET, 
                                                              args.output_path, 
                                                              args.n_topics, 
                                                              training_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the prompts", default="/shared/0/projects/research-jam-summer-2024/data/english_only/100k_results/wildchat_subset_en_100k_Mixtral-8x7B.jsonl")
    parser.add_argument("--output_path", type=str, help="Name of the file to save the generated text", default='/shared/0/projects/research-jam-summer-2024/data/topic_data/mallet-outputs-100k_Mixtral-8x7B')
    parser.add_argument("--n_topics", type=int, help="number of compments in LDA",default=100)
    args = parser.parse_args()


    main(args)