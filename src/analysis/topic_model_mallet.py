import matplotlib.pyplot as plt
import little_mallet_wrapper as lmw
import pandas as pd
import pdb
import argparse

path_to_mallet = "/shared/0/resources/mallet/mallet-2.0.8/bin/mallet"

def main(args):

    dataset = pd.read_json(args.input_path, orient='records', lines=True)

    

    training_data = [lmw.process_string(t) for t in dataset['text'].tolist()]
    training_data = [d for d in training_data if d.strip()]
    
    lmw.print_dataset_stats(training_data)

    num_topics = 20  # CHANGE THIS TO YOUR PREFERRED NUMBER OF TOPICS

    output_directory_path = '/shared/0/projects/research-jam-summer-2024/data/topic_data/mallet-outputs' # CHANGE THIS TO YOUR OUTPUT DIRECTORY

    topic_keys, topic_distributions = lmw.quick_train_topic_model(path_to_mallet, 
                                                              output_directory_path, 
                                                              num_topics, 
                                                              training_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the prompts", default="/shared/0/projects/research-jam-summer-2024/data/topic_data/100k-with-topics.jsonl")
    parser.add_argument("--output_path", type=str, help="Name of the file to save the generated text")
    parser.add_argument("--n_samples", type=int, help="number of samples",default=2000)
    parser.add_argument("--n_features", type=int, help="number of feature",default=1000)
    parser.add_argument("--n_components", type=int, help="number of compments in LDA",default=10)
    parser.add_argument("--n_top_words", type=int, help="number of top words",default=20)
    args = parser.parse_args()


    main(args)