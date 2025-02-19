#import matplotlib.pyplot as plt
import pdb
import os
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer



def plot_top_words(model,
                   feature_names:List[str], 
                   n_top_words:int, 
                   title:str,
                   output_path:str):
    """Plots the topic clusters with each cluster 20 topics.
    """
    fig, axes = plt.subplots(20, 5, figsize=(30, 150), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    plt.savefig(os.path.join(output_path, 'topic_features_figure.pdf'), dpi=150)


def extract_topics(model,
                   feature_names:List[str], 
                   n_top_words:int,
                   output_path:str):
    """Plots the topic clusters with each cluster 20 topics.
    """
    topic_features = []
    topic_weights = []
    with open(os.path.join(output_path, 'topic_features_weights.txt'), 'a') as the_file:
    
        for _, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[-n_top_words:]
            top_features = feature_names[top_features_ind]
            weights = topic[top_features_ind]
            topic_features.append(top_features)
            topic_weights.append(weights)
            the_file.write(f'{top_features}\n')
            the_file.write(f'{weights}\n')
    return topic_features, topic_weights


def load_tf_features(n_features: int, 
                     data_samples: List[str]):
    """Uses tf (raw term count) features for LDA.
    """
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words="english")
    tf = tf_vectorizer.fit_transform(data_samples)
    tf_feature_names = tf_vectorizer.get_feature_names_out()

    return tf, tf_feature_names


def lda_model(n_topics: int):
    """Loads the LDA model.
    """
    return LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="online",
        learning_offset=50.0,
        random_state=0)



def process_string(input:str):
    return input.lower()



def main(args):

    dataset = pd.read_json(args.input_path, orient='records', lines=True)

    training_data = [process_string(t) for t in dataset['human_turn_1'].tolist()]
    training_data = [d for d in training_data if d.strip()]

    args.n_samples = len(training_data)
    data_samples = training_data[:args.n_samples]

    tf, tf_feature_names = load_tf_features(args.n_features, data_samples)
    lda = lda_model(args.n_topics)
    lda.fit(tf)

    # topic_features, topic_weights = extract_topics(lda, tf_feature_names, args.n_top_words, args.output_path)
    plot_top_words(lda, tf_feature_names, args.n_top_words, "Top 100 Topics in LDA model", args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Name of the file with the prompts", default="/shared/0/projects/research-jam-summer-2024/data/english_only/100k_results/wildchat_subset_en_100k_Mixtral-8x7B.jsonl")
    parser.add_argument("--output_path", type=str, help="Name of the file to save the generated text", default='/shared/0/projects/research-jam-summer-2024/data/topic_data/lda-outputs-100k_Mixtral-8x7B')
    parser.add_argument("--n_samples", type=int, help="number of samples",default=2000)
    parser.add_argument("--n_features", type=int, help="number of feature",default=1000)
    parser.add_argument("--n_topics", type=int, help="number of compments in LDA",default=100)
    parser.add_argument("--n_top_words", type=int, help="number of top words",default=20)
    args = parser.parse_args()


    main(args)