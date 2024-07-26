import pdb
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def plot_top_words(model,
                   feature_names:List[str], 
                   n_top_words:int, 
                   title:str):
    """Plots the topic clusters with each cluster 20 topics.
    """
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
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
    plt.savefig("/shared/0/projects/research-jam-summer-2024/data/topic_data/100k-with-topics.pdf", dpi=150)


def extract_topics(model,
                   feature_names:List[str], 
                   n_top_words:int):
    """Plots the topic clusters with each cluster 20 topics.
    """
    topic_features = []
    topic_weights = []
    for _, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]
        topic_features.append(top_features)
        topic_weights.append(weights)
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


def lda_model(n_components: int):
    """Loads the LDA model.
    """
    return LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0)


def main(args):

    dataset = pd.read_json(args.input_path, orient='records', lines=True)

    topic_classes = []
    topic_texts   = []
    for _, row in tqdm(dataset.iterrows()):
        topic_classes.append(row['label'])
        topic_texts.append(row['text'])

    args.n_samples = len(topic_texts)
    data_samples = topic_texts[:args.n_samples]

    tf, tf_feature_names = load_tf_features(args.n_features, data_samples)
    lda = lda_model(args.n_components)
    lda.fit(tf)

    topic_features, topic_weights = extract_topics(lda, tf_feature_names, args.n_top_words)
    plot_top_words(lda, tf_feature_names, args.n_top_words, "Topics in LDA model")



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