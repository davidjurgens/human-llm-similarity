import pandas as pd
import numpy as np
import os
import re
from scipy.spatial.distance import jensenshannon, cosine
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True


# KEYS TO DO DATAFRAME MERGES ON (i.e., what uniquely identifies a row)
merge_keys = ['conversation_hash','hashed_ip','model','prompt']


# METRIC DICTIONARIES
lexical = {'word_count':'scalar',
           'word_length':'scalar',
           'perplexity':'scalar',
           'typo':'scalar',
          }

syntactic = {'pos':'distribution',
             'dep_dpth':'scalar',
             'dep_brth':'scalar',
             'dep_dep_dist':'scalar',
            }

semantic = {'sbert':'vector',
            'liwc':'distribution',
            'topic':'distribution',
           }

style = {'punctuation':'distribution',
         'capitalization':'scalar',
         'sentiment':'scalar',
         'politeness':'scalar',
         'formality':'scalar',
         'toxicity':'scalar',
         'readability':'scalar',
         'subjectivity':'scalar',
         'luar':'vector',
        }

all_metrics = {k:v for metrics_dict in [lexical, syntactic, semantic, style] for k,v in metrics_dict.items()}


metric_category = {}
for k in lexical: metric_category[k] = 'lexical'
for k in syntactic: metric_category[k] = 'syntactic'
for k in semantic: metric_category[k] = 'semantic'
for k in style: metric_category[k] = 'style'

# READ, MERGE, AND UPDATE DATA

def capital_rate(text):
    # count number of capital letters
    capital_count = sum(1 for char in text if char.isupper())

    # count number of aplhabet letters (i.e., letters that can be capitalized)
    total_letter_count = sum(1 for char in text if char.isalpha())

    if total_letter_count > 0:
        return capital_count / total_letter_count
    else:
        return np.nan

def word_length(utterance):
    return np.mean([len(word) for word in utterance.split()])

def list_to_dict(x):
    return {'category_'+str(i): x[i] for i in range(len(x))}

def add_total(x):
    x['balance'] = max(0, 1 - sum(x.values()))
    return x

def replace_with_pca(df):
    for k in all_metrics:
        if all_metrics[k] != 'scalar':
            if all_metrics[k] == 'distribution': 
                X = np.array([list(x.values()) for col in ['human_','llm_'] for x in df[col+k]])
            elif all_metrics[k] == 'vector':
                X = np.array([x for col in ['human_','llm_'] for x in df[col+k]])
            else:
                print(k, 'error')
            pca = PCA(n_components=2)
            pca.fit(X)

            print(k, X.shape)
            print(pca.explained_variance_ratio_)
            print()

            X_pc = pca.fit_transform(X)
            df['human_'+k] = X_pc[:df.shape[0]].tolist()
            df['llm_'+k] = X_pc[df.shape[0]:].tolist()
    return df

def make_human_vs_llm_df(f, base_dir, human_prefix='human_', sim_prefix='llm_', 
                         read_dep = True, read_emb = True, turn_3_prefix = 'turn_3_'):
    print(f)
    
    print('read metrics')
    # read base metrics
    df = pd.read_json(base_dir+f, orient='records', lines=True)
    #fix metrics
    df[human_prefix+'word_count'] = np.exp(df[human_prefix+'log_word_count']) 
    df[sim_prefix+'word_count'] = np.exp(df[sim_prefix+'log_word_count']) 
    df[human_prefix+'readability'] = df[human_prefix+'readability'].replace(-1, np.NaN) 
    df[sim_prefix+'readability'] = df[sim_prefix+'readability'].replace(-1, np.NaN)
    df[human_prefix+'capitalization'] = df[human_prefix+'turn_3'].apply(capital_rate)
    df[sim_prefix+'capitalization'] = df[sim_prefix+'turn_3'].apply(capital_rate)
    df[human_prefix+'word_length'] = df[human_prefix+'turn_3'].apply(word_length)
    df[sim_prefix+'word_length'] = df[sim_prefix+'turn_3'].apply(word_length)
    df[human_prefix+'topic'] = df[human_prefix+'topic'].apply(list_to_dict)
    df[sim_prefix+'topic'] = df[sim_prefix+'topic'].apply(list_to_dict)
    df[human_prefix+'liwc'] = df[human_prefix+'liwc'].apply(add_total)
    df[sim_prefix+'liwc'] = df[sim_prefix+'liwc'].apply(add_total)
    df[human_prefix+'punctuation'] = df[human_prefix+'punctuation'].apply(add_total)
    df[sim_prefix+'punctuation'] = df[sim_prefix+'punctuation'].apply(add_total)
    #subset columns
    df = df[[c for c in df.columns 
             if c in merge_keys or re.sub(human_prefix+'|'+sim_prefix,'',c) in all_metrics]]

    if read_dep:
        df = df.drop([human_prefix+'pos',sim_prefix+'pos'], axis=1)
        print('read dependency parse metrics')
        # read dependency parse metrics
        dep = pd.read_json(base_dir+re.sub('.jsonl','_POS_DEP.jsonl',f), orient='records', lines=True)
        dep = dep[[c for c in dep.columns if c in merge_keys or re.sub(human_prefix+'|'+sim_prefix,'',c) in all_metrics]]
        df = df.merge(dep, on = merge_keys, how = 'left')

    if read_emb:
        print('read embeddings')
        # read embeddings 
        b = np.load(base_dir+re.sub('.jsonl','_embeddings.npz',f), allow_pickle=True)
        emb = pd.DataFrame({human_prefix+'sbert': b[human_prefix+turn_3_prefix+'sbert'].tolist(),
                            sim_prefix+'sbert': b[sim_prefix+turn_3_prefix+'sbert'].tolist(),
                            human_prefix+'luar': b[human_prefix+turn_3_prefix+'luar'].tolist(),
                            sim_prefix+'luar': b[sim_prefix+turn_3_prefix+'luar'].tolist()
                            #'conversation_hash': b['conversation_hash']
                            })
        df = pd.concat([df, emb], axis=1)
    
    # which model
    df['model'] = re.sub('.jsonl','',f)
    
    return df


# CALCULATE SIMILARITY MEASURES BY ROW

def min_max_scale(x):
    min_x = min(x)
    max_x = max(x)
    return (x - min_x)/(max_x - min_x)

def zscore_scale(x):
    mean_x = np.mean(x)
    sd_x = np.std(x)
    return (x - mean_x)/sd_x

def jitter_prob(p, eps = 1./1000.):
    if p == 0: p = p + eps
    elif p == 1: p = p - eps
    return p



def dist(human, llm, metric_type, p=1):
    if metric_type == 'scalar':
        return 1 - np.abs((llm - human) ** p) # absolute difference (1 - |diff| is similarity)
    
    if metric_type == 'distribution':
        return 1 - jensenshannon([pd.Series(list(h.values())).apply(jitter_prob) for h in human], # JSD (1 - JSD is similarity)
                                 [pd.Series(list(l.values())).apply(jitter_prob) for l in llm], axis=1) #jitter to handle 0/1
    
    if metric_type == 'vector':
        return [1 - cosine(human[i], llm[i]) for i in human.index] # cosine similarity (1 - cosine dist)

def diff_unit(human, llm, metric_type, p = 1, rescale = False):
    if metric_type == 'scalar':
        if rescale: #step 1: put both human and llm on a 0-1 scale so abs difference is in [0,1]
            
            human = min_max_scale(human)
            llm = min_max_scale(llm)
        return dist(human, llm, metric_type, p) #step 2: distance
    
    if metric_type == 'distribution':
        return dist(human, llm, metric_type) # directly return distance (JSD is in [0,1])
    
    if metric_type == 'vector':
        cos = dist(human, llm, metric_type) #step 1: distance
        return [(x + 1) / 2 for x in cos] #step 2: rescale to 0-1 scale

def diff_zscore(human, llm, metric_type, p=1):
    diff = dist(human, llm, metric_type, p) #step 1: distance
    return zscore_scale(diff) #step 2: z-score

def dist_row_correlation(row, metrics_dict=all_metrics):
    human = np.hstack([row['human_'+k] for k in metrics_dict])
    llm = np.hstack([row['llm_'+k] for k in metrics_dict])
    return correlate(human,llm)

def row_mean(df, cols, method=''):
    return df[[method+k for k in cols]].apply(lambda x: np.mean(x), axis=1)

def aggregate_scores(df, method=''):
    if method!='': method = method + '_'
    df.loc[:,'agg_'+method+'lexical'] = row_mean(df, lexical, method)
    df.loc[:,'agg_'+method+'syntactic'] = row_mean(df, syntactic, method)
    df.loc[:,'agg_'+method+'semantic'] = row_mean(df, semantic, method)
    df.loc[:,'agg_'+method+'style'] = row_mean(df, style, method)
    df.loc[:,'overall_'+method+'score'] = row_mean(df, [k for k in df.columns if k.startswith('agg_')])

    sns.kdeplot(df['overall_'+method+'score'], label = 'overall')
    for k in df.columns: 
        if k.startswith('agg_'):
            sns.kdeplot(df[k], label = k)
    plt.legend(title= "", loc= "upper left")
    plt.show()
    
    return df


# CALCULATE SIMILARITY MEASURES BY COLUMN

def correlate(human, llm, corr_method='pearson'): #'pearson', 'kendall', 'spearman'
    return pd.DataFrame({'human':human, 'llm':llm}).corr(method=corr_method)['human']['llm']

def distribution_scale(human, llm, scale = min_max_scale):
    full_vector = [x[j] for v in [human,llm] for j in range(2) for x in v]
    std_vector = scale(np.array(full_vector))
    h0 = std_vector[:len(human)]
    h1 = std_vector[len(human):2*len(human)]
    l0 = std_vector[2*len(human):3*len(human)]
    l1 = std_vector[3*len(human):]
    return [[h0[i], h1[i]] for i in range(len(human))], [[l0[i], l1[i]] for i in range(len(human))]

def col_diff_correlate(human, llm, metric_type, corr_method):
    if metric_type == 'scalar':
        return correlate(human, llm, corr_method) #direct correlation of columns
    
    if metric_type == 'distribution':
        dict_keys = human.tolist()[0]
        return np.nanmean([correlate(human.apply(lambda x: x[k1]), #average correlation along each dimension
                                     llm.apply(lambda x: x[k1]), corr_method) for k1 in dict_keys])
            
    if metric_type == 'vector':
        n = len(human)
        X = np.array([x for col in [human,llm] for x in col])
        pca = PCA(n_components=X.shape[1])
        X_pc = pca.fit_transform(X)
        h = X_pc[:n]
        l = X_pc[n:]
        return np.nanmean([correlate(h[:,i], l[:,i], corr_method) for i in range(h.shape[1])])
#         return correlate([x for i in human.index for x in human[i]], #correlation of all vectors
#                          [x for i in llm.index for x in llm[i]])


def corr_metric(corr_method, var = 'model', human_prefix='human_', sim_prefix='llm_'):
    vals = []
    metric = []
    cor = []
    for lvl in set(metrics[var]):
        print(lvl)
        sub = metrics[metrics[var] == lvl]
        for k in all_metrics:
            #print(k)
            vals.append(lvl)
            metric.append(k)
            cor.append(col_diff_correlate(sub[human_prefix+k], sub[sim_prefix+k], all_metrics[k], corr_method))
    col_corr = pd.DataFrame({var: vals, 'metric': metric, 'cor': cor, 'corr_method': corr_method})
    col_corr['category'] = col_corr['metric'].replace(metric_category)
    return col_corr


