import string 
from tqdm import tqdm
import pickle

import scipy
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA

# Experiment 1
WEAT_words = {
'A':['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill'], 
'B':['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna'],
'C':['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career'],
'D':['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives'],
'E':['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition'],
'F':['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture'],
'G':['science', 'technology', 'physics', 'chemistry', 'einstein', 'nasa', 'experiment', 'astronomy'],
'H':['poetry', 'art', 'shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama'],
}


def has_punct(w):
    
    if any([c in string.punctuation for c in w]):
        return True
    return False

def has_digit(w):
    
    if any([c in '0123456789' for c in w]):
        return True
    return False

def limit_vocab(wv, w2i, vocab, exclude = None):
    vocab_limited = []
    for w in tqdm(vocab[:50000]): 
        if w.lower() != w:
            continue
        if len(w) >= 20:
            continue
        if has_digit(w):
            continue
        if '_' in w:
            p = [has_punct(subw) for subw in w.split('_')]
            if not any(p):
                vocab_limited.append(w)
            continue
        if has_punct(w):
            continue
        vocab_limited.append(w)
    
    if exclude:
        vocab_limited = list(set(vocab_limited) - set(exclude))
    
    print("size of vocabulary:", len(vocab_limited))
    
    wv_limited = np.zeros((len(vocab_limited), len(wv[0, :])))
    for i,w in enumerate(vocab_limited):
        wv_limited[i,:] = wv[w2i[w],:]
    
    w2i_limited = {w: i for i, w in enumerate(vocab_limited)}
    
    return vocab_limited, wv_limited, w2i_limited

def norm_stand(wv):
    W_norm = np.zeros(wv.shape)
    d = (np.sum(wv ** 2, 1) ** (0.5))
    W_norm = (wv.T / d).T
    return W_norm

def normalize(wv):
    
    # normalize vectors
    norms = np.apply_along_axis(LA.norm, 1, wv)
    wv = wv / norms[:, np.newaxis]
    return wv


def topK(w, wv, w2i, vocab, k=10):
    
    # extract the word vector for word w
    idx = w2i[w]
    vec = wv[idx, :]
    
    # compute similarity of w with all words in the vocabulary
    sim = wv.dot(vec)
#     sim = []
#     for i in range(len(wv)):
#         sim.append(1-scipy.spatial.distance.cosine(wv[i, :], vec))
#     sim = np.array(sim)
        
    # sort similarities by descending order
    sort_sim = (sim.argsort())[::-1]

    # choose topK
    best = sort_sim[:(k+1)]

    return [vocab[i] for i in best if i!=idx]


def similarity(w1, w2, wv, w2i):
    
    i1 = w2i[w1]
    i2 = w2i[w2]
    vec1 = wv[i1, :]
    vec2 = wv[i2, :]

    return 1-scipy.spatial.distance.cosine(vec1, vec2)



def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)

from sklearn.decomposition import PCA
from sklearn import preprocessing

def doPCA(pairs, wv, w2i):
        
    matrix = []
    cnt = 0
    
    if type(pairs[0]) is list:
        for a, b in pairs:
            if not (a in w2i and b in w2i): continue
            center = (wv[w2i[a], :] + wv[w2i[b], :])/2
            matrix.append(wv[w2i[a], :] - center)
            matrix.append(wv[w2i[b], :] - center)
            cnt += 1
    else:
        for a in pairs:
            if not (a in w2i): continue
            matrix.append(wv[w2i[a], :])
            cnt += 1
        
        embeds = np.array(matrix)
        wv_mean = np.mean(np.array(embeds), axis=0)
        wv_hat = np.zeros(embeds.shape).astype(float)
    
        for i in range(len(embeds)):
            wv_hat[i, :] = embeds[i, :] - wv_mean
        matrix = wv_hat
            
    matrix = np.array(matrix)
    pca = PCA()
    pca.fit(matrix)
    print('pairs used in PCA: ', cnt)
    return pca

# get tuples of biases and counts of masculine/feminine NN for each word (for bias-by-neighbors)
import operator
def bias_by_neighbors(wv, w2i, vocab, gender_bias_bef, size, neighbours_num = 100):
    
    tuples = []
    
    sorted_g = sorted(gender_bias_bef.items(), key=operator.itemgetter(1))
    female = [item[0] for item in sorted_g[:size]]
    male = [item[0] for item in sorted_g[-size:]]
#     vocab = male + female
    selected = female + male if size > 0 else vocab
    
    for w in selected:
        
        top = topK(w, wv, w2i, vocab, k=neighbours_num+5)[:neighbours_num]

        m = 0
        f = 0    
        for t in top:
            if gender_bias_bef[t] > 0:
                m+=1
            else:
                f+=1
            
        tuples.append((w, gender_bias_bef[w], m, f))

    return tuples

def get_tuples_prof(wv, w2i, vocab,  words, gender_bias_dict):
    
    wv = normalize(wv)
    
    tuples = []
    for w in words:
        if w not in gender_bias_dict:
            continue
            
        top = topK(w, wv, w2i, vocab, k=105)[:100]
            
        m = 0
        f = 0  
        for t in top:          
            if gender_bias_dict[t] > 0:
                m+=1
            else:
                f+=1
                
        tuples.append((w, gender_bias_dict[w], m, f))
        
    return tuples

# compute correlation between bias-by-projection and bias-by-neighbors

import scipy.stats

def pearson(a,b):
   
    return scipy.stats.pearsonr(a,b)

def compute_corr(tuples, i1, i2):
    
    a = []
    b = []
    for t in tuples:
        a.append(t[i1])
        b.append(t[i2])
    assert(len(a)==len(b))    
    print('pearson: ', scipy.stats.pearsonr(a,b))
    print('spearman: ', scipy.stats.spearmanr(a, b))
    
# Auxiliary finctions

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def visualize(vectors, y_true, y_pred, ax, title, random_state, num_clusters = 2):
    
    # perform TSNE
    
    X_embedded = TSNE(n_components=2, random_state=random_state).fit_transform(vectors)
    for x,p,y in zip(X_embedded, y_pred, y_true):
        if p:
            if y:
                ax.scatter(x[0], x[1], marker = '.', c = 'c')
            else:
                ax.scatter(x[0], x[1], marker = 'x', c = 'c')
        else:
            if y:
                ax.scatter(x[0], x[1], marker = '.', c = 'darkviolet')
            else:
                ax.scatter(x[0], x[1], marker = 'x', c = 'darkviolet')
                        
    
    ax.text(.01, .9, title ,transform=ax.transAxes, fontsize=15)

    
def extract_vectors(words, wv, w2i):
    
    X = [wv[w2i[x],:] for x in words]
    
    return X


def cluster_and_visualize(words, X, random_state, y_true, num=2):
    
    y_pred = KMeans(n_clusters=num, random_state=random_state).fit_predict(X)
#     fig, axs = plt.subplots(figsize=(6, 3))
#     visualize(X, y_true, y_pred, axs, 'Original', random_state)
    correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred) ]
    print('precision', max(sum(correct)/float(len(correct)), 1 - sum(correct)/float(len(correct))))

    
import scipy.stats
from sklearn import svm
def train_and_predict(wv, w2i, vocab, size_train, size_test, males, females):
    
    X_train = [wv[w2i[w],:] for w in males[:size_train]+females[:size_train]]
    Y_train = [1]*size_train + [0]*size_train
    X_test = [wv[w2i[w],:] for w in males[size_train:]+females[size_train:]]
    Y_test = [1]*size_test + [0]*size_test

    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, Y_train)

    preds = clf.predict(X_test)

    accuracy = [1 if y==z else 0 for y,z in zip(preds, Y_test)]
    acc = float(sum(accuracy))/len(accuracy)
    print('accuracy:', float(sum(accuracy))/len(accuracy))
    
    return acc
    
    
# Auxiliary functions for experiments by Caliskan et al.

import scipy
import scipy.misc as misc
import itertools


def s_word(w, A, B, wv, w2i, vocab, all_s_words):
    
    if w in all_s_words:
        return all_s_words[w]
    
    mean_a = []
    mean_b = []
    
    for a in A:
        mean_a.append(similarity(w, a, wv, w2i))
    for b in B:
        mean_b.append(similarity(w, b, wv, w2i))
        
    mean_a = sum(mean_a)/float(len(mean_a))
    mean_b = sum(mean_b)/float(len(mean_b))
    
    all_s_words[w] = mean_a - mean_b

    return all_s_words[w]


def s_group(X, Y, A, B,  wv, w2i, vocab, all_s_words):
    
    total = 0
    for x in X:
        total += s_word(x, A, B,  wv, w2i, vocab, all_s_words)
    for y in Y:
        total -= s_word(y, A, B,  wv, w2i, vocab, all_s_words)
        
    return total


def p_value_exhust(X, Y, A, B,  wv, w2i, vocab):
    
    if len(X) > 10:
        print('might take too long, use sampled version: p_value')
        return
    
    assert(len(X) == len(Y))
    
    all_s_words = {}
    s_orig = s_group(X, Y, A, B, wv, w2i, vocab, all_s_words) 
    
    union = set(X+Y)
    subset_size = int(len(union)/2)
    
    larger = 0
    total = 0
    for subset in set(itertools.combinations(union, subset_size)):
        total += 1
        Xi = list(set(subset))
        Yi = list(union - set(subset))
        if s_group(Xi, Yi, A, B, wv, w2i, vocab, all_s_words) > s_orig:
            larger += 1
    print('num of samples', total)
    return larger/float(total)

def association_diff(t, A, B, wv, w2i):
    
    mean_a = []
    mean_b = []
    
    for a in A:
        mean_a.append(similarity(t, a, wv, w2i))
    for b in B:
        mean_b.append(similarity(t, b, wv, w2i))
        
    mean_a = sum(mean_a)/float(len(mean_a))
    mean_b = sum(mean_b)/float(len(mean_b))
    
    return mean_a - mean_b

def effect_size(X, Y, A, B,  wv, w2i, vocab):
    
    assert(len(X) == len(Y))
    assert(len(A) == len(B))
    
    norm_x = []
    norm_y = []
    
    for x in X:
        norm_x.append(association_diff(x, A, B, wv, w2i))
    for y in Y:
        norm_y.append(association_diff(y, A, B, wv, w2i))
    
    std = np.std(norm_x+norm_y, ddof=1)
    norm_x = sum(norm_x) / float(len(norm_x))
    norm_y = sum(norm_y) / float(len(norm_y))
    
    return (norm_x-norm_y)/std


def p_value_sample(X, Y, A, B, wv, w2i, vocab):
    
    random.seed(10)
    np.random.seed(10)
    all_s_words = {}
    
    assert(len(X) == len(Y))
    length = len(X)
    
    s_orig = s_group(X, Y, A, B,  wv, w2i, vocab, all_s_words) 
    
    num_of_samples = min(1000000, int(scipy.special.comb(length*2,length)*100))
    print('num of samples', num_of_samples)
    larger = 0
    for i in range(num_of_samples):
        permute = np.random.permutation(X+Y)
        Xi = permute[:length]
        Yi = permute[length:]
        if s_group(Xi, Yi, A, B, space, all_s_words) > s_orig:
            larger += 1
    
    return larger/float(num_of_samples) 
    
    