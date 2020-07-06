import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW, fetch_TR9856
from web.datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS, fetch_ESSLI_1a, fetch_ESSLI_2b, \
    fetch_ESSLI_2c
from web.analogy import *
from six import iteritems
from web.embedding import Embedding
from web.evaluate import calculate_purity, evaluate_categorization, evaluate_on_semeval_2012_2, evaluate_analogy, \
evaluate_on_WordRep, evaluate_similarity

def evaluate_similarity_pearson(w, X, y):
    """
    Calculate Pearson correlation between cosine similarity of the model
    and human rated similarity of word pairs
    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.
    X: array, shape: (n_samples, 2)
      Word pairs
    y: vector, shape: (n_samples,)
      Human ratings
    Returns
    -------
    cor: float
      Pearson correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    missing_words = 0
    words = w.vocabulary.word_id
    for query in X:
        for query_word in query:
            if query_word not in words:
                missing_words += 1
    if missing_words > 0:
        print("Missing {} words. Will replace them with mean vector".format(missing_words))

    new_x = []
    new_y = []
    for i in range(len(X)):
        if X[i, 0] in words and X[i, 1] in words:
            new_x.append(X[i])
            new_y.append(y[i])
    
    X = np.array(new_x)
    y = np.array(new_y)
    
    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A = np.vstack(list(w.get(word, mean_vector) for word in X[:, 0]))
    B = np.vstack(list(w.get(word, mean_vector) for word in X[:, 1]))
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    return scipy.stats.pearsonr(scores, y.squeeze())

def evaluate_similarity(w, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    X: array, shape: (n_samples, 2)
      Word pairs

    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    missing_words = 0
    words = w.vocabulary.word_id
    for query in X:
        for query_word in query:
            if query_word not in words:
                missing_words += 1
#     if missing_words > 0:
#         print("Missing {} words. Will replace them with mean vector".format(missing_words))

    new_x = []
    new_y = []
    exist_cnt = 0
    
    for i in range(len(X)):
        if X[i, 0] in words and X[i, 1] in words:
            new_x.append(X[i])
            new_y.append(y[i])
            exist_cnt += 1
    
    print('exist {} in {}'.format(exist_cnt, len(X)))
    X = np.array(new_x)
    y = np.array(new_y)
    
    
    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A = np.vstack(w.get(word, mean_vector) for word in X[:, 0])
    B = np.vstack(w.get(word, mean_vector) for word in X[:, 1])
#     scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    scores = np.array([v1.dot(v2.T) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation


def evaluate_simi(wv, w2i, vocab):
    wv_dict = dict()
    for w in vocab:
        wv_dict[w] = wv[w2i[w], :]
        
    if isinstance(wv_dict, dict):
        w = Embedding.from_dict(wv_dict)

    # Calculate results on similarity
    print("Calculating similarity benchmarks")
    similarity_tasks = {
        "WS353": fetch_WS353(),
        "RG65": fetch_RG65(),
#         "WS353R": fetch_WS353(which="relatedness"),
#         "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "MTurk": fetch_MTurk(),
        "RW": fetch_RW(),
        "MEN": fetch_MEN(),
    }

#     similarity_results = {}

    for name, data in iteritems(similarity_tasks):
        print("Sample data from {}, num of samples: {} : pair \"{}\" and \"{}\" is assigned score {}".format(
            name, len(data.X), data.X[0][0], data.X[0][1], data.y[0]))
        score = evaluate_similarity(w, data.X, data.y)
        print("Spearman correlation of scores on {} {}".format(name, score))
#         score, p_value = evaluate_similarity_pearson(w, data.X, data.y)
#         print("Pearson correlation of scores on {} {}, p value: {}".format(name, score, p_value))

def evaluate_categorization(w, X, y, method="kmeans", seed=None):
    """
    Evaluate embeddings on categorization task.

    Parameters
    ----------
    w: Embedding or dict
      Embedding to test.

    X: vector, shape: (n_samples, )
      Vector of words.

    y: vector, shape: (n_samples, )
      Vector of cluster assignments.

    method: string, default: "all"
      What method to use. Possible values are "agglomerative", "kmeans", "all.
      If "agglomerative" is passed, method will fit AgglomerativeClustering (with very crude
      hyperparameter tuning to avoid overfitting).
      If "kmeans" is passed, method will fit KMeans.
      In both cases number of clusters is preset to the correct value.

    seed: int, default: None
      Seed passed to KMeans.

    Returns
    -------
    purity: float
      Purity of the best obtained clustering.

    Notes
    -----
    KMedoids method was excluded as empirically didn't improve over KMeans (for categorization
    tasks available in the package).
    """

    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    assert method in ["all", "kmeans", "agglomerative"], "Uncrecognized method"

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    new_x = []
    new_y = []
    exist_cnt = 0
    
    for idx, word in enumerate(X.flatten()):
        if word in w :
            new_x.append(X[idx])
            new_y.append(y[idx])
            exist_cnt += 1
    
    print('exist {} in {}'.format(exist_cnt, len(X)))
    X = np.array(new_x)
    y = np.array(new_y)
    
    words = np.vstack([w.get(word, mean_vector) for word in X.flatten()])
    ids = np.random.RandomState(seed).choice(range(len(X)), len(X), replace=False)

    # Evaluate clustering on several hyperparameters of AgglomerativeClustering and
    # KMeans
    best_purity = 0

    if method == "all" or method == "agglomerative":
        best_purity = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                       affinity="euclidean",
                                                                       linkage="ward").fit_predict(words[ids]))
        logger.debug("Purity={:.3f} using affinity={} linkage={}".format(best_purity, 'euclidean', 'ward'))
        for affinity in ["cosine", "euclidean"]:
            for linkage in ["average", "complete"]:
                purity = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                          affinity=affinity,
                                                                          linkage=linkage).fit_predict(words[ids]))
                logger.debug("Purity={:.3f} using affinity={} linkage={}".format(purity, affinity, linkage))
                best_purity = max(best_purity, purity)

    if method == "all" or method == "kmeans":
        purity = calculate_purity(y[ids], KMeans(random_state=seed, n_init=10, n_clusters=len(set(y))).
                                  fit_predict(words[ids]))
        logger.debug("Purity={:.3f} using KMeans".format(purity))
        best_purity = max(purity, best_purity)

    return best_purity

def evaluate_cate(wv, w2i, vocab, method="all", seed=None):
    """
    method: string, default: "all"
      What method to use. Possible values are "agglomerative", "kmeans", "all.
      If "agglomerative" is passed, method will fit AgglomerativeClustering (with very crude
      hyperparameter tuning to avoid overfitting).
      If "kmeans" is passed, method will fit KMeans.
      In both cases number of clusters is preset to the correct value.
    seed: int, default: None
      Seed passed to KMeans.
    """
    wv_dict = dict()
    for w in vocab:
        wv_dict[w] = wv[w2i[w], :]
        
    if isinstance(wv_dict, dict):
        w = Embedding.from_dict(wv_dict)    
    
    # Calculate results on categorization
    print("Calculating categorization benchmarks")
    categorization_tasks = {
        "AP": fetch_AP(),
#         "ESSLI_2c": fetch_ESSLI_2c(),
#         "ESSLI_2b": fetch_ESSLI_2b(),
        "ESSLI_1a": fetch_ESSLI_1a(),
        "Battig": fetch_battig(),
        "BLESS": fetch_BLESS(),
    }

    categorization_results = {}

    # Calculate results using helper function
    for name, data in iteritems(categorization_tasks):
        print("Sample data from {}, num of samples: {} : \"{}\" is assigned class {}".format(
            name, len(data.X), data.X[0], data.y[0]))
        categorization_results[name] = evaluate_categorization(w, data.X, data.y, method=method, seed=None)
        print("Cluster purity on {} {}".format(name, categorization_results[name]))

def evaluate_analogy_google(W, vocab):
    """Evaluate the trained w vectors on a variety of tasks"""

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]
    prefix = '/zf15/tw8cb/summer_2019/code/GloVe/eval/question-data/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                +  W[ind3[subset], :])
            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("%s:" % filenames[i])
        print('ACCURACY TOP1: %.2f%% (%d/%d)' %
            (np.mean(val) * 100, np.sum(val), len(val)))

    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    print('Semantic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print('Syntactic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


def evaluate_analogy_msr(W, vocab, file_name='EN-MSR.txt'):
    """Evaluate the trained word vectors on a variety of tasks"""

    prefix = '/zf15/tw8cb/summer_2019/code/GloVe/eval/question-data/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    with open('%s/%s' % (prefix, file_name), 'r') as f:
        full_data = []
        for line in f:
            tokens = line.rstrip().split(' ')
            full_data.append([tokens[0], tokens[1], tokens[2], tokens[4]])
        full_count += len(full_data)
        data = [x for x in full_data if all(word in vocab for word in x)]

    indices = np.array([[vocab[word] for word in row] for row in data])
    ind1, ind2, ind3, ind4 = indices.T

    predictions = np.zeros((len(indices),))
    num_iter = int(np.ceil(len(indices) / float(split_size)))
    for j in range(num_iter):
        subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

        pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
            +  W[ind3[subset], :])
        #cosine similarity if input W has been normalized
        dist = np.dot(W, pred_vec.T)

        for k in range(len(subset)):
            dist[ind1[subset[k]], k] = -np.Inf
            dist[ind2[subset[k]], k] = -np.Inf
            dist[ind3[subset[k]], k] = -np.Inf

        # predicted word index
        predictions[subset] = np.argmax(dist, 0).flatten()

    val = (ind4 == predictions) # correct predictions
    count_tot = count_tot + len(ind1)
    correct_tot = correct_tot + sum(val)

#     print("%s:" % filenames[i])
    print(len(val))
    print('ACCURACY TOP1-MSR: %.2f%% (%d/%d)' %
        (np.mean(val) * 100, np.sum(val), len(val)))

def evaluate_analogy_semeval2012(w_dict):
    score = evaluate_on_semeval_2012_2(w_dict)['all']
    print("Analogy prediction accuracy on {} {}".format("SemEval2012", score))

def evaluate_ana(wv, w2i, vocab):
    W_norm = np.zeros(wv.shape)
    d = (np.sum(wv ** 2, 1) ** (0.5))
    W_norm = (wv.T / d).T
    
    evaluate_analogy_msr(W_norm, w2i)
    evaluate_analogy_google(W_norm, w2i)
    
    wv_dict = dict()
    for w in vocab:
        wv_dict[w] = W_norm[w2i[w], :]
        
    if isinstance(wv_dict, dict):
        w = Embedding.from_dict(wv_dict)
    evaluate_analogy_semeval2012(w)

#     analogy_tasks = {
#         "Google": fetch_google_analogy(),
#         "MSR": fetch_msr_analogy()
#     }

#     analogy_results = {}

#     for name, data in iteritems(analogy_tasks):
#         analogy_results[name] = evaluate_analogy(w, data.X, data.y)
#         print("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))