## [Double-Hard Debias: Tailoring Word Embeddings for Gender Bias Mitigation](https://arxiv.org/abs/2005.00965)
[Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), [Xi Victoria Lin](http://victorialin.net/), [Nazneen Fatema Rajani](http://www.nazneenrajani.com/), [Bryan McCann](https://bmccann.github.io/), [Vicente Ordonez](https://www.vicenteordonez.com/), [Caiming Xiong](http://cmxiong.com/)

### Abstract
Word embeddings derived from humangenerated corpora inherit strong gender bias which can be further amplified by downstream models. Some commonly adopted debiasing approaches, including the seminal Hard Debias algorithm (Bolukbasi et al., 2016), apply post-processing procedures that project pre-trained word embeddings into a subspace orthogonal to an inferred gender subspace. We discover that semantic-agnostic corpus regularities such as word frequency captured by the word embeddings negatively impact the performance of these algorithms. We propose a simple but effective technique, Double-Hard Debias, which purifies the word embeddings against such corpus regularities prior to inferring and removing the gender subspace. Experiments on three bias mitigation benchmarks show that our approach preserves the distributional semantics of the pre-trained word embeddings while reducing gender bias to a significantly larger degree than prior approaches.

### Requirements
- Python 3/2
- [Word Embedding Benckmarks](https://github.com/kudkudak/word-embeddings-benchmarks)

### Data
- Word Embeddings: Please download all word embeddings([Google Drive](https://drive.google.com/drive/folders/1-WcKbViKdl-wBvSXoMq9p_PYCNBzeUe5?usp=sharing)) and save them into [data](./data) folder.
- Special word lists: You can find all word lists used in this project in [data](./data) folder.

### Double-Hard-Debias
You can find the detailed steps to implement Double-Hard Debias in [GloVe_Debias](./GloVe_Debias.ipynb)

### Evaluation
We evaluated Double-Hard Debias and other debiasing approaches on [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) and [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) embeddings. Please check the results in [GloVe_Eval](./GloVe_Eval.ipynb)
and [GloVe_Eval](./GloVe_Debias.ipynb)


### Citing
If you find our paper/code useful, please consider citing:

```
@InProceedings{wang2020doublehard,
author={Tianlu Wang and Xi Victoria Lin and Nazneen Fatema Rajani and Bryan McCann and Vicente Ordonez and Caiming Xiong},
title={Double-Hard Debias: Tailoring Word Embeddings for Gender Bias Mitigation},
booktitle = {Association for Computational Linguistics (ACL)},
month = {July},
year = {2020}
}
```

### Kudos
This project is developed based on [gender_bias_lipstick](https://github.com/gonenhila/gender_bias_lipstick) and [word-embeddings-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks). Thanks for their efforts!
