import gensim, logging, os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import nltk

corpus = nltk.corpus.brown.sents()
fname = 'brown_skipgram.model'
if os.path.exists(fname):
    # load the file if it has already been trained, to save repeating the slow training step below
    model = gensim.models.Word2Vec.load(fname)
else:
    # can take a few minutes, grab a cuppa
    model = gensim.models.Word2Vec(corpus, size=100, min_count=5, workers=2, iter=50)
    model.save(fname)

words = "woman women man girl boy green blue did".split()
for w1 in words:
    for w2 in words:
        print(w1, w2, model.similarity(w1, w2))

import numpy as np
import matplotlib.pyplot as plt

M = np.zeros((len(words), len(words)))
for i, w1 in enumerate(words):
    for j, w2 in enumerate(words):
        M[i, j] = model.similarity(w1, w2)

plt.imshow(M, interpolation='nearest')
plt.colorbar()

ax = plt.gca()
ax.set_xticklabels([''] + words, rotation=45)
ax.set_yticklabels([''] + words)

model.most_similar(positive=['woman'], topn=10)