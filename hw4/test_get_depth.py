from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np

lemma_set = set([])

for ss in wn.synsets('clever', 'a'):
    pos = ss.name().split('.')[1]
    for tt in ss.lemma_names():
        print(tt)
        jj = wn.lemmas(tt, pos)
        for lexeme in jj:
            lemma_set.add(lexeme.name())

print(lemma_set)
