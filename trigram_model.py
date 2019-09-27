import sys
from collections import defaultdict
import math
import random
import os
import os.path
from helper import Helper

"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile,'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def count_words_and_get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return (sum(word_counts.values()), set(word for word in word_counts if word_counts[word] > 1))


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    if Helper.is_empty(sequence):
        return []

    if n == 1:
        sequence.insert(0, 'START')
    else:
        for k in range(0, n - 1):
            sequence.insert(0, 'START')

    sequence.append('STOP')

    list_of_n_grams = []

    for i in range(0, len(sequence) - n + 1):
        n_grams = []

        for j in range(i, i + n):
            n_grams.append(sequence[j])

        list_of_n_grams.append(tuple(n_grams))

    return list_of_n_grams


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to count the total number of words
        # and to build a lexicon
        # For total number of words, 'START' and 'STOP' are excluded
        generator = corpus_reader(corpusfile)
        self.total_words, self.lexicon = count_words_and_get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            for unigram in get_ngrams(sentence, 1):
                self.unigramcounts[unigram] += 1

            for bigram in get_ngrams(sentence, 2):
                self.bigramcounts[bigram] += 1

            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return 0.0

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        return 0.0

    def generate_sentence(self,t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return 0.0

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        return float("-inf")

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        return float("inf")


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0

        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # ..

        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # ..

        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.


    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)
