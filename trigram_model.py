import sys
from collections import defaultdict
import math
import random
import numpy as np
import os
import os.path
from helper import Helper, start_token, stop_token, unk_token


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
                    yield [word if word in lexicon else unk_token for word in sequence]
                else:
                    yield sequence


# def count_words_and_get_lexicon(corpus):
#     word_counts = defaultdict(int)
#     for sentence in corpus:
#         for word in sentence:
#             word_counts[word] += 1
#     return (sum(word_counts.values()), set(word for word in word_counts if word_counts[word] > 1))


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    if Helper.is_empty(sequence):
        return []

    if n == 1:
        sequence.insert(0, start_token)
    else:
        for k in range(0, n - 1):
            sequence.insert(0, start_token)

    sequence.append(stop_token)

    list_of_n_grams = []

    for i in range(0, len(sequence) - n + 1):
        n_grams = []

        for j in range(i, i + n):
            n_grams.append(sequence[j])

        list_of_n_grams.append(tuple(n_grams))

    return list_of_n_grams


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add(unk_token)
        self.lexicon.add(start_token)
        self.lexicon.add(stop_token)

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        # Question to TA:
        # 1. Do we exclude 'START' and 'STOP' when calculating the total number of words for unigram probability
        # 2. Do we exclude 'START' and 'STOP' when calculating the total number of words (M) for perplexity

        # For the total number of words, 'START' and 'STOP' are excluded
        self.total_words = sum(count for word, count in self.unigramcounts.items() if word not in {start_token, stop_token})
        # self.total_words_for_perplexity = ???
        self.total_bigrams = sum(self.bigramcounts.values())
        self.total_trigrams = sum(self.trigramcounts.values())


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

        # Tricky thing is to make sure param is always in the form of tuple
        if not Helper.is_valid_n_grams(trigram, 3):
            raise Exception

        denominator = self.bigramcounts[trigram[0:2]]
        # If the denominator is zero, just return zero
        # Think of this as if p(b,c)=0, then p(a|b,c) should also be zero
        if denominator == 0:
            # Special case for ('START', 'START', 'someword',)
            # We can just calculate for ('START', 'someword',) because both
            # denote the same meaning, which is the probability of 'someword'
            # appears as the first word in a sentence
            # if trigram[0:2] == (start_token, start_token,):
            #     return raw_bigram_probability((start_token, trigram[3]))
            # But somehow, we don't need this as we always append
            # two start tokens to the sentence before calculating trigram
            return 0

        return self.trigramcounts[trigram] / denominator


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        # Tricky thing is to make sure param is always in the form of tuple
        if not Helper.is_valid_n_grams(bigram, 2):
            raise Exception

        denominator = self.unigramcounts[(bigram[0],)]

        # If the denominator is zero, just return zero
        # Think of this as if p(b)=0, then p(a|b) should also be zero
        if denominator == 0:
            return 0

        return self.bigramcounts[bigram] / denominator


    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # Tricky thing is to make sure param is always in the form of tuple
        if not Helper.is_valid_n_grams(unigram, 1):
            raise Exception

        # The denominator is always greater than zero, NaN is not a possibility
        return self.unigramcounts[unigram] / self.total_words

    # Generate trigram probability for every trigram that starts with a given bigram
    def generate_trigram_probability_distribution(self, given_bigram):
        word_list = []
        trigram_prob_dist = []
        # Ask TA is it OK if prob sums to let say 1.000000000000061 instead of 1
        # Maybe this relates to rounding

        for trigram in self.trigramcounts:
            if trigram[2] != start_token and given_bigram == trigram[0:2]:
                word_list.append(trigram[2])
                trigram_prob_dist.append(self.raw_trigram_probability(trigram))

        # Size of returned list can vary
        # For example, if size=6, that means there are only 6 words in the corpus
        # that come after the given bigram
        return word_list, trigram_prob_dist


    def generate_sentence(self, t=20, debug=False):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """

        bigram = [start_token, start_token]
        produced_words = []

        for i in range(0, t):
             # Get possible words and their probability
             word_list, trigram_prob_dist = self.generate_trigram_probability_distribution(tuple(bigram))
             if debug and len(word_list) < 20:
                 print('Observed bigram: ' + str(tuple(bigram)))
                 print('Word list: ' + str(tuple(word_list)))

             exp_prob_dist = np.random.multinomial(10, trigram_prob_dist)
             # This will return a list with the same size as trigram_prob_dist
             # Every element denotes how many times the word in that index
             # occurs in the experiment (10 times random sampling with replacement)
             # All elements of the list sum to 10
             if debug and len(word_list) < 20:
                 print('----------')
                 print(exp_prob_dist)
                 print('----------')

             # Find index with biggest occurence
             idx = np.argmax(exp_prob_dist)
             # Get the actual word
             word = word_list[idx]
             if debug and len(word_list) < 20:
                 print('Chosen word: ' + word)

             # Append word to the solution
             produced_words.append(word)
             # Shift bigram
             bigram[0] = bigram[1]
             bigram[1] = word

             if word == stop_token:
                 break

        return produced_words


    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        return lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(trigram[1:3]) + lambda3*self.raw_unigram_probability((trigram[2],))

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        log_probs_sum = 0.0
        for trigram in get_ngrams(sentence, 3):
            log_prob = self.smoothed_trigram_probability(trigram)

            if log_prob == 0:
                # Question to TA: How should we compute sentence_logprob if there is zero prob
                print('This trigram has zero probs: ' + str(trigram))
                raise Exception
                # return float("-inf")

            log_probs_sum += math.log2(log_prob)

        return log_probs_sum

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """

        corpus_unigram_counts = defaultdict(int)
        l = 0.0

        # Question to TA: We always use training lexicon right???
        generator = corpus_reader(corpus, self.lexicon)
        for sentence in generator:
            for unigram in get_ngrams(sentence, 1):
                corpus_unigram_counts[unigram] += 1

            log_prob = self.sentence_logprob(sentence)
            l += log_prob

        # Question to TA: Do we exclude 'START' and 'STOP' when calculating total words for perplexity
        corpus_total_words = sum(count for word, count in corpus_unigram_counts.items() if word not in {start_token, stop_token})
        # corpus_total_words = sum(corpus_unigram_counts.values())

        print(l)
        print(corpus_total_words)
        # return (2 ** ((-1)*l)) / corpus_total_words


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
