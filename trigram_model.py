import sys
from collections import defaultdict
import math
import random
import numpy as np
import os
import os.path
from copy import copy


"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

# Change token to constant for readibility
start_token = 'START'
stop_token = 'STOP'
unk_token = 'UNK'

def is_empty(list_of_dict):
    if list_of_dict is None:
        return True

    if len(list_of_dict) == 0:
        return True

    return False


def is_valid_n_grams(n_grams, n):
    if n_grams is None:
        return False

    if len(n_grams) == 0:
        return False

    if not isinstance(n_grams, tuple):
        return False

    if len(n_grams) != n:
        return False

    return True


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile,'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else unk_token for word in sequence]
                else:
                    yield sequence


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

    if is_empty(sequence):
        return []

    # I don't want the value of the original sequence changes
    sequence_copy = copy(sequence)

    if n == 1:
        sequence_copy.insert(0, start_token)
    else:
        for k in range(0, n - 1):
            sequence_copy.insert(0, start_token)

    sequence_copy.append(stop_token)

    list_of_n_grams = []

    for i in range(0, len(sequence_copy) - n + 1):
        n_grams = []

        for j in range(i, i + n):
            n_grams.append(sequence_copy[j])

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

        # As clarified by Prof. Benajiba, 'START' and 'STOP' tokens count towards the total number of words
        self.total_words = sum(self.unigramcounts.values())
        # If not, it should be something like:
        # self.total_words = sum(count for word, count in self.unigramcounts.items() if word not in {(start_token,), (stop_token,)})
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

        # Make sure param is always in the form of tuple of string of size 3
        if not is_valid_n_grams(trigram, 3):
            raise Exception

        denominator = self.bigramcounts[trigram[0:2]]

        # If the denominator is zero (no such bigram), just return zero
        # Think of this as if p(b,c)=0, then p(a|b,c) should also be zero
        if denominator == 0:
            # Special case for ('START', 'START', 'someword',)
            # We can just calculate for ('START', 'someword',) because both
            # denote the same meaning, which is the probability of 'someword'
            # appears as the first word in a sentence
            if trigram[0:2] == (start_token, start_token,):
                return self.raw_bigram_probability((start_token, trigram[2]))

            return 0

        return self.trigramcounts[trigram] / denominator


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        # Make sure param is always in the form of tuple of string of size 2
        if not is_valid_n_grams(bigram, 2):
            raise Exception

        denominator = self.unigramcounts[(bigram[0],)]

        # If the denominator is zero (no such unigram), just return zero
        # Think of this as if p(b)=0, then p(a|b) should also be zero
        if denominator == 0:
            return 0

        return self.bigramcounts[bigram] / denominator


    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # Make sure param is always in the form of tuple of string of size 1
        if not is_valid_n_grams(unigram, 1):
            raise Exception

        # The denominator is always greater than zero, NaN is not a possibility
        return self.unigramcounts[unigram] / self.total_words

    # Generate trigram probability for every trigram that starts with a certain bigram
    # This is a helper function for generate_sentence
    def generate_trigram_probability_distribution(self, given_bigram):
        word_list = []
        trigram_prob_dist = []

        for trigram in self.trigramcounts:
            if trigram[2] != start_token and given_bigram == trigram[0:2]:
                word_list.append(trigram[2])
                trigram_prob_dist.append(self.raw_trigram_probability(trigram))

        # Size of returned list can vary
        # For example, if size=6, that means there are only 6 words
        # in the corpus that appear after the given bigram
        return word_list, trigram_prob_dist


    def generate_sentence(self, t=20):
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

             exp_prob_dist = np.random.multinomial(10, trigram_prob_dist)
             # This will return a list with the same size as trigram_prob_dist
             # Every element denotes how many times the word in that index
             # occurs in the experiment (10 times random sampling with replacement)
             # All elements of the list sum to 10

             # Find index with the most occurences
             idx = np.argmax(exp_prob_dist)
             # Get the actual word
             word = word_list[idx]

             # Append word to the solution
             produced_words.append(word)
             # Shift bigram
             bigram[0] = bigram[1]
             bigram[1] = word

             # Halt if 'STOP' is produced
             if word == stop_token:
                 break

        return produced_words


    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """

        # Question to TA: Do we need to preprocess sentence to replace
        # OOV words with 'UNK'

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        trigram_probability = lambda1*self.raw_trigram_probability(trigram)
        bigram_probability = lambda2*self.raw_bigram_probability(trigram[1:3])
        unigram_probability = lambda3*self.raw_unigram_probability((trigram[2],))

        return trigram_probability + bigram_probability + unigram_probability

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        # Question to TA: Do we need to preprocess sentence to replace
        # OOV words with 'UNK'

        log_probs_sum = 0.0
        for trigram in get_ngrams(sentence, 3):
            log_prob = self.smoothed_trigram_probability(trigram)

            if log_prob == 0:
                # Question to TA: How should we compute sentence_logprob if there is zero prob
                # I assume we just return minus infinity
                print('This trigram has zero probs even after smoothing: ' + str(trigram))
                return float('-inf')
            else:
                log_probs_sum += math.log2(log_prob)

        return log_probs_sum

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """

        corpus_unigram_counts = defaultdict(int)
        l = 0.0

        for sentence in corpus:
            for unigram in get_ngrams(sentence, 1):
                corpus_unigram_counts[unigram] += 1

            log_prob = self.sentence_logprob(sentence)
            l += log_prob

        # As clarified by Prof. Benajiba, 'START' and 'STOP' tokens count towards the total number of words
        # For brown test, this would yield 173-ish perplexity
        corpus_total_words = sum(corpus_unigram_counts.values())
        # If not, it should be something like:
        # corpus_total_words = sum(count for word, count in corpus_unigram_counts.items() if word not in {(start_token,), (stop_token,)})
        # This would yield 300-ish perplexity for brown test

        l = l / corpus_total_words
        return 2 ** ((-1)*l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1) # trained on train_high.txt
        model2 = TrigramModel(training_file2) # trained on train_low.txt

        total = 0
        correct = 0

        # This will be for test_high hence the pp_high < pp_low condition
        for f in os.listdir(testdir1):
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            if pp_high < pp_low:
                correct = correct + 1

            total = total + 1

        # This will be for test_low hence the pp_low < pp_high condition
        for f in os.listdir(testdir2):
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))

            if pp_low < pp_high:
                correct = correct + 1

            total = total + 1

        return correct / total

if __name__ == "__main__":
    # Loading model
    data_dir = 'hw1_data/'
    if len(sys.argv) > 1:
        model = TrigramModel(sys.argv[1])
    else:
        model = TrigramModel(data_dir + 'brown_train.txt')

    # Test get_ngrams
    assert get_ngrams([], 1) == []
    assert get_ngrams(['natural', 'language', 'processing'], 1) == [(start_token,), ('natural',), ('language',), ('processing',), (stop_token,)]
    assert get_ngrams(['natural', 'language', 'processing'], 2) == [(start_token, 'natural',), ('natural', 'language',), ('language', 'processing',), ('processing', stop_token,)]
    assert get_ngrams(['natural', 'language', 'processing'], 3) == [(start_token, start_token, 'natural',), (start_token, 'natural', 'language',), ('natural','language', 'processing',), ('language', 'processing', stop_token,)]

    # Test count ngrams
    assert model.unigramcounts[(start_token,)] == 41614
    assert model.unigramcounts[(start_token,)] == model.unigramcounts[(stop_token,)]
    assert model.unigramcounts[('the',)] == 61428
    assert model.bigramcounts[(start_token, 'the',)] == 5478
    assert model.trigramcounts[(start_token, start_token, 'the',)] == 5478

    # Test raw probabilities
    assert model.raw_unigram_probability(('the',)) == 61428/1084179
    assert model.raw_bigram_probability(('the', 'jury',)) == 35/61428
    assert model.raw_trigram_probability(('the', 'jury', 'said',)) == 7/35

    # Zero probs
    assert model.raw_unigram_probability(('casdajnkuakk',)) == 0 # no such unigram
    assert model.raw_bigram_probability(('the', 'casdajnkuakk',)) == 0 # no such bigram (numerator is zero)
    assert model.raw_bigram_probability(('casdajnkuakk', 'the',)) == 0 # no such preceding unigram (denominator is zero), also return zero
    assert model.raw_trigram_probability(('the', 'jury', 'casdajnkuakk',)) == 0 # no such trigram (numerator is zero)
    assert model.raw_trigram_probability(('casdajnkuakk', 'the', 'jury',)) == 0 # no such preceding bigram (denominator is zero), also return zero

    # Special cases for (start_token, start_token, anything)
    assert model.raw_bigram_probability((start_token, 'the',)) == 5478/41614
    assert model.raw_trigram_probability((start_token, start_token, 'the',)) == model.raw_bigram_probability((start_token, 'the',))

    # Test smoothed_trigram_probability
    assert model.smoothed_trigram_probability(('casdajnkuakk', 'the', 'jury',)) == (1/3)*(0) + (1/3)*(35/61428) + (1/3)*(59/1084179) # trigram is not found, but bigram and unigram are found
    assert model.smoothed_trigram_probability(('the', 'casdajnkuakk', 'jury',)) == (1/3)*(0) + (1/3)*(0) + (1/3)*(59/1084179) # only unigram is found
    assert model.smoothed_trigram_probability(('the', 'jury', 'casdajnkuakk',)) == 0 # All trigram, bigram and unigram are not found

    # Test sentence_logprob
    assert model.sentence_logprob(['the', 'jury', 'said', 'casdajnkuakk']) == float('-inf')

    # Test perplexity
    if len(sys.argv) > 2:
        test_corpus = corpus_reader(sys.argv[2], model.lexicon)
    else:
        test_corpus = corpus_reader(data_dir + 'brown_test.txt', model.lexicon)

    # brown_test perplexity
    pp = model.perplexity(test_corpus)
    print('Perplexity for brown_test: ')
    print(pp)
    assert pp < 400.0

    # brown_train perplexity
    train_corpus = corpus_reader(data_dir + 'brown_train.txt', model.lexicon)
    pp = model.perplexity(train_corpus)
    print('Perplexity for brown_train: ')
    print(pp)
    assert pp < 20.0

    # Essay scoring experiment
    ets_toefl_data_dir = data_dir + 'ets_toefl_data/'
    acc = essay_scoring_experiment(ets_toefl_data_dir + 'train_high.txt', ets_toefl_data_dir + 'train_low.txt', ets_toefl_data_dir + 'test_high', ets_toefl_data_dir + 'test_low')
    print('Essay scoring accuracy: ')
    print(acc)
    assert pp > 0.8

    # Test generate random sentence
    random_sentence_1 = model.generate_sentence(10)
    print('Sample random sentence 1 of max length 10:')
    print(random_sentence_1)
    assert len(random_sentence_1) < 11
    if (len(random_sentence_1) < 10):
        assert random_sentence_1[len(random_sentence_1)-1] == stop_token

    random_sentence_2 = model.generate_sentence()
    print('Sample random sentence 2 of max length 20:')
    print(random_sentence_2)
    assert len(random_sentence_2) < 21
    if (len(random_sentence_2) < 20):
        assert random_sentence_2[len(random_sentence_2)-1] == stop_token
