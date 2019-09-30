import pytest
import trigram_model
import time
from helper import Helper, start_token, stop_token, unk_token


@pytest.fixture
def trigram_model_brown_train():
    print()
    return trigram_model.TrigramModel('hw1_data/brown_train.txt')


# TESTED
def test_get_ngrams():
    assert trigram_model.get_ngrams([], 1) == []
    assert trigram_model.get_ngrams(['natural', 'language', 'processing'], 1) == [(start_token,), ('natural',), ('language',), ('processing',), (stop_token,)]
    assert trigram_model.get_ngrams(['natural', 'language', 'processing'], 2) == [(start_token, 'natural',), ('natural', 'language',), ('language', 'processing',), ('processing', stop_token,)]
    assert trigram_model.get_ngrams(['natural', 'language', 'processing'], 3) == [(start_token, start_token, 'natural',), (start_token, 'natural', 'language',), ('natural','language', 'processing',), ('language', 'processing', stop_token,)]


# TESTED
def test_count_ngrams(trigram_model_brown_train):
    assert trigram_model_brown_train.unigramcounts[('the',)] == 61428
    assert trigram_model_brown_train.bigramcounts[(start_token, 'the',)] == 5478
    assert trigram_model_brown_train.trigramcounts[(start_token, start_token, 'the',)] == 5478


# TESTED
def test_raw_probability(trigram_model_brown_train):
    # print(trigram_model_brown_train.raw_unigram_probability((start_token,)))
    # print(trigram_model_brown_train.raw_bigram_probability((start_token, 'the')))
    # print(trigram_model_brown_train.raw_trigram_probability((start_token, start_token, 'the')))
    assert trigram_model_brown_train.raw_trigram_probability((start_token, start_token, 'the',)) == trigram_model_brown_train.raw_bigram_probability((start_token, 'the',))


# TESTED
def test_sentence_prob(trigram_model_brown_train):
    # print(trigram_model_brown_train.raw_unigram_probability((start_token,)))
    # print(trigram_model_brown_train.raw_bigram_probability((start_token, 'the')))
    # print(trigram_model_brown_train.raw_trigram_probability((start_token, start_token, 'the')))
    # In this case, prob ('with', 'a', 'snick',) == 0 because 'snick' should be 'UNK'
    assert trigram_model_brown_train.sentence_logprob(['the', 'blade', 'came', 'out', 'with', 'a', 'snick', '!']) == 0


# TESTED
def test_generate_trigram_prob_dist(trigram_model_brown_train):
    # Probability should sum to 1, with epsilon=10^-6 to accommodate rounding issue
    assert sum(trigram_model_brown_train.generate_trigram_probability_distribution((start_token, start_token))[1]) < 1.000001
    assert sum(trigram_model_brown_train.generate_trigram_probability_distribution((start_token, start_token))[1]) > 0.999999


# TESTED
def test_generate_sentence(trigram_model_brown_train):
    random_sentence_1 = trigram_model_brown_train.generate_sentence(10)
    # print(random_sentence_1)
    assert len(random_sentence_1) < 11
    if (len(random_sentence_1) < 10):
        assert random_sentence_1[len(random_sentence_1)-1] == stop_token

    random_sentence_2 = trigram_model_brown_train.generate_sentence(20)
    # print(random_sentence_2)
    assert len(random_sentence_2) < 21
    if (len(random_sentence_2) < 20):
        assert random_sentence_2[len(random_sentence_2)-1] == stop_token


# TESTED
def test_perplexity(trigram_model_brown_train):
    start = time.time()

    test_perplexity = trigram_model_brown_train.perplexity(trigram_model.corpus_reader('hw1_data/brown_test.txt', trigram_model_brown_train.lexicon))
    print(test_perplexity)
    assert test_perplexity < 400.0

    print('Seconds elapsed: ')
    print(time.time() - start)


# TESTED
def test_perplexity_train(trigram_model_brown_train):
    start = time.time()

    test_perplexity = trigram_model_brown_train.perplexity('hw1_data/brown_train.txt')
    print(test_perplexity)
    assert test_perplexity < 400.0

    print('Seconds elapsed: ')
    print(time.time() - start)


# WRONG
def test_essay_scoring(trigram_model_brown_train):
    start = time.time()

    accuracy = trigram_model.essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', 'hw1_data/ets_toefl_data/test_high', 'hw1_data/ets_toefl_data/test_low')
    print(accuracy)
    assert accuracy > 0.8

    print('Seconds elapsed: ')
    print(time.time() - start)
