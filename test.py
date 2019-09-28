import pytest
import trigram_model

@pytest.fixture
def trigram_model_brown_train():
    return trigram_model.TrigramModel('hw1_data/brown_train.txt')

def test_get_ngrams():
    assert trigram_model.get_ngrams([], 1) == []
    assert trigram_model.get_ngrams(['natural', 'language', 'processing'], 1) == [('START',), ('natural',), ('language',), ('processing',), ('STOP',)]
    assert trigram_model.get_ngrams(['natural', 'language', 'processing'], 2) == [('START', 'natural',), ('natural', 'language',), ('language', 'processing',), ('processing', 'STOP',)]
    assert trigram_model.get_ngrams(['natural', 'language', 'processing'], 3) == [('START', 'START', 'natural',), ('START', 'natural', 'language',), ('natural','language', 'processing',), ('language', 'processing', 'STOP',)]

def test_count_ngrams(trigram_model_brown_train):
    assert trigram_model_brown_train.unigramcounts[('the',)] == 61428
    assert trigram_model_brown_train.bigramcounts[('START', 'the',)] == 5478
    assert trigram_model_brown_train.trigramcounts[('START', 'START', 'the',)] == 5478

def test_generate_trigram_prob_dist(trigram_model_brown_train):
    # Probability should sum to 1, with epsilon=10^-6 to accommodate rounding issue
    assert sum(trigram_model_brown_train.generate_trigram_probability_distribution(('START', 'START'))[1]) < 1.000001
    assert sum(trigram_model_brown_train.generate_trigram_probability_distribution(('START', 'START'))[1]) > 0.999999

def test_generate_sentence(trigram_model_brown_train):
    random_sentence_1 = trigram_model_brown_train.generate_sentence(10)
    print(random_sentence_1)
    assert len(random_sentence_1) < 11
    if (len(random_sentence_1) < 10):
        assert random_sentence_1[len(random_sentence_1)-1] == 'STOP'

    random_sentence_2 = trigram_model_brown_train.generate_sentence(20)
    print(random_sentence_2)
    assert len(random_sentence_2) < 21
    if (len(random_sentence_2) < 20):
        assert random_sentence_2[len(random_sentence_2)-1] == 'STOP'
