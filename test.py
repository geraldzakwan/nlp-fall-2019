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
