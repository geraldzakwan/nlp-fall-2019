#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np

from collections import Counter
import string

# Participate in the 4705 lexical substitution competition (optional): YES
# Alias: peaky_blinders

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()

def lower(sentence):
    lowered_sentence = []
    for word in sentence:
        lowered_sentence.append(word.lower())

    return lowered_sentence

def remove_punctuation(sentence):
    cleaned_sentence = []
    for word in sentence:
        if word not in string.punctuation:
            cleaned_sentence.append(word)

    return cleaned_sentence

def remove_stopwords(sentence):
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)

    new_sentence = []
    for word in sentence:
        # word = word.lower()

        if word not in stop_words:
            new_sentence.append(word)

    return new_sentence

def compute_overlap(context, definition):
    overlap_count = 0
    for word in context:
        pass

def get_candidates(lemma, pos):
    # Part 1

    # Return solution as a set to make sure unique lemmas are returned
    possible_synonyms = set([])

    # Retrieve all lexemes for the particular lemma and pos
    lexemes = wn.lemmas(lemma, pos=pos)

    # print(lexemes)

    # Iterate over lexemes
    for lexeme in lexemes:
        # Get the synset for current lexeme
        synset = lexeme.synset()

        # Get the lexemes from the synset
        for candidate_lemma in synset.lemmas():
            # Retrieve the name from a lemma structure
            candidate_lemma_name = candidate_lemma.name()

            # Make sure we don't add input lemma as solution
            if candidate_lemma_name != lemma:
                # Check if lemma contains multiple words
                if len(candidate_lemma_name.split('_')) > 1:
                    # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
                    candidate_lemma_name = candidate_lemma_name.replace('_', ' ')

                # Add to the solution
                possible_synonyms.add(candidate_lemma_name)

    # Return the set
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    lemma = context.lemma
    pos = context.pos

    # print(lemma)
    # print(pos)
    # print()

    # Return solution as a set to make sure unique lemmas are returned
    synonyms_counter = Counter()

    # Retrieve all lexemes for the particular lemma and pos
    lexemes = wn.lemmas(lemma, pos=pos)

    # Iterate over lexemes
    for lexeme in lexemes:
        # Get the synset for current lexeme
        synset = lexeme.synset()

        # Get the lemmas from the synset
        for candidate_lemma in synset.lemmas():
            # Retrieve the name from a lemma structure
            candidate_lemma_name = candidate_lemma.name()

            # Make sure we don't add input lemma as solution
            if candidate_lemma_name != lemma:
                # print(candidate_lemma.name())
                # print(candidate_lemma.count())
                # print()

                # Check if lemma contains multiple words
                if len(candidate_lemma_name.split('_')) > 1:
                    # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
                    candidate_lemma_name = candidate_lemma_name.replace('_', ' ')

                # Add to the solution
                synonyms_counter[candidate_lemma_name] += candidate_lemma.count()

    # Return the set
    return synonyms_counter.most_common(1)[0][0] # replace for part 2

def get_most_frequent_lexeme(synset, input_lemma):
    max_lexeme = None
    max_count = -1

    # Get the lemmas from the synset
    for lexeme in synset.lemmas():
        print(lexeme.count())
        # Make sure we don't add input lemma as solution
        if lexeme.name() != input_lemma:
            if lexeme.count() > max_count:
                max_lexeme = lexeme
                max_count = lexeme.count()

    # Rare case when all counts equal to zero
    if max_count == -1:
        # Just return the first one
        return synset.lemmas()[0]

    return max_lexeme

def compute_overlap(cleaned_full_context, sense):
    overlap = 0
    cleaned_full_context_set = set(cleaned_full_context)

    raw_definition = sense.definition()
    definition = tokenize(raw_definition)
    examples = sense.examples()

    cleaned_definition = remove_stopwords(definition)
    cleaned_examples = remove_stopwords(examples)

    overlap += len(cleaned_full_context_set.intersection(set(cleaned_definition)))
    overlap += len(cleaned_full_context_set.intersection(set(cleaned_examples)))

    hypernyms = sense.hypernyms()
    for hypernym in hypernyms:
        raw_definition = hypernym.definition()
        definition = tokenize(raw_definition)
        examples = hypernym.examples()

        cleaned_definition = remove_stopwords(definition)
        cleaned_examples = remove_stopwords(examples)

        overlap += len(cleaned_full_context_set.intersection(set(cleaned_definition)))
        overlap += len(cleaned_full_context_set.intersection(set(cleaned_examples)))

    return overlap

def wn_simple_lesk_predictor(context):
    lemma = context.lemma
    pos = context.pos

    print(lemma)

    full_context = context.left_context + context.right_context

    # Some stack of preprocessings
    cleaned_full_context = remove_punctuation(full_context)
    cleaned_full_context = lower(cleaned_full_context)
    cleaned_full_context = remove_stopwords(cleaned_full_context)

    # Retrieve all lexemes for the particular lemma and pos
    lexemes = wn.lemmas(lemma, pos=pos)

    best_sense = None
    max_overlap = -1
    # Iterate over lexemes
    for lexeme in lexemes:
        # Get the synset for current lexeme
        synset = lexeme.synset()

        overlap = compute_overlap(cleaned_full_context, synset)

        if overlap > max_overlap:
            best_sense = synset
            max_overlap = overlap

    most_frequent_lexeme = get_most_frequent_lexeme(best_sense, lemma)
    lemma_name = most_frequent_lexeme.name()

    # Check if lemma contains multiple words
    if len(lemma_name.split('_')) > 1:
        # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
        lemma_name = lemma_name.replace('_', ' ')

    return lemma_name #replace for part 3

class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context):
        return None # replace for part 4

    def predict_nearest_with_context(self, context):
        return None # replace for part 5

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # print(get_candidates('slow', 'a'))
    # print(len(get_candidates('slow', 'a')))

    # print(remove_stopwords(['i', 'love', 'you']))
    # sys.exit()

    for context in read_lexsub_xml(sys.argv[1]):
        # print(type(context))
        # print(context)  # useful for debugging
        # print(context.pos)
        # get_candidates(context.lemma, context.pos)
        # get_candidates('slow', 'a')
        # print(wn_frequency_predictor(context))
        # prediction = smurf_predictor(context)
        prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # print(prediction)
        # sys.exit()
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
