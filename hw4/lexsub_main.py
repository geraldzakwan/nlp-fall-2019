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

def get_candidates(lemma, pos):
    # Return solution as a set to make sure unique lemmas are returned
    possible_synonyms = set([])

    # Retrieve all lexemes for the particular lemma and pos
    lexemes = wn.lemmas(lemma, pos=pos)

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

                # Add lemma to the solution
                possible_synonyms.add(candidate_lemma_name)

    # Return the set
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def return_frequency(context):
    # Counter with lemma as key and its count as value
    synonyms_counter = Counter()

    # Retrieve all lexemes for the particular lemma and pos
    lexemes = wn.lemmas(context.lemma, pos=context.pos)

    # Iterate over lexemes
    for lexeme in lexemes:
        # Get the synset for current lexeme
        synset = lexeme.synset()

        # Get the lemmas from the synset
        for candidate_lemma in synset.lemmas():
            candidate_lemma_name = candidate_lemma.name()

            # Make sure we don't add input lemma as solution
            if candidate_lemma_name != context.lemma:
                # Check if lemma contains multiple words
                if len(candidate_lemma_name.split('_')) > 1:
                    # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
                    candidate_lemma_name = candidate_lemma_name.replace('_', ' ')

                # Add to the solution, the same lemma can be added twice
                # if it appears together with input lemma in multiple synsets
                synonyms_counter[candidate_lemma_name] += candidate_lemma.count()

    # If there is a tie, pick an arbitrary lemma, whatever comes first
    # in the front of the list after sorted descendingly
    return synonyms_counter

def wn_frequency_predictor(context):
    # Counter with lemma as key and its count as value
    synonyms_counter = Counter()

    # Retrieve all lexemes for the particular lemma and pos
    lexemes = wn.lemmas(context.lemma, pos=context.pos)

    # Iterate over lexemes
    for lexeme in lexemes:
        # Get the synset for current lexeme
        synset = lexeme.synset()

        # Get the lemmas from the synset
        for candidate_lemma in synset.lemmas():
            candidate_lemma_name = candidate_lemma.name()

            # Make sure we don't add input lemma as solution
            if candidate_lemma_name != context.lemma:
                # Check if lemma contains multiple words
                if len(candidate_lemma_name.split('_')) > 1:
                    # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
                    candidate_lemma_name = candidate_lemma_name.replace('_', ' ')

                # Add to the solution, the same lemma can be added twice
                # if it appears together with input lemma in multiple synsets
                synonyms_counter[candidate_lemma_name] += candidate_lemma.count()

    # If there is a tie, pick an arbitrary lemma, whatever comes first
    # in the front of the list after sorted descendingly
    return synonyms_counter.most_common(1)[0][0]

def get_most_frequent_lexeme(synset, input_lemma):
    max_lexeme = None
    max_count = 0

    # Get the lemmas from the synset
    for lexeme in synset.lemmas():
        # print('EUY')
        # print(synset.lemmas())
        # print(lexeme)
        # print(lexeme.count())
        # Make sure we don't add input lemma as solution
        if lexeme.name() != input_lemma:
            if lexeme.count() > max_count:
                max_lexeme = lexeme
                max_count = lexeme.count()

    # Rare case when all counts equal to zero
    if max_count == 0:
        # raise Exception('Wew')
        # Just return the first one
        return synset.lemmas()[0]

    return max_lexeme

def get_most_frequent_synset(synset_overlap_list, input_lemma):
    max_count = 0
    most_frequent_synset = None

    for tup in synset_overlap_list:
        synset = tup[0]
        lexemes = synset.lemmas()

        # Important: Don't process synset that only has input lemma as its lexeme
        proceed = True
        if len(lexemes) == 1:
            if lexemes[0].name() == input_lemma:
                # print('PERNAH')
                # print(lexemes)
                proceed = False

        if proceed:
            synset = tup[0]
            count = 0

            for lexeme in synset.lemmas():
                count += lexeme.count()

            if count > max_count:
                most_frequent_synset = synset
                max_count = count

    return most_frequent_synset

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

    if overlap > 0:
        # print('YEY, OVERLAP')
        # print(overlap)
        # print('CONTEXT')
        # print(cleaned_full_context)
        # print('SENSE')
        # print(sense)
        # print('------------------')
        pass

    return overlap

def get_cleaned_full_context(context, window_size=-1):
    left_context = context.left_context
    right_context = context.right_context

    if window_size > 0:
        if len(left_context) > window_size:
            left_context = left_context[len(left_context) - window_size:len(left_context)]

        if len(right_context) > window_size:
            right_context = right_context[0:window_size]

    # Append left_context and right_context
    full_context = left_context + right_context

    # Some stack of preprocessings
    cleaned_full_context = remove_punctuation(full_context)
    cleaned_full_context = lower(cleaned_full_context)
    cleaned_full_context = remove_stopwords(cleaned_full_context)

    return cleaned_full_context

def wn_simple_lesk_predictor(context):
    cleaned_full_context = get_cleaned_full_context(context)

    # List of tuple: (synset, overlap_count with context)
    synset_overlap_list = []

    # Iterate over synsets
    for synset in wn.synsets(context.lemma, context.pos):
        # Important: Don't process synset that only has input lemma as its lexeme
        lexemes = synset.lemmas()
        proceed = True
        if len(lexemes) == 1:
            if lexemes[0].name() == context.lemma:
                # print('PERNAH')
                # print(lexemes)
                proceed = False

        if proceed:
            overlap = compute_overlap(cleaned_full_context, synset)

            synset_overlap_list.append((synset, overlap))

    # Sort based on overlap_count
    synset_overlap_list = sorted(synset_overlap_list, key=lambda x: x[1], reverse=True)
    # print('1')
    # print(synset_overlap_list)

    best_tup = synset_overlap_list[0]
    best_synset = best_tup[0]
    best_overlap = best_tup[1]
    # print('2')
    # print(best_sense)
    if best_overlap == 0:
        best_sense = get_most_frequent_synset(synset_overlap_list, context.lemma)
        # print('3')
        # print(best_sense)

    most_frequent_lexeme = get_most_frequent_lexeme(best_synset, context.lemma)
    # print('4')
    # print(most_frequent_lexeme)
    lemma_name = most_frequent_lexeme.name()

    # Check if lemma contains multiple words
    if len(lemma_name.split('_')) > 1:
        # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
        lemma_name = lemma_name.replace('_', ' ')

    return lemma_name #replace for part 3

class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        # print('Finish loading')

    def predict_nearest(self, context):
        possible_synonyms = list(get_candidates(context.lemma, context.pos))

        # Question: What if synonyms OOV???
        # For now, ignore
        considered_synonyms = []
        for synonym in possible_synonyms:
            if synonym in self.model.wv:
                considered_synonyms.append(synonym)

        return self.model.most_similar_to_given(context.lemma, considered_synonyms)

    def get_nearest_synonym(self, target_vector, considered_synonyms, metrics='cosine_similarity'):
        max_cosine_similarity = -1.0
        nearest_synonym = None

        for i in range(0, len(considered_synonyms)):
            syn_vector = self.model.wv[considered_synonyms[i]]

            if metrics == 'cosine_similarity':
                dot_prod = np.dot(target_vector, syn_vector)
                target_norm = np.linalg.norm(target_vector)
                syn_norm = np.linalg.norm(syn_vector)
                cosine_similarity = dot_prod / (target_norm * syn_norm)
            else:
                cosine_similarity = -1.0

            if cosine_similarity > max_cosine_similarity:
                nearest_synonym = considered_synonyms[i]
                max_cosine_similarity = cosine_similarity

        return nearest_synonym

    def predict_nearest_with_context(self, context, window_size=5, include_target=True):
        # print(return_frequency(context))

        cleaned_full_context = get_cleaned_full_context(context, window_size)

        if include_target:
            target_vector = self.model.wv[context.lemma]
        else:
            target_vector = np.zeros(300, dtype='float32')

        for word in cleaned_full_context:
            # Ignore oov this time
            if word in self.model.wv:
                target_vector = np.add(target_vector, self.model.wv[word])

        possible_synonyms = list(get_candidates(context.lemma, context.pos))

        # Question: What if synonyms OOV???
        # For now, ignore
        considered_synonyms = []
        for synonym in possible_synonyms:
            if synonym in self.model.wv:
                considered_synonyms.append(synonym)

        return self.get_nearest_synonym(target_vector, considered_synonyms)

    # Don't do nothing
    # def predict_nearest_with_context_average(self, context):
    #     cleaned_full_context = get_cleaned_full_context(context, 5)
    #
    #     target_vector = self.model.wv[context.lemma]
    #     for word in cleaned_full_context:
    #         # Ignore oov this time
    #         if word in self.model.wv:
    #             target_vector = np.add(target_vector, self.model.wv[word])
    #
    #     # Take the average
    #     target_vector = np.divide(target_vector, len(cleaned_full_context) + 1)
    #
    #     possible_synonyms = list(get_candidates(context.lemma, context.pos))
    #
    #     # Question: What if synonyms OOV???
    #     # For now, ignore
    #     considered_synonyms = []
    #     for synonym in possible_synonyms:
    #         if synonym in self.model.wv:
    #             considered_synonyms.append(synonym)
    #
    #     return self.get_nearest_synonym(target_vector, considered_synonyms)

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

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
        # prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # print(prediction)
        # sys.exit()
        # prediction = predictor.predict_nearest(context)
        # prediction = predictor.predict_nearest_with_context(context, 5, False)
        prediction = predictor.predict_nearest_with_context(context, 7)
        # prediction = predictor.predict_nearest_with_context(context, 1, False)
        # sys.exit()
        # prediction = predictor.predict_nearest_with_context_average(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
