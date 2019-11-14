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
import time

# Participate in the 4705 lexical substitution competition (optional): YES
# Alias: peaky_blinders

# Below is a set of normalization functions
# HELPER FUNCTION
def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()

# HELPER FUNCTION
def lower(sentence):
    lowered_sentence = []
    for word in sentence:
        lowered_sentence.append(word.lower())

    return lowered_sentence

# HELPER FUNCTION
def remove_punctuation(sentence):
    cleaned_sentence = []
    for word in sentence:
        if word not in string.punctuation:
            cleaned_sentence.append(word)

    return cleaned_sentence

# HELPER FUNCTION
def remove_numbers(sentence):
    cleaned_sentence = []
    for word in sentence:
        if not word.isnumeric():
            cleaned_sentence.append(word)

    return cleaned_sentence

# HELPER FUNCTION
def remove_stopwords(sentence):
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)

    new_sentence = []
    for word in sentence:
        if word not in stop_words:
            new_sentence.append(word)

    return new_sentence

# HELPER FUNCTION
def normalize(sentence):
    # Some stack of preprocessings from above subfunctions
    # to normalize a list of tokens
    sentence = remove_punctuation(sentence)
    sentence = remove_numbers(sentence)
    sentence = lower(sentence)
    sentence = remove_stopwords(sentence)

    return sentence

# PART 1
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
            candidate_lemma_name = candidate_lemma.name().lower()

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

# PART 2
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
            candidate_lemma_name = candidate_lemma.name().lower()

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

# PART 3
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
                proceed = False

        if proceed:
            overlap = compute_overlap(cleaned_full_context, synset)
            synset_overlap_list.append((synset, overlap))

    # Use subfunction because there are many cases
    best_synset = resolve_best_synset(synset_overlap_list)

    # Get the most frequent lexeme
    most_frequent_lexeme = get_most_frequent_lexeme(best_synset, context.lemma)
    lemma_name = most_frequent_lexeme.name()

    # Check if lemma name contains multiple words
    if len(lemma_name.split('_')) > 1:
        # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
        lemma_name = lemma_name.replace('_', ' ')

    return lemma_name

# HELPER FUNCTION
# For odd window_size, pad='left' means that we put more words in the left
# E.g. for window_size=5 and pad='left', there will be 3 words in the left_context
# and 2 words in right_context. Pad='right' does the opposite.
def get_cleaned_full_context(context, window_size=-1, pad='left', when_to_normalize='before'):
    left_context = context.left_context
    right_context = context.right_context

    if when_to_normalize == 'before':
        left_context = normalize(left_context)
        right_context = normalize(right_context)

    if window_size > 0:
        left_window_size = window_size // 2

        if window_size % 2 == 1:
            if pad == 'left':
                left_window_size = left_window_size + 1

        right_window_size = window_size - left_window_size

        if len(left_context) > left_window_size:
            left_context = left_context[len(left_context) - left_window_size:len(left_context)]

        if len(right_context) > right_window_size:
            right_context = right_context[0:right_window_size]

    if when_to_normalize == 'after':
        left_context = normalize(left_context)
        right_context = normalize(right_context)

    # Append left_context and right_context
    full_context = left_context + right_context

    return full_context

# HELPER FUNCTION
def get_most_frequent_lexeme(synset, input_lemma):
    max_lexeme = None
    max_count = 0

    # Get the lemmas from the synset
    for lexeme in synset.lemmas():
        # Make sure we don't add input lemma as solution
        if lexeme.name() != input_lemma:
            if lexeme.count() > max_count:
                max_lexeme = lexeme
                max_count = lexeme.count()

    # Case when all counts equal to zero (not sure why WordNet
    # would have count equals to zero for so many lexemes)
    if max_count == 0:
        # Just return the first one
        return synset.lemmas()[0]

    return max_lexeme

# HELPER FUNCTION
def get_most_frequent_synset(synset_overlap_list, input_lemma):
    # Set initial max_count = -1 to make sure that most_frequent_synset
    # will not be None if all lexeme counts are zero
    # In that case, the first lexeme (that is not the same with input_lemma)
    # will be returned as the substitute
    max_count = -1
    most_frequent_synset = None

    for tup in synset_overlap_list:
        synset = tup[0]
        lexemes = synset.lemmas()

        # Important: Don't process synset that only has input lemma as its lexeme
        proceed = True
        if len(lexemes) == 1:
            if lexemes[0].name() == input_lemma:
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

# HELPER FUNCTION
# Assumption: We don't count duplicate overlap
# So we can use set to quickly compute it (using intersection)
def compute_overlap(cleaned_full_context, sense):
    overlap = 0
    cleaned_full_context_set = set(cleaned_full_context)

    # Count overlap with synset definition & example
    raw_definition = sense.definition()
    definition = tokenize(raw_definition)
    examples = sense.examples()

    cleaned_definition = normalize(definition)
    cleaned_examples = normalize(examples)

    overlap += len(cleaned_full_context_set.intersection(set(cleaned_definition)))
    overlap += len(cleaned_full_context_set.intersection(set(cleaned_examples)))

    # Count overlap with definitions & examples from synset hypernyms
    hypernyms = sense.hypernyms()
    for hypernym in hypernyms:
        raw_definition = hypernym.definition()
        definition = tokenize(raw_definition)
        examples = hypernym.examples()

        cleaned_definition = normalize(definition)
        cleaned_examples = normalize(examples)

        overlap += len(cleaned_full_context_set.intersection(set(cleaned_definition)))
        overlap += len(cleaned_full_context_set.intersection(set(cleaned_examples)))

    return overlap

# HELPER FUNCTION
def resolve_best_synset(synset_overlap_list):
    # Sort based on overlap_count, descendingly
    synset_overlap_list = sorted(synset_overlap_list, key=lambda x: x[1], reverse=True)
    max_overlap = synset_overlap_list[0][1]

    # If no overlap, return the most frequent synset
    if max_overlap == 0:
        return get_most_frequent_synset(synset_overlap_list, context.lemma)

    # Get all synsets with max overlap
    synset_with_max_overlap = []
    for tup in synset_overlap_list:
        overlap = tup[1]

        if overlap < max_overlap:
            break

        synset_with_max_overlap.append(tup)

    # Only 1 synset with max overlap, return it
    if len(synset_with_max_overlap) == 1:
        return synset_with_max_overlap[0][0]

    # If more than one, resolve using get_most_frequent_synset,
    # this time using only synset with max overlap
    return get_most_frequent_synset(synset_with_max_overlap, context.lemma)

# HELPER FUNCTION
def is_candidate_lemma_valid(candidate_lemma_name, lemma):
    # if candidate_lemma_name != lemma:
    # if lemma not in candidate_lemma_name and candidate_lemma_name not in lemma:
    if candidate_lemma_name not in lemma:
        return True

    return False

# HELPER FUNCTION
# NOTE: THIS IS NOTE FOR PART 1
def get_best_predictor_candidates(lemma, pos):
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
            candidate_lemma_name = candidate_lemma.name().lower()

            if is_candidate_lemma_valid(candidate_lemma_name, lemma):
                # Check if lemma contains multiple words
                if len(candidate_lemma_name.split('_')) > 1:
                    if pos == 'a' or pos == 'r':
                        candidate_lemma_name = candidate_lemma_name.replace('_', '-')
                    else:
                        # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
                        candidate_lemma_name = candidate_lemma_name.replace('_', ' ')

                # Add lemma to the solution
                possible_synonyms.add(candidate_lemma_name)

        # This generates a quite huge boost, from 0.129 to 0.136
        hypernym_synsets = synset.hypernyms()
        for hi_syn in hypernym_synsets:
            for candidate_lemma in hi_syn.lemmas():
                # Retrieve the name from a lemma structure
                candidate_lemma_name = candidate_lemma.name().lower()

                if is_candidate_lemma_valid(candidate_lemma_name, lemma):
                    # Check if lemma contains multiple words
                    if len(candidate_lemma_name.split('_')) > 1:
                        if pos == 'a' or pos == 'r':
                            candidate_lemma_name = candidate_lemma_name.replace('_', '-')
                        else:
                            # Replace '_' with ' ', e.g. 'turn_around' -> 'turn around'
                            candidate_lemma_name = candidate_lemma_name.replace('_', ' ')

                    # Add lemma to the solution
                    possible_synonyms.add(candidate_lemma_name)

    # Return the set
    return possible_synonyms

class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    # PART 4
    def predict_nearest(self, context):
        possible_synonyms = list(get_candidates(context.lemma, context.pos))

        # Assumption: We can ignore synonym candidates
        # that are not in the word2vec vocabulary
        considered_synonyms = []
        for synonym in possible_synonyms:
            if synonym in self.model.wv:
                considered_synonyms.append(synonym)

        # From my experiment, every lemma is in the word2vec vocabulary. Nothing to handle.
        target_vector = self.model.wv[context.lemma]

        return self.get_nearest_synonym(target_vector, considered_synonyms)

        # We can also use the most_similar_to_given function provided by word2vec to
        # select the most similar word from considered_synonyms by using cosine_similarity
        # This yields the same result.
        # return self.model.most_similar_to_given(context.lemma, considered_synonyms)

    # This function is needed because word2vec doesn't provide most_similar_to_given
    # function with vector input (only for word input like above)
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
                raise Exception('Metrics is not yet supported')

            if cosine_similarity > max_cosine_similarity:
                nearest_synonym = considered_synonyms[i]
                max_cosine_similarity = cosine_similarity

        return nearest_synonym

    # PART 5
    def predict_nearest_with_context(self, context):
        # From Prof. Benajiba's description, we better limit the context to +-5 words.
        # around the target word. Thus, I set window_size equals to 5 for this experiment.
        # I use pad='right' because from my experiment it yields a better result.
        # pad='left' gives 0.121 precision and recall, meanwhile pad='right' gives 0.124.
        cleaned_full_context = get_cleaned_full_context(context, 5, 'right')

        target_vector = self.model.wv[context.lemma]
        # Sum the target word vector with all word vectors in the context
        # to obtain a single sentence vector
        for word in cleaned_full_context:
            # Assumption: We can ignore context words
            # that are not in the word2vec vocabulary
            if word in self.model.wv:
                target_vector = np.add(target_vector, self.model.wv[word])

        possible_synonyms = list(get_candidates(context.lemma, context.pos))

        # Assumption: We can ignore synonym candidates
        # that are not in the word2vec vocabulary
        considered_synonyms = []
        for synonym in possible_synonyms:
            if synonym in self.model.wv:
                considered_synonyms.append(synonym)

        # Because the inputs now are vectors, we can't use the
        # most_similar_to_given function like the previous one
        return self.get_nearest_synonym(target_vector, considered_synonyms)

    # PART 6
    def predict_best(self, context):
        cleaned_full_context = get_cleaned_full_context(context, 7, 'left', 'after')

        target_vector = self.model.wv[context.lemma]
        for word in cleaned_full_context:
            if word in self.model.wv:
                target_vector = np.add(target_vector, self.model.wv[word])

        possible_synonyms = list(get_best_predictor_candidates(context.lemma, context.pos))

        considered_synonyms = []
        for synonym in possible_synonyms:
            if synonym in self.model.wv:
                considered_synonyms.append(synonym)

        return self.get_nearest_synonym(target_vector, considered_synonyms)

if __name__=="__main__":
    # At submission time, this program should run your best predictor (part 6).

    # PART 1
    # print(get_candidates('slow', 'a'))
    # {'sluggish', 'obtuse', 'dim', 'tedious', 'dumb', 'irksome', 'ho-hum', 'dull', 'dense', 'wearisome', 'deadening', 'tiresome', 'boring'}

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        # prediction = smurf_predictor(context)

        # PART 2 - WordNet Frequency Baseline -> 0.98 precision and recall
        prediction = wn_frequency_predictor(context)

        # PART 3 - Simple Lesk Algorithm -> 0.89 precision and recall
        # prediction = wn_simple_lesk_predictor(context)

        # PART 4 - Most Similar Synonym -> 0.115 precision and recall
        # prediction = predictor.predict_nearest(context)

        # PART 5 - Context and Word Embeddings -> 0.124 precision and recall
        # prediction = predictor.predict_nearest_with_context(context)

        # PART 6 - Best Predictor
        # prediction = predictor.predict_best(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
