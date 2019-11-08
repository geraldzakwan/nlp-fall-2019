from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
# Keras is not used for this module
# import keras
import numpy as np

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)


def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label)
        elif rel == "right_arc":
            state.right_arc(label)

    return state.deps


class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None
    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_structure):
    deprels = dep_structure.deprels

    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1

    seq = []
    while state.buffer:
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


class FeatureExtractor(object):

    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)
        self.pos_vocab = self.read_vocab(pos_vocab_file)
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))

        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file:
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab

    def get_input_representation(self, words, pos, state, top_words=3):
        # TODO: Write this method for Part 2

        # This will be the returned array (size: 2*3)
        # The first three element corresponds to stack elements and
        # the last three corresponds to buffer elements
        word_indices = []

        # Take the top three words from stack and append them as indexes to word_indices
        for i in range(1, top_words+1):
            # Check if we need to pad, i.e. when stack has less than three words
            # Use '<NULL>' for padding.
            if len(state.stack) < i:
                # Append the index of '<NULL>'
                word_indices.append(self.word_vocab['<NULL>'])
            else:
                # Get the i-th top word index from the stack
                word_idx_from_stack = state.stack[(-1)*i]

                # Get the actual word from the sentence
                word = words[word_idx_from_stack]

                # None is used to represent '<ROOT>'
                if word == None:
                    # Append the index of '<ROOT>'
                    word_indices.append(self.word_vocab['<ROOT>'])
                else:
                    # Check special case for words with POS tag: <CD> or <NNP>
                    # In this case, append the index of the POS tag and not the word
                    if pos[word_idx_from_stack] == 'CD':
                        # Append the index of 'CD'
                        word_indices.append(self.word_vocab['<CD>'])
                    elif pos[word_idx_from_stack] == 'NNP':
                        # Append the index of 'NNP'
                        word_indices.append(self.word_vocab['<NNP>'])
                    else:
                        # IMPORTANT: If the word is not lowered, words like 'AND', 'WILL'
                        # and any first word starting with capital letter e.g. 'The'
                        # will be treated as UNK.
                        word = word.lower()

                        # This is the normal/typical case where we actually process a word
                        # Check if the word exists in the vocabulary
                        if word in self.word_vocab:
                            # If found, get the word index in the vocab
                            word_idx_from_vocab = self.word_vocab[word]

                            # Append the index of the word
                            word_indices.append(word_idx_from_vocab)
                        else:
                            # If not found, append the index of '<UNK>' instead
                            word_indices.append(self.word_vocab['<UNK>'])

        # Take the next three words from buffer and append them as indexes to word_indices
        # All the steps are the same as above so I'm not commenting the codes below
        for i in range(1, top_words+1):
            if len(state.buffer) < i:
                word_indices.append(self.word_vocab['<NULL>'])
            else:
                word_idx_from_buffer = state.buffer[(-1)*i]

                word = words[word_idx_from_buffer]

                if word == None:
                    word_indices.append(self.word_vocab['<ROOT>'])
                else:
                    if pos[word_idx_from_buffer] == 'CD':
                        word_indices.append(self.word_vocab['<CD>'])
                    elif pos[word_idx_from_buffer] == 'NNP':
                        word_indices.append(self.word_vocab['<NNP>'])
                    else:
                        word = word.lower()

                        if word in self.word_vocab:
                            word_idx_from_vocab = self.word_vocab[word]

                            word_indices.append(word_idx_from_vocab)
                        else:
                            word_indices.append(self.word_vocab['<UNK>'])

        # print(words)
        # print(pos)
        # print(state)
        # print(word_indices)
        # Finally, return the list of word indexes
        # Convert the list into a numpy array with type integer
        return np.array(word_indices, dtype='float32')

    def get_output_representation(self, output_pair):
        # TODO: Write this method for Part 2

        # Use the output_labels dictionary that is kindly provided
        # to get a unique index for a transition tuple (output_pair).
        # Index ranges inclusively from 0 to 90.
        transition_pair_index = self.output_labels[output_pair]

        # Create a one hot encoding vector as a numpy array with type integer
        # of shape (91). Fill all the elements with zero.
        one_hot_encoded_transition = np.zeros(len(self.output_labels), dtype='float32')

        # Change the value to 1 only for the input transition index
        one_hot_encoded_transition[transition_pair_index] = 1

        return one_hot_encoded_transition

def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0
    for dtree in conll_reader(in_file):
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)


    with open(sys.argv[1],'r') as in_file:

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)
