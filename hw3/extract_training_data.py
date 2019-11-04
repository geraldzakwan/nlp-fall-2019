from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
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

    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2
        word_indices = []

        # Take three top words from stack
        for i in range(1, 4):
            # Check if we need to pad, i.e. use <NULL>
            if len(state.stack) < i:
                # Append index of '<NULL>'
                word_indices.append(self.word_vocab['<NULL>'])
            else:
                # Get the i-th top word from the stack
                word_idx_from_stack = state.stack[(-1)*i]

                # Get the actual word from the sentence
                word = words[word_idx_from_stack]

                # Check special case for root
                if word == None:
                    # Append index of '<ROOT>'
                    word_indices.append(self.word_vocab['<ROOT>'])
                else:
                    # Check special case for <CD> and <NNP>
                    if pos[word_idx_from_stack] == 'CD':
                        # Append index of 'CD'
                        word_indices.append(self.word_vocab['<CD>'])
                    elif pos[word_idx_from_stack] == 'NNP':
                        # Append index of 'NNP'
                        word_indices.append(self.word_vocab['<NNP>'])
                    else:
                        # Check if it exists in vocab:
                        if word in self.word_vocab:
                            # Get the word index in vocab
                            word_idx_from_vocab = self.word_vocab[word]

                            # Append to return array
                            word_indices.append(word_idx_from_vocab)
                        else:
                            # Append index of 'UNK'
                            word_indices.append(self.word_vocab['<UNK>'])

        # Take next three words from buffer
        for i in range(1, 4):
            # Check if we need to pad, i.e. use <NULL>
            if len(state.buffer) < i:
                # Append index of '<NULL>'
                word_indices.append(self.word_vocab['<NULL>'])
            else:
                # Get the i-th top word from the stack
                word_idx_from_buffer = state.buffer[(-1)*i]

                # Get the actual word from the sentence
                word = words[word_idx_from_buffer]

                # Check special case for root
                if word == None:
                    # Append index of '<ROOT>'
                    word_indices.append(self.word_vocab['<ROOT>'])
                else:
                    # Check special case for <CD> and <NNP>
                    if pos[word_idx_from_buffer] == 'CD':
                        # Append index of 'CD'
                        word_indices.append(self.word_vocab['<CD>'])
                    elif pos[word_idx_from_buffer] == 'NNP':
                        # Append index of 'NNP'
                        word_indices.append(self.word_vocab['<NNP>'])
                    else:
                        # Check if it exists in vocab:
                        if word in self.word_vocab:
                            # Get the word index in vocab
                            word_idx_from_vocab = self.word_vocab[word]

                            # Append to return array
                            word_indices.append(word_idx_from_vocab)
                        else:
                            # Append index of 'UNK'
                            word_indices.append(self.word_vocab['<UNK>'])

        return np.array(word_indices, dtype='int')

    def get_output_representation(self, output_pair):
        # TODO: Write this method for Part 2

        # Use the already computed output labels to get unique index
        # for each transition, 0-90 inclusive
        transition_pair_index = self.output_labels[output_pair]

        # Build the one hot encoding vector, fill the index with 1, rest are 0
        one_hot_encoded_transition = np.zeros(len(self.output_labels), dtype='int')
        one_hot_encoded_transition[transition_pair_index] = 1

        return one_hot_encoded_transition
        # return keras.utils.to_categorical(1, num_classes=len(self.output_labels), dtype='int')


def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0
    for dtree in conll_reader(in_file):
        words = dtree.words()
        pos = dtree.pos()

        # print('WORDS')
        # print(words)
        #
        # print('POS')
        # print(pos)

        for state, output_pair in get_training_instances(dtree):
            # print('STATE')
            # print(state)
            # print('INPUT REPRESENTATION')
            # print(extractor.get_input_representation(words, pos, state))
            # print('---------------------------------------------------')
            inputs.append(extractor.get_input_representation(words, pos, state))

            # print('OUTPUT_PAIR')
            # print(output_pair)
            # print(extractor.get_output_representation(output_pair))
            # print('---------------------------------------------------')
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
