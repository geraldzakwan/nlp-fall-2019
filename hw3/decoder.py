from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def is_action_permitted(self, action, state):
        if action[0] == 'shift':
            # Can't shift if buffer only contains one word and stack is not empty
            if len(state.buffer) < 2 and len(state.stack) > 0:
                return False

            # Root can't be the target of left arc
            if len(state.stack) > 0:
                if state.stack[-1] == self.extractor.word_vocab['<ROOT>']:
                    return False

        # Can't do left or right arc if the stack is empty
        if action[0] == 'left_arc' or action[0] == 'right_arc':
            if len(state.stack) == 0:
                return False

        return True

    def sort_output(self, softmax_output):
        # Create list of tuple (action, probability)
        action_prob_tuples = []
        for i in range(0, len(softmax_output)):
            action_prob_tuples.append((self.output_labels[i], softmax_output[i]))

        # print(action_prob_tuples)

        # Sort by probability

        # IMPORTANT, SORT DESCENDINGLY WKWKWKWK
        # IF NOT, GET SOMETHING VERY VERY LOW:
        # Micro Avg. Labeled Attachment Score: 2.487500310937539e-05
        # Micro Avg. Unlabeled Attachment Score: 0.25653590706698837
        #
        # Macro Avg. Labeled Attachment Score: 1.802719699252995e-05
        # Macro Avg. Unlabeled Attachment Score: 0.2565859057508941
        return sorted(action_prob_tuples, key=lambda x: x[1], reverse=True)

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)

        while state.buffer:
            # TODO: Write the body of this loop for part 4

            input = self.extractor.get_input_representation(words, pos, state)

            # IMPORTANT: Reshape to (1, 6), 1 is the batch_size
            input.resize(1, len(input))

            # print('INPUT:')
            # print(type(input))
            # print(input.shape)
            # print(input)
            softmax_output = self.model.predict(input)
            # print('OUTPUT:')
            # print(type(softmax_output))
            # print(softmax_output.shape)
            # print(softmax_output)

            # IMPORTANT, since softmax_output is of shape (1, 91),
            # take its first element
            sorted_actions = self.sort_output(softmax_output[0])

            # Loop over actions, starting with the highest prob
            for i in range(0, len(sorted_actions)):
                action = sorted_actions[i][0]

                # Do action only if it's valid,
                # otherwise try next possible action
                if self.is_action_permitted(action, state):
                    if action[0] == 'shift':
                        state.shift()
                    elif action[0] == 'left_arc':
                        state.left_arc(action[1])
                    elif action[0] == 'right_arc':
                        state.right_arc(action[1])
                    break

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
