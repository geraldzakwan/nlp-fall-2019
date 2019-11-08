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

    # Explanation for this function is provided in the README.txt
    def is_action_permitted(self, action, state):
        # # Shift can't be done if:
        # if action[0] == 'shift':
        #     # Buffer only contains one word and stack is not empty
        #     if len(state.buffer) == 1 and len(state.stack) > 0:
        #         print('MASUK 2')
        #         return False
        #
        # # Left arc can't be done if:
        # elif action[0] == 'left_arc':
        #     # 1. Stack is empty
        #     if len(state.stack) == 0:
        #         print('MASUK 3')
        #         return False
        #
        #     # 2. Root is the target, remember left arc flows from buffer to stack
        #     if state.stack[-1] == self.extractor.word_vocab['<ROOT>']:
        #         print('MASUK 4')
        #         return False
        #
        # # Right arc can't be done if the stack is empty
        # elif action[0] == 'right_arc':
        #     if len(state.stack) == 0:
        #         print('MASUK 5')
        #         return False
        #
        # # Anything else is permitted
        # return True

        if action[0] == 'shift':
            if len(state.buffer) > 1:
                return True
            elif len(state.buffer) == 1 and len(state.stack) == 0:
                return True
            else:
                return False

        elif action[0] == 'left_arc':
            if len(state.stack) > 0:
                # This is pure stupidity, take me almost 2 hours to realize
                # if state.stack[-1] != self.extractor.word_vocab['<ROOT>']:
                if state.stack[-1] != 0:
                    return True
            else:
                return False

        elif action[0] == 'right_arc':
            if len(state.stack) > 0:
                return True
            else:
                return False

    # Explanation for this function is provided in the README.txt
    def sort_output(self, softmax_output):
        # Create list of tuple (action, probability)
        # Use self.output_labels list (which is kindly provided)
        # to map index in the softmax_output to the action tuple
        action_prob_tuples = []
        for i in range(0, len(softmax_output)):
            action_prob_tuples.append((self.output_labels[i], softmax_output[i]))

        # IMPORTANT: Sort by probability descendingly
        # Careful to not sort ascendingly, because the accuracy will be very very low
        # as we always predict the most unprobable one
        return sorted(action_prob_tuples, key=lambda x: x[1], reverse=True)

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)

        # print(words)
        # print(pos)
        # print(state)
        # sys.exit()
        it = 0

        while state.buffer:
            # TODO: Write the body of this loop for part 4

            # Use extractor object to get input representation
            input = self.extractor.get_input_representation(words, pos, state)

            # IMPORTANT: Reshape to (1, len(input)) or (1,-1)
            # Otherwise, an error will be yielded
            # 1, in this case, is the batch_size (we feed the input one by one)
            # input.resize(1, len(input))

            # Get the prediction from the neural net
            softmax_output = self.model.predict(input.reshape(1, -1))

            # IMPORTANT: softmax_output is a 2d array of shape (1, 91).
            # Thus, to get list of probabilities, take its first element
            # Sort the actions using subfunction
            sorted_actions = self.sort_output(softmax_output[0])

            # Loop over actions, starting with the one with highest probability
            for i in range(0, len(sorted_actions)):
                action = sorted_actions[i][0]

                # Execute the action only if it's valid given the current state
                # Otherwise, try the next possible action
                # Check using subfunction
                if self.is_action_permitted(action, state):
                    # if i > -1  and i < 10:
                    #     print('i = ' + str(i))
                    # Execute the action accordingly
                    # print(action[0])
                    # print(it)
                    if action[0] == 'shift':
                        state.shift()
                    # For left and right arc, supply the relationship type
                    elif action[0] == 'left_arc':
                        state.left_arc(action[1])
                    elif action[0] == 'right_arc':
                        state.right_arc(action[1])

                    # IMPORTANT: Don't forget to break the loop so that
                    # only one action is executed. The next action
                    # will be determined again by the neural net
                    break

            it = it + 1
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
