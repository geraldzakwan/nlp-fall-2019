# Identity

Name: Geraldi Dzakwan
UNI: gd2551

# Some assumptions/design decisions that I made:

- All words are lowered, e.g. words like 'The', 'AND' and 'WILL' will be
  converted into 'the', 'and' and 'will'. This is needed because vocabulary
  contains only lowered words. Thus, we need to lower the words so that
  they are not treated as unknown ('<UNK>'). If not handled, this issue can
  decrease the accuracy. In my experiment, it is around 4-5% lower than
  it should be if we don't lower the words.

- I create two subfunctions to help working on decoding (parsing) the sentence
  in the Parser class. They are:

  1. is_action_permitted(self, action, state)
     This function basically returns True if an action, e.g. ('left_arc', 'aux'),
     don't violate some constraints given the current state. It returns False
     otherwise. The constraints are well commented on the function. They are
     all the constraints stated in the problem description.

  2. sort_output(self, softmax_output)
     This function basically accepts the softmax output from the neural network.
     Softmax output is a list containing the probability where probability at
     index i refers to the probability of an action from output_label[i], which
     is kindly provided by Prof. Benajiba as self.output_labels.
     The function then creates a list of tuple (action, probability) and return
     the descendingly sorted array based on the probability value.
