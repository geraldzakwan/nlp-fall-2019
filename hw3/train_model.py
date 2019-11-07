from extract_training_data import FeatureExtractor
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense

def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3

    # Build a keras Sequential model
    model = Sequential()

    # word_types -> Number of words (including special cases) in the dictionary
    # See words.vocab. 15153 is the word_types value for basic configuration
    # 32 -> Embedding dimension default value, can be customized
    # 6 -> Input vector size, i.e. the word index from top 3 stack and buffer, can be customized
    model.add(Embedding(word_types, 32, input_length=6))

    # Flatten layer to flatten the output, i.e. make it a 1d array
    model.add(Flatten())

    # Two dense layers with 100 and 10 units respectively for basic configuration
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))

    # Add a dense output layer with 91 (output_labels) units
    model.add(Dense(output_labels, activation='softmax'))

    # Compile the model with an optimizer and a loss function
    # Default optimizer is Adam with learning rate = 0.01
    model.compile(keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy')

    return model


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
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")

    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)

    model.save(sys.argv[3])
