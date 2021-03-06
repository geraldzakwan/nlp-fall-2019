from extract_training_data import FeatureExtractor
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense

def build_model(word_types, pos_types, outputs, input_length=6, embedding_dim=32, dense_1=100, dense_2=10, learning_rate=0.01):
    # TODO: Write this function for part 3

    # Build a keras Sequential model
    model = Sequential()

    # word_types -> Number of words (including special cases) in the dictionary
    # 15153 is the word_types value for the default configuration
    # embedding_dim -> The word embedding dimension, default value is 32, can be customized
    # input_length -> Input vector size. Default value is 6, i.e. taking 3 words from stack and buffer each, can be customized
    model.add(Embedding(word_types, embedding_dim, input_length=input_length))

    # Flatten layer to flatten the output, i.e. make it a 1D array
    model.add(Flatten())

    # Two dense layers with 100 and 10 units respectively for the default configuration
    model.add(Dense(dense_1, activation='relu'))
    model.add(Dense(dense_2, activation='relu'))

    # Add a dense output layer with 91 (output_labels) units
    model.add(Dense(outputs, activation='softmax'))

    # Compile the model with an optimizer and a loss function
    # Default optimizer is Adam with learning rate = 0.01
    model.compile(keras.optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy')

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

    # To use default model, simply don't supply sys.argv[4]
    # To use modified model, run with 'modified' argument in the very back, i.e.
    # python train_model.py data/input_train_mod.npy data/target_train_mod.npy data/model_mod.h5 modified
    if len(sys.argv) > 4:
        if sys.argv[4] == 'modified':
            # Modified hyperparams
            model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels), input_length=6, embedding_dim=16, dense_1=100, dense_2=10, learning_rate=0.01)
        else:
            # Default model
            model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    else:
        # Default model
        model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))

    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")

    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)

    model.save(sys.argv[3])
