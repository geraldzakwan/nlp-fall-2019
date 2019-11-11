import numpy as np

animal = np.array([2.0, 3.0, 0.0, 3.0, 0.0, 3.0], dtype='float32')

word_vector = {}
word_vector['dog'] = np.array([0.0, 4.0, 0.0, 4.0, 2.0, 2.0], dtype='float32')
word_vector['cat'] = np.array([4.0, 0.0, 0.0, 3.0, 3.0, 10.0], dtype='float32')
word_vector['computer'] = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 5.0], dtype='float32')
word_vector['run'] = np.array([4.0, 3.0, 5.0, 0.0, 3.0, 4.0], dtype='float32')
word_vector['mouse'] = np.array([2.0, 10.0, 5.0, 4.0, 3.0, 0.0], dtype='float32')

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    # print('Dot: ' + str(np.dot(a, b)))
    # print('L2: ' + str(np.linalg.norm(b)))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    print('Euclidean distance')
    for word in word_vector:
        print(word, str(euclidean_distance(animal, word_vector[word])))
        print()

    print('------------------')

    print('L2 Norm animal:')
    print(np.linalg.norm(animal))
    print()

    print('Cosine similarity')
    for word in word_vector:
        print(word)
        print(str(cosine_similarity(animal, word_vector[word])))
        print()
