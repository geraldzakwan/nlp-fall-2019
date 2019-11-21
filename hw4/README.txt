Setup:

- Please put the word2vec file 'GoogleNews-vectors-negative300.bin.gz' in the
same directory (same level) as lexsub_main.py

- The libraries (and their versions) that I use:
  1. numpy 1.17.4
  2. nltk 3.4.5
  3. gensim 3.8.1

- To run the best predictor:
  1. python lexsub_main.py lexsub_trial.xml > best.predict
  2. perl score.pl best.predict gold.trial

General design decisions:

- Use the default tokenizer given by Prof. Benajiba. Some tokenizers provided by
NLTK, e.g. word_tokenize, don't seem to improve the result.

- For normalization, I do these:
  1. Remove punctuation listed in Python string.punctuation
  2. Remove numbers using Python isnumeric() function
  3. Lowercase all the words
  4. Remove stop words as suggested, using stopwords.words('english')

- For Part 1 get_candidates(), if a synonym contains multiple words,
it will be separated by spaces, e.g. 'turn around'

- For Part 2 wn_frequency_predictor(), if there is a tie, the synonym that is
"alphabetically biggest" is picked. See comments in the function for more detail.

- For Part 3 wn_simple_lesk_predictor(), synsets that only have the input lemma/target word
as their lexeme are not considered. This is to avoid the case that the returned synonym
will be the same as the input lemma. See comments in the function for more detail.

- Still for Part 3, see resolve_best_synset() function for several cases
that can happen when determining the best sense (e.g. when there is no overlap
or when there are more than one synsets with max overlap). Also, I don't
count duplicate overlaps, i.e. if the same word appears twice in the context
or in the sense definitions/examples, it counts as one.

- For Part 4, I ignore context words that are not in word2vec vocabulary. For
the input lemma/target word, I don't observe any OOV occurs. The metrics
that I use is cosine similarity.

- For Part 5, 5 context words are used as suggested by Prof. Benajiba. I use
2 left context words and 3 right context words and it yields 0.124 precision and recall.
Meanwhile, if 3 left context words and 2 right context words are used,
it gives lower precision and recall -> 0.121.

My best solution yields 0.141 precision and recall, which is nearly 2% improvement from Part 5.
It is implemented in the predict_best() function inside the class Word2VecSubst.
Allow +-5 minutes time (6 minutes at maximum) for the program to run.

The approaches for the best predictor are as below:

- Beside using the synsets that contain the input lemma, I also use the
hypernyms of those synsets to add even more synonym candidates. See function
get_best_predictor_candidates() for more detail. This is the most important
refinement. Including hyponyms on the other hand, lower the precision. Before this,
I use word2vec most_similar function on the input lemma to get more synonym
candidates. But, it runs too slowly (4-5 secs per example) so I disregard it.

- On the previous parts, I normalize the context before slicing it based on window size.
This time, I slice it first before normalizing it so that it may contain less
words than the window size. See function get_best_predictor_candidates() for more detail.
Thus, it is possible to not have a context at all. This is the second most important refinement.
I observe that it's better to not include context rather than include context that
is quite far away from target word.

- Use window_size of 7 with 4 words on the left and 3 words on the right.
This is the best one after experimenting on several window sizes (4, 5, 6, 7, 8, 9, 10).

- Change the checking condition of a synonym candidate in regards to the input lemma.
Previously, it is simply synonym != input_lemma. Now, I also disregard a candidate
in which it is a substring of the input lemma. This helps for some cases such as
when the input lemma is 'examination'. We don't want 'exam' as the substitute as
it is too general and less likely to be the right substitution for different senses.

- For adjectives and adverbs with multiple words, use the separator '-', e.g. well-fit.
But, this doesn't seem to yield any improvements as my predictor never actually
return a synonym with multiple words (they never have the biggest similarity).

Detailed Results for each part:

- PART 2 - WordNet Frequency Baseline
(venv) Kartikos-MBP:hw4 kartiko$ python lexsub_main.py lexsub_trial.xml > wn_frequency.predict
(venv) Kartikos-MBP:hw4 kartiko$ perl score.pl wn_frequency.predict gold.trial
Total = 298, attempted = 298
precision = 0.099, recall = 0.099
Total with mode 206 attempted 206
precision = 0.136, recall = 0.136

- PART 3 - Simple Lesk Algorithm
(venv) Kartikos-MBP:hw4 kartiko$ python lexsub_main.py lexsub_trial.xml > wn_simple_lesk.predict
(venv) Kartikos-MBP:hw4 kartiko$ perl score.pl wn_simple_lesk.predict gold.trial
Total = 298, attempted = 298
precision = 0.089, recall = 0.089
Total with mode 206 attempted 206
precision = 0.131, recall = 0.131

- PART 4 - Most Similar Synonym
(venv) Kartikos-MBP:hw4 kartiko$ python lexsub_main.py lexsub_trial.xml > wn_most_similar.predict
(venv) Kartikos-MBP:hw4 kartiko$ perl score.pl wn_most_similar.predict gold.trial
Total = 298, attempted = 298
precision = 0.115, recall = 0.115
Total with mode 206 attempted 206
precision = 0.170, recall = 0.170

- PART 5 - Context and Word Embeddings
(venv) Kartikos-MBP:hw4 kartiko$ python lexsub_main.py lexsub_trial.xml > w2v_context.predict
(venv) Kartikos-MBP:hw4 kartiko$ perl score.pl w2v_context.predict gold.trial
Total = 298, attempted = 298
precision = 0.124, recall = 0.124
Total with mode 206 attempted 206
precision = 0.189, recall = 0.189

- PART 6 - Best Predictor
(venv) Kartikos-MBP:hw4 kartiko$ python lexsub_main.py lexsub_trial.xml > best.predict
(venv) Kartikos-MBP:hw4 kartiko$ perl score.pl best.predict gold.trial
Total = 298, attempted = 298
precision = 0.141, recall = 0.141
Total with mode 206 attempted 206
precision = 0.223, recall = 0.223
