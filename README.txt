Some assumptions/design decisions that I made for this assignment:

1. 'START' and 'STOP' tokens are listed in the vocabulary and count towards the total words.
This is as clarified by Prof. Benajiba in the email and in https://piazza.com/class/k09vvrvx2l846o?cid=72.
This design have some effects:

  a. raw_unigram_probability becomes smaller for every word since the total words increase

  b. perplexity becomes smaller, 2^(-l/M) will go smaller as M (the total words) get higher.

    - Perplexity for brown_test
    If 'START' and 'STOP' are counted, my result is 173.51
    Without counting them, the perplexity is 300.18

    - Perplexity for brown_train
    If 'START' and 'STOP' are counted, my result is 14.49
    Without counting them, the perplexity is 18.03

    Those values seem fine according to https://piazza.com/class/k09vvrvx2l846o?cid=81

2. P('START', 'START', anything,) is equal to P('START', anything,) as both denote the same meaning:
probability of anything is the first word. Reference: https://piazza.com/class/k09vvrvx2l846o?cid=32

3. Out of vocabulary words handling:

   a. For raw_uni/bi/trigram_probability functions, I don't replace OOV words with 'UNK' because it is
   meant to be 'raw'. I just return zero if either the numerator or denominator is zero (or both).
   Reference: https://piazza.com/class/k09vvrvx2l846o?cid=18, https://piazza.com/class/k09vvrvx2l846o?cid=48

   b. For smoothed_trigram_probability, I also don't replace OOV words with 'UNK'. The reason is that
   linear interpolation should only account for unseen trigram and not unseen unigram. If an unseen word appears
   as the last element of the trigram, e.g. ('the', 'jury', unseen), the smoothed_trigram_probability will be zero.

   c. For sentence_logprob, I don't replace OOV words with 'UNK' as well. If there is at least one trigram
   that has smoothed_trigram_probability equals zero, the function will return float('-inf').

   d. However, when computing perplexity, OOV words will all be replaced by 'UNK'. This is done by
   corpus_reader function. Hence, all the cases above won't happen. That is, smoothed_trigram_probability
   will never be equal to zero thus making sentence_logprob will never return float('-inf').

4. I implement the generate_sentence (optional) function. I use numpy library, particularly numpy.random.multinomial,
to conduct sampling experiment as suggested by one of the TA here: https://piazza.com/class/k09vvrvx2l846o?cid=35.
The rest of the logic is explained in the comments inside the function.
