Some assumptions / design decisions that I made for this assignment:

KEY DISCUSSIONS FOR OOV/UNK:
https://piazza.com/class/k09vvrvx2l846o?cid=48
Basically: for raw probability, fuck UNK -> simply return 0
We then use the smoothed_trigram_probability to "smoothen" it

- For these functions: (a, b, c, d)
  1. For param=x, every OOV word will always be replace by 'UNK'. Thus, when counting the probability,
  there wouldn't be a case where the denominator is zero. This is the default for my implementation.
  2. For param=y, every OOV word will stay as it is. Thus, when counting the probability,
  there would be a case where the denominator is zero.

- For raw_unigram_probability, the total number of words:
  1. Exclude start and stop, why? Reference: https://piazza.com/class/k09vvrvx2l846o?cid=30 (Evan Lander)
  https://piazza.com/class/k09vvrvx2l846o?cid=44 (Evan Lander)

- Special case, probability of ('START', 'START', 'the',): https://piazza.com/class/k09vvrvx2l846o?cid=32

- For perplexity, the total number of words:
  1. Exclude start and stop, why? The default. Reference: https://piazza.com/class/k09vvrvx2l846o?cid=36
  https://piazza.com/class/k09vvrvx2l846o?cid=43
  2. Result when exclude
  3. Result when include?
