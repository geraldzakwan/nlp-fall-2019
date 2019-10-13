# Get started:

- No need to install any library (all are built-in libraries in Python 3)
- Please put the 'atis3.pcfg' and 'atis3_test.ptb' data file in the same directory as the Python files

# Some assumptions/design decisions that I made for this coding assignment:

- Part 1 (Sum of probabilities should be equal to 1)

  1. Two floats are considered the same if their difference is less than 0.0001.
  Reference: https://piazza.com/class/k09vvrvx2l846o?cid=108

  2. Other than checking if nonterminal symbols are upper-case and the probabilities
  for the same LHS symbol sum to 1, I also check that for each production rule:
    a. The RHS should be either two nonterminals or a terminal.
    b. A nonterminal is valid if it appears in some LHS.
    c. A terminal is valid if it doesn't appear in any LHS.
    Reference: https://piazza.com/class/k09vvrvx2l846o?cid=110

  Valid examples:
  - ('NP', ('NP', 'VP', ), 0.006)
  - ('NP', ('flight', ), 0.004)

  Invalid examples:
  - ('NP', ('VP', ), 0.006)
  - ('NP', ('flight', 'to', ), 0.004)
  - ('NP', (), 0.0) -> empty production rule
