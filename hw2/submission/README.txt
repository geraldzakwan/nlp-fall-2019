# Some assumptions/design decisions I made for this coding assignment:

- Part 1

  Other than checking if nonterminal symbols are upper-case and
  the probabilities for the same lhs symbol sum to 1.0, I also check:

  1. For each production rule triplet, the rhs should be
  two nonterminals or a terminal.

  Valid examples:
  - ('NP', ('NP', 'VP'), 0.006)
  - ('NP', ('flight'), 0.004)

  Invalid examples:
  - ('NP', ('VP'), 0.006)
  - ('NP', ('flight', 'to'), 0.004)
  - ('NP', (), 0.0)

  2. For nonterminal,

- Part 3

  # Example of a backpointer value:
  # Let say we want to define backpointer for parse_table[(0, 3,)]['NP']
  # It is something like (('NP', 0, 2), ('PP', 2, 3),)
