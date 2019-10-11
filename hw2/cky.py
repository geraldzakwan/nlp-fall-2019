"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 2 - Parsing with Context Free Grammars
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar.
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        # Init parse table
        parse_table = {}
        for i in range(0, len(tokens)+1):
            for j in range(i+1, len(tokens)+1):
                parse_table[(i, j,)] = {}

        # Fill parse table of span 1 (rules involving only terminal)
        for i in range(0, len(tokens)):
            rules = self.grammar.rhs_to_rules[(tokens[i],)]

            for rule in rules:
                nonterminal = rule[0]
                # No need backpointer in this case, set parse_table value to empty tuple is sufficient
                parse_table[(i, i+1,)][nonterminal] = ()

        # Fill parse table of span of length 2 and above (rules involving only nonterminal)
        for length in range(2, len(tokens)+1):
            # Iterate over all possibilites of span of current length
            for i in range(0, len(tokens)-length+1):
                j = i + length

                # Get combinations of every two spans that cover current span
                for k in range(i+1, j):
                    # Retrieve the nonterminals that they contain
                    some_nonterminal_1 = parse_table[(i, k ,)]
                    some_nonterminal_2 = parse_table[(k, j ,)]

                    # Remember we may have more than one nonterminal in these two spans
                    # Let's iterate both spans and make a combination of pairs
                    for nonterminal_1 in some_nonterminal_1:
                        for nonterminal_2 in some_nonterminal_2:
                            nonterminal_pair = (nonterminal_1, nonterminal_2, )

                            # Check if the pair can result in new nonterminal
                            if nonterminal_pair in self.grammar.rhs_to_rules:
                                rhs_to_rules = self.grammar.rhs_to_rules[nonterminal_pair]

                                # Add every new nonterminal to chart current span i, j
                                # No need backpointer in this case, set parse_table value to empty tuple is sufficient
                                for rule in rhs_to_rules:
                                    new_nonterminal = rule[0]
                                    parse_table[(i, j,)][new_nonterminal] = ()

        # Check if there is nonterminal 'TOP' in the whole span
        # It indicates whether or not a sentence belongs to the grammar/language
        return 'TOP' in parse_table[(0, len(tokens), )]

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3

        # Init parse table and probs table
        parse_table = {}
        probs_table = {}

        for i in range(0, len(tokens)+1):
            for j in range(i+1, len(tokens)+1):
                parse_table[(i, j,)] = {}
                probs_table[(i, j,)] = {}

        # Fill parse table of span of length 2 and above (rules involving only nonterminal)
        for i in range(0, len(tokens)):
            rules = self.grammar.rhs_to_rules[(tokens[i],)]

            # Question to TA: Do we need to pick the most probable nonterminal in span of length 1?
            # In this case, I keep all if more than one apply
            for rule in rules:
                nonterminal = rule[0]
                prob = math.log2(rule[2])

                parse_table[(i, i+1,)][nonterminal] = tokens[i]
                probs_table[(i, i+1,)][nonterminal] = prob

        # Fill parse table of span of length 2 and above (rules involving only nonterminal)
        for length in range(2, len(tokens)+1):
            # Iterate over all possibilites of span of current length
            for i in range(0, len(tokens)-length+1):
                j = i + length

                # Get combinations of every two spans that cover current span
                for k in range(i+1, j):
                    # Retrieve the nonterminals that they contain
                    some_nonterminal_1 = parse_table[(i, k ,)]
                    some_nonterminal_2 = parse_table[(k, j ,)]

                    # Remember we may have more than one nonterminal in these two spans
                    # Let's iterate both spans and make a combination of pairs
                    for nonterminal_1 in some_nonterminal_1:
                        for nonterminal_2 in some_nonterminal_2:
                            nonterminal_pair = (nonterminal_1, nonterminal_2, )

                            # Check if the pair can result in new nonterminal
                            if nonterminal_pair in self.grammar.rhs_to_rules:
                                rhs_to_rules = self.grammar.rhs_to_rules[nonterminal_pair]

                                # Add every new nonterminal to current span i, j
                                # of the parse table along with its backpointer
                                for rule in rhs_to_rules:
                                    new_nonterminal = rule[0]

                                    # Log prob of left child, right child and new nonterminal
                                    prob_subtree_1 = probs_table[(i, k, )][nonterminal_1] # This is already in the form of log
                                    prob_subtree_2 = probs_table[(k, j, )][nonterminal_2] # This is already in the form of log
                                    prod_rule_prob = math.log2(rule[2])
                                    new_prob = prob_subtree_1 + prob_subtree_2 + prod_rule_prob

                                    # Example of a backpointer value:
                                    # Let say we want to define backpointer for parse_table[(0, 3,)]['NP']
                                    # It is something like (('NP', 0, 2), ('PP', 2, 3),)
                                    first_backpointer = (nonterminal_pair[0], i, k, )
                                    second_backpointer = (nonterminal_pair[1], k, j, )

                                    # Check if we already assign a backpointer previously
                                    if new_nonterminal not in parse_table[(i, j, )]:
                                        # If no backpointer is assigned yet, simply assign it
                                        parse_table[(i, j, )][new_nonterminal] = (first_backpointer, second_backpointer, )

                                        # Assign the probability as well
                                        probs_table[(i, j, )][new_nonterminal] = new_prob
                                    else:
                                        # Check if current prob is bigger than previous prob
                                        if new_prob > probs_table[(i, j, )][new_nonterminal]:
                                            # print('Example of replacing backpointer with another one that has higher prob: ')
                                            # print('Previous backpointer and the probability: ')
                                            # print(parse_table[(i, j, )][new_nonterminal])
                                            # print(probs_table[(i, j, )][new_nonterminal])
                                            # print('New backpointer and the probability: ')
                                            # print(first_backpointer, second_backpointer, )
                                            # print(new_prob)

                                            # Replace prob and backpointer if it is
                                            parse_table[(i, j, )][new_nonterminal] = (first_backpointer, second_backpointer, )

                                            probs_table[(i, j, )][new_nonterminal] = new_prob

        return parse_table, probs_table

def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    # Hint: Recursively traverse the parse chart
    # Base case is when the dict value is just a terminal string
    if isinstance(chart[(i, j, )][nt], str):
        return (nt, chart[(i, j, )][nt], )
    else:
        # First, get the backpointers
        left_backpointer = chart[(i, j, )][nt][0]
        right_backpointer = chart[(i, j, )][nt][1]

        # Then, get their index and nonterminal
        left_nt = left_backpointer[0]
        right_nt = right_backpointer[0]

        left_i = left_backpointer[1]
        right_i = right_backpointer[1]

        left_j = left_backpointer[2]
        right_j = right_backpointer[2]

        # Finally, recurse over left and right child
        return (get_tree(chart, left_i, left_j, left_backpointer), get_tree(chart, right_i, right_j, right_backpointer))

if __name__ == "__main__":

    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)

        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        print(parser.is_in_language(toks))
        print(parser.is_in_language('miami flights cleveland from to .'.split(' ')))

        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)

        print(table[(0, 6, )])
        print(probs[(0, 6, )])
        print('-----------------')

        print(get_tree(table, 0, len(toks), 'TOP'))
