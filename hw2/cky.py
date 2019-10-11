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
        return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        # Init table with
        table = {}
        for i in range(0, len(tokens)):
            for j in range(i+1, len(tokens)):
                table[(i, j,)] = {}

        probs = None

        for i in range(0, len(tokens)-1):
            rules = self.grammar.rhs_to_rules[(tokens[i],)]

            for rule in rules:
                nonterminal = rule[0]
                # What is the backpointer of length 1, is it just the terminal string?
                table[(i, i+1,)][nonterminal] = tokens[i]

        for length in range(2, len(tokens)):
        # for length in range(2, 3):
            # print(length)

            for i in range(0, len(tokens)-length):
                # print('------')
                j = i + length

                # print(i, j)
                for k in range(i+1, j):
                    # print(i,k)
                    # print(k,j)

                    # Get pair of nonterminals from chart of length size - 1
                    some_nonterminal_1 = table[(i, k ,)]
                    # print('NONTERMINAL1:')
                    # print(some_nonterminal_1)
                    some_nonterminal_2 = table[(k, j ,)]
                    # print('NONTERMINAL2:')
                    # print(some_nonterminal_2)

                    # Remember we may have more than one nonterminal in a chart cell
                    # Let's iterate both and make a combination of pairs
                    for nonterminal_1 in some_nonterminal_1:
                        for nonterminal_2 in some_nonterminal_2:
                            nonterminal_pair = (nonterminal_1, nonterminal_2, )

                            # Check if that pair can result in new nonterminal
                            if nonterminal_pair in self.grammar.rhs_to_rules:
                                rhs_to_rules = self.grammar.rhs_to_rules[nonterminal_pair]
                                # print('Yes, we find a combination')
                                # print(nonterminal_pair)
                                # print(rhs_to_rules)

                                # Add every new_nonterminal to chart i,j
                                # Example of backpointer value:
                                # Let say, we want to define backpointer for table[(0, 3,)]['NP']
                                # It can be something like (('NP',0,2), ('PP',2,3),)
                                for rule in rhs_to_rules:
                                    new_nonterminal = rule[0]
                                    first_backpointer = (nonterminal_pair[0], i, k, )
                                    second_backpointer = (nonterminal_pair[1], k, j, )
                                    table[(i, j,)][new_nonterminal] = (first_backpointer, second_backpointer, )

                                    # print((first_backpointer, second_backpointer, ))
                            else:
                                # print('This nonterminal_pair is trash')
                                # print(nonterminal_pair)
                                pass


        return table, probs


def get_tree(chart, i,j,nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    return None


if __name__ == "__main__":

    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        #print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        print(table)
        # assert check_probs_format(probs)
