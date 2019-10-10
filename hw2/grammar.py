"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 2 - Parsing with Context Free Grammars
Yassine Benajiba
"""

import sys
import math
from collections import defaultdict

class Pcfg(object):
    """
    Represent a probabilistic context free grammar.
    """

    def __init__(self, grammar_file):
        # Same as lhs, just the backpointer, e.g.:
        # rhs_to_rules[('ABOUT', 'NP')] = [('PP', ('ABOUT', 'NP'), probability), other terminals that have this lhs]
        self.rhs_to_rules = defaultdict(list)

        # Basically a list of production rules, e.g.:
        # lhs_to_rules['PP'] = [('PP', ('ABOUT', 'NP'), probability), other tuple of production rules]
        self.lhs_to_rules = defaultdict(list)

        self.startsymbol = None
        self.read_rules(grammar_file)

    def read_rules(self,grammar_file):
        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()

    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False.
        """
        for nonterminal in self.lhs_to_rules:
            # First, check if nonterminal follows the format, i.e. all uppercase
            if not Pcfg.check_nonterminal_format(nonterminal):
                print('The first nonterminal to not follow the format: ' + nonterminal)
                return False

            # Second, check the validity of the production rule, e.g. either has two nonterminals or a terminal
            if not Pcfg.check_production_rules(self.lhs_to_rules[nonterminal]):
                print('The first nonterminal whose production rule doesn\'t follow the format: ' + nonterminal)
                return False

            # Third, check if all production rules sum up to 1
            if not Pcfg.check_production_rules_total_probability(self.lhs_to_rules[nonterminal]):
                print('The first nonterminal whose production rule probabilities don\'t sum up closely to 1: ' + nonterminal)
                return False

        return True

    @staticmethod
    def check_nonterminal_format(nonterminal):
        return nonterminal.isupper()

    @staticmethod
    def check_terminal_format(terminal):
        return True

    @staticmethod
    def check_production_rules(production_triplets):
        for production_triplet in production_triplets:
            production_rule = production_triplet[1]

            # The production rule needs to be either two nonterminals or a terminal
            if len(production_rule) == 2:
                if not (Pcfg.check_nonterminal_format(production_rule[0]) and Pcfg.check_nonterminal_format(production_rule[1])):
                    return False
            elif len(production_rule) == 1:
                if not Pcfg.check_terminal_format(production_rule[0]):
                    return False
            else:
                # Empty production rule is not valid
                return False

        return True

    @staticmethod
    def check_production_rules_total_probability(production_triplets):
        total_probability = float(0.0)
        for production_triplet in production_triplets:
            total_probability = total_probability + production_triplet[2]

        if total_probability != float(1.0):
            print('Example of nonterminal whose production rule probabilities don\'t sum up exactly to 1 (but close): ' + str(production_triplets[0][0]))
            print(total_probability)

        # Two floats are considered the same if their difference is less than 0.0001
        # Reference: https://piazza.com/class/k09vvrvx2l846o?cid=108
        return math.isclose(total_probability, float(1.0), abs_tol=0.0001)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        grammar_filepath = sys.argv[1]
    else:
        grammar_filepath = 'atis3.pcfg'

    with open(grammar_filepath,'r') as grammar_file:
        grammar = Pcfg(grammar_file)

    # print(Number of nonterminalslen(grammar.lhs_to_rules))
    # print('-----------------------------------')

    print(grammar.verify_grammar())
