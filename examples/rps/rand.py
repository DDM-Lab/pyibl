# Copyright 2024 Carnegie Mellon University

"""
A player who, by default, chooses moves uniformly at random. If bias is set it should
be a 2-tuple, the probabilities of rock or paper being chosen, and a correspondingly
skewed distribution is then used for picking moves at random.
"""

import random
from rps_game import RPSPlayer, MOVES

class RandomPlayer(RPSPlayer):

    def __init__(self, bias=None):
        super().__init__()
        if bias:
            self._rock, self._paper = bias
            if not (self._rock >= 0 and self._paper >= 0
                    and (self._rock + self._paper) <= 1):
                raise RuntimError(f"bias ({bias}) should be a tuple of the probabilities that rock and paper are chosen")
        else:
            self._rock = None

    def move(self):
        if not self._rock:
            return random.choice(MOVES)
        r = random.random()
        if r <= self._rock:
            return "rock"
        elif r <= self._rock + self._paper:
            return "paper"
        else:
            return "scissors"
