# Copyright 2024 Carnegie Mellon University

"""
A player using a win stay, lose shift strategy. After a loss the next move is an upgrade
of the previous one with probability upgrade, and otherwise a downgrade. The initial move
is chosen randomly. Note that with the default upgrade value of 0.5 the shift after a
is simply to a uniformly random choice between the other two moves.
"""

import random
from rps_game import RPSPlayer, MOVES

class WinStayLoseShiftPlayer(RPSPlayer):

    def __init__(self, upgrade=0.5):
        super().__init__()
        if upgrade < 0 or upgrade > 1:
            raise RuntimeError(f"the upgrade parameter ({updagrede})should be a probability, non-negative but less than one")
        self._upgrade = upgrade
        self.reset()

    def reset(self):
        self._last_move = None
        self._last_outcome = None

    def move(self):
        if self._last_outcome == "win":
            mv = self._last_move
        elif not self._last_outcome:
            mv = random.choice(MOVES)
        elif random.random() < self._upgrade:
            # upgrade
            mv = MOVES[(MOVES.index(self._last_move) + 1) % 3]
        else:
            # downgrade
            mv = MOVES[(MOVES.index(self._last_move) - 1) % 3]
        self._last_move = mv
        return mv

    def result(self, opponent_move, outcome, wins, ties, losses):
        self._last_outcome = outcome
