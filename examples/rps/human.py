# Copyright 2024 Carnegie Mellon University

"""
An RPSPlayer subclass which simply solicits moves from a human player using the terminal,
and also prints to the terminal the results of each round of play.
"""

from rps_game import RPSPlayer, MOVES
import sys

def read_move():
    while True:
        print("Enter your next move: r(ock), p(aper) or s(cissors): ", end="", flush=True)
        s = sys.stdin.readline().strip()
        if s:
            for m in MOVES:
                if m.startswith(s):
                    return m


class HumanPlayer(RPSPlayer):

    def move(self):
        self._last_move = read_move()
        return self._last_move

    def result(self, opponent_move, outcome, wins, ties, losses):
        print(f"You played {self._last_move}, your opponent played {opponent_move}, you {outcome} "
              f"(so far you have won {wins}, tied {ties} and lost {losses})")
