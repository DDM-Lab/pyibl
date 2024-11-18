# Copyright 2024 Carnegie Mellon University

"""
A framework for placing the Rock, Paper, Scissors game. Players instances of subclasses
of RPSPlayer.

Also included is a command line interface to run pairs of players, typically of two
different types, against one another for a given number of rounds and a number of
virtual participant pairs. The players are described as a module name, dot, and a
constructor name, optionally followed by parenthesized arguments to the constructor. For
example,
    python rps_game.py wsls.WinStayLoseShiftPlayer "rand.RandomPlayer(bias=(0.8, 0.1))"
"""

import click
from importlib import import_module
import matplotlib.pyplot as plt
from os import listdir
from os.path import splitext
import pandas as pd
from re import fullmatch
import sys
from tqdm import trange


MOVES = ["rock", "paper", "scissors"]
RESULTS = ["tie", "win", "lose"]


class RPSPlayer:
    """
    Subclass this abstract class to create a kind of player. The move() method must
    be overridden to respond with one of the possible MOVES. If desired, the result()
    method may also be overridden to inform the player of the result of the most recent
    round of play. Similarly the reset() method may be overridden; it is typically
    called between virtual games to reset the virtual participant if that participant
    retains state between rounds.
    """

    def __init__(self):
        self._awaiting_result = False

    def reset(self):
        """If there's anything to do this method must be overridden"""
        pass

    def do_reset(self):
        self.reset()
        self._awaiting_result = False

    def move(self):
        """Must be overriden by a subclass"""
        raise NotImplementedError("The move() method must be overridden")

    def do_move(self):
        if self._awaiting_result:
            raise RuntimeError("Cannot make a move until the previous round has been resolved")
        m = self.move()
        self._awaiting_result = True
        return m

    def result(self, opponent_move, outcome, wins, ties, losses):
        """If there's anything to do this method must be overridden"""
        pass

    def do_result(self, opponent_move, outcome, wins, ties, losses):
        if not self._awaiting_result:
            return
        self.result(opponent_move, outcome, wins, ties, losses)
        self._awaiting_result = False



class RPSGame:
    """
    Plays one or more games between two player objects, each of a given number of rounds.
    The player objects are reset between games. Returns a Pandas DataFrame collecting
    the results of all the rounds of all the games.
    """

    def __init__(self, player1, player2, rounds=1, participants=1):
        self._players = [player1, player2]
        self._rounds = rounds
        self._participants = participants

    def play(self, show_progress=False):
        results = []
        for participant in (trange(1, self._participants + 1) if show_progress
                            else range(1, self._participants + 1)):
            wins = [0, 0]
            for p in self._players:
                p.reset()
            for round in range(1, self._rounds + 1):
                moves = [p.do_move() for p in self._players]
                outcomes = [RESULTS[(MOVES.index(moves[i]) - MOVES.index(moves[(i + 1) % 2])) % 3]
                            for i in range(2)]
                for i in range(2):
                    if outcomes[i] == "win":
                        wins[i] += 1
                for p, om, oc, win, loss in zip(self._players,
                                                reversed(moves),
                                                outcomes,
                                                wins,
                                                reversed(wins)):
                    p.do_result(om, oc, win, round - (win + loss), loss)
                results.append([participant, round,
                                moves[0], moves[1],
                                outcomes[0], outcomes[1],
                                wins[0], wins[1]])
        return pd.DataFrame(results,
                            columns=("participant pair,round,"
                                     "player 1 move,player 2 move,"
                                     "player 1 outcome,player 2 outcome,"
                                     "player 1 total wins,player 2 total wins").split(","))


def plot_wins_losses(df, player_no=1, title=None, file=None):
    if file:
        df.to_csv(file)
    other_player = 1 if player_no==2 else 2
    df["wins"] = df.apply(lambda x: x[f"player {player_no} total wins"] / x["round"], axis=1)
    df["losses"] = df.apply(lambda x: x[f"player {other_player} total wins"] / x["round"], axis=1)
    df["ties"] = 1 - (df["wins"] + df["losses"])
    rounds = max(df["round"])
    xmargin = rounds / 80
    df.groupby("round")[["wins", "ties", "losses"]].mean().plot(figsize=(10, 6),
                                                                color=("green", "gray", "firebrick"),
                                                                ylim=(-0.03, 1.03),
                                                                title=title,
                                                                xlabel="round",
                                                                xlim=(1 - xmargin, rounds + xmargin),
                                                                xticks=(range(1, rounds+1) if rounds < 8
                                                                        else None),
                                                                ylabel="fraction winning/losing")
    plt.show()


def make_player(s):
    if m := fullmatch(r"(\w+)\.(\w+)(\(.*\))?", s):
        mname = m.group(1)
        cname = m.group(2)
        args = m.group(3)
        module = import_module(mname)
        c = cname + (args or "()")
        return eval(f"module.{c}"), c
    else:
        raise RuntimeError(f"Don't know how to create player {s}")

@click.command()
@click.option("--rounds", "-r", type=int, default=100,
              help="The number of rounds to play")
@click.option("--participants", "-p", type=int, default=200,
              help="The number of participant pairs to play")
@click.option("--file", "-f", type=str, default=None,
              help="A CSV file into which to write the results")
@click.argument("player1")
@click.argument("player2")
def main(player1, player2, rounds=1, participants=1, file=None, show_progress=None):
    if file and not splitext(file)[1]:
        file += ".csv"
    if show_progress is None:
        show_progress = not player1.startswith("human") and not player2.startswith("human")
    p1, n1 = make_player(player1)
    p2, n2 = make_player(player2)
    plot_wins_losses(RPSGame(p1, p2, rounds, participants).play(show_progress),
                     title=f"{n1} versus\n{n2}\naveraged over {participants:,} participant pairs",
                     file=file)


if __name__ == '__main__':
    main()
