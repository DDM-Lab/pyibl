# Copyright 2024 Carnegie Mellon University

from itertools import repeat
from pyibl import Agent
from rps_game import RPSPlayer, MOVES, RESULTS


"""
Several different IBL models for playing Rock, Paper, Scissors.
"""

class IBLPlayer(RPSPlayer):
    """
    A base class for a players using a single PyIBL Agent. The attributes argument is
    passed to the Agent constructor. The payoffs argument is a 3-tuple of the payoffs
    for the possible results in the order lose, tie, win.
    """

    def __init__(self, attributes=None, payoffs=(-1, 0, 1), kwd={}):
        super().__init__()
        self._agent = Agent(attributes)
        for a in ("noise", "decay", "temperature"):
            if v := kwd.get(a):
                setattr(self._agent, a, v)
        # The _payoffs in the object are stored in a different order than the human
        # friendly order used for the parameter, instead matching he order in which
        # they appear in rps_game.RESULTS: tie, win, lose
        self._payoffs = payoffs[1:] + payoffs[:1]
        self._initial_payoff = 1.2 * max(payoffs)

    def reset(self):
        self._agent.reset(self._agent.default_utility in (None, False))

    def respond(self, outcome):
        self._agent.respond(self._payoffs[RESULTS.index(outcome)])


class BasicIBLPlayer(IBLPlayer):
    """
    A simplistic IBL model that simply notes how well we do for each possible move. While
    this might work against a not very smart opponent, say one that almost always picks
    rock, against players that are learning and responding from the history of our moves
    it is unlike to do well.
    """

    def __init__(self, **kwd):
        super().__init__(kwd)
        self._agent.default_utility = self._initial_payoff
        self.reset()

    def reset(self):
        super().reset()
        self._agent.reset()

    def move(self):
        return self._agent.choose(MOVES)

    def result(self, opponent_move, outcome, wins, ties, losses):
        self.respond(outcome)


class ContextualIBLPlayer(IBLPlayer):
    """
    A slightly smarter IBL model that chooses its move based on what move our opponent
    made in the last round.
    """

    def __init__(self, **kwd):
        super().__init__(["move", "opponent_previous_mode"], kwd=kwd)
        self._agent.default_utility = self._initial_payoff
        self.reset()

    def reset(self):
        super().reset()
        self._opponent_previous = None

    def move(self):
        return self._agent.choose(zip(MOVES, repeat(self._opponent_previous)))[0]

    def result(self, opponent_move, outcome, wins, ties, losses):
        self.respond(outcome)
        self._opponent_previous = opponent_move


NONE_MATCH = 0.5

def move_sim(x, y):
    if x == y:
        return 1
    elif x is None or y is None:
        return NONE_MATCH
    else:
        return 0

def shift(element, list):
    # Adds element to the front of the list, shifting the existing elements towards the
    # back, with the oldest element falling off the end of the list.
    list.pop()
    list.insert(0, element)


class LagIBLPlayer(IBLPlayer):
    """
    An IBL model that keeps track of the past N (= log) moves of both out opponent and
    our own move, thus capturing how our opponent responds to our moves. Because there
    are so many possibilities we use partial matching with not yet seen possibilities
    viewed as half as salient as those that match perfectly.
    """

    def __init__(self, lag=1, mismatch_penalty=1, **kwd):
        self._lag = lag
        move_attrs = (["opp-" + str(i) for i in range(1, lag + 1)] +
                      ["own-" + str(i) for i in range(1, lag + 1)])
        super().__init__(["move"] + move_attrs, kwd=kwd)
        self._agent.mismatch_penalty = mismatch_penalty
        self._agent.similarity(move_attrs, move_sim)
        self.reset()
        self._agent.populate(self.choices(), self._initial_payoff)

    def reset(self):
        self._opp_prev = [None] * self._lag
        self._own_prev = [None] * self._lag

    def choices(self):
        return [[move] + lst
                for move, lst in zip(MOVES, repeat(self._opp_prev + self._own_prev))]

    def move(self):
        self._move = self._agent.choose(self.choices())[0]
        return self._move

    def result(self, opponent_move, outcome, wins, ties, losses):
        self.respond(outcome)
        shift(opponent_move, self._opp_prev)
        shift(self._move, self._own_prev)
