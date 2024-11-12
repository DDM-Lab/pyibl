================================
Rock paper scissors game example
================================

Here is described a Python implementation of the Rock, Paper, Scissors game [#f2]_, and how to
connect a variety of models to it. Rock, Paper, Scissors is a game where two players
compete, each choosing one of three possible moves. It is most interesting when
iterated many times, as players may possibly be able to learn about their opponent’s
biases and exploit them. Note that there are three outcomes possible in any round, two where one
player wins and the other loses, and a third where the players tie. This can have implications for
strategies, as, for example, maximizing a player’s numbers wins is not the same as minimizing that player’s number of loses.
The game has been extensively studied.

:download:`Click here to download </_downloads/rps.zip>` a zipped archive for the various files of code
described in this section, along with a ``requirements.txt`` file. The recommended way of running these
examples yourself is to create and activate a virtual environment using venv or conda, doing ``pip install -r requirements.txt`` in
it, and then in it running Python on the desired file.

The implementation of the game, which is completely independent of PyIBL,  is in ``rps-game.py``.

.. code-block:: python
    :linenos:

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
                         title=f"{n1} versus\n{n2}\n(averaged over {participants} participants)",
                         file=file)


    if __name__ == '__main__':
        main()


This defines a class, ``RPSPlayer``, which is
subclassed to implement various player types.
There is a further ``RPSGame`` class which is constructed with two players which are
subclasses of ``RPSPlayer``;
the two players are typically, though not necessarily, of different subclasses.
The ``RPSGame`` object calls the players repeatedly for a number of rounds,
typically for several or many  virtual participant pairs, and gathers the results,
returning them as a `Pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
The ``rps_game.py`` file also contains a function, ``plot_winsₗosses`` that takes such a DataFrame and plots
the wins and losses of the first player against the second using the `Matplotlib <https://matplotlib.org/>`_ library.

When creating a subclass of ``RPSPlayer``
its ``move()`` method must be overridden to return one of the string values ``"rock"``, ``"paper"`` or ``"scissors"``.
Usually the method ``result()`` is also overridden, allowing display or capture
of the results of a round of the game, though for some very simple models this may not be necessary.
Similarly the ``reset()`` method may be overridden if the model is retaining state carried from round to round that
may need to be reset between virtual participants.
A simple human player is defined with the subclass ``HumanPlayer`` in ``human.py``;
note that the ``HuamnPlayer`` overrides ``move()`` to request a move from the player and returns it; and
overrides ``result()`` to display the results:

.. code-block:: python
    :linenos:

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

It would also be relatively straightforward to create a web-based interface as an ``RPSPlayer``, which
would allow a human-human game to be played.

In addition to this human player, a number of models are implemented, several using PyIBL, and are described in subsequent subsections.

Finally, ``rps-game.py`` implements a command line interface, creating an ``RPSGame`` with players of designated types,
playing potentially many rounds with many virtual pairs of those players, and then plotting the results of these games.
Because running large number of participants can require long periods of time, particularly for some kinds of models and/or
large numbers of rounds, a progress indicator is usually shown while results are being computed.
For example, to run 1,000 pairs of ``WinStayLoseShiftPlayer`` (described further below) against a ``RandoomPlayer`` (also described further below),
the latter biased to return rock 80% of the time, paper 20%, and scissors never, with each pair playing 60 rounds, you could call::

    python rps_game.py --participants=1000 --rounds=60 wsls.WinStayLoseShiftPlayer "rand.RandomPlayer(bias=(0.8, 0.2))"

This will result in display of a graph much like the following, though it may differ
slightly in detail since both models are stochastic.

.. image:: _static/rps-1.png
   :width: 700

By supplying a ``--file`` argument you can also save the resulting DataFrame describing the full results
into a CSV file. For example, if we add ``--file=results.csv`` the first few lines of the resulting file will look something
like the following, though again differing in detail since the models are stochastic.

    ,participant pair,round,player 1 move,player 2 move,player 1 outcome,player 2 outcome,player 1 total wins,player 2 total wins
    0,1,1,scissors,rock,lose,win,0,1
    1,1,2,rock,rock,tie,tie,0,1
    2,1,3,scissors,rock,lose,win,0,2
    3,1,4,rock,rock,tie,tie,0,2
    4,1,5,paper,rock,win,lose,1,2
    5,1,6,paper,rock,win,lose,2,2
    6,1,7,paper,rock,win,lose,3,2
    7,1,8,paper,rock,win,lose,4,2
    8,1,9,paper,rock,win,lose,5,2
    9,1,10,paper,rock,win,lose,6,2
    10,1,11,paper,rock,win,lose,7,2
    11,1,12,paper,rock,win,lose,8,2
    12,1,13,paper,rock,win,lose,9,2
    13,1,14,paper,paper,tie,tie,9,2
    ...

Or, when imported into a spreadsheet:

.. image:: _static/rps-2.png
   :width: 700



.. [#f2] Dyson, B. J. *et al*. Negative outcomes evoke cyclic irrational decisions in Rock, Paper, Scissors.
         *Sci. Rep*. 6, 20479; doi: 10.1038/srep20479 (2016).




Random model
------------

Perhaps the simplest model, ``RandomPlayer`` in ``rand.py``, simply chooses a move at random.

.. code-block:: python
    :linenos:

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


By default it uses a uniform random distribution to choose between the three moves.
While this is "optimal" play in the sense that no other player can exploit it to win more than one third
of the time, neither can the random player exploit other players' biases.

If desired the distribution used to choose moves may be skewed by setting the ``bias`` parameter, which should
be a 2-tuple of the probabilities of choosing rock and paper. for example, to choose rock 80% of the time,
paper never, and scissors 20% of the time, ``RandomPlayer(bias=(0.8, 0)``.


Win stay lose shift model
-------------------------

A strategy that has been widely studied is Win Stay, Lose Shift, which is demonstrated here.

.. code-block:: python
    :linenos:

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


When shifting after a loss a choice needs to be made between the two other possible moves. By default
this choice is made uniformly at random. However an upgrade or downgrade can be made preferred
by setting the ``upgrade`` parameter to the probability that after a loss the next move is an upgrade
from this player's previous move. Note that strategies where a loss is always followed by an upgrade
or a downgrade can easily be used simply by setting ``upgrade`` to ``1`` or ``0``, respectively.

For example, to play a win stay, lose shift strategy, with a loss resulting in a 20%
chance of an upgrade and an 80% chance of a downgrade, against a biased random strategy favoring rock
80% of the time we could do::

    python rps_game.py "wsls.WinStayLoseShiftPlayer(upgrade=0.2)" "rand.RandomPlayer(bias=(0.8, 0.15))"

This results in a plot similar to the following.

.. image:: _static/rps-3.png
   :width: 700


IBL models
----------

Neither of the above models have any dependence upon PyIBL, but we can easily create PyIBL models, too, which allow us to
see how various IBL models can fare playing Rock, Paper, Scissors. Three such models are in ``ibl.py``.

.. code-block:: python
    :linenos:

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


All three models use one PyIBL Agent for each player, and the code managing that agent is shared by having each model
a subclass of an base ``IBLPlayer`` class. What attributes the created Agent has is set with the ``attributes`` argument
to the ``IBLPlayer`` constructor. When making such an IBL model we must decide what the utilities are of
winning, losing or tying a game. By default ``IBLPlayer`` sets these to 1 point for a win, 0 points for a tie, and -1 points
for a loss. These values can be adjusted with the ``payoffs`` argument to the constructor. The default prefers winning, but
views tying as preferable to losing; but for some investigations it may be appropriate to aim solely at maximizing wins, or
minimizing losses, so ``payoffs`` something like ``(0, 0, 1)`` or ``(0, 1, 1)``, respectively, might be appropriate.
The ``IBLPlayer`` class also makes available to its subclasses a suitable value for prepopulated instances and the like for encouraging
exploration, ``_initial_payoff``, based on these possible payoff values. In addition, the usual IBL parameters (noise, decay and blending temperature)
can also be modified simply by change parameters passed to the constructor.

The simplest IBL model, ``BasicIBLPlayer``,  simply bases its choice on the history of results from having made each move, with no concern for
the moves made in the preceding rounds. This can work against unsophisticated opponents, such as one that favors rock 60% of the time::

    python rps_game.py ibl.BasicIBLPlayer "rand.RandomPlayer(bias=(0.6, 0.2))"

.. image:: _static/rps-4.png
   :width: 700


But this basic model is readily defeated by a more sophisticated opponent, such as one employing a win stay, lose shift strategy::

    python rps_game.py ibl.BasicIBLPlayer "wsls.WinStayLoseShiftPlayer(upgrade=1)"

.. image:: _static/rps-5.png
   :width: 700

A smarter model can use a PyIBL attribute to base its move selection on the opponents previous move. This does better against
the wind stay, stay loses shift opponent, above::

    python rps_game.py ibl.ContextualIBLPlayer "wsls.WinStayLoseShiftPlayer(upgrade=1)"

.. image:: _static/rps-6.png
   :width: 700

But if the win stay, lose shift opponent randomly selects what it does on a loss, the ``ContextualIBLPlayer`` will fare poorly against it::

    python rps_game.py ibl.ContextualIBLPlayer "wsls.WinStayLoseShiftPlayer(upgrade=0.5)"

.. image:: _static/rps-7.png
   :width: 700

To improve this IBL model further we want to have it capture not just the experience of what opponent moves most likely follow its own moves,
but how it responds to our moves. In a ``LagIBLModel`` we capture the results based on both players' past moves.
By default it only uses the immediately preceding moves of both players, but by setting the ``lag`` parameter this can be increased to
multiple past moves.
A full set of results for all pairs of possible moves is large, especially if we consider multiple past moves, we use partial matching
to treat unseen results as of some value, but not as much as one's we've really seen. The similarly function, ``sim()``, returns 0.5,
for attributes that do not match instances we've seen, allowing instances only some of whose attributes match to still contributed to
the blended value, albeit with less weight than those that have more matching attributes.

With a lag of only one, matching just the preceding moves of both places, this model handily dominates the win stay, lose shift strategy
even with an evenly distributed random selection of upgrades/downgraes::

    python rps_game.py ibl.LagIBLPlayer "wsls.WinStayLoseShiftPlayer(upgrade=0.5)"

.. image:: _static/rps-8.png
   :width: 700

Because most of these models, both the conventional ones and the IBL ones, have various parameters that can be modified
it is easy to compare the results of a variety of differing strategies.
