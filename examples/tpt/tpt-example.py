"""This is an example of using PyIBL, a Python implementation of a subset of Instance
Based Learning Theory, distributed by Carnegie Mellon University's Dynamic Decision Making
Laboratory. This example uses PyIBL on iterated, binary choice tasks, where the (virtual)
participant chooses between a safe option with a fixed outcome; and a risky option with
two different possible outcomes, which being determined probabilistically. This is
implemented in this module by the function iterated_choice.

This module applies iterated_choice to the sixty problems of the Technion Prediction
Tournament (2008) (http://tx.technion.ac.il/~erev/Comp/Comp.html) problems, which are all
of this form. The sixty problems differ only in four parameters: the safe payoff, the two
possible risky payoffs, and a probability for determining which of the risky payoffs will
apply each time it is needed. The payoffs are all real numbers greater than -30 and less
than 30. These problem parameters are read from an accompanying tab separated data file;
that file also contains several other fields we ignore.

This module's function main() runs these sixty problems, each for one hundred trials, with
one hundred different participants. For each participant/problem pair there is learning
throughout the one hundred trials, but none between problems or participants. At the
conclusion a table is shown with, for each problem, the mean (and standard deviation)
across the one hundred participants of the fraction of the one hundred trials in which the
risky option was chosen.

At the beginning of the one hundred trials for each participant/problem a default utility
of 30 is used to prime the instance matching process. Because this default utility exceeds
any value that can actually be returned by a choice it encourages initial exploration.

Assuming PyIBL is installed this can be run by something like

  python tpt-example.py

Depending upon details of you Python installation how you call Python may be slightly
different.

Parameters of the PyIBL Agent, and of the experiment, may be altered by changing the
values of the constants PARTICIPANTS, TRIALS, NOISE, DECAY and DEFAULT_UTILITY.
"""

import csv
import prettytable
import random
import statistics
import sys

from pyibl import Agent

PARTICIPANTS = 100
TRIALS = 100
NOISE = 1.5
DECAY = 5
DEFAULT_UTILITY = 30
DATA_FILE = "tpt-data.csv"

def iterated_choice(*, agent, trials, high, prob, low, safe):
    """Runs an iterated, binary choice *trials* times. The possible decisions are
    ``'risky'`` or ``'safe'``. If ``'safe'`` is chosen the payoff is *safe*. If
    ``'risky'`` is chosen the payoff is *high* with a probability of *prob*, and otherwise
    *low*. The ``Agent`` used is *agent*. Returns the number of times ``'risky'`` was
    chosen.
    """
    riskyChosen = 0
    agent.reset()
    for trial in range(trials):
        choice = agent.choose('risky', 'safe')
        if choice == 'safe':
            agent.respond(safe)
        else:
            assert choice == 'risky'
            riskyChosen += 1
            agent.respond(high if random.random() <= prob else low)
    return riskyChosen

def main():
    # Create and configure the agent, setting its parameters to values given earlier
    # in this file, and arranging for a detailed log to be written, if desired.
    a = Agent("TPT", noise=NOISE, decay=DECAY, default_utility=DEFAULT_UTILITY)

    # Read the TPT problem parameters from the CSV file.
    with open("tpt-data.tsv", newline='') as tsvfile:
        data = [(int(problem), float(high), float(probability), float(low), float(safe))
                for (problem, high, probability, low, safe, ignore1, ignore2)
                in csv.reader(tsvfile, dialect='excel-tab')]

    # Create a matrix to hold how frequently risky was chosen for each
    # particpant/problem pair.
    summary = [[0]*PARTICIPANTS for d in range(len(data))]

    # Run the actual experiment, printing out some progress information along the way.
    for participant in range(PARTICIPANTS):
        sys.stdout.write(str(participant + 1) if (participant + 1) % 10 == 0 else '.')
        if (participant + 1) % 50 == 0:
            print()
        sys.stdout.flush()
        for (problem, h, probability, l, s) in data:
            n = iterated_choice(agent=a, trials=TRIALS, high=h, prob=probability,
                                low=l, safe=s)
            summary[problem - 1][participant] = n / TRIALS

    # Print a table summarizing how frequently the participants selected risky
    # for each problem.
    print()
    print(f"Results averaged over {PARTICIPANTS} participants, each running {TRIALS} trials of each problem:")
    tab = prettytable.PrettyTable()
    tab.add_column("Problem", [d[0] for d in data])
    tab.add_column("High", [d[1] for d in data])
    tab.align["High"] = 'r'
    tab.add_column("Low", [d[3] for d in data])
    tab.align["Low"] = 'r'
    tab.add_column("Probability", [f"{float(d[2]):.2f}" for d in data])
    tab.add_column("Safe", [d[4] for d in data])
    tab.align["Safe"] = 'r'
    tab.add_column("Mean risky choice",
                   [f"{statistics.mean(summary[p]):.2f}" for p in range(len(data))])
    tab.add_column("Standard deviation",
                   [f"{statistics.stdev(summary[p]):.3f}" for p in range(len(data))])
    print(tab)

if __name__ == '__main__':
    main()
