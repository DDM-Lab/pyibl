"""This is an example of using PyIBL, a Python implementation of a
subset of Instance Based Learning Theory, distributed by Carnegie
Mellon University's Dynamic Decision Making Laboratory. This example,
the Diners' Dilemma, uses PyIBL on a multi-agent, iterated, binary
choice task. There are at least two, and no more than ten, agents,
each making a choice between cheap and expensive. Each choice has both
a cost and a benefit associated with it. Each agent receives the
benefit associated with the choice that agent made, but each agent
pays the same fraction of the overall cost of all the agent's choices.
The utility of the choice to the agent is the difference between
that agent's benefit and its share of the total cost.

There are sixty different variations used in this experiment. The
number of agents, and associated costs and benefits of the two choices
for each variation are read from an associated CSV format data file.
This data file also contains some other fields, not used for this
example.

Each variation is run for 100 trials, by 100 different groups of the
appropriate size. For each agent in each group/variation pair there is
learning over the 100 trials. But there is no learning between groups,
variations or agents. The costs and benefits are all positive, real
numbers, no greater than four. At the conclusion a table is shown
with, for each variation, the mean (and standard deviation) across the
one hundred groups of the average number of expensive options chosen
during the one hundred trials.

At the beginning of the one hundred trials for each group/variation a
default utility of 30 is used to prime the instance matching process.
Because this default utility exceeds any value that can actually be
returned by a choice it encourages initial exploration.

Assuming PyIBL is installed this can be run by something like

  python dd-example.py

Depending upon details of you Python installation how you call Python may be slightly
different. Also note that this example takes ten minutes or longer to run, depending
upon your hardware, etc.

Parameters of the PyIBL Agents, and of the experiment, may be altered by changing the
values of the constants GROUPS, TRIALS, NOISE, DECAY and DEFAULT_UTILITY.
"""

import csv
import itertools
import math
import prettytable
import random
import statistics
import sys

from pyibl import Agent

GROUPS = 100
TRIALS = 100
NOISE = 1.5
DECAY = 5
DEFAULT_UTILITY = 30
DATA_FILE = "dd-data.csv"


def main():

    # Read the descriptions of the variations. The first row of the CSV file is a header,
    # and the values of those column headers are used below to extract fields.
    with open(DATA_FILE, newline='') as csvfile:
        data = list(csv.DictReader(csvfile))
    max_group_size = max([int(row['group_size']) for row in data])

    # Create the Agents, and configure them.
    agents = [ Agent(f"DD-{i + 1}", noise=NOISE, decay=DECAY, default_utility=DEFAULT_UTILITY)
               for i in range(max_group_size) ]

    # Create a matrix to hold the results.
    summary = [[0]*GROUPS for d in range(len(data))]

    # Actually run the experiment, printing out a progress indication as it goes.
    for group in range(GROUPS):
        sys.stdout.write(str(group + 1) if (group + 1) % 10 == 0 else '.')
        if (group + 1) % 50 == 0:
            print()
        sys.stdout.flush()
        for row, variation in zip(data, itertools.count()):
            for a in agents:
                a.reset()
            group_size = int(row['group_size'])
            for trial in range(TRIALS):
                choices = [ a.choose("cheap", "expensive") for a in agents ]
                average_cost = math.fsum([float(row[c + "_cost"]) for c in choices]) / group_size
                summary[variation][group] += choices.count("expensive") / (GROUPS * TRIALS)
                for a, c in zip(agents, choices):
                    a.respond(float(row[c + "_benefit"]) - average_cost)

    # Print the results.
    print()
    print(f"Results averaged over {GROUPS} groups, each running {TRIALS} trials of each variation:")
    tab = prettytable.PrettyTable()
    tab.add_column("Variation", [str(i + 1) for i in range(len(data))])
    tab.add_column("Group size", [d['group_size'] for d in data])
    tab.add_column("Cheap benefit", [d['cheap_benefit'] for d in data])
    tab.add_column("Cheap cost", [d['cheap_cost'] for d in data])
    tab.add_column("Expensive benefit", [d['expensive_benefit'] for d in data])
    tab.add_column("Expensive cost", [d['expensive_cost'] for d in data])
    tab.add_column("Mean expensive choice",
                   [f"{statistics.mean(summary[v]):.2f}" for v in range(len(data))])
    tab.add_column("Standard deviation",
                   [f"{statistics.stdev(summary[v]):.3f}" for v in range(len(data))])
    print(tab)


if __name__ == '__main__':
    main()

