# Copyright 2025 Carnegie Mellon University
# Example of networked binary choice games implemented with PyIBL

import click
from collections import Counter
import csv
from datetime import datetime
from itertools import count
import matplotlib.pyplot as plt
from pyibl import Agent
import random
from tqdm import tqdm

DEFAULT_ROUNDS = 60
DEFAULT_PARTICIPANT_SETS = 500

DEFAULT_NOISE = 0.25
DEFAULT_DECAY = 0.5

GAMES = {"independence": {("A", "A"): (5, 5),
                          ("A", "B"): (5, 0),
                          ("B", "A"): (0, 5),
                          ("B", "B"): (0, 0)},
         "interdependence": {("A", "A"): (5, 5),
                             ("A", "B"): (0, 5),
                             ("B", "A"): (5, 0),
                             ("B", "B"): (0, 0)}}

PREPOPULATED_VALUE = 6

PAIRINGS = {
    "disconnected": [[(0, 1), (2, 3), (4, 5)]],
    "ring": [[(0, 1), (2, 3), (4,5)],
             [(1, 2), (3, 4), (0, 5)]],
    "fully-connected": [[(4, 5), (2, 3), (0, 1)],
                        [(3, 5), (2, 4), (0, 1)],
                        [(3, 4), (2, 5), (0, 1)],
                        [(4, 5), (1, 3), (0, 2)],
                        [(3, 5), (1, 4), (0, 2)],
                        [(3, 4), (1, 5), (0, 2)],
                        [(4, 5), (1, 2), (0, 3)],
                        [(2, 5), (1, 4), (0, 3)],
                        [(2, 4), (1, 5), (0, 3)],
                        [(3, 5), (1, 2), (0, 4)],
                        [(2, 5), (1, 3), (0, 4)],
                        [(2, 3), (1, 5), (0, 4)],
                        [(3, 4), (1, 2), (0, 5)],
                        [(2, 4), (1, 3), (0, 5)],
                        [(2, 3), (1, 4), (0, 5)]]}

PAIRS = len(list(PAIRINGS.values())[0][0])
for kind in PAIRINGS.values():
    for pairing in kind:
        assert len(pairing) == PAIRS
NODES = 2 * PAIRS

def run_one(agents, network, game, participant_sets, rounds, progress, csv_writer, plot_file):
    counts = [Counter() for i in range(rounds)]
    for pp in range(participant_sets):
        for a in agents:
            a.reset()
        for r in range(rounds):
            for p in random.choice(PAIRINGS[network]):
                p = random.sample(p, k=len(p)) # counterbalance
                choices = tuple(agents[i].choose("AB") for i in p)
                counts[r].update((choices,))
                payoffs = GAMES[game][choices]
                for i, j in zip(p, count()):
                    agents[i].respond(payoffs[j])
                csv_writer.writerow((network, game, pp + 1, r + 1,
                                     p[0], choices[0], payoffs[0],
                                     p[1], choices[1], payoffs[1]))
            progress.update()
    plt.plot(tuple(range(1, rounds + 1)),
             tuple(counts[r][("A", "A")] / (participant_sets * PAIRS) for r in range(rounds)),
             label='("A", "A")', color="green", linestyle="solid")
    plt.plot(tuple(range(1, rounds + 1)),
             tuple(counts[r][("A", "B")] / (participant_sets * PAIRS)  for r in range(rounds)),
             label='("A", "B")', color="blue", linestyle="dotted")
    plt.plot(tuple(range(1, rounds + 1)),
             tuple(counts[r][("B", "A")] / (participant_sets * PAIRS)  for r in range(rounds)),
             label='("B", "A")', color="violet", linestyle="dotted")
    plt.plot(tuple(range(1, rounds + 1)),
             tuple(counts[r][("B", "B")] / (participant_sets * PAIRS)  for r in range(rounds)),
             label='("B", "B")', color="red", linestyle="dashed")
    plt.legend()
    plt.title(f"{game} game, {network} topology\n"
              f"{participant_sets} participant sets, noise={agents[0].noise}, decay={agents[0].decay}")
    plt.xlim((0, rounds + 2))
    plt.xticks((1, rounds / 2, rounds))
    plt.xlabel("round")
    plt.ylim((-0.05, 1.05))
    plt.yticks((0, 0.5, 1))
    plt.ylabel("fraction of pairs choosing")
    if plot_file:
        plt.savefig(plot_file)
    else:
        plt.show()
    plt.clf()


@click.command()
@click.option("--rounds", "-r", default=DEFAULT_ROUNDS, type=int,
              help="The number of rounds to play")
@click.option("--participant-sets", "-p", default=DEFAULT_PARTICIPANT_SETS, type=int,
              help="The number of participant sets to simulate")
@click.option("-noise", "-s", default=DEFAULT_NOISE, type=float,
              help="The IBL activation noise to use")
@click.option("--decay", "-d", default=DEFAULT_DECAY, type=float,
              help="The IBL decay parameter to use")
def main(rounds, participant_sets, noise, decay):
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("network,game,participant set,round,"
                    "player one,player one move,player one payoff,"
                    "player two,player two move,player two payoff").split(","))
        agents = [Agent(str(i), default_utility=PREPOPULATED_VALUE, noise=noise, decay=decay)
                  for i in range(1, NODES + 1)]
        with tqdm(total=(len(GAMES) * len(PAIRINGS) * participant_sets * rounds)) as t:
            for n in PAIRINGS.keys():
                for g in GAMES.keys():
                    run_one(agents, n, g, participant_sets, rounds,
                            t, w, f"{n}-{g}.png")


if __name__ == '__main__':
    main()
