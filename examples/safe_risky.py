"""This is an example of using PyIBL, a Python implementation of a
subset of Instance Based Learning Theory, distributed by Carnegie
Mellon University's Dynamic Decision Making Laboratory. This example
uses PyIBL on iterated, binary choice tasks, where the (virtual)
participant chooses between a safe option with a fixed outcome; and a
risky option with two different possible outcomes, which determined
probabilistically.

It is run for three different versions of the risky choice. The safe choice always wins a
reward of zero, while the risky choice returns -5 or +5, with three different
probabilities of which. The run() function runs the model against this task. The run
function takes parameters describing the probably of the positive or negative risky
outcome, how many rounds each virtual participant plays, and the number of virtual
participants.

The run() function is called by a main() function that runs ten thousand virtual
participants for sixty rounds each, for each of three different probabilities, and plots
the average results for each probability by round.

Besides PyIBL, the Python packages matplotlib and tqdm must be installed.

Assuming they and PyIBL are installed this can be run by something like

  python safe_risky.py

Depending upon details of you Python installation how you call Python may be slightly
different.

Parameters of the PyIBL Agent, and of the experiment, may be altered by changing the
values of the constants PARTICIPANTS, ROUNDS, NOISE, DECAY and DEFAULT_UTILITY.

"""

import pyibl
import random

import matplotlib.pyplot as plt

from tqdm import tqdm

PARTICIPANTS = 10_000
ROUNDS = 60
NOISE = 0.25
DECAY = 0.5
DEFAULT_UTILITY = 30

def run(rounds, participants, risky_wins=0.5):
    risky_chosen = [0] * rounds
    a = pyibl.Agent(default_utility=DEFAULT_UTILITY, noise=NOISE, decay=DECAY)
    for p in tqdm(range(participants)):
        a.reset()
        for r in range(rounds):
            c = a.choose("safe", "risky")
            if c == "safe":
                a.respond(0)
            else:
                risky_chosen[r] += 1
                a.respond(5 if random.random() < risky_wins else -5)
    return [ n / participants for n in risky_chosen ]

def main():
    for p in (0.6, 0.5, 0.4):
        plt.plot(range(1, ROUNDS + 1), run(ROUNDS, PARTICIPANTS, p), label=f"p={p}")
    plt.ylim([0, 1])
    plt.ylabel("fraction choosing risky")
    plt.xlabel("round")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
