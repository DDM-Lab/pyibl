# Copyright 2024 Carnegie Mellon University
# Binary choice example using PyIBL

import matplotlib.pyplot as plt
import numpy as np
from pyibl import Agent
from random import random
from tqdm import tqdm

HIGH_PAYOUTS = [4, 6, 12]
SAFE_PAYOUT = 3
PLOT_FILE = "binary-choice.png"
ROUNDS = 60
PARTICIPANTS = 10_000
PREPOPULATED_MULTIPLIER = 1.2

def run_condition(high_payout, progress):
    results = []
    high_probability = SAFE_PAYOUT / high_payout
    for participant in range(PARTICIPANTS):
        agent = Agent(default_utility=(PREPOPULATED_MULTIPLIER * high_payout))
        round_results = [None] * ROUNDS
        for round in range(ROUNDS):
            choice = agent.choose(["safe", "risky"])
            if choice == "safe":
                payoff = SAFE_PAYOUT
            elif random() < high_probability:
                payoff = high_payout
            else:
                payoff = 0
            agent.respond(payoff)
            round_results[round] = int(choice == "risky")
        results.append(round_results)
        progress.update()
    return results

def main():
    progress = tqdm(total=(len(HIGH_PAYOUTS) * PARTICIPANTS))
    for payout in HIGH_PAYOUTS:
        plt.plot(range(1, ROUNDS + 1),
                 np.mean(np.asarray(run_condition(payout, progress)), axis=0),
                 label=f"risky high payoff = {payout} points")
    plt.xticks([1] + [10 * n for n in range(1, round((ROUNDS + 10) / 10))])
    plt.ylim([0, 1])
    plt.yticks([round(n / 4, 2) for n in range(5)])
    plt.ylabel("fraction choosing risky")
    plt.xlabel("round")
    plt.legend()
    plt.title(f"Safe ({SAFE_PAYOUT} points) versus risky, {PARTICIPANTS:,} participants")
    plt.savefig(PLOT_FILE)

if __name__ == '__main__':
    main()
