# Copyright 2024 Carnegie Mellon University
# Binary choice example using PyIBL and multiple processes

from alhazen import IteratedExperiment
import matplotlib.pyplot as plt
import numpy as np
from pyibl import Agent
from random import random

HIGH_PAYOUTS = [4, 6, 12]
SAFE_PAYOUT = 3
PLOT_FILE = "binary-choice.png"
ROUNDS = 60
PARTICIPANTS = 10_000
PREPOPULATED_MULTIPLIER = 1.2
PROCESSES = 0

class BinaryChoice(IteratedExperiment):

    def prepare_condition(self, condition, context):
        context["high-probability"] = SAFE_PAYOUT / condition

    def run_participant_prepare(self, participant, condition, context):
        self.agent = Agent(default_utility=(PREPOPULATED_MULTIPLIER * condition))

    def run_participant_run(self, round, participant, condition, context):
        choice = self.agent.choose(["safe", "risky"])
        if choice == "safe":
            payoff = SAFE_PAYOUT
        elif random() < context["high-probability"]:
            payoff = condition
        else:
            payoff = 0
        self.agent.respond(payoff)
        return int(choice == "risky")

def main():
      exp = BinaryChoice(rounds=ROUNDS,
                         conditions=HIGH_PAYOUTS,
                         participants=PARTICIPANTS,
                         process_count=PROCESSES)
      results = exp.run()
      for condition in exp.conditions:
          plt.plot(range(1, ROUNDS + 1), np.mean(np.asarray(results[condition]), axis=0),
                   label=f"risky high payoff = {condition} points")
      plt.xticks([1] + [10 * n for n in range(1, round((ROUNDS + 10) / 10))])
      plt.ylim([0, 1])
      plt.yticks([round(n / 4, 2) for n in range(5)])
      plt.ylabel("fraction choosing risky")
      plt.xlabel("round")
      plt.legend()
      plt.title(f"Safe ({SAFE_PAYOUT} points) verus risky, {PARTICIPANTS:,} participants")
      plt.savefig(PLOT_FILE)

if __name__ == '__main__':
    main()
