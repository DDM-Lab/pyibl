import pyibl
import random

import matplotlib.pyplot as plt

from tqdm import tqdm

PARTICIPANTS = 10_000
ROUNDS = 60

risky_chosen = [0] * ROUNDS
a = pyibl.Agent()
for p in tqdm(range(PARTICIPANTS)):
    a.reset()
    a.default_utility = 3.2
    for r in range(ROUNDS):
        choice = a.choose(["safe", "risky"])
        if choice == "risky":
            payoff = 3 if random.random() < 1/3 else 0
            risky_chosen[r] += 1
        else:
            payoff = 1
        a.respond(payoff)

plt.plot(range(ROUNDS), [ v / PARTICIPANTS for v in risky_chosen])
plt.ylim([0, 1])
plt.ylabel("fraction choosing risky")
plt.xlabel("round")
plt.title(f"Safe (1 always) versus risky (3 × ⅓, 0 × ⅔)\nσ={a.noise}, d={a.decay}")
plt.show()
