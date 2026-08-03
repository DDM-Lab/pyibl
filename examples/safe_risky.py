# Copyright 2014–2026 Carnegie Mellon University

import pyibl
import random

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    _MISSING_TQDM_WARNED = False

    def tqdm(iterable=None, **kwargs):
        global _MISSING_TQDM_WARNED
        if not _MISSING_TQDM_WARNED:
            print("tqdm is not installed; progress bars are disabled. Install it with: pip install tqdm")
            _MISSING_TQDM_WARNED = True
        return iterable if iterable is not None else range(0)

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

if plt is None:
    print("matplotlib is not installed; plotting is disabled. Install it with: pip install matplotlib")
    print(f"Final risky choice fraction: {risky_chosen[-1] / PARTICIPANTS:.3f}")
else:
    plt.plot(range(ROUNDS), [v / PARTICIPANTS for v in risky_chosen])
    plt.ylim([0, 1])
    plt.ylabel("fraction choosing risky")
    plt.xlabel("round")
    plt.title(f"Safe (1 always) versus risky (3 × ⅓, 0 × ⅔)\nσ={a.noise}, d={a.decay}")
    plt.show()
