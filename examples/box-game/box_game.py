# Copyright 2019–2025 Carnegie Mellon University

import click
import csv
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import pyibl
import random
from tqdm import tqdm

DEFAULT_ROUNDS = 50
DEFAULT_PARTICIPANTS = 1000
DEFAULT_NOISE = 0.25
DEFAULT_DECAY = 0.5
DEFAULT_TEMPERATURE = 1
LOGFILE = "box-game-log.csv"

CONDITIONS = [{"name": n, "p": p, "q": q} for n, p, q in [("1 way",        1,    0.5),
                                                          ("2 way (0.75)", 0.75, 0.375),
                                                          ("2 way (0.5)",  0.5,  0.25),
                                                          ("no signaling", None, None)]]

def run(condition, rounds=DEFAULT_ROUNDS, participants=DEFAULT_PARTICIPANTS,
        noise=DEFAULT_NOISE, decay=DEFAULT_DECAY, temperature=DEFAULT_TEMPERATURE,
        logwriter=None, progress=None):
    for c in CONDITIONS:
        if c["name"] == condition:
            cond_p = c["p"]
            cond_q = c["q"]
            break
    else:
        raise ValueError(f"Unknown condition {condition}")
    selection_agent = pyibl.Agent(noise=noise, decay=decay, temperature=temperature)
    attack_agent = pyibl.Agent(attributes=(["attack"] + ([] if cond_p is None else ["warning"])),
                               noise=noise, decay=decay, temperature=temperature)
    successful_attacks = 0
    failed_attacks = 0
    withdrawals = 0
    if cond_p is None:
        attack_agent.populate ([{"attack": False}],
                               0)
    else:
        attack_agent.populate ([{"attack": False, "warning": 0},
                                {"attack": False, "warning": 1}],
                               0)
    for v in [100, -50]:
        selection_agent.populate([0, 1], v)
        if cond_p is None:
            attack_agent.populate([{"attack": True}],
                                  v)
        else:
            attack_agent.populate([{"attack": True, "warning": 0},
                                   {"attack": True, "warning": 1}],
                                  v)
    for p in range(participants):
        total = 0
        selection_agent.reset(True)
        attack_agent.reset(True)
        for r in range(rounds):
            selected = selection_agent.choose((0, 1))
            covered = random.random() < 0.5
            if cond_p is None:
                attack = attack_agent.choose([{"attack": True},
                                              {"attack": False}])["attack"]
            else:
                if covered:
                    warned = int(random.random() < (1- cond_p))
                else:
                    warned = int(random.random() < cond_q)
                attack = attack_agent.choose([{"attack": True, "warning": warned},
                                              {"attack": False, "warning": warned}])["attack"]
            if not attack:
                withdrawals += 1
                payoff = 0
            elif covered:
                failed_attacks += 1
                payoff = -50
            else:
                successful_attacks += 1
                payoff = 100
            total += payoff
            attack_agent.respond(payoff)
            selection_agent.respond(payoff)
            logwriter.writerow([condition, p + 1, r + 1, selected,
                                (int(warned) if cond_p is not None else None),
                                int(covered), int(attack), payoff, total])
        if progress:
            progress.update()
    return [n / (participants * rounds)
            for n in [successful_attacks, failed_attacks, withdrawals]]

@click.command()
@click.option("--rounds", "-r", default=DEFAULT_ROUNDS,
              help="number of rounds to play")
@click.option("--participants", "-p", default=DEFAULT_PARTICIPANTS,
              help="number of virtual participants to simulate")
@click.option("--noise", "-n", default=DEFAULT_NOISE,
              help="noise for the two agents")
@click.option("--decay", "-d", default=DEFAULT_DECAY,
              help="decay parameter for the two agents")
@click.option("--temperature", "-t", default=DEFAULT_TEMPERATURE,
              help="blending temperature for the two agents")
def main(rounds, participants, noise, decay, temperature):
    results = {"successful attack": [], "failed attack": [], "withdrew": []}
    colors = ("red", "green", "blue")
    with tqdm(total=(participants * len(CONDITIONS))) as p:
        with open(LOGFILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow("Condition,Subject,Trial,Selected,Warning,Covered,Action,Outcome,Cum_Outcome".split(","))
            for c in CONDITIONS:
                cname = c["name"]
                r = run(cname, rounds=rounds, participants=participants,
                        noise=noise, decay=decay, temperature=temperature,
                        logwriter=w, progress=p)
                for k, v in zip(results.keys(), r):
                    results[k].append(round(v, 2))
    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(len(CONDITIONS))
    wid = 0.25
    for (kind, vals), mult, c  in zip(results.items(), count(), colors):
        offset = wid * mult
        rects = ax.bar(x + offset, vals, wid, label=kind, color=c)
        ax.bar_label(rects, padding=3)
        mult += 1
    ax.set_xticks(x + wid, [c["name"] for c in CONDITIONS])
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 0.6)
    ax.set_title(f"{participants} participants, {rounds} rounds\n"
                 f"noise={noise}, decay={decay}, temperature={temperature}")
    plt.show()


if __name__ == "__main__":
    main()
