"""Internal helpers for Agent reporting, tracing, and aggregate row assembly."""

import csv
import math

import numpy as np

from ._optional_deps import PrettyTable, require_prettytable


def extract_instance_utility(instance):
    first_attr = instance["attributes"][0]
    assert first_attr[0] == "_utility"
    return first_attr[1]


def append_aggregate_details(agent, history, blended_utility):
    if agent._aggregate_details is None:
        return
    for detail in history:
        attributes = detail["attributes"]
        row = [
            agent._aggregate_iteration,
            agent.time,
            None,
            attributes[0][1],
            tuple(attribute[1] for attribute in attributes[1:]) if agent._attributes else attributes[1][1],
            blended_utility,
            detail["retrieval_probability"],
            detail["activation"],
            detail["base_level_activation"],
            detail.get("activation_noise", 0),
        ]
        similarities = detail.get("similarities")
        if similarities:
            agent._aggregate_similarities = True
            row.append(detail.get("mismatch", 0))
            row.extend(similarities.get(attribute, np.nan) for attribute in agent._attributes)
        agent._aggregate_details.append(row)


def print_trace(agent, query, utility, history, sqrt2):
    require_prettytable()
    print()
    if agent.attributes:
        print(", ".join(f"{key}: {value}" for key, value in query.items()), end="")
    else:
        print(query["_decision"], end="")
    print(f" -> {utility} @ time={agent.time}")
    tab = PrettyTable()
    fields = (["id"] + (list(agent.attributes) or ["decision"]) +
              ["created", "occurrences", "outcome", "base activation", "activation noise"])
    if agent._memory.mismatch:
        fields.append("mismatch adjustment")
    fields.extend(["total activation", "exp(act / temp)", "retrieval probability"])
    tab.field_names = fields
    for entry in history:
        attrs = dict(entry["attributes"])
        row = [entry["name"]]
        if agent.attributes:
            for attribute in agent.attributes:
                row.append(attrs.get(attribute, ""))
        else:
            row.append(attrs["_decision"])
        row.append(entry["creation_time"])
        row.append(entry["references"] if agent._memory.optimized_learning else list(entry["references"]))
        row.append(attrs["_utility"])
        row.append(entry["base_level_activation"])
        row.append(entry.get("activation_noise") or 0.0)
        if agent._memory.mismatch:
            row.append(entry["mismatch"])
        row.append(entry["activation"])
        row.append(math.exp(entry["activation"] / (agent.temperature or sqrt2 * agent.noise)))
        row.append(entry["retrieval_probability"])
        tab.add_row(row)
    print(tab, flush=True)


def print_instance_data(data, pretty, file):
    if not data:
        return
    if pretty:
        require_prettytable()
        tab = PrettyTable()
        tab.field_names = data[0].keys()
        for row in data:
            tab.add_row(row.values())
        print(tab, file=file, flush=True)
    else:
        writer = csv.DictWriter(file, data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)
