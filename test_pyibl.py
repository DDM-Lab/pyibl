# Copyright 2014-2024 Carnegie Mellon University

import math
import pytest
import random
import re
import sys

from collections import defaultdict
from contextlib import contextmanager
from math import isclose
from operator import __eq__
from pprint import pp

from pyibl import *
import insider
from pyactup import Chunk

@contextmanager
def randomseed(n=0):
    old = random.getstate()
    try:
        random.seed(n)
        yield
    finally:
        random.setstate(old)

def test_agent_init():
    a = Agent()
    assert re.fullmatch(r"agent-\d+", a.name)
    assert a.attributes == ()
    assert isclose(a.noise, 0.25)
    assert isclose(a.decay, 0.5)
    assert a.temperature is None
    assert a.mismatch_penalty is None
    assert not a.optimized_learning
    assert a.default_utility is None
    assert not a.default_utility_populates
    assert a.time == 0
    with pytest.warns(UserWarning):
        a = Agent(name="Test Agent",
                  attributes=["a1", "a2"],
                  noise=0,
                  decay=0,
                  temperature=1,
                  mismatch_penalty=1,
                  optimized_learning=True,
                  default_utility=1)
    a2 = Agent(name="Test Agent")
    assert a != a2
    assert a.name == "Test Agent"
    assert a2.name == "Test Agent"
    assert a.attributes == ("a1", "a2")
    assert isclose(a.noise, 0)
    assert isclose(a.decay, 0)
    assert isclose(a.temperature, 1)
    assert isclose(a.mismatch_penalty, 1)
    assert a.optimized_learning
    assert isclose(a.default_utility, 1)
    assert not a.default_utility_populates
    assert a.time == 0
    assert a2.attributes == ()
    assert isclose(a2.noise, 0.25)
    assert isclose(a2.decay, 0.5)
    assert a2.temperature is None
    assert a2.mismatch_penalty is None
    assert not a2.optimized_learning
    assert a2.default_utility is None
    assert not a2.default_utility_populates
    assert a2.time == 0
    with pytest.raises(ValueError):
        Agent(noise=-0.001)
    with pytest.raises(ValueError):
        Agent(decay=-0.001)
    with pytest.raises(ValueError):
        Agent(temperature=-0.001)
    with pytest.raises(ValueError):
        Agent(mismatch_penalty=-0.001)
    with pytest.raises(TypeError):
        Agent(attributes=1)
    assert Agent("a b c").attributes == ("a", "b", "c")
    assert Agent("b,c,a").attributes == ("b", "c", "a")
    assert Agent("b,c,a").attributes == ("b", "c", "a")
    assert Agent(name="Orange", attributes="c,b,a").attributes == ("c", "b", "a")
    with pytest.raises(ValueError):
        Agent("a b c d e c f g")

def test_noise():
    a = Agent()
    assert isclose(a.noise, 0.25)
    with pytest.warns(UserWarning):
        a.noise = 0
    assert a.noise == 0
    a.noise = 1
    assert isclose(a.noise, 1)
    a.noise = None
    assert a.noise == 0
    a.noise = False
    assert a.noise == 0
    with pytest.raises(ValueError):
        a.noise = -1

def test_noise_distribution():
    a = Agent()
    assert a.noise_distribution is None
    a.noise_distribution = lambda: random.random() - 0.5
    assert a.noise_distribution
    a.noise_distribution = None
    assert a.noise_distribution is None
    with pytest.raises(ValueError):
        a.noise_distribution = 1

def test_temperature():
    a = Agent()
    assert a.temperature is None
    a.temperature = 1
    assert isclose(a.temperature, 1)
    a.temperature = None
    assert a.temperature is None
    a.temperature = False
    assert a.temperature is None
    with pytest.raises(ValueError):
        a.temperature = 0
    with pytest.raises(ValueError):
        a.temperature = -1
    with pytest.raises(ValueError):
        a.temperature = 0.0001
    a.temperature = 1
    a.noise = 0
    with pytest.raises(ValueError):
        a.temperature = None
    a.noise = 0.0001
    with pytest.raises(ValueError):
        a.temperature = None

def test_decay():
    a = Agent()
    assert isclose(a.decay, 0.5)
    a.decay = 0
    assert a.decay == 0
    a.decay = 1
    assert isclose(a.decay, 1)
    a.decay = None
    assert a.decay == 0
    a.decay = False
    assert a.decay == 0
    with pytest.raises(ValueError):
        a.decay = -1
    a.optimized_learning = True
    a.decay = False
    assert a.decay == 0
    with pytest.raises(ValueError):
        a.decay = 1
    with pytest.raises(ValueError):
        a.decay = 3.14159265359
    a.optimized_learning = False
    a.decay = 1
    with pytest.raises(ValueError):
        a.optimized_learning = True
    a.decay = 2.7182818
    with pytest.raises(ValueError):
        a.optimized_learning = True

def test_mismatch_penalty():
    a = Agent()
    assert a.mismatch_penalty is None
    a.mismatch_penalty = 0
    assert a.mismatch_penalty == 0
    a.mismatch_penalty = 1
    assert isclose(a.mismatch_penalty, 1)
    a.mismatch_penalty = None
    assert a.mismatch_penalty is None
    a.mismatch_penalty = False
    assert a.mismatch_penalty is None
    with pytest.raises(ValueError):
        a.mismatch_penalty = -1

def test_default_utility():
    a = Agent()
    assert a.default_utility is None
    a.default_utility = 0
    assert a.default_utility == 0
    a.default_utility = 1
    assert isclose(a.default_utility, 1)
    a.default_utility = -10
    assert isclose(a.default_utility, -10)
    a.default_utility = None
    assert a.default_utility is None
    a.default_utility = False
    assert a.default_utility is None
    a.default_utility = lambda x: 1
    assert a.default_utility
    with pytest.warns(UserWarning):
        a.mismatch_penalty = 1
    a.mismatch_penalty = None
    a.default_utility = 10
    with pytest.warns(UserWarning):
        a.mismatch_penalty = 1
    a.mismatch_penalty = None
    a.default_utility = 0
    with pytest.warns(UserWarning):
        a.mismatch_penalty = 1
    a.default_utility = None
    assert a.mismatch_penalty == 1
    with pytest.warns(UserWarning):
        a.default_utility = 10
    with pytest.warns(UserWarning):
        a.default_utility = lambda x: 100
    with pytest.warns(UserWarning):
        a.default_utility = 0
    a = Agent(default_utility=0)
    results = set()
    for _ in range(3):
        c = a.choose(["a", "b", "c"])
        results.add(c)
        a.respond(-1)
    assert results == {"a", "b", "c"}

def test_advance():
    a = Agent()
    for i in range(2):
        assert a.time == 0
        assert a.advance(0) == 0
        assert a.time == 0
        assert a.advance() == 1
        assert a.time == 1
        assert a.advance(2) == 3
        assert a.time == 3
        with pytest.raises(ValueError):
            a.advance(-1)
        with pytest.raises(ValueError):
            a.advance(1.00000001)
        with pytest.raises(ValueError):
            a.advance("foo")
        assert a.time == 3
        a.reset()
    a.advance(target=10)
    assert a.time == 10
    a.advance()
    assert a.time == 11
    assert a.advance() == 12
    assert a.advance(1, 14) == 14
    assert a.advance(2, 15) == 16
    assert a.advance(1, 17) == 17
    with pytest.raises(ValueError):
        a.advance(target=11)
    with pytest.raises(ValueError):
        a.advance(target=-1)
    with pytest.raises(ValueError):
        a.advance(target=20.0001)
    with pytest.raises(ValueError):
        a.advance(target="not a number")

def test_choose_simple():
    choices = ["a", "b"]
    for d in (0.0, 0.1, 0.5, 1.0):
        for n in (0.0, 0.1, 0.25, 1.0):
            for t in (0.5, 1.0):
                a = Agent(noise=n, temperature=t, decay=d, default_utility=1)
                assert a.time == 0
                r1 = a.choose(choices)
                assert r1 in choices
                assert a.time == 1
                a.respond(0)
                assert a.time == 1
                r2 = a.choose(choices)
                assert r2 in choices
                assert r1 != r2
                assert a.time == 2
                a.respond(0.5)
                assert a.time == 2
                assert a.choose((choices + ["c"])) == "c"
                assert a.time == 3
                a.respond(2)
                assert a.choose() == "c"
    with pytest.raises(RuntimeError):
        a.choose(choices)
    a.respond(10000)
    assert a.choose() == "c"
    a.respond(10000)
    assert a.choose("bc") == "c"
    a.respond(10000)
    with pytest.raises(ValueError):
        a.choose(["a", ["c"]])
    with pytest.raises(ValueError):
        a.choose(["a", "b", None, "d"])
    a = Agent(temperature=1, noise=0)
    a.populate("A", 10)
    a.populate("B", 5)
    assert a.choose("AB") == "A"
    a.respond(0)
    choice, details = a.choose(details=True)
    assert choice == "B"
    assert len(details) == 2
    bd = details[0]
    assert bd["choice"] == "B" and isclose(bd["blended_value"], 5.0)
    p = bd["retrieval_probabilities"]
    assert len(p) == 1
    assert p[0]["utility"] == 5
    assert isclose(p[0]["retrieval_probability"], 1.0)
    bd = details[1]
    assert bd["choice"] == "A" and isclose(bd["blended_value"], 4.142135623730951)
    p = bd["retrieval_probabilities"]
    assert len(p) == 2
    assert p[0]["utility"] == 10
    assert isclose(p[0]["retrieval_probability"], 0.4142135623730951)
    assert p[0]["utility"] == 10 and isclose(p[0]["retrieval_probability"], 0.4142135623730951)
    assert p[1]["utility"] == 0 and isclose(p[1]["retrieval_probability"], 0.585786437626905)

def test_respond():
    a = Agent(temperature=1, noise=0)
    a.populate("A", 10)
    a.populate("B", 9)
    assert a.choose("AB") == "A"
    assert a.respond(0) is None
    assert a.choose() == "B"
    assert a.respond(0, "A") is None
    assert a.choose() == "B"
    assert isclose(a.respond().expectation, 9.0)
    assert a.choose() == "B"
    df = a.respond(None, "A")
    assert df._attributes["_decision"] == "A"
    assert isclose(df.expectation, 2.8019727339170046)
    insts = a.instances(None)
    assert insts[:-1] == [{'decision': 'A', 'outcome': 10, 'created': 0, 'occurrences': (0,)},
                          {'decision': 'B', 'outcome': 9, 'created': 0, 'occurrences': (0, 3)},
                          {'decision': 'A', 'outcome': 0, 'created': 1, 'occurrences': (1, 2)}]
    d = insts[-1]
    assert isclose(d["outcome"], 2.8019727339170046)
    del d["outcome"]
    assert d ==  {"decision": "A", "created": 4, "occurrences": (4,)}

def test_populate():
    a = Agent()
    assert len(a.instances(None)) == 0
    a.populate("a", 10)
    inst = a.instances(None)
    assert len(inst) == 1
    assert inst[0]["decision"] == "a" and inst[0]["outcome"] == 10 and inst[0]["created"] == 0
    a.populate("bcdef", 10)
    inst = a.instances(None)
    assert len(inst) == 6
    for i in range(6):
        assert inst[i]["decision"] in "abcdef" and inst[i]["outcome"] == 10 and inst[i]["created"] == 0
    for i in range(50):
        assert a.choose("abcdef") in "abcdef"
        a.respond(random.random() * 5)
    assert len(a.instances(None)) == 56
    a.populate("xyz", 30)
    a.populate("uvwx", 40, 22)
    inst = next(i for i in a.instances(None) if i["decision"] == "y")
    assert inst["outcome"] == 30 and inst["created"] == 50
    assert a.time == 50
    assert a.choose("uvwx") in "uvwx"
    with pytest.raises(ValueError):
        a.populate("uvwx", 1, a.time + 1)

def test_reset():
    a = Agent(default_utility=1, noise=0.37, decay=0.55)
    assert not a.optimized_learning
    assert isclose(a.noise, 0.37)
    assert isclose(a.decay, 0.55)
    assert a.time == 0
    assert len(a.instances(None)) == 0
    a.default_utility_populates = True
    a.choose("abcde")
    assert a.time == 1
    assert len(a.instances(None)) == 5
    a.respond(0)
    assert a.time == 1
    assert len(a.instances(None)) == 6
    a.reset()
    assert a.time == 0
    assert isclose(a.noise, 0.37)
    assert isclose(a.decay, 0.55)
    assert not a.optimized_learning
    assert len(a.instances(None)) == 0
    a.choose("abc")
    assert a.time == 1
    assert len(a.instances(None)) == 3
    a.reset()
    a.optimized_learning = True
    assert a.time == 0
    assert isclose(a.noise, 0.37)
    assert isclose(a.decay, 0.55)
    assert a.optimized_learning
    assert len(a.instances(None)) == 0
    a.choose("abcd")
    assert a.time == 1
    assert len(a.instances(None)) == 4
    a.default_utility_populates = False
    a.reset()
    assert a.time == 0
    assert a.optimized_learning
    assert isclose(a.noise, 0.37)
    assert isclose(a.decay, 0.55)
    assert len(a.instances(None)) == 0
    a.choose("abcde")
    assert a.time == 1
    assert len(a.instances(None)) == 0
    a.respond(0)
    assert a.time == 1
    assert len(a.instances(None)) == 1
    a.reset()
    a.optimized_learning = False
    a.default_utility_populates = True
    a.choose("abc")
    a.respond(0.01)
    a.choose("abc")
    a.respond(0.01)
    a.choose("abc")
    a.respond(0.01)
    assert a.time == 3
    assert len(a.instances(None)) == 6
    a.choose("abc")
    a.respond(1000)
    a.choose("abc")
    a.respond(1000)
    assert a.time == 5
    assert len(a.instances(None)) == 7
    a.reset(preserve_prepopulated=True)
    assert a.time == 0
    assert len(a.instances(None)) == 3
    a.reset(preserve_prepopulated=False)
    assert a.time == 0
    assert len(a.instances(None)) == 0

def test_random_choice():
    choices = list(range(5))
    results = [0]*len(choices)
    n = 5000
    for i in range(n):
        a = Agent(default_utility=1)
        results[a.choose(choices)] += 1
    for r in results:
        assert isclose(r / n, 1 / len(choices), rel_tol=0.1)

def test_alternate_choice():
    a = Agent(default_utility=10, noise=0.05)
    previous = a.choose([True, False])
    for i in range(20):
        assert a.time == i + 1
        a.respond(-10**i)
        assert a.time == i + 1
        assert a.choose([True, False]) != previous
        previous = not previous

def test_many_choices():
    a = Agent(temperature=1, noise=0)
    choices = list(range(100))
    for i in choices:
        a.populate([i], 1000 + (100 - i) * 0.001)
    for i in choices:
        assert a.choose(choices) == i
        a.respond(0)
    a.default_utility=1001
    assert a.choose(list(range(1000))) >= 100

SAFE_RISKY_PARTICIPANTS = 80
SAFE_RISKY_ROUNDS = 50

def safe_risky(noise=0.25, decay=0.5, temperature=None, optimized_learning=False, risky_wins=0.5):
    risky_chosen = 0
    a = Agent(noise=noise,
              decay=decay,
              temperature=temperature,
              optimized_learning=optimized_learning,
              default_utility=10)
    for p in range(SAFE_RISKY_PARTICIPANTS):
        a.reset()
        for r in range(SAFE_RISKY_ROUNDS):
            if a.choose(["safe", "risky"]) == "safe":
                a.respond(0)
            else:
                risky_chosen += 1
                a.respond(5 if random.random() < risky_wins else -5)
    return risky_chosen / (SAFE_RISKY_PARTICIPANTS * SAFE_RISKY_ROUNDS)

def test_safe_risky():
    # Note that tiny changes to the code could change the values being asserted.
    with randomseed():
        x = safe_risky()
        assert isclose(x, 0.25675)
        x = safe_risky(optimized_learning=True)
        assert isclose(x, 0.2215)
        x = safe_risky(decay=2)
        assert isclose(x, 0.1375)
        x = safe_risky(temperature=1, noise=0)
        assert isclose(x, 0.0865)
        x = safe_risky(risky_wins=0.6)
        assert isclose(x, 0.3445)
        x = safe_risky(risky_wins=0.4)
        assert isclose(x, 0.158)
        results = []
        results.append(safe_risky())
        results.append(safe_risky(optimized_learning=True))
        results.append(safe_risky(decay=2))
        results.append(safe_risky(temperature=1, noise=0))
        results.append(safe_risky(risky_wins=0.6))
        results.append(safe_risky(risky_wins=0.4))
        assert all(isclose(r, x) for r, x in zip(results, [0.224, 0.313, 0.13325, 0.161, 0.38125, 0.135]))

def form_choice(d):
    n = random.randrange(6)
    if n == 0:
        return d
    elif n == 1:
        return d
    elif n == 2:
        d["ignore-unused"] = 17
        return d
    elif n == 3:
        return [ d["button"], d["illuminated"] ]
    elif n == 4:
        return [ d["button"], d["illuminated"], "ignore-unused" ]
    elif n == 5:
        return ( d["button"], d["illuminated"] )
    else:
        return ( d["button"], d["illuminated"], "ignore-unused" )

def test_attributes():
    # Note that tiny changes to the code could change the values being asserted.
    with randomseed():
        left_chosen = 0
        illuminated_chosen = 0
        a = Agent(attributes=["button", "illuminated"], default_utility=5)
        left = { "button": "left" }
        right = { "button": "right" }
        for i in range(2000):
            left["illuminated"] = random.random() < 0.5
            right["illuminated"] = random.random() < 0.5
            formed_left = form_choice(left)
            if random.randrange(2):
                choice = a.choose([formed_left, form_choice(right)])
            else:
                choice = a.choose([form_choice(right), formed_left])
            illum = False
            if choice == formed_left:
                is_left = True
                if left["illuminated"]:
                    illum = True
            else:
                is_left = False
                if right["illuminated"]:
                    illum = True
            if is_left:
                left_chosen += 1
            if illum:
                illuminated_chosen += 1
            a.respond((1 if is_left else 2) * (2 if illum else 1))
        assert left_chosen > 200 and left_chosen < 800
        assert illuminated_chosen > 1200 and illuminated_chosen < 1800
        a = Agent(attributes=["attribute_1", "attribute_2"], default_utility=1)
        results = set()
        for i in range(3):
            results.add(tuple(a.choose([{"attribute_1": 1, "attribute_2": 2},
                                        {"attribute_1": 3},
                                        {"attribute_2": 4}])))
            a.respond(0)
        assert len(results) == 3
        a = Agent(attributes=["x"], default_utility=10)
        with pytest.raises(ValueError):
            a.choose([["not hashable"], 0])
        with pytest.raises(ValueError):
            a.choose([0, 1, 2, 1, 3])

def partial_matching_agent():
    a = Agent(temperature=1, noise=0, attributes=["button", "color", "size"], mismatch_penalty=5)
    a.populate([{"button": "a", "color": "red", "size": 5}], 100)
    a.populate([{"button": "b", "color": "blue", "size": 10}], 110)
    a.populate([{"button": "c", "color": "magenta", "size": 4}], 400)
    return a

def color_similarity(x, y):
    if x == y:
        return 1
    elif x == "magenta" or y == "magenta":
        return 0.9
    else:
        return 0.1

def test_partial_matching():
    a = partial_matching_agent()
    assert a.choose([{"button": "a", "color": "red", "size": 5},
                     {"button": "b", "color": "blue", "size": 10}])["button"] == "b"
    a = partial_matching_agent()
    a.similarity("button", lambda x, y: 1)
    a.similarity("color", color_similarity)
    a.similarity("size", positive_linear_similarity)
    assert a.choose([{"button": "a", "color": "red", "size": 5},
                     {"button": "b", "color": "blue", "size": 20}])["button"] == "a"
    a.respond(10)
    assert a.choose([{"button": "a", "color": "red", "size": 5},
                     {"button": "b", "color": "blue", "size": 20}])["button"] == "b"
    a = partial_matching_agent()
    a.similarity("button", lambda x, y: 1)
    a.similarity("color", color_similarity)
    a.similarity("size", positive_quadratic_similarity)
    assert a.choose([{"button": "a", "color": "red", "size": 5},
                     {"button": "b", "color": "blue", "size": 20}])["button"] == "b"
    a.respond(10)
    assert a.choose([{"button": "a", "color": "red", "size": 5},
                     {"button": "b", "color": "blue", "size": 20}])["button"] == "a"

def test_partial_activations():
    a = Agent(temperature=1, noise=0, attributes=["attr"])
    a.populate([{"attr": 1}, {"attr": 2}, {"attr": 3}], 0)
    a.populate(({"attr": 1}, {"attr": 2}, {"attr": 3}), 9)
    a.populate([{"attr": 1}], 4)
    a.populate([{"attr": 2}], 6)
    a.details = True
    c = a.choose([{"attr": 1}, {"attr": 2}, {"attr": 3}])
    assert c["attr"] == 2
    assert isclose(a.details[0][0]["blended"], 4.333333333333333)
    assert isclose(a.details[0][1]["blended"], 5)
    assert isclose(a.details[0][2]["blended"], 4.5)
    for d1 in a.details[0]:
        for d2 in d1["activations"]:
            assert d2.get("mismatch") is None
    a.respond(-20)
    a.mismatch_penalty = 5
    a.details.clear()
    a._memory._similarities.clear()
    c = a.choose([{"attr": 1}, {"attr": 2}, {"attr": 3}])
    assert c["attr"] == 3
    assert isclose(a.details[0][0]["blended"], 4.333333333333333)
    assert isclose(a.details[0][1]["blended"], -3.009431025426018)
    assert isclose(a.details[0][2]["blended"], 4.5)
    for d1 in a.details[0]:
        for d2 in d1["activations"]:
            assert d2.get("mismatch") is None
    a.respond(-20)
    a.similarity("attr", True)
    a.details.clear()
    c = a.choose([{"attr": 1}, {"attr": 2}, {"attr": 3}])
    assert c["attr"] == 1
    assert isclose(a.details[0][0]["blended"], 4.179723593644641)
    assert isclose(a.details[0][1]["blended"], -2.2435213562801173)
    assert isclose(a.details[0][2]["blended"], -6.775779766415437)
    for d1 in a.details[0]:
        for d2 in d1["activations"]:
            assert isclose(d2["mismatch"], 0) or isclose(d2["mismatch"], -5)
    a.respond(-20)
    a.similarity("attr", bounded_linear_similarity(-20, 10))
    a.details.clear()
    c = a.choose([{"attr": 1}, {"attr": 2}, {"attr": 3}])
    assert c["attr"] == 2
    assert isclose(a.details[0][0]["blended"], -4.348084167341177)
    assert isclose(a.details[0][1]["blended"], -4.191899625084612)
    assert isclose(a.details[0][2]["blended"], -4.3259594953443665)
    for d1 in a.details[0]:
        for d2 in d1["activations"]:
            assert (isclose(d2["mismatch"], 0)
                    or isclose(d2["mismatch"], -0.16666666666666663)
                    or isclose(d2["mismatch"], -0.33333333333333326))
    a.respond(0)
    a.noise = 0.25
    a.details.clear()
    c = a.choose([{"attr": 1}, {"attr": 2}, {"attr": 3}])
    v1 = a.details[0][0]["activations"]
    v2 = a.details[0][1]["activations"]
    v3 = a.details[0][2]["activations"]
    assert len(v1) == len(v2) and len(v1) == len(v3)

def test_insider():
    # Note that tiny changes to the code could change the value being asserted.
    with randomseed():
        x = insider.run()
        assert isclose(x, 0.6795)

def test_delayed_feedback():
    with randomseed():
        a = Agent(noise=0.1, decay=1)
        a.populate(["a"], 10)
        assert a.choose("a") == "a"
        dra = a.respond()
        assert not dra.is_resolved
        assert isclose(dra.outcome, 10)
        a.populate(["b"], 20)
        assert a.choose("ab") == "b"
        drb = a.respond()
        assert not dra.is_resolved
        assert isclose(dra.outcome, 10)
        assert isclose(dra.expectation, 10)
        assert not drb.is_resolved
        assert isclose(drb.outcome, 20)
        assert a.choose("ab") == "b"
        a.respond(-10000)
        assert a.choose("ab") == "a"
        a.respond(0)
        assert not dra.is_resolved
        assert isclose(dra.outcome, 10)
        assert not drb.is_resolved
        assert isclose(drb.outcome, 20)
        inst = a.instances(None)
        assert len(inst) == 4
        assert next(i for i in inst
                    if i["decision"]=="a" and isclose(i["outcome"],10) and i["created"]==0
                    and i["occurrences"]==(0,1))
        assert next(i for i in inst
                    if i["decision"]=="b" and isclose(i["outcome"],20) and i["created"]==1
                    and i["occurrences"]==(1,2))
        assert next(i for i in inst
                    if i["decision"]=="b" and isclose(i["outcome"],-10000) and i["created"]==3
                    and i["occurrences"]==(3,))
        assert next(i for i in inst
                    if i["decision"]=="a" and isclose(i["outcome"],0) and i["created"]==4
                    and i["occurrences"]==(4,))
        assert isclose(dra.update(15), 10)
        assert dra.is_resolved
        assert isclose(dra.outcome, 15)
        assert isclose(dra.expectation, 10)
        inst = a.instances(None)
        assert len(inst) == 5
        assert next(i for i in inst
                    if i["decision"]=="a" and isclose(i["outcome"],10) and i["created"]==0
                    and i["occurrences"]==(0,))
        assert next(i for i in inst
                    if i["decision"]=="b" and isclose(i["outcome"],20) and i["created"]==1
                    and i["occurrences"]==(1,2))
        assert next(i for i in inst
                    if i["decision"]=="b" and isclose(i["outcome"],-10000) and i["created"]==3
                    and i["occurrences"]==(3,))
        assert next(i for i in inst
                    if i["decision"]=="a" and isclose(i["outcome"],0) and i["created"]==4
                    and i["occurrences"]==(4,))
        assert next(i for i in inst
                    if i["decision"]=="a" and isclose(i["outcome"],15) and i["created"]==1
                    and i["occurrences"]==(1,))
        assert not drb.is_resolved
        assert isclose(drb.outcome, 20)
        assert isclose(dra.update(20), 15)
        assert dra.is_resolved
        assert isclose(dra.outcome, 20)
        assert isclose(dra.expectation, 10)
        assert not drb.is_resolved
        assert isclose(drb.outcome, 20)

def test_instances(tmp_path):
    a = Agent(default_utility=15, default_utility_populates=True)
    choices = "abcdefghijklm"
    for i in range(100):
        assert a.choose(choices)
        a.respond(random.random() * 8)
    assert len(a.instances(None)) == 100 + len(choices)
    p = tmp_path / "instances.txt"
    a.instances(file=p)
    s = p.read_text()
    assert re.search("decision.+outcome.+created.+occurrences", s)
    assert re.search(r"m.+15.+0.+\(0,\)", s)
    assert len(s.split("\n")) > 100 + len(choices)
    p = tmp_path / "instances.csv"
    a.instances(file=p, pretty=False)
    lines = p.read_text().split("\n")
    assert len(lines) > 100 + len(choices)

def test_details():
    a = Agent(temperature=1, noise=0, decay=10)
    a.details = True
    assert a.details == []
    a.populate([False], 10)
    a.populate([True], 20)
    assert a.choose([False, True])
    assert len(a.details) == 1 and len(a.details[0]) == 2
    a.respond(3)
    assert not a.choose([False, True])
    a.respond(15)
    assert len(a.details) == 2 and len(a.details[0]) == 2 and len(a.details[1]) == 2
    assert not a.details[0][0]["decision"]
    assert isclose(a.details[0][0]["blended"], 10.0)
    assert a.details[0][1]["decision"]
    assert isclose(a.details[0][1]["blended"], 20.0)
    assert not a.details[1][0]["decision"]
    assert isclose(a.details[1][0]["blended"], 10.0)
    assert a.details[1][1]["decision"]
    assert isclose(a.details[1][1]["blended"], 3.0165853658536586)
    assert not a.choose()
    assert a.respond(0, True) is None
    assert len(a.details) == 3 and len(a.details[-1]) == 2
    first, second = a.details[-1]
    assert not first["decision"] and isclose(first["blended"], 14.999915325994918)
    assert second["decision"] and isclose(second["blended"], 3.2897807667338075)
    old = a.details
    new = ["a"]
    a.details = new
    assert not a.choose([True, False])
    assert a.details[0] == "a"
    assert len(a.details) == 2

def test_trace(capsys):
    a = Agent(default_utility=10)
    a.choose("abcd")
    a.respond(5)
    assert len(capsys.readouterr().out) == 0
    a.trace = True
    a.choose("abcd")
    a.respond(4)
    assert re.search("decision.+base activation.+activation noise.+retrieval probability",
                     capsys.readouterr().out)
    x = capsys.readouterr().out
    assert a.trace
    a.trace = False
    assert not a.trace
    a.choose("abcd")
    a.respond(6)
    assert capsys.readouterr().out == x
    a.trace = True
    a.choose("abcd")
    a.respond(0)
    assert len(capsys.readouterr().out) > len(x)
    a.temperature = 1
    a.noise = 0
    a.choose("abcd")            # shouldn't raise error with zero noise

def test_positive_linear_similarity():
    assert isclose(positive_linear_similarity(1, 2), 0.5)
    assert isclose(positive_linear_similarity(2, 1), 0.5)
    assert isclose(positive_linear_similarity(1, 10), 0.09999999999999998)
    assert isclose(positive_linear_similarity(10, 100), 0.09999999999999998)
    assert isclose(positive_linear_similarity(1, 2000), 0.0004999999999999449)
    assert isclose(positive_linear_similarity(1999, 2000), 0.9995)
    assert isclose(positive_linear_similarity(1, 1), 1)
    assert isclose(positive_linear_similarity(0.001, 0.002), 0.5)
    assert isclose(positive_linear_similarity(10.001, 10.002), 0.9999000199960006)
    for i in range(40):
        n = 10 ** i
        assert isclose(positive_linear_similarity(2e-20 * n, 3e-20 * n), 0.6666666666666667)
        assert isclose(positive_linear_similarity(3e-20 * n, 2e-20 * n), 0.6666666666666667)
    with pytest.raises(ValueError):
        positive_linear_similarity(0, 1)
    with pytest.raises(ValueError):
        positive_linear_similarity(1, -1)
    with pytest.raises(ValueError):
        positive_linear_similarity(0, 0)
    with pytest.raises(TypeError):
        positive_linear_similarity("one", 1)
    with pytest.raises(TypeError):
        positive_linear_similarity(2, "one")

def test_positive_quadratic_similarity():
    assert isclose(positive_quadratic_similarity(1, 2), 0.25)
    assert isclose(positive_quadratic_similarity(2, 1), 0.25)
    assert isclose(positive_quadratic_similarity(1, 10), 0.009999999999999995)
    assert isclose(positive_quadratic_similarity(10, 100), 0.009999999999999995)
    assert isclose(positive_quadratic_similarity(1, 2000), 2.4999999999994493e-07)
    assert isclose(positive_quadratic_similarity(1999, 2000), 0.9990002500000001)
    assert isclose(positive_quadratic_similarity(1, 1), 1)
    assert isclose(positive_quadratic_similarity(0.001, 0.002), 0.25)
    assert isclose(positive_quadratic_similarity(10.001, 10.002), 0.9998000499880025)
    for i in range(40):
        n = 10 ** i
        assert isclose(positive_quadratic_similarity(2e-20 * n, 3e-20 * n), 0.44444444444444453)
        assert isclose(positive_quadratic_similarity(3e-20 * n, 2e-20 * n), 0.44444444444444453)
    with pytest.raises(ValueError):
        positive_quadratic_similarity(0, 1)
    with pytest.raises(ValueError):
        positive_quadratic_similarity(1, -1)
    with pytest.raises(ValueError):
        positive_quadratic_similarity(0, 0)
    with pytest.raises(TypeError):
        positive_quadratic_similarity("one", 1)
    with pytest.raises(TypeError):
        positive_quadratic_similarity(2, "one")

def test_bounded_linear_similarity():
    f = bounded_linear_similarity(-1, 1)
    assert isclose(f(0, 1), 0.5)
    assert isclose(f(-0.1, 0.1), 0.9)
    assert isclose(f(-1, 1), 0.0)
    assert isclose(f(0, 0), 1.0)
    assert isclose(f(0, sys.float_info.epsilon), 0.9999999999999999)
    with pytest.warns(UserWarning):
        assert isclose(f(-2, 1), 0.0)
    with pytest.warns(UserWarning):
        assert isclose(f(-1, 2), 0.0)
    with pytest.warns(UserWarning):
        assert isclose(f(-2, 0), 0.5)
    with pytest.warns(UserWarning):
        assert isclose(f(0, 2), 0.5)
    with pytest.raises(TypeError):
        f("Zero", 0)
    with pytest.raises(TypeError):
        f(0, "Zero")
    with pytest.raises(TypeError):
        f(None, 0)
    with pytest.raises(TypeError):
        f(None, None)
    f = bounded_linear_similarity(0, 100)
    for i in range(95):
        assert isclose(f(i, i+5), 0.95)
        assert isclose(f(i+5, i), 0.95)
    f = bounded_linear_similarity(-1000, -900)
    for i in range(56):
        assert isclose(f(-1000 + i, -1000 + i + 44), 0.56)
        assert isclose(f(-1000 + i + 44, -1000 + i), 0.56)
    with pytest.raises(TypeError):
        assert bounded_linear_similarity("zero", 1)
    with pytest.raises(TypeError):
        assert bounded_linear_similarity(0, "one")
    with pytest.raises(TypeError):
        assert bounded_linear_similarity(None, 1)
    with pytest.raises(TypeError):
        assert bounded_linear_similarity(0, None)
    with pytest.raises(ValueError):
        assert bounded_linear_similarity(1, -2)
    with pytest.raises(ValueError):
        assert bounded_linear_similarity(0, 0)

def test_bounded_quadratic_similarity():
    f = bounded_quadratic_similarity(-1, 1)
    assert isclose(f(0, 1), 0.25)
    assert isclose(f(-0.1, 0.1), 0.81)
    assert isclose(f(-1, 1), 0.0)
    assert isclose(f(0, 0), 1.0)
    assert isclose(f(0, sys.float_info.epsilon), 0.9999999999999998)
    with pytest.warns(UserWarning):
        assert isclose(f(-2, 1), 0.0)
    with pytest.warns(UserWarning):
        assert isclose(f(-1, 2), 0.0)
    with pytest.warns(UserWarning):
        assert isclose(f(-2, 0), 0.25)
    with pytest.warns(UserWarning):
        assert isclose(f(0, 2), 0.25)
    with pytest.raises(TypeError):
        f("Zero", 0)
    with pytest.raises(TypeError):
        f(0, "Zero")
    with pytest.raises(TypeError):
        f(None, 0)
    with pytest.raises(TypeError):
        f(None, None)
    f = bounded_quadratic_similarity(0, 100)
    for i in range(95):
        assert isclose(f(i, i+5), 0.9025)
        assert isclose(f(i+5, i), 0.9025)
    f = bounded_quadratic_similarity(-1000, -900)
    for i in range(56):
        assert isclose(f(-1000 + i, -1000 + i + 44), 0.31360000000000005)
        assert isclose(f(-1000 + i + 44, -1000 + i), 0.31360000000000005)
    with pytest.raises(TypeError):
        assert bounded_quadratic_similarity("zero", 1)
    with pytest.raises(TypeError):
        assert bounded_quadratic_similarity(0, "one")
    with pytest.raises(TypeError):
        assert bounded_quadratic_similarity(None, 1)
    with pytest.raises(TypeError):
        assert bounded_quadratic_similarity(0, None)
    with pytest.raises(ValueError):
        assert bounded_quadratic_similarity(1, -2)
    with pytest.raises(ValueError):
        assert bounded_quadratic_similarity(0, 0)

def test_fixed_noise():
    a = Agent("a", mismatch_penalty=1)
    a.similarity("a", lambda x, y: 1)
    a.populate([{"a": i} for i in [1, 2]], 0)
    a.choose([[1],[2]])
    a.respond(0)
    def run_one(same):
        c, d = a.choose(details=True)
        d = {bd["choice"] if isinstance(bd["choice"], int) else bd["choice"][0]:
             set(rp["retrieval_probability"] for rp in bd["retrieval_probabilities"]) for bd in d}
        a.respond(0)
        if same:
            assert d[1] == d[2]
        else:
            assert d[1] != d[2]
    run_one(False)
    a.fixed_noise = True
    run_one(True)
    a = Agent(mismatch_penalty=1)
    a.similarity([], lambda x, y: 1)
    a.populate([1, 2], 0)
    a.choose([1,2])
    a.respond(0)
    run_one(False)
    a.fixed_noise = True
    run_one(True)

def test_similarity():
    def mismatch_value(agent, decision, chunk_name):
        activations = next(d for d in agent.details[-1]
                           if d["decision"] == decision)["activations"]
        return next(d for d in activations if d["name"] == chunk_name).get("mismatch")
    a = Agent(mismatch_penalty=1)
    a.similarity(function=True)
    Chunk._name_counter = 0
    a.populate([1], 0)
    a.populate([2], 0)
    a.details = True
    a.choose([1, 2])
    a.respond(0)
    assert isclose(mismatch_value(a, 1, "0000"), 0)
    assert isclose(mismatch_value(a, 1, "0001"), -1)
    assert isclose(mismatch_value(a, 2, "0000"), -1)
    assert isclose(mismatch_value(a, 2, "0001"), 0)
    a.similarity(weight=0.5)
    a.choose()
    a.respond(0)
    assert isclose(mismatch_value(a, 1, "0000"), 0)
    assert isclose(mismatch_value(a, 1, "0001"), -0.5)
    assert isclose(mismatch_value(a, 2, "0000"), -0.5)
    assert isclose(mismatch_value(a, 2, "0001"), 0)
    a.similarity([], lambda x, y: 1 - abs(x - y))
    a.choose()
    a.respond(0)
    assert isclose(mismatch_value(a, 1, "0000"), 0)
    assert isclose(mismatch_value(a, 1, "0001"), -0.5)
    assert isclose(mismatch_value(a, 2, "0000"), -0.5)
    assert isclose(mismatch_value(a, 2, "0001"), 0)
    a.similarity(None, None, 2)
    a.choose()
    a.respond(0)
    assert isclose(mismatch_value(a, 1, "0000"), 0)
    assert isclose(mismatch_value(a, 1, "0001"), -2)
    assert isclose(mismatch_value(a, 2, "0000"), -2)
    assert isclose(mismatch_value(a, 2, "0001"), 0)
    a.similarity([])
    a.choose()
    a.respond(0)
    assert mismatch_value(a, 1, "0000") is None
    assert mismatch_value(a, 2, "0001") is None

def test_discrete_blend():
    a = Agent("a b", temperature=1, noise=0)
    a.populate([{"a": 1, "b": 1}], 10)
    a.populate([{"a": 2, "b": 1}], 15)
    a.populate([{"a": 1, "b": 2}], 20)
    a.populate([{"a": 2, "b": 2}], 25)
    a.advance()
    a.populate([{"a": 2, "b": 2}], 25)
    a.advance()
    b, p = a.discrete_blend("a", {"b": 1})
    assert b in {1, 2}
    assert isclose(p[1], 0.5)
    assert isclose(p[2], 0.5)
    d = defaultdict(int)
    for _ in range(200):
        b, p = a.discrete_blend("a", {"b": 1})
        d[b] += 1
    assert d[1] > 10
    assert d[2] > 10
    a.mismatch_parameter=1
    a.similarity("a", True)
    b, p = a.discrete_blend("b", {"a": 2})
    assert b == 2
    assert isclose(p[1], 0.29289321881345254)
    assert isclose(p[2], 0.7071067811865476)

def test_index():
    a = Agent("x y")
    assert set(a._memory.index) == {"x", "y"}
    a.similarity("y", True)
    assert set(a._memory.index) == {"x"}
    a.populate({"x": 1, "y": 2}, 0)
    a.similarity("y")
    assert set(a._memory.index) == {"x"}
    a.reset()
    assert set(a._memory.index) == {"x", "y"}
    a.similarity("y x", True)
    assert set(a._memory.index) == set()

def test_aggregate():
    CARDS = [(n, c) for n in range(1, 5) for c in "ryg"]
    WINNER = (1, "r")
    # Note that tiny changes to the code could change the values being asserted.
    with randomseed():
        a = Agent(["n", "color"], mismatch_penalty=1)
        a.populate(CARDS, 3.2)
        a.similarity(["n"], bounded_linear_similarity(1, 4))
        assert a.aggregate_details is None
        a.aggregate_details = True
        for p in range(100):
            a.reset(True)
            succeeded = False
            for r in range(8):
                choice = a.choose(CARDS)
                payoff = int(choice[1] == WINNER[1])
                if choice[0] == WINNER[0]:
                    payoff += 2
                elif abs(choice[0] - WINNER[0]) == 1:
                    payoff += 1
                if payoff == 3 and not succeeded:
                    succeeded = True
                a.respond(payoff)
        agg = a.aggregate_details
        assert agg.shape == (49180, 12)
        assert (list(agg.columns.values) ==
                ['iteration', 'time', 'choice', 'utility', 'option', 'blended_value', 'retrieval_probability',
                 'activation', 'base_level_activation', 'activation_noise', 'mismatch', 'n.similarity'])
        def compcol(colname, vals, approx=True):
            assert all(map((isclose if approx else __eq__), random.choices(agg.loc[:, colname], k=12), vals))
        compcol("time", [8, 2, 1, 8, 2, 5, 1, 5, 6, 6, 2, 2], False)
        compcol("choice", [(4, 'y'), (1, 'y'), (3, 'y'), (3, 'r'), (1, 'y'), (4, 'y'),
                           (4, 'r'), (2, 'y'), (1, 'g'), (2, 'r'), (1, 'r'), (3, 'y')],
                False)
        compcol("utility", [2.0, 3.2, 3.2, 3.2, 2.0, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2])
        compcol("option", [(1, 'r'), (1, 'r'), (1, 'y'), (2, 'r'), (4, 'g'), (1, 'g'),
                           (3, 'g'), (2, 'g'), (1, 'r'), (2, 'y'), (4, 'g'), (4, 'g')],
                False)
        compcol("blended_value",
                [2.021612840195993, 2.6902198237681407, 1.6405438566798678, 3.2000000000000006,
                 3.0068505933731045, 1.6333426037125909, 3.2, 0.23433458965941562, 3.133741402642344,
                 2.4666357733346214, 3.0742098285628225, 2.270098816718304])
        compcol("retrieval_probability",
                [0.007345545287107604, 0.4697221026713707, 0.14848217961656338, 0.0769918015437266,
                 0.020650233305125002, 0.011389819296455399, 0.700977500920395, 0.03718489199834408,
                 0.005343248158715588, 0.001313776850371587, 0.01355438512875705, 0.3864726452559221])
        compcol("activation",
                [-1.6019323881245635, -2.2421036877870337, -0.664349263698488, -0.604465686750661,
                 -0.46715941300391617, -0.6413420002009673, -0.4805733379506427, -1.5039784100777807,
                 -0.6182989854058272, -2.7000726287931593, -2.340731038447224, -0.9620280803896725])
        compcol("base_level_activation",
                [0.0, -0.3465735902799726, -0.3465735902799726, -0.8047189562170503, -0.8047189562170503,
                 -0.9729550745276566, -0.9729550745276566, -0.8958797346140275, -0.3465735902799726,
                 -0.6931471805599453, -0.5493061443340549, 0.0])
        compcol("activation_noise",
                [-1.073248337921886, -0.11873070862491725, -0.7452154776196454, 0.08958047959071777,
                 0.13214943521333966, -0.035577656298113665, -0.9657671733017608, -0.5717495182730941,
                 1.3722697306935068, 0.45348597893575326, -0.14214170314902874, 0.016820647397743353])
        compcol("mismatch",
                [-0.33333333333333326, -0.33333333333333326, 0.0, -0.33333333333333326, -0.33333333333333326,
                 0.0, -0.6666666666666666, 0.0, -1.0, -0.33333333333333326, -0.33333333333333326,
                 -0.33333333333333326])
        compcol("n.similarity",
                [0.6666666666666667, 0.33333333333333337, 0.0, 0.6666666666666667, 0.33333333333333337,
                 0.6666666666666667, 0.6666666666666667, 0.33333333333333337, 0.6666666666666667,
                 0.33333333333333337, 0.6666666666666667, 0.6666666666666667])
    last_rand = 0
    def rand2(reset=False):
        global last_rand
        if reset:
            last_rand = 0
            last_rand = (19 * last_rand + 3) % (3 * 19 + 1)
            return last_rand % 2
    def cmp(v1, v2):
        v1 = list(v1)
        v2 = list(v2)
        assert len(v1) == len(v2)
        for x, y in zip(v1, v2):
            assert isclose(x, y)
    def deterministic_model(n, **params):
        rand2(True)
        agent = Agent(temperature=1, noise=0, **params)
        agent.aggregate_details = True
        for p in range(3):
            agent.reset()
            if not params.get("default_utility"):
                agent.populate(["s"], 2.2)
            agent.populate(["r"], 2.201)
            for r in range(6 - p):
                if agent.choose(["s", "r"]) == "s":
                    agent.respond(1)
                else:
                    agent.respond(2 if rand2() else 0)
        agg = agent.aggregate_details
        assert agg.shape == (n, 10)
        cmp(agg["activation_noise"], [0]*n)
        cmp(agg["activation"], agg["base_level_activation"])
        return agent
    a = deterministic_model(51)
    agg = a.aggregate_details
    assert list(agg["iteration"]) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                      2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    assert list(agg["time"]) == [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6,
                                 6, 6, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                                 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    assert list(agg["choice"]) == ['r', 'r', 's', 's', 's', 's', 's', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 'r',
                                   'r', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 'r', 'r', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 's', 's']
    cmp(agg["retrieval_probability"],
        [1.0, 1.0, 1.0, 0.4142135623730951,
         0.585786437626905, 0.36602540378443865, 0.6339745962155614, 0.4494897427831781,
         0.550510257216822, 0.22654091966098644, 0.7734590803390136, 0.4641016151377546,
         0.5358983848622454, 0.1637143175276618, 0.8362856824723383, 0.4721359549995794,
         0.5278640450004206, 0.12786907869062805, 0.872130921309372, 0.47722557505166113,
         0.5227744249483388, 1.0, 1.0, 1.0, 0.4142135623730951, 0.585786437626905,
         0.36602540378443865, 0.6339745962155614, 0.4494897427831781, 0.550510257216822,
         0.22654091966098644, 0.7734590803390136, 0.4641016151377546, 0.5358983848622454,
         0.1637143175276618, 0.8362856824723383, 0.4721359549995794, 0.5278640450004206,
         1.0, 1.0, 1.0, 0.4142135623730951, 0.585786437626905, 0.36602540378443865,
         0.6339745962155614, 0.4494897427831781, 0.550510257216822, 0.22654091966098644,
         0.7734590803390136, 0.4641016151377546, 0.5358983848622454])
    a = deterministic_model(51, mismatch_penalty=1.5)
    agg = a.aggregate_details
    assert list(agg["iteration"]) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                      2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    assert list(agg["time"]) == [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6,
                                 6, 6, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                                 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    assert list(agg["choice"]) == ['r', 'r', 's', 's', 's', 's', 's', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 'r',
                                   'r', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 'r', 'r', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 's', 's']
    cmp(agg["retrieval_probability"],
        [1.0, 1.0, 1.0, 0.4142135623730951, 0.585786437626905, 0.36602540378443865, 0.6339745962155614,
         0.4494897427831781, 0.550510257216822, 0.22654091966098644, 0.7734590803390136, 0.4641016151377546,
         0.5358983848622454, 0.1637143175276618, 0.8362856824723383, 0.4721359549995794, 0.5278640450004206,
         0.12786907869062805, 0.872130921309372, 0.47722557505166113, 0.5227744249483388, 1.0, 1.0, 1.0,
         0.4142135623730951, 0.585786437626905, 0.36602540378443865, 0.6339745962155614, 0.4494897427831781,
         0.550510257216822, 0.22654091966098644, 0.7734590803390136, 0.4641016151377546, 0.5358983848622454,
         0.1637143175276618, 0.8362856824723383, 0.4721359549995794, 0.5278640450004206, 1.0, 1.0, 1.0,
         0.4142135623730951, 0.585786437626905, 0.36602540378443865, 0.6339745962155614, 0.4494897427831781,
         0.550510257216822, 0.22654091966098644, 0.7734590803390136, 0.4641016151377546, 0.5358983848622454])
    a = deterministic_model(48, default_utility=2.2, default_utility_populates=True)
    agg = a.aggregate_details
    cmp(agg["retrieval_probability"],
        [1.0, 1.0, 0.4142135623730951, 0.585786437626905, 0.36602540378443865, 0.6339745962155614,
         0.4494897427831781, 0.550510257216822, 0.22654091966098644, 0.7734590803390136, 0.4641016151377546,
         0.5358983848622454, 0.1637143175276618, 0.8362856824723383, 0.4721359549995794, 0.5278640450004206,
         0.12786907869062805, 0.872130921309372, 0.47722557505166113, 0.5227744249483388, 1.0, 1.0,
         0.4142135623730951, 0.585786437626905, 0.36602540378443865, 0.6339745962155614, 0.4494897427831781,
         0.550510257216822, 0.22654091966098644, 0.7734590803390136, 0.4641016151377546, 0.5358983848622454,
         0.1637143175276618, 0.8362856824723383, 0.4721359549995794, 0.5278640450004206, 1.0, 1.0,
         0.4142135623730951, 0.585786437626905, 0.36602540378443865, 0.6339745962155614, 0.4494897427831781,
         0.550510257216822, 0.22654091966098644, 0.7734590803390136, 0.4641016151377546, 0.5358983848622454])
    a = deterministic_model(36, default_utility=2.2, default_utility_populates=False)
    agg = a.aggregate_details
    cmp(agg["retrieval_probability"],
        [1.0, 0.4142135623730951, 0.585786437626905, 1.0, 0.4494897427831781, 0.550510257216822, 1.0,
         0.4641016151377546, 0.5358983848622454, 1.0, 0.22966848451216434, 0.7703315154878356, 1.0,
         0.2612674239835993, 0.7387325760164007, 1.0, 0.4142135623730951, 0.585786437626905, 1.0, 0.4494897427831781,
         0.550510257216822, 1.0, 0.4641016151377546, 0.5358983848622454, 1.0, 0.22966848451216434, 0.7703315154878356,
         1.0, 0.4142135623730951, 0.585786437626905, 1.0, 0.4494897427831781, 0.550510257216822, 1.0,
         0.4641016151377546, 0.5358983848622454])
    a = deterministic_model(51, optimized_learning=True)
    agg = a.aggregate_details
    assert list(agg["iteration"]) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                      2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    assert list(agg["time"]) == [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6,
                                 6, 6, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                                 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    assert list(agg["choice"]) == ['r', 'r', 's', 's', 's', 's', 's', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 'r',
                                   'r', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 'r', 'r', 's', 's', 's', 's',
                                   's', 's', 's', 's', 's', 's', 's']
    cmp(agg["retrieval_probability"],
        [1.0, 1.0, 1.0, 0.41421356237309503, 0.585786437626905, 0.36602540378443865, 0.6339745962155614,
         0.4494897427831781, 0.5505102572168219, 0.2612038749637415, 0.7387961250362586, 0.4641016151377546,
         0.5358983848622454, 0.20521309615767264, 0.7947869038423273, 0.4721359549995794, 0.5278640450004206,
         0.16952084719853724, 0.8304791528014627, 0.47722557505166113, 0.5227744249483388, 1.0, 1.0, 1.0,
         0.41421356237309503, 0.585786437626905, 0.36602540378443865, 0.6339745962155614, 0.4494897427831781,
         0.5505102572168219, 0.2612038749637415, 0.7387961250362586, 0.4641016151377546, 0.5358983848622454,
         0.20521309615767264, 0.7947869038423273, 0.4721359549995794, 0.5278640450004206, 1.0, 1.0, 1.0,
         0.41421356237309503, 0.585786437626905, 0.36602540378443865, 0.6339745962155614, 0.4494897427831781,
         0.5505102572168219, 0.2612038749637415, 0.7387961250362586, 0.4641016151377546, 0.5358983848622454])
    assert a.aggregate_details is not agg
    a.choose(["s", "r"])
    assert a.aggregate_details.shape == (55, 10)
    a.aggregate_details = True
    assert a.aggregate_details.empty
    a.populate(["s"], 2.2)
    a.populate(["r"], 2.201)
    assert a.aggregate_details.empty
    a.reset()
    a.populate(["s", "r"], 2.2)
    for r in range(3):
        a.choose(["s", "r"])
        a.respond(2)
    assert a.aggregate_details.shape == (9, 10)


def test_plot():
    def run_model(agent):
        agent.populate(["s", "r"], 3.2)
        for p in range(100):
            agent.reset(True)
            for r in range(50):
                choice = agent.choose(["s", "r"])
                if choice == "r":
                    agent.respond(3 if random.random() < 1/4 else 0)
                else:
                    agent.respond(1)
    a = Agent()
    run_model(a)
    with pytest.raises(RuntimeError):
        a.plot("choice")
    a = Agent()
    a.aggregate_details = True
    run_model(a)
    a.plot("choice", filename="plots/choice.png")
    a.plot("choice", filename="plots/risky.png", exclude=["s"], title="Fraction choosing risky", earliest=3)
    a.plot("bv", filename="plots/bv.png", max=3)
    a.plot("probability", filename="plots/probability.png")
    a.plot("probability", filename="plots/safe_prob.png", exclude=["s", "not present"], earliest=3)
    a.plot("activation", filename="plots/activation.png", latest=40)
    a.plot("baselevel", filename="plots/baselevel.png", include=["r", "not present"], limits=(-2, 1))
    a.plot("baselevel", filename="plots/empty.png", max=2.5, min=2.5)

    def sim(x, y):
        return 1 - abs(x - y) / 8
    OPTIONS = [("black", 2), ("black", 4), ("black", 6), ("black", 8),("black", 10),
               ("gold", 1), ("gold", 3), ("gold", 5), ("gold", 7), ("gold", 9)]
    a = Agent(["c", "n"], mismatch_penalty=1)
    a.similarity(["n"], sim)
    a.populate(OPTIONS, 150)
    a.aggregate_details = True
    for p in range(80):
        a.reset(True)
        options = list(OPTIONS)
        for r in range(40):
            choice = a.choose(options)
            if choice[0] == "black":
                a.respond(choice[1]**2 + (2 if random.random() < 0.3 else -2))
            else:
                a.respond(100 - choice[1]**2 + (3 if random.random() < 0.7 else -3))
            if round(r + p / 40) % 7 == 0:
                options.pop()
    a.plot("choice", filename="plots/pm_choice.png")
    a.plot("bv", max=140, legend=False, filename="plots/pm_bv.png")
    a.plot("mismatch", min=90, max=140, filename="plots/pm_mismatch.png")
    a.plot("n.similarity", filename="plots/pm_similarity.png")
    with pytest.raises(ValueError):
        a.plot("c.similarity")
    with pytest.raises(ValueError):
        a.plot("foo")
    with pytest.raises(ValueError):
        a.plot("")
