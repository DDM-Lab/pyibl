Tutorial
========

.. _tutorial:

Likely the easiest way to get started with PyIBL is by looking at some examples of its use.
While much of what is in this chapter should be understandable even without much knowledge of Python, to
write your own models you'll need to know how to write Python code.
If you are new to Python, a good place to start may be
the `Python Tutorial <https://docs.python.org/3.8/tutorial/>`_.

Note that when running the examples below yourself the results may differ in detail because
IBL models are stochastic.


A first example
---------------

In the code blocks that follow, lines the user has typed begin with any of the three prompts,

.. code-block::

    $
    >>>
    ...

Other lines are printed by Python or some other command.

First we launch Python, and make PyIBL available to it. While the output here was
captured in a Linux distribution and virtual environment in which you launch Python version 3.8 by typing ``python``,
your installation my differ and you may launch it with ``python3``, ``py``, or something else
entirely; or start an interactive session in a completely different way using a
graphical IDE.

.. code-block::

    $ python
    Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from pyibl import Agent

Next we create an :class:`Agent`, named ``'My Agent'``.

.. code-block:: python

    >>> a = Agent(name="My Agent")
    >>> a
    <Agent My Agent 140441174486264>

We  have to tell the agent what do if we ask it to choose between options
it has never previously experienced. One way to do this is to set a default
by setting the agent's :attr:`Agent.default_utility` property.

.. code-block:: python

    >>> a.default_utility = 10.0

Now we can ask the agent to choose between two options, that we'll just describe
using two strings. When you try this yourself you may get the opposite answer
as the IBLT theory is stochastic, which is particularly
obvious in cases like this where there is no reason yet to prefer one answer to the other.

.. code-block:: python

    >>> a.choose(["The Green Button", "The Red Button"])
    'The Green Button'

Now return a response to the model. We'll supply 1.0.

.. code-block:: python

    >>> a.respond(1.0)

Because that value is significantly less than the default utility when we ask
the agent to make the same choice again, we expect it with high
probability to pick the other button.

.. code-block:: python

    >>> a.choose(["The Green Button", "The Red Button"])
    'The Red Button'

We'll give it an even lower utility than we did the first one.

.. code-block:: python

    >>> a.respond(-2.0)

If we stick with these responses the model will tend to favor the first button selected.
Again, your results may differ in detail because of randomness.

.. code-block:: python

    >>> a.choose(["The Green Button", "The Red Button"])
    'The Green Button'
    >>> a.respond(1.0)
    >>> a.choose(["The Green Button", "The Red Button"])
    'The Red Button'
    >>> a.respond(-2.0)
    >>> a.choose(["The Green Button", "The Red Button"])
    'The Green Button'
    >>> a.respond(1.0)
    >>> a.choose(["The Green Button", "The Red Button"])
    'The Green Button'
    >>> a.respond(1.0)
    >>> a.choose(["The Green Button", "The Red Button"])
    'The Green Button'
    >>> a.respond(1.0)
    >>> a.choose(["The Green Button", "The Red Button"])
    'The Green Button'
    >>> a.respond(1.0)

But doing this by hand isn't very useful for modeling.
Instead, let's write a function that asks the model to make
this choice, and automates the reply.

.. code-block:: python

    >>> def choose_and_respond():
    ...     result = a.choose(["The Green Button", "The Red Button"])
    ...     if result == "The Green Button":
    ...         a.respond(1.0)
    ...     else:
    ...         a.respond(-2.0)
    ...     return result
    ...
    >>> choose_and_respond()
    'The Green Button'

Let's ask the model to make this choice a thousand times, and see how many
times it picks each button. But let's do this from a clean slate. So, before
we run it, we'll call :meth:`reset` to clear the agent's memory.

.. code-block:: python

    >>> a.reset()
    >>> results = { "The Green Button" : 0, "The Red Button" : 0 }
    >>> for _ in range(1000):
    ...     results[choose_and_respond()] += 1
    ...
    >>> results
    {'The Red Button': 11, 'The Green Button': 989}

As we expected the model prefers the green button, but because of randomness, does
try the red one occasionally.

Now let's add some other choices. We'll make a more complicated function
that takes a dictionary of choices and the responses they generate,
and see how they do. This will make use of a bit more Python.
The default utility is still 10, and so long
as the responses are well below that we can reasonably expect the first
few trials to sample them all before favoring those that give the best
results; but after the model gains more experience, it will favor whatever
color or colors give the highest rewards.

.. code-block:: python

    >>> def choose_and_respond(choices):
    ...     result = a.choose(choices)
    ...     a.respond(responses[result])
    ...     return result
    ...
    >>> a.reset()
    >>> responses = { "green": -5, "blue": 0, "yellow": -4,
    ...               "red": -6, "violet": 0 }
    ...
    >>> choices = list(responses.keys())
    >>> choices
    ['green', 'blue', 'yellow', 'red', 'violet']
    >>> results = { k: 0 for k in choices }
    >>> results
    {'green': 0, 'blue': 0, 'yellow': 0, 'red': 0, 'violet': 0}
    >>> for _ in range(5):
    ...     results[choose_and_respond(choices)] += 1
    ...
    >>> results
    {'green': 1, 'blue': 1, 'yellow': 1, 'red': 1, 'violet': 1}
    >>> for _ in range(995):
    ...     results[choose_and_respond(choices)] += 1
    ...
    >>> results
    {'green': 10, 'blue': 488, 'yellow': 8, 'red': 8, 'violet': 486}

The results are as we expected.


Multiple agents
---------------

A PyIBL model is not limited to using just one agent. It can use as many as
the modeler wishes. For this example we'll have ten players competing for rewards.
Each player, at each turn, will pick either ``'safe'`` or ``'risky'``.
Any player picking ``'safe'`` will always receive 1 point. All those
players picking ``'risky'`` will share 7 points divided evenly between them; if fewer
than seven players pick ``'risky'`` those who did will do better than
if they had picked ``'safe'``, but if more than seven players pick ``'risky'``
they will do worse.

.. code-block:: python

    >>> from pyibl import Agent
    >>> agents = [ Agent(name=n, default_utility=20)
    ...            for n in "ABCDEFGHIJ" ]
    >>> def play_round():
    ...     choices = [ a.choose(['safe', 'risky']) for a in agents ]
    ...     risky = [ a for a, c in zip(agents, choices) if c == 'risky' ]
    ...     reward = 7 / len(risky)
    ...     for a in agents:
    ...         if a in risky:
    ...             a.respond(reward)
    ...         else:
    ...             a.respond(1)
    ...     return (reward, "".join([ a.name for a in risky ]))

Here's what running it for ten rounds looks like.

.. code-block:: python

    >>> for _ in range(10):
    ...     print(play_round())
    ...
    (1.4, 'BDFHI')
    (1.4, 'ACEGJ')
    (1.75, 'DFGI')
    (1.4, 'BDEHJ')
    (0.875, 'ABCEFGHI')
    (0.7777777777777778, 'ABCDFGHIJ')
    (1.4, 'ACEFG')
    (1.75, 'BHIJ')
    (1.0, 'ACDEGHJ')

By just looking at a small sample of runs we can't really discern any
interesting patterns. Instead we'll write a function that runs the
agents many times, and gathers some statistics. We'll work out how
many agents pick risky, on average, by counting the length of the
second value returned by ``play_round()``; how many times each of the
agents picked risky; and what the aggregate payoff was to each agent.
And then run it for 1,000 rounds.

Note that before running it we get a clean slate by calling each agent's
``reset`` method. And for the payoffs we round the results to one decimal
place, as Python by default would be showing them to about 16 decimal
places, and we don't need that kind of precision.

.. code-block:: python

    >>> from statistics import mean, median, mode
    >>> from itertools import count
    >>> def run_agents(rounds):
    ...     for a in agents:
    ...         a.reset()
    ...     by_round = []
    ...     by_agent = [0]*len(agents)
    ...     agent_payoffs = [0]*len(agents)
    ...     for _ in range(rounds):
    ...         payoff, chose_risky = play_round()
    ...         by_round.append(len(chose_risky))
    ...         for a, i in zip(agents, count()):
    ...             if a.name in chose_risky:
    ...                 by_agent[i] += 1
    ...                 agent_payoffs[i] += payoff
    ...             else:
    ...                 agent_payoffs[i] += 1
    ...     print(mean(by_round), median(by_round), mode(by_round))
    ...     print(by_agent)
    ...     print([float(f"{p:.1f}") for p in agent_payoffs])
    ...
    >>> run_agents(1000)
    6.408 7.0 7
    [856, 283, 681, 851, 313, 230, 874, 706, 820, 794]
    [1106.2, 1001.0, 1056.5, 1097.7, 1004.9, 1001.8, 1102.2, 1052.7, 1092.1, 1076.9]

Note that this time when we ran it seven of the agents chose risky over two thirds of the
time, but three, b, e and f, chose it less than one third of the time, but all received
about the same reward over the course of 1,000 rounds, just a little better than
if they'd all always chosen safe.

Let's run it for a few more 1,000 round blocks.

.. code-block:: python

    >>> run_agents(1000)
    6.483 6.0 6
    [335, 884, 630, 472, 165, 875, 857, 706, 886, 673]
    [1007.9, 1091.8, 1029.9, 1007.6, 1000.2, 1100.9, 1080.3, 1051.5, 1103.7, 1043.2]
    >>> run_agents(1000)
    6.476 7.0 7
    [323, 318, 267, 299, 888, 847, 834, 902, 912, 886]
    [1005.1, 1003.8, 1001.4, 1001.0, 1088.2, 1078.6, 1063.1, 1094.0, 1098.0, 1090.7]
    >>> run_agents(1000)
    6.455 6.0 6
    [525, 572, 716, 558, 666, 707, 828, 641, 502, 740]
    [1031.6, 1030.3, 1067.6, 1034.4, 1051.9, 1075.7, 1112.5, 1048.9, 1026.9, 1065.3]
    >>> run_agents(1000)
    6.408 7.0 7
    [856, 283, 681, 851, 313, 230, 874, 706, 820, 794]
    [1106.2, 1001.0, 1056.5, 1097.7, 1004.9, 1001.8, 1102.2, 1052.7, 1092.1, 1076.9]

We see that a similar pattern holds, with a majority of the agents, when seen
over the full 1,000 rounds, having largely favored a risky
strategy, but a minority, again over the full 1,000 rounds, having favored a safe strategy.
But which agents these are, of course, varies from block to block; and, perhaps, if we
looked at more local sequences of decisions, we might see individual agent's strategies
shifting over time.


Attributes
----------

The choices an agent decides between are not limited to atomic entities
as we've used in the above. They can be structured using "attributes."
Such attributes need to be declared when the agent is created.

As a concrete example, we'll have our agent decide which of two buttons,
``'left'`` or ``'right'``, to push. But one of these buttons will be
illuminated. Which is illuminated at any time is decided randomly, with
even chances for either. Pushing the left button earns a base reward of
1, and the right button of 2; but when a button is illuminated its reward
is doubled.

We'll define our agent to have two attributes, ``"button"`` and ``"illuminatted"``.
The first is which button, and the second is whether or not that button is illumiunated.
In this example the the ``"button"`` value is the decision to be made, and
``"illuminatted"`` value represents the context, or situation, in which this decision is being made.

We'll start by creating an agent, and two choices, one for each button.

.. code-block:: python

    >>> from pyibl import Agent
    >>> from random import random
    >>> a = Agent(["button", "illuminated"], default_utility=5)
    >>> left = { "button": "left", "illuminated": False }
    >>> right = { "button": "right", "illuminated": False }

While we've created them both with the button un-illuminated, the code
that actually runs the experiment will turn one of them on, randomly.

.. code-block:: python

    >>> def push_button():
    ...     if random() <= 0.5:
    ...         left["illuminated"] = True
    ...     else:
    ...         left["illuminated"] = False
    ...     right["illuminated"] = not left["illuminated"]
    ...     result = a.choose([left, right])
    ...     reward = 1 if result["button"] == "left" else 2
    ...     if result["illuminated"]:
    ...         reward *= 2
    ...     a.respond(reward)
    ...     return result
    ...
    >>> push_button()
    {'button': 'right', 'illuminated': True}

Now we'll ``reset`` the agent, and then run it 2,000 times, counting how many times each button
is picked, and how many times an illuminated button is picked.

.. code-block:: python

    >>> a.reset()
    >>> results = {'left': 0, 'right': 0, True: 0, False: 0}
    >>> for _ in range(2000):
    ...     result = push_button()
    ...     results[result["button"]] += 1
    ...     results[result["illuminated"]] += 1
    ...
    >>> results
    {'left': 518, 'right': 1482, True: 1483, False: 517}


As we might have expected the right button is favored, as are illuminated ones, but
since an illuminated left is as good as a non-illuminated right neither completely
dominates.


Partial matching
----------------

In the previous examples experience from prior experiences only
applied if the prior decisions, or their contexts, matched exactly the
ones being considered for the current choice. But often we want to
choose the option that most closely matches, though not necessarily
exactly, for some definition of "closely." To do this we define a
similarity function for those attributes we want to partially match,
and specify a ``mismatch_penalty`` parameter.

In this example there will be a continuous function, ``f()``, that maps
a number between zero and one to a reward value. At each round the model
will be passed five random numbers between zero and one, and be asked to
select the one that it expects will give the greatest reward. We'll start
by defining an agent that expects choices to have a single attribute, ``n``.

.. code-block:: python

    >>> from pyibl import Agent
    >>> from random import random
    >>> import math
    >>> import sys
    >>> a = Agent(["n"])

We'll define a similarity function for that attribute, a function of two variables, two different values
of the attribute to be compared. When the attribute
values are the same the value should be 1, and when they are maximally
dissimilar, 0. The similarity function we're supplying is scaled linearly, and its
value ranges from 0, if one of its arguments is 1 and the other is 0, and otherwise scales up
to 1 when they are equal. So, for example, 0.31 and 0.32 have a large similarity, 0.99, but
0.11 and 0.93 have a small similarity, 0.18.

.. code-block:: python

    >>> a.similarity(["n"], lambda x, y: 1 - abs(x - y))

The :attr:`mismatch_penalty` is a non-negative number that says how much to
penalize past experiences for poor matches to the options currently
under consideration. The larger its value, the more mismatches
are penalized. We'll experiment with different values of the ``mismatch_penalty``
in our model

Let's define a function that will run our model, with the number of
iterations, the ``mismatch_penalty``, and the reward function supplied as parameters.
Note that we reset the agent at the beginning of this function.
We then supply one starting datum for the model to use, the value of the reward
function when applied to zero. After asking the agent to choose one of five,
randomly assigned values, our ``run_model`` function will work out which would have
given the highest reward, and keep track of how often the model did make that choice.
We'll round that fraction of correct choices made down to two decimal places to
make sure it is displayed nicely.

.. code-block:: python

    >>> def run_model(trials, mismatch, f):
    ...     a.reset()
    ...     a.mismatch_penalty = mismatch
    ...     a.populate([{"n": 0}], f(0))
    ...     number_correct = 0
    ...     fraction_correct = []
    ...     for t in range(trials):
    ...         options = [ {"n": random()} for _ in range(5) ]
    ...         choice = a.choose(options)
    ...         best = -sys.float_info.max
    ...         best_choice = None
    ...         for o in options:
    ...             v = f(o["n"])
    ...             if o == choice:
    ...                 a.respond(v)
    ...             if v > best:
    ...                 best = v
    ...                 best_choice = o
    ...         if choice == best_choice:
    ...             number_correct += 1
    ...         fraction_correct.append(float(f"{number_correct / (t + 1):.2f}"))
    ...     return fraction_correct

For our reward function we'll define a quadratic function that has a single peak of value 5 when called on 0.72, and
drops off on either side, down to 2.4 when called on 0 and down to 4.6 when called on 1.

.. code-block:: python

    >>> def f(x):
    ...    return 5 * (1 - math.pow(x - 0.72, 2))

Let's first run it with a mismatch penalty of zero, which means all values
will be considered equally good, and the result will simply be random guessing.

.. code-block:: python

    >>> run_model(100, 0, f)
    [0.0, 0.0, 0.0, 0.25, 0.2, 0.17, 0.14, 0.25, 0.22, 0.2, 0.18,
     0.25, 0.31, 0.29, 0.27, 0.31, 0.29, 0.28, 0.26, 0.25, 0.24, 0.23,
     0.22, 0.21, 0.2, 0.19, 0.19, 0.18, 0.17, 0.2, 0.19, 0.19, 0.18,
     0.18, 0.17, 0.17, 0.19, 0.18, 0.18, 0.17, 0.17, 0.17, 0.16, 0.16,
     0.16, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13,
     0.12, 0.14, 0.16, 0.15, 0.15, 0.15, 0.15, 0.16, 0.16, 0.15, 0.17,
     0.16, 0.16, 0.16, 0.16, 0.17, 0.17, 0.16, 0.16, 0.17, 0.17, 0.17,
     0.18, 0.18, 0.19, 0.19, 0.18, 0.18, 0.19, 0.19, 0.19, 0.18, 0.18,
     0.18, 0.18, 0.19, 0.18, 0.18, 0.18, 0.19, 0.2, 0.2, 0.2, 0.2, 0.2]

As we can see, it looks like random guessing, getting things right only about 20% of the time.

Now let's try it with a mismatch penalty of 1, which won't pay too much attention to
how closely the values match those we've seen before, but will pay a little bit of attention to it.

.. code-block:: python

    >>> run_model(100, 1, f)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14, 0.12, 0.11, 0.1, 0.18, 0.25,
     0.31, 0.29, 0.27, 0.31, 0.29, 0.33, 0.32, 0.3, 0.29, 0.32, 0.35,
     0.33, 0.36, 0.35, 0.37, 0.36, 0.34, 0.33, 0.35, 0.34, 0.33, 0.32,
     0.34, 0.36, 0.35, 0.34, 0.36, 0.35, 0.37, 0.36, 0.35, 0.34, 0.36,
     0.37, 0.36, 0.35, 0.37, 0.36, 0.35, 0.37, 0.38, 0.39, 0.4, 0.41,
     0.4, 0.41, 0.41, 0.4, 0.39, 0.4, 0.41, 0.41, 0.42, 0.42, 0.42,
     0.43, 0.42, 0.41, 0.42, 0.42, 0.42, 0.42, 0.41, 0.42, 0.42, 0.41,
     0.41, 0.41, 0.41, 0.4, 0.4, 0.4, 0.41, 0.41, 0.4, 0.4, 0.39, 0.39,
     0.4, 0.4, 0.4, 0.4, 0.41, 0.42, 0.41, 0.42, 0.41, 0.42]

While it started out guessing, since it had only minimal information, as it
learns more the model does much better, reaching correct answers about 40% of the time,
twice as good a random.

If we use a much larger mismatch penalty, 30, we'll see an even greater improvement,
converging on being correct about 90% of the time.

.. code-block:: python

    >>> run_model(100, 30, f)
    [0.0, 0.0, 0.33, 0.5, 0.6, 0.5, 0.57, 0.62, 0.67, 0.6, 0.55, 0.58,
     0.62, 0.64, 0.6, 0.62, 0.65, 0.67, 0.68, 0.7, 0.71, 0.68, 0.7,
     0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.77, 0.78, 0.79, 0.79,
     0.8, 0.81, 0.81, 0.82, 0.82, 0.82, 0.83, 0.83, 0.84, 0.84, 0.84,
     0.85, 0.85, 0.85, 0.86, 0.86, 0.86, 0.87, 0.87, 0.87, 0.87, 0.88,
     0.88, 0.88, 0.88, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.9,
     0.9, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.9, 0.9,
     0.9, 0.9, 0.9, 0.89, 0.89, 0.89, 0.89, 0.9, 0.9, 0.9, 0.9, 0.9,
     0.9, 0.9, 0.9, 0.9, 0.91, 0.9, 0.9, 0.9, 0.9, 0.9]


Inspecting the model's internal state and computations
------------------------------------------------------

Sometimes, possibly for debugging, possibly for writing detailed log files,
and possibly for making unusual models, we want to be able to see what's
going on inside PyIBL. Several tools are provided to facilitate this.

The :meth:`instances` method show's all instances currently in an agent's memory.

Consider this simple, binary choice model, that selects between a safe choice,
always returning 1, and a risky choice which returns 2 fifty percent of the time,
and 0 otherwise.

.. code-block:: python

    >>> a = Agent(default_utility=20)
    >>> def run_once():
    ...     if a.choose(["safe", "risky"]) == "safe":
    ...         a.respond(1)
    ...     elif random() <= 0.5:
    ...         a.respond(2)
    ...     else:
    ...         a.respond(0)


If we run it once, and then look at its memory we see three instances, two
that were created using the ``default_utility``, and one actually experienced.
As usual, if you run this yourself, it may differ in detail since PyIBL models
are stochastic.

.. code-block:: python

    >>> run_once()
    >>> a.instances()
    +----------+---------+---------+-------------+
    | decision | outcome | created | occurrences |
    +----------+---------+---------+-------------+
    |   safe   |    20   |    0    |     [0]     |
    |  risky   |    20   |    0    |     [0]     |
    |  risky   |    2    |    1    |     [1]     |
    +----------+---------+---------+-------------+

Let's run it ten more times and look again.

.. code-block:: python

    >>> for _ in range(10):
    ...     run_once()
    ...
    >>> a.instances()
    +----------+---------+---------+------------------+
    | decision | outcome | created |   occurrences    |
    +----------+---------+---------+------------------+
    |   safe   |    20   |    0    |       [0]        |
    |  risky   |    20   |    0    |       [0]        |
    |  risky   |    0    |    1    |    [1, 8, 10]    |
    |   safe   |    1    |    2    | [2, 5, 7, 9, 11] |
    |  risky   |    2    |    3    |    [3, 4, 6]     |
    +----------+---------+---------+------------------+

There are now five different instances, but all the actually
experienced ones have been reinforced two or more times.

If we want to see how PyIBL uses these values when computing a next
iteration we can turn on tracing in the agent.

.. code-block:: python

    >>> a.trace = True
    >>> run_once()

    safe → 1.6981119704016256
    +------+----------+---------+------------------+---------+---------------------+----------------------+---------------------+-----------------------+
    |  id  | decision | created |   occurrences    | outcome |   base activation   |   activation noise   |   total activation  | retrieval probability |
    +------+----------+---------+------------------+---------+---------------------+----------------------+---------------------+-----------------------+
    | 0132 |   safe   |    0    |       [0]        |    20   | -1.2424533248940002 |  0.6743933445745868  | -0.5680599803194134 |  0.036742735284296085 |
    | 0135 |   safe   |    2    | [2, 5, 7, 9, 11] |    1    |  1.0001744608971734 | -0.41339471775300435 |  0.586779743144169  |   0.9632572647157039  |
    +------+----------+---------+------------------+---------+---------------------+----------------------+---------------------+-----------------------+

    risky → 0.23150913644216178
    +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+-----------------------+
    |  id  | decision | created | occurrences | outcome |   base activation   |   activation noise   |   total activation  | retrieval probability |
    +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+-----------------------+
    | 0133 |  risky   |    0    |     [0]     |    20   | -1.2424533248940002 | -0.2948777661505001  | -1.5373310910445004 | 0.0009256525969768487 |
    | 0134 |  risky   |    1    |  [1, 8, 10] |    0    |  0.4111940833223344 | 0.48087038114494485  |  0.8920644644672793 |   0.8925763051517108  |
    | 0136 |  risky   |    3    |  [3, 4, 6]  |    2    | 0.09087765648075839 | 0.049537460190590174 | 0.14041511667134857 |   0.1064980422513124  |
    +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+-----------------------+

From this we see PyIBL computing blended values for the two options,
safe and risky, of 1.6981 and 0.2315, respectively. For the former, it
computed the activation of two relevant chunks, resulting in retrieval
probabilities it used to combine the possible outcomes of 20 and 1,
though heavily discounting the former because it's activation is so
low, because of decay. Similarly for the risky choice, though with
three instances reflecting three outcomes in the agent's memory.

To gain programmatic access to this data we can use the :attr:`details` of an agent.
Here we run the model one more time and print the result details.

.. code-block:: python

    >>> from pprint import pp
    >>> a.trace = False
    >>> a.details = True
    >>> run_once()
    >>> pp(a.details)
    [[{'decision': 'safe',
       'activations': [{'name': '0132',
                        'creation_time': 0,
                        'attributes': (('_utility', 20), ('_decision', 'safe')),
                        'reference_count': 1,
                        'references': [0],
                        'base_level_activation': -1.2824746787307684,
                        'activation_noise': 0.2343700698730416,
                        'activation': -1.0481046088577268,
                        'retrieval_probability': 0.0010536046644722481},
                       {'name': '0135',
                        'creation_time': 2,
                        'attributes': (('_utility', 1), ('_decision', 'safe')),
                        'reference_count': 6,
                        'references': [2, 5, 7, 9, 11, 12],
                        'base_level_activation': 1.184918357959952,
                        'activation_noise': 0.1904030286066936,
                        'activation': 1.3753213865666456,
                        'retrieval_probability': 0.9989463953355276}],
       'blended': 1.0200184886249728},
      {'decision': 'risky',
       'activations': [{'name': '0133',
                        'creation_time': 0,
                        'attributes': (('_utility', 20), ('_decision', 'risky')),
                        'reference_count': 1,
                        'references': [0],
                        'base_level_activation': -1.2824746787307684,
                        'activation_noise': -0.04867278847092464,
                        'activation': -1.331147467201693,
                        'retrieval_probability': 0.006544349628104761},
                       {'name': '0134',
                        'creation_time': 1,
                        'attributes': (('_utility', 0), ('_decision', 'risky')),
                        'reference_count': 3,
                        'references': [1, 8, 10],
                        'base_level_activation': 0.2724966041059383,
                        'activation_noise': 0.16088543930698337,
                        'activation': 0.43338204341292164,
                        'retrieval_probability': 0.9624144271846764},
                       {'name': '0136',
                        'creation_time': 3,
                        'attributes': (('_utility', 2), ('_decision', 'risky')),
                        'reference_count': 3,
                        'references': [3, 4, 6],
                        'base_level_activation': 0.027153555019573457,
                        'activation_noise': -0.8079194823991546,
                        'activation': -0.7807659273795811,
                        'retrieval_probability': 0.03104122318721878}],
       'blended': 0.1929694389365328}]]

We could use this information, for example, to write detailed log
files of many iterations of our model while it runs over thousands of
iterations.
