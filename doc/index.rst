.. Copyright 2014-2022 Carnegie Mellon University

PyIBL version 5.0

.. contents::


Introduction
============

PyIBL is a Python implementation of a subset of Instance Based Learning Theory (IBLT) [#f1]_.
It is made and distributed by the
`Dynamic Decision Making Laboratory <http://ddmlab.com>`_
of
`Carnegie Mellon University <http://cmu.edu>`_
for making computational cognitive models supporting research
in how people make decisions in dynamic environments.

Typically PyIBL is used by creating an experimental framework in the Python programming language, which
uses one or more PyIBL :class:`Agent` objects. The framework
then asks these agents to make decisions, and informs the agents of the results of
those decisions. The framework, for example, may be strictly algorithmic, may interact with human
subjects, or may be embedded in a web site.

PyIBL is a library, or module, of `Python <http://www.python.org/>`_ code,  useful for creating
Python programs; it is not a stand alone application.
Some knowledge of Python programming is essential for using it.


.. [#f1] Cleotilde Gonzalez, Javier F. Lerch and Christian Lebiere (2003),
         `Instance-based learning in dynamic decision making,
         <http://www.sciencedirect.com/science/article/pii/S0364021303000314>`_
         *Cognitive Science*, *27*, 591-635. DOI: 10.1016/S0364-0213(03)00031-4.

Installing PyIBL
================

The latest version of PyIBL can be downloaded and installed from PyPi with ``pip``:

  .. parsed-literal:: pip install pyibl

Use of a virtual environment for Python, such as ``venv`` or Anaconda is recommended.

PyIBL requires Python version 3.8 or later.

Note that PyIBL is simply a Python module, a library, that is run as part of a larger
Python program. To build and run models using PyIBL you do need to do
some Python programming. If you're new to Python, a good place to
start learning it is `The Python Tutorial <https://docs.python.org/3.8/tutorial/>`_.
To write and run a Python program you need to create and edit Python
source files, and then run them. If you are comfortable using the command
line, you can simply create and edit the files in your favorite text editor,
and run them from the command line. Many folks, though, are happier using
a graphical Integrated Development Environment (IDE).
`Many Python IDEs are available <https://wiki.python.org/moin/IntegratedDevelopmentEnvironments>`_.
One  is
`IDLE <https://docs.python.org/3.8/library/idle.html>`_,
which comes packaged with Python itself, so if you installed Python
you should have it available.

This document is also available at `http://pyibl.ddmlab.com/ <http://pyibl.ddmlab.com/>`_.

Mailing list and reporting bugs
-------------------------------

Users and others interested in PyIBL are encouraged to subscribe to the
`PyIBL Users mailing list <https://lists.andrew.cmu.edu/mailman/listinfo/pyibl-users>`_.

The PyIBL community is small enough that this serves both as an announcement list and a discussion list.
Announcements of new versions of PyIBL will be made here, and it is a good place to ask questions about
PyIBL or to solicit other help in using it.

It is also a good place to report any bugs, or suspected bugs, in PyIBL. If, however, you would
prefer not to report them to this list, please feel free to instead send them to
`Don Morrison <mailto:dfm2@cmu.edu>`_.



Tutorial
========

Likely the easiest way to get started with PyIBL is by looking at some examples of its use.
While much of what is in this chapter should be understandable even without much knowledge of Python, to
write your own models you'll need to know how to write Python code.
If you are new to Python, a good place to start may be
the `Python Tutorial <https://docs.python.org/3.8/tutorial/>`_.

Note that when running the examples below yourself the results may differ in detail because
IBL models are stochastic.


A first example of using PyIBL
------------------------------

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
    Python 3.9.0 | packaged by conda-forge | (default, Oct 14 2020, 22:59:50)
    [GCC 7.5.0] on linux
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

    >>> from pyibl import Agent, similarity
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

We could use this information to, for example, to write detailed log
files of many iterations of our model while it runs over thousands of
iterations.


Reference
=========

.. automodule:: pyibl

.. autoclass:: Agent

   .. autoattribute:: name

   .. autoattribute:: attributes

   .. automethod:: choose

   .. automethod:: respond

   .. automethod:: populate

   .. autoattribute:: default_utility

   .. autoattribute:: default_utility_populates

   .. automethod:: reset

   .. autoattribute:: time

   .. automethod:: advance

   .. autoattribute:: noise

   .. autoattribute:: decay

   .. autoattribute:: temperature

   .. autoattribute:: mismatch_penalty

   .. automethod:: similarity

   .. autoattribute:: optimized_learning

   .. automethod:: instances

   .. autoattribute:: details

   .. autoattribute:: trace

   .. automethod:: discrete_blend

   .. autoattribute:: fixed_noise

.. autoclass:: DelayedResponse

   .. autoattribute:: is_resolved

   .. autoattribute:: outcome

   .. autoattribute:: expectation

   .. automethod:: update

.. autofunction:: positive_linear_similarity

.. autofunction:: positive_quadratic_similarity

.. autofunction:: bounded_linear_similarity

.. autofunction:: bounded_quadratic_similarity


Internals
=========

PyIBL is built on top of `PyACTUp <http://halle.psy.cmu.edu/pyactup/>`_, a Python implementation of
a portion of `ACT-R <http://act-r.psy.cmu.edu/>`_'s declarative memory. This chapter describes the computations
underlying decisions made by PyIBL, which are mostly carried out in the underlying PyACTUp code.

The fundamental unit of memory in PyIBL is an instance (a "chunk" in PyACTUp), which combines the
attributes of a choice with the result it led to, along with timing data.


Activation
----------

A fundamental part of retrieving an instance from an agent's memory is computing the activation of that instance,
a real number describing
how likely it is to be recalled, based on how frequently and recently it has been experienced by the :class:`Agent`, and how well it
matches the attributes of what is to be retrieved.

The activation, :math:`A_{i}` of instance *i* is a sum of three
components,

  .. math:: A_{i} = B_{i} + \epsilon_{i} + P_{i}

the base-level activation, the activation noise, and the partial matching correction.

Base-level activation
~~~~~~~~~~~~~~~~~~~~~

The base-level activation, :math:`B_{i}`, describes the frequency and recency of the instance *i*,
and depends upon the :attr:`decay` parameter of the :class:`Agent`, *d*. In the normal case, when the
agent's :attr:`optimized_learning` parameter is ``False``, the base-level activation is computed using
the amount of time that has elapsed since each of the past appearances of *i*, which in the following
are denoted as the various :math:`t_{ij}`.

  .. math:: B_{i} = \ln(\sum_{j} t_{ij}^{-d})

If the agent's :attr:`optimized_learning` parameter is ``True`` an approximation is used instead, often less taxing of
computational resources. It is particularly useful if the same instances are expected to be seen many times, and assumes
that repeated experiences of the various instances are distributed roughly evenly over time.
Instead of using the times of all the past occurrences of *i*, it uses :math:`L_i`, the amount of time since
the first appearance of *i*, and :math:`n_i`, a count of the number of times *i* has appeared.

  .. math:: B_{i} = \ln(\frac{n_i}{1 - d}) - d \ln(L_i)

The ``optimized_learning`` parameter may also be set to a positive integer. This specifies a number of most recent
reinforcements of a chunk to be used to compute the base-level activation in the normal way, with the contributions
of any older than those approximated using a formula similar to the preceding.

Note that setting the ``decay`` parameter to ``None`` disables the computation of base-level
activation. That is, the base-level component of the total activation is zero in this case.

Activation noise
~~~~~~~~~~~~~~~~

The activation noise, :math:`\epsilon_{i}`, implements the stochasticity of retrievals from an agent's memory.
It is sampled from a logistic distribution centered on zero. An :class:`Agent` has a scale
parameter, :attr:`noise`, for this distribution. It is resampled each time the activation is computed.

Note that setting the ``noise`` parameter to zero results in supplying
no noise to the activation. This does not quite make operation of
PyIBL deterministic, since retrievals of instances with the same
activations are resolved randomly.

Partial Matching
~~~~~~~~~~~~~~~~

If the agent's :attr:`mismatch_penalty` parameter is ``None``, the partial matching correction, :math:`P_{i}`, is zero.
Otherwise :math:`P_{i}` depends upon the similarities of the attributes of the instance to those attributes
being sought in the retrieval and the value of the `mismatch_penalty` parameter.

PyIBL represents similarities as real numbers between zero and one, inclusive, where two values being completely similar, identical,
has a value of one; and being completely dissimilar has a value of zero; with various other degrees of similarity being
positive, real numbers less than one.

How to compute the similarity of two instances is determined by the programmer, using the
method :meth:`similarity`.
A function is supplied to this method to be applied to values of the
attributes of given names, this function returning a similarity value. In addition, the ``similarity`` method
can assign a weight, :math:`\omega`, to these attributes, allowing the mismatch contributions of multiple attributes
to be scaled with respect to one another. If not explicitly supplied this weight defaults to one.

A function is supplied to this method to be applied to values of the
attributes of given names, this function returning a similarity value. In addition, the ``similarity`` method
can assign a weight, :math:`\omega`, to these slots, allowing the mismatch contributions of multiple slots
to be scaled with respect to one another. If not explicitly supplied this weight defaults to one.

If the ``mismatch`` parameter has real value :math:`\mu`, the similarity of attribute *k* of
instance *i* to the desired
value of that attribute :math:`S_{ik}`, and the similarity weight of attribute *k* is :math:`\omega_{k}`,
the partial matching correction is

  .. math:: P_{i} = \mu \sum_{k} \omega_{k} (S_{ik} - 1)

The value of :math:`\mu` should be positive, so :math:`P_{i}` is negative, and increasing dissimilarities
reduce the total activation, scaled by the value of :math:`\mu`.

Attributes for which no similarity function is defined are always matched exactly, non-matching instances not
being considered at all.


Blending
--------

Once the activations of all the relevant instances have been computed, they are used to compute
a blended value of the utility, an average of the utilities of those instances weighted by a function
of the instances' activations, the probability of retrieval.

A parameter, the :attr:`temperature`, or :math:`\tau`, is used in constructing this blended value.
In PyIBL the value of this parameter is by default the :attr:`noise` parameter used for activation noise,
multiplied by :math:`\sqrt{2}`. However it can be set independently of the ``noise``, if preferred.

If *m* is the set of instances matching the criteria, and, for :math:`i \in m`, the activation
of instance *i* is :math:`a_{i}`, we define a weight, :math:`w_{i}`, used to compute a probability of retrieval
describing the contribution *i* makes to the blended value

  .. math:: w_{i} = e^{a_{i} / \tau}

The probabilty of retrieval is simply the weight divided by the sum of all the weights for a given potential outcome.
If :math:`u_{i}` is the utility, that is the outcome value, stored in instance *i*,
the  blended value, *BV*, is then

  .. math:: BV =\, \sum_{i \in m}{\, \frac{w_{i}}{\sum_{j \in m}{w_{j}}} \; u_{i}}


Changes to PyIBL
================

Changes between versions 4.2 and 5.0
------------------------------------

* PyIBL now requires Python 3.8 or later.
* Substantial changes were made to the internal representations of agents and instances enabling faster IBL computations in
  many use cases involving large numbers of instances.
* :class:`Agent` attributes can now be any non-empty string, and the arguments to the :class:`Agent` constructor
  are now in a different order.
* Changed the arguments to :meth:`choose`, :meth:`populate` and :meth:`forget`.
* Similarity functions are now per-:class:`Agent` instead of global, and are set with the :meth:`similarity` method.
* Similarities can now have weights, also set with the :meth:`similarity` method, allowing easier balancing
  of the contributions of multiple attributes.
* The :meth:`advance` method has been added to the API.
* The :meth:`choose2` method has been replaced by an optional argument to :meth:`choose`.
* The :meth:`populate_at` method has been replaced by an optional argument to :meth:`populate`.
* There is a new method :meth:`discrete_blend` useful for creating models using a different paradigm
  from PyIBL’s usual :meth:`choose`/:meth:`respond` cycle.
* It is now possible to set :attr:`optimized_learning` as an :class:`Agent` parameter in the usual way, instead
  of as an argument to :meth:`reset`. In addition, :attr:`optimized_learning` can now take positive integers
  as its value, enabling a mixed mode of operation.
* The default value of :attr:`default_utility_populates` is now ``False``, and it can be set at :class:`Agent`
  creation time with an argument to the constructor.
* There is a new :class:`Agent` property, :attr:`fixed_noise`, allowing a variant noise generation scheme
  for unusual models.
* General tidying and minor bug fixes.

When upgrading existing version 4.x models to version 5.0 or later some syntactic changes will nearly always
have to be made. In particular, PyIBL no longer abuses Python’s keyword arguments, and lists of choices now need
to be passed to :meth:`choose` and :meth:`populate`, which now also take their arguments in a different order.
In simple cases this is as easy as surrounding the formerly trailing arguments by square bracket, and swapping
the result two arguments. For more complex cases it may be necessary to pass a list of dictionaries.
For example, what in version 4.x would have been expressed as

.. code-block:: python

    a.populate(10, "red", "blue")
    a.choose("red", "blue")

could be expressed in version 5.0 as

.. code-block:: python

    a.populate(["red", "blue"], 10)
    a.choose(["red", "blue"])

If you are using partial matching you will also have to replace calls to the :func:`similarity` function by
the :class:`Agent`’s :meth:`similarity` method. This method also takes slightly different arguments than
the former function.
For example, what in version 4.x would have been expressed as

.. code-block:: python

    similarity(cubic_similarity, "weight", "volume")

could be expressed in version 5.0 as

.. code-block:: python

    a.similarity(["weight", "volume"], cubic_similarity)


Changes between versions 4.2 and  4.2.0.1
-----------------------------------------

* PyIBL is now distributed via PyPi and need no longer be downloaded from the DDMLab website.


Changes between versions 4.1 and  4.2
-------------------------------------

* The :meth:`choose2` method has been added to the API.
* The :meth:`respond` method now takes a second, optional argument.
* There is more flexability possible when partially matching attributes.
* PyIBL now requires Pythonn verison 3.7 or later.
* General tidying and minor bug fixes.


Changes between versions 4.0 and 4.1
------------------------------------

* The API for :class:`DelayedFeedback` has been changed.
* The :meth:`reset()` now has an additional, optional argument, *preserve_prepopulated*.
* Some minor bug fixes.


Changes between versions 3.0 and 4.0
------------------------------------

* Situations and SituationDecisions are no longer needed. Choices are now ordinary
  Python objects, such as dicts and lists.
* The overly complex logging mechanism was a rich source of confusion and bugs. It
  has been eliminated, and replaced by a simpler mechanism, :attr:`details`, which
  facilitates the construction of rich log files in whatever forms may be desired.
* Populations were rarely used, badly understood and even when they
  were used were mostly just used to facilitate logging from multiple
  agents; in version 4.0 populations have been eliminated, though they may come
  back in a different form in a future version of PyIBL.
* Methods and attributes are now uniformly spelled in ``snake_case`` instead of ``camelCase``.
* Many attributes of Agents can now be specified when they are created.
* Similarities are now shared between Agents, by attribute name, rather than being
  specific to an Agent.
* Several common similarity functions are predefined.
* The current :attr:`time` can now be queried.
* Delayed feedback is now supported.
* PyIBL is now built on top of `PyACTUp <http://halle.psy.cmu.edu/pyactup/>`_.
* Some bugs have been fixed, and things have been made generally tidier internally.


Changes between versions 2.0 and 3.0
------------------------------------

* Similarity and partial matching are now implemented.
* SituationDecisions have changed completely, and are no longer created by an Agent.
* Logging has changed substantially: there can be multiple, differently configured
  logs; it is now possible to have per-Agent logs, not just Population-wide logs;
  and logging configuration now controls not just which columns are shown, but
  the order in which they appear.
* Default values of noise and decay are now 0.25 and 0.5, respectively, matching
  oral common practice in ACT-R, instead of ACT-R's out of the box defaults, which
  are rarely useful.
* General internal tidying

  .. warning::
      Note that version 3.0 was never publicly released though
      preliminary internal development versions of it it were used for a
      variety of experiments, both within the DDMLab and elsewhere.

Changes between versions 1.0 and 2.0
------------------------------------

* Agents are now publicly visible objects that can be passed around and moved from
  one Population to another. The API has completely changed so that you no longer
  cite an agent by name in a Population.
* Options presented to Agents are no longer merely decisions, but include situations as well.
* Logging is configured with strings rather than constants.
* Logging can now be configured to include or exclude unused options and instances.
* Bug fixes, particularly in logging.
* Better documentation.
* General internal tidying
