# Copyright 2014-2024 Carnegie Mellon University

"""PyIBL is an implementation of a subset of Instance Based Learn Theory (IBLT). The
principle class is :class:`Agent`, an instance of which is a cognitive entity learning and
making decisions based on its experience from prior decisions, primarily by calls to its
:meth:`choose` and :meth:`respond` methods. The decisions an agent is choosing
between can be further decorated with information about their current state. There are
facilities for inspecting details of the IBL decision making process programmatically
facilitating debugging, logging and fine grained control of complex models.
"""

__version__ = "5.1.5"

PYACTUP_MINIMUM_VERSION = "2.2.3"

if "dev" in __version__:
    print("PyIBL version", __version__)

import collections.abc as abc
import csv
import io
import math
import matplotlib.pyplot as plt
import numbers
import numpy as np
import os
import pandas as pd
import pyactup
import random
import sys
import warnings

from collections import Counter, defaultdict
from itertools import count
from numbers import Real
from packaging import version
from prettytable import PrettyTable
from warnings import warn

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = (lambda message, category, filename, lineno, line=None:
                          formatwarning_orig(message, category, filename, lineno, line=''))

if version.parse(pyactup.__version__) < version.parse(PYACTUP_MINIMUM_VERSION):
    warn(f"PyACTUp version {pyactup.__version__} is older than that required by this version of PyIBL")

__all__ = ["Agent", "DelayedResponse",
           "positive_linear_similarity", "positive_quadratic_similarity",
           "bounded_linear_similarity", "bounded_quadratic_similarity"]

LEGEND_LIMIT = 10

SQRT2 = math.sqrt(2)
AGGREGATE_COLUMNS = tuple(("iteration,time,choice,utility,option,blended_value,"
                           "retrieval_probability,activation,base_level_activation,"
                           "activation_noise").split(","))


class Agent:
    """A cognitive entity learning and making decisions based on its experience from prior decisions.
    The main entry point to PyIBL. An Agent has a *name*, a string, which can be retrieved
    with the :attr:`name` property. The name cannot be changed after an agent is created.
    If, when creating an agent, the *name* argument is not supplied or is ``None``, a name
    will be created of the form ``'Anonymous-Agent-n'``, where *n* is a unique integer.

    An :class:`Agent` also has zero or more *attributes*, named by strings. The attribute
    names can be retrieved with the :attr:`attributes` property, and cannot be changed
    after an agent is created. Attribute names must be non-empty strings. The value of
    *attributes*, if present, should be an :class:`Iterable` of strings. As a convenience
    if none of the attribute names contain spaces or commas a string consisting of the
    names, separated by commas or spaces may be used instead of a list.

    The agent properties :attr:`noise`, :attr:`decay`, :attr:`temperature`,
    :attr:`mismatch_penalty`, :attr:`optimized_learning`, :attr:`default_utility`,
    :attr:`default_utility_populates` and :attr:`fixed_noise` can be initialized when
    creating an Agent.

    """

    _agent_number = 0

    def __init__(self,
                 attributes=[],
                 name=None,
                 noise=pyactup.DEFAULT_NOISE,
                 decay=pyactup.DEFAULT_DECAY,
                 temperature=None,
                 mismatch_penalty=None,
                 optimized_learning=False,
                 default_utility=None,
                 default_utility_populates=False,
                 fixed_noise=False):
        self._attributes = pyactup.Memory._ensure_slot_names(attributes)
        if name is None:
            Agent._agent_number += 1
            name = f"agent-{Agent._agent_number}"
        elif not (isinstance(name, str) and len(name) > 0):
            raise TypeError(f"Agent name {name} is not a non-empty string")
        self._name = name
        self._memory = pyactup.Memory(optimized_learning=optimized_learning,
                                      threshold=None,
                                      index=(self._attributes or ("_decision",)))
        self.temperature = temperature # set temperature BEFORE noise
        self.noise = noise
        self.decay = decay
        self.mismatch_penalty = mismatch_penalty
        self.default_utility = default_utility
        self.default_utility_populates = default_utility_populates
        self._details = None
        self._aggregate_details = None
        self._aggregate_similarities = False
        self._aggregate_iteration = 0
        self._trace = False
        self._fixed_noise = fixed_noise
        self.reset()
        self._test_default_utility()

    def __repr__(self):
        return f"<Agent {str(self)} {id(self)}>"

    def __str__(self):
        return str(self._name)

    @property
    def name(self):
        """The name of this Agent.
        It is a string, provided when the agent was created, and cannot be changed
        thereafter.
        """
        return self._name

    @property
    def attributes(self):
        """A tuple of the names of the attributes included in all situations associated with decisions this agent will be asked to make.
        These names are assigned when the agent is created and cannot be
        changed, and are strings. The order of them in the returned
        tuple is the same as that in which they were given when the
        agent was created.
        """
        return self._attributes

    def _preferred_index(self):
        return [a for a in self.attributes if not self._memory._similarities.get(a)]

    def reset(self, preserve_prepopulated=False):
        """Erases this agent's memory and resets its time to zero.
        If it is ``True`` it deletes all those not created at time zero. IBLT parameters
        such as :attr:`noise` and :attr:`decay` are not affected. If ``False`` any
        prepopulated instances, including those created automatically if a
        :attr:`default_utility` is provided and :attr:`default_utility_populates` is
        ``True`` are removed, but the settings of those properties are not altered.
        """
        self._memory.reset(preserve_prepopulated=preserve_prepopulated,
                           index=self._preferred_index())
        self._last_learn_time = 0
        self._previous_choices = None
        self._pending_decision = None
        self._aggregate_iteration += 1

    @property
    def time(self):
        """This agent's current time.
        Time in PyIBL is a dimensionless quantity, typically just counting the number of
        choose/respond cycles that have occurred since the Memory was last :meth:`reset`.
        """
        return self._memory.time

    def advance(self, increment=1, target=None):
        """Advances the time of this agent by *increment* time steps.
        The *increment*, which defaults to ``1``, should be a non-negative integer; if it
        is not a :exc:`ValueError` is raised.

        If *target* is provided instead of *increment*, it should be the future time to
        which to advance. If both *target* and *increment* are supplied the time is set
        to the later of *target* and the current time advanced by *increment*.

        Returns the updated :attr:`time`.

        A :exc:`ValueError` is raised if it is in the past or is not a non-negative
        integer.
        """
        def ensure_num(n, name):
            try:
                result = int(n)
            except:
                result = None
            if result is None or result != n or result < 0:
                raise ValueError(f"The {name}, {n}, is not a non-negative integer")
            return result
        t = self.time
        if increment is not None:
            t = self.time + ensure_num(increment, "increment")
        if target is not None:
            n = ensure_num(target, "target")
            if n < self.time:
                raise ValueError(f"The target time, {target}, is earlier than the "
                                 f"current time, {self.time}")
            t = max(n, t)
        self._memory.time = t
        return self.time

    @property
    def noise(self):
        """The amount of noise to add during instance activation computation. This is typically a
        positive, possibly floating point, number between about ``0.2`` and ``0.8``. It
        defaults to ``0.25``. If explicitly zero, no noise is added during activation
        computation. Setting :attr:`noise` to ``None`` or ``False`` is equivalent to
        setting it to zero.

        If an explicit :attr:`temperature` is not set, the value of :attr:`noise` is also
        used to compute a default temperature for the blending operation on result
        utilities.

        Attempting to set :attr:`noise` to a negative number raises a
        :exc:`ValueError`.

        """
        return self._memory.noise

    @noise.setter
    def noise(self, value):
        if value != getattr(self._memory, "noise", None):
            self._memory.noise = float(value) if value is not None else 0.0

    @property
    def noise_distribution(self):
        """ Provide an alternative distribution from which noise is sampled.
        If ``None`` the default logistic distribution is used. Otherwise the value of this
        attribute should be a callable that takes no arguments and returns a real number.
        It will be called once each time activation noise is required and the value,
        scaled as usual by the :attr:`noise` parameter, will be used as the activation
        noise. A :exc:`ValueError` is raised if an attempt is made to set this attribute
        to anything other than a callable or ``None``.

        .. warning::
            It is rarely appropriate to use ``noise_distribution``. The default logistic
            distribution is almost always a more appropriate choice. The ability to change
            the distribution is provided only for esoteric purposes, and care should be
            exercised lest biologically implausible models result.
        """
        return self._memory.noise_distribution

    @noise_distribution.setter
    def noise_distribution(self, value):
        self._memory.noise_distribution = value

    @property
    def fixed_noise(self):
        """ Whether or not to constrain activation noise to remain constant at any one time.
        In some complicated models it may be necessary to compute the activation of an
        instance more than once at the same time step. Normally in this case each such
        computation generates its own, independent value of the activation noise. For
        some esoteric purposes it may be preferred to use the same activation noise for
        these different perspectives of an instance's activation, which can be achieved by
        setting this property to ``True``. By default :attr:`fixed_noise` is ``False``.
        """
        return self._fixed_noise

    @fixed_noise.setter
    def fixed_noise(self, value):
        self._fixed_noise = bool(value)

    @property
    def temperature(self):
        """The temperature parameter used for blending values.
        If ``None``, the default, the square root of ``2`` times the value of
        :attr:`noise` will be used. If the temperature is too close to zero, which
        can also happen if it is ``None`` and the :attr:`noise` is too low, or negative, a
        :exc:`ValueError` is raised.
        """
        return self._memory.temperature

    @temperature.setter
    def temperature(self, value):
        self._memory.temperature = value

    @property
    def decay(self):
        """Controls the rate at which activation for previously experienced instances in memory decay with the passage of time.
        Time in this sense is dimensionless, and simply the number of choose/respond
        cycles that have occurred since the agent was created or last :meth:`reset`. The
        :attr:`decay` is typically between about ``0.1`` to about ``2.0``. The default
        value is ``0.5``. If zero memory does not decay. Setting :attr:`decay` to ``None``
        or ``False`` is equivalent to setting it to zero. The :attr:`decay` must be less
        than ``1`` if this :class:`Agent` is using :attr:`optimized_learning`. Attempting
        to set :attr:`decay` to a negative number raises a :exc:`ValueError`.
        """
        return self._memory.decay

    @decay.setter
    def decay(self, value):
        if value != getattr(self._memory, "decay", None):
            self._memory.decay = float(value) if value is not None else 0.0

    @property
    def mismatch_penalty(self):
        """The mismatch penalty applied to partially matching values when computing activations.
        If ``None`` no partial matching is done. Otherwise any defined similarity
        functions (see :func:`similarity`) are called as necessary, and the resulting
        values are multiplied by the mismatch penalty and subtracted from the activation.
        For any attributes and decisions for which similarity functions are not defined
        only instances matching exactly on these attributes or decisions are considered.

        Attempting to set this parameter to a value other than ``None`` or a non-negative
        real number raises a :exc:`ValueError`.

        """
        return self._memory.mismatch

    @mismatch_penalty.setter
    def mismatch_penalty(self, value):
        v = value
        if value is False:
            v = None
        if v is not None and (not isinstance(v, Real) or v < 0):
            raise ValueError(f"The mismatch_penalty, {value}, is neither a non-negative "
                             f"real number nor None")
        self._memory.mismatch = v
        self._test_default_utility()

    @property
    def optimized_learning(self):
        """Whether or not this :class:`Agent` uses the optimized learning approximation when computing instance activations.
        If ``False``, the default, optimized learning is not used. If ``True`` is is used
        for all cases. If a positive integer, that number of the most recent rehearsals of
        an instance are used exactly, with any older rehearsals having their contributions
        to the activation approximated.

        Optimized learning can only be used if the :attr:`decay` is less than one.
        Attempting to set this parameter to ``True`` or an integer when :attr:`decay` is
        one or greater raises a :exc:`ValueError`.

        The value of this attribute can only be changed when the :class:`Agent` does not
        contain any instances, typically immediately after it is created or :meth:`reset`.
        Otherwise a :exc:`RuntimeError` is raised.

        .. warning::
            Care should be taken when adjusting the :attr:`time` manually and using
            optimized learning as operations that depend upon activation may no longer
            raise an exception if they are called when ``advance`` has not been called
            after an instance has been created or reinforced, producing biologically
            implausible results.
        """
        return self._memory.optimized_learning

    @optimized_learning.setter
    def optimized_learning(self, value):
        self._memory.optimized_learning = value

    @property
    def details(self):
        """A :class:`MutableSequence` into which details of this Agent's internal computations will be added.
        If ``None``, the default, such details are not accumulated. It can be explicitly
        set to a :class:`MutableSequence` of the modeler's choice, typically a list, into
        which details are accumulated. Setting it to ``True`` sets the value to a fresh,
        empty list, whose value can be ascertained by consulting the value of
        :attr:`details`.

        .. warning::
            In complex models, or models with many iterations, the ``details`` can gather
            a lot of information quickly. It is often best to ``clear()`` or otherwise
            reset the ``details`` frequently.

        A :exc:`ValueError` is raised if an attempt is made to set its value to anything
        other than ``None``, ``True`` or a :class:`MutableSequence`.

        >>> from pprint import pp
        >>> a = Agent(default_utility=10, default_utility_populates=True)
        >>> a.choose(["a", "b", "c"])
        'c'
        >>> a.respond(5)
        >>> a.details = True
        >>> a.choose()
        'a'
        >>> pp(a.details)
        [[{'decision': 'a',
           'activations': [{'name': '0000',
                            'creation_time': 0,
                            'attributes': (('_utility', 10), ('_decision', 'a')),
                            'references': (0,),
                            'base_activation': -0.3465735902799726,
                            'activation_noise': -0.1925212278297397,
                            'activation': -0.5390948181097123,
                            'retrieval_probability': 1.0}],
           'blended': 10.0},
          {'decision': 'b',
           'activations': [{'name': '0001',
                            'creation_time': 0,
                            'attributes': (('_utility', 10), ('_decision', 'b')),
                            'references': (0,),
                            'base_activation': -0.3465735902799726,
                            'activation_noise': -0.21036659990743722,
                            'activation': -0.5569401901874098,
                            'retrieval_probability': 1.0}],
           'blended': 10.0},
          {'decision': 'c',
           'activations': [{'name': '0002',
                            'creation_time': 0,
                            'attributes': (('_utility', 10), ('_decision', 'c')),
                            'references': (0,),
                            'base_activation': -0.3465735902799726,
                            'activation_noise': -0.0970213443277114,
                            'activation': -0.44359493460768396,
                            'retrieval_probability': 0.16296805894917318},
                           {'name': '0003',
                            'creation_time': 1,
                            'attributes': (('_utility', 5), ('_decision', 'c')),
                            'references': (1,),
                            'base_activation': 0.0,
                            'activation_noise': 0.1349273092778319,
                            'activation': 0.1349273092778319,
                            'retrieval_probability': 0.8370319410508268}],
           'blended': 5.814840294745866}]]
        """
        return self._details

    @details.setter
    def details(self, value):
        if value == 0:
            value = None
        elif value == True:
            value = []
        if not (value is None or isinstance(value, abc.MutableSequence)):
            raise ValueError("the value of details must be None or a list or other MutableSequence")
        self._details = value

    @property
    def aggregate_details(self):
        """A Pandas `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ aggregating information about the computations performed by this Agent.
        Set to ``True`` to begin aggregating information, or to ``False`` (the default) to
        not collect this information. If already collecting information setting this to
        ``True`` will clear the already gathered information and start over.

        This collecting of information is most useful if the Agent is used in a common
        pattern: a number of rounds of a task are performed, each round a choose/respond
        cycle, and this sequence of rounds is repeated over a number of distinct, virtual
        participants, :meth:`reset` being called between each participant.

        Each row of the resulting DataFrame corresponds to the state of one of the
        instances in the Agent at a particular time for a particular virtual participant.
        The columns in the DataFrame are

        ``iteration``
            the number of times :meth:`reset` has been called before recording the data in
            this row, which is typically an identifier for the virtual participant

        ``time``
            the :attr:`time` in this Agent at which the data in this row was recorded,
            typically the round of the task

        ``choice``
            the choice made in this round; if this Agent has attributes, the choice will
            be a tuple containing the values of the attributes in the order in which the
            attributes were declared when creating the Agent

        ``utility``
            the utility in this instance

        ``option``
            the option in this instance; note that this typically differs from the choice
            column as the choice is the choice made at this time while the option is the
            option in this instance, whether or not this instance contributed to the
            "winning" choice; however, as with choice, if the Agent was defined to have
            attributes the value in this column will be a tuple listing the values of
            those attributes in the order provided when the Agent was defined

        ``blended_value``
            the blended value computed for the option in this instance at this time; note
            that the same blended value is reported for each row corresponding to an
            instance with the same option at the same time and iteration

        ``retrieval_probability``
            the probability of retrieval of this instance at this time

        ``activation``
            the activation of this instance at this time

        ``base_level_activation``
            the base level term of the activation computation at this time

        ``activation_noise``
            the noise term of  the activation computation at this time

        If partial matching was used at least once in this Agent since the aggregate
        details started being record there will be additional columns following those
        above. The first, ``mismatch``, is the total mismatch penalty for this instance
        for this query. After this there is one column for each attribute that was
        partially matched, the similarity value computed for this attribute.

        Note that each time the ``aggregate_details`` attribute is queried a new DataFrame
        is created from an internal data structure. If you wish to modify a DataFrame and
        have those changes retained you need to retain this DataFrame.

        .. warning::
            Collecting ``aggregate_details`` incurs costs, both in space and time, and for
            large, complex models some thought may be needed to balance these costs
            against the value and convenience of the results.

        >>> a = Agent()
        >>> a.aggregate_details = True
        >>> a.populate(["a", "b"], 2.2)
        >>> for participant in range(100):
                a.reset(True)
                for round in range(60):
                    choice = a.choose(["a", "b"])
                    if choice == "a":
                        a.respond(1)
                    else:
                        a.respond(2 if random.random() < 0.5 else 0)
        >>> a.aggregate_details
               iteration  time choice  utility option  blended_value  retrieval_probability  activation  base_level_activation  activation_noise
        0              1     1      a      2.2      a       2.200000               1.000000    0.091998               0.000000          0.091998
        1              1     1      a      2.2      b       2.200000               1.000000   -0.233124               0.000000         -0.233124
        2              1     2      b      1.0      a       1.479895               0.600088   -0.390199               0.000000         -0.390199
        3              1     2      b      2.2      b       2.200000               1.000000    0.185066              -0.346574          0.531639
        4              1     3      b      1.0      a       1.261019               0.782484   -0.352501              -0.346574         -0.005927
        5              1     3      b      2.0      b       2.004449               0.977755    0.321220               0.000000          0.321220
        6              1     4      b      1.0      a       1.026956               0.977537   -0.250698              -0.549306          0.298608
        7              1     4      b      2.0      b       2.010839               0.945803    0.507820               0.534800         -0.026980
        8              1     5      b      1.0      a       1.053339               0.955551    0.111650              -0.693147          0.804797
        9              1     5      b      0.0      b       1.662770               0.169062   -0.211333               0.000000         -0.211333
        ...          ...   ...    ...      ...    ...            ...                    ...         ...                    ...               ...
        11990        100    56      b      1.0      a       1.000052               0.999956    2.261437               2.175180          0.086257
        11991        100    56      b      2.0      b       1.366890               0.683311    0.762205               0.813690         -0.051485
        11992        100    57      a      1.0      a       1.000002               0.999998    2.427721               2.088158          0.339563
        11993        100    57      a      2.0      b       0.422011               0.210943    0.814716               0.783826          0.030889
        11994        100    58      b      1.0      a       1.000018               0.999985    2.148301               2.153508         -0.005207
        11995        100    58      b      2.0      b       1.733465               0.866600    1.085368               0.756193          0.329175
        11996        100    59      b      1.0      a       1.000011               0.999991    2.089526               2.077563          0.011963
        11997        100    59      b      2.0      b       1.246271               0.623060    1.708573               0.730428          0.978145
        11998        100    60      b      1.0      a       1.000004               0.999996    2.318136               2.026204          0.291932
        11999        100    60      b      2.0      b       1.081265               0.540324    1.061905               1.107370         -0.045465
        [12000 rows x 10 columns]

        """
        if self._aggregate_details is None:
            return None
        cols = AGGREGATE_COLUMNS
        if self._aggregate_similarities:
            cols += ("mismatch",) + tuple(f"{a}.similarity" for a in self._attributes)
        result = pd.DataFrame(self._aggregate_details, columns=cols)
        result.dropna(axis="columns", how="all", inplace=True)
        return result

    @aggregate_details.setter
    def aggregate_details(self, value):
        if value:
            self._aggregate_details = []
        else:
            self._aggregate_details = None
        self._aggregate_similarities = False
        self._aggregate_iteration = 0

    @property
    def trace(self):
        """A boolean which, if ``True``, causes the :class:`Agent` to print details of its computations to standard output.
        Intended for use as a tool for debugging models. By default it is ``False``.

        The output is divided into blocks, the first line of which describes the
        choice being described and the blended value of its outcome. This is followed by
        a tabular description of various intermediate values used to arrive at this
        blended value.

        ::

         >>> a = Agent(default_utility=10, default_utility_populates=True)
         >>> a.choose(["a", "b", "c"])
         'b'
         >>> a.respond(5)
         >>> a.choose()
         'a'
         >>> a.respond(7.2)
         >>> a.choose()
         'c'
         >>> a.respond(2.3)
         >>> a.choose()
         'a'
         >>> a.respond(7.2)
         >>> a.trace = True
         >>> a.choose()

         a → 7.214829206137316 @ time=5
         +------+----------+---------+-------------+---------+---------------------+---------------------+---------------------+---------------------+-----------------------+
         |  id  | decision | created | occurrences | outcome |   base activation   |   activation noise  |   total activation  |   exp(act / temp)   | retrieval probability |
         +------+----------+---------+-------------+---------+---------------------+---------------------+---------------------+---------------------+-----------------------+
         | 0006 |    a     |    0    |     [0]     |    10   | -0.8047189562170503 | 0.23439403910287038 | -0.5703249171141799 | 0.19926444531722448 |  0.00529614504904138  |
         | 0010 |    a     |    2    |    [2, 4]   |   7.2   | 0.45574639440832615 |  0.8249453921296758 |  1.2806917865380019 |  37.425166810260265 |   0.9947038549509586  |
         +------+----------+---------+-------------+---------+---------------------+---------------------+---------------------+---------------------+-----------------------+

         b → 8.633125874767709 @ time=5
         +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+----------------------+-----------------------+
         |  id  | decision | created | occurrences | outcome |   base activation   |   activation noise   |   total activation  |   exp(act / temp)    | retrieval probability |
         +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+----------------------+-----------------------+
         | 0007 |    b     |    0    |     [0]     |    10   | -0.8047189562170503 | -0.16726717777620997 | -0.9719861339932603 |  0.063979539230306   |   0.7266251749535416  |
         | 0009 |    b     |    1    |     [1]     |    5    | -0.6931471805599453 | -0.6244610552806152  | -1.3176082358405605 | 0.024070725797184517 |  0.27337482504645844  |
         +------+----------+---------+-------------+---------+---------------------+----------------------+---------------------+----------------------+-----------------------+

         c → 5.881552425492787 @ time=5
         +------+----------+---------+-------------+---------+---------------------+--------------------+----------------------+--------------------+-----------------------+
         |  id  | decision | created | occurrences | outcome |   base activation   |  activation noise  |   total activation   |  exp(act / temp)   | retrieval probability |
         +------+----------+---------+-------------+---------+---------------------+--------------------+----------------------+--------------------+-----------------------+
         | 0008 |    c     |    0    |     [0]     |    10   | -0.8047189562170503 | 0.5923008644042377 | -0.21241809181281257 | 0.548367776208054  |   0.4651366786354268  |
         | 0011 |    c     |    3    |     [3]     |   2.3   | -0.3465735902799726 | 0.1835398166993702 | -0.1630337735806024  | 0.6305712354751412 |   0.5348633213645733  |
         +------+----------+---------+-------------+---------+---------------------+--------------------+----------------------+--------------------+-----------------------+

         'b'
         """
        return self._trace

    @trace.setter
    def trace(self, value):
        self._trace = bool(value)

    @property
    def default_utility(self):
        """The utility, or a function to compute the utility, if there is no matching instance.
        If when :meth:`choose` is called, for some choice passed to it there is no
        existing instance that matches the choice the value of this property is consulted.
        Note that an instance added with :meth:`populate` counts as matching, and will
        prevent the interrogation of this property. If partial matching
        (:attr:`mismatch_penalty`) is enabled, any instance that even partially matches a
        choice will prevent the iterrogation of this property.

        .. warning::
            It is rarely appropriate to use :attr:`default_utility` when partial matching
            is being used. If they are used together the :attr:`default_utility` will
            only be applied the first time a relevant choice is made, as all subsequent
            choices will partially match. This generally gives unpredicatable results and
            is rarely useful. Instead, when using partial matching it is usually better
            to explicitly prepopulate appropriate instances using :meth:`populate`.

        The value of this property may be a :class:`Real`, in which case when needed it is
        simply used as the default utility. If it is not a Real, it is assumed to be a
        function that takes one argument, one of the choices passed to :meth:`choose`.
        When a default utility is needed that function will be called, passing the choice
        in question to it, and the value returned, which should be a Real, will be used.
        If at that time the value is not a function of one argument, or it does not return
        a Real, a :exc:`RuntimeError` is raised.

        The :attr:`default_utility_populates` property, which is ``False`` by default,
        controls whether or not an instance is added for each interrogation of
        the :attr:`default_utility` property. If an instance is added, it is added
        as by :meth:`populate` at the current time.

        Setting :attr:`default_utility` to ``None`` or ``False`` (the initial default)
        causes no default utility to be used. In this case, if :meth:`choose` is called
        for a decision in a situation for which there is no instance available, a
        :exc:`RuntimeError` is raised.
        """
        return self._default_utility

    @default_utility.setter
    def default_utility(self, value):
        if value is False:
            value = None
        self._callable_default_utility =  not (value is None or isinstance(value, numbers.Real))
        self._default_utility = value
        self._test_default_utility()

    def _test_default_utility(self):
        try:
            if self._default_utility is not None and self._memory.mismatch is not None:
                warn("Setting a default_utility and using partial matching "
                     "simultaneously is usually ill-advised")
        except AttributeError:
            # We were called before self was completely initialized.
            pass

    @property
    def default_utility_populates(self):
        """Whether or not a default utility provided by the :attr:`default_utility` property is also entered as an instance in memory.
        This property has no effect if default_utility has been set to ``None`` or ``False``.
        """
        return self._default_utility_populates

    @default_utility_populates.setter
    def default_utility_populates(self, value):
        self._default_utility_populates = bool(value)

    def populate(self, choices, outcome, when=None):
        """Adds instances to memory, one for each of the *choices*, with the given outcome, at the current time, without advancing that time.
        The *outcome* should be a :class:`Real` number.
        The *choices* are as described in :meth:`choose`.
        If provided, *when* should be a time, a dimensionless quantity, typically a count
        of the number of choose/respond cycles that have occurred since the agent was
        created or last :meth:`reset`; providing *time* will cause the instances to be
        added at that specified time instead of the current time. The *time* provided
        may not be in the future.

        This is typically used to enable startup of a model by adding instances before the
        first call to :meth:`choose`. When used in this way the timestamp associated with
        this occurrence of the instance will be zero. Subsequent occurrences are possible
        if :meth:`respond` is called with the same outcome after :meth:`choose` has
        returned the same decision in the same situation, in which case those reinforcing
        occurrences will have later timestamps. An alternative mechanism to facilitate
        sartup of a model is setting the :attr:`default_utility` property of the agent.
        While rarely done, a modeler can even combine the two mechanisms, if desired.

        It is also possible to call :meth:`populate` after choose/respond cycles have
        occurred. In this case the instances are added with the current time as the
        timestamp. This is one less than the timestamp that would be used were an instance
        to be added by being experienced as part of a choose/respond cycle instead. Each
        agent keeps internally a clock, the number of choose/respond cycles that have
        occurred since it was created or last :meth:`reset`. When :meth:`choose` is called
        it advances that clock by one *before* computing the activations of the existing
        instances, as it must since the activation computation depends upon all
        experiences having been in the past. That advanced clock is the timestamp used
        when an instance is added or reinforced by :meth:`respond`.

        The *when* argument, which if provided should be a time, can be used to add
        instances at other times.

        Raises a :exc:`ValueError` if *outcome* is not a :class:`Real` number, if any of
        the *choices* are malformed or duplicates, or if *when* is in the future.

        .. warning::
            In normal use you should only call :meth:`populate` before any choose/respond
            cycles. If, for exotic purposes, you do wish to call it after, caution should
            be exercised to avoid biologically implausible models. It should not normally
            be necessary to use the *when* argument, which is provided only for esoteric
            uses. In particular instances may not be added in the future, as that would
            result in tears as operations such as :meth:`choose` will raise
            an :exc:`Exception`.
        """
        if when is not None:
            if when > self.time:
                raise ValueError(f"The when argument ({when}) must not be in the future")
            return self._at_time(when, lambda: self.populate(choices, outcome))
        Agent._outcome_value(outcome)
        for choice in self._make_queries(choices):
            self._memory.learn(Agent._add_utility(choice, outcome))
            self._last_learn_time = max(self._last_learn_time, self._memory.time)

    @staticmethod
    def _attribute_value(value, attribute):
        if not isinstance(value, abc.Hashable):
            raise ValueError(
                f"{value} is not hashable and cannot be used as the value of attribute {attribute}")
        return value

    def _canonicalize_choice(self, choice):
        if self.attributes:
            if isinstance(choice, abc.Mapping):
                return { a: Agent._attribute_value(choice.get(a), a)
                         for a in self._attributes }
            elif isinstance(choice, abc.Sequence):
                return { a: Agent._attribute_value(c, a)
                         for c, a in zip(choice, self._attributes) }
            else:
                raise ValueError(f"{choice} cannot be used as a choice")
        elif choice is None:
            raise ValueError(f"None cannot be used as a choice")
        elif isinstance(choice, abc.Hashable):
            return { "_decision": choice }
        else:
            raise ValueError(f"{choice} is not hashable and cannot be used as a choice")

    def _make_queries(self, choices):
        result = [ self._canonicalize_choice(c) for c in choices ]
        if len(set(tuple(d.items()) for d in result)) != len(result):
            raise ValueError("duplicate choices")
        return result

    @staticmethod
    def _add_utility(attributes, utility):
        result = {"_utility": utility}
        result.update(attributes)
        return result

    @staticmethod
    def _outcome_value(value):
        if not isinstance(value, numbers.Real):
            raise ValueError(
                f"outcome {value} does not have a (non-complex) numeric value")
        return value

    def _at_time(self, when, callback):
        if not isinstance(when, int):
            raise ValueError(f"Time {when} is not an integer")
        if when > self.time:
            raise ValueError(f"Time {when} cannot be greater than the current time, {self.time}")
        saved = self._memory.time
        try:
            self._memory._time = when
            callback()
        finally:
            self._memory._time = saved

    def choose(self, choices=None, details=False):
        """Selects which of the *choices* is expected to result in the largest payoff, and returns it.
        The expected form of the *choices* depends upon whether or not this :class:`Agent`
        has any attributes or not. If it does not, each of the *choices* should be a
        :class:`Hashable` that is not ``None``, representing an atomic choice; if any of
        the *choices* are not hashable or are ``None`` a :exc:`ValueError` is raised.

        If this :class:`Agent` does have attributes (that is, the *attributes* argument
        was supplied and non-empty when it was created, or, equivalently, the value of the
        :attr:`attributes` property is a non-empty list), then each of the *choices* can
        be either a :class:`Mapping`, typically a :class:`dict`, mapping attribute names
        to their values, or a :class:`Sequence`, typically a :class:`list`, containing
        attribute values in the order they were declared when this :class:`Agent` was
        created and would be returned by :attr:`attributes`. Attributes not present
        (either there is no key in the :class:`Mapping`, or a :class:`Sequence` shorter
        than the number of attributes) have a value of `None`, while values not
        corresponding to attributes of the :class:`Agent` (either a key in the
        :class:`Mapping` that does not match an attribute name, or a :class:`Sequence`
        longer than the number of attributes) are ignored. Whether a :class:`Mapping` or a
        :class:`Sequence`, all the attribute values must be :class:`Hashable`, and are
        typically strings or numbers. If any of the *choices* do not have one of these
        forms a :exc:`ValueError` is raised.

        In either case, if any pair of the *choices* duplicate each other, even if of
        different forms (e.g. dictionary versus list), and after adding default ``None``
        values and removing ignored values, a :exc:`ValueError` is raised.

        It is also possible to supply no *choices*, in which case those used in the most
        recent previous call to this method are reused. If there was no such previous call
        since the last time this :class:`Agent` was :meth:`reset` a :exc:`ValueError` is
        raised.

        For each of the *choices* this method finds all instances in memory that match,
        and computes their activations at the current time based upon when in the past
        they have been seen, modified by the value of the :attr:`decay` property, and with
        noise added as controlled by the :attr:`noise` property. If partial matching has
        been enabled with :attr:`mismatch_penalty` such matching instances need not match
        exactly, and the similarities modified by the mismatch penalty are subtracted from
        the activations. If partial matching is not enabled only those instances that
        match exactly are consulted. "Exact" matches are based on Python's ``==``
        operator, not ``is``. Thus, for example ``0``, ``0.0`` and ``False`` all match one
        another, as do ``1``, ``1.0`` and ``True``.

        Looking at the activations of the whole ensemble of instances matching a choice a
        retrieval probability is computed for each possible outcome, and these are
        combined to arrive at a blended value expected for the choice. This blending
        operation depends upon the value of the :attr:`temperature` property; if none is
        supplied a default is computed based on the value of the :attr:`noise` parameter.
        The value chosen and returned is that element of *choices* with the highest
        blended value. In case of a tie one will be chosen at random.

        After a call to :attr:`choose` a corresponding call must be made to
        :meth:`respond` before calling :meth:`choose` again, or a :exc:`RuntimeError` will
        be raised.

        If the *details* argument is also supplied and is ``True`` two values are returned
        as a 2-tuple, the first as above and the second containing data used to arrive at
        the selection. While the comparison of blended values used by :meth:`choose` is
        the appropriate process for most models, for some specialized purposes the modeler
        may wish to implement a different decision procedure. This additional information,
        when combined with supplying a second argment to :meth:`respond`, facilitates the
        construction of such more complicated models.

        The second return value is a list of dicts, one for each choice. These dicts have
        entries for the choice, the blended value, and a list of retrieval probability
        descriptions. The retrieval probability descriptions are themselves dicts, one for
        each instance consulted in constructing the given choice's blended value. Each of
        these latter dicts has two entries, one for the utility stored in the instance and
        the other its probability of retrieval.

        Because of noise the results returned by :attr:`choose` are stochastic, so the
        results of running the following examples may differ in their details from those
        shown.

    >>> from pprint import pp
    >>> a = Agent(name="Button Pusher", default_utility=10)
    >>> a.choose(["left", "right"])
    'left'
    >>> a.respond(5)
    >>> a.choose()
    'right'
    >>> a = Agent(["species", "state"], "Pet Shop")
    >>> a.populate([["parrot", "dead"]], 0)
    >>> a.populate([["parrot", "squawking"]], 10)
    >>> a.choose([["parrot", "dead"], ["parrot", "squawking"]])
    ['parrot', 'squawking']
    >>> a = Agent(name="Cheese Shop")
    >>> a.populate(["Tilset", "Wensleydale"], 10)
    >>> a.choose(["Tilset", "Wensleydale"])
    'Tilset'
    >>> a.respond(1)
    >>> choice, data = a.choose(["Tilset", "Wensleydale"], details=True)
    >>> choice
    'Wensleydale'
    >>> pp(data)
    [{'choice': 'Wensleydale',
      'blended_value': 10.0,
      'retrieval_probabilities': [{'utility': 10, 'retrieval_probability': 1.0}]},
     {'choice': 'Tilset',
      'blended_value': 2.1449539686120187,
      'retrieval_probabilities': [{'utility': 10,
                                   'retrieval_probability': 0.12721710762355765},
                                  {'utility': 1,
                                   'retrieval_probability': 0.8727828923764424}]}]

        """
        if self._pending_decision:
            raise RuntimeError("choice requested before previous outcome was supplied")
        choices = list(choices if choices is not None else [])
        if not choices:
            if self._previous_choices:
                choices = self._previous_choices
            else:
                raise ValueError("no choices were supplied and no default ones are available")
        queries = self._make_queries(choices)
        self._previous_choices = choices
        det = [] if self._details is not None else None
        try:
            if details or det is not None or self._trace or self._aggregate_details is not None:
                history = []
                self._memory.activation_history = history
            else:
                history = None
            if self._last_learn_time >= self._memory.time:
                self._memory.advance(self._last_learn_time - self._memory.time + 1)
            utilities = []
            ret_probs = []
            agg_len = len(self._aggregate_details) if self._aggregate_details is not None else None
            def do_choose(history):
                for c, q in zip(choices, queries):
                    u = self._memory.blend("_utility", q)
                    if u is None:
                        if self._default_utility is not None:
                            if self._callable_default_utility:
                                u = self._default_utility(c)
                            else:
                                u = self._default_utility
                            if self._default_utility_populates:
                                self._at_time(0, lambda: self._memory.learn(Agent._add_utility(q, u)))
                        else:
                            raise RuntimeError(f"No experience available for choice {c}")
                    utilities.append(u)
                    if details:
                        ret_probs.append([{"utility": Agent._extract_instance_utility(inst),
                                           "retrieval_probability": inst["retrieval_probability"]}
                                          for inst in self._memory.activation_history])
                    if det is not None:
                        d = dict(q) if self.attributes else {"decision": q["_decision"]}
                        d["activations"] = history
                        d["blended"] = u
                        det.append(d)
                    if history is not None:
                        if self._trace:
                            self._print_trace(q, u, history)
                        if (ad := self._aggregate_details) is not None:
                            agg = None
                            for d in history:
                                # cf. the definition of AGGREGATE_COLUMNS near the top of
                                #     this file
                                agg = [self._aggregate_iteration,
                                       self.time,
                                       None,
                                       (att := d["attributes"])[0][1],
                                       tuple(a[1] for a in att[1:]) if self._attributes else att[1][1],
                                       u,
                                       d["retrieval_probability"],
                                       d["activation"],
                                       d["base_level_activation"],
                                       d.get("activation_noise", 0)]
                                if sim := d.get("similarities"):
                                    self._aggregate_similarities = True
                                    agg.append(d.get("mismatch", 0))
                                    agg.extend([sim.get(a, np.nan) for a in self._attributes])
                                if agg is not None:
                                    ad.append(agg)
                        history = []
                        self._memory.activation_history = history
            if (not self._fixed_noise):
                do_choose(history)
            else:
                with self._memory.fixed_noise:
                    do_choose(history)
        finally:
            self._memory.activation_history = None
        if self._details is not None:
            self._details.append(det)
        if self._trace:
            print(f"\n   {'='*140}")
        best_indecies = [0]
        best_utility = utilities[0]
        for u, i in zip(utilities[1:], count(1)):
            if u > best_utility:
                best_utility = u
                best_indecies = [i]
            elif u == best_utility:
                best_indecies.append(i)
        best = random.choice(best_indecies)
        self._pending_decision = (best, choices, queries, utilities)
        if agg_len is not None:
            for v in self._aggregate_details[agg_len:]:
                v[2] = tuple(queries[best].values()) if self._attributes else queries[best]["_decision"]
        result = choices[best]
        if details:
            return result, sorted(({"choice": c,
                                    "blended_value": bv,
                                    "retrieval_probabilities": rp}
                                   for c, bv, rp in zip(choices, utilities, ret_probs)),
                                  key=lambda x: x.get("blended_value"),
                                  reverse=True)
        else:
            return result

    def _extract_instance_utility(inst):
        first_attr = inst["attributes"][0]
        assert first_attr[0] == "_utility"
        return first_attr[1]

    def _print_trace(self, query, utility, history):
        print()
        if self.attributes:
            print(", ".join(list(f"{k}: {v}" for k, v in query.items())), end="")
        else:
            print(query["_decision"], end="")
        print(f" → {utility} @ time={self.time}")
        tab = PrettyTable()
        fields = (["id"] + (list(self.attributes) or ["decision"]) +
                  ["created", "occurrences", "outcome", "base activation", "activation noise"])
        if self._memory.mismatch:
            fields.append("mismatch adjustment")
        fields.extend(["total activation", "exp(act / temp)", "retrieval probability"])
        tab.field_names = fields
        for h in history:
            attrs = dict(h["attributes"])
            row = [h["name"]]
            if self.attributes:
                for a in self.attributes:
                    row.append(attrs.get(a, ""))
            else:
                row.append(attrs["_decision"])
            row.append(h["creation_time"])
            row.append(h["references"] if self._memory.optimized_learning else list(h["references"]))
            row.append(attrs["_utility"])
            row.append(h["base_level_activation"])
            row.append(h.get("activation_noise") or 0.0)
            if self._memory.mismatch:
                row.append(h["mismatch"])
            row.append(h["activation"])
            row.append(math.exp(h["activation"] / (self.temperature or SQRT2 * self.noise)))
            row.append(h["retrieval_probability"])
            tab.add_row(row)
        print(tab, flush=True)

    def respond(self, outcome=None, choice=None):
        """Provide the *outcome* resulting from the most recent decision selected by :meth:`choose`.
        The *outcome* should be a real number, where larger numbers are considered "better."
        This results in the creation or reinforcemnt of an instance in memory for the
        decision with the given outcome, and is the fundamental way in which the PyIBL
        model "learns from experience."

        By default the choice selected by :meth:`choose` is the one to which the outcome
        is attached. In unusual cases, however, the modeler may prefer to select a
        different choice; for example, if using a different decision procedure based on
        the information returned by a ``True`` value of the second argument to
        :meth:`Choose`, or if performing model tracing of an individual human's behavior
        on the experiment being modeled. To support these unusual cases a second argument
        may be passed to :meth:`respond`, which is the choice to actually be made. If it
        is not one of the choices provided in the original call to :meth:`choose` a
        ``ValueError`` is raised.

        It is also possible to delay feedback, by calling :meth:`respond` without
        arguments, or with the first argument being ``None``. This tells the
        :class:`Agent` to assume it has received feedback equal to that it expected, that
        is, the blended value resulting from past experiences. In this case
        :meth:`respond` returns a value, a :class:`DelayedRespone` object, which can be
        used subsequently to update the response.

        .. warning::
            Delayed feedback is an experimental feature and care should be exercised in
            its use to avoid biologically implausible models.

        If there has not been a call to :meth:`choose` since the last time :meth:`respond`
        was called a :exc:`RuntimeError` is raised. If *outcome* is neither ``None`` nor a
        real number a :exc:`ValueError` is raised.
        """
        if not self._pending_decision:
            raise RuntimeError(
                f"outcome {outcome} supplied when no decision requiring an outcome is pending")
        best, choices, queries, utilities = self._pending_decision
        if choice is None:
            i = best
        else:
            try:
                i = choices.index(choice)
            except ValueError:
                raise ValueError(f"{choice} is not one of choices originally provided")
        if outcome is not None:
            self._memory.learn(Agent._add_utility(queries[i], Agent._outcome_value(outcome)))
            self._last_learn_time = self._memory.time
            self._pending_decision = None
        else:
            self._memory.learn(Agent._add_utility(queries[i], utilities[i]))
            self._last_learn_time = self._memory.time
            result = DelayedResponse(self, queries[i], utilities[i])
            self._pending_decision = None
            return result

    def discrete_blend(self, outcome_attribute, conditions):
        """Returns the most likely to be retrieved, existing value of *outcome_attribute* subject to the *conditions*.
        That is, the existing value from the instances in this :class:`Agent` such that
        the likelihood one of those instances will be retrieved weighted by their
        probabilities of retrieval. The *outcome_attribute* should an attribute name in
        this :class:`Agent`. The *conditions* should be a ``Mapping`` mapping attribute
        names to values, like the various choices provided to :meth:`choose`. Also returns
        a second value, a dictionary mapping possible values of *outcome_attribute* to
        their probabilities of retrieval.

        This method can be useful for building specialized models using schemes that do
        not correspond to the paradigm exposed by the usual ``choose``/``respond`` cycles.
        """
        pyactup.Memory._ensure_slot_name(outcome_attribute)
        conditions = self._make_queries([conditions])[0]
        if outcome_attribute in conditions:
            del conditions[outcome_attribute]
        return self._memory.discrete_blend(outcome_attribute, conditions)

    def instances(self, file=sys.stdout, pretty=True):
        """Prints or returns all the instances currently stored in this :class:`Agent`.
        If *file* is ``None`` a list of dictionaries is returned, each corresponding
        to an instance. If *file* is a string it is taken as a file name, which is opened
        for writing, and the results printed thereto; otherwise *file* is assumed to be
        an open, writable ``file``. By default the file is standard out, typically
        resulting in the instances being printed to the console.

        When printing to a file if *pretty* is true, the default, a format intended for
        reading by humans is used. Otherwise comma separated values (CSV) format, more
        suitable for importing into spreadsheets, numpy, and the like, is used.
        """
        attrs = [ (a, a) for a in self.attributes ]
        if not attrs:
            attrs = [ ("decision", "_decision") ]
        result = []
        for c in self._memory.values():
            d = {name: c[a] for name, a in attrs}
            d["outcome"] = c["_utility"]
            d["created"] = c._creation
            d["occurrences"] = tuple(map(int, c.references))
            result.append(d)
        if file is None:
            return result
        if isinstance(file, io.TextIOBase):
            Agent._print_instance_data(result, pretty, file)
        else:
            with open(file, "w+", newline=(None if pretty else "")) as f:
                Agent._print_instance_data(result, pretty, f)

    @staticmethod
    def _print_instance_data(data, pretty, file):
        if not data:
            return
        if pretty:
            tab = PrettyTable()
            tab.field_names = data[0].keys()
            for d in data:
                tab.add_row(d.values())
            print(tab, file=file, flush=True)
        else:
            w = csv.DictWriter(file, data[0].keys())
            w.writeheader()
            for d in data:
                w.writerow(d)

    def plot(self, kind, title=None, xlabel=None, ylabel=None,
             include=None, exclude=None, min=None, max=None, earliest=None, latest=None,
             legend=None, limits=None, filename=None, show=None):
        """Generates a variety of plots of the evolution of this Agent's state over time.
        These plots can often be helpful in understanding how a PyIBL model is working.
        To generate these plots :attr:`aggregate_details` must be set to ``True``. The
        constraints on the model for producing useful :attr:`aggregate_details` similarly
        apply for producing useful plots. Note that these plots are of values averaged
        over all the virtual participants. The *kind* argument to :meth:`plot` should be
        a string, one of the following

        ``"choice"``
            a plot of the number of times various choices are made by the model as a
            function of time

        ``"bv"``
            a plot of the blended values of the options considered as a function of time

        ``"probability"``
            a plot of the probabilities of retrieval of the various instances as a function
            of time

        ``"activation"``
            a plot of the activations of the various instances as a function of time

        ``"baselevel"``
            a plot of the base level term in the activation computations of the various
            instances as a function of time

        ``"mismatch"``
            the total mismatch penalty terms in the activation computations of the various
            instances as a function of time

        In addition, the *kind* may be a string comprised of the name of an attribute
        concatenated with ``".similarity"``, which will produce a plot of the average
        similarity value of this attribute's value to that sought as a function of time.

        Whether or not the plot is displayed and/or saved to a file can be specified with
        the optional arguments *show* (a Boolean) and *filename* (a string). If *show* is
        not provided, by default the file will be displayed if and only if no *filename* is
        provided. In addition the plot is always returned by this method as a
        `matplotlib Figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html>`_.

        Details of the plot can be manipulated with several other optional arguments to
        :meth:`plot`:

        *title*
            the title applied to the plot

        *xlabel* and *ylabel*
            the labels of the corresponding axes of the plot

        *include*and *exclude*
            these should be lists of options; if *include* is provided only options in
            this list are displayed; any options in *exclude* are not displayed; if an
            option appears in both lists it is not displayed

        *min* and *max*
            these should be Real numbers; if *min* is supplied only results for instances
            with a utility greater or equal to this value are displayed; if *max* is
            supplied only results for instances with a utility less than or equal to this
            value are displayed

        *earliest* and *latest*
            these should be Real numbers; if *earliest* is supplied only results at this
            time or later are displayed; if *latest* is supplied only results at this time
            or earlier are displayed

        *legend*
            if ``True`` a legend is displayed on the plot; if ``False`` no legend is
            displayed; other values can also be supplied, which are passed on to
            `matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_;
            by default a legend is shown only if there are not so many items in the plot
            that the legend would likely overflow the available space

        *limits*
            this should be a list or tuple of two Real numbers, the smallest and greatest
            values of the Y axis of the plot; if not supplied suitable defaults are
            assumed

        Since :meth:`plot` always returns a Figure, more sophisticated manipulations can
        be made of it and the result displayed explicitly.

        >>> a = Agent()
        >>> a.aggregate_details = True
        >>> a.populate(["a", "b", "c"], 3.2)
        >>> for participant in range(1000):
                a.reset(True)
                for round in range(100):
                    choice = a.choose(["a", "b", "c"])
                    if choice == "a":
                        a.respond(1)
                    elif choice == "b":
                        a.respond(2 if random.random() < 1/2 else 0)
                    elif a.time <= 50:
                        a.respond(3 if random.random() < 1/3 else 0)
                    else:
                        a.respond(3 if random.random() < 2/3 else 0)
        >>> a.plot("bv")

        .. image:: _static/bv_plot.png

        """
        if min and not isinstance(min, numbers.Real):
            raise ValueError("The min value, {min}, is neither a Real number nor None")
        if max and not isinstance(max, numbers.Real):
            raise ValueError("The max value, {max}, is neither a Real number nor None")
        if show is None:
            show = filename is None
        agg = self.aggregate_details
        if agg is None:
            raise RuntimeError("Can't make plot unless aggregate_details is set")
        plot_kind = {"choice":      ChoicePlot("choice", "Fraction making choice", True),
                     "bv":          OptionPlot("blended_value", "Mean blended value"),
                     "probability": InstancePlot("retrieval_probability",
                                                 "Mean probability of retrieval", True),
                     "activation":  InstancePlot("activation", "Mean total activation"),
                     "baselevel":   InstancePlot("base_level_activation", "Mean base level activation"),
                     "mismatch":    InstancePlot("mismatch", "Mean total mismatch penalty")
                     }.get(kind)
        if not plot_kind:
            if isinstance(kind, str) and kind.endswith(".similarity"):
                if kind in agg:
                    plot_kind = InstancePlot(kind, f"Mean similarities of {kind[0:-11]}",  True)
                else:
                    raise ValueError(f"The {kind[0:-11]} attribute is either absent or not partially matched")
            else:
                raise ValueError(f"Unknown plot kind {kind}")
        if kind == "mismatch" and "mismatch" not in agg:
            raise ValueError("Can't generate a mismatch plot when no attributes were partially matched")
        data = plot_kind.get_data(agg, include, exclude, min, max, earliest, latest)
        plt.clf()
        for k, (t, v) in data.items():
            plt.plot(t, v, label=str(k))
        if title is not False:
            plt.title(title or f"{plot_kind._description} versus time")
        if legend is None:
            legend = data and (len(data) <= LEGEND_LIMIT)
        if legend is True:
            plt.legend()
        elif legend:
            plt.legend(legend)
        plt.xlabel(xlabel if xlabel is not None else "Time")
        plt.ylabel(ylabel if ylabel is not None else  plot_kind._description)
        if limits:
            plt.ylim(limits)
        elif plot_kind._default_ylim:
            plt.ylim(plot_kind._default_ylim)
        if filename:
            plt.savefig(filename)
        if show:
            plt.show()
        return plt.gcf()

    @staticmethod
    def _ensure_plot_items(supplied, possibilities, argname):
        try:
            if len(s := set(supplied)) < len(supplied):
                raise ValueError(f"Duplicate {argname} in {supplied}")
            elif len(s.intersection(possibilities)) < len(s):
                raise ValueError(f"Unused {argname} in {supplied}")
        except:
            raise TypeError(f"{argname} should be a sequence of strings")

    def similarity(self, attributes=None, function=None, weight=None):
        """Assigns a function and/or corresponding weight to be used when computing the similarity of attribute values.
        The *attributes* are names of attributes of the :class:`Agent`. The value of
        *attributes*, if present, should be an :class:`Iterable` of strings. As a
        convenience if none of the attribute namess contain spaces or commas a string
        consisting of the names, separated by commas or spaces may be used instead of a
        list. For an :class:`Agent` that has no attributes the *attributes* argument
        should be empty or omitted.

        The *function* argument should be a ``Callable``, taking two arguments, or
        ``True``. The similarity value returned should be a real number between zero and
        one, inclusive. If ``True`` is passed as the value of *function* a default
        similarity function is used which returns one if its two arguments are ``==`` and
        zero otherwise. If, when called, the function returns a number outside that range
        a :exc:`RuntimeError` is raised. If, when the similarity function is called, the
        return value is not a real number a :exc:`ValueError` is raised.

        The *weight* should be a positive, real number, and defaults to one.

        Similarity functions are only called when the `Agent` has a
        :attr:`mismatch_penalty` specified. When a similarity function is called it is
        passed two arguments, attribute values to compare. The function should be
        commutative; that is, if called with the same arguments in the reverse order, it
        should return the same value. It should also be stateless, always returning the
        same values if passed the same arguments. If either of these constraints is
        violated no error is raised, but the results will, in most cases, be meaningless.

        If one of *function* or *weight* is omitted but the other is supplied, the
        supplied item is set with the omitted one unchanged. If called with neither
        *function* nor *weight* the similarity function is removed.

        In the following examples the height and width are assumed to range from zero to
        ten, and similarity of either is computed linearly, as the difference between them
        normalized by the maximum length of ten. The colors pink and red are considered
        50% similar, and all other color pairs are similar only if identical, with the
        similarity weighted half as much as the usual default.

        >>> a.similarity(["height", "width"], lambda v1, v2: 1 - abs(v1 - v2) / 10)
        >>> def color_similarity(c1, c2):
        ...     if c1 == c2:
        ...         return 1
        ...     elif c1 in ("red", "pink") and c2 in ("red", "pink"):
        ...         return 0.5
        ...     else:
        ...         return 0
        ...
        >>> a.similarity("color", color_similarity, 0.5)

        """
        self._memory.similarity((pyactup.Memory._ensure_slot_names(attributes)
                                 or [ "_decision" ]),
                                function,
                                weight)
        try:
            self._memory.index = self._preferred_index()
        except RuntimeError:
            pass


class Plot():

    def __init__(self, column, description, limit_range=False):
        self._column = column
        self._description = description
        self._default_ylim = (-0.05, 1.05) if limit_range else None

    def get_data(self, data, include, exclude, min, max):
        raise NotImplementedError()


class ChoicePlot(Plot):
    def __init__(self, column, description, limit_range=False):
        super().__init__(column, description, limit_range)

    def get_data(self, data, include, exclude, min, max, earliest, latest):
        counters = {}
        sums = defaultdict(int)
        times = set()
        for ch in data.choice.unique():
            d = data[data.choice==ch]
            if earliest is not None:
                d = d[d.time >= earliest]
            if latest is not None:
                d = d[d.time <= latest]
            grouped = d.groupby(["iteration", "time"])
            # There's probably a better way to do this within Pandas, right?
            cnt = Counter([t for (i, t), ignore in grouped[["iteration", "time"]]])
            times.update(cnt.keys())
            for k in cnt:
                sums[k] += cnt[k]
            counters[ch] = cnt
        result = {}
        for ch, cnt in counters.items():
            if include and ch not in include:
                continue
            if exclude and ch in exclude:
                continue
            for k, v in cnt.items():
                cnt[k] /= sums[k]
            for t in times:
                if t not in cnt:
                    cnt[t] = 0
            result[ch] = tuple(zip(*sorted(cnt.items())))
        return result


class OptionPlot(Plot):
    def __init__(self, column, description, limit_range=False):
        super().__init__(column, description, limit_range)

    def get_data(self, data, include, exclude, min, max, earliest, latest):
        result = {}
        for opt in data.option.unique():
            if include and opt not in include:
                continue
            if exclude and opt in exclude:
                continue
            d = data[(data.option==opt)]
            if earliest is not None:
                d = d[d.time >= earliest]
            if latest is not None:
                d = d[d.time <= latest]
            grouped = d.groupby("time")
            result[opt] = ([t for t, ignore in grouped["time"]],
                           grouped[self._column].mean())
        return result


class InstancePlot(Plot):
    def __init__(self, column, description, limit_range=False):
        super().__init__(column, description, limit_range)

    def get_data(self, data, include, exclude, min, max, earliest, latest):
        result = {}
        for opt, util in data[["option", "utility"]].drop_duplicates().itertuples(index=False, name=None):
            if include and opt not in include:
                continue
            if exclude and opt in exclude:
                continue
            if min and util < min:
                continue
            if max and util > max:
                continue
            d = data[(data.option==opt) & (data.utility==util)]
            if earliest is not None:
                d = d[d.time >= earliest]
            if latest is not None:
                d = d[d.time <= latest]
            grouped = d.groupby("time")
            result[f"{opt}, {util}"] = ([t for t, ignore in grouped["time"]],
                                        grouped[self._column].mean())
        return result


class DelayedResponse:
    """A representation of an intermediate state of the computation of a decision, as returned from :meth:`respond` called with no arguments.
    """

    def __init__(self, agent, attributes, expectation):
        self._agent = agent
        self._time = agent.time
        self._attributes = attributes
        self._resolved = False
        self._expectation = expectation
        self._outcome = expectation

    @property
    def is_resolved(self):
        """Whether or not ground truth feedback to the :class:`Agent` regarding this decision has yet been delivered by the user.
        """
        return self._resolved

    @property
    def expectation(self):
        """ The expected value learned when this :class:`DelayedReponse` was created.
        """
        return self._expectation

    @property
    def outcome(self):
        """The most recent response learned by the :class:`Agent` for this decision.
        When :attr:`is_resolved` is ``False`` this will be the reward expected by the
        :class:`Agent` when the decision was made. After it has been resolved by calling
        :meth:`update`, delivering the ground truth reward, this will be that real value.
        """
        return self._outcome

    def update(self, outcome):
        """Replaces the current reward learned, either expected or ground truth, by a new ground truth value.

        The *outcome* is a real number. Typically this value replaces that learned when
        :meth:`respond` was called, though it
        might instead replace the value supplied by an earlier call to :meth:`update`.
        It is always learned at the time of the original call to :meth:`respond`.

        The most recent previous value of the learned reward, either the expected value,
        or that set by a previous call of :meth:`update`, is returned.

        Raises a :exc:`ValueError` if *outcome* is not a real number.

        Because of noise the results returned by :attr:`choose` are stochastic the results
        of running the following examples will differ in their details from those shown.

        >>> a = Agent(default_utility=10, default_utility_populates)
        >>> a.choose(["a", "b"])
        'b'
        >>> a.respond(2)
        >>> a.choose()
        'a'
        >>> a.respond(3)
        >>> a.choose()
        'a'
        >>> r = a.respond()
        >>> a.choose()
        'a'
        >>> a.respond(7)
        >>> a.instances()
        +----------+-------------------+---------+-------------+
        | decision |      outcome      | created | occurrences |
        +----------+-------------------+---------+-------------+
        |    a     |         10        |    0    |     [0]     |
        |    b     |         10        |    0    |     [0]     |
        |    b     |         2         |    1    |     [1]     |
        |    a     |         3         |    2    |     [2]     |
        |    a     | 8.440186635799552 |    3    |     [3]     |
        |    a     |         7         |    4    |     [4]     |
        +----------+-------------------+---------+-------------+
        >>> r.update(1)
        8.440186635799552
        >>> a.instances()
        +----------+---------+---------+-------------+
        | decision | outcome | created | occurrences |
        +----------+---------+---------+-------------+
        |    a     |    10   |    0    |     [0]     |
        |    b     |    10   |    0    |     [0]     |
        |    b     |    2    |    1    |     [1]     |
        |    a     |    3    |    2    |     [2]     |
        |    a     |    1    |    3    |     [3]     |
        |    a     |    7    |    4    |     [4]     |
        +----------+---------+---------+-------------+
        """
        outcome = Agent._outcome_value(outcome)
        old = self._outcome
        self._agent._memory.forget(Agent._add_utility(self._attributes, self._outcome), self._time)
        self._agent._at_time(self._time,
                             lambda: self._agent._memory.learn(Agent._add_utility(self._attributes, outcome)))
        self._resolved = True
        self._outcome = outcome
        return old


def positive_linear_similarity(x, y):
    """Returns a similarity value of two positive :class:`Real` numbers, scaled linearly by the larger of them.
If *x* and *y* are equal the value is one, and otherwise a positive float less than one
that gets smaller the greater the difference between *x* and *y*.

If either *x* or *y* is not positive a :exc:`ValueError` is raised.

>>> positive_linear_similarity(1, 2)
0.5
>>> positive_linear_similarity(2, 1)
0.5
>>> positive_linear_similarity(1, 10)
0.09999999999999998
>>> positive_linear_similarity(10, 100)
0.09999999999999998
>>> positive_linear_similarity(1, 2000)
0.0004999999999999449
>>> positive_linear_similarity(1999, 2000)
0.9995
>>> positive_linear_similarity(1, 1)
1
>>> positive_linear_similarity(0.001, 0.002)
0.5
>>> positive_linear_similarity(10.001, 10.002)
0.9999000199960006
"""
    if x <= 0 or y <= 0:
        raise ValueError(f"the arguments, {x} and {y}, are not both positive")
    if x == y:
        return 1
    if x > y:
        x, y = y, x
    return 1 - (y - x) / y

def positive_quadratic_similarity(x, y):
    """Returns a similarity value of two positive :class:`Real` numbers, scaled quadratically by the larger of them.
If *x* and *y* are equal the value is one, and otherwise a positive float less than one
that gets smaller the greater the difference between *x* and *y*.

If either *x* or *y* is not positive a :exc:`ValueError` is raised.

>>> positive_quadratic_similarity(1, 2)
0.25
>>> positive_quadratic_similarity(2, 1)
0.25
>>> positive_quadratic_similarity(1, 10)
0.009999999999999995
>>> positive_quadratic_similarity(10, 100)
0.009999999999999995
>>> positive_quadratic_similarity(1, 2000)
2.4999999999994493e-07
>>> positive_quadratic_similarity(1999, 2000)
0.9990002500000001
>>> positive_quadratic_similarity(1, 1)
1
>>> positive_quadratic_similarity(0.001, 0.002)
0.25
>>> positive_quadratic_similarity(10.001, 10.002)
0.9998000499880025
"""
    return positive_linear_similarity(x, y)**2

def bounded_linear_similarity(minimum, maximum):
    """Returns a function of two arguments that returns a similarity value reflecting a linear scale between *minimum* and *maximum*.
The two arguments to the function returned should be :class:`Real` numbers between
*minimum* and *maximum*, inclusive. If the two arguments to the function returned are
equal they are maximally similar, and one is returned. If the absolute value of their
difference is as large as possible, they are maximally different, and zero is returned.
Otherwise a scaled value on a linear scale between these two extrema, measuring the
magnitude of the difference between the arguments to the returned function is used, a
value between zero and one being returned.

Raises a :exc:`ValueError` if either *minimum* or *maximum* is not a Real number, or if
*minimum* is not less than *maximum*.

When the returned function is called if either of its arguments is not a Real number a
:exc:`ValueError` is then raised. If either of those arguments is less than *minimum*,
or greater than *maximum*, a warning is issued, and either *minimum* or *maximum*,
respectively, is instead used as the argument's value.

>>> f = bounded_linear_similarity(-1, 1)
>>> f(0, 1)
0.5
>>> f(-0.1, 0.1)
0.9
>>> f(-1, 1)
0.0
>>> f(0, 0)
1.0
>>> sys.float_info.epsilon
2.220446049250313e-16
>>> f(0, _)
0.9999999999999999

    """
    if minimum >= maximum:
        raise ValueError(f"minimum, {minimum}, is not less than maximum, {maximum}")
    def _similarity(x, y):
        if x < minimum:
            warn(f"{x} is less than {minimum}, so {minimum} is instead being used in computing similarity")
            x = minimum
        elif x > maximum:
            warn(f"{x} is greater than {maximum}, so {maximum} is instead being used in computing similarity")
            x = maximum
        if y < minimum:
            warn(f"{y} is less than {minimum}, so {minimum} is instead being used in computing similarity")
            y = minimum
        elif y > maximum:
            warn(f"{y} is greater than {maximum}, so {maximum} is instead being used in computing similarity")
            y = maximum
        return 1 - abs(x - y) / abs(maximum - minimum)
    return _similarity

def bounded_quadratic_similarity(minimum, maximum):
    """Returns a function of two arguments that returns a similarity value reflecting a quadratic scale between *minimum* and *maximum*.
Both arguments to the function returned should be :class:`Real` numbers between *minimum*
and *maximum*, inclusive. If the two arguments to the function returned are equal they are
maximally similar, and one is returned. If the absolute value of their difference is as
large as possible, they are maximally different, and zero is returned. Otherwise a scaled
value on a quadratic scale between these two extrema, measuring the magnitude of the
difference between the arguments to the returned function is used, a value between zero
and one being returned.

Raises a :exc:`ValueError` if either *minimum* or *maximum* is not a Real number, or if
*minimum* is not less than *maximum*.

When the returned function is called if either of its arguments is not a Real number a
:exc:`ValueError` is then raised. If either of those arguments is less than *minimum*,
or greater than *maximum*, a warning is issued, and either *minimum* or *maximum*,
respectively, is instead used as the argument's value.

>>> f = bounded_quadratic_similarity(-1, 1)
>>> f(0, 1)
0.25
>>> f(-0.1, 0.1)
0.81
>>> f(-1, 1)
0.0
>>> f(0, 0)
1.0
>>> sys.float_info.epsilon
2.220446049250313e-16
>>> f(0, _)
0.9999999999999998

    """
    f = bounded_linear_similarity(minimum, maximum)
    return lambda x, y: f(x, y)**2


# Local variables:
# fill-column: 90
# End:
