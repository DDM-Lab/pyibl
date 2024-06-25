Changes
****************

.. _chanages:


Version 5.1
===========

from version 5.1.4 to 5.1.5
---------------------------

* improved the output of the instances() method
* adjusted some issues raised by changes to recent versions of several of PyIBL’s dependencies
* improved the documentation


from version 5.1.3 to 5.1.4
---------------------------

* Fixed a bug exhibited when :attr:`default_utility` was set to exactly zero.

from version 5.1.1 to 5.1.3
---------------------------

* Added :attr:`noise_distribution`.


from version 5.1 to 5.1.1
-------------------------

* Fixed a bug which caused some rows to be omitted from the :attr:`aggregate_details`;
  since the :meth:`plot` method depends upon the :attr:`aggregate_details` plots weere
  sometimes incorrect, too.


Version 5.0
===========

from version 5.0.4 to 5.1
-------------------------

* Added the :attr:`aggregate_details` attribute.
* Added the :meth:`plot` method.

from version 5.0.2 to 5.0.4
---------------------------

* The sequence of all attribute names is now a tuple of strings rather than a list.

from version 5.0.1 to 5.0.3
---------------------------

* Add further information to trace output.
* Update some copyrights and documentation.

from version 5.0 to 5.0.1
-------------------------

* Update copyrights, documentation and dependencies.
* Fix a bug afflicting tracing when noise is zero.


from version 4.2 to 5.0
-----------------------

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


Older versions
==============

from version 4.2 to  4.2.0.1
----------------------------

* PyIBL is now distributed via PyPi and need no longer be downloaded from the DDMLab website.


from version 4.1 to  4.2
------------------------

* The :meth:`choose2` method has been added to the API.
* The :meth:`respond` method now takes a second, optional argument.
* There is more flexability possible when partially matching attributes.
* PyIBL now requires Pythonn verison 3.7 or later.
* General tidying and minor bug fixes.


from version 4.0 to 4.1
-----------------------

* The API for :class:`DelayedFeedback` has been changed.
* The :meth:`reset()` now has an additional, optional argument, *preserve_prepopulated*.
* Some minor bug fixes.


from version 3.0 to 4.0
-----------------------

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


from version 2.0 to 3.0
-----------------------

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
      preliminary internal development versions of it were used for a
      variety of experiments, both within the DDMLab and elsewhere.

from version 1.0 to 2.0
-----------------------

* Agents are now publicly visible objects that can be passed around and moved from
  one Population to another. The API has completely changed so that you no longer
  cite an agent by name in a Population.
* Options presented to Agents are no longer merely decisions, but include situations as well.
* Logging is configured with strings rather than constants.
* Logging can now be configured to include or exclude unused options and instances.
* Bug fixes, particularly in logging.
* Better documentation.
* General internal tidying


