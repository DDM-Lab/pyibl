Internals
=========

.. _internals:


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

The activation, :math:`A_{i}(t)` of instance *i* at time *t* is a sum of three
components,

  .. math:: A_{i}(t) = B_{i}(t) + M_{i} + \epsilon

the base-level activation, the partial matching correction, and the activation noise.

Base-level activation
~~~~~~~~~~~~~~~~~~~~~

The base-level activation, :math:`B_{i}(t)`, describes the frequency and recency of the instance *i*,
and depends upon the :attr:`decay` parameter of the :class:`Agent`, *d*. In the normal case, when the
agent's :attr:`optimized_learning` parameter is ``False``, the base-level activation is computed using
the amount of time that has elapsed since each of the past experiences of *i*; in the following this
set of times of experiences of *i* before *t* is denoted by :math:`\mathcal{T}_{i}(t)`.


  .. math:: B_{i}(t) = \ln \left( \sum_{t' \in \mathcal{T}_{i}(t)} (t - t')^{-d} \right)

If the agent's :attr:`optimized_learning` parameter is ``True`` an approximation is used instead, sometimes less taxing of
computational resources. It is particularly useful if the same instances are expected to be seen many times, and assumes
that repeated experiences of the various instances are distributed roughly evenly over time.
Instead of using the times of all the past occurrences of *i*, it uses :math:`t_{0}`, the time of
the first appearance of *i*, and :math:`n_i(t)`, a count of the number of times *i* has appeared before time *t*.

  .. math:: B_{i}(t) = \ln(\frac{n_{i}(t)}{1 - d}) - d \ln(t - t_{0})

The ``optimized_learning`` parameter may also be set to a positive integer. This specifies a number of most recent
reinforcements of a chunk to be used to compute the base-level activation in the normal way, with the contributions
of any older than those approximated using a formula similar to the preceding.

Note that setting the ``decay`` parameter to ``None`` disables the computation of base-level
activation. That is, the base-level component of the total activation is zero in this case.

Partial Matching
~~~~~~~~~~~~~~~~

If the agent's :attr:`mismatch_penalty` parameter is ``None``, the partial matching correction, :math:`M_{i}`, is zero.
Otherwise :math:`M_{i}` depends upon the similarities of the attributes of the instance to those attributes
being sought in the retrieval and the value of the `mismatch_penalty` parameter.

PyIBL represents similarities as real numbers between zero and one, inclusive, where two values being completely similar, ``==``,,
has a value of one; and being completely dissimilar has a value of zero; with various other degrees of similarity being
positive, real numbers less than one.

How to compute the similarity of two instances is determined by the programmer, using the
method :meth:`similarity`.
A function is supplied to this method to be applied to values of the
attributes of given names, this function returning a similarity value. In addition, the ``similarity`` method
can assign a weight, :math:`\omega`, to these attributes, allowing the mismatch contributions of multiple attributes
to be scaled with respect to one another. If not explicitly supplied this weight defaults to one.

If the ``mismatch`` parameter has positive real value :math:`\mu`, the similarity of attribute *j* of
instance *i* to the desired
value of that attribute is :math:`S_{ij}`, the similarity weight of attribute *j* is :math:`\omega_{j}`,
and the set of all attributes for which a similarity function is defined is :math:`\mathcal{F}`,
the partial matching correction is

  .. math:: M_{i} = \mu \sum_{j \in \mathcal{F}} \omega_{j} (S_{ij} - 1)

The value of :math:`\mu` should be positive, and thus :math:`M_{i}` is negative, so it is not so much that increased
similarities increase the activation as dissimilarities reduce it, and increased similarities simply cause it
to be reduced less, scaled by the value of :math:`\mu`.

Attributes for which no similarity function is defined are always matched exactly, non-matching instances not
being considered at all.

Activation noise
~~~~~~~~~~~~~~~~

The activation noise, :math:`\epsilon`, implements the stochasticity of retrievals from an agent's memory.
It is sampled from a distribution, scaled by the ``noise`` parameter. This sampling occurs every time the
activation of an instance needs to be calculated, and is typically different each time.

By default this distribution is a logistic distribution centered on zero. The distribution used can be
changed for special purposes by using an :class:`Agent`’s ``noise_distribution`` attribute, though for
nearly all uses the default is the right choice.

An :class:`Agent` has a scale parameter, ``noise``. If this parameter is denoted as :math:`\sigma`, and
if the value sampled from the distribution is :math:`\xi`, the activation noise is

  .. math:: \epsilon = \sigma \xi

Note that setting the ``noise`` parameter to zero results in supplying
no noise to the activation. This does not quite make operation of
PyIBL deterministic, since retrievals of instances with the same
activations are resolved randomly.


Blending
--------

Once the activations of all the relevant instances have been computed, they are used to compute
a blended value of the utility, an average of the utilities of those instances weighted by a function
of the instances' activations, the probability of retrieval.

A parameter, the :attr:`temperature`, or :math:`\tau`, is used in constructing this blended value.
In PyIBL the value of this parameter is by default the :attr:`noise` parameter used for activation noise,
multiplied by :math:`\sqrt{2}`. However it can be set independently of the ``noise``, if preferred, and is
often set to ``1``.

For a given option being considered, *k*, let :math:`\mathcal{M}_{k}` be the set of all matching instances.
Then the probability of retrieval of instance :math:`i \in \mathcal{M}_{k}` at time *t* is

  .. math:: P_{i}(t) = \frac{e^{A_{i}(t) / \tau}}{\sum_{i' \in \mathcal{M}_{k}}{e^{A_{i'}(t) / \tau}}}

From these we can compute the blended value at time *t*, :math:`V_{k}(t)` of this option's various utilities in the
instances for this option. If :math:`u_{i}` is the utility that was provided as the first argument in the call to
:meth:`respond` that completed the experience of instance *i*, or equivalently was supplied by a call to
:meth:`populate` or with the :attr:`default_utility`, this blended value is

  .. math:: V_{k}(t) = \sum_{i \in \mathcal{M}_{k}}{P_{i}(t) u_{i}}
