Reference
=========

.. _reference:


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

   .. automethod:: discrete_blend

   .. automethod:: instances

   .. autoattribute:: details

   .. autoattribute:: trace

   .. autoattribute:: aggregate_details

   .. automethod:: plot

   .. autoattribute:: noise_distribution

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

