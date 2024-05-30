.. Copyright 2014-2024 Carnegie Mellon University

.. meta::
   :description: Documentation of PyIBL, a Python implementation of a subset of Instance Based Learning Theory for modeling decisions from experience
   :keywords lang=en: pyibl python ibl iblt instance learning act-r cmu ddmlab decisions experience cognitive model modeling

.. ᵢ_introduction

PyIBL is a Python implementation of a subset of Instance Based Learning Theory (IBLT) [#f1]_.
It is made and distributed by the
`Dynamic Decision Making Laboratory <http://ddmlab.com>`_
of
`Carnegie Mellon University <http://cmu.edu>`_
for making computational cognitive models supporting research
in how people make decisions in dynamic environments.
Here is documented version |version| of PyIBL.

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

.. image:: _static/front_image.png

|
|

Contents
********

.. toctree::
   :maxdepth: 3

   Introduction <self>
   installation
   tutorial
   reference
   internals
   examples
   changes
