Embedding-Based Scam Title Example
**********************************

This tutorial demonstrates using a Hugging Face text embedding model with
PyIBL attribute similarity. The key idea is:

Install the optional dependency first:

.. code-block:: bash

   pip install -e .[embedding]

- ``agent.embedding(function=...)`` registers the embedding model
- ``agent.embedding("title")`` enables embedding-based similarity for that attribute
- labeled examples are populated as positive/negative utilities
- prediction is a ``choose()`` between ``scam`` and ``safe`` label options

The complete runnable example is in
``examples/embedding/scam_tutorial.py``, with training data in
``examples/embedding/scam_titles.csv``.

.. literalinclude:: ../examples/embedding/scam_tutorial.py
   :language: python
