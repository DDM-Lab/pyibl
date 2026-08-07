# PyIBL 6.0: Instance Embeddings and Behavior Prediction 

## This is a fork of the PyIBL python library that introduces 2 major new components

### Instance Embeddings 

The example of this is in the \examples\embedding folder 

### Behavior Prediction

The example of this is in the \examples\prediction folder 

# PyIBL

PyIBL is a Python implementation of a subset of Instance Based Learning Theory
(IBLT) (Cleotilde Gonzalez, Javier F. Lerch and Christian Lebiere (2003),
Instance-based learning in dynamic decision making, Cognitive Science, 27,
591-635. DOI: 10.1016/S0364-0213(03)00031-4). It is made and distributed by
the Dynamic Decision Making Laboratory of Carnegie Mellon University for
making computational cognitive models supporting research in how people make
decisions in dynamic environments.

PyIBL requires Python version 3.8 or later. PyIBL also works in recent
versions of PyPy.

The latest released version of PyIBL may be installed from PyPi with pip:

    pip install pyibl

For local development as an editable install (so changes in this checkout are
immediately available to environments where you install it):

    pip install -e .

Optional feature dependencies can be installed as extras:

    pip install -e .[plotting]   # matplotlib + pandas for plotting/dataframe features
    pip install -e .[tables]     # prettytable for pretty trace/chunk printing
    pip install -e .[progress]   # tqdm for progress bars in optional scripts

PyIBL now ships a local `pyactup` package in the same repository, so a separate
`pip install pyactup` is not required.

Optional embedding tutorial dependencies can be installed with:

    pip install -e .[embedding]

Repository package layout:

- `pyibl/` contains the PyIBL package sources
- `pyactup/` contains the bundled PyACTUp package sources
- legacy top-level module files are no longer used as runtime sources

For further information, including the documentation see the
[online documentation](https://ddm-lab.github.io/pyibl-documentation).

PyIBL is copyright 2014-2026 by Carnegie Mellon University. It may be
freely used, and modified, but only for research purposes.
