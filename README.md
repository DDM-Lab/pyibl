# PyIBL 6.0

PyIBL 6.0 is the next major evolution of PyIBL focused on three priorities:

1. performance inspired by SpeedyIBL
2. a more mutable and hackable internal architecture
3. modern extensions such as embedding-based similarity and behavior prediction workflows

This branch is an active refactor. The goal is to preserve the familiar Agent-level
API while improving the execution engine underneath it.

## What PyIBL 6.0 Is

PyIBL is a Python implementation of core ideas from Instance Based Learning Theory (IBLT):
an agent chooses, receives outcomes, and updates memory over repeated decisions.

PyIBL 6.0 expands this with:

- embedding-aware similarity for semantic generalization
- prediction-oriented examples and workflows
- a performance roadmap informed by the SpeedyIBL implementation
- a package layout that separates core API surface from internal engine modules

## Design Intent for 6.0

- Keep the user-facing modeling flow simple: choose -> respond.
- Improve runtime speed in hot paths (activation, blending, and retrieval loops).
- Increase mutability for experimentation with alternative memory/similarity strategies.
- Reduce coupling between conceptual layers so internals can evolve faster.

## Installation

Install for local development:

        pip install -e .

Optional extras:

        pip install -e .[plotting]    # matplotlib + pandas
        pip install -e .[tables]      # prettytable
        pip install -e .[progress]    # tqdm
        pip install -e .[embedding]   # sentence-transformers
        pip install -e .[all]         # all optional features

## Project Structure

Top-level layout:

        pyibl/
            pyibl/
                __init__.py
                core.py
                agent.py
                memory.py
                chunk.py
                similarity.py
                plotting.py
            examples/
                embedding/
                prediction/
                binary-choice/
                binary-network/
                box-game/
                rps/
                safe_risky.py
            SpeedyIBL/
                Codes/
                speedyIBL.ipynb
                tutorial_speedyibl.ipynb
            doc/
            test_pyibl.py
            setup.py
            pyproject.toml
            requirements.txt

Module intent:

- pyibl/core.py: primary implementation path and public model behavior.
- pyibl/agent.py: agent-facing API boundary.
- pyibl/memory.py and pyibl/chunk.py: mutable memory data structures and chunk storage.
- pyibl/similarity.py: similarity functions and matching behavior.
- pyibl/plotting.py: optional plotting/dataframe utilities.
- examples/: runnable tutorials and experiment templates.
- SpeedyIBL/: reference implementation used to guide optimization direction.

## Why the SpeedyIBL Folder Is Here

SpeedyIBL is a performance reference used to inform PyIBL 6.0 changes. It helps guide:

- data layout decisions for fast updates
- loop-level optimization opportunities
- tradeoffs between readability, mutability, and speed

PyIBL 6.0 is not a direct copy of SpeedyIBL. The goal is to adapt the best ideas while
retaining PyIBL's usability and model transparency.

## Initial 6.0 Refactor Roadmap

1. Stabilize package boundaries and module responsibilities.
2. Continue migrating engine responsibilities into pyibl-local internals.
3. Benchmark choose/respond throughput and memory growth against baseline behavior.
4. Add targeted tests for mutability-sensitive pathways.
5. Document migration guidance for modelers.

## Documentation and Background

- Existing PyIBL docs: https://ddm-lab.github.io/pyibl-documentation
- Foundational paper: Gonzalez, Lerch, and Lebiere (2003), Cognitive Science 27, 591-635.

## License and Usage

PyIBL is copyright 2014-2026 by Carnegie Mellon University.
It may be freely used and modified for research purposes.
