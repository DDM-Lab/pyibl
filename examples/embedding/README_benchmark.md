Embedding Speed Benchmark
=========================

This benchmark compares four engines on the same large embedding-style scam-title task:

- `ibl6_local`: this repository's current PyIBL 6.0 refactor
- `ibl4_speedy_repo`: legacy PyIBL 4.1.2 from `SpeedyIBL/Codes/pyibl.py`
- `speedyibl_repo`: SpeedyIBL from `SpeedyIBL/Codes/speedyibl.py`
- `pyibl5_pip`: pip-installed PyIBL 5.x in a separate Python environment

How it works
------------

1. Expands `scam_titles.csv` into a larger synthetic dataset.
2. Encodes unique titles once with sentence-transformers.
3. Precomputes a cosine similarity matrix over title IDs.
4. Runs each engine in a subprocess with adapter wrappers and identical
   `runs x epochs` loops.
5. Prints elapsed time, throughput, and accuracy.

Why this is fair
----------------

- Embedding compute is done once before timing each engine loop.
- All engines consume the same title IDs and the same similarity values.
- Each engine receives equivalent choice/respond workloads.

Requirements
------------

- In the current repo environment:
  - `sentence-transformers`
  - `numpy`
- For `pyibl5_pip` comparison:
  - a separate Python executable where `pyibl==5.*` is installed

Example setup for pyibl 5.x
---------------------------

```bash
python3 -m venv .venv-pyibl5
.venv-pyibl5/bin/pip install --upgrade pip
.venv-pyibl5/bin/pip install "pyibl>=5,<6" numpy
```

Example setup for legacy IBL4 adapter
-------------------------------------

```bash
python3 -m venv .venv-ibl4
.venv-ibl4/bin/pip install --upgrade pip
.venv-ibl4/bin/pip install "pyactup==1.1.4" ordered-set prettytable numpy
```

Run the comparison
------------------

From repository root:

```bash
python examples/embedding/benchmark_embedding_speed.py \
  --dataset-size 4000 \
  --runs 10 \
  --epochs 20 \
  --write-expanded-csv examples/embedding/scam_titles_expanded.csv \
  --ibl4-python .venv-ibl4/bin/python \
  --pyibl5-python .venv-pyibl5/bin/python
```

Run a subset of engines
-----------------------

```bash
python examples/embedding/benchmark_embedding_speed.py \
  --engines ibl6_local speedyibl_repo ibl4_speedy_repo \
  --dataset-size 3000 --runs 5 --epochs 10
```

Notes
-----

- The SpeedyIBL repository implementation uses a different option format than PyIBL.
  The benchmark script adapts this interface internally.
- `pyibl5_pip` is skipped unless `--pyibl5-python` is provided.
- You can isolate dependency stacks per engine with:
  `--ibl6-python`, `--ibl4-python`, `--speedy-python`, and `--pyibl5-python`.