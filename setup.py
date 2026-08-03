# Copyright 2014-2026 Carnegie Mellon University

import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")
CORE_TEXT = (ROOT / "pyibl" / "core.py").read_text(encoding="utf-8")
VERSION = re.search(r'^__version__\s*=\s*"([^"]+)"', CORE_TEXT, re.MULTILINE).group(1)

setup(name="pyibl",
      version=VERSION,
      description="A Python implementation of a subset of Instance Based Learning Theory",
      license="Free for research purposes",
      author="Dynamic Decision Making Laboratory of Carnegie Mellon University",
      author_email="dfm2@cmu.edu",
      url="https://ddm-lab.github.io/pyibl-documentation/",
      platforms=["any"],
      long_description=DESCRIPTION,
      long_description_content_type="text/markdown",
      packages=find_packages(include=["pyibl", "pyibl.*", "pyactup", "pyactup.*"]),
      install_requires=[
          "numpy",
          "pylru",
          "ordered_set",
          "packaging"],
      extras_require={
          "plotting": ["matplotlib", "pandas"],
          "tables": ["prettytable"],
          "progress": ["tqdm"],
          "embedding": ["sentence-transformers"],
          "all": ["matplotlib", "pandas", "prettytable", "tqdm", "sentence-transformers"],
      },
      tests_require=["pytest"],
      python_requires=">=3.8",
      classifiers=["Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3 :: Only",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10",
                   "Programming Language :: Python :: 3.11",
                   "Operating System :: OS Independent"])
