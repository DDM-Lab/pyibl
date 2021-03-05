# Copyright 2014-2021 Carnegie Mellon University

from setuptools import setup
import re

with open("pyibl.py") as f:
    VERSION = re.search(r"""^\s+__version__\s*=\s*['"]([0-9]+\.[0-9]+(?:\.[0-9]+)?(?:\.dev[0-9]+)?)['"]\s*$""",
                        f.read(),
                        re.MULTILINE).group(1)

with open("README.md") as f:
    DESCRIPTION = f.read()

setup(name="pyibl",
      version=VERSION,
      description="A Python implementation of a subset of Instance Based Learning Theory",
      license="Free for research purposes",
      author="Dynamic Decision Making Laboratory of Carnegie Mellon University",
      author_email="dfm2@cmu.edu",
      url="http://pyibl.ddmlab.com/",
      platforms=["any"],
      long_description=DESCRIPTION,
      long_description_content_type="text/markdown",
      py_modules=["pyibl"],
      install_requires=[
          "pyactup>=1.1.2",
          "ordered_set",
          "prettytable",
          "packaging"],
      tests_require=["pytest"],
      python_requires=">=3.7",
      classifiers=["Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3 :: Only",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Operating System :: OS Independent"])
