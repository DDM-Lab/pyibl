"""Chunk data structure used by the local PyIBL memory engine."""

import numpy as np

__all__ = ["Chunk"]


class Chunk(dict):
    """A learned item.

    A chunk acts much like a dictionary, and its slots can be retrieved with the
    usual ``[]`` notation, or with ``.get()``.
    """

    __slots__ = ["_name", "_memory", "_creation", "_references", "_reference_count"]

    _name_counter = 0

    def __init__(self, memory, content):
        self._name = f"{Chunk._name_counter:04d}"
        Chunk._name_counter += 1
        self._memory = memory
        self.update(content)
        self._creation = memory._time
        self._references = np.empty(
            1 if self._memory._optimized_learning != 0 else 0,
            dtype=np.float64,
        )
        self._reference_count = 0

    def __repr__(self):
        return "<Chunk {} {} {}>".format(self._name, dict(self), self._reference_count)

    def __str__(self):
        return f"Chunk-{self._name}"

    @property
    def memory(self):
        """The Memory object that contains this chunk."""
        return self._memory

    @property
    def reference_count(self):
        """A non-negative integer count of chunk reinforcements."""
        return self._reference_count

    @property
    def references(self):
        """Times at which this chunk has been reinforced."""
        return tuple(
            self._references[
                : (
                    self._reference_count
                    if self._memory._optimized_learning is None
                    else min(self._reference_count, self._memory._optimized_learning)
                )
            ]
        )
