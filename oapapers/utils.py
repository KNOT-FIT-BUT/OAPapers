# -*- coding: UTF-8 -*-
"""
Created on 15.06.23

:author:     Martin Dočekal
"""
import bisect
import ctypes
import multiprocessing

import numpy as np
import scipy
from microdict import mdict
from typing import Dict, Sequence, Union, TypeVar, Optional, Mapping, Iterable, Tuple, Iterator, List

from windpyutils.generic import arg_sort
from windpyutils.structures.sorted import SortedMap, Comparable


def convert_dict_to_mdict(original: Dict, mdict_type: str) -> mdict:
    """
    Converts dictionary to microdict.

    :param original: Original dictionary.
    :param mdict_type: Type of microdict. Such as "i64:i64".
    :return: converted microdict
    """

    new_mapping = mdict.create(mdict_type)

    new_mapping.update(original)

    return new_mapping


class SharedSortedMapOfSequencesOfIntegers(Mapping[int, Sequence[int]]):
    """
    Memory efficient sorted map using shared memory for values that are sequences of integers. It is also using
    shared memory for keys that are also integers.
    """

    def __init__(self, init_values: Optional[Union[Mapping[int, Sequence[int]], Iterable[Tuple[int, Sequence[int]]]]]):
        """
        :param init_values: voluntary initial values for the map.
        """

        self.keys_storage = []
        self.values_storage = []

        if init_values is not None:
            if isinstance(init_values, Mapping):
                self.keys_storage = list(init_values.keys())
                values = list(init_values.values())
            else:
                self.keys_storage, values = zip(*init_values)
            # sort keys
            sorted_indices = arg_sort(self.keys_storage)

            self.keys_storage = [self.keys_storage[i] for i in sorted_indices]
            self.values_storage = [values[i] for i in sorted_indices]

        self.keys_storage = multiprocessing.Array(ctypes.c_int64, self.keys_storage, lock=False)

        values_storage = []
        start_offsets = []
        for v in self.values_storage:
            start_offsets.append(len(values_storage))
            values_storage.extend(v)

        # create views
        self.values_storage = multiprocessing.Array(ctypes.c_int64, values_storage, lock=False)
        self.start_offsets = multiprocessing.Array(ctypes.c_int64, start_offsets, lock=False)

    def __getitem__(self, key: int) -> List[int]:
        insert_index, already_in = self.insertions_index(key)
        if not already_in:
            raise KeyError(f"Key {key} is not in the map.")

        start_offset = self.start_offsets[insert_index]
        end_offset = self.start_offsets[insert_index + 1] if insert_index + 1 < len(self.start_offsets) else len(self.values_storage)
        return self.values_storage[start_offset:end_offset]

    def __iter__(self) -> Iterator[int]:
        return iter(self.keys_storage)

    def __len__(self) -> int:
        return len(self.keys_storage)

    def insertions_index(self, x: int) -> Tuple[int, bool]:
        """
        Returns insertions index for given value that remains the value sorted and flag that signalizes whether the
        value is already in.

        :param x: value for which the insertion point should be found
        :return: insertion index and already in flag
        """
        try:
            searched_i = bisect.bisect_left(self.keys_storage, x)
        except TypeError:
            raise KeyError(f"Key {x} is not in the map.")

        try:
            on_index = self.keys_storage[searched_i]
            if on_index == x:
                return searched_i, True
        except IndexError:
            pass

        return searched_i, False


class SharedCSRMatrixFactory:
    """
    Factory for shared CSR matrices
    """

    def __init__(self, matrix: scipy.sparse.csr_matrix):
        """
        :param matrix: Original matrix.
        """

        # copy data to shared memory

        self.data = multiprocessing.Array(np.ctypeslib.as_ctypes_type(matrix.data.dtype), matrix.data, lock=False)
        self.indices = multiprocessing.Array(np.ctypeslib.as_ctypes_type(matrix.indices.dtype), matrix.indices, lock=False)
        self.indptr = multiprocessing.Array(np.ctypeslib.as_ctypes_type(matrix.indptr.dtype), matrix.indptr, lock=False)
        self.shape = matrix.shape

    def create(self) -> scipy.sparse.csr_matrix:
        """
        Creates sparse matrix using shared memory.

        :return: Created shared matrix.
        """
        return scipy.sparse.csr_matrix((self.data, self.indices, self.indptr), shape=self.shape, copy=False)


