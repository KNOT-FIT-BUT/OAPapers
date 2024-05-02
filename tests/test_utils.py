# -*- coding: UTF-8 -*-
"""
Created on 24.08.23

:author:     Martin Dočekal
"""
import copy
from unittest import TestCase

from scipy.sparse import csr_matrix

from oapapers.utils import SharedSortedMapOfSequencesOfIntegers, SharedCSRMatrixFactory


class TestSharedSortedMapOfSequencesOfIntegers(TestCase):
    def setUp(self) -> None:
        self.filled = SharedSortedMapOfSequencesOfIntegers({10: [10, 11, 12], 9: [9, 8], 8: [8, 9], 7: []})

    def test_len(self):
        self.assertEqual(4, len(self.filled))

    def test_in(self):
        self.assertTrue(10 in self.filled)
        self.assertTrue(7 in self.filled)
        self.assertTrue(8 in self.filled)
        self.assertTrue(9 in self.filled)
        self.assertFalse(99 in self.filled)
        self.assertFalse(None in self.filled)

    def test_getitem(self):
        self.assertEqual([10, 11, 12], self.filled[10])
        self.assertEqual([], self.filled[7])
        self.assertEqual([8, 9], self.filled[8])
        self.assertEqual([9, 8], self.filled[9])

        with self.assertRaises(KeyError):
            self.filled[99]
        with self.assertRaises(KeyError):
            self.filled[None]

    def test_iter(self):
        self.assertSequenceEqual([7, 8, 9, 10], list(self.filled))

    def test_keys(self):
        self.assertSequenceEqual([7, 8, 9, 10], list(self.filled.keys()))

    def test_values(self):
        self.assertSequenceEqual([[], [8, 9], [9, 8], [10, 11, 12]], list(self.filled.values()))

    def test_items(self):
        self.assertSequenceEqual([(7, []), (8, [8, 9]), (9, [9, 8]), (10, [10, 11, 12])], list(self.filled.items()))


class TestSharedCSRMatrixFactory(TestCase):

    def setUp(self) -> None:
        self.matrix = csr_matrix([[1, 0, 0], [0, 5, 0], [0, 0, 9]])
        self.shared = SharedCSRMatrixFactory(copy.deepcopy(self.matrix))

    def test_create(self):
        created = self.shared.create()
        self.assertEqual(self.matrix.shape, created.shape)
        self.assertSequenceEqual(list(self.matrix.data), list(created.data))
        self.assertSequenceEqual(list(self.matrix.indices), list(created.indices))
        self.assertSequenceEqual(list(self.matrix.indptr), list(created.indptr))