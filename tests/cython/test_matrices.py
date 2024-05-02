# -*- coding: UTF-8 -*-
""""
Created on 01.09.23

:author:     Martin Dočekal
"""

import unittest

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from oapapers.cython.matrices import top_k_sparse_inner_product


class TestTopKSparseInnerProduct(unittest.TestCase):
    def test_wrong_shape(self):
        with self.assertRaises(AssertionError):
            q = csr_matrix([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
            d = csc_matrix([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
            top_k_sparse_inner_product(q, d, 1)

    def test_positive_k(self):
        with self.assertRaises(AssertionError):
            q = csr_matrix([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
            d = csc_matrix([[1, 0], [2, 3], [0, -3]], dtype=np.float32)
            top_k_sparse_inner_product(q, d, 0)

    def test_empty_query(self):
        with self.assertRaises(AssertionError):
            q = csr_matrix([], dtype=np.float32)
            d = csc_matrix([[1, 0], [2, 3], [0, -3]], dtype=np.float32)
            top_k_sparse_inner_product(q, d, 1)

    def test_empty_documents(self):
        with self.assertRaises(AssertionError):
            q = csr_matrix([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
            d = csc_matrix([], dtype=np.float32)
            top_k_sparse_inner_product(q, d, 1)

    def test_compute(self):
        q = csr_matrix([
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 1]
        ], dtype=np.float32)
        d = csc_matrix([
            [2, 0, 4],
            [1.5, 3, 0],
            [0, -1, 1]
        ], dtype=np.float32)
        res = top_k_sparse_inner_product(q, d, 2)
        self.assertSequenceEqual([[2, 0], [1, 0], [2, 0]], res[0].tolist())
        self.assertSequenceEqual([[4.0, 2.0], [2.0, 1.5], [1.0, 0.0]], res[1].tolist())

        res = top_k_sparse_inner_product(q, d, 1)
        self.assertSequenceEqual([[2], [1], [2]], res[0].tolist())
        self.assertSequenceEqual([[4.0], [2.0], [1.0]], res[1].tolist())

        res = top_k_sparse_inner_product(q, d, 3)
        self.assertSequenceEqual([[2, 0, 1], [1, 0, 2], [2, 0, 1]], res[0].tolist())
        self.assertSequenceEqual([[4.0, 2.0, 0.0], [2.0, 1.5, 1.0], [1.0, 0.0, -1.0]], res[1].tolist())

        res = top_k_sparse_inner_product(q, d, 4)
        self.assertSequenceEqual([[2, 0, 1], [1, 0, 2], [2, 0, 1]], res[0].tolist())
        self.assertSequenceEqual([[4.0, 2.0, 0.0], [2.0, 1.5, 1.0], [1.0, 0.0, -1.0]], res[1].tolist())



if __name__ == '__main__':
    unittest.main()
