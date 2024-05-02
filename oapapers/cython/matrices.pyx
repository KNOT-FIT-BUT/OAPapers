# -*- coding: UTF-8 -*-
""""
Created on 01.09.23
Utils for matrices

:author:     Martin Dočekal
"""
import numpy as np
cimport numpy as cnp
from scipy.sparse import csr_matrix, csc_matrix


cdef _top_k_sparse_inner_product(cnp.ndarray[cnp.int32_t, ndim=1] q_indptr,
                                 cnp.ndarray[cnp.int32_t, ndim=1] q_indices,
                                 cnp.ndarray[cnp.float32_t, ndim=1] q_data,
                                 cnp.ndarray[cnp.int32_t, ndim=1] x_indptr,
                                 cnp.ndarray[cnp.int32_t, ndim=1] x_indices,
                                 cnp.ndarray[cnp.float32_t, ndim=1] x_data,
                                 int k):
    """
    Computes top-k inner product between queries and all vectors in x. The result is sorted in descending order.    
    
    It is memory efficient implementation that is not materializing the whole matrix of inner products.
    
    :param q_indptr: CSR sparse matrix indptr of queries in form of numpy array.
    :param q_indices: CSR sparse matrix indices of queries in form of numpy array.
    :param q_data: CSR sparse matrix data of queries in form of numpy array.
    :param x_indptr: CSC sparse matrix indptr of vectors in form of numpy array.
    :param x_indices: CSC sparse matrix indices of vectors in form of numpy array.
    :param x_data: CSC sparse matrix data of vectors in form of numpy array.
    :param k: Number of top scores
        if the number of vectors is less than k, then the key will be set to number of vectors automatically.
    :return:
        indices: Indices of top-k vectors for each query. Shape [n_queries, k]
        values: inner product Values of top-k vectors for each query. Shape [n_queries, k]
    """

    cdef int n_queries = q_indptr.shape[0] - 1
    cdef int n_vectors = x_indptr.shape[0] - 1

    k = min(k, n_vectors)

    cdef cnp.ndarray[cnp.int32_t, ndim=2] indices = np.full((n_queries, k), -1, dtype=q_indices.dtype)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] values = np.full((n_queries, k), -np.inf, dtype=q_data.dtype)

    cdef int q_start, q_end, x_start, x_end
    cdef int i, j, q_row, q_offset
    cdef cnp.float32_t inner_product

    for q_row in range(n_queries):
        q_start = q_indptr[q_row]
        q_end = q_indptr[q_row + 1]

        for x_col in range(n_vectors):
            x_start = x_indptr[x_col]
            x_end = x_indptr[x_col + 1]

            q_offset = q_start

            # compute inner product
            inner_product = 0.0
            for i in range(x_start, x_end):
                while q_offset < q_end and q_indices[q_offset] < x_indices[i]:
                    q_offset += 1

                if q_offset < q_end and q_indices[q_offset] == x_indices[i]:
                    inner_product += q_data[q_offset] * x_data[i]

            # insert into top-k
            j = 0
            while j < k and values[q_row, j] > inner_product:
                j += 1

            if j < k:
                for i in range(k - 1, j, -1):   # shift smaller values to the right
                    values[q_row, i] = values[q_row, i - 1]
                    indices[q_row, i] = indices[q_row, i - 1]

                values[q_row, j] = inner_product
                indices[q_row, j] = x_col

    return indices, values

cpdef top_k_sparse_inner_product(queries: csr_matrix, x: csc_matrix, k: int):
    """
    Computes top-k inner product between queries and all vectors in x. The result is sorted in descending order.
    
    It is memory efficient implementation that is not materializing the whole matrix of inner products.
    
    :param queries: CSR Sparse matrix of queries. Shape [n_queries, n_features]
    :param x: CSC Sparse matrix of vectors. Shape [n_features, n_vectors]
    :param k: Number of top scores
    :return: 
        indices: Indices of top-k vectors for each query. Shape [n_queries, k]
                -1 on given position means that there is not enough vectors in x to fill the top-k.
        values: inner product Values of top-k vectors for each query. Shape [n_queries, k]
    """
    assert queries.shape[1] == x.shape[0], "Queries and vectors must have same number of features."
    assert k > 0, "k must be positive."
    assert queries.shape[0] > 0, "There must be at least one query."
    assert x.shape[1] > 0, "There must be at least one vector."

    return _top_k_sparse_inner_product(queries.indptr, queries.indices, queries.data, x.indptr, x.indices, x.data, k)


