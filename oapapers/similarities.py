# -*- coding: UTF-8 -*-
"""
Created on 15.03.23

Module with various similarity functions.

:author:     Martin Dočekal
"""
from collections import Counter
from typing import AbstractSet


def similarity_score(n_1: AbstractSet[str], n_2: AbstractSet[str]) -> float:
    """
    Calculates similarities of two sets of tokens.

    The similarity score S is harmonic mean of Jaccard index J and a containment metric C (such as in s2orc paper
    https://aclanthology.org/2020.acl-main.447.pdf).


    S = (2*J*C)/(J+C)
    J = |N1 ∩ N2| / |N1 ∪ N2|
    C = |N1 ∩ N2| / min(|N1|,|N2|)

    :param n_1: first set of tokens
    :param n_2: second set of tokens
    :return: similarity score, 0 for empty
    """

    if len(n_1) == 0 or len(n_2) == 0:
        return 0

    intersect_cardinality = len(n_1 & n_2)
    if intersect_cardinality == 0:
        return 0

    # |N1 ∪ N2| = |N1| + |N2| - |N1 ∩ N2|
    union_cardinality = len(n_1) + len(n_2) - intersect_cardinality

    j = intersect_cardinality / union_cardinality
    c = intersect_cardinality / min(len(n_1), len(n_2))

    return 2 * j * c / (j + c)


def containment_score(n_1: AbstractSet[str], n_2: AbstractSet[str]) -> float:
    """
    Calculates similarities of two sets of tokens.

    C = |N1 ∩ N2| / min(|N1|,|N2|)

    :param n_1: first set of tokens
    :param n_2: second set of tokens
    :return: similarity score, 0 for empty
    """

    if len(n_1) == 0 or len(n_2) == 0:
        return 0

    intersect_cardinality = len(n_1 & n_2)
    if intersect_cardinality == 0:
        return 0

    return intersect_cardinality / min(len(n_1), len(n_2))


def dice_similarity_score(n_1: Counter[str], n_2: Counter[str]) -> float:
    """
    Calculates dice similarity coefficient of two multisets of tokens.


    S = 2|N1 ∩ N2| / (|N1| + |N2|)

    :param n_1: first multiset of tokens
    :param n_2: second multiset of tokens
    :return: similarity score, 0 for empty
    """
    cardinality_n_1 = sum(n_1.values())
    cardinality_n_2 = sum(n_2.values())

    if cardinality_n_1 == 0 or cardinality_n_2 == 0:
        return 0

    intersect_cardinality = sum((n_1 & n_2).values())

    return 2 * intersect_cardinality / (cardinality_n_1 + cardinality_n_2)
