# -*- coding: UTF-8 -*-
"""
Created on 15.03.23

:author:     Martin Dočekal
"""

from typing import Optional, Sequence, AbstractSet

from oapapers.similarities import containment_score


def match_authors(a_normalized: AbstractSet[str], a_normalized_initials: AbstractSet[str],
                  b_normalized: AbstractSet[str], b_normalized_initials: AbstractSet[str],
                  match_threshold: float) -> bool:
    """
    Determines whether two authors names match based on normalized versions

    :param a_normalized: normalized version of an author name
    :param a_normalized_initials: normalized initial version of an author name
    :param b_normalized: normalized version of an author name
    :param b_normalized_initials: normalized initial version of an author name
    :param match_threshold: Score threshold for matching. All above or equal are ok.
    :return: true if there is a match
    """

    if containment_score(a_normalized, b_normalized) >= match_threshold or \
            containment_score(a_normalized, b_normalized_initials) >= match_threshold or \
            containment_score(a_normalized_initials, b_normalized) >= match_threshold:
        return True

    return False


def match_authors_groups(a: Optional[Sequence[AbstractSet[str]]], a_initials: Optional[Sequence[AbstractSet[str]]],
                         b: Optional[Sequence[AbstractSet[str]]], b_initials: Optional[Sequence[AbstractSet[str]]],
                         match_threshold: float) -> bool:
    """
    Determines whether at least one author pair among a and b matches.

    :param a: one sequence of authors
    :param a_initials: normalized initial version of a
    :param b: another sequence of authors
    :param b_initials: normalized initial version of b
    :param match_threshold: Score threshold for matching. All above or equal are ok.
    :return: True if there is at least one matching pair
    """

    if any(match_authors(x, x_i, y_a, y_a_i, match_threshold)
           for x, x_i in zip(a, a_initials)
           for y_a, y_a_i in zip(b, b_initials)):
        return True

    return False

