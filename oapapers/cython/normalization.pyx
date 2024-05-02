# -*- coding: UTF-8 -*-
""""
Created on 07.08.22
Function for normalization.

:author:     Martin Dočekal
"""

import re
from collections import Counter
from typing import List, Sequence, Tuple, Union

from unidecode import unidecode


re_pattern_non_word = re.compile(r'[\W_]+', re.UNICODE)
"""
\W
    Matches any character which is not a word character. This is the opposite of \w. 
    If the ASCII flag is used this becomes the equivalent of [^a-zA-Z0-9_]. 
    If the LOCALE flag is used, matches characters which are neither alphanumeric in the current 
    locale nor the underscore.
"""

cdef str remove_repeating_characters(str s):
    """
    Removes repeating characters in a row.

    :param s: string for removing
    :return: string without repeating characters in a row
    """
    cdef str res = ""
    cdef str last = ""
    for c in s:
        if c != last:
            res += c
            last = c
    return res


cpdef str normalize_string(s: str):
    """
    Normalizes string.

    Normalization:
        all non word characters are converted to space
        conversion to lower case
        transliterates any unicode string into the closest possible representation in ascii text
        removes all repeated characters in a row and replaces them with one character
            as it is common error to write e.g. "Timmerman" instead of "Timmermann"

    :param s: string for normalization
    :return: normalized string
    """

    norm_s = re_pattern_non_word.sub(" ", unidecode(s)).lower()
    if len(norm_s.strip()) == 0:
        norm_s = s.lower()
    norm_s = remove_repeating_characters(norm_s)
    return norm_s

cpdef normalize_multiple_strings(s: Sequence[str]):
    """
    Normalizes multiple strings.

    Normalization:
        all non word characters are converted to space
        conversion to lower case
        transliterates any unicode string into the closest possible representation in ascii text
        removes all repeated characters in a row and replaces them with one character
            as it is common error to write e.g. "Timmerman" instead of "Timmermann"

    :param s: strings for normalization
    :return: normalized strings
    """
    return [normalize_string(x) for x in s]


def normalize_and_tokenize_string(s: str) -> List[str]:
    """
    Normalizes string and splits it into word tokens. Using whitespace as a delimiter.

    Normalization:
        all non word characters are converted to space
        conversion to lower case
        transliterates any unicode string into the closest possible representation in ascii text
        removes all repeated characters in a row and replaces them with one character
            as it is common error to write e.g. "Timmerman" instead of "Timmermann"

    :param s: string for normalization
    :return: list with tokens
    """
    return normalize_string(s).split()


cpdef similarity_score(n_1, n_2):
    """
    Computes similarity score for two sets.
    
    :param n_1: first set
    :param n_2: second set
    :return: similarity score
    """

    # convert to sets
    n_1 = set(n_1)
    n_2 = set(n_2)

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


cpdef similarity_scores_for_list_of_strings(x: str, list2: Sequence[str]):
    """
    Computes similarity scores for two lists of strings. Performs normalization and tokenization before comparison.

    :param x: string that will be compared with all strings in list2
    :param list2: second list of strings
    :return: similarity scores
    """
    res = []
    norm_x = normalize_and_tokenize_string(x)
    for s2 in list2:
        res.append(similarity_score(norm_x, normalize_and_tokenize_string(s2)))
    return res


cpdef normalize_authors(authors: Sequence[str]):
    """
    Normalizes authors.

    :param authors: authors for normalization
    :return: sequence of normalized authors
    """

    res = []
    for author in authors:
        res.append(set(normalize_and_tokenize_string(author)))
    return res


cpdef convert_authors_to_initials_normalized_version(authors: Sequence[str]):
    """
    Converts names to initials version (e.g. John Ronald Reuel Tolkien => J R R Tolkien)
    :param authors: authors for conversion
    :return: sequence of normalized authors converted to initials version
    """
    init_authors = []
    for a in authors:
        parts = normalize_and_tokenize_string(a)

        init_parts = []

        for p in parts[:-1]:
            init_parts.append(p[0])
        if len(parts) > 0:
            init_parts.append(parts[-1])

        init_authors.append(set(init_parts))

    return init_authors


cpdef initial_and_normalized_authors(authors: Sequence[Union[str, Tuple[str]]],
                                   pre_normalized: bool = False):
    """
    It is the same like calling convert_authors_to_initials_normalized_version and normalize_authors
    but it is more efficient.

    :param authors: authors for conversion
    :param pre_normalized: if True then authors are already normalized and tokenized
    :return:
        sequence of normalized authors converted to initials version
        sequence of normalized authors
    """
    init_authors = []
    norm_authors = []
    for a in authors:
        parts = a if pre_normalized else normalize_and_tokenize_string(a)

        init_parts = []
        for p in parts[:-1]:
            init_parts.append(p[0])
        if len(parts) > 0:
            init_parts.append(parts[-1])

        init_authors.append(set(init_parts))
        norm_authors.append(set(parts))

    return init_authors, norm_authors

