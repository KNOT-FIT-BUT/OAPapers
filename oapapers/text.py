# -*- coding: UTF-8 -*-
"""
Created on 10.03.23
Utils for improving text quality.

:author:     Martin Dočekal
"""
import enum
import itertools
import re
from enum import Enum

from pyphen import Pyphen
from typing import Tuple, Sequence, Optional

from windpyutils.structures.span_set import SpanSet, SpanSetOverlapsEqRelation


class DeHyphenator:
    """
    There were issues with hyphenation in the data. The hyphens that are used to break long words at the end of a line.
    """

    def __init__(self, lang_code: str = "en_US", regex: str = r"(\w+)-\s+(\w+)"):
        """
        :param lang_code: Language code of the text to dehyphenate.
        """
        self.hyphenator = Pyphen(lang=lang_code)
        self.regex = re.compile(regex)

    def sub(self, match: re.Match) -> str:
        """
        Substitution function for the regex.
        """
        candidate = match.group(1) + match.group(2)
        if (match.group(1).lower(), match.group(2).lower()) in set(self.hyphenator.iterate(candidate.lower())):
            return candidate

        return match.group(0)

    def replacements(self, text: str) -> Tuple[list[Tuple[int, int]], list[str]]:
        """
        Returns replacements for the text.

        :param text: Text to dehyphenate.
        :return: List of tuples of start and end index of the replacement and the list of replacement themselves
        """
        matches = list(self.regex.finditer(text))
        return [(m.start(), m.end()) for m in matches], [self.sub(m) for m in matches]

    def __call__(self, text: str) -> str:
        """
        Dehyphenates the text.

        :param text: Text to dehyphenate.
        :return: Dehyphenated text.
        """
        return self.regex.sub(self.sub, text)


class SpanCollisionHandling(Enum):
    """
    Handling of span collisions.
    """

    SKIP = enum.auto()  #: the replacement span will be skipped
    REMOVE = enum.auto()  #: remove the span that collides with the replacement span
    MERGE = enum.auto()  #: merge the span that collides with the replacement span
    RAISE = enum.auto()  #: raise an exception if there is a collision


def replace_at(text: str, replacement_spans: Sequence[Tuple[int, int]], replace_with: Sequence[str],
               spans: Sequence[Sequence[Tuple[int, int]]] = [],
               collisions: SpanCollisionHandling = SpanCollisionHandling.RAISE) -> \
        Tuple[str, Sequence[Sequence[Optional[Tuple[int, int]]]]]:
    """
    Replaces text at given spans with given text.
    It also handles offsets of associated spans with given text.

    :param text: text to be replaced
    :param replacement_spans: starts and ends of disjunctive spans to be replaced
    :param replace_with: new text on given spans
    :param spans: associated spans that should be updated
        it is sequence of sequences of spans as it enables to have multiple groups of spans
    :param collisions: defines how to handle collisions with citations and references
    :raise ValueError: if the number of spans and replace_with is not the same
    :raise ValueError: if there are spans that collide with citations or references and collisions is set to RAISE
    :return:
        new text
        updated spans or None if the span was removed
    """

    if len(replacement_spans) != len(replace_with):
        raise ValueError("The number of replacement_spans and replace_with must be the same.")

    if collisions == SpanCollisionHandling.SKIP or collisions == SpanCollisionHandling.RAISE:
        span_set = SpanSet(((s[0], s[1]) for s in itertools.chain.from_iterable(spans)), None,
                           eq_relation=SpanSetOverlapsEqRelation())

        if collisions == SpanCollisionHandling.RAISE and any(s in span_set for s in replacement_spans):
            raise ValueError("There are replacement_spans that collide with spans.")

        # we need to remove replacement_spans spans that collide with spans
        keep_mask = [s not in span_set for s in replacement_spans]
        replacement_spans = [s for s, k in zip(replacement_spans, keep_mask) if k]
        replace_with = [s for s, k in zip(replace_with, keep_mask) if k]

    elif collisions == SpanCollisionHandling.REMOVE:
        # we need to remove all spans that collide with replacement_spans
        spans_set = SpanSet(replacement_spans, None, eq_relation=SpanSetOverlapsEqRelation())
        spans = [[None if s in spans_set else s for s in ss] for ss in spans]

    # we need to sort spans by start
    arg_sort = sorted(range(len(replacement_spans)), key=lambda x: replacement_spans[x][0])
    sorted_spans = [replacement_spans[i] for i in arg_sort]
    sorted_replace_with = [replace_with[i] for i in arg_sort]
    # check disjunctive
    for i in range(1, len(replacement_spans)):
        if sorted_spans[i][0] < sorted_spans[i - 1][1]:
            raise ValueError("The spans replacement_spans be disjunctive.")

    # we need to find all citations and references that are in the spans
    # and update them in following way:
    #   if the span is before the citation/reference then we need to update the start and end
    #   if the span is after the citation/reference then we need to do nothing
    #   if the span collides with the citation/reference then we need to merge them as all the other
    #       collision handling options are already handled

    offset = 0
    for i, (start, end) in enumerate(sorted_spans):
        start = start + offset
        end = end + offset

        diff = len(sorted_replace_with[i]) - (end - start)
        new_spans = []
        # we need to update citations and references
        for span_seq in spans:
            new_spans.append([])
            for s in span_seq:
                if s is None:
                    new_spans[-1].append(None)
                    continue
                s_start, s_end = s
                if s_end > start and end > s_start:
                    # we need to merge them
                    s_start = min(s_start, start)
                    s_end = max(s_end, end) + diff

                if s_start >= end:
                    s_start += diff
                    s_end += diff

                new_spans[-1].append((s_start, s_end))

        spans = new_spans
        text = text[:start] + sorted_replace_with[i] + text[end:]
        offset += diff

    return text, spans


def clean_title(title: str) -> str:
    """
    Cleans the title.

    There were observed some parsing errors in document titles (mainly in bibliography)
    that are supposed to be fixed here.

    :param title: Title to clean.
    :return: Cleaned title.
    """

    # Due to parssing errors there are several titles that are containing an identifier in the postfix.
    # remove substrings like:
    # . CoRR, abs/1606.01400
    # . doi:10.1111/j.1751-9020.2007.00007.x 17
    # . arXiv, 1503.01215
    title = re.sub(r"\W?\s*(arXiv|CoRR|doi|math|Archive Number)[:,\.]?\s*((((abs)|(PR))\/)|(number))?\s*[0-9]+(\.[0-9]+)?.*", "", title,
                   flags=re.IGNORECASE)

    return title
