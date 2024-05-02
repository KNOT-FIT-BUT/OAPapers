# -*- coding: UTF-8 -*-
""""
Created on 24.11.22
Module for enhancing citation spans.

:author:     Martin Dočekal
"""
import enum
import itertools
import re
import sys
from typing import Dict, List

from windpyutils.structures.span_set import SpanSet, SpanSetOverlapsEqRelation

from oapapers.bib_entry import Bibliography
from oapapers.document import Document
from oapapers.hierarchy import TextContent, RefSpan, Hierarchy

HARVARD_RE_INSIDE_BRACKETS = re.compile(
    r"\b(?!(?:Although|Also)\b)(?:e\.g\.\,?\s*)?((?:[A-Z](?:(?:' ?)|(?:` ?))?[A-Za-z'`-]+?))(?:,? (?:(?:and |& )?((?:[A-Z](?:(?:' ?)|(?:` ?))?[A-Za-z'`-]+?))|(?:et al.?)))*(?:, *((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?))",
    re.MULTILINE)
HARVARD_IN_TEXT_RE = re.compile(
    r"\b([A-Z](?:(?:' ?)|(?:` ?))?[A-Za-z'`-]+)\s+(?:(?:and|&) ([A-Z](?:(?:' ?)|(?:` ?))?[A-Za-z'`-]+))?(?:et al.?)?\s*\(((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?)\)",
    re.MULTILINE)
HARVARD_RE = re.compile(
    rf"(?:\(\s*)?{HARVARD_RE_INSIDE_BRACKETS.pattern}(?:\s*\))?",
    re.MULTILINE)
HARVARD_COMPLETE_RE = re.compile(rf"(?:{HARVARD_RE.pattern}|{HARVARD_IN_TEXT_RE.pattern})", re.MULTILINE)
VANCOUVER_PARENTHESIS_RE = re.compile(
    r"(\(\s*(([0-9]+)|[0-9]+\s*-\s*[0-9]+|[0-9]+(\s*,\s*[0-9]+)+)\s*\))")  # couldn't use something like(\(([0-9]+(,|-|)\s*)+\)) as it causes catastrophic backtracking
VANCOUVER_SQUARE_BRACKETS_RE = re.compile(r"(\[\s*(([0-9]+)|[0-9]+\s*-\s*[0-9]+|[0-9]+(\s*,\s*[0-9]+)+)\s*\])")
NORMALIZED_RE = re.compile(r"(\[CITE:(UNK|[0-9]+)])")

EXPAND_SPAN_RE = re.compile("((\[[0-9]+\s*-\s*[0-9]+\])|(\[[0-9]+\]\s*-\s*\[[0-9]+\])|(\([0-9]+\)\s*-\s*\([0-9]+\)))")
VANCOUVER_RE = re.compile(
    rf"(({VANCOUVER_PARENTHESIS_RE.pattern})|({VANCOUVER_SQUARE_BRACKETS_RE.pattern}))")
NUMBER_RE = re.compile("[0-9]+")


class CitationStyle(enum.Enum):
    HARVARD = "HARVARD"
    VANCOUVER_PARENTHESIS = "VANCOUVER_PARENTHESIS"
    VANCOUVER_SQUARE_BRACKETS = "VANCOUVER_SQUARE_BRACKETS"
    NORMALIZED = "NORMALIZED"
    UNKNOWN = "UNKNOWN"


def identify_citation_style_of_doc(doc: Document) -> CitationStyle:
    """
    Returns citation style of this document.

    :param doc: document for identification
        it uses its hierarchy to identify the citation style
    :return: citation style
    """

    return identify_citation_style_of_hier(doc.hierarchy)


def identify_citation_style_of_hier(hier: Hierarchy) -> CitationStyle:
    """
    Returns citation style of given hierarchy.

    :param doc: document for identification
        it uses its hierarchy to identify the citation style
    :return: citation style
    """

    counts = [
        0,  # HARVARD
        0,  # VANCOUVER_PARENTHESIS
        0,  # VANCOUVER_SQUARE_BRACKETS
        0,  # NORMALIZED
    ]

    for tc in hier.text_content():
        counts[0] += len(HARVARD_RE.findall(tc.text))
        counts[1] += len(VANCOUVER_PARENTHESIS_RE.findall(tc.text))
        counts[2] += len(VANCOUVER_SQUARE_BRACKETS_RE.findall(tc.text))
        counts[3] += len(NORMALIZED_RE.findall(tc.text))

    max_count_index = max(enumerate(counts), key=lambda x: x[1])[0]

    if any(counts[max_count_index] == c for i, c in enumerate(counts) if i != max_count_index):
        return CitationStyle.UNKNOWN

    return [
        CitationStyle.HARVARD,
        CitationStyle.VANCOUVER_PARENTHESIS,
        CitationStyle.VANCOUVER_SQUARE_BRACKETS,
        CitationStyle.NORMALIZED
    ][max_count_index]


def identify_citation_style(text: str) -> CitationStyle:
    """
    Returns citation style of this document.

    :param text: text of the document
    :return: citation style
    """

    counts = [
        len(HARVARD_RE.findall(text)),
        len(VANCOUVER_PARENTHESIS_RE.findall(text)),
        len(VANCOUVER_SQUARE_BRACKETS_RE.findall(text)),
        len(NORMALIZED_RE.findall(text)),
    ]
    max_count_index = max(enumerate(counts), key=lambda x: x[1])[0]

    if any(counts[max_count_index] == c for i, c in enumerate(counts) if i != max_count_index):
        return CitationStyle.UNKNOWN

    return [
        CitationStyle.HARVARD,
        CitationStyle.VANCOUVER_PARENTHESIS,
        CitationStyle.VANCOUVER_SQUARE_BRACKETS,
        CitationStyle.NORMALIZED
    ][max_count_index]


def repair_span_offsets(text_content: TextContent, start_offset: int, offset_diff: int) -> None:
    """
    Repair offsets of spans in text content.

    :param text_content: text content whose spans should be repaired
    :param start_offset: start offset from which the repair should be done
    :param offset_diff: difference that should be added to offsets
    """
    if offset_diff != 0:
        # shift to new offset
        for s in itertools.chain(text_content.citations, text_content.references):
            if s.start >= start_offset:
                s.start += offset_diff
                s.end += offset_diff


def expand_single_citation(text_content: TextContent, start_offset: int, end_offset: int,
                           vancouver_ids: Dict[int, int]):
    """
    Expanding of citations in format like:
    [3-6] o or [3] - [6].
    to
        [3][4][5][6]

    :param text_content: text content with citation for expanding
        changes are done in place
    :param start_offset: start of citation span for expanding
    :param end_offset: end of citation span for expanding
    :param vancouver_ids: vancouver ids that will be used to obtain index into bibliography
        An id must be at position that matches with bibliography index
    """
    span = text_content.text[start_offset: end_offset]
    parts = span.split("-")

    if len(parts) == 2:

        num_left = re.findall(NUMBER_RE, parts[0])
        num_right = re.findall(NUMBER_RE, parts[1])
        if len(num_left) == 1 and len(num_right) == 1:
            num_left = int(num_left[0])
            num_right = int(num_right[0])
            if num_left < num_right:
                new_spans = [f"[{i}]" if span[0] == "[" else f"({i})" for i in range(num_left, num_right + 1)]
                new_span = "".join(new_spans)
                text_content.text = text_content.text[:start_offset] + new_span + text_content.text[end_offset:]

                # remove all collisions
                new_citations = []
                for c in text_content.citations:
                    if end_offset <= c.start or c.end <= start_offset:
                        new_citations.append(c)
                text_content.citations = new_citations

                offset_diff = len(new_span) - len(span)
                repair_span_offsets(text_content, start_offset, offset_diff)

                # insert
                off = start_offset
                for i, s in zip(range(num_left, num_right + 1), new_spans):
                    start, end = off, off + len(s)
                    try:
                        r_span = RefSpan(vancouver_ids[i], start, end)
                    except KeyError:
                        # this bib was filtered out
                        r_span = RefSpan(None, start, end)
                    text_content.citations.append(r_span)
                    off = end

                text_content.citations = sorted(text_content.citations, key=lambda r: r.start)


def expand_citations(hierarchy: Hierarchy, vancouver_ids: Dict[int, int]):
    """
    Performs expansion of dashed citations with multiple references:
        [3-6], [3] - [6], or (3) - (6).
        Not (3-6) to prevent collisions with ordinary parenthesis as it uses simple regex.

    Make sure that the hierarchy uses vancouver style citations.

    :param hierarchy: hierarchy containing spans that should be enhanced
        considers only text content whose parent (sub)hierarchy has None headline
        (works in place)
    :param vancouver_ids: vancouver ids that will be used to obtain index into bibliography
        An id must be at position that matches with bibliography index.
    """

    for tc in hierarchy.text_content(lambda x: x.headline is None):  # take only plain text, skip formulas
        offset = 0
        for m in EXPAND_SPAN_RE.finditer(tc.text):
            start_offset = offset + m.start()
            end_offset = offset + m.end()
            # we are only interested in those that are marked as citation
            if any(end_offset > c.start and c.end > start_offset for c in tc.citations):
                len_before = len(tc.text)
                expand_single_citation(tc, start_offset, end_offset, vancouver_ids)
                offset = len(tc.text) - len_before


def merge_citations(tc: TextContent, square_brackets: bool = True):
    """
    Merges citations to dashed format. It removes the linking with bibliography. Expects vancouver style citations.

    :param tc: text content with citations
    :param square_brackets: if True then citations are merged to [1-3] format, otherwise to (1-3)
    """

    start_index = None
    end_index = None
    start_num = None
    end_num = None
    new_citations = []

    left_bracket = "[" if square_brackets else "("
    right_bracket = "]" if square_brackets else ")"

    def add_new_citation():
        if end_index > start_index:
            # merge
            new_cit_text = f"{left_bracket}{start_num}-{end_num}{right_bracket}"
            tc.text = tc.text[:tc.citations[start_index].start] + new_cit_text + \
                      tc.text[tc.citations[end_index].end:]
            start_offset = tc.citations[start_index].start
            diff_offset = len(new_cit_text) - (tc.citations[end_index].end - tc.citations[start_index].start)
            new_citations.append(RefSpan(None, start_offset, start_offset + len(new_cit_text)))
            repair_span_offsets(tc, start_offset, diff_offset)
        else:
            # leave as is
            new_citations.append(tc.citations[start_index])

    for i, c in enumerate(tc.citations):
        span = tc.text[c.start:c.end]
        num = re.findall(NUMBER_RE, span)

        if len(num) == 1:
            num = int(num[0])
            if start_index is None:
                start_index = i
                end_index = i
                start_num = num
                end_num = num
            else:
                if num == end_num + 1 and \
                        len(tc.text[tc.citations[i - 1].end:c.start].strip()) == 0 and \
                        left_bracket == span[0] and right_bracket == span[-1]:  # all expanded citations have brackets
                    end_index = i
                    end_num = num
                else:
                    add_new_citation()
                    start_index = i
                    end_index = i
                    start_num = num
                    end_num = num

    if start_index is not None:
        add_new_citation()

    tc.citations = new_citations


def repair_single_merged_citation_span(text_content: TextContent, vancouver_ids: Dict[int, int]):
    """
    Repairs unidentified citation spans that covers multiple citations in given text content.
        E.g.: Lamma et al. [5,6,7] apply an extension of logic
            citation spans: Lamma et al. [5,6,7]
            but should be just [5,6,7]

    :param text_content: text content that will be checked for badly annotated citations and repaired in place
    :param vancouver_ids: vancouver ids that will be used to obtain index into bibliography
        Is used for operations on vancouver citation style (expansion of citations).
    """
    new_citations = []
    for cit_offset, cit_span in enumerate(text_content.citations):
        replaced = False
        if cit_span.index is None:
            span_text = text_content.text[cit_span.start:cit_span.end]
            for m in VANCOUVER_RE.finditer(span_text):
                offset = cit_span.start + m.start()
                citation_span = span_text[m.start():m.end()]
                parts = citation_span.split(",")

                for p in parts:
                    i = int(NUMBER_RE.search(p).group(0))
                    end_offset = offset + len(p) + 1
                    try:
                        r_span = RefSpan(vancouver_ids[i], offset, end_offset)
                    except KeyError:
                        # this bib was filtered out
                        r_span = RefSpan(None, offset, end_offset)
                    replaced = True
                    new_citations.append(r_span)
                    offset = end_offset
                new_citations[-1].end -= 1
        if not replaced:
            new_citations.append(cit_span)

    text_content.citations = sorted(new_citations, key=lambda r: r.start)


def repair_merged_citation_spans(hierarchy: Hierarchy, vancouver_ids: Dict[int, int]):
    """
    Repairs unidentified citation spans that covers multiple citations.
        E.g.: Lamma et al. [5,6,7] apply an extension of logic
            citation spans: Lamma et al. [5,6,7]
            but should be just [5,6,7]

    :param hierarchy: hierarchy containing spans that should be enhanced
        (works in place)
    :param vancouver_ids: vancouver ids that will be used to obtain index into bibliography
        Is used for operations on vancouver citation style (expansion of citations).
    """
    for tc in hierarchy.text_content(lambda x: x.headline is None):  # take only plain text, skip formulas
        repair_single_merged_citation_span(tc, vancouver_ids)


def create_vancouver_mapping(hierarchy: Hierarchy, interpolate: bool = True) -> Dict[int, int]:
    """
    Creates mapping of in text vancouver style citation ids that could be used to obtain index into bibliography.

    :param hierarchy: hierarchy whose citation spans will be used to create the mapping
    :param interpolate: As there were observed some "non-linearities" in this mapping, probably due to improper
        parsing of bibliography, this method is doing the best effort match, thus the result is rather heuristical.
        e.g.
            let's say that we can obtain following mapping from content:
                vancouver:  1 3 | 5 7
                bib. index: 0 2 | 5 7
            We can see that the first part before | is v -1 and the second part is just v. If interpolate is activate
            this method is able to fill the gaps to:
                vancouver:  1 2 3 | 5 6 7
                bib. index: 0 1 2 | 5 6 7

    :return: vancouver ids mapping (yes this is the thing you need for the other methods in this module)
    """
    res = {}
    for tc in hierarchy.text_content():
        for c in tc.citations:
            if c.index is not None:
                cit_ids = re.findall(NUMBER_RE, tc.text[c.start: c.end])
                if len(cit_ids) == 1:
                    res[int(cit_ids[0])] = c.index

    if interpolate:
        segments = []  # stores lists: (first vancouver, last vancouver, difference to bib. index)
        for v, i in sorted(res.items(), key=lambda m: m[0]):
            d = i - v
            if len(segments) == 0 or segments[-1][2] != i - v:
                segments.append([v, v, d])
            else:
                segments[-1][1] = v

        # fill the gaps
        for first, last, diff in segments:
            for x in range(first + 1, last):
                if x not in res:
                    res[x] = x + diff

    return res


def remove_suspicious_citations(hierarchy: Hierarchy, max_id: int = 1_000, max_expansion: int = 100):
    """
    Removes suspicious citations from the hierarchy.
    Suspicious citation is a citation that has too big ids or if it is citation with multiple references [1000-9999]
    than it is suspicious if the difference is too big.

    :param hierarchy: hierarchy that will be checked for suspicious citations and repaired in place
    :param max_id: maximal vancouver id that can be used for citation
    :param max_expansion: maximal number of citations that can be defined by dashed citation span
    """
    hierarchy.citation_spans()
    for tc in hierarchy.text_content():
        new_citations = []
        for cit in tc.citations:
            if cit.index is None:
                span_text = tc.text[cit.start:cit.end]
                expand_spans = list(EXPAND_SPAN_RE.finditer(span_text))
                if len(expand_spans) == 0:
                    number = re.findall(NUMBER_RE, span_text)
                    if len(number) == 1:
                        number = int(number[0])
                        if number <= max_id:
                            new_citations.append(cit)
                else:
                    parts = span_text.split("-")
                    if len(parts) == 2:
                        number_left = re.findall(NUMBER_RE, parts[0])
                        num_right = re.findall(NUMBER_RE, parts[1])
                        if len(number_left) == 1 and len(num_right) == 1:
                            number_left = int(number_left[0])
                            num_right = int(num_right[0])
                            if number_left <= max_id and num_right <= max_id and \
                                    num_right - number_left < max_expansion:
                                new_citations.append(cit)
                                continue
            else:
                new_citations.append(cit)
        tc.citations = new_citations


def repair_crossing_spans(hierarchy: Hierarchy):
    """
    Repairs crossing spans in hierarchy.

    The byproduct of this method is that it also sort the spans by their start.

    :param hierarchy: hierarchy containing spans that should be repaired
        (works in place)
    """

    for tc in hierarchy.text_content():
        tc.citations.sort(key=lambda x: x.start)
        tc.references.sort(key=lambda x: x.start)

        for i in range(len(tc.citations) - 1):
            if tc.citations[i].end > tc.citations[i + 1].start:
                tc.citations[i].end = tc.citations[i + 1].start

        for i in range(len(tc.references) - 1):
            if tc.references[i].end > tc.references[i + 1].start:
                tc.references[i].end = tc.references[i + 1].start


def repair_span_boundaries_in_hierarchy(hierarchy: Hierarchy, cit_style: CitationStyle):
    """
    Repairs boundaries of citation spans in whole hierarchy.
    It will try to find the closest bracket

    :param hierarchy: hierarchy containing spans that should be repaired
        (works in place)
    :param cit_style: citation style of the hierarchy
    """

    try:
        match_re = {
            CitationStyle.HARVARD: HARVARD_RE,
            CitationStyle.VANCOUVER_PARENTHESIS: VANCOUVER_PARENTHESIS_RE,
            CitationStyle.VANCOUVER_SQUARE_BRACKETS: VANCOUVER_SQUARE_BRACKETS_RE,
        }[cit_style]
    except KeyError:
        return

    for tc in hierarchy.text_content():
        # sort spans
        tc.citations.sort(key=lambda x: x.start)
        for s in match_re.finditer(tc.text):
            matches = []
            for c in tc.citations:
                if s.end() > c.start and c.end > s.start():
                    matches.append(c)
            
            if len(matches) == 1:
                matches[0].start = s.start()
                matches[0].end = s.end()
            elif len(matches) > 1:
                # just the first and last
                matches[0].start = s.start()
                matches[-1].end = s.end()


def enhance_citations_spans(hierarchy: Hierarchy, interpolate: bool = True, max_id: int = 1_000,
                            max_expansion: int = 100):
    """
    Method for repairing/enhancing citation spans. It is doing following work:
        Performs expansion of dashed citations with multiple references.
            [3-6], [3] - [6], or (3) - (6).
            Not (3-6) to prevent collisions with ordinary parenthesis as it uses simple regex.
            considers only text content whose parent (sub)hierarchy has None headline
        Repairs unidentified citation spans that covers multiple citations.
            E.g.: Lamma et al. [5,6,7] apply an extension of logic
                citation spans: Lamma et al. [5,6,7]
                but should be just [5,6,7]
    Is designed for vancouver style citations.


    It also tries to repair boundaries of citation spans not just for vancouver style citations.

    :param hierarchy: hierarchy containing spans that should be enchanced
        (works in place)
    :param interpolate: whether the interpolation should be used when the vancouver citation ids are mapped to
        bibliography indices.
        As there were observed some "non-linearities" in this mapping, probably due to improper
        parsing of bibliography, this method is doing the best effort match, thus the result is rather heuristical.
        e.g.
            let's say that we can obtain following mapping from content:
                vancouver:  1 3 | 5 7
                bib. index: 0 2 | 5 7
            We can see that the first part before | is v -1 and the second part is just v. If interpolate is activate
            this method is able to fill the gaps to:
                vancouver:  1 2 3 | 5 6 7
                bib. index: 0 1 2 | 5 6 7
    :param max_id: maximal vancouver id that can be used for citation
        it is overriden by the max id from the known bib. mapping
    :param max_expansion: maximal number of citations that can be defined by dashed citation span
    """
    citation_style = identify_citation_style_of_hier(hierarchy)

    if citation_style is not None and citation_style in {CitationStyle.VANCOUVER_PARENTHESIS,
                                                         CitationStyle.VANCOUVER_SQUARE_BRACKETS}:
        vancouver_ids = create_vancouver_mapping(hierarchy, interpolate=interpolate)
        max_id = max(max_id, max(vancouver_ids.keys()) if len(vancouver_ids) > 0 else 0)
        remove_suspicious_citations(hierarchy, max_id=max_id, max_expansion=max_expansion)
        expand_citations(hierarchy, vancouver_ids)
        repair_merged_citation_spans(hierarchy, vancouver_ids)

    repair_span_boundaries_in_hierarchy(hierarchy, citation_style)


def add_missing_harvard_style_citations(doc: Document, authors_match_threshold: float = sys.float_info.min):
    """
    Method for adding missing harvard style citations to the document.
    It is using simple regex to find the citations, and then it is checking whether the citation is already present
    in the document. If not, it is added to the document. Every matched citation is checked against the bibliography
    and if it is not present in the bibliography, it is not considered.

    :param doc: document to which the missing citations should be added
        works in place
    :param authors_match_threshold: threshold for matching the authors in the citation and the bibliography
     The score is dice similarity between authors names pair.
     By default, it is set to mode when at least one word in a name is enough to match the authors.
    """
    bibliography = Bibliography(doc.bibliography,
                                authors_match_threshold=authors_match_threshold)
    for t_c in doc.text_content():
        known_citation_spans_starts = []
        known_citation_spans_ends = []

        for c in t_c.citations:
            known_citation_spans_starts.append(c.start)
            known_citation_spans_ends.append(c.end)

        known_citation_spans = SpanSet(known_citation_spans_starts, known_citation_spans_ends, True,
                                       SpanSetOverlapsEqRelation())

        new_citations_spans = []
        for match in HARVARD_COMPLETE_RE.finditer(t_c.text):
            if match.span() not in known_citation_spans:
                author_a = match.group(1) if match.group(1) is not None else match.group(
                    4)  # 4: is for in text citation
                author_b = match.group(2) if match.group(2) is not None else match.group(
                    5)  # 5: is for in text citation
                year = match.group(3) if match.group(3) is not None else match.group(6)  # 6: is for in text citation

                if author_a is None or year is None:
                    continue

                try:
                    year = int(year)

                    authors = [author_a]
                    if author_b is not None:
                        authors.append(author_b)

                    new_citations_spans.append(RefSpan(
                        index=bibliography.index(None, authors, year),
                        start=match.span()[0],
                        end=match.span()[1]
                    ))
                except ValueError:
                    continue

        t_c.citations.extend(new_citations_spans)
        # sort them by start
        t_c.citations.sort(key=lambda x: x.start)


def match_unk_citation_spans_with_bib(doc: Document, authors_match_threshold: float = sys.float_info.min):
    """
    Method for matching unmatched harvard style citations with document's bibliography.

    :param doc: document to which the citations should be identified
        works in place
    :param authors_match_threshold: threshold for matching the authors in the citation and the bibliography
     The score is containment score between authors names pair.
     By default, it is set to mode when at least one word in a name is enough to match the authors.
    """
    bibliography = Bibliography(doc.bibliography,
                                authors_match_threshold=authors_match_threshold)
    for t_c in doc.text_content():
        for c in t_c.citations:
            if c.index is not None:
                continue

            span = t_c.text[c.start:c.end]

            matches = list(HARVARD_RE_INSIDE_BRACKETS.finditer(span))
            if len(matches) == 1:
                match = matches[0]
                if match.group(1) is None or match.group(3) is None:
                    continue

                try:
                    year = int(match.group(3))

                    authors = [match.group(1)]
                    if match.group(2) is not None:
                        authors.append(match.group(2))
                    c.index = bibliography.index(None, authors, year)
                except ValueError:
                    continue


def group_citations(t_c: TextContent) -> List[List[RefSpan]]:
    """
    Method for grouping citations in the text content.

    Example of citation group: [1,2,3]
     .. as was presented in [1,2,3] ...


    :param t_c: text content to be processed
    :return: list of citation groups
    """

    groups = []

    for c in t_c.citations:
        if len(groups) == 0 or not re.match(r"^\W*$", t_c.text[groups[-1][-1].end: c.start]):
            # new cit. group
            groups.append([])

        groups[-1].append(c)

    return groups
