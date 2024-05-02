# -*- coding: UTF-8 -*-
""""
Created on 11.04.22

:author:     Martin Dočekal
"""
import copy
import math
import re
import sys
from abc import abstractmethod, ABC

from typing import List, Optional, Tuple, Pattern, AbstractSet, Iterable

from oapapers.datasets import OADataset
from oapapers.document import Document, OARelatedWorkDocument


class Filter(ABC):
    """
    Base class for document filters.
    """

    @abstractmethod
    def __call__(self, document: Document) -> bool:
        """
        Check whether given document passes this filter.
        :param document: document for filtration
        :return: true passes filter, false otherwise
        """
        pass


class FilterWithID:

    def __init__(self, f: Filter):
        """
        Performs filtering using given filter and returns tuple with documents id and filter result.

        :param f: filter that should be used
        """

        self.filter = f

    def __call__(self, document: Document) -> Tuple[bool, int]:
        """
        Check whether given document passes this filter.

        :param document: document for filtration
        :return: whethr passes filter, id of document
        """
        return self.filter(document), document.id


class AlwaysFilter(Filter):
    """
    Always filter returns given result for every document.

    Example:
        >>> f = AlwaysFilter(False)
        >>> f(document)
        False

        >>> f = AlwaysFilter(True)
        >>> f(document)
        True
    """

    def __init__(self, res: bool):
        """
        Initialization of always filter.

        :param res: this values will be always returned
        """

        self.res = res

    def __call__(self, document: Document) -> bool:
        return self.res


class CombinedFilter(Filter):
    """
    Combination of filters.

    applies all filters in given order and the results is:
        f1(d) and f2(d) and f3(f)
    """

    def __init__(self, filters: List[Filter]):
        """
        Initialization of combined filter.

        :param filters: filters that should be used
        """

        self.filters = filters

    def __call__(self, document: Document) -> bool:
        return all(f(document) for f in self.filters)


class SecNonEmptyHeadlinesFilter(Filter):
    """
    Filters out documents with empty headline on section lvl.
    """

    def __init__(self, min_sec_height: int = 2):
        """
        Initialization of filter.

        :param min_sec_height: min height of a sub-hierarchy to be considered as section
        """
        self.min_sec_height = min_sec_height

    def __call__(self, document: Document) -> bool:
        for section in document.hierarchy.sections(self.min_sec_height):
            if not section.headline:
                return False
        return True


class NumberOfSectionsFilter(Filter):
    """
    Filters out all documents that have less than min_sec or more than max_sec sections.
    It expects that all provided documents have at least two lvl hierarchy, so the section exists. No exception is
    raised when otherwise.
    """

    def __init__(self, min_sec: float = 0, max_sec: float = math.inf, min_sec_height: int = 2, rw: bool = False):
        """
        Initialization of filter.

        :param min_sec: minimal number of sections in a document
        :param max_sec: maximal number of section in a document
        :param min_sec_height: min height of a sub-hierarchy to be considered as section
        :param rw: whether to work only on related work section
            it expects that the documents are OARelatedWorkDocuments
        """
        if min_sec > max_sec:
            raise AttributeError("The max_sec must be greater or equal to min_sec.")

        self.min_sec = min_sec
        self.max_sec = max_sec
        self.min_sec_height = min_sec_height
        self.rw = rw

    def __call__(self, document: Document) -> bool:

        content = document.hierarchy
        if self.rw:
            content = document.related_work

        return self.min_sec <= sum(1 for _ in content.sections(self.min_sec_height)) <= self.max_sec


class NumberOfTextPartsInSectionFilter(Filter):
    """
    Filters out all documents that have less than min_par or more than max_par text parts in a top lvl section.
    """

    def __init__(self, min_par: float = 0, max_par: float = math.inf, rw: bool = False):
        """
        Initialization of filter.

        :param min_par: minimal number of text parts in a section
        :param max_par: maximal number of text parts in a section
        :param rw: whether to work only on related work section
            it expects that the documents are OARelatedWorkDocuments
        """
        if min_par > max_par:
            raise AttributeError("The max_par must be greater or equal to min_par.")

        self.min_par = min_par
        self.max_par = max_par
        self.rw = rw

    def __call__(self, document: Document) -> bool:
        content = document.hierarchy
        if self.rw:
            document: OARelatedWorkDocument
            return self.min_par <= len(list(document.related_work.text_content())) <= self.max_par
        return all(self.min_par <= len(list(sec.text_content())) <= self.max_par for sec in content.content)


class NumberOfCitationsFilter(Filter):
    """
    Filters out all documents that have less than min_cit or more than max_cit citations in citations field.
    """

    def __init__(self, min_cit: float = 0, max_cit: float = math.inf):
        """
        Initialization of filter.

        :param min_cit: minimal number of citations
        :param max_cit: maximal number of citations
        """
        if min_cit > max_cit:
            raise AttributeError("The max_cit must be greater or equal to min_cit.")

        self.min_cit = min_cit
        self.max_cit = max_cit

    def __call__(self, document: Document) -> bool:
        return self.min_cit <= len(document.citations) <= self.max_cit


class CitationsFracFilter(Filter):
    """
    Filters out all documents that have lower/higher proportion of known citation spans to all citation spans in given
    hierarchy.

    If there are no citations in the document, it acts as there is zero coverage.
    """

    def __init__(self, min_cit_frac: float = 0, max_cit_frac: float = 1.0, rw: bool = False):
        """
        Initialization of filter.

        :param min_cit_frac: minimal proportion of known citations
        :param max_cit_frac: maximal proportion of known citations
        :param rw: whether to work only on related work section
            it expects that the documents are OARelatedWorkDocuments
        """
        if min_cit_frac > max_cit_frac:
            raise AttributeError("The max_cit_frac must be greater or equal to min_cit_frac.")

        self.min_cit_frac = min_cit_frac
        self.max_cit_frac = max_cit_frac
        self.rw = rw

    def __call__(self, document: Document) -> bool:
        known_content_cit = 0
        all_content_cit = 0

        content = document.hierarchy
        if self.rw:
            content = document.related_work

        for t in content.text_content():
            for c in t.citations:
                all_content_cit += 1
                if c.index is not None and document.bibliography[c.index].id is not None:
                    known_content_cit += 1

        frac = known_content_cit / all_content_cit if all_content_cit > 0 else 0

        return self.min_cit_frac <= frac <= self.max_cit_frac


class CitationsGroupsFracFilter(Filter):
    """
    Filters out all documents that have lower/higher proportion of known citation span groups to all
    citation span groups in given hierarchy.

    Example of citation group: [1,2,3]
     .. as was presented in [1,2,3] ...

    """

    def __init__(self, min_cit_frac: float = 0, max_cit_frac: float = 1.0, rw: bool = False):
        """
        Initialization of filter.

        :param min_cit_frac: minimal proportion of known citation span groups
        :param max_cit_frac: maximal proportion of known citation span groups
        :param rw: whether to work only on related work section
            it expects that the documents are OARelatedWorkDocuments
        """
        if min_cit_frac > max_cit_frac:
            raise AttributeError("The max_cit_frac must be greater or equal to min_cit_frac.")

        self.min_cit_frac = min_cit_frac
        self.max_cit_frac = max_cit_frac
        self.rw = rw

    def __call__(self, document: Document) -> bool:
        known_content_cit_groups = 0
        all_content_cit_groups = 0

        content = document.hierarchy
        if self.rw:
            content = document.related_work

        for t in content.text_content():
            previous_c = None
            previous_c_hit = False
            for c in t.citations:
                if previous_c is None or not re.match(r"^\W*$", t.text[previous_c.end: c.start]):
                    # new cit. group
                    known_content_cit_groups += previous_c_hit
                    all_content_cit_groups += 1
                    previous_c_hit = False

                if c.index is not None and document.bibliography[c.index].id is not None:
                    previous_c_hit = True
                previous_c = c
            known_content_cit_groups += previous_c_hit

        frac_res = known_content_cit_groups/all_content_cit_groups if all_content_cit_groups > 0 else 0

        return self.min_cit_frac <= frac_res <= self.max_cit_frac


class FullRecordFilter(Filter):
    """
    Filters out all documents that don't have title, authors, and content.
    """

    def __call__(self, document: Document) -> bool:
        return document.title is not None and len(document.authors) > 0 and document.hierarchy.has_text_content


class CouldEstablishMultLvlHierFilter(Filter):
    """
    Filters out all documents for which the multi-level hierarchy could not be established.
    Expects flat section structure.
    """

    def __call__(self, document: Document) -> bool:
        return document.hierarchy.flat_2_multi()


class IsValidAfterHierPruneFilter(Filter):
    """
    Filters out all documents that will be empty (without text content) after pruning .
    """

    def __init__(self, prune_empty_headlines_nodes: bool, prune_nodes_without_text_content: bool,
                 prune_according_to_name_assigned_to_chars_in_headline: Optional[Tuple[Pattern, float]],
                 min_height: int = 2, copy_doc: bool = True):
        """
        initialize filter with hierarchy pruning methods that should be applied

        :param prune_empty_headlines_nodes: Removes all sub-nodes with empty headline that have at least given height.
        :param prune_nodes_without_text_content: Removes all sub-nodes without text content (in whole sub-hierarchy).
        :param prune_according_to_name_assigned_to_chars_in_headline: Removes all sub-nodes with lower coverage of
            valid characters in headline that have at least given height.
            Provide tuple with matching pattern and minimal coverage.
        :param min_height: sub-hierarchy is considered when it has at least such height
        :param copy_doc: If False it will actually do the pruning on original document, thus the documents will be changed.
        :raise ValueError: when no pruning method is selected
        """

        if not prune_empty_headlines_nodes and not prune_nodes_without_text_content and \
                prune_according_to_name_assigned_to_chars_in_headline is None:
            raise ValueError("At least one pruning method must be selected.")

        self.prune_empty_headlines_nodes = prune_empty_headlines_nodes
        self.prune_nodes_without_text_content = prune_nodes_without_text_content
        self.prune_according_to_name_assigned_to_chars_in_headline = \
            prune_according_to_name_assigned_to_chars_in_headline
        self.min_height = min_height
        self.copy = copy_doc

    def __call__(self, document: Document) -> bool:
        hierarchy = document.hierarchy
        if self.copy:
            hierarchy = copy.deepcopy(hierarchy)

        if self.prune_empty_headlines_nodes:
            hierarchy.prune_empty_headlines_nodes(self.min_height)

        if self.prune_according_to_name_assigned_to_chars_in_headline:
            hierarchy.prune_according_to_name_assigned_to_chars_in_headline(
                self.prune_according_to_name_assigned_to_chars_in_headline[0],
                self.prune_according_to_name_assigned_to_chars_in_headline[1],
                self.min_height
            )

        if self.prune_nodes_without_text_content:
            hierarchy.prune_nodes_without_text_content()

        return hierarchy.has_text_content


class HasHeadlineFilter(Filter):
    """
    Filters out all documents that doesn't have at least one headline that matches given regex.
    """

    def __init__(self, rgx: Pattern, min_text_parts: int = 0, min_depth: float = 0, max_depth: float = math.inf):
        """
        Initialization of filter.

        :param rgx: regex that should be matched by at least one headline
        :param min_text_parts: minimal number of text parts to be counted
        :param min_depth: minimal depth of a node (root has zero depth)
        :param max_depth: maximal depth of a node
        """
        self.rgx = rgx
        self.min_text_parts = min_text_parts
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, document: Document) -> bool:
        return any(
            len(list(sec.text_content())) >= self.min_text_parts for sec in
            document.hierarchy.get_part(self.rgx, max_h=1, min_depth=self.min_depth, max_depth=self.max_depth)
        )


class FractionOfCitedDocumentsWithMultiSectionContentFilter(Filter):
    """
    Filters out all documents that don't have given proportion of cited documents with multi-section content.

    If there are no citations, fraction is 0.
    """

    def __init__(self, dataset: OADataset, min_cit_frac: float = 0, max_cit_frac: float = 1.0):
        """
        Initialization of filter.

        :param dataset: dataset with cited documents
        :param min_cit_frac: minimal proportion of cited documents with multi-section content
        :param max_cit_frac: maximal proportion of cited documents with multi-section content
        """
        self.min_cit_frac = min_cit_frac
        self.max_cit_frac = max_cit_frac
        self.dataset = dataset

    def __call__(self, document: Document) -> bool:
        if self.min_cit_frac <= 0 and self.max_cit_frac >= 1.0:
            return True

        frac = sum(1 for i in document.citations if i in self.dataset and len(self.dataset.get_by_id(i).hierarchy.content) > 1) \
               / len(document.citations) if len(document.citations) > 0 else 0

        return self.min_cit_frac <= frac <= self.max_cit_frac


class FieldsOfStudyFilter(Filter):
    """
    Filters out all documents that don't have given field of study.
    """

    def __init__(self, fields_of_study: Iterable[str], case_sensitive: bool = False):
        """
        Initialization of filter.

        :param fields_of_study: set of field of study, at least one must be present
        :param case_sensitive: if True, fields of study are case-sensitive
        """
        self.case_sensitive = case_sensitive
        self.fields_of_study = set(fields_of_study) if case_sensitive else set(f.lower() for f in fields_of_study)

    def __call__(self, document: Document) -> bool:
        if document.fields_of_study is None:
            return False

        fields_of_study_names = [f if isinstance(f, str) else f[0] for f in document.fields_of_study]
        f = set(fields_of_study_names) if self.case_sensitive else set(f.lower() for f in fields_of_study_names)
        return len(f & self.fields_of_study) > 0
