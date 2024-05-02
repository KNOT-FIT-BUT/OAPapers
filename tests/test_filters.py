# -*- coding: UTF-8 -*-
""""
Created on 11.04.22

:author:     Martin Dočekal
"""
import re
import unittest
from pathlib import Path
from unittest import TestCase

from oapapers.bib_entry import BibEntry
from oapapers.datasets import OADataset
from oapapers.document import Document
from oapapers.hierarchy import Hierarchy, TextContent, RefSpan
from oapapers.filters import Filter, CombinedFilter, SecNonEmptyHeadlinesFilter, NumberOfSectionsFilter, \
    AlwaysFilter, NumberOfTextPartsInSectionFilter, NumberOfCitationsFilter, FullRecordFilter, \
    IsValidAfterHierPruneFilter, HasHeadlineFilter, CitationsFracFilter, CitationsGroupsFracFilter, \
    FractionOfCitedDocumentsWithMultiSectionContentFilter

SCRIPT_DIR = Path(__file__).parent


class MockFilter(Filter):

    def __init__(self, res: bool):
        self.res = res

    def __call__(self, document: Document) -> bool:
        return self.res


class TestCombinedFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.document = Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737",
                                 title="title 1", authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                 citations=[1, 2, 3], hierarchy=Hierarchy(
                headline="title 1", content=[
                    Hierarchy("section 1", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                    Hierarchy("section 2", [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                    Hierarchy("section 3", [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                ]
            ), bibliography=[], non_plaintext_content=[], uncategorized_fields={})

    def test_single(self):
        f = CombinedFilter([MockFilter(True)])
        self.assertTrue(f(self.document))

        f = CombinedFilter([MockFilter(False)])
        self.assertFalse(f(self.document))

    def test_combined(self):
        f = CombinedFilter([MockFilter(True), MockFilter(True), MockFilter(True)])
        self.assertTrue(f(self.document))

        f = CombinedFilter([MockFilter(True), MockFilter(False), MockFilter(True)])
        self.assertFalse(f(self.document))


class TestSecNonEmptyHeadlinesFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                headline="title 1", content=[
                    Hierarchy("section 1", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                    Hierarchy("section 2", [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                    Hierarchy("section 3", [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                ]
            ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=11, mag_id=1, doi=None, title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy("section 2",
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=20, mag_id=2, doi=None, title="title 1", authors=["a", "b"],
                                   year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          ]

    def test_non_empty(self):
        f = SecNonEmptyHeadlinesFilter(min_sec_height=1)
        self.assertListEqual([True, True, False], [f(d) for d in self.documents])


class TestNumberOfSectionsFilter(unittest.TestCase):
    def setUp(self) -> None:
        # sections: 1, 5, 3, 4
        self.documents = [Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                headline="title 1", content=[
                    Hierarchy("section 1", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                ]
            ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=11, mag_id=1, doi=None, title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=None, mag_id=None, doi=None, title="title 1", authors=["a", "b"],
                                   year=2010,
                                   fields_of_study=["f1", "f2"], citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy("section 2",
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=None, mag_id=None, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          ]

    def test_min(self):
        f = NumberOfSectionsFilter(min_sec=4, min_sec_height=1)
        self.assertListEqual([False, True, False, True], [f(d) for d in self.documents])

    def test_max(self):
        f = NumberOfSectionsFilter(max_sec=3, min_sec_height=1)
        self.assertListEqual([True, False, True, False], [f(d) for d in self.documents])

    def test_min_max(self):
        f = NumberOfSectionsFilter(min_sec=3, max_sec=4, min_sec_height=1)
        self.assertListEqual([False, False, True, True], [f(d) for d in self.documents])

    def test_min_max_attribute_error(self):
        _ = NumberOfSectionsFilter(min_sec=4, max_sec=4, min_sec_height=1)
        with self.assertRaises(AttributeError):
            _ = NumberOfSectionsFilter(min_sec=5, max_sec=4, min_sec_height=1)


class TestAlwaysFilter(unittest.TestCase):
    def setUp(self) -> None:
        # sections: 1, 5, 3, 4
        self.documents = [Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737",
                                   title="title 1", authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                headline="title 1", content=[
                    Hierarchy("section 1", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                ]
            ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737",
                                   title="title 1", authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737",
                                   title="title 1", authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy("section 2",
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          ]

    def test_false(self):
        f = AlwaysFilter(False)
        self.assertListEqual([False, False, False, False], [f(d) for d in self.documents])

    def test_true(self):
        f = AlwaysFilter(True)
        self.assertListEqual([True, True, True, True], [f(d) for d in self.documents])


class TestNumberOfTextPartsInSectionFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                headline="title 1", content=[
                    Hierarchy("section 1", [
                        Hierarchy(headline=None, content=TextContent("text 1", [], [])),
                        Hierarchy(headline=None, content=TextContent("text 2", [], []))
                    ]),
                ]
            ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 2", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 3", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 4", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 5", [], []))]),
                                      Hierarchy("section 2",
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 2", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 3", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 2", [], [])),
                                                 Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          ]

    def test_min(self):
        f = NumberOfTextPartsInSectionFilter(min_par=2)
        self.assertListEqual([True, False, True, False], [f(d) for d in self.documents])

    def test_max(self):
        f = NumberOfTextPartsInSectionFilter(max_par=3)
        self.assertListEqual([True, True, False, True], [f(d) for d in self.documents])

    def test_min_max(self):
        f = NumberOfTextPartsInSectionFilter(min_par=2, max_par=3)
        self.assertListEqual([True, False, False, False], [f(d) for d in self.documents])

    def test_min_max_attribute_error(self):
        _ = NumberOfTextPartsInSectionFilter(min_par=4, max_par=4)
        with self.assertRaises(AttributeError):
            _ = NumberOfTextPartsInSectionFilter(min_par=5, max_par=4)


class TestNumberOfCitationsFilter(TestCase):
    def setUp(self) -> None:
        self.documents = [Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                headline="title 1", content=[
                    Hierarchy("section 1", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                    Hierarchy("section 2", [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                    Hierarchy("section 3", [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                ]
            ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3, 4, 5], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy("section 2",
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          ]

    def test_min(self):
        f = NumberOfCitationsFilter(min_cit=3)
        self.assertListEqual([True, True, False], [f(d) for d in self.documents])

    def test_max(self):
        f = NumberOfCitationsFilter(max_cit=3)
        self.assertListEqual([True, False, True], [f(d) for d in self.documents])

    def test_min_max(self):
        f = NumberOfCitationsFilter(min_cit=1, max_cit=3)
        self.assertListEqual([True, False, True], [f(d) for d in self.documents])

    def test_min_max_attribute_error(self):
        _ = NumberOfCitationsFilter(min_cit=4, max_cit=4)
        with self.assertRaises(AttributeError):
            _ = NumberOfCitationsFilter(min_cit=5, max_cit=4)


class TestCitationsFracFilter(TestCase):
    def setUp(self) -> None:
        # known citations: 2
        # citations spans: 7
        # known citations spans: 4
        # citations groups: 4
        # known citations groups: 3
        self.document = Document(
            id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1", authors=["a", "b"],
            year=2010,
            fields_of_study=["f1", "f2"], citations=[1, 4],
            hierarchy=Hierarchy(headline="title 1", content=[
                Hierarchy("section 1", [
                    Hierarchy(headline=None, content=TextContent("text 1 [1, 2, 3]",
                                                                 [
                                                                     RefSpan(0, 7, 9), RefSpan(1, 11, 12),
                                                                     RefSpan(2, 14, 16)
                                                                 ], []))
                ]),
                Hierarchy("section 2", [
                    Hierarchy(headline=None, content=TextContent("text 2 [4][5]",
                                                                 [
                                                                     RefSpan(3, 7, 10), RefSpan(4, 10, 13)
                                                                 ], []))
                ]),
                Hierarchy("section 3", [
                    Hierarchy(headline=None, content=TextContent("text 3 [1] and [4]",
                                                                 [
                                                                     RefSpan(0, 7, 10), RefSpan(3, 15, 18)
                                                                 ], []))
                ]),
            ]),
            bibliography=[
                BibEntry(1, "Bib 1", None, ("author",)), BibEntry(None, "Bib 2", None, ("author",)),
                BibEntry(None, "Bib 3", None, ("author",)), BibEntry(4, "Bib 4", None, ("author",)),
                BibEntry(None, "Bib 5", None, ("author",))
            ],
            non_plaintext_content=[], uncategorized_fields={})

    def test_min(self):
        self.assertTrue(CitationsFracFilter(min_cit_frac=0.57)(self.document))
        self.assertFalse(CitationsFracFilter(min_cit_frac=0.6)(self.document))

    def test_max(self):
        self.assertTrue(CitationsFracFilter(max_cit_frac=0.6)(self.document))
        self.assertFalse(CitationsFracFilter(max_cit_frac=0.5)(self.document))

    def test_min_max(self):
        self.assertTrue(CitationsFracFilter(min_cit_frac=0.57, max_cit_frac=0.6)(self.document))
        self.assertFalse(CitationsFracFilter(min_cit_frac=0.2, max_cit_frac=0.5)(self.document))
        self.assertFalse(CitationsFracFilter(min_cit_frac=0.6, max_cit_frac=0.8)(self.document))

    def test_min_max_attribute_error(self):
        _ = CitationsFracFilter(min_cit_frac=0.1, max_cit_frac=0.2)
        with self.assertRaises(AttributeError):
            _ = CitationsFracFilter(min_cit_frac=0.4, max_cit_frac=0.2)


class TestCitationsGroupsFracFilter(TestCase):
    def setUp(self) -> None:
        # known citations: 2
        # citations spans: 7
        # known citations spans: 4
        # citations groups: 4
        # known citations groups: 3
        self.document = Document(
            id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1", authors=["a", "b"],
            year=2010,
            fields_of_study=["f1", "f2"], citations=[1, 4],
            hierarchy=Hierarchy(headline="title 1", content=[
                Hierarchy("section 1", [
                    Hierarchy(headline=None, content=TextContent("text 1 [1, 2, 3]",
                                                                 [
                                                                     RefSpan(0, 7, 9), RefSpan(1, 11, 12),
                                                                     RefSpan(2, 14, 16)
                                                                 ], []))
                ]),
                Hierarchy("section 2", [
                    Hierarchy(headline=None, content=TextContent("text 2 [4][5]",
                                                                 [
                                                                     RefSpan(3, 7, 10), RefSpan(4, 10, 13)
                                                                 ], []))
                ]),
                Hierarchy("section 3", [
                    Hierarchy(headline=None, content=TextContent("text 3 [1] and [2]",
                                                                 [
                                                                     RefSpan(0, 7, 10), RefSpan(1, 15, 18)
                                                                 ], []))
                ]),
            ]),
            bibliography=[
                BibEntry(1, "Bib 1", None, ("author",)), BibEntry(None, "Bib 2", None, ("author",)),
                BibEntry(None, "Bib 3", None, ("author",)), BibEntry(4, "Bib 4", None, ("author",)),
                BibEntry(None, "Bib 5", None, ("author",))
            ],
            non_plaintext_content=[], uncategorized_fields={})

    def test_min(self):
        self.assertTrue(CitationsGroupsFracFilter(min_cit_frac=0.75)(self.document))
        self.assertFalse(CitationsGroupsFracFilter(min_cit_frac=0.76)(self.document))

    def test_max(self):
        self.assertTrue(CitationsGroupsFracFilter(max_cit_frac=0.75)(self.document))
        self.assertFalse(CitationsGroupsFracFilter(max_cit_frac=0.5)(self.document))

    def test_min_max(self):
        self.assertTrue(CitationsGroupsFracFilter(min_cit_frac=0.25, max_cit_frac=0.8)(self.document))
        self.assertFalse(CitationsGroupsFracFilter(min_cit_frac=0.2, max_cit_frac=0.5)(self.document))
        self.assertFalse(CitationsGroupsFracFilter(min_cit_frac=0.8, max_cit_frac=0.9)(self.document))

    def test_min_max_attribute_error(self):
        _ = CitationsGroupsFracFilter(min_cit_frac=0.1, max_cit_frac=0.2)
        with self.assertRaises(AttributeError):
            _ = CitationsGroupsFracFilter(min_cit_frac=0.4, max_cit_frac=0.2)


class TestFullRecordFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                headline="title 1", content=[
                    Hierarchy("section 1", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                    Hierarchy("section 2", [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                    Hierarchy("section 3", [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                ]
            ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                []),
                                      Hierarchy("section 2",
                                                []),
                                      Hierarchy("section 3",
                                                []),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          ]

    def test_full_record(self):
        f = FullRecordFilter()
        self.assertListEqual([True, False, False], [f(d) for d in self.documents])


class TestIsValidAfterHierPruneFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                headline="title 1", content=[
                    Hierarchy("", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                    Hierarchy("", [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                    Hierarchy(None, [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                ]
            ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("", [], []))]),
                                      Hierarchy("section 2",
                                                [Hierarchy(headline=None, content=TextContent("", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("section 1",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy(None,
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                                      Hierarchy("section 3",
                                                [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                                   authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                                   citations=[1, 2, 3], hierarchy=Hierarchy(
                                  headline="title 1", content=[
                                      Hierarchy("ØÓÖ ½ ØÓÖ ¾",
                                                [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                                      Hierarchy("ÐÙ×ØØÖ",
                                                [Hierarchy(headline=None, content=TextContent("text 2", [], []))])
                                  ]
                              ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
                          ]
        self.regex_latin = re.compile(r"^LATIN (CAPITAL|SMALL) LETTER .$", re.IGNORECASE | re.UNICODE)

    def test_every_pruning(self):
        f = IsValidAfterHierPruneFilter(True, True, (self.regex_latin, 0.75), min_height=1)
        self.assertListEqual([False, False, True, False], [f(d) for d in self.documents])
        f = IsValidAfterHierPruneFilter(True, True, (self.regex_latin, 0.75), min_height=1, copy_doc=False)
        self.assertListEqual([False, False, True, False], [f(d) for d in self.documents])

    def test_empty_headlines(self):
        f = IsValidAfterHierPruneFilter(True, False, None, min_height=1)
        self.assertListEqual([False, False, True, True], [f(d) for d in self.documents])
        f = IsValidAfterHierPruneFilter(True, False, None, min_height=1, copy_doc=False)
        self.assertListEqual([False, False, True, True], [f(d) for d in self.documents])

    def test_without_text_content(self):
        f = IsValidAfterHierPruneFilter(False, True, None, min_height=1)
        self.assertListEqual([True, False, True, True], [f(d) for d in self.documents])
        f = IsValidAfterHierPruneFilter(False, True, None, min_height=1, copy_doc=False)
        self.assertListEqual([True, False, True, True], [f(d) for d in self.documents])

    def test_prune_according_to_name_assigned_to_chars_in_headline(self):
        f = IsValidAfterHierPruneFilter(False, False, (self.regex_latin, 0.75), min_height=1)
        self.assertListEqual([False, False, True, False], [f(d) for d in self.documents])
        f = IsValidAfterHierPruneFilter(False, False, (self.regex_latin, 0.75), min_height=1, copy_doc=False)
        self.assertListEqual([False, False, True, False], [f(d) for d in self.documents])

    def test_nothing(self):
        with self.assertRaises(ValueError):
            _ = IsValidAfterHierPruneFilter(False, False, None)


class TestHasHeadlineFilter(TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                     authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                     citations=[1, 2, 3], hierarchy=Hierarchy(
                    headline="title 1", content=[
                        Hierarchy("section 1", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                    ]
                ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
            Document(id=1, s2orc_id=11, mag_id=1, doi=None, title="title 1",
                     authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                     citations=[1, 2, 3], hierarchy=Hierarchy(
                    headline="title 1", content=[
                        Hierarchy("section 1",
                                  [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                        Hierarchy(None,
                                  [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                        Hierarchy("abstract",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                        Hierarchy("section 3",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                        Hierarchy("section 3",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                    ]
                ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
            Document(id=1, s2orc_id=None, mag_id=None, doi=None, title="title 1", authors=["a", "b"], year=2010,
                     fields_of_study=["f1", "f2"], citations=[1, 2, 3], hierarchy=Hierarchy(
                    headline="title 1", content=[
                        Hierarchy("section 1",
                                  [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                        Hierarchy("section 2",
                                  [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                        Hierarchy("section 3",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                    ]
                ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
            Document(id=1, s2orc_id=None, mag_id=None, doi="10.32473/flairs.v35i.130737", title="title 1",
                     authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                     citations=[1, 2, 3], hierarchy=Hierarchy(
                    headline="title 1", content=[
                        Hierarchy("Abstract",
                                  [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                        Hierarchy(None,
                                  [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                        Hierarchy("section 3",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                        Hierarchy("section 3",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                    ]
                ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
        ]

    def test_match(self):
        abstract_regex = re.compile(r"^((^|\s|\()((((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|"
                                    r"[0-9]+|[a-z])(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+"
                                    r"|[a-z]))*\.?)($|\s|\)))?\s*abstract\s*$", re.IGNORECASE)
        f = HasHeadlineFilter(abstract_regex)
        self.assertListEqual([False, True, False, True], [f(d) for d in self.documents])


class TestFractionOfCitedDocumentsWithMultiSectionContentFilter(TestCase):
    FIXTURE_FILE = str(
        SCRIPT_DIR / "fixtures" / "test_fraction_of_cited_documents_with_multi_section_content_filter.jsonl")

    def setUp(self) -> None:
        self.documents = [
            Document(id=1, s2orc_id=321, mag_id=123, doi="10.32473/flairs.v35i.130737", title="title 1",
                     authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                     citations=[1, 2, 3], hierarchy=Hierarchy(
                    headline="title 1", content=[
                        Hierarchy("section 1", [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                    ]
                ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
            Document(id=1, s2orc_id=11, mag_id=1, doi=None, title="title 1",
                     authors=["a", "b"], year=2010, fields_of_study=["f1", "f2"],
                     citations=[4, 5, 6], hierarchy=Hierarchy(
                    headline="title 1", content=[
                        Hierarchy("section 1",
                                  [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                        Hierarchy(None,
                                  [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                        Hierarchy("abstract",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                        Hierarchy("section 3",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                        Hierarchy("section 3",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                    ]
                ), bibliography=[], non_plaintext_content=[], uncategorized_fields={}),
            Document(id=1, s2orc_id=None, mag_id=None, doi=None, title="title 1", authors=["a", "b"], year=2010,
                     fields_of_study=["f1", "f2"], citations=[1, 4, 3], hierarchy=Hierarchy(
                    headline="title 1", content=[
                        Hierarchy("section 1",
                                  [Hierarchy(headline=None, content=TextContent("text 1", [], []))]),
                        Hierarchy("section 2",
                                  [Hierarchy(headline=None, content=TextContent("text 2", [], []))]),
                        Hierarchy("section 3",
                                  [Hierarchy(headline=None, content=TextContent("text 3", [], []))]),
                    ]
                ), bibliography=[], non_plaintext_content=[], uncategorized_fields={})
        ]

        self.references = OADataset(self.FIXTURE_FILE)
        self.references.open()

    def tearDown(self) -> None:
        self.references.close()

    def test_match_min(self):
        self.filter = FractionOfCitedDocumentsWithMultiSectionContentFilter(self.references, min_cit_frac=0.5)
        self.assertSequenceEqual([True, False, True], [self.filter(d) for d in self.documents])

    def test_match_max(self):
        self.filter = FractionOfCitedDocumentsWithMultiSectionContentFilter(self.references, max_cit_frac=0.5)
        self.assertSequenceEqual([False, True, False], [self.filter(d) for d in self.documents])

    def test_match_min_max(self):
        self.filter = FractionOfCitedDocumentsWithMultiSectionContentFilter(self.references, min_cit_frac=0.5,
                                                                            max_cit_frac=0.75)
        self.assertSequenceEqual([False, False, True], [self.filter(d) for d in self.documents])


if __name__ == '__main__':
    unittest.main()
