# -*- coding: UTF-8 -*-
""""
Created on 28.02.22

:author:     Martin Dočekal
"""
import os
import unittest

from oapapers.bib_entry import BibEntry
from oapapers.datasets import OADataset, OARelatedWork
from oapapers.document import Document
from oapapers.hierarchy import Hierarchy, TextContent, RefSpan
from oapapers.stats import DocumentsStats, RelatedWorkStats

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures")


class TestDocumentsStats(unittest.TestCase):
    def setUp(self) -> None:
        self.stats = DocumentsStats()

    def test_process(self):
        documents = [
            Document(
                id=0,
                s2orc_id=11,
                mag_id=1,
                doi="something",
                title="Title 0",
                authors=["Author A.", "Author B."],
                year=2022,
                fields_of_study=["Mathematics"],
                citations=[3, 12],
                hierarchy=Hierarchy(
                    "Title 0",
                    [
                        Hierarchy(
                            "Headline 1",
                            [
                                Hierarchy(None, [
                                    Hierarchy(None, TextContent("So [1] far so good.", [RefSpan(2, 4, 5)], [])),
                                    Hierarchy(None, TextContent("sentence 2", [], []))
                                ]),
                                Hierarchy("formula", TextContent("1 + 1 = 3", [], []))
                            ]
                        ),
                        Hierarchy(
                            "Headline 2", [
                                Hierarchy(None, [
                                    Hierarchy(None, TextContent("No citations here.", [], []))
                                ])
                            ]
                        )
                    ]

                ),
                bibliography=[
                    BibEntry(None, "BIB ENTRY 1", None, ("Author 1",)),
                    BibEntry(1, "BIB ENTRY 2", None, ("Author 1",)),
                    BibEntry(2, "BIB ENTRY 3", None, ("Author 1",)),
                ],
                non_plaintext_content=[("figure", " description of figure 1"), ("table", "description of table 1")],
                uncategorized_fields={}
            ),
            Document(
                id=1,
                s2orc_id=12,
                mag_id=2,
                doi="something else",
                title="Title 1",
                authors=["Author C."],
                year=2022,
                fields_of_study=["Biology"],
                citations=[3],
                hierarchy=Hierarchy(
                    "Title 1",
                    [
                        Hierarchy(
                            "Headline 1",
                            [
                                Hierarchy(
                                    "Sub-headline 1", [
                                        Hierarchy(None, [
                                            Hierarchy(
                                                None,
                                                TextContent(text="So (Morreall, 2016) far so good.",
                                                            citations=[RefSpan(3, 4, 18)], references=[])
                                            )
                                        ]),
                                    ]
                                )
                            ]
                        ),
                        Hierarchy(
                            "Headline 2", [
                                Hierarchy(None, [
                                    Hierarchy(
                                        None,
                                        TextContent(text="No citations here.", citations=[], references=[])
                                    )
                                ])
                            ]
                        ),
                        Hierarchy(
                            "Headline 3", [
                                Hierarchy(None, [
                                    Hierarchy(
                                        None,
                                        TextContent(text="Unknown citation in here (Someone, 2011).",
                                                    citations=[RefSpan(None, 26, 39)], references=[])
                                    )
                                ])
                            ]
                        ),
                    ]
                ),
                bibliography=[
                    BibEntry(None, "BIB ENTRY 1", None, ("Author 1", )),
                    BibEntry(1, "BIB ENTRY 2", None, ("Author 1", )),
                    BibEntry(2, "BIB ENTRY 3", None, ("Author 1", )),
                    BibEntry(3, "BIB ENTRY 4", None, ("Author 1", )),
                    BibEntry(None, "BIB ENTRY 5", None, ("Author 1", ))
                ],
                non_plaintext_content=[("figure", " description of figure 1")],
                uncategorized_fields={}
            ),
            Document(
                id=2,
                s2orc_id=100,
                mag_id=10,
                doi="another",
                title="Title 2",
                authors=["Author C.", "Author D.", "Author E."],
                year=1994,
                fields_of_study=["Language"],
                citations=[3, 32],
                hierarchy=Hierarchy(
                    "Title 2",
                    [
                        Hierarchy(
                            "Headline 1",
                            [
                                Hierarchy(None, [
                                    Hierarchy(
                                        None,
                                        TextContent(text="So (Morreall, 2016) far so good.",
                                                    citations=[RefSpan(3, 4, 18)], references=[])
                                    )
                                ])
                            ]
                        ),
                        Hierarchy(
                            "",
                            [
                                Hierarchy(None, [
                                    Hierarchy(
                                        None,
                                        TextContent(text="Not processed citations here (Morreall, 2016).", citations=[],
                                                    references=[])
                                    )
                                ])
                            ]
                        ),
                        Hierarchy(
                            "Headline 3",
                            [
                                Hierarchy(None, [
                                    Hierarchy(
                                        None,
                                        TextContent(text="Unknown citation in here (Someone, 2011).",
                                                    citations=[RefSpan(None, 26, 39)], references=[])
                                    )
                                ])
                            ]
                        ),
                    ]
                ),
                bibliography=[
                    BibEntry(None, "BIB ENTRY 1", None, ("Author 1",)),
                    BibEntry(1, "BIB ENTRY 2", None, ("Author 1",)),
                    BibEntry(2, "BIB ENTRY 3", None, ("Author 1",)),
                    BibEntry(3, "BIB ENTRY 4", None, ("Author 1",)),
                    BibEntry(None, "BIB ENTRY 5", None, ("Author 1",))
                ],
                non_plaintext_content=[("figure", " description of figure 1")],
                uncategorized_fields={}
            ),
            Document(
                id=3,
                s2orc_id=200,
                mag_id=20,
                doi="another one",
                title="Title 3",
                authors=["Author F."],
                year=2022,
                fields_of_study=["Mathematics"],
                citations=[],
                hierarchy=Hierarchy(
                    "Title 3",
                    [
                        Hierarchy(
                            "Headline 1",
                            [
                                Hierarchy(None, [
                                    Hierarchy(
                                        None,
                                        TextContent(text="So far so good.", citations=[], references=[])
                                    )
                                ])
                            ]
                        ),
                        Hierarchy(
                            "Headline 2",
                            [
                                Hierarchy(None, [
                                    Hierarchy(
                                        None,
                                        TextContent(text="No citations here.", citations=[], references=[])
                                    )
                                ])
                            ]
                        )
                    ]

                ),
                bibliography=[
                    BibEntry(None, "BIB ENTRY 1", None, ("Author 1",)),
                    BibEntry(1, "BIB ENTRY 2", None, ("Author 1",)),
                ],
                non_plaintext_content=[],
                uncategorized_fields={}
            )
        ]

        for d in documents:
            self.stats.process(d)

        self.assertEqual(4, self.stats.num_of_documents)
        self.assertEqual(2, self.stats.num_of_documents_with_non_referenced_citations)
        self.assertEqual(2, self.stats.num_of_documents_with_all_known_citations)
        self.assertEqual(3, self.stats.num_of_documents_with_non_empty_headlines)
        self.assertEqual(1, self.stats.num_of_documents_with_suspicion_for_missing_reference)

        self.assertEqual({56: 1, 91: 1, 119: 1, 33: 1}, self.stats.hist_of_num_of_chars_per_document)
        self.assertEqual({15: 2, 18: 1, 7: 1}, self.stats.hist_of_num_of_words_per_document)
        self.assertEqual({2: 1, 3: 2, 4: 1}, self.stats.hist_of_num_of_sentences_per_document)
        self.assertEqual({2: 2, 3: 2}, self.stats.hist_of_num_of_paragraphs_per_document)
        self.assertEqual({2: 2, 3: 1, 4: 1}, self.stats.hist_of_num_of_sections_per_document)

        self.assertEqual({90: 1, 142: 1, 146: 1, 60: 1}, self.stats.hist_of_num_of_chars_per_document_with_headlines)
        self.assertEqual({2: 2, 3: 2}, self.stats.hist_of_num_of_top_lvl_sections_per_document)
        self.assertEqual({3: 1, 1: 9}, self.stats.hist_of_num_of_text_parts_per_top_lvl_section)
        self.assertEqual({19: 1, 18: 3, 32: 2, 41: 2, 46: 1, 15: 1, 10: 1, 9: 1}, self.stats.hist_of_text_parts_len_chars)

        self.assertEqual({2022: 3, 1994: 1}, self.stats.hist_of_years)
        self.assertEqual({1: 2, 2: 1, 3: 1}, self.stats.hist_of_num_of_authors_per_document)
        self.assertEqual({0: 1, 1: 1, 2: 2}, self.stats.hist_of_num_of_citations_per_document)
        self.assertEqual({0: 1, 1: 2, 2: 1}, self.stats.hist_of_num_of_non_plaintexts_per_document)
        self.assertEqual({"mathematics": 2, "biology": 1, "language": 1}, self.stats.hist_of_fields)
        self.assertEqual({'HARVARD': 2, 'UNKNOWN': 1, 'VANCOUVER_SQUARE_BRACKETS': 1},
                         self.stats.hist_of_citations_styles)


class TestDocumentsStatsStr(unittest.TestCase):

    def setUp(self) -> None:
        self.stats = DocumentsStats()
        self.stats.num_of_documents = 10
        self.stats.num_of_documents_with_all_known_citations = 3
        self.stats.num_of_documents_with_non_referenced_citations = 1
        self.stats.num_of_documents_with_non_empty_headlines = 4
        self.stats.num_of_documents_with_suspicion_for_missing_reference = 5
        self.stats.num_of_citation_spans = 10
        self.stats.num_of_bibliography_entries = 3
        self.stats.num_of_bibliography_entries_with_id = 2

        self.stats.hist_of_years = {1: 1}
        self.stats.hist_of_num_of_authors_per_document = {2: 2}
        self.stats.hist_of_num_of_chars_per_document = {120: 1}
        self.stats.hist_of_num_of_chars_per_document_with_headlines = {120: 2}
        self.stats.hist_of_num_of_words_per_document = {3: 20}
        self.stats.hist_of_num_of_sentences_per_document = {3: 1}
        self.stats.hist_of_num_of_paragraphs_per_document = {3: 2}
        self.stats.hist_of_num_of_sections_per_document = {3: 3}
        self.stats.hist_of_num_of_top_lvl_sections_per_document = {3: 4}

        self.stats.hist_of_num_of_text_parts_per_top_lvl_section = {4: 4}
        self.stats.hist_of_text_parts_len_chars = {4: 5}
        self.stats.hist_of_num_of_citations_per_document = {5: 5}
        self.stats.hist_of_num_of_non_plaintexts_per_document = {6: 6}
        self.stats.hist_of_fields = {"mathematics": 10}
        self.stats.hist_of_citations_styles = {
            "HARVARD": 20, "VANCOUVER_PARENTHESIS": 10, "VANCOUVER_SQUARE_BRACKETS": 5
        }

    def test_str(self):

        self.assertEqual(
            "Number of documents\t10\n"
            "Number of documents with all known citations\t3\n"
            "Number of documents with non referenced citations\t1\n"
            "Number of documents with just non empty headlines\t4\n"
            "Number of documents with suspicion for at least one missing reference\t5\n"
            "Number of citation spans\t10\n"
            "Number of bibliography entries	3\n"
            "Number of bibliography entries with id	2\n"
            "Histogram of years\n"
            "1 ████████████████████████████████████████ 1\n"
            "Histogram of authors per document\n"
            "2 ████████████████████████████████████████ 2\n"
            "Histogram of chars per document\n"
            "120 ████████████████████████████████████████ 1\n"
            "Histogram of chars per document with headlines\n"
            "120 ████████████████████████████████████████ 2\n"
            "Histogram of words per document\n"
            "3 ████████████████████████████████████████ 20\n"
            "Histogram of sentences per document\n"
            "3 ████████████████████████████████████████ 1\n"
            "Histogram of paragraphs per document\n"
            "3 ████████████████████████████████████████ 2\n"
            "Histogram of sections per document\n"
            "3 ████████████████████████████████████████ 3\n"
            "Histogram of top lvl sections per document\n"
            "3 ████████████████████████████████████████ 4\n"
            "Histogram of text parts per top lvl section\n"
            "4 ████████████████████████████████████████ 4\n"
            "Histogram of chars per text part\n"
            "4 ████████████████████████████████████████ 5\n"
            "Histogram of citations per document\n"
            "5 ████████████████████████████████████████ 5\n"
            "Histogram of non-plaintexts per document\n"
            "6 ████████████████████████████████████████ 6\n"
            "Histogram of fields of study\n"
            "mathematics ████████████████████████████████████████ 10\n"
            "Histogram of citation styles\n"
            "HARVARD                   ████████████████████████████████████████ 20\n"
            "VANCOUVER_PARENTHESIS     ████████████████████ 10\n"
            "VANCOUVER_SQUARE_BRACKETS ██████████ 5\n"
            , str(self.stats)
        )


class TestRelatedWorkStats(unittest.TestCase):
    def setUp(self) -> None:
        path_to_references_dataset = os.path.join(FIXTURES_PATH, "references.jsonl")
        self.references_dataset = OADataset(path_to_references_dataset, path_to_references_dataset+".index").open()
        self.stats = RelatedWorkStats(self.references_dataset)

    def tearDown(self) -> None:
        self.references_dataset.close()

    def test_process(self):
        p = os.path.join(FIXTURES_PATH, "related_work.jsonl")
        with OARelatedWork(p, p+".index") as related_work_dataset:
            for d in related_work_dataset:
                self.stats.process(d)

        self.assertEqual(1, self.stats.num_of_targets)
        self.assertEqual(1, self.stats.num_of_citations)
        self.assertEqual({'on', '[cite:1]', 'introduction', '1', 'polytopes', 'section', 'of', 'regular'},
                         self.stats.target_vocabulary)
        self.assertEqual({"abstract", "of", "this", "is", "not", "a", "review:", "part", "two", "section", "1"},
                            self.stats.abstracts_vocabulary)
        self.assertEqual({"abstract", "introduction", "section", "1", "of", "this", "is", "not", "a", "review:", "part", "two",
                          "[cite:0]"},
                            self.stats.all_input_content_vocabulary)

        self.assertEqual({1: 1}, self.stats.hist_of_num_of_sections_per_target)
        self.assertEqual({1: 1}, self.stats.hist_of_num_of_paragraphs_per_target)
        self.assertEqual({1: 1}, self.stats.hist_of_num_of_sentences_per_target)
        self.assertEqual({8: 1}, self.stats.hist_of_num_of_words_per_target)
        self.assertEqual({55: 1}, self.stats.hist_of_num_of_chars_per_target)

        self.assertEqual({1: 1}, self.stats.hist_of_num_of_sections_per_input_abstracts)
        self.assertEqual({1: 1}, self.stats.hist_of_num_of_paragraphs_per_input_abstracts)
        self.assertEqual({1: 1}, self.stats.hist_of_num_of_sentences_per_input_abstracts)
        self.assertEqual({11: 1}, self.stats.hist_of_num_of_words_per_input_abstracts)
        self.assertEqual({52: 1}, self.stats.hist_of_num_of_chars_per_input_abstracts)

        self.assertEqual({3: 1}, self.stats.hist_of_num_of_sections_per_input_all_content)
        self.assertEqual({2: 1}, self.stats.hist_of_num_of_paragraphs_per_input_all_content)
        self.assertEqual({2: 1}, self.stats.hist_of_num_of_sentences_per_input_all_content)
        self.assertEqual({23: 1}, self.stats.hist_of_num_of_words_per_input_all_content)
        self.assertEqual({117: 1}, self.stats.hist_of_num_of_chars_per_input_all_content)


if __name__ == '__main__':
    unittest.main()
