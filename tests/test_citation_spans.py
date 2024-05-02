# -*- coding: UTF-8 -*-
"""
Created on 24.11.22

:author:     Martin Dočekal
"""
import unittest
from unittest import TestCase

from oapapers.bib_entry import BibEntry
from oapapers.citation_spans import expand_single_citation, expand_citations, repair_single_merged_citation_span, \
    repair_merged_citation_spans, enhance_citations_spans, create_vancouver_mapping, merge_citations, \
    add_missing_harvard_style_citations, identify_citation_style, CitationStyle, match_unk_citation_spans_with_bib, \
    repair_span_boundaries_in_hierarchy, group_citations, identify_citation_style_of_doc
from oapapers.document import Document
from oapapers.hierarchy import TextContent, RefSpan, Hierarchy


class TestExpandSingleCitation(TestCase):
    def setUp(self) -> None:
        self.tc = TextContent(
            "These are highly regarded authors [7]. As was proven in [1-4] and disproven in [5] - [8].",
            citations=[
                RefSpan(6, 34, 37),
                RefSpan(4, 79, 82),
                RefSpan(5, 85, 88),
            ], references=[])

    def test_dash_inside(self):
        expand_single_citation(self.tc, 56, 61, {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6})
        self.assertEqual(
            TextContent(
                "These are highly regarded authors [7]. As was proven in [1][2][3][4] and disproven in [5] - [8].",
                citations=[
                    RefSpan(6, 34, 37),
                    RefSpan(0, 56, 59),
                    RefSpan(1, 59, 62),
                    RefSpan(2, 62, 65),
                    RefSpan(3, 65, 68),
                    RefSpan(4, 86, 89),
                    RefSpan(5, 92, 95),
                ], references=[]),
            self.tc
        )

    def test_dash_outside(self):
        expand_single_citation(self.tc, 79, 88, {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6})
        self.assertEqual(
            TextContent(
                "These are highly regarded authors [7]. As was proven in [1-4] and disproven in [5][6][7][8].",
                citations=[
                    RefSpan(6, 34, 37),
                    RefSpan(4, 79, 82),
                    RefSpan(5, 82, 85),
                    RefSpan(6, 85, 88),
                    RefSpan(None, 88, 91),
                ], references=[]),
            self.tc
        )


class TestMergeCitations(TestCase):
    def test_nothing(self):
        tc = TextContent(
            "This is text without citations. Just some brackets (5-1).",
            citations=[], references=[RefSpan(None, 0, 4)])
        merge_citations(tc)
        self.assertEqual("This is text without citations. Just some brackets (5-1).", tc.text)
        self.assertEqual([], tc.citations)
        self.assertEqual([RefSpan(None, 0, 4)], tc.references)

    def test_non_expanded(self):
        tc = TextContent(
            "These are highly regarded authors [7]. As was proven in [8][2][7][4] and disproven in [8][3][7][6]. Also see [9] and [10].",
            citations=[
                RefSpan(6, 34, 37),
                RefSpan(7, 56, 59),
                RefSpan(1, 59, 62),
                RefSpan(6, 62, 65),
                RefSpan(3, 65, 68),
                RefSpan(7, 86, 89),
                RefSpan(2, 89, 92),
                RefSpan(6, 92, 95),
                RefSpan(5, 95, 98),
                RefSpan(8, 109, 112),
                RefSpan(9, 117, 121),

            ], references=[
                RefSpan(None, 73, 82),  # disproven
                RefSpan(None, 121, 122),  # .

            ])
        merge_citations(tc)
        self.assertEqual(
            "These are highly regarded authors [7]. As was proven in [8][2][7][4] and disproven in [8][3][7][6]. Also see [9] and [10].",
            tc.text)
        self.assertEqual(
            [
                RefSpan(6, 34, 37),
                RefSpan(7, 56, 59),
                RefSpan(1, 59, 62),
                RefSpan(6, 62, 65),
                RefSpan(3, 65, 68),
                RefSpan(7, 86, 89),
                RefSpan(2, 89, 92),
                RefSpan(6, 92, 95),
                RefSpan(5, 95, 98),
                RefSpan(8, 109, 112),
                RefSpan(9, 117, 121),

            ], tc.citations)

        self.assertEqual(
            [
                RefSpan(None, 73, 82),  # disproven
                RefSpan(None, 121, 122),  # .
            ], tc.references)

    def test_mixed(self) -> None:
        tc = TextContent(
            "These are highly regarded authors [7]. As was proven in [1][2][3][4] and disproven in [5][6][7][8]. Also see [9] and [10].",
            citations=[
                RefSpan(6, 34, 37),
                RefSpan(0, 56, 59),
                RefSpan(1, 59, 62),
                RefSpan(2, 62, 65),
                RefSpan(3, 65, 68),
                RefSpan(4, 86, 89),
                RefSpan(5, 89, 92),
                RefSpan(6, 92, 95),
                RefSpan(7, 95, 98),
                RefSpan(8, 109, 112),
                RefSpan(9, 117, 121),

            ], references=[
                RefSpan(None, 73, 82),  # disproven
                RefSpan(None, 121, 122),  # .
            ])

        merge_citations(tc)
        self.assertEqual(
            "These are highly regarded authors [7]. As was proven in [1-4] and disproven in [5-8]. Also see [9] and [10].",
            tc.text
        )
        self.assertEqual(
            [
                RefSpan(6, 34, 37),
                RefSpan(None, 56, 61),
                RefSpan(None, 79, 84),
                RefSpan(8, 95, 98),
                RefSpan(9, 103, 107)
            ],
            tc.citations
        )
        self.assertEqual(
            [
                RefSpan(None, 66, 75),  # disproven
                RefSpan(None, 107, 108),  # .
            ], tc.references)

    def test_only_expanded(self):
        tc = TextContent(
            "As was proven in [1][2][3][4] and disproven in [5][6][7][8].",
            citations=[
                RefSpan(0, 17, 20),
                RefSpan(1, 20, 23),
                RefSpan(2, 23, 26),
                RefSpan(3, 26, 29),
                RefSpan(4, 47, 50),
                RefSpan(5, 50, 53),
                RefSpan(6, 53, 56),
                RefSpan(7, 56, 59),
            ], references=[
                RefSpan(None, 7, 13),  # proven
                RefSpan(None, 34, 43),  # disproven
            ])

        merge_citations(tc)
        self.assertEqual(
            "As was proven in [1-4] and disproven in [5-8].",
            tc.text
        )
        self.assertEqual(
            [
                RefSpan(None, 17, 22),
                RefSpan(None, 40, 45),
            ],
            tc.citations)

        self.assertEqual(
            [
                RefSpan(None, 7, 13),  # proven
                RefSpan(None, 27, 36),  # disproven
            ], tc.references)


class TestCitationSpans(TestCase):

    def test_expand_citations(self):
        hier = Hierarchy(None, TextContent(
            "These are highly regarded authors [7]. As was proven in [1-4] and disproven in [5] - [8].",
            citations=[
                RefSpan(6, 34, 37),
                RefSpan(None, 56, 61),
                RefSpan(4, 79, 82),
                RefSpan(5, 85, 88),
            ], references=[
                RefSpan(None, 46, 52),  # proven
                RefSpan(None, 66, 75),  # disproven
            ]))

        expand_citations(hier, {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6})

        self.assertEqual(
            TextContent(
                "These are highly regarded authors [7]. As was proven in [1][2][3][4] and disproven in [5][6][7][8].",
                citations=[
                    RefSpan(6, 34, 37),
                    RefSpan(0, 56, 59),
                    RefSpan(1, 59, 62),
                    RefSpan(2, 62, 65),
                    RefSpan(3, 65, 68),
                    RefSpan(4, 86, 89),
                    RefSpan(5, 89, 92),
                    RefSpan(6, 92, 95),
                    RefSpan(None, 95, 98),
                ], references=[
                    RefSpan(None, 46, 52),  # proven
                    RefSpan(None, 73, 82),  # disproven
                ]),
            hier.content
        )

    def test_repair_single_merged_citation_span(self):
        tc = TextContent(
            "Lamma et al. [5,6,7] apply an extension of logic",
            citations=[
                RefSpan(None, 0, 20)
            ], references=[])
        repair_single_merged_citation_span(tc, vancouver_ids={5: 4, 7: 5})
        self.assertEqual(TextContent(
            "Lamma et al. [5,6,7] apply an extension of logic",
            citations=[
                RefSpan(4, 13, 16),
                RefSpan(None, 16, 18),
                RefSpan(5, 18, 20),
            ], references=[]), tc)

    def test_repair_merged_citation_spans(self):
        hier = Hierarchy(None, TextContent(
            "These are highly regarded authors [7]. Lamma et al. [5,6,7] apply an extension of logic. "
            "Others et. al. [4] also mention something.",
            citations=[
                RefSpan(6, 34, 37),
                RefSpan(None, 39, 59),
                RefSpan(None, 89, 107),
            ], references=[]))

        repair_merged_citation_spans(hier, {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6})

        self.assertEqual(
            TextContent(
                "These are highly regarded authors [7]. Lamma et al. [5,6,7] apply an extension of logic. "
                "Others et. al. [4] also mention something.",
                citations=[
                    RefSpan(6, 34, 37),
                    RefSpan(4, 52, 55),
                    RefSpan(5, 55, 57),
                    RefSpan(6, 57, 59),
                    RefSpan(3, 104, 107),
                ], references=[]),
            hier.content
        )

    def test_enhance_citations_spans(self):
        hier = Hierarchy(None, TextContent(
            "These are highly regarded authors [7]. Lamma et al. [5-7] apply an extension of logic. "
            "Others et. al. [4] also mention something [3].",
            citations=[
                RefSpan(6, 34, 37),
                RefSpan(None, 52, 57),
                RefSpan(None, 87, 105),
                RefSpan(2, 129, 132),
            ], references=[
                RefSpan(None, 17, 25),  # regarded
                RefSpan(None, 67, 76),  # extension
            ]))

        enhance_citations_spans(hier)

        self.assertEqual(
            TextContent(
                "These are highly regarded authors [7]. Lamma et al. [5][6][7] apply an extension of logic. "
                "Others et. al. [4] also mention something [3].",
                citations=[
                    RefSpan(6, 34, 37),
                    RefSpan(4, 52, 55),
                    RefSpan(5, 55, 58),
                    RefSpan(6, 58, 61),
                    RefSpan(3, 106, 109),
                    RefSpan(2, 133, 136),
                ], references=[
                    RefSpan(None, 17, 25),  # regarded
                    RefSpan(None, 71, 80),  # extension
                ]),
            hier.content
        )


class TestCreateVancouverMapping(TestCase):
    def setUp(self) -> None:
        self.hier = Hierarchy(None, [
            Hierarchy(None, TextContent(
                "These are highly regarded authors [7]. Lamma et al. [5,6,7] apply an extension of logic. "
                "Others et. al. [3] also mention something.",
                citations=[
                    RefSpan(7, 34, 37),
                    RefSpan(None, 39, 59),
                    RefSpan(2, 89, 107),
                ], references=[])),
            Hierarchy(None, TextContent(
                "Lamma et al. [1,5] apply an extension of logic.",
                citations=[
                    RefSpan(0, 13, 16),
                    RefSpan(5, 16, 18)
                ], references=[]), )
        ])

    def test_create_vancouver_mapping_no_interpolation(self):
        self.assertEqual({1: 0, 3: 2, 5: 5, 7: 7}, create_vancouver_mapping(self.hier, interpolate=False))

    def test_create_vancouver_mapping_interpolation(self):
        self.assertEqual({1: 0, 2: 1, 3: 2, 5: 5, 6: 6, 7: 7}, create_vancouver_mapping(self.hier, interpolate=True))


class TestAddMissingHarvardStyleCitations(TestCase):

    def setUp(self) -> None:
        # document with few already identified harvard style citations,
        # missing citations and citation like spans that do not refer to any bib entry
        self.doc = Document(
            id=1,
            s2orc_id=None,
            mag_id=None,
            doi=None,
            title="Test document",
            authors=[],
            year=None,
            fields_of_study=[],
            citations=[],
            hierarchy=Hierarchy(None, [
                Hierarchy(None, TextContent(
                    "As was presented (Earlier in 2013) word translation  (Mikolov et al., 2014) is a fascinating task. ",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "Keyphrase extraction (Docekal and Smrz, 2022) and another task is document summarization (Rush et al., 2015).",
                    citations=[
                        RefSpan(0, 89, 108),
                    ], references=[])),
                Hierarchy(None, TextContent(
                    "These highly regarded authors Lamma et al.  (2016) apply an extension of logic on another authors work (Maraka, 2020).",
                    citations=[], references=[])),
                ]),
            bibliography=[
                BibEntry(0, "Word translation", 2014, ("Mikolov Tomas",)),
                BibEntry(1, "Document summarization", 2015, ("Rush Michael",)),
                BibEntry(2, "Keyphrase extraction", 2022, ("Docekal Martin", "Smrz Pavel")),
                BibEntry(3, "Publication about everything and more", 2016, ("Lamma Alvares Pedro Luka De La Muerte", "Lion", "Opossum")),
                BibEntry(4, "Nothing and something", 2020, ("Uranium G",)),
            ],
            non_plaintext_content=[],
            uncategorized_fields={}
        )

    def test_add_missing_harvard_style_citations(self):
        add_missing_harvard_style_citations(self.doc)

        self.assertSequenceEqual(
            [RefSpan(0, 53, 75)],
            self.doc.hierarchy.content[0].content.citations
        )
        self.assertSequenceEqual(
            [RefSpan(2, 21, 45), RefSpan(0, 89, 108)],
            self.doc.hierarchy.content[1].content.citations
        )
        self.assertSequenceEqual(
            [RefSpan(3, 30, 50)],
            self.doc.hierarchy.content[2].content.citations
        )


class TestIdentifyCitationStyle(unittest.TestCase):

    def test_citation_style_couldnt_decide(self):
        self.assertEqual(CitationStyle.UNKNOWN, identify_citation_style(""))
        self.assertEqual(CitationStyle.UNKNOWN, identify_citation_style("(Morreall, 2016) [1] (10)"))

    def test_citation_style_harvard(self):
        txt = "The dominant theory of humor is the Incongruity Theory (Morreall, 2016). It says that we are finding humor in perceiving something unexpected (incongruous) that violates expectations that were set up by the joke. There are samples, in the provided dataset, that uses the incongruity to create humor. Moreover, according to Hossain et al. (2020a), we can see a positive influence of incongruity on systems results for the dataset."
        self.assertEqual(CitationStyle.HARVARD, identify_citation_style(txt))

        txt = "Official results (2) were achieved with a Convolutional Neural Networks (CNNs) (LeCun et al., 1999; Fukushima and Miyake, 1982), but we also tested numerous other approaches such as SVM (Cortes and Vapnik, 1995) and pre-trained (7) transformer model (Vaswani et al., 2017)."
        self.assertEqual(CitationStyle.HARVARD, identify_citation_style(txt))

    def test_citation_style_vancouver_parenthesis(self):
        txt = "The dominant theory of humor is the Incongruity Theory (1). It says that we are finding humor in perceiving something unexpected (incongruous) that violates expectations that were set up by the joke. There are samples, in the provided dataset, that uses the incongruity to create humor. Moreover, according to Hossain et al. (2020a), we can see a positive influence of incongruity on systems results for the dataset."
        self.assertEqual(CitationStyle.VANCOUVER_PARENTHESIS, identify_citation_style(txt))

        txt = "Official results (2) were achieved with a Convolutional Neural Networks (CNNs) (4, 5), but we also tested numerous other approaches such as SVM (Cortes and Vapnik, 1995) and pre-trained (7) transformer model (Vaswani et al., 2017)."
        self.assertEqual(CitationStyle.VANCOUVER_PARENTHESIS, identify_citation_style(txt))

    def test_citation_style_vancouver_square_brackets(self):
        txt = "The dominant theory of humor is the Incongruity Theory [1]. It says that we are finding humor in perceiving something unexpected (incongruous) that violates expectations that were set up by the joke. There are samples, in the provided dataset, that uses the incongruity to create humor. Moreover, according to Hossain et al. (2020a), we can see a positive influence of incongruity on systems results for the dataset."
        self.assertEqual(CitationStyle.VANCOUVER_SQUARE_BRACKETS, identify_citation_style(txt))

        txt = "Official results [2] were achieved with a Convolutional Neural Networks (CNNs) [4, 5], but we also tested numerous other approaches such as SVM (Cortes and Vapnik, 1995) and pre-trained [7] transformer model (Vaswani et al., 2017)."
        self.assertEqual(CitationStyle.VANCOUVER_SQUARE_BRACKETS, identify_citation_style(txt))


class TestIdentifyCitationStyleOfDoc(unittest.TestCase):

    def test_citation_style_couldnt_decide(self):
        doc = Document(
            id=1,
            s2orc_id=None,
            mag_id=None,
            doi=None,
            title="Test document",
            authors=[], year=None,
            fields_of_study=[],
            citations=[],
            hierarchy=Hierarchy(None, [
                Hierarchy(None, TextContent(
                    "As was presented (Earlier in 2013) word translation  (Mikolov et al., 2014) is a fascinating task. ",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "Keyphrase extraction [1] and another task is document summarization [2].",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "These highly regarded authors (13) apply an extension of logic on another authors work (50).",
                    citations=[], references=[])),
                ]),
            bibliography=[],
            non_plaintext_content=[],
            uncategorized_fields={}
        )

        self.assertEqual(CitationStyle.UNKNOWN, identify_citation_style_of_doc(doc))

    def test_citation_style_harvard(self):
        doc = Document(
            id=1,
            s2orc_id=None,
            mag_id=None,
            doi=None,
            title="Test document",
            authors=[], year=None,
            fields_of_study=[],
            citations=[],
            hierarchy=Hierarchy(None, [
                Hierarchy(None, TextContent(
                    "As was presented (Earlier in 2013) word translation  (Mikolov et al., 2014) is a fascinating task. ",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "The dominant theory of humor is the Incongruity Theory (Morreall, 2016).",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "These highly regarded authors (Franc, 1994) apply an extension of logic on another authors work (50).",
                    citations=[], references=[])),
            ]),
            bibliography=[],
            non_plaintext_content=[],
            uncategorized_fields={}
        )

        self.assertEqual(CitationStyle.HARVARD, identify_citation_style_of_doc(doc))

    def test_citation_style_vancouver_parenthesis(self):
        doc = Document(
            id=1,
            s2orc_id=None,
            mag_id=None,
            doi=None,
            title="Test document",
            authors=[], year=None,
            fields_of_study=[],
            citations=[],
            hierarchy=Hierarchy(None, [
                Hierarchy(None, TextContent(
                    "As was presented (1) word translation  (2) is a fascinating task. ",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "The dominant theory of humor is the Incongruity Theory (Morreall, 2016).",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "These highly regarded authors [13] apply an extension of logic on another authors work (50).",
                    citations=[], references=[])),
            ]),
            bibliography=[],
            non_plaintext_content=[],
            uncategorized_fields={}
        )

        self.assertEqual(CitationStyle.VANCOUVER_PARENTHESIS, identify_citation_style_of_doc(doc))

    def test_citation_style_vancouver_square_brackets(self):
        doc = Document(
            id=1,
            s2orc_id=None,
            mag_id=None,
            doi=None,
            title="Test document",
            authors=[], year=None,
            fields_of_study=[],
            citations=[],
            hierarchy=Hierarchy(None, [
                Hierarchy(None, TextContent(
                    "As was presented [1] word translation  [2] is a fascinating task. ",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "The dominant theory of humor is the Incongruity Theory (Morreall, 2016).",
                    citations=[], references=[])),
                Hierarchy(None, TextContent(
                    "These highly regarded authors (13) apply an extension of logic on another authors work [50].",
                    citations=[], references=[])),
            ]),
            bibliography=[],
            non_plaintext_content=[],
            uncategorized_fields={}
        )

        self.assertEqual(CitationStyle.VANCOUVER_SQUARE_BRACKETS, identify_citation_style_of_doc(doc))


class TestMatchUnkCitationSpansWithBib(unittest.TestCase):
    def setUp(self) -> None:
        # document with few already identified harvard style citations,
        # missing citations and citation like spans that do not refer to any bib entry
        self.doc = Document(
            id=1,
            s2orc_id=None,
            mag_id=None,
            doi=None,
            title="Test document",
            authors=[],
            year=None,
            fields_of_study=[],
            citations=[],
            hierarchy=Hierarchy(None, [
                Hierarchy(None, TextContent(
                    "As was presented (Earlier in 2013) word translation  (Mikolov et al., 2014) is a fascinating task. ",
                    citations=[
                        RefSpan(4, 54, 74)  # badly identified citation should not be changed
                    ], references=[])),
                Hierarchy(None, TextContent(
                    "Keyphrase extraction (Docekal and Smrz, 2022) and another task is document summarization (Rush et al., 2015).",
                    citations=[
                        RefSpan(None, 22, 44), RefSpan(None, 89, 108)
                    ], references=[])),
                Hierarchy(None, TextContent(
                    "These highly regarded authors (Lamma et al., 2016) apply an extension of logic on another authors work (Maraka, 2020).",
                    citations=[
                        RefSpan(None, 103, 117)
                    ], references=[])),
                ]),
            bibliography=[
                BibEntry(0, "Word translation", 2014, ("Mikolov Tomas",)),
                BibEntry(1, "Document summarization", 2015, ("Rush Michael",)),
                BibEntry(2, "Keyphrase extraction", 2022, ("Docekal Martin", "Smrz Pavel")),
                BibEntry(3, "Publication about everything and more", 2016, ("Lamma Alvares Pedro Luka De La Muerte", "Lion", "Opossum")),
                BibEntry(4, "Nothing and something", 2020, ("Uranium G",)),
            ],
            non_plaintext_content=[],
            uncategorized_fields={}
        )

    def test_match_unk_citation_spans_with_bib_badly_identified_shouldnt_be_corrected(self):
        match_unk_citation_spans_with_bib(self.doc)
        self.assertSequenceEqual(
            [RefSpan(4, 54, 74)],
            self.doc.hierarchy.content[0].content.citations
        )

    def test_match_unk_citation_spans_with_bib(self):
        match_unk_citation_spans_with_bib(self.doc)
        self.assertSequenceEqual(
            [RefSpan(2, 22, 44), RefSpan(1, 89, 108)],
            self.doc.hierarchy.content[1].content.citations
        )

    def test_match_unk_citation_spans_with_bib_unknown(self):
        match_unk_citation_spans_with_bib(self.doc)
        self.assertSequenceEqual(
            [RefSpan(None, 103, 117)],
            self.doc.hierarchy.content[2].content.citations
        )


class TestRepairSpanBoundariesInHierarchy(unittest.TestCase):

    def test_no_spans(self):
        tc = TextContent("Some text", [], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.HARVARD)
        self.assertEqual(tc, hier.content)

    def test_no_spans_harvard_citation_that_is_ok(self):
        tc = TextContent("Some text (Author, 2023)", [RefSpan(3, 10, 24)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.HARVARD)
        self.assertEqual(tc, hier.content)

    def test_no_spans_harvard_citation_that_is_not_ok(self):
        tc = TextContent("Some text (Author, 2023)", [RefSpan(3, 11, 23)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.HARVARD)
        self.assertEqual(TextContent("Some text (Author, 2023)", [RefSpan(3, 10, 24)], []), hier.content)

    def test_no_spans_vancouver_parenthesis_citation_that_is_ok(self):
        tc = TextContent("Some text (1)", [RefSpan(3, 10, 13)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.VANCOUVER_PARENTHESIS)
        self.assertEqual(tc, hier.content)

    def test_no_spans_vancouver_parenthesis_citation_that_is_not_ok(self):
        tc = TextContent("Some text (1)", [RefSpan(3, 11, 12)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.VANCOUVER_PARENTHESIS)
        self.assertEqual(TextContent("Some text (1)", [RefSpan(3, 10, 13)], []), hier.content)

    def test_no_spans_vancouver_brackets_citation_that_is_ok(self):
        tc = TextContent("Some text [1]", [RefSpan(3, 10, 13)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.VANCOUVER_SQUARE_BRACKETS)
        self.assertEqual(tc, hier.content)

    def test_no_spans_vancouver_brackets_citation_that_is_not_ok(self):
        tc = TextContent("Some text [1]", [RefSpan(3, 11, 12)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.VANCOUVER_SQUARE_BRACKETS)
        self.assertEqual(TextContent("Some text [1]", [RefSpan(3, 10, 13)], []), hier.content)

    def test_multiple_harvard_spans_ok(self):
        tc = TextContent("Some text (Author, 2023) also see another work (Other, 1994)", [RefSpan(3, 10, 24), RefSpan(3, 47, 60)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.HARVARD)
        self.assertEqual(tc, hier.content)

    def test_multiple_harvard_spans_not_ok(self):
        tc = TextContent("Some text (Author, 2023) also see another work (Other, 1994)", [RefSpan(3, 11, 23), RefSpan(3, 55, 59)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.HARVARD)
        self.assertEqual(TextContent("Some text (Author, 2023) also see another work (Other, 1994)", [RefSpan(3, 10, 24), RefSpan(3, 47, 60)], []), hier.content)

    def test_match_but_no_span_harvard(self):
        tc = TextContent("Some text (Author, 2023) also see another work (Other, 1994)", [RefSpan(3, 48, 59)], [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.HARVARD)
        self.assertEqual(TextContent("Some text (Author, 2023) also see another work (Other, 1994)", [RefSpan(3, 47, 60)], []), hier.content)

    def test_repair_group_citation(self):
        tc = TextContent("The server's behaviour has a strong relation to high revenue; for example, if the server "
                         "becomes overloaded the response time can grow to unacceptable levels that can lead to the "
                         "user leaving the website. To maintain acceptable response time and minimise server "
                         "overload, clusters of multiple web servers have been developed [3,47].",
                         [
                             RefSpan(None, 326, 328),
                             RefSpan(None, 328, 331),
                         ],
                         [])
        hier = Hierarchy(None, tc)
        repair_span_boundaries_in_hierarchy(hier, CitationStyle.VANCOUVER_SQUARE_BRACKETS)
        self.assertEqual(
            TextContent("The server's behaviour has a strong relation to high revenue; for example, if the server "
                        "becomes overloaded the response time can grow to unacceptable levels that can lead to the "
                        "user leaving the website. To maintain acceptable response time and minimise server "
                        "overload, clusters of multiple web servers have been developed [3,47].",
                        [
                            RefSpan(None, 325, 328),
                            RefSpan(None, 328, 331),
                        ],
                        []),
            hier.content
        )


class TestGroupCitations(unittest.TestCase):

    def test_no_citations(self):
        tc = TextContent("Some text", [], [])
        self.assertSequenceEqual([], group_citations(tc))

    def test_one_citation(self):
        tc = TextContent("Some text (Author, 2023)", [RefSpan(3, 10, 24)], [])
        self.assertSequenceEqual([[RefSpan(3, 10, 24)]], group_citations(tc))

    def test_two_citations(self):
        tc = TextContent("Some text (Author, 2023) also see another work (Other, 1994)", [RefSpan(3, 10, 24), RefSpan(3, 47, 60)], [])
        self.assertSequenceEqual([[RefSpan(3, 10, 24)], [RefSpan(3, 47, 60)]], group_citations(tc))

    def test_one_citation_group(self):
        tc = TextContent("Some text (Author, 2023; Buthor, 2022) also see another work (Author, 1994)",
                         [RefSpan(3, 10, 24), RefSpan(4, 25, 38), RefSpan(5, 61, 75)], [])
        self.assertSequenceEqual([[RefSpan(3, 10, 24), RefSpan(4, 25, 38)], [RefSpan(5, 61, 75)]], group_citations(tc))
