# -*- coding: UTF-8 -*-
""""
Created on 04.02.22

:author:     Martin Dočekal
"""
import json
import os
import unittest
from dataclasses import asdict
from unittest import TestCase

from oapapers.document import Document, OARelatedWorkDocument
from oapapers.citation_spans import CitationStyle
from oapapers.bib_entry import BibEntry
from oapapers.hierarchy import Hierarchy, TextContent, PositionFromEnd, RefSpan
from oapapers.papers_list import PapersList, PapersListRecord

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
FIXTURES_DIR = os.path.join(SCRIPT_DIR, "fixtures")


class TestDocument(TestCase):
    def setUp(self) -> None:
        self.d = Document(1,
                          321,
                          123,
                          "10.32473/flairs.v35i.130737",
                          "title",
                          ["author1", "author2"],
                          2021,
                          ["Mathematics"],
                          [1, 2, 3],
                          Hierarchy(
                              "title",
                              [
                                  Hierarchy(
                                      "headline1", TextContent("text1", [RefSpan(0, 10, 100)], [])
                                  ),
                                  Hierarchy(
                                      "headline2",
                                      TextContent("text2", [RefSpan(None, 20, 200)], [RefSpan(0, 20, 200)])
                                  ),
                              ]
                          ),
                          [BibEntry(2, "BIB TITLE", 2020, ("Author 1",))],
                          [("figure", "desc 1"), ("table", "desc 2")],
                          uncategorized_fields={})

    def test_str(self):
        self.assertEqual(
            '{"id":1,"s2orc_id":321,"mag_id":123,"doi":"10.32473/flairs.v35i.130737","title":"title","authors":["author1","author2"],'
            '"year":2021,"fields_of_study":["Mathematics"],"citations":[1,2,3],'
            '"hierarchy":{'
            '"headline":"title","content":['
            '{"headline":"headline1","content":{"text":"text1","citations":[{"index":0,"start":10,"end":100}],"references":[]}},'
            '{"headline":"headline2","content":{"text":"text2","citations":[{"index":null,"start":20,"end":200}'
            '],'
            '"references":[{"index":0,"start":20,"end":200}]}}]},'
            '"bibliography":[{"id":2,"title":"BIB TITLE","year":2020,"authors":["Author 1"]}],'
            '"non_plaintext_content":[["figure","desc 1"],["table","desc 2"]]}',
            str(self.d))

    def test_filter_citations(self):
        self.d.filter_citations({1, 3})
        self.assertListEqual([1, 3], self.d.citations)

        self.assertEqual(Hierarchy(
            "title",
            [
                Hierarchy(
                    "headline1", TextContent("text1", [RefSpan(0, 10, 100)], [])
                ),
                Hierarchy(
                    "headline2",
                    TextContent("text2", [RefSpan(None, 20, 200)], [RefSpan(0, 20, 200)])
                ),
            ]
        ), self.d.hierarchy)
        self.assertSequenceEqual([
            BibEntry(None, "BIB TITLE", 2020, ("Author 1",))
        ], self.d.bibliography)

    def test_text_content(self):
        self.assertListEqual(
            [
                TextContent("text1", [RefSpan(0, 10, 100)], []),
                TextContent("text2", [RefSpan(None, 20, 200)], [RefSpan(0, 20, 200)])
            ], list(self.d.text_content()))

        self.d.hierarchy = Hierarchy(
            "title",
            [
                Hierarchy(
                    "headline1",
                    [
                        Hierarchy(None,
                                  [
                                      Hierarchy(None, TextContent("text0", [], [])),
                                      Hierarchy(None, TextContent("text1", [], []))
                                  ]),
                        Hierarchy(None, TextContent("text2", [RefSpan(0, 10, 100)], []))
                    ]
                ),
                Hierarchy(
                    "headline2",
                    TextContent("text3", [RefSpan(None, 20, 200)], [RefSpan(0, 20, 200)])
                ),
            ]
        )

        self.assertListEqual(
            [
                TextContent("text0", [], []),
                TextContent("text1", [], []),
                TextContent("text2", [RefSpan(0, 10, 100)], []),
                TextContent("text3", [RefSpan(None, 20, 200)], [RefSpan(0, 20, 200)])
            ], list(self.d.text_content()))

    def test_text_content_parent_condition(self):
        self.assertListEqual(
            [
                TextContent("text1", [RefSpan(0, 10, 100)], []),
            ], list(self.d.text_content(lambda x: x.headline == "headline1")))

    def test_translate_ids(self):
        translation_mapping = {1: 10, 2: 20, 3: 30}
        self.d.translate_ids(translation_mapping)
        self.assertEqual(10, self.d.id)
        self.assertListEqual([10, 20, 30], self.d.citations)
        self.assertListEqual([0, None], [c.index for t_c in self.d.text_content() for c in t_c.citations])

    def test_translate_ids_two_distinct_to_one_same(self):
        translation_mapping = {1: 10, 2: 10, 3: 30}
        self.d.translate_ids(translation_mapping)
        self.assertEqual(10, self.d.id)
        self.assertListEqual([10, 30], self.d.citations)
        self.assertListEqual([0, None], [c.index for t_c in self.d.text_content() for c in t_c.citations])

    def test_deduplicate_authors(self):
        self.d.deduplicate_authors()

        self.assertSequenceEqual(["author1", "author2"], self.d.authors)

        self.d.authors = ["author 1", "A Zirizzotti", "author 2", "A. Zirizzotti"]
        self.d.deduplicate_authors()
        self.assertSequenceEqual(["author 1", "A Zirizzotti", "author 2"], self.d.authors)


class TestDocumentSpanFixSpanBoundaries(TestCase):
    def setUp(self) -> None:
        self.d = Document(1,
                          321,
                          123,
                          "10.32473/flairs.v35i.130737",
                          "title",
                          ["author1", "author2"],
                          2021,
                          ["Mathematics"],
                          [10, 20, 30],
                          Hierarchy(
                              "title",
                              [
                                  Hierarchy(
                                      "headline1",
                                      TextContent("I'm text with citation [123] and reference to Figure 1. .",
                                                  [RefSpan(1, 23, 28)],
                                                  [RefSpan(0, 46, 55)])
                                  ),
                                  Hierarchy(
                                      "headline2",
                                      TextContent("I'm text with citation (231 ) and reference to (Table 2 ).",
                                                  [RefSpan(2, 23, 27)],
                                                  [RefSpan(1, 47, 55)])
                                  ),
                                  Hierarchy(
                                      "headline3",
                                      TextContent("I'm text with citation [ 321][ 123 ] and reference to ( Table 2).",
                                                  [RefSpan(3, 25, 29), RefSpan(1, 31, 34)],
                                                  [RefSpan(1, 56, 64)])
                                  ),
                              ]
                          ),
                          [
                              BibEntry(None, "Title 1", None, ("Author 1",)),
                              BibEntry(10, "Title 2", None, ("Author 2",)),
                              BibEntry(20, "Title 3", None, ("Author 3",)),
                              BibEntry(30, "Title 4", None, ("Author 4",)),
                          ],
                          [("figure", "desc 1"), ("table", "desc 2")],
                          uncategorized_fields={})

    def test_fix_span_boundaries(self):
        self.d.fix_span_boundaries()

        gt = [
            ([RefSpan(1, 23, 28)], [RefSpan(0, 46, 55)]),
            ([RefSpan(2, 23, 29)], [RefSpan(1, 47, 57)]),
            ([RefSpan(3, 23, 29), RefSpan(1, 29, 36)], [RefSpan(1, 54, 64)])
        ]

        for i, t_c in enumerate(self.d.text_content()):
            self.assertListEqual(gt[i][0], t_c.citations, msg=f"Troubles with citations in text content: {i}")
            self.assertListEqual(gt[i][1], t_c.references, msg=f"Troubles with references in text content: {i}")


class TestDocumentSpanNormalization(TestCase):
    def setUp(self) -> None:
        self.d = Document(1,
                          321,
                          123,
                          "10.32473/flairs.v35i.130737",
                          "title",
                          ["author1", "author2"],
                          2021,
                          ["Mathematics"],
                          [10, 20, 30],
                          Hierarchy(
                              "title",
                              [
                                  Hierarchy(
                                      "headline1",
                                      TextContent("I'm text with citation [123] and reference to Figure 1. .",
                                                  [RefSpan(1, 23, 28)],
                                                  [RefSpan(0, 46, 55)])
                                  ),
                                  Hierarchy(
                                      "headline2",
                                      TextContent("I'm text with citation [231 ] and reference to (Table 2 ).",
                                                  [RefSpan(2, 23, 29)],
                                                  [RefSpan(1, 47, 57)])
                                  ),
                                  Hierarchy(
                                      "headline3",
                                      TextContent("I'm text with citation [ 321][ 123 ] and reference to ( Table 2).",
                                                  [RefSpan(3, 23, 29), RefSpan(1, 29, 36)],
                                                  [RefSpan(1, 54, 64)])
                                  ),
                              ]
                          ),
                          [
                              BibEntry(None, "Title 1", None, ("Author 1",)),
                              BibEntry(10, "Title 2", None, ("Author 2",)),
                              BibEntry(20, "Title 3", None, ("Author 3",)),
                              BibEntry(30, "Title 4", None, ("Author 4",)),
                          ],
                          [("figure", "desc 1"), ("table", "desc 2")],
                          uncategorized_fields={})

    def test_normalize_single_span_citations(self):
        s = RefSpan(1, 10, 15)
        self.assertEqual(
            "This cite [CITE:1]",
            self.d.normalize_single_span("This cite [123]  ", s, False)
        )
        self.assertEqual(RefSpan(1, 10, 18), s)

        s = RefSpan(1, 10, 15)
        self.assertEqual(
            "This cite [CITE:1] is normalized.",
            self.d.normalize_single_span("This cite [123] is normalized.", s, False)
        )
        self.assertEqual(RefSpan(1, 10, 18), s)

        s = RefSpan(1, 10, 16)
        self.assertEqual(
            "This cite [CITE:1] is normalized.",
            self.d.normalize_single_span("This cite [123 ] is normalized.", s, False)
        )
        self.assertEqual(RefSpan(1, 10, 18), s)

        s = RefSpan(1, 10, 16)
        self.assertEqual(
            "This cite [CITE:1] is normalized.",
            self.d.normalize_single_span("This cite [ 123] is normalized.", s, False)
        )
        self.assertEqual(RefSpan(1, 10, 18), s)

        s = RefSpan(1, 10, 16)
        self.assertEqual(
            "This cite [CITE:1] is normalized.",
            self.d.normalize_single_span("This cite [ 123] is normalized.", s, False)
        )
        self.assertEqual(RefSpan(1, 10, 18), s)

        s = RefSpan(1, 10, 17)
        self.assertEqual(
            "This cite [CITE:1] is normalized.",
            self.d.normalize_single_span("This cite [ 123 ] is normalized.", s, False)
        )
        self.assertEqual(RefSpan(1, 10, 18), s)

        s = RefSpan(1, 9, 14)
        self.assertEqual(
            "This cite [CITE:1] is normalized.",
            self.d.normalize_single_span("This cite[123] is normalized.", s, False)
        )
        self.assertEqual(RefSpan(1, 10, 18), s)

        s = RefSpan(1, 10, 15)
        self.assertEqual(
            "This cite [CITE:1] is normalized.",
            self.d.normalize_single_span("This cite [123]is normalized.", s, False)
        )
        self.assertEqual(RefSpan(1, 10, 18), s)

        s = RefSpan(3, 9, 14)
        self.assertEqual(
            "This cite [CITE:3] is normalized.",
            self.d.normalize_single_span("This cite[123]is normalized.", s, False)
        )
        self.assertEqual(RefSpan(3, 10, 18), s)

        s = RefSpan(None, 9, 14)
        self.assertEqual(
            "This cite [CITE:UNK] is normalized.",
            self.d.normalize_single_span("This cite[123]is normalized.", s, False)
        )
        self.assertEqual(RefSpan(None, 10, 20), s)

    def test_normalize_single_span_references(self):
        s = RefSpan(0, 10, 15)
        self.assertEqual(
            "This cite [REF:0] is normalized.",
            self.d.normalize_single_span("This cite [123] is normalized.", s, True)
        )
        self.assertEqual(RefSpan(0, 10, 17), s)

        s = RefSpan(0, 10, 16)
        self.assertEqual(
            "This cite [REF:0] is normalized.",
            self.d.normalize_single_span("This cite [123 ] is normalized.", s, True)
        )
        self.assertEqual(RefSpan(0, 10, 17), s)

        s = RefSpan(0, 10, 16)
        self.assertEqual(
            "This cite [REF:0] is normalized.",
            self.d.normalize_single_span("This cite [ 123] is normalized.", s, True)
        )
        self.assertEqual(RefSpan(0, 10, 17), s)

        s = RefSpan(0, 10, 16)
        self.assertEqual(
            "This cite [REF:0] is normalized.",
            self.d.normalize_single_span("This cite [ 123] is normalized.", s, True)
        )
        self.assertEqual(RefSpan(0, 10, 17), s)

        s = RefSpan(0, 10, 17)
        self.assertEqual(
            "This cite [REF:0] is normalized.",
            self.d.normalize_single_span("This cite [ 123 ] is normalized.", s, True)
        )
        self.assertEqual(RefSpan(0, 10, 17), s)

        s = RefSpan(0, 9, 14)
        self.assertEqual(
            "This cite [REF:0] is normalized.",
            self.d.normalize_single_span("This cite[123] is normalized.", s, True)
        )
        self.assertEqual(RefSpan(0, 10, 17), s)

        s = RefSpan(0, 10, 15)
        self.assertEqual(
            "This cite [REF:0] is normalized.",
            self.d.normalize_single_span("This cite [123]is normalized.", s, True)
        )
        self.assertEqual(RefSpan(0, 10, 17), s)

        s = RefSpan(1, 9, 14)
        self.assertEqual(
            "This cite [REF:1] is normalized.",
            self.d.normalize_single_span("This cite[123]is normalized.", s, True)
        )
        self.assertEqual(RefSpan(1, 10, 17), s)

    def test_normalize_citation_spans(self):
        gt_contents = [
            TextContent("I'm text with citation [CITE:1] and reference to Figure 1. .",
                        [RefSpan(1, 23, 31)], [RefSpan(0, 49, 58)]),
            TextContent("I'm text with citation [CITE:2] and reference to (Table 2 ).",
                        [RefSpan(2, 23, 31)], [RefSpan(1, 49, 59)]),
            TextContent("I'm text with citation [CITE:3] [CITE:1] and reference to ( Table 2).",
                        [RefSpan(3, 23, 31), RefSpan(1, 32, 40)], [RefSpan(1, 58, 68)])
        ]

        self.d.normalize_citation_spans()

        self.assertListEqual(
            gt_contents,
            list(self.d.text_content())
        )

    def test_normalize_reference_spans(self):
        gt_contents = [
            TextContent("I'm text with citation [123] and reference to [REF:0] .",
                        [RefSpan(1, 23, 28)], [RefSpan(0, 46, 53)]),
            TextContent("I'm text with citation [231 ] and reference to [REF:1] .",
                        [RefSpan(2, 23, 29)], [RefSpan(1, 47, 54)]),
            TextContent("I'm text with citation [ 321][ 123 ] and reference to [REF:1] .",
                        [RefSpan(3, 23, 29), RefSpan(1, 29, 36)], [RefSpan(1, 54, 61)])
        ]

        self.d.normalize_reference_spans()

        self.assertListEqual(
            gt_contents,
            list(self.d.text_content())
        )

    def test_normalize_spans(self):
        gt_contents = [
            TextContent("I'm text with citation [CITE:1] and reference to [REF:0] .",
                        [RefSpan(1, 23, 31)], [RefSpan(0, 49, 56)]),
            TextContent("I'm text with citation [CITE:2] and reference to [REF:1] .",
                        [RefSpan(2, 23, 31)], [RefSpan(1, 49, 56)]),
            TextContent("I'm text with citation [CITE:3] [CITE:1] and reference to [REF:1] .",
                        [RefSpan(3, 23, 31), RefSpan(1, 32, 40)], [RefSpan(1, 58, 65)]),
            TextContent("[REF:1] I'm text with citation [CITE:3] [CITE:1] and reference to.",
                        [RefSpan(3, 31, 39), RefSpan(1, 40, 48)], [RefSpan(1, 0, 7)]),
            TextContent(
                "Lemma 12 [CITE:0] [CITE:1] [CITE:2] can be easily proved by recursing on (33a)-(33b) and (37) .",
                [RefSpan(0, 9, 17), RefSpan(1, 18, 26), RefSpan(2, 27, 35)], []),
            TextContent("The proofs of 12 [CITE:0] [CITE:1] [CITE:2] and Lemma 13 are obvious and omitted.",
                        [RefSpan(0, 17, 25), RefSpan(1, 26, 34), RefSpan(2, 35, 43)], []),
        ]

        self.d.hierarchy.content.append(Hierarchy(
            "headline 4",
            TextContent("( Table 2) I'm text with citation [ 321][ 123 ] and reference to.",
                        [RefSpan(3, 34, 40), RefSpan(1, 40, 47)],
                        [RefSpan(1, 0, 10)])
        ))
        self.d.hierarchy.content.append(Hierarchy(
            "headline 5",
            TextContent("Lemma 12(1)(2)(3) can be easily proved by recursing on (33a)-(33b) and (37) .",
                        [RefSpan(0, 8, 11), RefSpan(1, 11, 14), RefSpan(2, 14, 17)], [])
        ))
        self.d.hierarchy.content.append(Hierarchy(
            "headline 6",
            TextContent("The proofs of 12 (1)(2)(3) and Lemma 13 are obvious and omitted.",
                        [RefSpan(0, 17, 20), RefSpan(1, 20, 23), RefSpan(2, 23, 26)], [])
        ))

        self.d.normalize_spans()
        self.assertListEqual(
            gt_contents,
            list(self.d.text_content())
        )


class TestDocumentIdentifyCiteSpans(TestCase):
    def setUp(self) -> None:
        self.d = Document(1,
                          321,
                          123,
                          "10.32473/flairs.v35i.130737",
                          "title",
                          ["author1", "author2"],
                          2021,
                          ["Mathematics"],
                          [1, 2, 3],
                          Hierarchy(
                              "title",
                              [
                                  Hierarchy(
                                      "headline1",
                                      TextContent("I'm text with citation [123] and reference to Figure 1. .",
                                                  [
                                                      RefSpan(0, 23, 28)
                                                  ],
                                                  [RefSpan(0, 46, 55)])
                                  ),
                                  Hierarchy(
                                      "headline2",
                                      TextContent("I'm text with citation [231 ] and reference to (Table 2 ).",
                                                  [RefSpan(2, 23, 29)],
                                                  [RefSpan(1, 47, 57)])
                                  ),
                                  Hierarchy(
                                      "headline3",
                                      TextContent("I'm text with citation [ 321][ 123 ] and reference to ( Table 2).",
                                                  [
                                                      RefSpan(0, 23, 29),
                                                      RefSpan(1, 29, 36)
                                                  ],
                                                  [RefSpan(1, 54, 64)])
                                  ),
                              ]
                          ),
                          [
                              BibEntry(None, "Amazing paper", 1904, ("Gustav Gustavson",)),
                              BibEntry(None, "History of game development", None, ("Daniel Vávra",)),
                              BibEntry(1, "Title 3", None, ("Author 3",)),
                              BibEntry(2, "Title 5", None, ("Author 3",)),
                              BibEntry(3, "Title 7", None, ("Author 3",)),
                          ],
                          [("figure", "desc 1"), ("table", "desc 2")],
                          uncategorized_fields={})

        self.papers_list = PapersList([
            PapersListRecord("Amazing paper", 1904, ["Gustav Gustavson", "Charles Winchester III."]),
            PapersListRecord("Random Paper", 1904, ["Random Bloke"]),
            PapersListRecord("Interesting Paper", None, ["Interesting Author"]),
            PapersListRecord("My story", None, ["Gustav Gustavson"]),
            PapersListRecord("History of game development", None, ["Daniel Vávra"]),
        ])
        self.indices_2_id = [0, 1, 2, 3, 4]

    def test_identify_cite_spans(self):
        new_cites = self.d.identify_bibliography(self.papers_list, self.indices_2_id)
        self.assertEqual(2, new_cites)
        cite_spans = list(self.d.citation_spans())
        self.assertSequenceEqual([
            RefSpan(0, 23, 28), RefSpan(2, 23, 29), RefSpan(0, 23, 29), RefSpan(1, 29, 36)
        ], cite_spans)
        self.assertSequenceEqual([
            BibEntry(0, "Amazing paper", 1904, ("Gustav Gustavson",)),
            BibEntry(4, "History of game development", None, ("Daniel Vávra",)),
            BibEntry(1, "Title 3", None, ("Author 3",)),
            BibEntry(2, "Title 5", None, ("Author 3",)),
            BibEntry(3, "Title 7", None, ("Author 3",)),
        ], self.d.bibliography)
        self.assertSequenceEqual([0, 1, 2, 3, 4], self.d.citations)


class TestOARelatedWorkDocument(TestCase):
    def setUp(self) -> None:
        self.d = OARelatedWorkDocument(id=119311038,
                                       s2orc_id=321,
                                       mag_id=123,
                                       doi="10.32473/flairs.v35i.130737",
                                       title="On regular polytopes",
                                       authors=["Luis J. Boya", "Cristian Rivera"],
                                       year=2012,
                                       fields_of_study=[
                                           "Mathematics",
                                           "Physics",
                                           "Regular polytope",
                                           "Abelian group",
                                           "Dimension (graph theory)",
                                           "Polytope",
                                           "Platonic solid",
                                           "Lie algebra",
                                           "Combinatorics",
                                           "Triality",
                                           "Group (mathematics)"
                                       ],
                                       citations=[119311039],
                                       hierarchy=Hierarchy(
                                           headline="On regular polytopes",
                                           content=[
                                               Hierarchy(
                                                   headline="Abstract",
                                                   content=[
                                                       Hierarchy(
                                                           headline=None,
                                                           content=[
                                                               Hierarchy(
                                                                   headline=None,
                                                                   content=TextContent(
                                                                       text="Abstract of On regular polytopes section [CITE:0]",
                                                                       citations=[
                                                                           RefSpan(index=0, start=41, end=49)
                                                                       ],
                                                                       references=[]
                                                                   )
                                                               )
                                                           ]
                                                       ),
                                                       Hierarchy(
                                                           headline=None,
                                                           content=[
                                                               Hierarchy(
                                                                   headline=None,
                                                                   content=TextContent(
                                                                       text="Abstract of On regular polytopes section 2",
                                                                       citations=[],
                                                                       references=[]
                                                                   )
                                                               )
                                                           ]
                                                       )
                                                   ]
                                               ),
                                               Hierarchy(
                                                   headline="Introduction",
                                                   content=[
                                                       Hierarchy(
                                                           headline=None,
                                                           content=[
                                                               Hierarchy(
                                                                   headline=None,
                                                                   content=TextContent(
                                                                       text="Sentence in introduction",
                                                                       citations=[],
                                                                       references=[]
                                                                   )
                                                               )
                                                           ]
                                                       )
                                                   ]
                                               ),
                                           ]
                                       ),
                                       related_work=Hierarchy(
                                           headline="Related Work",
                                           content=[
                                               Hierarchy(
                                                   headline=None,
                                                   content=[
                                                       Hierarchy(
                                                           headline=None,
                                                           content=TextContent(
                                                               text="Introduction [CITE:0] of On regular polytopes section 1",
                                                               citations=[
                                                                   RefSpan(index=1, start=13, end=21)
                                                               ],
                                                               references=[]
                                                           )
                                                       )
                                                   ]
                                               )
                                           ]
                                       ),
                                       related_work_orig_path=(2,),
                                       bibliography=[
                                           BibEntry(119311039, "Fake citation", 1973, ("Mr. Fake",)),
                                           BibEntry(119311040, "Regular Polytopes", 1973, ("H S Coxeter",)),
                                       ],
                                       non_plaintext_content=[],
                                       uncategorized_fields={}
                                       )

    def test_str(self):
        self.assertEqual(json.loads(str(self.d)),
                         json.loads(
                             """{"id":119311038,"s2orc_id":321,"mag_id":123,"doi":"10.32473/flairs.v35i.130737","title":"On regular polytopes","authors":["Luis J. Boya","Cristian Rivera"],"year":2012,"fields_of_study":["Mathematics","Physics","Regular polytope","Abelian group","Dimension (graph theory)","Polytope","Platonic solid","Lie algebra","Combinatorics","Triality","Group (mathematics)"],"citations":[119311039],"hierarchy":{"headline":"On regular polytopes","content":[{"headline":"Abstract","content":[{"headline":null,"content":[{"headline":null,"content":{"text":"Abstract of On regular polytopes section [CITE:0]","citations":[{"index":0,"start":41,"end":49}],"references":[]}}]},{"headline":null,"content":[{"headline":null,"content":{"text":"Abstract of On regular polytopes section 2","citations":[],"references":[]}}]}]},{"headline":"Introduction","content":[{"headline":null,"content":[{"headline":null,"content":{"text":"Sentence in introduction","citations":[],"references":[]}}]}]}]},"related_work":{"headline":"Related Work","content":[{"headline":null,"content":[{"headline":null,"content":{"text":"Introduction [CITE:0] of On regular polytopes section 1","citations":[{"index":1,"start":13,"end":21}],"references":[]}}]}]},"related_work_orig_path":[2],"bibliography":[{"id":119311039,"title":"Fake citation","year":1973,"authors":["Mr. Fake"]},{"id":119311040,"title":"Regular Polytopes","year":1973,"authors":["H S Coxeter"]}],"non_plaintext_content":[]}"""))

    def test_abstract(self):
        self.assertEqual(asdict(self.d.abstract),
                         json.loads(
                             """{"headline": "Abstract", "content": [{"headline": null, "content": [{"headline": null, "content": {"text": "Abstract of On regular polytopes section [CITE:0]", "citations": [{"index": 0, "start": 41, "end": 49}], "references": []}}]}, {"headline": null, "content": [{"headline": null, "content": {"text": "Abstract of On regular polytopes section 2", "citations": [], "references": []}}]}]}"""))

    def test_related_work(self):
        self.assertEqual(asdict(self.d.related_work),
                         json.loads(
                             """{"headline": "Related Work", "content": [{"headline": null, "content": [{"headline": null, "content": {"text": "Introduction [CITE:0] of On regular polytopes section 1", "citations": [{"index": 1, "start": 13, "end": 21}], "references": []}}]}]}"""))


class TestPositionFromEnd(TestCase):
    def test_lt(self):
        self.assertTrue(PositionFromEnd(2) < PositionFromEnd(1))
        self.assertTrue(PositionFromEnd(2) < PositionFromEnd(0))
        self.assertTrue(5 < PositionFromEnd(0))
        self.assertFalse(PositionFromEnd(0) < 5)
        self.assertFalse(PositionFromEnd(0) < PositionFromEnd(0))


if __name__ == '__main__':
    unittest.main()
