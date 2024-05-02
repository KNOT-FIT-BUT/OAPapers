# -*- coding: UTF-8 -*-
""""
Created on 17.08.22

:author:     Martin Dočekal
"""
import copy
import re
from unittest import TestCase

from windpyutils.structures.maps import ImmutIntervalMap

from oapapers.hierarchy import Hierarchy, TextContent, RefSpan, SerialNumberFormat, PositionFromEnd
from oapapers.text import SpanCollisionHandling


class TestTextContent(TestCase):

    def test_replace_raises(self):
        t = TextContent("text for testing", [], [])
        with self.assertRaises(ValueError):
            t.replace_at([(5, 8)], ["replaced", "replaced"])

        with self.assertRaises(ValueError):
            t.replace_at([(5, 8), (6, 9)], ["replaced", "replaced"])

    def test_replace_at_no_spans(self):
        t = TextContent("text for testing", [], [])
        t.replace_at([(5, 8)], ["replaced"])
        self.assertEqual(TextContent("text replaced testing", [], []), t)

    def test_replace_at_no_collision(self):
        t = TextContent("context with citations [1] and reference 2", [RefSpan(0, 23, 26)], [RefSpan(0, 41, 42)])
        t.replace_at([(0, 7), (31, 40)], ["replaced", "some"])
        self.assertEqual(TextContent("replaced with citations [1] and some 2", [RefSpan(0, 24, 27)], [RefSpan(0, 37, 38)]), t)

    def test_replace_at_collision_skip(self):
        t = TextContent("context with citations [1] and reference 2", [RefSpan(0, 23, 26)], [RefSpan(0, 41, 42)])
        t.replace_at([(0, 7), (24, 24)], ["replaced", "some"], SpanCollisionHandling.SKIP)
        self.assertEqual(TextContent("replaced with citations [1] and reference 2", [RefSpan(0, 24, 27)], [RefSpan(0, 42, 43)]), t)

    def test_replace_at_collision_remove(self):
        t = TextContent("context with citations [1] and reference 2", [RefSpan(0, 23, 26)], [RefSpan(0, 41, 42)])
        t.replace_at([(0, 7), (24, 24)], ["replaced", "some"], SpanCollisionHandling.REMOVE)
        self.assertEqual(TextContent("replaced with citations [some1] and reference 2", [], [RefSpan(0, 46, 47)]), t)

    def test_replace_at_collision_merge(self):
        t = TextContent("context with citations [1] and reference 2", [RefSpan(0, 23, 26)], [RefSpan(0, 41, 42)])
        t.replace_at([(0, 7), (24, 24)], ["replaced", "some"], SpanCollisionHandling.MERGE)
        self.assertEqual(TextContent("replaced with citations [some1] and reference 2", [RefSpan(0, 24, 31)], [RefSpan(0, 46, 47)]), t)

        t = TextContent("context with citations [1] and reference 2", [RefSpan(0, 23, 26)], [RefSpan(0, 41, 42)])
        t.replace_at([(0, 7), (23, 26)], ["replaced", "(Harvard, 2022)"], SpanCollisionHandling.MERGE)
        self.assertEqual(TextContent("replaced with citations (Harvard, 2022) and reference 2", [RefSpan(0, 24, 39)], [RefSpan(0, 54, 55)]), t)

    def test_replace_at_collision_raise(self):
        t = TextContent("context with citations [1] and reference 2", [RefSpan(0, 23, 26)], [RefSpan(0, 41, 42)])
        with self.assertRaises(ValueError):
            t.replace_at([(0, 7), (24, 24)], ["replaced", "some"], SpanCollisionHandling.RAISE)


class TestHierarchy(TestCase):
    def setUp(self) -> None:
        self.hierarchy = Hierarchy(
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
                        Hierarchy(None, TextContent("text2", [RefSpan(1, 10, 100)], []))
                    ]
                ),
                Hierarchy(
                    "headline2",
                    TextContent("text3", [RefSpan(2, 20, 200)], [RefSpan(0, 20, 200)])
                ),
            ]
        )

    def test_height(self):
        self.assertEqual(3, self.hierarchy.height)
        self.assertEqual(0, Hierarchy(None, TextContent("text0", [], [])).height)
        self.assertEqual(0, Hierarchy(None, []).height)
        self.assertEqual(1, Hierarchy(None,
                                      [
                                          Hierarchy(None, TextContent("text0", [], [])),
                                          Hierarchy(None, TextContent("text1", [], []))
                                      ]).height)

    def test_sections(self):
        self.assertSequenceEqual([self.hierarchy.content[0]], list(self.hierarchy.sections()))
        self.assertSequenceEqual([], list(Hierarchy(None, []).sections()))
        self.assertSequenceEqual([
            self.hierarchy.content[0], self.hierarchy.content[0].content[0]
        ], list(self.hierarchy.sections(1)))

    def test_has_text_content(self):
        self.assertTrue(self.hierarchy.has_text_content)
        self.assertTrue(Hierarchy(None, TextContent("text0", [], [])).has_text_content)
        self.assertFalse(Hierarchy(None, []).has_text_content)
        self.assertFalse(Hierarchy(None, [Hierarchy(None, []), Hierarchy(None, [])]).has_text_content)
        self.assertFalse(Hierarchy(None, TextContent("", [], [])).has_text_content)

    def test_prune_empty_headlines_nodes(self):
        h = copy.deepcopy(self.hierarchy)
        h.prune_empty_headlines_nodes(999)
        self.assertEqual(self.hierarchy, h)
        h.prune_empty_headlines_nodes(2)
        self.assertEqual(self.hierarchy, h)
        self.hierarchy.content[0].content = self.hierarchy.content[0].content[1:]
        h.prune_empty_headlines_nodes(1)
        self.assertEqual(self.hierarchy, h)

    def test_nodes_with_height(self):
        self.assertSequenceEqual([
            Hierarchy(None, TextContent("text0", [], [])), Hierarchy(None, TextContent("text1", [], [])),
            Hierarchy(None, TextContent("text2", [RefSpan(1, 10, 100)], [])),
            Hierarchy("headline2", TextContent("text3", [RefSpan(2, 20, 200)], [RefSpan(0, 20, 200)]))
        ], list(self.hierarchy.nodes_with_height(0)))

        res = list(self.hierarchy.nodes_with_height(1))
        self.assertSequenceEqual(self.hierarchy.content[0].content[:1], res)
        self.assertSequenceEqual([self.hierarchy.content[0]], list(self.hierarchy.nodes_with_height(2)))
        self.assertSequenceEqual([], list(self.hierarchy.nodes_with_height(99)))

    def test_paths_to_nodes_with_height(self):
        self.assertSequenceEqual([], list(self.hierarchy.paths_to_nodes_with_height(10)))
        res = list(self.hierarchy.paths_to_nodes_with_height(0))
        self.assertSequenceEqual([
            [self.hierarchy, self.hierarchy.content[0], self.hierarchy.content[0].content[0],
             self.hierarchy.content[0].content[0].content[0]],
            [self.hierarchy, self.hierarchy.content[0], self.hierarchy.content[0].content[0],
             self.hierarchy.content[0].content[0].content[1]],
            [self.hierarchy, self.hierarchy.content[0], self.hierarchy.content[0].content[1]],
            [self.hierarchy, self.hierarchy.content[1]]
        ], res)

    def test_prune_nodes_without_text_content(self):
        h = copy.deepcopy(self.hierarchy)
        h.prune_nodes_without_text_content()
        self.assertEqual(self.hierarchy, h)

        res = Hierarchy(None, TextContent("text0", [], []))
        res.prune_nodes_without_text_content()
        self.assertEqual(Hierarchy(None, TextContent("text0", [], [])), res)

        res = Hierarchy(None, [])
        res.prune_nodes_without_text_content()
        self.assertEqual(Hierarchy(None, []), res)

        res = Hierarchy(None, [Hierarchy(None, []), Hierarchy(None, [])])
        res.prune_nodes_without_text_content()
        self.assertEqual(Hierarchy(None, []), res)

        res = Hierarchy(None, [Hierarchy(None, []), Hierarchy(None, [])])
        res.prune_nodes_without_text_content()
        self.assertEqual(Hierarchy(None, []), res)

        h.content[1].content = []
        gt = copy.deepcopy(h)
        gt.content = h.content[:1]
        h.prune_nodes_without_text_content()
        self.assertEqual(gt, h)

    def test_prune_named_text_blocks_nothing_to_prune(self):
        h = copy.deepcopy(self.hierarchy)
        h.prune_named_text_blocks({"headline to prune"})
        self.assertEqual(self.hierarchy, h)

    def test_prune_named_text_blocks(self):
        h = copy.deepcopy(self.hierarchy)
        h.content.append(Hierarchy("HEADLINE to prune", TextContent("text4", [], [])))
        h.prune_named_text_blocks({"headline to prune"})
        self.assertEqual(self.hierarchy, h)

    def test_prune_named_text_blocks_no_lower_case(self):
        h = copy.deepcopy(self.hierarchy)
        h.content.append(Hierarchy("HEADLINE to prune", TextContent("text4", [], [])))
        saved = copy.deepcopy(h)
        h.prune_named_text_blocks({"headline to prune"}, lower_case=False)
        self.assertEqual(saved, h)

    def test_text_content(self):
        self.assertListEqual(
            [
                TextContent("text0", [], []),
                TextContent("text1", [], []),
                TextContent("text2", [RefSpan(1, 10, 100)], []),
                TextContent("text3", [RefSpan(2, 20, 200)], [RefSpan(0, 20, 200)])
            ], list(self.hierarchy.text_content()))

    def test_text_content_parent_condition(self):
        self.assertListEqual(
            [
                TextContent("text3", [RefSpan(2, 20, 200)], [RefSpan(0, 20, 200)])
            ], list(self.hierarchy.text_content(lambda x: x.headline == "headline2")))

    def test_pre_order(self):
        sub_hiers = [
            self.hierarchy,
            self.hierarchy.content[0],
            self.hierarchy.content[0].content[0],
            self.hierarchy.content[0].content[0].content[0], self.hierarchy.content[0].content[0].content[1],
            self.hierarchy.content[0].content[1],
            self.hierarchy.content[1],
        ]
        res = list(self.hierarchy.pre_order())
        self.assertListEqual(sub_hiers, res)

    def test_guess_serial_number_strict(self):
        self.assertEqual(None, Hierarchy("LIST OF TABLES", []).guess_serial_number(True))
        self.assertEqual((1,), Hierarchy("INTRODUCTION", []).guess_serial_number(True))
        self.assertEqual(None, Hierarchy("World War II", []).guess_serial_number(True))
        self.assertEqual((5,), Hierarchy("5. Methodology", []).guess_serial_number(True))
        self.assertEqual((5,), Hierarchy("5. Ořech", []).guess_serial_number(True))
        self.assertEqual((5,), Hierarchy("V. Methodology", []).guess_serial_number(True))
        self.assertEqual((4, 2), Hierarchy("4.2 Key notions", []).guess_serial_number(True))
        self.assertEqual((2,), Hierarchy("CHAPTER II", []).guess_serial_number(True))
        self.assertEqual((1,), Hierarchy("Chapter 1", []).guess_serial_number(True))
        self.assertEqual((4, 1),
                         Hierarchy("IV.1 Computational demonstration on pattern recognition", []).guess_serial_number(
                             True))
        self.assertEqual((1, 2, 3, 3), Hierarchy("1.2.3.c", []).guess_serial_number(True))
        self.assertEqual(None, Hierarchy("İ. King", []).guess_serial_number(True))


class TestWithoutSetUpHierarchy(TestCase):

    def test_convert_letter_to_int(self):
        self.assertEqual(1, Hierarchy.convert_letter_to_int("A"))
        self.assertEqual(2, Hierarchy.convert_letter_to_int("B"))
        self.assertEqual(3, Hierarchy.convert_letter_to_int("C"))
        self.assertEqual(4, Hierarchy.convert_letter_to_int("D"))
        with self.assertRaises(AssertionError):
            Hierarchy.convert_letter_to_int("AD")

    def test_serial_number_format_match(self):
        self.assertTrue(Hierarchy.serial_number_format_match((SerialNumberFormat.ARABIC,),
                                                             (SerialNumberFormat.ARABIC,))
                        )
        self.assertTrue(Hierarchy.serial_number_format_match((SerialNumberFormat.ARABIC,
                                                              SerialNumberFormat.ROMAN),
                                                             (SerialNumberFormat.ARABIC,
                                                              SerialNumberFormat.ROMAN))
                        )
        self.assertTrue(Hierarchy.serial_number_format_match((SerialNumberFormat.UNKNOWN,
                                                              SerialNumberFormat.ARABIC,
                                                              SerialNumberFormat.ROMAN),
                                                             (SerialNumberFormat.ROMAN,
                                                              SerialNumberFormat.ARABIC,
                                                              SerialNumberFormat.ROMAN))
                        )
        self.assertTrue(Hierarchy.serial_number_format_match((SerialNumberFormat.ROMAN,
                                                              SerialNumberFormat.ARABIC),
                                                             (SerialNumberFormat.ROMAN,
                                                              SerialNumberFormat.ARABIC,
                                                              SerialNumberFormat.ROMAN))
                        )

        self.assertFalse(Hierarchy.serial_number_format_match((SerialNumberFormat.ROMAN,
                                                               SerialNumberFormat.ARABIC),
                                                              (SerialNumberFormat.ARABIC,
                                                               SerialNumberFormat.ARABIC,
                                                               SerialNumberFormat.ROMAN))
                         )

    def test_serial_numbers_sparsity(self):
        self.assertEqual(2, Hierarchy.serial_numbers_sparsity((1,), (4,)))
        self.assertEqual(0, Hierarchy.serial_numbers_sparsity((1,), (1, 1)))
        self.assertEqual(4, Hierarchy.serial_numbers_sparsity((1, 1), (4, 2)))
        self.assertEqual(4, Hierarchy.serial_numbers_sparsity((1,), (3, 3)))
        self.assertEqual(2, Hierarchy.serial_numbers_sparsity((3, 1, 2), (4, 2)))
        self.assertEqual(3, Hierarchy.serial_numbers_sparsity((4, 2, 2), (4, 2, 4, 2)))

    def test_serial_numbers_sparsity_pos_from_end(self):
        self.assertEqual(0, Hierarchy.serial_numbers_sparsity((1,), (PositionFromEnd(0),)))
        self.assertEqual(2, Hierarchy.serial_numbers_sparsity((1, 2), (3, PositionFromEnd(0),)))
        self.assertEqual(0, Hierarchy.serial_numbers_sparsity((1,), (1, PositionFromEnd(0),)))
        self.assertEqual(-1, Hierarchy.serial_numbers_sparsity((1, PositionFromEnd(0)), (1,)))
        self.assertEqual(0, Hierarchy.serial_numbers_sparsity((PositionFromEnd(1),), (PositionFromEnd(0),)))
        self.assertEqual(1, Hierarchy.serial_numbers_sparsity((PositionFromEnd(2),), (PositionFromEnd(0),)))
        self.assertEqual(-1, Hierarchy.serial_numbers_sparsity((PositionFromEnd(0),), (PositionFromEnd(1),)))

    def test_serial_numbers_sparsity_invalid_input(self):
        self.assertEqual(-1, Hierarchy.serial_numbers_sparsity((4,), (1,)))
        self.assertEqual(-1, Hierarchy.serial_numbers_sparsity((1,), (1,)))

    def test_serial_number_is_subsequent(self):
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (2,)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (1,)))

        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (4,)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (4,), max_sparsity=1))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (4,), max_sparsity=2))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (4,), max_sparsity=999))

        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (1, 1)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (1, 3)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (2, 3)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (2, 3), max_sparsity=3))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (3, 1)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (2, 3)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((3, 1, 2), (4, 2)))

        self.assertTrue(Hierarchy.serial_number_is_subsequent((2, 2), (2, 3)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((2, 2), (3,)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, 2), (3, 1)))

        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, 2), (2,)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, 2), (2, 2)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, 2), (1, 1)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, 2), (2, 1, 2)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent(None, (2, 1, 2)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, 2), None))

        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (PositionFromEnd(0),)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((2,), (2, PositionFromEnd(0))))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, PositionFromEnd(0)), (2,)))

    def test_serial_number_is_subsequent_with_format(self):
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (2,),
                                                              (SerialNumberFormat.ARABIC,),
                                                              (SerialNumberFormat.ARABIC,)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (2,),
                                                              (SerialNumberFormat.UNKNOWN,),
                                                              (SerialNumberFormat.ARABIC,)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (4,),
                                                              (SerialNumberFormat.ROMAN,),
                                                              (SerialNumberFormat.ROMAN,), max_sparsity=2))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (1, 1),
                                                              (SerialNumberFormat.ROMAN,),
                                                              (SerialNumberFormat.ROMAN,
                                                               SerialNumberFormat.ARABIC)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (2, 3),
                                                              (SerialNumberFormat.ARABIC,),
                                                              (SerialNumberFormat.ARABIC,
                                                               SerialNumberFormat.LATIN), max_sparsity=3))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((2, 2), (2, 3),
                                                              (SerialNumberFormat.ROMAN,
                                                               SerialNumberFormat.LATIN),
                                                              (SerialNumberFormat.ROMAN,
                                                               SerialNumberFormat.LATIN)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((2, 2), (3,),
                                                              (SerialNumberFormat.ARABIC,
                                                               SerialNumberFormat.LATIN),
                                                              (SerialNumberFormat.ARABIC,)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((2, 2), (3, 1),
                                                              (SerialNumberFormat.ARABIC,
                                                               SerialNumberFormat.ARABIC),
                                                              (SerialNumberFormat.ARABIC,
                                                               SerialNumberFormat.ARABIC), max_sparsity=4))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, 2), (2, 2),
                                                               (SerialNumberFormat.ARABIC,
                                                                SerialNumberFormat.ARABIC),
                                                               (SerialNumberFormat.ARABIC,
                                                                SerialNumberFormat.ARABIC)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (2,),
                                                               (SerialNumberFormat.ARABIC,),
                                                               (SerialNumberFormat.ROMAN,)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (4,),
                                                               (SerialNumberFormat.LATIN,),
                                                               (SerialNumberFormat.ROMAN,)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((1,), (1, 1),
                                                               (SerialNumberFormat.ROMAN,),
                                                               (SerialNumberFormat.ARABIC,
                                                                SerialNumberFormat.ARABIC)))

        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (PositionFromEnd(0),), (SerialNumberFormat.ARABIC,),
                                                              (SerialNumberFormat.UNKNOWN,)))
        self.assertTrue(Hierarchy.serial_number_is_subsequent((2,), (2, PositionFromEnd(0)),
                                                              (SerialNumberFormat.ARABIC,),
                                                              (SerialNumberFormat.ARABIC, SerialNumberFormat.UNKNOWN)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, PositionFromEnd(0)), (2,),
                                                               (SerialNumberFormat.ARABIC, SerialNumberFormat.UNKNOWN,),
                                                               (SerialNumberFormat.ARABIC,)))

    def test_serial_number_is_direct_subsequent(self):
        self.assertTrue(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 4)))
        self.assertTrue(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 3, 1)))
        self.assertTrue(Hierarchy.serial_number_is_direct_subsequent((3, 3, 1), (3, 4)))

        self.assertFalse(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 5)))
        self.assertFalse(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 3, 2)))
        self.assertFalse(Hierarchy.serial_number_is_direct_subsequent((3, 3, 1), (3, 5)))
        self.assertFalse(Hierarchy.serial_number_is_direct_subsequent((3,), (3, 3, 1)))
        self.assertFalse(Hierarchy.serial_number_is_direct_subsequent((3, 3), (4, 1)))

        self.assertTrue(Hierarchy.serial_number_is_subsequent((1,), (PositionFromEnd(0),)))
        self.assertFalse(Hierarchy.serial_number_is_subsequent((2, PositionFromEnd(0)), (2,)))

    def test_serial_number_is_direct_subsequent_with_format(self):
        self.assertTrue(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 4),
                                                                     (SerialNumberFormat.ARABIC,
                                                                      SerialNumberFormat.ARABIC),
                                                                     (SerialNumberFormat.ARABIC,
                                                                      SerialNumberFormat.ARABIC)))
        self.assertTrue(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 4),
                                                                     (SerialNumberFormat.UNKNOWN,
                                                                      SerialNumberFormat.ARABIC),
                                                                     (SerialNumberFormat.ARABIC,
                                                                      SerialNumberFormat.ARABIC)))
        self.assertTrue(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 3, 1),
                                                                     (SerialNumberFormat.ROMAN,
                                                                      SerialNumberFormat.ARABIC),
                                                                     (SerialNumberFormat.ROMAN,
                                                                      SerialNumberFormat.ARABIC,
                                                                      SerialNumberFormat.LATIN)))
        self.assertTrue(Hierarchy.serial_number_is_direct_subsequent((3, 3, 1), (3, 4),
                                                                     (SerialNumberFormat.ROMAN,
                                                                      SerialNumberFormat.ARABIC,
                                                                      SerialNumberFormat.ARABIC),
                                                                     (SerialNumberFormat.ROMAN,
                                                                      SerialNumberFormat.ARABIC)))

        self.assertFalse(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 5),
                                                                      (SerialNumberFormat.ARABIC,
                                                                       SerialNumberFormat.ARABIC),
                                                                      (SerialNumberFormat.ARABIC,
                                                                       SerialNumberFormat.ARABIC)))
        self.assertFalse(Hierarchy.serial_number_is_direct_subsequent((3, 3), (3, 4),
                                                                      (SerialNumberFormat.ROMAN,
                                                                       SerialNumberFormat.ARABIC),
                                                                      (SerialNumberFormat.ARABIC,
                                                                       SerialNumberFormat.ARABIC)))

        self.assertTrue(Hierarchy.serial_number_is_direct_subsequent((2,), (2, PositionFromEnd(0)),
                                                                     (SerialNumberFormat.ARABIC,),
                                                                     (SerialNumberFormat.ARABIC,
                                                                      SerialNumberFormat.UNKNOWN)))
        self.assertFalse(Hierarchy.serial_number_is_direct_subsequent((2, PositionFromEnd(0)), (2,),
                                                                      (SerialNumberFormat.ARABIC,
                                                                       SerialNumberFormat.UNKNOWN,),
                                                                      (SerialNumberFormat.ARABIC,)))

    def test_search_longest_sparse_subsequent(self):
        self.assertSequenceEqual([(0, 5), (7, 8), (11, 12)],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(1,), (2,), (3,), (4,), (4, 1), None, None, (4, 2), None, None, None, (5,)])
                                 )

        self.assertSequenceEqual([(0, 5), (11, 12)],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(1,), (2,), (3,), (5,), (5, 1), None, None, (4, 2), None, None, None, (7,)])
                                 )

        self.assertSequenceEqual([(0, 3), (5, 7), (9, 10), (13, 14)],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(1,), (2,), (3,), (1,), (2,), (4,), (4, 1), None, None, (4, 2), None, None, None,
                                      (5,)])
                                 )
        # select the left most one
        self.assertSequenceEqual([(0, 2), (4, 6), (8, 9), (12, 13)],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(1,), (2,), (1,), (2,), (3,), (3, 1), None, None, (4, 2), None, None, None, (5,)])
                                 )

        self.assertSequenceEqual([(1, 5), (7, 8), (11, 12)],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(10,), (2,), (3,), (4,), (4, 1), None, None, (4, 2), None, None, None, (5,)])
                                 )

        self.assertSequenceEqual([],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [None, None, None, None])
                                 )

        self.assertSequenceEqual([],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [1, None, None, None])
                                 )

        self.assertSequenceEqual([],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [None, None, None, 1])
                                 )

        self.assertSequenceEqual([],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [])
                                 )

    def test_search_longest_sparse_subsequent_sticky(self):
        self.assertSequenceEqual([(0, 5), (7, 8), (11, 12)],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(1,), (2,), (3,), (4,), (4, 1), None, None, (4, 2), None, None, None, (5,)],
                                     sticky=[0, 11])
                                 )

        self.assertSequenceEqual([],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(1,), (2,), (3,), (5,), (5, 1), None, None, (4, 2), None, None, None, (7,)],
                                     sticky=[3, 5])
                                 )

        self.assertSequenceEqual([(3, 7), (9, 10), (13, 14)],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(1,), (2,), (3,), (1,), (2,), (4,), (4, 1), None, None, (4, 2), None, None, None,
                                      (5,)], sticky=[3])
                                 )

        self.assertSequenceEqual([(3, 7), (9, 10), (13, 14)],
                                 Hierarchy.search_longest_sparse_subsequent(
                                     [(1,), (2,), (3,), (1,), (2,), (4,), (4, 1), None, None, (4, 2), None, None, None,
                                      (5,)], sticky=[3])
                                 )

    def test_fix_alone_number_headlines(self):
        sections = [
            Hierarchy("4.1.1. Relevant International Declarations",
                      [Hierarchy(None, TextContent("Sentence 1", [], []))]),
            Hierarchy("4.1.2.", [
                Hierarchy(None, [
                    Hierarchy(None, TextContent("Sentence 1", [], [])),
                    Hierarchy(None, TextContent("Sentence 2", [], []))
                ])
            ]),
            Hierarchy("4.1.3. Case Study Two: The International", []),
        ]
        anchor = ImmutIntervalMap({(0, 5): True})
        Hierarchy.fix_alone_number_headlines(sections, anchor)
        self.assertSequenceEqual([
            Hierarchy("4.1.1. Relevant International Declarations",
                      [Hierarchy(None, TextContent("Sentence 1", [], []))]),
            Hierarchy("4.1.2. Sentence 1", [
                Hierarchy(None, [
                    Hierarchy(None, TextContent("Sentence 2", [], []))
                ])
            ]),
            Hierarchy("4.1.3. Case Study Two: The International", []),
        ], sections)

        sections = [
            Hierarchy("4.1.1. Relevant International Declarations",
                      [Hierarchy(None, TextContent("Sentence 1", [], []))]),
            Hierarchy("4.1.2.", [
                Hierarchy(None, [
                    Hierarchy(None, TextContent("Sentence 1", [], []))
                ])
            ]),
            Hierarchy("4.1.3. Case Study Two: The International", []),
        ]
        Hierarchy.fix_alone_number_headlines(sections, anchor)
        self.assertSequenceEqual([
            Hierarchy("4.1.1. Relevant International Declarations",
                      [Hierarchy(None, TextContent("Sentence 1", [], []))]),
            Hierarchy("4.1.2. Sentence 1", []),
            Hierarchy("4.1.3. Case Study Two: The International", []),
        ], sections)

    def test_fix_alone_number_headlines_no_change(self):
        sections = [
            Hierarchy("4.1.1. Relevant International Declarations",
                      [Hierarchy(None, TextContent("Sentence 1", [], []))]),
            Hierarchy("4.1.2.", [
                Hierarchy(None, [
                    Hierarchy(None, TextContent("Sentence 1", [], [])),
                    Hierarchy(None, TextContent("Sentence 2", [], []))
                ])
            ]),
            Hierarchy("4.1.4. Case Study Two: The International", []),
        ]
        anchor = ImmutIntervalMap({(0, 0): True, (2, 5): True})
        Hierarchy.fix_alone_number_headlines(sections, anchor)
        self.assertSequenceEqual([
            Hierarchy("4.1.1. Relevant International Declarations",
                      [Hierarchy(None, TextContent("Sentence 1", [], []))]),
            Hierarchy("4.1.2.", [
                Hierarchy(None, [
                    Hierarchy(None, TextContent("Sentence 1", [], [])),
                    Hierarchy(None, TextContent("Sentence 2", [], []))
                ])
            ]),
            Hierarchy("4.1.4. Case Study Two: The International", []),
        ], sections)

    def test_merge_split_headlines(self):
        sections = [
            Hierarchy("CHAPTER II", []),
            Hierarchy("REVIEW OF ENERGY STORAGE DEVICES FOR POWER", []),
            Hierarchy("ELECTRONICS APPLICATIONS", []),
            Hierarchy("2.1 Introduction", [Hierarchy(None, TextContent("Power electronics applications", [], []))]),
        ]
        ser_num = [(2,), None, None, (2, 1)]
        anchor_map = ImmutIntervalMap({(0, 0): True, (3, 3): True})

        self.assertSequenceEqual([
            Hierarchy("CHAPTER II REVIEW OF ENERGY STORAGE DEVICES FOR POWER ELECTRONICS APPLICATIONS", []),
            Hierarchy("2.1 Introduction", [Hierarchy(None, TextContent("Power electronics applications", [], []))]),
        ], Hierarchy.merge_split_headlines(sections, ser_num, anchor_map))

        sections = [
            Hierarchy("PART (I) INTRODUCTION", []),
            Hierarchy("1. THE GOALS AND THE STRUCTURE OF THIS BOOK", []),
            Hierarchy("1.1. The Goals of this Book", [])
        ]
        ser_num = [(1,), (1,), (1, 1)]
        anchor_map = ImmutIntervalMap({(0, 3): True})

        self.assertSequenceEqual([
            Hierarchy("PART (I) INTRODUCTION", []),
            Hierarchy("1. THE GOALS AND THE STRUCTURE OF THIS BOOK", []),
            Hierarchy("1.1. The Goals of this Book", []),
        ], Hierarchy.merge_split_headlines(sections, ser_num, anchor_map))

        sections = [
            Hierarchy("CHAPTER II", []),
            Hierarchy("LOW DENSITY PARITY CHECK CODES",
                      [Hierarchy(None, TextContent("This chapter presents", [], []))]),
            Hierarchy("2.1. Product Accumulate Codes:", [])
        ]
        ser_num = [(2,), None, (2, 1)]
        anchor_map = ImmutIntervalMap({(0, 0): True, (2, 2): True})

        self.assertSequenceEqual([
            Hierarchy("CHAPTER II LOW DENSITY PARITY CHECK CODES",
                      [Hierarchy(None, TextContent("This chapter presents", [], []))]),
            Hierarchy("2.1. Product Accumulate Codes:", [])
        ], Hierarchy.merge_split_headlines(sections, ser_num, anchor_map))

        sections = [
            Hierarchy("CHPTER II", []),
            Hierarchy("REVIEW OF ENERGY STORAGE DEVICES FOR POWER", []),
            Hierarchy("ELECTRONICS APPLICATIONS", []),
            Hierarchy("2.1 Introduction", [Hierarchy(None, TextContent("Power electronics applications", [], []))]),
        ]
        ser_num = [None, None, None, (2, 1)]
        anchor_map = ImmutIntervalMap({(3, 3): True})

        self.assertSequenceEqual([
            Hierarchy("CHPTER II REVIEW OF ENERGY STORAGE DEVICES FOR POWER ELECTRONICS APPLICATIONS", []),
            Hierarchy("2.1 Introduction", [Hierarchy(None, TextContent("Power electronics applications", [], []))]),
        ], Hierarchy.merge_split_headlines(sections, ser_num, anchor_map))

    def test_split_merged_headlines(self):
        """
        <div xmlns="http://www.tei-c.org/ns/1.0"><head>6. Conclusions and future work</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>Appendix A. Notational representation of social argument</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>B. Operational semantics </head></div>
        :return:
        """
        sections = [
            Hierarchy("6. Conclusions and future work", []),
            Hierarchy("Appendix A. Notational representation of social argument",
                      [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Operational semantics", [Hierarchy(None, TextContent("text", [], []))]),
        ]
        ser_num = [(6,), (PositionFromEnd(0),), (2,)]
        ser_num_format = [(SerialNumberFormat.ARABIC,), (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.LATIN,)]

        self.assertSequenceEqual([
            Hierarchy("6. Conclusions and future work", []),
            Hierarchy("Appendix", []),
            Hierarchy("A. Notational representation of social argument",
                      [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Operational semantics", [Hierarchy(None, TextContent("text", [], []))]),
        ], Hierarchy.split_merged_headlines(sections, ser_num, ser_num_format))

    def test_merge_split_headlines_non_strict(self):
        sections = [
            Hierarchy("CHPTER II", []),
            Hierarchy("LOW DENSITY PARITY CHECK CODES",
                      [Hierarchy(None, TextContent("This chapter presents", [], []))]),
            Hierarchy("2.1. Product Accumulate Codes:", [])
        ]
        ser_num = [None, None, (2, 1)]
        anchor_map = ImmutIntervalMap({(2, 2): True})

        res = Hierarchy.merge_split_headlines(sections, ser_num, anchor_map)
        self.assertSequenceEqual([
            Hierarchy("CHPTER II LOW DENSITY PARITY CHECK CODES",
                      [Hierarchy(None, TextContent("This chapter presents", [], []))]),
            Hierarchy("2.1. Product Accumulate Codes:", [])
        ], res)

    def test_guess_serial_number(self):
        self.assertEqual(None, Hierarchy("LIST OF TABLES", []).guess_serial_number(False))
        self.assertEqual((1,), Hierarchy("INTRODUCTION", []).guess_serial_number(False))
        self.assertEqual((2,), Hierarchy("World War II", []).guess_serial_number(False))
        self.assertEqual((5,), Hierarchy("5. Methodology", []).guess_serial_number(False))
        self.assertEqual((5,), Hierarchy("V. Methodology", []).guess_serial_number(False))
        self.assertEqual((4, 2), Hierarchy("4.2 Key notions", []).guess_serial_number(False))
        self.assertEqual((2,), Hierarchy("CHAPTER II", []).guess_serial_number(False))
        self.assertEqual((1,), Hierarchy("Chapter 1", []).guess_serial_number(False))
        self.assertEqual((4, 1),
                         Hierarchy("IV.1 Computational demonstration on pattern recognition", []).guess_serial_number(
                             False))
        self.assertEqual((1, 2, 3, 3), Hierarchy("1.2.3.c", []).guess_serial_number(False))

    def test_guess_serial_number_with_format(self):
        self.assertEqual(None,
                         Hierarchy("LIST OF TABLES", []).guess_serial_number(False, num_format=True))
        self.assertEqual(((1,), (SerialNumberFormat.UNKNOWN,)),
                         Hierarchy("INTRODUCTION", []).guess_serial_number(False, num_format=True))
        self.assertEqual(((2,), (SerialNumberFormat.ROMAN,)),
                         Hierarchy("World War II", []).guess_serial_number(False, num_format=True))
        self.assertEqual(((5,), (SerialNumberFormat.ARABIC,)),
                         Hierarchy("5. Methodology", []).guess_serial_number(False, num_format=True))
        self.assertEqual(((5,), (SerialNumberFormat.ROMAN,)),
                         Hierarchy("V. Methodology", []).guess_serial_number(False, num_format=True))
        self.assertEqual(((4, 2), (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC)),
                         Hierarchy("4.2 Key notions", []).guess_serial_number(False, num_format=True))
        self.assertEqual(((2,), (SerialNumberFormat.ROMAN,)),
                         Hierarchy("CHAPTER II", []).guess_serial_number(False, num_format=True))
        self.assertEqual(((1,), (SerialNumberFormat.ARABIC,)),
                         Hierarchy("Chapter 1", []).guess_serial_number(False, num_format=True))
        self.assertEqual(((4, 1), (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC)),
                         Hierarchy("IV.1 Computational demonstration on pattern recognition", []).guess_serial_number(
                             False, num_format=True))
        self.assertEqual(((1, 2, 3, 3), (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC,
                                         SerialNumberFormat.ARABIC, SerialNumberFormat.LATIN)),
                         Hierarchy("I.2.3.c", []).guess_serial_number(False, num_format=True))

    def test_guess_content_serial_numbers(self):
        h = Hierarchy("1. introduction", [
            Hierarchy("LIST OF TABLES", []), Hierarchy("INTRODUCTION", []), Hierarchy("World War II", []),
            Hierarchy("5. Methodology", []), Hierarchy("V. Methodology", []), Hierarchy("4.2 Key notions", []),
            Hierarchy("CHAPTER II", []), Hierarchy("Chapter 1", []),
            Hierarchy("IV.1 Computational demonstration on pattern recognition", []), Hierarchy("1.2.3.c", [])
        ])

        ser_numbers, formats = h.guess_content_serial_numbers(False)

        self.assertSequenceEqual(
            [None, (1,), (2,), (5,), (5,), (4, 2), (2,), (1,), (4, 1), (1, 2, 3, 3)],
            ser_numbers
        )
        self.assertSequenceEqual(
            [
                (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.ROMAN,),
                (SerialNumberFormat.ARABIC,), (SerialNumberFormat.ROMAN,),
                (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC), (SerialNumberFormat.ROMAN,),
                (SerialNumberFormat.ARABIC,), (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
                (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC,
                 SerialNumberFormat.LATIN)],
            formats
        )

    def test_guess_content_serial_numbers_strict(self):
        h = Hierarchy("1. introduction", [
            Hierarchy("LIST OF TABLES", []), Hierarchy("INTRODUCTION", []), Hierarchy("World War II", []),
            Hierarchy("5. Methodology", []), Hierarchy("5. Ořech", []), Hierarchy("V. Methodology", []),
            Hierarchy("4.2 Key notions", []), Hierarchy("CHAPTER II", []), Hierarchy("Chapter 1", []),
            Hierarchy("IV.1 Computational demonstration on pattern recognition", []), Hierarchy("1.2.3.c", []),
            Hierarchy("İ. King", [])
        ])

        ser_numbers, formats = h.guess_content_serial_numbers(True)

        self.assertSequenceEqual(
            [None, (1,), None, (5,), (5,), (5,), (4, 2), (2,), (1,), (4, 1), (1, 2, 3, 3), None],
            ser_numbers
        )
        self.assertSequenceEqual(
            [
                (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.UNKNOWN,),
                (SerialNumberFormat.ARABIC,), (SerialNumberFormat.ARABIC,), (SerialNumberFormat.ROMAN,),
                (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
                (SerialNumberFormat.ROMAN,), (SerialNumberFormat.ARABIC,),
                (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
                (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC,
                 SerialNumberFormat.LATIN), (SerialNumberFormat.UNKNOWN,)],
            formats
        )

    def test_correct_the_roman_format_mismatch_not_enough_context(self):
        ser_nums = [(1,)]
        ser_nums_format = [(SerialNumberFormat.ROMAN,)]
        ser_nums_res, ser_nums_format_res = Hierarchy.correct_the_roman_format_mismatch(ser_nums, ser_nums_format)
        self.assertSequenceEqual([(1,)], ser_nums_res)
        self.assertSequenceEqual([(SerialNumberFormat.ROMAN,)], ser_nums_format_res)

    def test_correct_the_roman_format_mismatch_without_roman(self):
        ser_nums = [(1,), (2,), (3,)]
        ser_nums_format = [(SerialNumberFormat.ARABIC,), (SerialNumberFormat.ARABIC,), (SerialNumberFormat.ARABIC,)]
        ser_nums_res, ser_nums_format_res = Hierarchy.correct_the_roman_format_mismatch(ser_nums, ser_nums_format)
        self.assertSequenceEqual([(1,), (2,), (3,)], ser_nums_res)
        self.assertSequenceEqual(
            [(SerialNumberFormat.ARABIC,), (SerialNumberFormat.ARABIC,), (SerialNumberFormat.ARABIC,)],
            ser_nums_format_res)

    def test_correct_the_roman_format_mismatch_easy(self):
        ser_nums = [(1,), (2,), (100,), (500,), (5,)]
        ser_nums_format = [(SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,), (SerialNumberFormat.ROMAN,),
                           (SerialNumberFormat.ROMAN,), (SerialNumberFormat.LATIN,)]
        ser_nums_res, ser_nums_format_res = Hierarchy.correct_the_roman_format_mismatch(ser_nums, ser_nums_format)
        self.assertSequenceEqual([(1,), (2,), (3,), (4,), (5,)], ser_nums_res)
        self.assertSequenceEqual([(SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,),
                                  (SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,)], ser_nums_format_res)

    def test_correct_the_roman_format_mismatch_with_none(self):
        ser_nums = [(1,), (2,), (100,), (500,), None]
        ser_nums_format = [(SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,), (SerialNumberFormat.ROMAN,),
                           (SerialNumberFormat.ROMAN,), (SerialNumberFormat.UNKNOWN,)]
        ser_nums_res, ser_nums_format_res = Hierarchy.correct_the_roman_format_mismatch(ser_nums, ser_nums_format)
        self.assertSequenceEqual([(1,), (2,), (3,), (4,), None], ser_nums_res)
        self.assertSequenceEqual([(SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,),
                                  (SerialNumberFormat.LATIN,), (SerialNumberFormat.UNKNOWN,)], ser_nums_format_res)

    def test_correct_the_roman_format_mismatch_subsection(self):
        ser_nums = [(1,), (1, 1), (1, 2), (1, 100), (1, 100, 1), (1, 500), (2,)]
        ser_nums_format = [
            (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.LATIN, SerialNumberFormat.LATIN),
            (SerialNumberFormat.LATIN, SerialNumberFormat.LATIN),
            (SerialNumberFormat.LATIN, SerialNumberFormat.ROMAN),
            (SerialNumberFormat.LATIN, SerialNumberFormat.ROMAN, SerialNumberFormat.LATIN),
            (SerialNumberFormat.LATIN, SerialNumberFormat.ROMAN),
            (SerialNumberFormat.LATIN,)
        ]
        ser_nums_res, ser_nums_format_res = Hierarchy.correct_the_roman_format_mismatch(ser_nums, ser_nums_format)
        self.assertSequenceEqual([(1,), (1, 1), (1, 2), (1, 3), (1, 3, 1), (1, 4), (2,)], ser_nums_res)
        self.assertSequenceEqual([
            (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.LATIN, SerialNumberFormat.LATIN),
            (SerialNumberFormat.LATIN, SerialNumberFormat.LATIN),
            (SerialNumberFormat.LATIN, SerialNumberFormat.LATIN),
            (SerialNumberFormat.LATIN, SerialNumberFormat.LATIN, SerialNumberFormat.LATIN),
            (SerialNumberFormat.LATIN, SerialNumberFormat.LATIN),
            (SerialNumberFormat.LATIN,)
        ], ser_nums_format_res)

    def test_correct_the_roman_format_mismatch_reset_counter(self):
        ser_nums = [(1,), (1,), (2,), (100,), (500,), (2,)]
        ser_nums_format = [
            (SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ARABIC,)
        ]

        ser_nums_res, ser_nums_format_res = Hierarchy.correct_the_roman_format_mismatch(ser_nums, ser_nums_format)
        self.assertSequenceEqual([(1,), (1,), (2,), (3,), (4,), (2,)], ser_nums_res)
        self.assertSequenceEqual([
            (SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.ARABIC,)
        ], ser_nums_format_res)

    def test_correct_the_roman_format_mismatch_reset_counter_roman_subsection(self):
        ser_nums = [(1,), (1,), (2,), (3,), (2,)]
        ser_nums_format = [(SerialNumberFormat.LATIN,), (SerialNumberFormat.ROMAN,), (SerialNumberFormat.ROMAN,),
                           (SerialNumberFormat.ROMAN,), (SerialNumberFormat.LATIN,)]
        ser_nums_res, ser_nums_format_res = Hierarchy.correct_the_roman_format_mismatch(ser_nums, ser_nums_format)
        self.assertSequenceEqual([(1,), (1,), (2,), (3,), (2,)], ser_nums_res)
        self.assertSequenceEqual([(SerialNumberFormat.LATIN,), (SerialNumberFormat.ROMAN,), (SerialNumberFormat.ROMAN,),
                                  (SerialNumberFormat.ROMAN,), (SerialNumberFormat.LATIN,)], ser_nums_format_res)

    def test_repair_reset_counter_ser_nums(self):
        ser_nums = [
            (1,), (1,), (1, 1), (2,), (2, 1), (2,), None, (2, 1), (2, 2), (2, 2, 1), (3,), (3, 1), (3, 2), (4,), None,
            (5,), None
        ]
        ser_nums_formats = [
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ]
        sticky = [0, 5, 10, 13, 15]
        """
        Hierarchy("PART (I) INTRODUCTION", []),
        Hierarchy("1. THE GOALS AND THE STRUCTURE OF THIS BOOK", []),
        Hierarchy("1.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("2. THE GOALS AND THE STRUCTURE OF THIS BOOK", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("2.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("PART II. second section", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("II.2 second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("CHAPTER III. third section", []),
        Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("CHAPTER IV.", []),
        Hierarchy("fourth section", [Hierarchy(None, TextContent("text", [], []))]),
        Hierarchy("CHAPTER V.", []),
        Hierarchy("fifth section", [Hierarchy(None, TextContent("text", [], []))]),
        """
        res, res_format = Hierarchy.repair_reset_counter_ser_nums(ser_nums, sticky, ser_nums_formats, 1)
        self.assertSequenceEqual([
            (1,), (1, 1), (1, 1, 1), (1, 2), (1, 2, 1), (2,), None, (2, 1), (2, 2), (2, 2, 1), (3,), (3, 1), (3, 2),
            (4,), None, (5,), None
        ], res)
        self.assertSequenceEqual([
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ], res_format)

    def test_repair_reset_counter_ser_nums_no_need_for_num_repair(self):
        ser_nums = [
            (1,), (1, 1), (1, 2), None, (1, 3), (2,), (2, 1), None, (2, 2), (2, 3), (2, 3, 1), (3,), (3, 1), (3, 2),
            (4,), None,
            (5,), None
        ]
        ser_nums_formats = [
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ]
        sticky = [0, 5, 11, 14, 16]
        res, res_format = Hierarchy.repair_reset_counter_ser_nums(ser_nums, sticky, ser_nums_formats, 1)
        self.assertSequenceEqual([
            (1,), (1, 1), (1, 2), None, (1, 3), (2,), (2, 1), None, (2, 2), (2, 3), (2, 3, 1), (3,), (3, 1), (3, 2),
            (4,), None,
            (5,), None
        ], res)
        self.assertSequenceEqual([
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ], res_format)

    def test_repair_reset_counter_ser_nums_just_format_repair(self):
        ser_nums = [
            (1,), (1, 1), (1, 2), None, (1, 3), (2,), (2, 1), None, (2, 2), (2, 3), (2, 3, 1), (3,), (3, 1), (3, 2),
            (4,), None,
            (5,), None
        ]
        ser_nums_formats = [
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ]
        sticky = [0, 5]
        res, res_format = Hierarchy.repair_reset_counter_ser_nums(ser_nums, sticky, ser_nums_formats, 1)
        self.assertSequenceEqual([
            (1,), (1, 1), (1, 2), None, (1, 3), (2,), (2, 1), None, (2, 2), (2, 3), (2, 3, 1), (3,), (3, 1), (3, 2),
            (4,), None,
            (5,), None
        ], res)
        self.assertSequenceEqual([
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ], res_format)

    def test_repair_reset_counter_ser_nums_just_sticky_introduction(self):
        ser_nums = [
            (1,), (1,), (1, 1), (2,), (2, 1), (2,), None, (2, 1), (2, 2), (2, 2, 1), (3,), (3, 1), (3, 2), (4,), None,
            (5,), None
        ]
        ser_nums_formats = [
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ]
        sticky = [0]

        res, res_format = Hierarchy.repair_reset_counter_ser_nums(ser_nums, sticky, ser_nums_formats, 1)
        self.assertSequenceEqual([
            (1,), (1, 1), (1, 1, 1), (1, 2), (1, 2, 1), (2,), None, (2, 1), (2, 2), (2, 2, 1), (3,), (3, 1), (3, 2),
            (4,), None, (5,), None
        ], res)
        self.assertSequenceEqual([
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ], res_format)

    def test_repair_reset_counter_ser_nums_no_sticky(self):
        ser_nums = [
            (1,), (1, 1), (2,), (2, 1), (2,), None, (2, 1), (2, 2), (2, 2, 1), (3,), (3, 1), (3, 2), (4,), (5,)
        ]
        ser_nums_formats = [
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ARABIC,),
            (SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.ARABIC),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,)
        ]
        res, res_format = Hierarchy.repair_reset_counter_ser_nums(ser_nums, [], ser_nums_formats,
                                                                  1)
        self.assertSequenceEqual(ser_nums, res)
        self.assertSequenceEqual(ser_nums_formats, res_format)

    def test_flat_2_multi(self):
        h = Hierarchy("Document 2", [
            Hierarchy("I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2 second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. third section", []),
            Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 2", [
            Hierarchy("I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.2 second section part 2", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("II.2.1 first part second section part 2",
                              [Hierarchy(None, TextContent("text", [], []))]),
                ]),
            ]),
            Hierarchy("III. third section", [
                Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])

        self.assertEqual(gt, h)

        h = Hierarchy("Document 2", [
            Hierarchy("I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2 second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. third section", []),
            Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 2", [
            Hierarchy("I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.2 second section part 2", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
                    Hierarchy("II.2.1 first part second section part 2",
                              [Hierarchy(None, TextContent("text", [], []))]),
                ]),
            ]),
            Hierarchy("III. third section", [
                Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])

        self.assertEqual(gt, h)

        h = Hierarchy("Document 2", [
            Hierarchy("I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2 second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. third section", []),
            Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 2", [
            Hierarchy("I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.2 second section part 2", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("II.2.1 first part second section part 2",
                              [Hierarchy(None, TextContent("text", [], []))]),
                ]),
            ]),
            Hierarchy("III. third section", [
                Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])

        self.assertEqual(gt, h)

        h = Hierarchy("Document 2", [
            Hierarchy("CHAPTER I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER II. second section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2 second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER III. third section", []),
            Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER IV.", []),
            Hierarchy("fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER V.", []),
            Hierarchy("fifth section", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 2", [
            Hierarchy("CHAPTER I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER II. second section", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.2 second section part 2", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("II.2.1 first part second section part 2",
                              [Hierarchy(None, TextContent("text", [], []))]),
                ]),
            ]),
            Hierarchy("CHAPTER III. third section", [
                Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("CHAPTER IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])

        self.assertEqual(gt, h)

        h = Hierarchy("Document 3", [
            Hierarchy("PART (I) INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("1. THE GOALS AND THE STRUCTURE OF THIS BOOK", []),
            Hierarchy("1.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("2. THE GOALS AND THE STRUCTURE OF THIS BOOK", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("2.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("PART II. second section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2 second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER III. third section", []),
            Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER IV.", []),
            Hierarchy("fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER V.", []),
            Hierarchy("fifth section", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 3", [
            Hierarchy("PART (I) INTRODUCTION", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("1. THE GOALS AND THE STRUCTURE OF THIS BOOK", [
                    Hierarchy("1.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))])
                ]),
                Hierarchy("2. THE GOALS AND THE STRUCTURE OF THIS BOOK", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("2.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))])
                ]),
            ]),
            Hierarchy("PART II. second section", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.2 second section part 2", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("II.2.1 first part second section part 2",
                              [Hierarchy(None, TextContent("text", [], []))]),
                ]),
            ]),
            Hierarchy("CHAPTER III. third section", [
                Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("CHAPTER IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertEqual(gt, h)

        h = Hierarchy("Document 4", [
            Hierarchy("PART (I) INTRODUCTION", []),
            Hierarchy("1. THE GOALS AND THE STRUCTURE OF THIS BOOK", []),
            Hierarchy("1.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("2. THE GOALS AND THE STRUCTURE OF THIS BOOK", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("2.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("PART II. second section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2 second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER III. third section", []),
            Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER IV.", []),
            Hierarchy("fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER V.", []),
            Hierarchy("fifth section", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 4", [
            Hierarchy("PART (I) INTRODUCTION", [
                Hierarchy("1. THE GOALS AND THE STRUCTURE OF THIS BOOK", [
                    Hierarchy("1.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))])
                ]),
                Hierarchy("2. THE GOALS AND THE STRUCTURE OF THIS BOOK", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("2.1. The Goals of this Book", [Hierarchy(None, TextContent("text", [], []))])
                ]),
            ]),
            Hierarchy("PART II. second section", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("Alone headline", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.2 second section part 2", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("II.2.1 first part second section part 2",
                              [Hierarchy(None, TextContent("text", [], []))]),
                ]),
            ]),
            Hierarchy("CHAPTER III. third section", [
                Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("CHAPTER IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertEqual(gt, h)

        h = Hierarchy("Document 5", [
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", []),
            Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 5", [
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", [
                Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertEqual(gt, h)
        h = Hierarchy("Document 5", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", []),
            Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 5", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", [
                Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertEqual(gt, h)

    def test_flat_2_multi_match_cap_gap_fill(self):
        h = Hierarchy("Document 5", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 Evaluation", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("RESULTS", []),
            Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("AUTHOR CONTRIBUTIONS", []),
            Hierarchy("A. SOTA", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Nice paper", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 5", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("II.1 Evaluation", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("RESULTS", [
                Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("AUTHOR CONTRIBUTIONS", [
                Hierarchy("A. SOTA", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Nice paper", [Hierarchy(None, TextContent("text", [], []))])
            ]),
        ])
        self.assertEqual(gt, h)

    def test_example_from_paper(self):
        h = Hierarchy("Document Title", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER II. MODEL", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 Evaluation", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.3 Metrics", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("* H4 23 y 6", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. group of", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.3.1 Correlation", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("RESULTS", []),
            Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("D. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("AUTHOR CONTRIBUTIONS", []),
            Hierarchy("A. SOTA", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Data", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document Title", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CHAPTER II. MODEL", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("II.1 Evaluation", [
                    Hierarchy(None, TextContent("text", [], [])),
                ]),
                Hierarchy("II.3 Metrics", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("* H4 23 y 6", [Hierarchy(None, TextContent("text", [], []))]),
                    Hierarchy("II. group of", [Hierarchy(None, TextContent("text", [], []))]),
                    Hierarchy("II.3.1 Correlation", [Hierarchy(None, TextContent("text", [], []))]),
                ]),
            ]),
            Hierarchy("RESULTS", [
                Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("D. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("AUTHOR CONTRIBUTIONS", [
                Hierarchy("A. SOTA", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Data", [Hierarchy(None, TextContent("text", [], []))])
            ]),
        ])
        self.assertEqual(gt, h)

    def test_search_endings(self):
        content = [
            Hierarchy("some stuff", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("CONCLUSION that should be ommited", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("acknowledgement that should be omitted", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("acknowledgement", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("some other stuff", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Appendix", []),
        ]
        content_check = copy.deepcopy(content)
        ser_num = [None, (4,), None, None, None, None, None]
        sticky = [1]
        Hierarchy.search_endings(content, ser_num, sticky)

        self.assertSequenceEqual(content_check, content)
        self.assertSequenceEqual([None, (4,), None, None, (PositionFromEnd(1),), None, (PositionFromEnd(0),)], ser_num)
        self.assertSequenceEqual([1, 4, 6], sticky)

    def test_match_cap_gap_filler(self):
        content = [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("some stuff", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", []),
            Hierarchy("III.A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("AUTHOR CONTRIBUTIONS", []),
            Hierarchy("A. SOTA", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Nice paper", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("some other stuff", [Hierarchy(None, TextContent("text", [], []))]),
        ]
        content_check = copy.deepcopy(content)
        ser_num = [None, (1,), None, None, (3,), (3, 1,), (3, 2,), (3, 3,), (4,), None, (1,), (2,), None]
        ser_num_formats = [
            (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.ROMAN,), (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.LATIN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.LATIN,),
            (SerialNumberFormat.ROMAN, SerialNumberFormat.LATIN,), (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.UNKNOWN,)
        ]
        anchors = [1, 4, 5, 6, 7, 8]
        Hierarchy.match_cap_gap_filler(content, ser_num, ser_num_formats, anchors)

        self.assertSequenceEqual(content_check, content)
        self.assertSequenceEqual(
            [None, (1,), None, (2,), (3,), (3, 1,), (3, 2,), (3, 3,), (4,), (5,), (1,), (2,), None],
            ser_num)
        self.assertSequenceEqual([1, 4, 5, 6, 7, 8], anchors)

    def test_match_cap_gap_filler_not_enough_evidence(self):
        content = [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("some stuff", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", []),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("AUTHOR CONTRIBUTIONS", []),
            Hierarchy("A. SOTA", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Nice paper", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("some other stuff", [Hierarchy(None, TextContent("text", [], []))]),
        ]
        content_check = copy.deepcopy(content)
        ser_num = [None, (1,), None, None, (3,), (4,), None, (1,), (2,), None]
        ser_num_formats = [
            (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.ROMAN,), (SerialNumberFormat.UNKNOWN,),
            (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.ROMAN,), (SerialNumberFormat.ROMAN,),
            (SerialNumberFormat.UNKNOWN,), (SerialNumberFormat.LATIN,), (SerialNumberFormat.LATIN,),
            (SerialNumberFormat.UNKNOWN,)
        ]
        anchors = [1, 4, 5]
        Hierarchy.match_cap_gap_filler(content, ser_num, ser_num_formats, anchors)

        self.assertSequenceEqual(content_check, content)
        self.assertSequenceEqual([None, (1,), None, None, (3,), (4,), None, (1,), (2,), None],
                                 ser_num)
        self.assertSequenceEqual([1, 4, 5], anchors)

    def test_flat_2_multi_appendix(self):
        h = Hierarchy("Document 6", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", []),
            Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Appendix", []),
            Hierarchy("A. Model Details", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Additional Results", [Hierarchy(None, TextContent("text", [], []))]),
        ])

        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 6", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", [
                Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Appendix", [
                Hierarchy("A. Model Details", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Additional Results", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
        ])
        self.assertEqual(gt, h)

        h = Hierarchy("Document 7", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", []),
            Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Appendix", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("A. Model Details", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Additional Results", [Hierarchy(None, TextContent("text", [], []))]),
        ])

        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 7", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", [
                Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Appendix", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("A. Model Details", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Additional Results", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
        ])
        self.assertEqual(gt, h)

        h = Hierarchy("Document 8", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", []),
            Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Appendix A. Model Details", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Additional Results", [Hierarchy(None, TextContent("text", [], []))]),
        ])

        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 8", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", [
                Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Appendix", [
                Hierarchy("A. Model Details", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Additional Results", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
        ])
        self.assertEqual(gt, h)

    def test_flat_2_multi_endings(self):
        h = Hierarchy("Document 6", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", []),
            Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("ACKNOWLEDGEMENT", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("REFERENCES", [Hierarchy(None, TextContent("text", [], []))])
        ])

        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 6", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("I. INTRODUCTION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. MODEL AND EXACT DIAGONALIZATION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III. RESULTS", [
                Hierarchy("A. Chemical Potential", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("B. Specific Heat, Entropy", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("C. Spin Susceptibility", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. CONCLUSION", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("ACKNOWLEDGEMENT", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("REFERENCES", [Hierarchy(None, TextContent("text", [], []))]),
        ])
        self.assertEqual(gt, h)

    def test_flat_2_multi_shallow(self):
        h = Hierarchy("Document 1", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Introduction", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Methods", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Subjects", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Measurement of confounding factors", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Statistical analyses", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Results", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Discussion", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertTrue(h.flat_2_multi())
        gt = Hierarchy("Document 1", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Introduction", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Methods", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Subjects", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Measurement of confounding factors", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Statistical analyses", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Results", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Discussion", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertEqual(gt, h)

    def test_flat_2_multi_shallow_with_one_number(self):
        h = Hierarchy("Document 1", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Introduction", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. Methods", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Subjects", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Measurement of confounding factors", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Statistical analyses", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Results", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Discussion", [Hierarchy(None, TextContent("text", [], []))])
        ])
        # it should pass, but not make multi level hier as there is too few headlines with number
        self.assertTrue(h.flat_2_multi())
        gt = Hierarchy("Document 1", [
            Hierarchy("Abstract", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Introduction", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. Methods", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Subjects", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Measurement of confounding factors", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Statistical analyses", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Results", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("Discussion", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertEqual(gt, h)

    def test_flat_2_multi_with_typo_before_num(self):
        h = Hierarchy("Document 2", [
            Hierarchy("I.", []),
            Hierarchy("first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2", [
                Hierarchy(None, TextContent("second section part 2", [], [])),
                Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("- III. third section", []),
            Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])
        self.assertTrue(h.flat_2_multi())

        gt = Hierarchy("Document 2", [
            Hierarchy("I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [
                Hierarchy(None, TextContent("text", [], [])),
                Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("II.2 second section part 2", [
                    Hierarchy(None, TextContent("text", [], [])),
                    Hierarchy("II.2.1 first part second section part 2",
                              [Hierarchy(None, TextContent("text", [], []))]),
                ]),
            ]),
            Hierarchy("- III. third section", [
                Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
                Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            ]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ])

        self.assertEqual(gt, h)

    def test_flat_2_multi_failing(self):
        self.assertFalse(Hierarchy("Document 2", [
            Hierarchy("I. first section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II. second section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.1 second section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2 second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("II.2.1 first part second section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.1 third section part 1", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("III.2 third section part 2", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("IV. fourth section", [Hierarchy(None, TextContent("text", [], []))]),
            Hierarchy("V. fifth section", [Hierarchy(None, TextContent("text", [], []))])
        ]).flat_2_multi())

    def test_chars_name_coverage(self):
        pattern = re.compile(r"^LATIN (CAPITAL|SMALL) LETTER .$", re.IGNORECASE | re.UNICODE)
        self.assertEqual(1, Hierarchy.chars_name_coverage("Hello", pattern))
        self.assertEqual(0, Hierarchy.chars_name_coverage("", pattern))
        self.assertEqual(0.5, Hierarchy.chars_name_coverage("a ", pattern))
        self.assertEqual(0.0, Hierarchy.chars_name_coverage("ØÓÖ ½ ØÓÖ ¾", pattern))
        self.assertEqual(0.0, Hierarchy.chars_name_coverage("1 4 5", pattern))

    def test_get_part(self):
        hierarchy = Hierarchy(
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
                        Hierarchy("headline3",
                                  [
                                      Hierarchy(None,
                                                [
                                                    Hierarchy(None, TextContent("text0", [], [])),
                                                    Hierarchy(None, TextContent("text1", [], []))
                                                ])
                                  ])
                    ]
                ),
                Hierarchy(
                    "headline2",
                    TextContent("text3", [RefSpan(2, 20, 200)], [RefSpan(0, 20, 200)])
                ),
            ]
        )

        self.assertEqual(2, len(hierarchy.get_part(re.compile(r".*headline.*"))))
        self.assertEqual(0, len(hierarchy.get_part(re.compile(r".*noneheadline[12].*"))))
        self.assertEqual(0, len(hierarchy.get_part(re.compile(r".*noneheadline3.*"))))
        self.assertEqual(2, len(hierarchy.get_part(re.compile(r".*headline[23].*"))))

        self.assertEqual(2, len(hierarchy.get_part(re.compile(r".*headline.*"), max_depth=1)))
        self.assertEqual(1, len(hierarchy.get_part(re.compile(r".*title.*"), max_depth=1)))
        self.assertEqual(0, len(hierarchy.get_part(re.compile(r".*title.*"), min_depth=1)))
        self.assertEqual(1, len(hierarchy.get_part(re.compile(r".*headline.*"), min_depth=2, max_depth=2)))

        # test paths
        paths = {x[1] for x in hierarchy.get_part(re.compile(r".*headline.*"), return_path=True)}
        self.assertEqual({(0,), (1,)}, paths)

        paths = {x[1] for x in hierarchy.get_part(re.compile(r".*headline3.*"), return_path=True)}
        self.assertEqual({(0, 1)}, paths)
