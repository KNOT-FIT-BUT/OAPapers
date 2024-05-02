# -*- coding: UTF-8 -*-
"""
Created on 15.03.23

:author:     Martin Dočekal
"""
from collections import Counter
from unittest import TestCase

from oapapers.similarities import similarity_score, dice_similarity_score, containment_score


class Test(TestCase):

    def test_similarity_score_empty(self):
        self.assertEqual(0, similarity_score(set(), set()))
        self.assertEqual(0, similarity_score({"a", "b"}, set()))
        self.assertEqual(0, similarity_score(set(), {"a", "b"}))

    def test_similarity_score_non_empty(self):
        self.assertEqual(1.0, similarity_score({"a", "b"}, {"a", "b"}))
        self.assertAlmostEqual(0.666666667, similarity_score({"a"}, {"a", "b"}))

    def test_similarity_score_no_intersection(self):
        self.assertEqual(0.0, similarity_score({"o", "l"}, {"a", "b"}))

    def test_containment_score_empty(self):
        self.assertEqual(0, containment_score(set(), set()))
        self.assertEqual(0, containment_score({"a", "b"}, set()))
        self.assertEqual(0, containment_score(set(), {"a", "b"}))

    def test_containment_score_non_empty(self):
        self.assertEqual(1.0, containment_score({"a", "b"}, {"a", "b"}))
        self.assertAlmostEqual(1.0, containment_score({"a"}, {"a", "b"}))
        self.assertAlmostEqual(1.0, containment_score({"a", "b"}, {"a"}))
        self.assertAlmostEqual(0.5, containment_score({"d", "b"}, {"a", "b", "c"}))
        self.assertAlmostEqual(1.0, containment_score({"c", "b", "a"}, {"a", "b", "c"}))
        self.assertAlmostEqual(1/3, containment_score({"d", "a", "e"}, {"a", "b", "c"}))

    def test_containment_score_no_intersection(self):
        self.assertEqual(0.0, containment_score({"o", "l"}, {"a", "b"}))

    def test_dice_similarity_score_empty(self):
        self.assertEqual(0, dice_similarity_score(Counter(), Counter()))
        self.assertEqual(0, dice_similarity_score(Counter({"a": 1, "b": 1}), Counter()))
        self.assertEqual(0, dice_similarity_score(Counter(), Counter({"a": 1, "b": 1})))

    def test_dice_similarity_score_non_empty(self):
        self.assertEqual(1.0, dice_similarity_score(Counter({"a": 1, "b": 1}), Counter({"a": 1, "b": 1})))
        self.assertAlmostEqual(0.666666667, dice_similarity_score(Counter({"a": 1}), Counter({"a": 1, "b": 1})))

    def test_dice_similarity_score_no_intersection(self):
        self.assertEqual(0.0, dice_similarity_score(Counter({"o": 2, "l": 1}), Counter({"a": 1, "b": 1})))
