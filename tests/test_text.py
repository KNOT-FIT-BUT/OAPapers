# -*- coding: UTF-8 -*-
"""
Created on 10.03.23

:author:     Martin Dočekal
"""
import re
from unittest import TestCase

from oapapers.text import DeHyphenator, replace_at, SpanCollisionHandling, clean_title


class TestDeHyphenator(TestCase):
    def setUp(self):
        self.dehyphenator = DeHyphenator("en_US")

    def test_sub(self):
        self.assertEqual("measurements", self.dehyphenator.sub(re.match(r"(\w+)-\s+(\w+)", "mea- surements")))
        self.assertEqual("T- minutes", self.dehyphenator.sub(re.match(r"(\w+)-\s+(\w+)", "T- minutes")))

    def test_replacements(self):
        spans, replacements = self.dehyphenator.replacements("mea- surements and more anal- ysis and Be- low and incor- porating")
        self.assertSequenceEqual([(0, 14), (24, 34), (39, 46), (51, 66)], spans)
        self.assertSequenceEqual(["measurements", "analysis", "Below", "incorporating"], replacements)

    def test_call(self):
        self.assertEqual("measurements", self.dehyphenator("mea- surements"))
        self.assertEqual("T- minutes", self.dehyphenator("T- minutes"))
        self.assertEqual("mutli-document summarization", self.dehyphenator("mutli-document summarization"))

    def test_call_with_more(self):
        self.assertEqual("measurements and more analysis and Below and incorporating", self.dehyphenator("mea- surements and more anal- ysis and Be- low and incor- porating"))


class TestReplaceAt(TestCase):

    def test_replace_raises(self):
        with self.assertRaises(ValueError):
            replace_at("text for testing", [(5, 8)], ["replaced", "replaced"], [])

        with self.assertRaises(ValueError):
            replace_at("text for testing", [(5, 8), (6, 9)], ["replaced", "replaced"], [])

    def test_replace_at_no_spans(self):
        text, spans = replace_at("text for testing", [(5, 8)], ["replaced"], [])
        self.assertEqual("text replaced testing", text)
        self.assertSequenceEqual([], spans)

    def test_replace_at_no_collision(self):
        text, spans = replace_at("context with citations [1] and reference 2", [(0, 7), (31, 40)], ["replaced", "some"], [[(23, 26)], [(41, 42)]])
        self.assertEqual("replaced with citations [1] and some 2", text)
        self.assertSequenceEqual([[(24, 27)], [(37, 38)]], spans)

    def test_replace_at_collision_skip(self):
        text, spans = replace_at("context with citations [1] and reference 2", [(0, 7), (24, 24)], ["replaced", "some"], [[(23, 26)], [(41, 42)]], SpanCollisionHandling.SKIP)
        self.assertEqual("replaced with citations [1] and reference 2", text)
        self.assertSequenceEqual([[(24, 27)], [(42, 43)]], spans)

    def test_replace_at_collision_remove(self):
        text, spans = replace_at("context with citations [1] and reference 2", [(0, 7), (24, 24)], ["replaced", "some"], [[(23, 26)], [(41, 42)]], SpanCollisionHandling.REMOVE)
        self.assertEqual("replaced with citations [some1] and reference 2", text)
        self.assertSequenceEqual([[None], [(46, 47)]], spans)

    def test_replace_at_collision_merge(self):
        text, spans = replace_at("context with citations [1] and reference 2", [(0, 7), (24, 24)], ["replaced", "some"], [[(23, 26)], [(41, 42)]], SpanCollisionHandling.MERGE)
        self.assertEqual("replaced with citations [some1] and reference 2", text)
        self.assertSequenceEqual([[(24, 31)], [(46, 47)]], spans)

        text, spans = replace_at("context with citations [1] and reference 2", [(0, 7), (23, 26)], ["replaced", "(Harvard, 2022)"], [[(23, 26)], [(41, 42)]], SpanCollisionHandling.MERGE)
        self.assertEqual("replaced with citations (Harvard, 2022) and reference 2", text)
        self.assertSequenceEqual([[(24, 39)], [(54, 55)]], spans)

    def test_replace_at_collision_raise(self):
        with self.assertRaises(ValueError):
            replace_at("context with citations [1] and reference 2", [(0, 7), (24, 24)], ["replaced", "some"], [[(23, 26)], [(41, 42)]], SpanCollisionHandling.RAISE)


class TestCleanTitle(TestCase):
    def test_clean_title_ok(self):
        self.assertEqual("On the problems of doi and its application", clean_title("On the problems of doi and its application"))
        self.assertEqual("GTP 3.0 and others 122.333", clean_title("GTP 3.0 and others 122.333"))

    def test_clean_title_with_invalid_postfix(self):
        self.assertEqual("Fast rhetorical structure theory discourse parsing",
                         clean_title("Fast rhetorical structure theory discourse parsing. CoRR, abs/1505.02425"))

        self.assertEqual("Active Temporal Multiplexing of Photons",
                         clean_title("Active Temporal Multiplexing of Photons. arXiv, 1503.01215"))

        self.assertEqual("Probabilistic analysis and extraction of video content",
                         clean_title("Probabilistic analysis and extraction of video content Doi: 10.1109/ICIP.1999.822861. 58. Naphade,Probabilistic multimedia objects (MULTIJECTS): a novel approach to video indexing and retrieval"))

        self.assertEqual("An axiomatic characterization of the Brownian map",
                         clean_title("An axiomatic characterization of the Brownian map, math.PR/1506.03806. [47] , Liouville quantum gravity and the Brownian map I: The QLE(8/3,0) metric"))

        self.assertEqual("Rabies-Tanzania (Serengeti National Park), civet, human exp., novel lyssavirus",
                            clean_title("Rabies-Tanzania (Serengeti National Park), civet, human exp., novel lyssavirus, Archive Number: 20120314.1070293. ProMED-mail"))