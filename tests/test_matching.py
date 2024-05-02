# -*- coding: UTF-8 -*-
"""
Created on 15.03.23

:author:     Martin Dočekal
"""
from collections import Counter
from unittest import TestCase

from oapapers.matching import match_authors, match_authors_groups


class Test(TestCase):
    def test_match_authors(self):
        # authors ["Xiu", "Martin Dočekal", "Vlad,the Impaler", "John Ronald Reuel Tolkien"]
        authors_normalized = [
            set(["xiu"]),
            set(["martin", "docekal"]),
            set(["vlad", "the", "impaler"]),
            set(["john", "ronald", "reuel", "tolkien"]),
        ]
        authors_normalized_initials = [
            set(["xiu"]),
            set(["m", "docekal"]),
            set(["v", "t", "impaler"]),
            set(["j", "r", "tolkien"]),
        ]

        self.assertTrue(match_authors(authors_normalized[0], authors_normalized_initials[0],
                                                 set(["xiu"]), set(["xiu"]), 1.0))
        self.assertFalse(match_authors(authors_normalized[0], authors_normalized_initials[0],
                                                  set(["karel"]), set(["karel"]), 1.0))

        self.assertTrue(match_authors(authors_normalized[1], authors_normalized_initials[1],
                                                 set(["m", "docekal"]),
                                                 set(["m", "docekal"]), 1.0))
        self.assertFalse(match_authors(authors_normalized[0], authors_normalized_initials[0],
                                                  set(["merlin", "docekal"]),
                                                  set(["m", "docekal"]), 1.0))

        self.assertTrue(match_authors(authors_normalized[3], authors_normalized_initials[3],
                                                 set(["j", "r", "tolkien"]),
                                                 set(["j", "r", "tolkien"]), 0.75))

        self.assertFalse(match_authors(authors_normalized[3], authors_normalized_initials[3],
                                                  set(["j", "a", "tolkien"]),
                                                  set(["j", "a", "tolkien"]), 0.75))

    def test_match_authors_groups(self):
        authors_normalized = [
            set(["xiu"]),
            set(["martin", "docekal"]),
            set(["vlad", "the", "impaler"]),
            set(["john", "ronald", "reuel", "tolkien"]),
        ]

        authors_normalized_initials = [
            set(["xiu"]),
            set(["m", "docekal"]),
            set(["v", "t", "impaler"]),
            set(["j", "r", "tolkien"]),
        ]

        self.assertTrue(match_authors_groups(authors_normalized, authors_normalized_initials,
                                                        [set(["xiu"])], [set(["xiu"])], 1.0))

        self.assertFalse(match_authors_groups(authors_normalized, authors_normalized_initials,
                                                            [set(["aladin"])], [set(["karel"])], 1.0))

        self.assertTrue(match_authors_groups(authors_normalized, authors_normalized_initials,
                                                        [set(["m", "docekal"])],
                                                        [set(["m", "docekal"])], 1.0))
