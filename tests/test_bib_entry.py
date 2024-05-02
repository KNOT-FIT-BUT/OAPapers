# -*- coding: UTF-8 -*-
"""
Created on 15.03.23

:author:     Martin Dočekal
"""
from unittest import TestCase

from oapapers.bib_entry import Bibliography, BibEntry


class TestBibliography(TestCase):

    def setUp(self) -> None:
        # let's fill the bibliography with several paper and authors
        self.bib = Bibliography([
            BibEntry(id=1, title="Philosophical talk about nothing", year=2021, authors=("Xiu", "Martin Dočekal", "Vlad,the Impaler", "J R R Tolkien")),
            BibEntry(id=2, title="On the benefits of mirrors", year=1890, authors=("Dorian Gray",)),
            BibEntry(id=3, title="Fire, the greatest element", year=68, authors=("Nero",)),
            BibEntry(id=2, title="On the benefits of AI", year=2030, authors=("Dorian Gray",)),
        ])

    def test_index_exact(self):
        self.assertEqual(self.bib.index("Philosophical talk about nothing", ("Xiu", "Martin Dočekal", "Vlad,the Impaler", "J R R Tolkien"), 2021), 0)
        self.assertEqual(self.bib.index("On the benefits of mirrors", ("Dorian Gray",), 1890), 1)
        self.assertEqual(self.bib.index("Fire, the greatest element", ("Nero",), 68), 2)

    def test_index_unknown(self):
        with self.assertRaises(ValueError):
            self.bib.index("Unknown title", ("Unknown author",), 0)

        with self.assertRaises(ValueError):
            self.bib.index(None, ["Dorian Gray"], 1880, elimination_method=False)

        with self.assertRaises(ValueError):
            self.bib.index("On the benefits of", ("Dorian Gray",), 1880)

    def test_index_similar(self):
        self.assertEqual(self.bib.index(None, ["M Docekal"], 2021), 0)
        self.assertEqual(self.bib.index(None, ["Dorian Gray"], 1890), 1)
        self.assertEqual(self.bib.index(None, ["Nero"], 68), 2)

        # with title
        self.assertEqual(self.bib.index("Philosophical talk about", ["John Ronald Reuel Tolkien"], 2021), 0)
        self.assertEqual(self.bib.index("the benefits of mirrors", ["D Gray"], 1890), 1)
        self.assertEqual(self.bib.index("Fire, the element", ["Nero"], 68), 2)

    def test_add_append_and_index(self):
        self.bib.append(BibEntry(id=4, title="There is no such thing as bad publicity", year=1945,
                                 authors=("Joseph", "Stalin")))
        self.assertEqual(4, self.bib.index("There is no such thing as bad publicity", ("Joseph", "Stalin"), 1945))



