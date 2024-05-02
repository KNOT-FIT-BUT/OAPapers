# -*- coding: UTF-8 -*-
""""
Created on 07.08.22

:author:     Martin Dočekal
"""

from unittest import TestCase

from oapapers.cython.normalization import normalize_and_tokenize_string, normalize_authors, \
    convert_authors_to_initials_normalized_version, initial_and_normalized_authors, similarity_score, \
    similarity_scores_for_list_of_strings, normalize_multiple_strings


class TestNormalizeString(TestCase):
    def test_normalize_string(self):
        self.assertListEqual(["name", "surname"], normalize_and_tokenize_string("Name Surname"))
        self.assertListEqual(["name", "surname"], normalize_and_tokenize_string("Name,Surname"))
        self.assertListEqual(["martin", "docekal"], normalize_and_tokenize_string("Martin Dočekal"))
        self.assertListEqual(["timerman"], normalize_and_tokenize_string("Timmermann"))


class TestNormalizeMultipleStrings(TestCase):
    def test_normalize_multiple_strings(self):
        self.assertListEqual(
            [
                "name surname", "name surname", "martin docekal", "timerman"
            ],
            normalize_multiple_strings(["Name Surname", "Name,Surname", "Martin Dočekal", "Timmermann"])
        )


class TestNormalizeAuthors(TestCase):

    def test_normalize_authors(self):
        authors = ["Martin Dočekal", "Vlad,the Impaler", "Rosseinsky D.R."]

        normalized = normalize_authors(authors)

        self.assertSequenceEqual(
            [
                {"martin", "docekal"}, {"vlad", "the", "impaler"},
                {"roseinsky", "d", "r"},
            ], normalized)


    def test_convert_authors_to_initials_normalized_version(self):
        authors = ["Xiu", "Martin Dočekal", "Vlad,the Impaler", "John Ronald Reuel Tolkien"]
        converted = convert_authors_to_initials_normalized_version(authors)

        self.assertSequenceEqual(
            [
                {"xiu"}, {"m", "docekal"}, {"v", "t", "impaler"},
                {"j", "r", "r", "tolkien"},
            ], converted)

    def test_initial_and_normalized_authors(self):
        authors = ["Xiu", "Martin Dočekal", "Vlad,the Impaler", "John Ronald Reuel Tolkien"]
        init_ver, norm_ver = initial_and_normalized_authors(authors)

        self.assertSequenceEqual(
            [
                {"xiu"}, {"m", "docekal"}, {"v", "t", "impaler"},
                {"j", "r", "r", "tolkien"},
            ], init_ver)

        self.assertSequenceEqual(
            [
                {"xiu"}, {"martin", "docekal"}, {"vlad", "the", "impaler"},
                {"john", "ronald", "reuel", "tolkien"},
            ], norm_ver)

    def test_normalize_problematic(self):
        authors = ["Xiu", "Martin Dočekal", "(/)", ""]
        init_ver, norm_ver = initial_and_normalized_authors(authors)

        self.assertSequenceEqual(
            [
                {"xiu"}, {"m", "docekal"}, {"(/)"}, set([])
            ], init_ver)

        self.assertSequenceEqual(
            [
                {"xiu"}, {"martin", "docekal"}, {"(/)"}, set([])
            ], norm_ver)


class TestSimilarityScore(TestCase):

    def test_similarity_score_empty(self):
        self.assertEqual(0, similarity_score(set(), set()))
        self.assertEqual(0, similarity_score({"a", "b"}, set()))
        self.assertEqual(0, similarity_score(set(), {"a", "b"}))

    def test_similarity_score_non_empty(self):
        self.assertEqual(1.0, similarity_score({"a", "b"}, {"a", "b"}))
        self.assertAlmostEqual(0.666666667, similarity_score({"a"}, {"a", "b"}))

    def test_similarity_score_no_intersection(self):
        self.assertEqual(0.0, similarity_score({"o", "l"}, {"a", "b"}))


class TestSimilarityScoresForListOfStrings(TestCase):

    def test_similarity_scores_for_list_of_strings(self):
        res = similarity_scores_for_list_of_strings("a b", ["a b", "a", "o l"])
        self.assertEqual(3, len(res))
        self.assertAlmostEqual(1.0, res[0])
        self.assertAlmostEqual(0.666666667, res[1])
        self.assertAlmostEqual(0.0, res[2])
