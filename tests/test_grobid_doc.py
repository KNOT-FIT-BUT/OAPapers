# -*- coding: UTF-8 -*-
""""
Created on 16.05.22

:author:     Martin Dočekal
"""
import copy
import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copyfile, rmtree
from unittest import TestCase

import grobid_files_contents
from oapapers.grobid_doc import GROBIDMetaDoc, GROBIDDoc, remove_prefix
from oapapers.papers_list import PapersList, PapersListRecord

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_PATH = os.path.join(SCRIPT_PATH, "tmp")

GROBID_PATH = os.path.join(SCRIPT_PATH, "fixtures/grobid")

XML_615_FIXTURE_PATH = os.path.join(GROBID_PATH, "1/615.tei.xml")
XML_100085_FIXTURE_PATH = os.path.join(GROBID_PATH, "106/100085.tei.xml")

XML_615_TMP_PATH = os.path.join(TMP_PATH, "615.tei.xml")
XML_100085_TMP_PATH = os.path.join(TMP_PATH, "100085.tei.xml")


class TestBase(TestCase):
    def setUp(self) -> None:
        self.clear_tmp()

    def tearDown(self) -> None:
        self.clear_tmp()

    @staticmethod
    def clear_tmp():
        for f in Path(TMP_PATH).glob('*'):
            if not str(f).endswith("placeholder"):
                if f.is_dir():
                    rmtree(f)
                else:
                    os.remove(f)


class TestGROBIDMetaDoc(TestBase):
    def setUp(self) -> None:
        super(TestGROBIDMetaDoc, self).setUp()
        copyfile(XML_615_FIXTURE_PATH, XML_615_TMP_PATH)
        copyfile(XML_100085_FIXTURE_PATH, XML_100085_TMP_PATH)

    def test_all_from_path(self):
        meta_doc = GROBIDMetaDoc(XML_100085_TMP_PATH)
        self.assertEqual("DIAMOND STORAGE RING APERTURES", meta_doc.title)
        self.assertIsNone(meta_doc.year)
        self.assertListEqual(["N Wyles", "J Jones", "H Owen", "J Varley"], list(meta_doc.authors))

        meta_doc = GROBIDMetaDoc(XML_615_TMP_PATH)
        self.assertIsNone(meta_doc.title)
        self.assertEqual(2003, meta_doc.year)
        self.assertListEqual([], list(meta_doc.authors))

    def test_all_from_element(self):
        tree = ET.parse(XML_100085_TMP_PATH)
        source = tree.getroot()

        meta_doc = GROBIDMetaDoc(source)
        self.assertEqual("DIAMOND STORAGE RING APERTURES", meta_doc.title)
        self.assertIsNone(meta_doc.year)
        self.assertListEqual(["N Wyles", "J Jones", "H Owen", "J Varley"], list(meta_doc.authors))

    def test_year_out_of_interval(self):
        tree = ET.parse(XML_615_TMP_PATH)
        source = tree.getroot()
        node = source.find(f"{GROBIDMetaDoc.PREFIX}teiHeader/{GROBIDMetaDoc.PREFIX}fileDesc/"
                           f"{GROBIDMetaDoc.PREFIX}publicationStmt/{GROBIDMetaDoc.PREFIX}date")
        node.text = "1215"
        node.attrib["when"] = "1215"
        meta_doc = GROBIDMetaDoc(source)
        self.assertIsNone(meta_doc.title)
        self.assertIsNone(meta_doc.year)
        self.assertListEqual([], list(meta_doc.authors))


class TestStaticGROBIDMetaDoc(TestCase):
    def test_parse_name(self):
        pers_name = ET.fromstring("""
        <persName xmlns="http://www.tei-c.org/ns/1.0"><forename type="first">J</forename><surname>Jones</surname></persName>
        """)
        res_name = GROBIDMetaDoc.parse_name(pers_name)
        self.assertEqual("J Jones", res_name)

    def test_parse_name_additional_field(self):
        pers_name = ET.fromstring("""
                <persName xmlns="http://www.tei-c.org/ns/1.0">
                    <forename type="first">Harris</forename>
                    <forename type="middle">F</forename>
                    <roleName>Farmer J</roleName>
                </persName>
                """)
        res_name = GROBIDMetaDoc.parse_name(pers_name)
        self.assertEqual("Harris F", res_name)

    def test_parse_name_missing_fields(self):
        pers_name = ET.fromstring("""
                <persName xmlns="http://www.tei-c.org/ns/1.0">
                    <roleName>Farmer J</roleName>
                </persName>
                """)
        res_name = GROBIDMetaDoc.parse_name(pers_name)
        self.assertEqual("", res_name)


class TestGROBIDDoc(TestBase):
    def setUp(self) -> None:
        super(TestGROBIDDoc, self).setUp()
        copyfile(XML_100085_FIXTURE_PATH, XML_100085_TMP_PATH)
        self.papers_list = PapersList([
            PapersListRecord("DIAMOND STORAGE RING APERTURES", None, ["N Wyles", "J Jones", "H Owen", "J Varley"]),
            PapersListRecord("Beam Lifetime Studies for the SLS Storage Ring", 1999, ["M Böge", "A Streun"]),
            PapersListRecord("Apertures for Injection", None, ["S Tazzari"])]
        )

    def test_doc(self):
        doc = GROBIDDoc(XML_100085_TMP_PATH)

        self.assertEqual(grobid_files_contents.grobid_titles[0], doc.title)
        self.assertEqual(grobid_files_contents.grobid_years_not_matched[0], doc.year)
        self.assertSequenceEqual(grobid_files_contents.grobid_authors[0], list(doc.authors))
        self.assertEqual(grobid_files_contents.grobid_hierarchy_not_matched[0], doc.hierarchy)
        self.assertListEqual(grobid_files_contents.grobid_non_plaintext[0], doc.non_plaintext_content)
        self.assertEqual(grobid_files_contents.grobid_bibliography_not_matched[0], doc.bibliography)

    def test_doc_stub(self):
        doc = GROBIDDoc(XML_100085_TMP_PATH, True)

        self.assertEqual(grobid_files_contents.grobid_titles[0], doc.title)
        self.assertEqual(grobid_files_contents.grobid_years_not_matched[0], doc.year)
        self.assertSequenceEqual(grobid_files_contents.grobid_authors[0], list(doc.authors))
        hier = copy.deepcopy(grobid_files_contents.grobid_hierarchy_not_matched[0])
        hier.content = hier.content[:1]
        self.assertEqual(hier, doc.hierarchy)
        self.assertListEqual(grobid_files_contents.grobid_non_plaintext[0], doc.non_plaintext_content)
        self.assertEqual(grobid_files_contents.grobid_bibliography_not_matched[0], doc.bibliography)

    def test_doc_matched(self):
        doc = GROBIDDoc(XML_100085_TMP_PATH)
        doc.match_bibliography(0, self.papers_list)
        self.assertEqual(grobid_files_contents.title_100085, doc.title)
        self.assertEqual(grobid_files_contents.grobid_years_not_matched[0], doc.year) # the year is matched by mag
        self.assertSequenceEqual(grobid_files_contents.authors_100085, list(doc.authors))
        self.assertEqual(grobid_files_contents.hierarchy_100085, doc.hierarchy)
        self.assertListEqual(grobid_files_contents.non_plaintext_100085, doc.non_plaintext_content)
        self.assertEqual(grobid_files_contents.bibliography_100085, doc.bibliography)


class TestRemovePrefix(TestCase):
    def test_remove_prefix(self):
        self.assertEqual("string without prefix", remove_prefix("string without prefix", "prefix"))
        self.assertEqual(" without prefix", remove_prefix("string without prefix", "string"))


if __name__ == '__main__':
    unittest.main()

