# -*- coding: UTF-8 -*-
""""
Created on 01.03.22

:author:     Martin Dočekal
"""
import argparse
import csv
import json
import multiprocessing
import os
import sys
import unittest
from io import StringIO
from pathlib import Path
from shutil import copyfile, rmtree, copytree
from typing import Set, List, Union
from unittest.mock import patch

from oapapers.__main__ import write_references, create_related_work, \
    filter_and_print, get_last_line, extend, extend_mag_with_core, enhance_papers_with_mag, \
    get_ids_of_filter_passing_documents, convert_dataset, convert_s2orc, convert_core, filter_oa_dataset_with_reviews, \
    filter_related_work, pruning_hier, deduplication, identify_bibliography, identify_citation_spans, \
    create_papers_2_mag_mapping, create_s2orc_metadata, enrich_bibliography_from_citation_graph
from oapapers.bib_entry import BibEntry
from oapapers.datasets import OADataset, OARelatedWork
from oapapers.document import Document
from oapapers.document_datasets import DocumentDataset
from oapapers.filters import Filter
from oapapers.hierarchy import Hierarchy, TextContent

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_PATH = os.path.join(SCRIPT_PATH, "tmp")

REFERENCES_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/references.jsonl")
REFERENCES_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/references.jsonl")

REFERENCES_FOR_FILTRATION_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/filtration/references.jsonl")
REFERENCES_FOR_FILTRATION_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/references.jsonl")

REVIEWS_FOR_FILTRATION_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/filtration/related_work.jsonl")
REVIEWS_FOR_FILTRATION_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/related_work.jsonl")

REFERENCES_FILTERED_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/filtration/references_filtered.jsonl")
REFERENCES_FILTERED_FIXTURES_PATH_2 = os.path.join(SCRIPT_PATH, "fixtures/references_filtered.jsonl")
REFERENCES_FILTERED_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/references_filtered.jsonl")

REVIEWS_FILTERED_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/filtration/related_work_filtered.jsonl")
REVIEWS_FILTERED_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/related_work_filtered.jsonl")

WRITE_REFERENCES_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/write_references.jsonl")
WRITE_REFERENCES_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/write_references.jsonl")

REVIEWS_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/reviews.jsonl")
REVIEWS_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/reviews.jsonl")

SCOPUS_REVIEW_LIST_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/scopus_review_list.jsonl")
SCOPUS_REVIEW_LIST_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/scopus_review_list.jsonl")

RELATED_WORK_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/related_work.jsonl")
RELATED_WORK_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/related_work.jsonl")

S2ORC_PAPER_FIXTURE_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc_papers.jsonl")
S2ORC_PAPER_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/s2orc_papers.jsonl")

REFERENCES_RELATED_WORK_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/references_related_work.jsonl")
REFERENCES_RELATED_WORK_ANOTHER_FIXTURES_PATH = os.path.join(SCRIPT_PATH,
                                                             "fixtures/references_related_work_another.jsonl")
REFERENCES_RELATED_WORK_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/references_related_work.jsonl")

EXTENDING_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/extending.jsonl")
EXTENDING_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/extending.jsonl")

FOR_EXTEND_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/for_extend.jsonl")
FOR_EXTEND_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/for_extend.jsonl")

EXTENDED_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/extended.jsonl")
EXTENDED_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/extended.jsonl")

CONVERTED_PAPERS_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/converted_papers.jsonl")
CONVERTED_PAPERS_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/converted_papers.jsonl")

PAPERS_FOR_CONVERSION_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/papers_for_conversion.jsonl")
PAPERS_FOR_CONVERSION_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/papers_for_conversion.jsonl")

PAPERS_FOR_RELATED_WORK_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/papers_for_related_work.jsonl")
PAPERS_FOR_RELATED_WORK_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/papers_for_related_work.jsonl")

PAPERS_FOR_RELATED_WORK_REFERENCES_FIXTURES_PATH = os.path.join(SCRIPT_PATH,
                                                                "fixtures/papers_for_related_work_references.jsonl")
PAPERS_FOR_RELATED_WORK_REFERENCES_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/papers_for_related_work_references.jsonl")

FOR_PRUNING_PAPERS_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/for_pruning_papers.jsonl")
FOR_PRUNING_PAPERS_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/for_pruning_papers.jsonl")

PRUNED_PAPERS_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/pruned_papers.jsonl")
PRUNED_PAPERS_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/pruned_papers.jsonl")


class BaseTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        self.stdout = StringIO()
        self.stderr = StringIO()

        sys.stdout = self.stdout
        sys.stderr = self.stderr

        self.maxDiff = None

    def tearDown(self) -> None:
        super().tearDown()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.clear_tmp()

    @staticmethod
    def clear_tmp():
        for f in Path(TMP_PATH).glob('*'):
            if not str(f).endswith("placeholder"):
                if f.is_dir():
                    rmtree(f)
                else:
                    os.remove(f)

    def make_unordered(self, record: Union[dict, list]) -> Union[dict, list]:
        """
        Converts all lists to sets.

        :param record: record to convert
        :return: converted record
        """
        def convert_var(v):
            if isinstance(v, list):
                return frozenset(convert_var(x) for x in v)
            elif isinstance(v, dict):
                return {k: convert_var(x) for k, x in v.items()}
            else:
                return v

        if isinstance(record, dict):
            return {k: convert_var(v) for k, v in record.items()}

        return [convert_var(x) for x in record]

    def compare_json(self, gt: str, res: str, unordered: bool = False):
        """
        Checks two json files that they have the same content.

        :param gt: path to ground truth file
        :param res: path to file with results
        :param unordered: All lists are changed to sets
        """

        with open(gt, "r") as gt_f, open(res, "r") as res_f:
            gt = json.load(gt_f)
            res = json.load(res_f)

            if unordered:
                gt = self.make_unordered(gt)
                res = self.make_unordered(res)

            self.assertEqual(gt, res)

    def compare_jsonl(self, gt: str, res: str, unordered: bool = False):
        """
        Checks two jsonl files that they have the same content.

        :param gt: path to ground truth file
        :param res: path to file with results
        :param unordered: All lists are changed to sets
        """
        with open(gt, "r") as gt_f, open(res, "r") as res_f:
            if unordered:
                def convert(line):
                    def convert_var(v):
                        if isinstance(v, list):
                            return frozenset(convert_var(x) for x in v)
                        elif isinstance(v, dict):
                            return frozenset((k, convert_var(x)) for k, x in v.items())
                        else:
                            return v

                    return {k: convert_var(v) for k, v in json.loads(line).items()}

                for i, (gt, res) in enumerate(zip(gt_f.readlines(), res_f.readlines())):
                    self.assertDictEqual(convert(gt), convert(res), msg=f"Problem with line {i}")
            else:
                gt_lines = [json.loads(line) for line in gt_f.readlines()]
                res_lines = [json.loads(line) for line in res_f.readlines()]

                self.assertListEqual(gt_lines, res_lines)

    def check_index(self, path_to_index: str, path_to_indexed_file: str, id_in_jsonl: str = "id"):
        index = []
        with open(path_to_index, newline='') as f:
            for r in csv.DictReader(f, delimiter="\t"):
                index.append((int(r["file_line_offset"]), int(r["key"])))

        with open(path_to_indexed_file, "r") as f:
            lines = f.readlines()

            self.assertEqual(len(lines), len(index))

            for i, (line, (line_offset, key)) in enumerate(zip(lines, index)):
                f.seek(line_offset)
                from_offset_line = f.readline()

                self.assertEqual(line, from_offset_line, msg=f"The lines differ for item {i}.")
                self.assertEqual(json.loads(line)[id_in_jsonl], key)


class MockFilter(Filter):
    def __init__(self, allowed_ids):
        self.allowed_ids = allowed_ids
        self.called = False

    def __call__(self, document: Document) -> bool:
        return document.id in self.allowed_ids


class TestGetIdsIfFilterPassingReferencesDataset(BaseTestCase):
    def setUp(self) -> None:
        super(TestGetIdsIfFilterPassingReferencesDataset, self).setUp()

        copyfile(REFERENCES_FIXTURES_PATH, REFERENCES_TMP_PATH)
        copyfile(REFERENCES_FIXTURES_PATH + ".index", REFERENCES_TMP_PATH + ".index")
        self.oa_dataset = OADataset(REFERENCES_TMP_PATH, REFERENCES_TMP_PATH + ".index").open()

    def tearDown(self) -> None:
        super(TestGetIdsIfFilterPassingReferencesDataset, self).tearDown()
        self.oa_dataset.close()

    def test_get_ids_of_filter_passing_references(self):
        ids = get_ids_of_filter_passing_documents(self.oa_dataset, MockFilter([119311037]))
        self.assertEqual([119311037], ids)


class MockDocumentDataset(DocumentDataset):
    def __init__(self, docs: List[Document]):
        super().__init__()
        self.documents = docs

    def __len__(self):
        return len(self.documents)

    def _get_item(self, item: int) -> Document:
        d = self.documents[item]
        return d if self.transform is None else self.transform(d)


class TestConvertDataset(BaseTestCase):
    def setUp(self) -> None:
        super(TestConvertDataset, self).setUp()
        self.dataset = MockDocumentDataset([
            Document(
                id=3, s2orc_id=3, mag_id=10, doi=None,
                title="Title 0", authors=["Author A", "Author B"], year=2022,
                fields_of_study=["computer science", "NLP"], citations=[4, 5, 0],
                hierarchy=Hierarchy(
                    "Title 0",
                    [
                        Hierarchy(
                            "Section 0",
                            [
                                Hierarchy(None,
                                          [
                                              Hierarchy(None, TextContent("sentence 0", [], []))
                                          ]
                                          )
                            ]
                        ),
                    ]
                ),
                bibliography=[
                    BibEntry(None, "Title 1", 2020, ("Author 0", "Author 1")),
                    BibEntry(None, "Unknown Title", 2000, ("Unknown Author 0", "Unknown Author 1"))
                ],
                non_plaintext_content=[("figure", "figure 0 decription")],
                uncategorized_fields={
                    "origin": "core"
                }
            ),
            Document(
                id=4, s2orc_id=4, mag_id=11, doi=None,
                title="Title 1", authors=["Author A", "Author B"], year=2020,
                fields_of_study=[], citations=[],
                hierarchy=Hierarchy(
                    "Title 1",
                    [
                        Hierarchy(
                            "Section 0",
                            [
                                Hierarchy(None,
                                          [
                                              Hierarchy(None, TextContent("sentence 1", [], []))
                                          ]
                                          )
                            ]
                        ),
                    ]
                ),
                bibliography=[],
                non_plaintext_content=[],
                uncategorized_fields={
                    "origin": "core"
                }
            ),
            Document(
                id=5, s2orc_id=50, mag_id=5, doi=None, title="Title 3", authors=["Author A", "Author B"], year=2019,
                fields_of_study=[], citations=[],
                hierarchy=Hierarchy("Title 3", []),
                bibliography=[],
                non_plaintext_content=[],
                uncategorized_fields={
                    "origin": "core"
                }
            ),
        ])

    def test_conversions(self):
        convert_dataset(self.dataset, CONVERTED_PAPERS_TMP_PATH)
        self.compare_jsonl(CONVERTED_PAPERS_FIXTURES_PATH, CONVERTED_PAPERS_TMP_PATH)
        self.check_index(CONVERTED_PAPERS_TMP_PATH + ".index", CONVERTED_PAPERS_TMP_PATH)


S2ORC_METADATA_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc/metadata.jsonl")
S2ORC_METADATA_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/metadata.jsonl")

S2ORC_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc/s2orc.jsonl")
S2ORC_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/s2orc.jsonl")

S2ORC_CONVERTED_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc/converted.jsonl")

class TestConvertS2ORC(BaseTestCase):
    def setUp(self) -> None:
        super(TestConvertS2ORC, self).setUp()
        copyfile(S2ORC_METADATA_FIXTURES_PATH, S2ORC_METADATA_TMP_PATH)
        copyfile(S2ORC_METADATA_FIXTURES_PATH + ".index", S2ORC_METADATA_TMP_PATH + ".index")
        copyfile(S2ORC_FIXTURES_PATH, S2ORC_TMP_PATH)
        copyfile(S2ORC_FIXTURES_PATH + ".index", S2ORC_TMP_PATH + ".index")

        self.args = argparse.Namespace(**{
            "metadata": S2ORC_METADATA_TMP_PATH,
            "s2orc": S2ORC_TMP_PATH,
            "result": CONVERTED_PAPERS_TMP_PATH,
            "workers": 0,
            "gpu": False,
            "from_i": 0,
            "to_i": None,
        })

    def test_conversions(self):
        convert_s2orc(self.args)
        self.compare_jsonl(S2ORC_CONVERTED_FIXTURES_PATH, CONVERTED_PAPERS_TMP_PATH)
        self.check_index(CONVERTED_PAPERS_TMP_PATH + ".index", CONVERTED_PAPERS_TMP_PATH)

    def test_conversion_multiproc(self):
        if multiprocessing.cpu_count() <= 1:
            self.skipTest("There is not enough cpus.")
            return
        self.args.workers = -1
        convert_s2orc(self.args)
        self.compare_jsonl(S2ORC_CONVERTED_FIXTURES_PATH, CONVERTED_PAPERS_TMP_PATH)
        self.check_index(CONVERTED_PAPERS_TMP_PATH + ".index", CONVERTED_PAPERS_TMP_PATH)


GROBID_FOR_CONVERSION_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/grobid_for_conversion")
GROBID_FOR_CONVERSION_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/grobid_for_conversion")

MAG_FOR_CONVERSION_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/mag_for_core_convert.jsonl")
MAG_FOR_CONVERSION_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/mag_for_core_convert.jsonl")

S2ORC_FOR_CONVERSION_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc_for_core_convert.jsonl")
S2ORC_FOR_CONVERSION_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/s2orc_for_core_convert.jsonl")

S2ORC_ABSTRACTS_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc/abstracts.jsonl")
S2ORC_ABSTRACTS_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/abstracts.jsonl")

S2ORC_CITATION_GRAPH_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc/citation_graph.json")
S2ORC_CITATION_GRAPH_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/citation_graph.json")

S2ORC_PAPERS_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc/papers.jsonl")
S2ORC_PAPERS_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/papers.jsonl")

S2ORC_METADATA_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/s2orc/metadata.jsonl")
S2ORC_METADATA_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/metadata.jsonl")


class TestCreateS2ORCMetadata(BaseTestCase):
    def setUp(self) -> None:
        super(TestCreateS2ORCMetadata, self).setUp()
        self.clear_tmp()
        copyfile(S2ORC_ABSTRACTS_FIXTURES_PATH, S2ORC_ABSTRACTS_TMP_PATH)
        copyfile(S2ORC_ABSTRACTS_FIXTURES_PATH+".index", S2ORC_ABSTRACTS_TMP_PATH+".index")

        copyfile(S2ORC_CITATION_GRAPH_FIXTURES_PATH, S2ORC_CITATION_GRAPH_TMP_PATH)

        copyfile(S2ORC_PAPERS_FIXTURES_PATH, S2ORC_PAPERS_TMP_PATH)
        copyfile(S2ORC_PAPERS_FIXTURES_PATH+".index", S2ORC_PAPERS_TMP_PATH+".index")

        self.args = argparse.Namespace(**{
            "abstracts": S2ORC_ABSTRACTS_TMP_PATH,
            "citation_graph": S2ORC_CITATION_GRAPH_TMP_PATH,
            "papers": S2ORC_PAPERS_TMP_PATH,
            "result": S2ORC_METADATA_TMP_PATH,
            "workers": 0,
            "gpu": False
        })

    def test_conversions(self):
        create_s2orc_metadata(self.args)
        self.compare_jsonl(S2ORC_METADATA_FIXTURES_PATH, S2ORC_METADATA_TMP_PATH)
        self.check_index(S2ORC_METADATA_TMP_PATH + ".index", S2ORC_METADATA_TMP_PATH, id_in_jsonl="corpusid")


class TestConvertCORE(BaseTestCase):
    def setUp(self) -> None:
        super(TestConvertCORE, self).setUp()
        self.clear_tmp()
        copyfile(MAG_FOR_CONVERSION_FIXTURES_PATH, MAG_FOR_CONVERSION_TMP_PATH)
        copyfile(MAG_FOR_CONVERSION_FIXTURES_PATH + ".index", MAG_FOR_CONVERSION_TMP_PATH + ".index")
        copyfile(S2ORC_FOR_CONVERSION_FIXTURES_PATH, S2ORC_FOR_CONVERSION_TMP_PATH)
        copyfile(S2ORC_FOR_CONVERSION_FIXTURES_PATH + ".index", S2ORC_FOR_CONVERSION_TMP_PATH + ".index")
        copytree(GROBID_FOR_CONVERSION_FIXTURES_PATH, GROBID_FOR_CONVERSION_TMP_PATH)

        self.args = argparse.Namespace(**{
            "original": GROBID_FOR_CONVERSION_TMP_PATH,
            "mag": MAG_FOR_CONVERSION_TMP_PATH,
            "s2orc": S2ORC_FOR_CONVERSION_TMP_PATH,
            "result": CONVERTED_PAPERS_TMP_PATH,
            "batch": 128,
            "workers": 0,
            "gpu": False
        })

    def test_conversions(self):
        convert_core(self.args)
        self.compare_jsonl(CONVERTED_PAPERS_FIXTURES_PATH, CONVERTED_PAPERS_TMP_PATH)
        self.check_index(CONVERTED_PAPERS_TMP_PATH + ".index", CONVERTED_PAPERS_TMP_PATH)

    def test_conversion_multiproc(self):
        if multiprocessing.cpu_count() <= 1:
            self.skipTest("There is not enough cpus.")
            return
        self.args.workers = -1
        convert_core(self.args)
        self.compare_jsonl(CONVERTED_PAPERS_FIXTURES_PATH, CONVERTED_PAPERS_TMP_PATH)
        self.check_index(CONVERTED_PAPERS_TMP_PATH + ".index", CONVERTED_PAPERS_TMP_PATH)


class TestWriteReferences(BaseTestCase):
    def setUp(self) -> None:
        super(TestWriteReferences, self).setUp()
        copyfile(PAPERS_FOR_RELATED_WORK_FIXTURES_PATH, PAPERS_FOR_RELATED_WORK_TMP_PATH)

    def test_write_references(self):
        with OADataset(PAPERS_FOR_RELATED_WORK_TMP_PATH, PAPERS_FOR_RELATED_WORK_TMP_PATH+".index") as dataset:
            write_references(WRITE_REFERENCES_TMP_PATH, dataset, {119311037, 119311039}, {119311039})

            self.compare_jsonl(WRITE_REFERENCES_FIXTURES_PATH, WRITE_REFERENCES_TMP_PATH)


class TestCreateRelatedWork(BaseTestCase):
    def setUp(self) -> None:
        super(TestCreateRelatedWork, self).setUp()
        self.args = argparse.Namespace(**{
            "documents": PAPERS_FOR_RELATED_WORK_TMP_PATH,
            "reviews": RELATED_WORK_TMP_PATH,
            "references": REFERENCES_RELATED_WORK_TMP_PATH,
            "references_dataset": None,
            "workers": 0,
            "unordered": False,
        })

        copyfile(PAPERS_FOR_RELATED_WORK_FIXTURES_PATH, PAPERS_FOR_RELATED_WORK_TMP_PATH)
        copyfile(PAPERS_FOR_RELATED_WORK_FIXTURES_PATH + ".index", PAPERS_FOR_RELATED_WORK_TMP_PATH + ".index")

    def test_create_related_work(self):
        create_related_work(self.args)

        self.compare_jsonl(RELATED_WORK_FIXTURES_PATH, RELATED_WORK_TMP_PATH, unordered=True)
        self.compare_jsonl(REFERENCES_RELATED_WORK_FIXTURES_PATH, REFERENCES_RELATED_WORK_TMP_PATH, unordered=True)

        self.check_index(RELATED_WORK_TMP_PATH + ".index", RELATED_WORK_TMP_PATH)
        self.check_index(REFERENCES_RELATED_WORK_TMP_PATH + ".index", REFERENCES_RELATED_WORK_TMP_PATH)

    def test_create_related_work_with_different_references(self):
        self.args.references_dataset = PAPERS_FOR_RELATED_WORK_REFERENCES_TMP_PATH
        copyfile(PAPERS_FOR_RELATED_WORK_REFERENCES_FIXTURES_PATH, PAPERS_FOR_RELATED_WORK_REFERENCES_TMP_PATH)
        copyfile(PAPERS_FOR_RELATED_WORK_REFERENCES_FIXTURES_PATH + ".index",
                 PAPERS_FOR_RELATED_WORK_REFERENCES_TMP_PATH + ".index")
        create_related_work(self.args)

        self.compare_jsonl(RELATED_WORK_FIXTURES_PATH, RELATED_WORK_TMP_PATH, unordered=True)
        self.compare_jsonl(REFERENCES_RELATED_WORK_ANOTHER_FIXTURES_PATH, REFERENCES_RELATED_WORK_TMP_PATH, unordered=True)

        self.check_index(RELATED_WORK_TMP_PATH + ".index", RELATED_WORK_TMP_PATH)
        self.check_index(REFERENCES_RELATED_WORK_TMP_PATH + ".index", REFERENCES_RELATED_WORK_TMP_PATH)


class TestCreateRelatedWorkMultiprocessing(TestCreateRelatedWork):
    def setUp(self) -> None:
        super(TestCreateRelatedWorkMultiprocessing, self).setUp()
        self.args = argparse.Namespace(**{
            "documents": PAPERS_FOR_RELATED_WORK_TMP_PATH,
            "reviews": RELATED_WORK_TMP_PATH,
            "references": REFERENCES_RELATED_WORK_TMP_PATH,
            "references_dataset": None,
            "workers": -1,
            "unordered": False,
        })


class IdsFilter(Filter):
    def __init__(self, allowed_ids: Set[int]):
        self.allowed_ids = allowed_ids

    def __call__(self, document: Document) -> bool:
        return document.id in self.allowed_ids


class TestFilterAndPrintReviews(BaseTestCase):
    def setUp(self) -> None:
        super(TestFilterAndPrintReviews, self).setUp()
        copyfile(REFERENCES_FIXTURES_PATH, REFERENCES_TMP_PATH)
        copyfile(REFERENCES_FIXTURES_PATH + ".index", REFERENCES_TMP_PATH + ".index")

    def test_filtration(self):
        reviews_filter = IdsFilter({119311037})
        allowed_ref = {119311039}
        with OADataset(REFERENCES_TMP_PATH, REFERENCES_TMP_PATH + ".index") as reviews, \
                open(REFERENCES_FILTERED_TMP_PATH, "w") as res_f, open(REFERENCES_FILTERED_TMP_PATH + ".index",
                                                                       "w") as res_index_f:
            filter_and_print(reviews, reviews_filter, res_f, res_index_f,
                                                                          allowed_ref)

        self.compare_jsonl(REFERENCES_FILTERED_FIXTURES_PATH_2, REFERENCES_FILTERED_TMP_PATH)
        self.check_index(REFERENCES_FILTERED_FIXTURES_PATH_2 + ".index", REFERENCES_FILTERED_TMP_PATH)


class TestFilterAndPrint(BaseTestCase):

    def setUp(self) -> None:
        super(TestFilterAndPrint, self).setUp()
        copyfile(REFERENCES_FIXTURES_PATH, REFERENCES_TMP_PATH)
        copyfile(REFERENCES_FIXTURES_PATH + ".index", REFERENCES_TMP_PATH + ".index")

    def test_filtration(self):
        reviews_filter = IdsFilter({119311037})
        allowed_ref = {119311039}
        with OADataset(REFERENCES_TMP_PATH, REFERENCES_TMP_PATH + ".index") as reviews, \
                open(REFERENCES_FILTERED_TMP_PATH, "w") as res_f, open(REFERENCES_FILTERED_TMP_PATH + ".index",
                                                                       "w") as res_index_f:
             filter_and_print(reviews, reviews_filter, res_f, res_index_f,
                                     allowed_ref)

        self.compare_jsonl(REFERENCES_FILTERED_FIXTURES_PATH_2, REFERENCES_FILTERED_TMP_PATH)
        self.check_index(REFERENCES_FILTERED_FIXTURES_PATH_2 + ".index", REFERENCES_FILTERED_TMP_PATH)


class TestGetLastLine(BaseTestCase):
    def setUp(self) -> None:
        super(TestGetLastLine, self).setUp()
        copyfile(REFERENCES_FIXTURES_PATH + ".index", REFERENCES_TMP_PATH + ".index")

    def test_last_line(self):
        self.assertEqual("119311039	1178\r\n", get_last_line(REFERENCES_TMP_PATH + ".index"))


class TestExtend(BaseTestCase):
    def setUp(self) -> None:
        super(TestExtend, self).setUp()
        copyfile(EXTENDING_FIXTURES_PATH, EXTENDING_TMP_PATH)
        copyfile(EXTENDING_FIXTURES_PATH + ".index", EXTENDING_TMP_PATH + ".index")
        copyfile(FOR_EXTEND_FIXTURES_PATH, FOR_EXTEND_TMP_PATH)
        copyfile(FOR_EXTEND_FIXTURES_PATH + ".index", FOR_EXTEND_TMP_PATH + ".index")
        self.args = argparse.Namespace(**{
            "first": EXTENDING_TMP_PATH,
            "second": FOR_EXTEND_TMP_PATH,
            "result": EXTENDED_TMP_PATH,
            "match_threshold": 1.0,
            "batch": 8,
            "docs": 2,
            "workers": 0,
            "max_year_diff": 0,
        })

    def test_extend(self):
        extend(self.args)
        self.compare_jsonl(EXTENDED_FIXTURES_PATH, EXTENDED_TMP_PATH, unordered=True)
        self.check_index(EXTENDED_TMP_PATH + ".index", EXTENDED_TMP_PATH)


MAG_REVIEW_LIST_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/mag_review_list.jsonl")
MAG_REVIEW_LIST_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/mag_review_list.jsonl")
MAG_REVIEW_LIST_INDEX_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/mag_review_list.jsonl.index")
MAG_REVIEW_LIST_INDEX_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/mag_review_list.jsonl.index")

MAG_CORE_EXT_LIST_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/mag_core_ext_review_list.jsonl")
MAG_CORE_EXT_LIST_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/mag_core_ext_review_list.jsonl")
MAG_CORE_EXT_LIST_INDEX_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/mag_core_ext_review_list.jsonl.index")
MAG_CORE_EXT_LIST_INDEX_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/mag_core_ext_review_list.jsonl.index")

GROBID_FOR_MAG_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/grobid_for_mag_ext")
GROBID_FOR_MAG_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/grobid_for_mag_ext")

IDENTIFIED_REFERENCES_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/identified_references.txt")
IDENTIFIED_REFERENCES_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/identified_references.txt")


class TestExtendMagWithCore(BaseTestCase):
    def setUp(self) -> None:
        super(TestExtendMagWithCore, self).setUp()
        self.clear_dirs_in_tmp()
        copytree(GROBID_FOR_MAG_FIXTURES_PATH, GROBID_FOR_MAG_TMP_PATH)
        copyfile(MAG_REVIEW_LIST_FIXTURES_PATH, MAG_REVIEW_LIST_TMP_PATH)
        copyfile(MAG_REVIEW_LIST_INDEX_FIXTURES_PATH, MAG_REVIEW_LIST_INDEX_TMP_PATH)
        self.args = argparse.Namespace(**{
            "mag": MAG_REVIEW_LIST_TMP_PATH,
            "core": GROBID_FOR_MAG_TMP_PATH,
            "result": MAG_CORE_EXT_LIST_TMP_PATH,
            "batch": 8,
            "workers": 0,
            "identified_references": IDENTIFIED_REFERENCES_TMP_PATH
        })

    def tearDown(self) -> None:
        super(TestExtendMagWithCore, self).tearDown()
        self.clear_dirs_in_tmp()

    @staticmethod
    def clear_dirs_in_tmp():
        if os.path.isdir(GROBID_FOR_MAG_TMP_PATH):
            rmtree(GROBID_FOR_MAG_TMP_PATH)

    def test_extend_mag(self):
        extend_mag_with_core(self.args)
        self.compare_jsonl(MAG_CORE_EXT_LIST_FIXTURES_PATH, MAG_CORE_EXT_LIST_TMP_PATH, True)
        self.check_index(MAG_CORE_EXT_LIST_INDEX_TMP_PATH, MAG_CORE_EXT_LIST_TMP_PATH, "PaperId")

        with open(IDENTIFIED_REFERENCES_TMP_PATH, "r") as res, open(IDENTIFIED_REFERENCES_FIXTURES_PATH, "r") as gt:
            self.assertSequenceEqual([set(x) for x in gt.readlines()], [set(x) for x in res.readlines()])

    def test_extend_mag_mult_proc(self):
        if multiprocessing.cpu_count() == 0:
            self.skipTest("This test can only be run on the multi cpu device.")
            return
        self.args.workers = -1
        extend_mag_with_core(self.args)
        self.compare_jsonl(MAG_CORE_EXT_LIST_FIXTURES_PATH, MAG_CORE_EXT_LIST_TMP_PATH, True)
        self.check_index(MAG_CORE_EXT_LIST_INDEX_TMP_PATH, MAG_CORE_EXT_LIST_TMP_PATH, "PaperId")

        with open(IDENTIFIED_REFERENCES_TMP_PATH, "r") as res, open(IDENTIFIED_REFERENCES_FIXTURES_PATH, "r") as gt:
            self.assertSequenceEqual([set(x) for x in gt.readlines()], [set(x) for x in res.readlines()])


class TestExtendMagWithCoreMultProc(TestExtendMagWithCore):
    def setUp(self) -> None:
        super().setUp()
        self.args.workers = -1


PAPERS_FOR_ENHANCEMENT_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/papers_for_enhancement.jsonl")
PAPERS_FOR_ENHANCEMENT_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/papers_for_enhancement.jsonl")
PAPERS_ENHANCED_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/papers_enhanced.jsonl")
PAPERS_ENHANCED_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/papers_enhanced.jsonl")
PAPERS_ENHANCED_JUST_REFERENCES_FIXTURES_PATH = os.path.join(SCRIPT_PATH,
                                                             "fixtures/papers_enhanced_just_references.jsonl")
PAPERS_ENHANCED_JUST_REFERENCES_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/papers_enhanced_just_references.jsonl")

PAPERS_ENHANCED_MAPPING_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/papers_enhanced_mapping.txt")
PAPERS_ENHANCED_MAPPING_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/papers_enhanced_mapping.txt")


MAG_FOR_ENHANCE_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/mag_for_enhance_list.jsonl")
MAG_FOR_ENHANCE_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/mag_for_enhance_list.jsonl")


class TestCreatePapers2MagMapping(BaseTestCase):
    def setUp(self) -> None:
        super(TestCreatePapers2MagMapping, self).setUp()
        copyfile(PAPERS_FOR_ENHANCEMENT_FIXTURES_PATH, PAPERS_FOR_ENHANCEMENT_TMP_PATH)
        copyfile(PAPERS_FOR_ENHANCEMENT_FIXTURES_PATH + ".index", PAPERS_FOR_ENHANCEMENT_TMP_PATH + ".index")
        copyfile(MAG_FOR_ENHANCE_FIXTURES_PATH, MAG_FOR_ENHANCE_TMP_PATH)
        self.args = argparse.Namespace(**{
            "dataset": PAPERS_FOR_ENHANCEMENT_TMP_PATH,
            "mag": MAG_FOR_ENHANCE_TMP_PATH,
            "batch": 8,
            "workers": 0,
            "force_gpu_split": False,
        })

    def test_mapping(self):
        create_papers_2_mag_mapping(self.args)
        with open(PAPERS_ENHANCED_MAPPING_FIXTURES_PATH, "r") as gt:
            self.assertSequenceEqual(gt.read().splitlines(), self.stdout.getvalue().splitlines())

    def test_mapping_mult_proc(self):
        if multiprocessing.cpu_count() == 0:
            self.skipTest("This test can only be run on the multi cpu device.")
            return
        self.args.workers = -1
        create_papers_2_mag_mapping(self.args)
        with open(PAPERS_ENHANCED_MAPPING_FIXTURES_PATH, "r") as gt:
            self.assertSequenceEqual(gt.read().splitlines(), self.stdout.getvalue().splitlines())


class TestEnhancePapersWithMag(BaseTestCase):
    def setUp(self) -> None:
        super(TestEnhancePapersWithMag, self).setUp()
        copyfile(PAPERS_FOR_ENHANCEMENT_FIXTURES_PATH, PAPERS_FOR_ENHANCEMENT_TMP_PATH)
        copyfile(PAPERS_FOR_ENHANCEMENT_FIXTURES_PATH + ".index", PAPERS_FOR_ENHANCEMENT_TMP_PATH + ".index")
        copyfile(MAG_FOR_ENHANCE_FIXTURES_PATH, MAG_FOR_ENHANCE_TMP_PATH)
        copyfile(MAG_FOR_ENHANCE_FIXTURES_PATH + ".index", MAG_FOR_ENHANCE_TMP_PATH + ".index")
        self.args = argparse.Namespace(**{
            "mag": MAG_FOR_ENHANCE_TMP_PATH,
            "dataset": PAPERS_FOR_ENHANCEMENT_TMP_PATH,
            "to_mag": PAPERS_ENHANCED_MAPPING_FIXTURES_PATH,
            "result": PAPERS_ENHANCED_TMP_PATH,
            "just_references": False,
            "match_threshold": 0.8,
            "workers": 0,
        })

    def test_enhanced(self):
        enhance_papers_with_mag(self.args)
        self.compare_jsonl(PAPERS_ENHANCED_FIXTURES_PATH, PAPERS_ENHANCED_TMP_PATH, True)
        self.check_index(PAPERS_ENHANCED_TMP_PATH + ".index", PAPERS_ENHANCED_TMP_PATH, "id")

    def test_enhance_mult_proc(self):
        if multiprocessing.cpu_count() == 0:
            self.skipTest("This test can only be run on the multi cpu device.")
            return

        self.args.workers = -1
        enhance_papers_with_mag(self.args)
        self.compare_jsonl(PAPERS_ENHANCED_FIXTURES_PATH, PAPERS_ENHANCED_TMP_PATH, True)
        self.check_index(PAPERS_ENHANCED_TMP_PATH + ".index", PAPERS_ENHANCED_TMP_PATH, "id")


class TestEnhancePapersWithMagJustReferences(BaseTestCase):
    def setUp(self) -> None:
        super(TestEnhancePapersWithMagJustReferences, self).setUp()
        copyfile(PAPERS_FOR_ENHANCEMENT_FIXTURES_PATH, PAPERS_FOR_ENHANCEMENT_TMP_PATH)
        copyfile(PAPERS_FOR_ENHANCEMENT_FIXTURES_PATH + ".index", PAPERS_FOR_ENHANCEMENT_TMP_PATH + ".index")
        copyfile(MAG_FOR_ENHANCE_FIXTURES_PATH, MAG_FOR_ENHANCE_TMP_PATH)
        copyfile(MAG_FOR_ENHANCE_FIXTURES_PATH + ".index", MAG_FOR_ENHANCE_TMP_PATH + ".index")
        self.args = argparse.Namespace(**{
            "mag": MAG_FOR_ENHANCE_TMP_PATH,
            "dataset": PAPERS_FOR_ENHANCEMENT_TMP_PATH,
            "to_mag": PAPERS_ENHANCED_MAPPING_FIXTURES_PATH,
            "result": PAPERS_ENHANCED_JUST_REFERENCES_TMP_PATH,
            "just_references": True,
            "match_threshold": 0.8,
            "batch": 8,
            "workers": 0,
        })

    def test_enhanced(self):
        enhance_papers_with_mag(self.args)
        self.compare_jsonl(PAPERS_ENHANCED_JUST_REFERENCES_FIXTURES_PATH, PAPERS_ENHANCED_JUST_REFERENCES_TMP_PATH,
                           True)
        self.check_index(PAPERS_ENHANCED_JUST_REFERENCES_TMP_PATH + ".index", PAPERS_ENHANCED_JUST_REFERENCES_TMP_PATH,
                         "id")

    def test_enhance_mult_proc(self):
        if multiprocessing.cpu_count() == 0:
            self.skipTest("This test can only be run on the multi cpu device.")
            return

        self.args.workers = -1
        enhance_papers_with_mag(self.args)
        self.compare_jsonl(PAPERS_ENHANCED_JUST_REFERENCES_FIXTURES_PATH, PAPERS_ENHANCED_JUST_REFERENCES_TMP_PATH,
                           True)
        self.check_index(PAPERS_ENHANCED_JUST_REFERENCES_TMP_PATH + ".index", PAPERS_ENHANCED_JUST_REFERENCES_TMP_PATH,
                         "id")


class TestFilterOADatasetWithReviews(BaseTestCase):
    def setUp(self) -> None:
        super(TestFilterOADatasetWithReviews, self).setUp()
        copyfile(REVIEWS_FOR_FILTRATION_FIXTURES_PATH, REVIEWS_FOR_FILTRATION_TMP_PATH)
        copyfile(REVIEWS_FOR_FILTRATION_FIXTURES_PATH + ".index", REVIEWS_FOR_FILTRATION_TMP_PATH + ".index")

        copyfile(REFERENCES_FOR_FILTRATION_FIXTURES_PATH, REFERENCES_FOR_FILTRATION_TMP_PATH)
        copyfile(REFERENCES_FOR_FILTRATION_FIXTURES_PATH + ".index", REFERENCES_FOR_FILTRATION_TMP_PATH + ".index")

    def test_filter_oa_dataset_with_reviews(self):
        rw = OARelatedWork(REVIEWS_FOR_FILTRATION_TMP_PATH, REVIEWS_FOR_FILTRATION_TMP_PATH + ".index")
        ref = OADataset(REFERENCES_FOR_FILTRATION_TMP_PATH, REFERENCES_FOR_FILTRATION_TMP_PATH + ".index")

        rw_filter = MockFilter({119311040, 119311041})
        ref_filter = MockFilter({119311037, 119311038, 119311039})
        with rw, ref:
            filter_oa_dataset_with_reviews(rw, ref, rw_filter, ref_filter, REVIEWS_FILTERED_TMP_PATH,
                                           REFERENCES_FILTERED_TMP_PATH)

            self.compare_jsonl(REVIEWS_FILTERED_FIXTURES_PATH, REVIEWS_FILTERED_TMP_PATH)
            self.check_index(REVIEWS_FILTERED_TMP_PATH + ".index", REVIEWS_FILTERED_TMP_PATH)

            self.compare_jsonl(REFERENCES_FILTERED_FIXTURES_PATH, REFERENCES_FILTERED_TMP_PATH)
            self.check_index(REFERENCES_FILTERED_TMP_PATH + ".index", REFERENCES_FILTERED_TMP_PATH)


class TestFilterRelatedWork(BaseTestCase):

    def setUp(self) -> None:
        super(TestFilterRelatedWork, self).setUp()
        copyfile(RELATED_WORK_FIXTURES_PATH, RELATED_WORK_TMP_PATH)
        copyfile(RELATED_WORK_FIXTURES_PATH + ".index", RELATED_WORK_TMP_PATH + ".index")
        copyfile(REFERENCES_FIXTURES_PATH, REFERENCES_TMP_PATH)
        copyfile(REFERENCES_FIXTURES_PATH + ".index", REFERENCES_TMP_PATH + ".index")

        self.args = argparse.Namespace(**{
            "related_work": RELATED_WORK_TMP_PATH,
            "references": REFERENCES_TMP_PATH,
            "res_related_work": REVIEWS_FILTERED_TMP_PATH,
            "res_references": REFERENCES_FILTERED_TMP_PATH,
            "sec_non_empty_headlines_ref": True,
            "has_abstract_rev": 0,
            "has_abstract_ref": 0,
            "min_cit": 1,
            "max_cit": 5,
            "min_cit_frac": 0.1,
            "max_cit_frac": 0.9,
            "min_cit_group_frac": 0.2,
            "max_cit_group_frac": 1.0,
            "min_sec_rev": 2,
            "max_sec_rev": 10,
            "min_sec_ref": 4,
            "max_sec_ref": 20,
            "min_par_rev": 8,
            "max_par_rev": 40,
            "min_par_ref": 16,
            "max_par_ref": 80,
            "min_fraction_of_cited_documents_with_multi_section_content_filter": 0,
            "max_fraction_of_cited_documents_with_multi_section_content_filter": 1,
            "workers": 0,
        })

    @patch("oapapers.__main__.filter_oa_dataset_with_reviews")
    def test_filtering(self, mock):
        filter_related_work(self.args)
        call_args = mock.call_args_list[0].args
        reviews_filters = {f.__class__.__name__: f for f in call_args[2].filters}
        references_filters = {f.__class__.__name__: f for f in call_args[3].filters}

        self.assertEqual(RELATED_WORK_TMP_PATH, call_args[0].path_to)
        self.assertEqual(REFERENCES_TMP_PATH, call_args[1].path_to)

        self.assertEqual({
            "NumberOfCitationsFilter", "CitationsFracFilter", "CitationsGroupsFracFilter", "NumberOfSectionsFilter",
            "NumberOfTextPartsInSectionFilter", "FractionOfCitedDocumentsWithMultiSectionContentFilter",
        }, set(reviews_filters.keys()))

        self.assertEqual({
            "NumberOfSectionsFilter", "NumberOfTextPartsInSectionFilter", "SecNonEmptyHeadlinesFilter"
        }, set(references_filters.keys()))

        self.assertEqual(1, reviews_filters["NumberOfCitationsFilter"].min_cit)
        self.assertEqual(5, reviews_filters["NumberOfCitationsFilter"].max_cit)

        self.assertEqual(0.1, reviews_filters["CitationsFracFilter"].min_cit_frac)
        self.assertEqual(0.9, reviews_filters["CitationsFracFilter"].max_cit_frac)

        self.assertEqual(0.2, reviews_filters["CitationsGroupsFracFilter"].min_cit_frac)
        self.assertEqual(1.0, reviews_filters["CitationsGroupsFracFilter"].max_cit_frac)

        self.assertEqual(2, reviews_filters["NumberOfSectionsFilter"].min_sec)
        self.assertEqual(10, reviews_filters["NumberOfSectionsFilter"].max_sec)

        self.assertEqual(4, references_filters["NumberOfSectionsFilter"].min_sec)
        self.assertEqual(20, references_filters["NumberOfSectionsFilter"].max_sec)

        self.assertEqual(8, reviews_filters["NumberOfTextPartsInSectionFilter"].min_par)
        self.assertEqual(40, reviews_filters["NumberOfTextPartsInSectionFilter"].max_par)

        self.assertEqual(16, references_filters["NumberOfTextPartsInSectionFilter"].min_par)
        self.assertEqual(80, references_filters["NumberOfTextPartsInSectionFilter"].max_par)

        self.assertEqual(0, reviews_filters["FractionOfCitedDocumentsWithMultiSectionContentFilter"].min_cit_frac)
        self.assertEqual(1, reviews_filters["FractionOfCitedDocumentsWithMultiSectionContentFilter"].max_cit_frac)

        self.assertEqual(REVIEWS_FILTERED_TMP_PATH, call_args[4])
        self.assertEqual(REFERENCES_FILTERED_TMP_PATH, call_args[5])


class TestPruningHier(BaseTestCase):
    def setUp(self) -> None:
        super(TestPruningHier, self).setUp()
        copyfile(FOR_PRUNING_PAPERS_FIXTURES_PATH, FOR_PRUNING_PAPERS_TMP_PATH)
        copyfile(FOR_PRUNING_PAPERS_FIXTURES_PATH + ".index", FOR_PRUNING_PAPERS_TMP_PATH + ".index")

        self.args = argparse.Namespace(**{
            "original": FOR_PRUNING_PAPERS_TMP_PATH,
            "result": PRUNED_PAPERS_TMP_PATH,
            "empty_headlines": True,
            "no_text": True,
            "plain_latin": 0.75,
            "named_text_blocks": True,
            "workers": 0,
            "from_index": 0,
            "to_index": None,
            "filter_ids": True
        })

    def test_pruning(self):
        pruning_hier(self.args)
        self.compare_jsonl(PRUNED_PAPERS_FIXTURES_PATH, PRUNED_PAPERS_TMP_PATH)
        self.check_index(PRUNED_PAPERS_TMP_PATH + ".index", PRUNED_PAPERS_TMP_PATH)


FOR_DEDUPLICATION_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/for_deduplication.jsonl")
FOR_DEDUPLICATION_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/for_deduplication.jsonl")
DEDUPLICATED_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/deduplicated.jsonl")
DEDUPLICATED_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/deduplicated.jsonl")


class TestDeduplication(BaseTestCase):
    def setUp(self) -> None:
        super(TestDeduplication, self).setUp()
        copyfile(FOR_DEDUPLICATION_FIXTURES_PATH, FOR_DEDUPLICATION_TMP_PATH)
        copyfile(FOR_DEDUPLICATION_FIXTURES_PATH + ".index", FOR_DEDUPLICATION_TMP_PATH + ".index")

        self.args = argparse.Namespace(**{
            "dataset": FOR_DEDUPLICATION_TMP_PATH,
            "result": DEDUPLICATED_TMP_PATH,
            "match_threshold": 1.0,
            "max_year_diff": 0,
            "batch": 2,
            "workers": 0
        })

    def test_deduplication(self):
        deduplication(self.args)
        self.compare_jsonl(DEDUPLICATED_FIXTURES_PATH, DEDUPLICATED_TMP_PATH)
        self.check_index(DEDUPLICATED_TMP_PATH + ".index", DEDUPLICATED_TMP_PATH)

FOR_IDENTIFICATION_OF_CITATION_SPANS_FIXTURES_PATH = os.path.join(SCRIPT_PATH,
                                                             "fixtures/for_identification_of_citation_spans.jsonl")
FOR_IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH = os.path.join(SCRIPT_PATH,
                                                             "tmp/for_identification_of_citation_spans.jsonl")
IDENTIFICATION_OF_CITATION_SPANS_FIXTURES_PATH = os.path.join(SCRIPT_PATH,
                                                              "fixtures/identification_of_citation_spans.jsonl")
IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH = os.path.join(SCRIPT_PATH,
                                                         "tmp/identification_of_citation_spans.jsonl")


class TestIdentifyCitationSpans(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        copyfile(FOR_IDENTIFICATION_OF_CITATION_SPANS_FIXTURES_PATH, FOR_IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH)
        copyfile(FOR_IDENTIFICATION_OF_CITATION_SPANS_FIXTURES_PATH + ".index",
                 FOR_IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH + ".index")

        self.args = argparse.Namespace(**{
            "dataset": FOR_IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH,
            "result": IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH,
            "workers": 0,
        })

    def test_identify_citation_spans(self):
        identify_citation_spans(self.args)
        self.compare_jsonl(IDENTIFICATION_OF_CITATION_SPANS_FIXTURES_PATH, IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH)
        self.check_index(IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH + ".index", IDENTIFICATION_OF_CITATION_SPANS_TMP_PATH)


class TestMultProcIdentifyCitationSpans(TestIdentifyCitationSpans):
    def setUp(self) -> None:
        super().setUp()
        self.args.workers = 2


FOR_IDENTIFICATION_OF_CITATIONS_FIXTURES_PATH = os.path.join(SCRIPT_PATH,
                                                             "fixtures/for_identification_of_citations.jsonl")
FOR_IDENTIFICATION_OF_CITATIONS_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/for_identification_of_citations.jsonl")
IDENTIFIED_CITATIONS_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/identified_citations.jsonl")
IDENTIFIED_CITATIONS_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/identified_citations.jsonl")


class TestIdentifyBibliography(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        copyfile(FOR_IDENTIFICATION_OF_CITATIONS_FIXTURES_PATH, FOR_IDENTIFICATION_OF_CITATIONS_TMP_PATH)
        copyfile(FOR_IDENTIFICATION_OF_CITATIONS_FIXTURES_PATH + ".index",
                 FOR_IDENTIFICATION_OF_CITATIONS_TMP_PATH + ".index")

        self.args = argparse.Namespace(**{
            "dataset": FOR_IDENTIFICATION_OF_CITATIONS_TMP_PATH,
            "result": IDENTIFIED_CITATIONS_TMP_PATH,
            "batch": 2,
            "docs": 256,
            "match_threshold": 0.8,
            "max_year_diff": 0,
            "workers": 0,
            "from_i": 0,
            "to_i": None,
            "search": None,
            "force_gpu_split": False,
            "title_db": None
        })

    def test_identify_bibliography(self):
        identify_bibliography(self.args)
        self.compare_jsonl(IDENTIFIED_CITATIONS_FIXTURES_PATH, IDENTIFIED_CITATIONS_TMP_PATH)
        self.check_index(IDENTIFIED_CITATIONS_TMP_PATH + ".index", IDENTIFIED_CITATIONS_TMP_PATH)


class TestMultProcIdentifyBibliography(TestIdentifyBibliography):
    def setUp(self) -> None:
        super().setUp()
        self.args.workers = 2


FOR_IDENTIFICATION_OF_BIB_FROM_GRAPH_FIXTURES_PATH = os.path.join(SCRIPT_PATH,
                                                                     "fixtures/enrich_bib_from_cit_graph/for_identification_of_bib.jsonl")
FOR_IDENTIFICATION_OF_BIB_FROM_GRAPH_TMP_PATH = os.path.join(SCRIPT_PATH,
                                                                "tmp/for_identification_of_bib.jsonl")
CITATION_GRAPH_FIXTURE_PATH = os.path.join(SCRIPT_PATH,
                                             "fixtures/enrich_bib_from_cit_graph/citation_graph.json")
CITATION_GRAPH_TMP_PATH = os.path.join(SCRIPT_PATH,
                                            "tmp/citation_graph.json")

IDENTIFIED_CITATIONS_WITH_GRAPH_FIXTURES_PATH = os.path.join(SCRIPT_PATH,
                                                                "fixtures/enrich_bib_from_cit_graph/identified_citations.jsonl")


class TestIdentifyBibliographyFromCitationGraph(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        copyfile(FOR_IDENTIFICATION_OF_BIB_FROM_GRAPH_FIXTURES_PATH, FOR_IDENTIFICATION_OF_BIB_FROM_GRAPH_TMP_PATH)
        copyfile(FOR_IDENTIFICATION_OF_BIB_FROM_GRAPH_FIXTURES_PATH + ".index",
                 FOR_IDENTIFICATION_OF_BIB_FROM_GRAPH_TMP_PATH + ".index")
        copyfile(CITATION_GRAPH_FIXTURE_PATH, CITATION_GRAPH_TMP_PATH)

        self.args = argparse.Namespace(**{
            "dataset": FOR_IDENTIFICATION_OF_BIB_FROM_GRAPH_FIXTURES_PATH,
            "result": IDENTIFIED_CITATIONS_TMP_PATH,
            "citation_graph": CITATION_GRAPH_TMP_PATH,
            "id": "s2orc_id",
            "search": None,
            "title_match_threshold": 0.75,
            "authors_match_threshold": 0.75,
            "year_diff_threshold": 2,
            "workers": 0,
            "from_i": 0,
            "to_i": None,
        })

    def test_enrich_bibliography(self):
        enrich_bibliography_from_citation_graph(self.args)
        self.compare_jsonl(IDENTIFIED_CITATIONS_WITH_GRAPH_FIXTURES_PATH, IDENTIFIED_CITATIONS_TMP_PATH)
        self.check_index(IDENTIFIED_CITATIONS_TMP_PATH + ".index", IDENTIFIED_CITATIONS_TMP_PATH)


class TestMultProcIdentifyBibliographyFromCitationGraph(TestIdentifyBibliography):
    def setUp(self) -> None:
        super().setUp()
        self.args.workers = 2


if __name__ == '__main__':
    unittest.main()
