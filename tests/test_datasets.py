# -*- coding: UTF-8 -*-
""""
Created on 21.04.22

:author:     Martin Dočekal
"""
import os
import unittest
from pathlib import Path
from shutil import copyfile, rmtree

from oapapers.bib_entry import BibEntry
from oapapers.datasets import OADataset, OARelatedWork
from oapapers.document import Document, OARelatedWorkDocument
from oapapers.hierarchy import Hierarchy, TextContent, RefSpan

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_PATH = os.path.join(SCRIPT_PATH, "tmp")

REFERENCES_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/references.jsonl")
REFERENCES_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/references.jsonl")
REFERENCES_FIXTURES_INDEX_PATH = os.path.join(SCRIPT_PATH, "fixtures/references.jsonl.index")
REFERENCES_TMP_INDEX_PATH = os.path.join(SCRIPT_PATH, "tmp/references.jsonl.index")

RELATED_WORK_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/related_work.jsonl")
RELATED_WORK_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/related_work.jsonl")
RELATED_WORK_FIXTURES_INDEX_PATH = os.path.join(SCRIPT_PATH, "fixtures/related_work.jsonl.index")
RELATED_WORK_TMP_INDEX_PATH = os.path.join(SCRIPT_PATH, "tmp/related_work.jsonl.index")


class BaseTestCase(unittest.TestCase):
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


class TestReferences(BaseTestCase):
    def setUp(self) -> None:
        copyfile(REFERENCES_FIXTURES_PATH, REFERENCES_TMP_PATH)
        copyfile(REFERENCES_FIXTURES_INDEX_PATH, REFERENCES_TMP_INDEX_PATH)

        self.references = OADataset(REFERENCES_TMP_PATH, REFERENCES_TMP_INDEX_PATH)

    def test_read(self):
        gt_doc = Document(id=119311039,
                          s2orc_id=1240,
                          mag_id=124,
                          doi="10.32473/flairs.v35i.130730",
                          title="This is not a review: part two",
                          authors=["Wolfgang Hollik", "Jonas M. Lindert"],
                          year=2013,
                          fields_of_study=[
                              "Mathematics", "Physics", "Regular polytope", "Abelian group",
                              "Dimension (graph theory)", "Polytope", "Platonic solid",
                              "Lie algebra", "Combinatorics", "Triality", "Group (mathematics)"
                          ],
                          citations=[119311037],
                          hierarchy=Hierarchy(
                              headline="This is not a review: part two",
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
                                                          text="Abstract of This is not a review: part two section 1",
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
                                                          text="Introduction [CITE:0] of This is not a review: part two section 1",
                                                          citations=[
                                                              RefSpan(
                                                                  index=0,
                                                                  start=13,
                                                                  end=21
                                                              )
                                                          ],
                                                          references=[]
                                                      )
                                                  )
                                              ]
                                          )
                                      ]
                                  )
                              ]
                          ),
                          bibliography=[
                              BibEntry(None, "Regular Polytopes", 1973, ("H S Coxeter",))
                          ],
                          non_plaintext_content=[
                              ("figure", "description of FIGREF0"), ("table", "description of TABREF0")
                          ],
                          uncategorized_fields={}
                          )

        with self.references:
            self.assertEqual(gt_doc, self.references.get_by_id(119311039))
            self.assertEqual(gt_doc, self.references[1])

            documents = list(self.references)
            self.assertEqual(2, len(documents))
            self.assertEqual(gt_doc, documents[1])

            self.references.workers = 2
            documents = list(self.references)
            self.assertEqual(2, len(documents))
            self.assertEqual(gt_doc, documents[1])


class TestRelatedWork(BaseTestCase):
    def setUp(self) -> None:
        copyfile(RELATED_WORK_FIXTURES_PATH, RELATED_WORK_TMP_PATH)
        copyfile(RELATED_WORK_FIXTURES_INDEX_PATH, RELATED_WORK_TMP_INDEX_PATH)

        self.related_work = OARelatedWork(RELATED_WORK_TMP_PATH, RELATED_WORK_TMP_INDEX_PATH)

    def test_read(self):
        gt_doc = OARelatedWorkDocument(id=119311038,
                                       s2orc_id=20,
                                       mag_id=2,
                                       doi=None,
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
                                                               text="Introduction [CITE:1] of On regular polytopes section 1",
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
                                           BibEntry(None, "Fake citation", 1973, ("Mr. Fake",)),
                                           BibEntry(119311039, "This is not a review: part two", 2013, ("Wolfgang Hollik", "Jonas M. Lindert"))
                                       ],
                                       non_plaintext_content=[("figure", "description of FIGREF2"), ("table", "description of TABREF0")],
                                       uncategorized_fields={}
                                       )

        with self.related_work:
            self.assertEqual(gt_doc, self.related_work.get_by_id(119311038))
            self.assertEqual(gt_doc, self.related_work[0])


if __name__ == '__main__':
    unittest.main()
