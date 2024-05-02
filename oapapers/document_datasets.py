# -*- coding: UTF-8 -*-
""""
Created on 24.01.22
Module for reading document datasets.

:author:     Martin Dočekal
"""
import ctypes
import gzip
import inspect
import itertools
import math
import multiprocessing
import random
import sys
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import partial
from json import JSONDecodeError

from typing import Dict, Union, List, Generator, Optional, Any, Callable, Iterable, Tuple

import spacy
from windpyutils.files import MapAccessFile, RandomLineAccessFile
from windpyutils.parallel.own_proc_pools import FunctorWorker, FunctorWorkerFactory, FactoryFunctorPool
from windpyutils.structures.maps import ImmutIntervalMap
from windpyutils.structures.sorted import SortedMap

from oapapers.bib_entry import BibEntry
from oapapers.citation_spans import enhance_citations_spans
from oapapers.cython.normalization import normalize_and_tokenize_string
from oapapers.document import Document
from oapapers.grobid_doc import GROBIDDoc
from oapapers.hierarchy import Hierarchy, TextContent, RefSpan
from oapapers.myjson import json_loads
from oapapers.papers_list import MAGPapersList, COREPapersList, PapersList, PapersListRecordWithAllIds
from oapapers.text import SpanCollisionHandling, replace_at, DeHyphenator, clean_title


class DocumentDataset(ABC):
    """
    Abstract class for datasets of documents.

    :ivar stub_mode: True activates stub mode which provides stub documents.
        Stub documents are documents that might not contain whole content, but just short "preview".
        Also, the citation matching (matching bibliography in dataset) might be omitted.
    """

    def __init__(self):
        self._transform = None
        self._transform_with_index = False
        self.stub_mode = False
        self.preload_filter = None

    @abstractmethod
    def __len__(self):
        """
        Number of documents in dataset.
        """
        pass

    def __iter__(self) -> Generator[Union[Document, Any], None, None]:
        """
        sequence iteration over whole file
        :return: generator of documents or transformed documents when the transformation is activated
        """
        yield from self.iter_range()

    def iter_range(self, f: int = 0, t: Optional[int] = None,
                   unordered: bool = False) -> Generator[Union[Document, Any], None, None]:
        """
        sequence iteration over given range
        :param f: from
        :param t: to
        :param unordered: if True the documents are not returned in order
            might speed up the iteration in multiprocessing mode
        :return: generator of documents or transformed documents when the transformation is activated
        """
        if t is None:
            t = len(self)
        for i in range(f, t):
            yield self[i]

    @property
    def transform(self):
        """
        Transformation that should be applied on a document.
        """
        return self._transform

    @transform.setter
    def transform(self, t: Optional[Union[Callable[[Document], Any], Callable[[Document, int], Any]]]):
        """
        Sets transformation that should be applied on a document.

        :param t: new transformation
            accepts document and returns transformed document
            voluntary accepts document and its index and returns transformed document
        """
        self._transform = t
        self._transform_with_index = t is not None and len(inspect.signature(t).parameters) == 2

    @property
    def transform_with_index(self):
        """
        Whether the transformation accepts document and its index.
        """
        return self._transform_with_index

    def apply_transform(self, doc: Document, index: int) -> Any:
        """
        Applies transformation on a document.

        :param doc: document to transform
        :param index: index of the document
        :return: transformed document
        """
        if self._transform is None:
            return doc
        elif self._transform_with_index:
            return self._transform(doc, index)
        else:
            return self._transform(doc)

    def __getitem__(self, selector: Union[int, slice]) -> Union[Document, List[Document]]:
        """
        Get document from dataset.

        :param selector: line number (from zero) or slice
        :return: the document, or list of documents when the slice is used
        """

        if isinstance(selector, slice):
            return [self._get_item(i) for i in range(len(self))[selector]]
        else:
            return self._get_item(selector)

    @abstractmethod
    def _get_item(self, selector: int) -> Union[Document, Any]:
        """
        Get document from dataset.

        :param selector: id of a document
        :return: the document or transformed document
        """
        pass



class DatasetMultProcWorker(FunctorWorker):
    """
    Helper for multiprocessing.
    Allows to obtain documents in parallel.
    """

    def __init__(self, dataset: DocumentDataset, max_chunks_per_worker: float = math.inf):
        super().__init__(max_chunks_per_worker)
        self.dataset = dataset

    def begin(self):
        if hasattr(self.dataset.transform, "begin"):
            self.dataset.transform.begin()

    def end(self):
        if hasattr(self.dataset.transform, "end"):
            self.dataset.transform.end()

    def __call__(self, i: int) -> Document:
        """
        Obtains document on given index.

        :param i: index of given document
        :return: the document
        """
        try:
            return self.dataset[i]
        except Exception as e:
            print(f"There was an exception in process {multiprocessing.current_process()}")
            traceback.print_exc()
            raise e


class DatasetMultProcWorkerFactory(FunctorWorkerFactory):
    """
    Factory for creating dataset workers.
    """

    def __init__(self, dataset: DocumentDataset, max_chunks_per_worker: float = math.inf):
        """
        :param dataset: dataset it should work on
        :param max_chunks_per_worker: Defines maximal number of chunks that a worker will do before it will stop
            New worker will replace it when used with pool that supports replace queue.

            This is particular useful when you observe increasing memory, as it seems there is a known problem
                with that: https://stackoverflow.com/questions/21485319/high-memory-usage-using-python-multiprocessing
        """
        self.dataset = dataset
        self.max_chunks_per_worker = max_chunks_per_worker

    def create(self) -> DatasetMultProcWorker:
        return DatasetMultProcWorker(self.dataset, self.max_chunks_per_worker)


class Archive:
    """
    Reading of s2orc dataset archive.
    """

    def __init__(self, path_to: str):
        """
        Initialization of archive. No extraction is done in that phase.

        :param path_to: Path to s2orc datase archive.
        :param tmp_folder: If none system tmp will be used. Otherwise, you can specify which folder you want to use.
        """

        self._path_to = path_to
        self._archive_descriptor = None
        self.verbose = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> "Archive":
        """
        Opens the archive if it is closed, else it is just empty operation.

        :return: Returns the object itself.
        """

        if self._archive_descriptor is None:
            self._archive_descriptor = gzip.open(self._path_to, 'rb')

        return self

    def close(self):
        """
        Closes the archive.
        """

        if self._archive_descriptor is not None:
            self._archive_descriptor.close()

    def __iter__(self) -> Generator[Dict, None, None]:
        """
        Iterates over the archive and generates json lines.

        :return: arhive contents generator
        """
        if self._archive_descriptor is None:
            raise RuntimeError("Please open the archive first.")

        for line in self._archive_descriptor:
            if len(line.strip()) == 0:
                # skips empty line that could be at the end of a file
                continue
            yield json_loads(line.decode('utf-8'))


class S2ORCDocumentDataset(DocumentDataset):
    """
    S2ORC dataset reader.

    S2ORC is dataset presented in
        http://dx.doi.org/10.18653/v1/2020.acl-main.447
    . The original official github is https://github.com/allenai/s2orc., but now it is part of semantic scholar API
        https://api.semanticscholar.org/.

    """
    lock = multiprocessing.Lock()

    def __init__(self, metadata: str, s2orc: str, workers: int = 0, chunk_size: int = 10):
        """
        initialization of dataset

        :param metadata: path to metadata jsonl indexed file
        :param s2orc: path to s2orc jsonl indexed file
        :param workers: activates multiprocessing and determines number of workers that should be used
            the multiprocessing is used during iteration through whole dataset
        :param chunk_size: chunk size for single process when the multiprocessing is activated
        """
        super().__init__()

        self._prepare_spacy()

        self.workers = workers
        self.chunk_size = chunk_size
        self.max_chunks_per_worker = 10_000
        self.dehyphenator = DeHyphenator()
        self._metadata = MapAccessFile(metadata, metadata + ".index", int)
        self._manager = multiprocessing.Manager()

        if workers > 0:
            self._metadata.mapping = SortedMap(self._metadata.mapping)
            self._metadata.mapping.keys_storage = multiprocessing.Array(ctypes.c_int64,
                                                                        self._metadata.mapping.keys_storage, lock=False)
            self._metadata.mapping.values_storage = multiprocessing.Array(ctypes.c_int64,
                                                                          self._metadata.mapping.values_storage, lock=False)

        self._ids = list(self._metadata.mapping.keys())
        self._s2orc = MapAccessFile(s2orc, s2orc + ".index", int)
        self._s2orc.mapping = SortedMap(self._s2orc.mapping)
        self._s2orc.mapping.keys_storage = multiprocessing.Array(ctypes.c_int64,
                                                                 self._s2orc.mapping.keys_storage, lock=False)
        self._s2orc.mapping.values_storage = multiprocessing.Array(ctypes.c_int64,
                                                                   self._s2orc.mapping.values_storage, lock=False)

        self.spacy_batch_size = 1000

        self.cnt = 0
        self.total_time = 0
        self.assemble_metadata_time = 0
        self.prepare_content_time = 0
        self.split_into_sentences_time = 0
        self.create_hierarchy_time = 0
        self.assemble_content_time = 0
        self.add_missing_bib_time = 0
        self.enhance_citations_time = 0
        self.profile = False

    def __enter__(self):
        self._metadata.__enter__()
        self._s2orc.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._metadata.__exit__(exc_type, exc_value, traceback)
        self._s2orc.__exit__(exc_type, exc_value, traceback)

    def __len__(self):
        return len(self._ids)

    def __iter__(self) -> Generator[Union[Document, Any], None, None]:
        """
        sequence iteration over whole file
        :return: generator of documents or transformed documents when the transformation is activated
        """
        yield from self.iter_range()

    def iter_range(self, f: int = 0, t: Optional[int] = None,
                   unordered: bool = False) -> Generator[Union[Document, Any], None, None]:
        """
        sequence iteration over given range
        :param f: from
        :param t: to
        :param unordered: if True the documents are not returned in order
            might speed up the iteration in multiprocessing mode
        :return: generator of documents or transformed documents when the transformation is activated
        """
        if self._metadata.file is None or self._s2orc.file is None:
            raise RuntimeError("Firstly open the dataset.")

        with nullcontext() if self.workers <= 0 else \
                FactoryFunctorPool(self.workers, DatasetMultProcWorkerFactory(self, self.max_chunks_per_worker),
                                   results_queue_maxsize=10.0, verbose=True, join_timeout=5) as pool:

            m = partial(map, self._get_item) if self.workers <= 0 \
                else partial((pool.imap_unordered if unordered else pool.imap), chunk_size=self.chunk_size)

            if t is None:
                t = len(self)

            for document in m(range(f, t)):
                yield document

    def _prepare_spacy(self):
        """
        Loads the spacy models and selects the components we need.
        """
        self._spacy = spacy.load("en_core_sci_sm")
        self._spacy_stub = spacy.blank("en")
        self._spacy_stub.add_pipe("sentencizer")

    @property
    def spacy(self) -> spacy.Language:
        """
        Returns spacy model instance.
        """
        if self.stub_mode:
            return self._spacy_stub
        return self._spacy

    @classmethod
    def get_authors(cls, doc: Dict) -> List[str]:
        """
        Getting authors from the document.

        :param doc: document
        :return: list of authors
        """
        res = []
        for author_span in doc["annotations"]["authors"]:
            first_names = list(
                cls.materialize_spans(
                    cls.selects_sub_spans(
                        doc["annotations"]["authorfirstname"], int(author_span["start"]), int(author_span["end"])
                    ), doc["text"]
                )
            )
            last_names = list(
                cls.materialize_spans(
                    cls.selects_sub_spans(
                        doc["annotations"]["authorlastname"], int(author_span["start"]), int(author_span["end"])
                    ), doc["text"]
                )
            )
            res.append(" ".join(first_names + last_names))

        return res

    @staticmethod
    def materialize_spans(spans: Iterable[Dict], text: str) -> Generator[str, None, None]:
        """
        Materializes the spans from the document. It means that it selects the text of a spans.

        :param spans: spans from the document
        :param text: text of the document
        :return: list of strings that are materialized spans
        """

        for span in spans:
            yield text[span["start"]:span["end"]]

    @staticmethod
    def selects_sub_spans(spans: Iterable[Dict], start: int, end: int) -> Generator[Dict, None, None]:
        """
        Selects the spans that are in given range.

        :param spans: spans from the document
        :param start: start of the range
        :param end: end of the range
        :return: spans that are in given range
        """

        for span in spans:
            if span["start"] >= start and span["end"] <= end:
                yield span

    def assemble_metadata(self, doc_id: int, json_metadata: Dict) -> Dict[str, Any]:
        """
        Assembles metadata from the document.

        :param doc_id: id of the document
        :param json_metadata: metadata from the document
        :return: metadata
        """

        """example of json_metadata:
        {
           "corpusid":118105413,
           "externalids":{
              "ACL":null,
              "DBLP":null,
              "ArXiv":null,
              "MAG":"379921969",
              "CorpusId":"118105413",
              "PubMed":null,
              "DOI":null,
              "PubMedCentral":null
           },
           "url":"https://www.semanticscholar.org/paper/022f1585041bbe99bb9ab26d6b379353ae3ad15e",
           "title":"Self-organization of isotopic and drift-ware turbulence",
           "authors":[
              {
                 "authorId":"93856515",
                 "name":"A. Pushkarev"
              }
           ],
           "venue":"",
           "publicationvenueid":null,
           "year":2013,
           "referencecount":0,
           "citationcount":0,
           "influentialcitationcount":0,
           "isopenaccess":false,
           "s2fieldsofstudy":[
              {
                 "category":"Physics",
                 "source":"s2-fos-model"
              },
              {
                 "category":"Physics",
                 "source":"external"
              }
           ],
           "publicationtypes":null,
           "publicationdate":"2013-12-18",
           "journal":null,
           "updated":"2022-01-27T17:31:22.629Z",
           "abstract":"",
           "citing":[]
        }
        """

        return {
            "id": doc_id,
            "s2orc_id": doc_id,
            "mag_id": int(json_metadata["externalids"]["MAG"]) if json_metadata["externalids"]["MAG"] is not None else None,
            "doi": json_metadata["externalids"]["DOI"],
            "title": json_metadata["title"],
            "authors": [] if json_metadata["authors"] is None else list(a["name"] for a in json_metadata["authors"]),
            "year": json_metadata["year"],
            "fields_of_study": [] if json_metadata["s2fieldsofstudy"] is None else sorted(
                set(f["category"] for f in json_metadata["s2fieldsofstudy"])),
            "abstract": json_metadata["abstract"],
            "citations": sorted(json_metadata["citing"]),
        }

    def assemble_paragraphs(self, doc: Dict) -> List[Dict[str, Any]]:
        """
        Assembles paragraphs from the document.
            Every paragraph will have type, section, text, cite_spans, ref_spans, start, end

        :param doc: s2orc document content representation (values of 'content' key)
        :return: list of paragraphs
            There are two types of paragraphs:
                - paragraph
                - formula
        """

        paragraphs = []

        # assemble spans of sections from section headers, place between two header is a section
        sections_spans = []
        headers = []
        if doc["annotations"]["sectionheader"] is not None:
            # deduplicate section headers
            dedup = sorted(set((int(sec["start"]), int(sec["end"])) for sec in doc["annotations"]["sectionheader"]),
                           key=lambda x: x[0])

            for i, sec in enumerate(dedup):
                headline_number = ""
                for sec_orig in doc["annotations"]["sectionheader"]:
                    if int(sec_orig["start"]) == sec[0] and int(sec_orig["end"]) == sec[1]:
                        # if there is something like {"attributes":{"n":"1"}
                        if "attributes" in sec_orig and "n" in sec_orig["attributes"]:
                            headline_number = sec_orig["attributes"]["n"] + " "
                        break
                if i == len(dedup) - 1:
                    sections_spans.append((int(sec[0]), math.inf))
                else:
                    sections_spans.append((int(sec[0]), int(dedup[i + 1][0]) - 1))

                headers.append(headline_number + doc["text"][sec[0]:sec[1]])

        try:
            sec_headline_map = ImmutIntervalMap(
                {
                    sec: sec_title
                    for sec, sec_title in zip(sections_spans, headers)
                }
            )
        except KeyError as e:
            print(doc["annotations"]["sectionheader"], sections_spans)
            raise e

        if doc["annotations"]["paragraph"] is not None:
            for par_span in doc["annotations"]["paragraph"]:
                # "bibref":"[{\"attributes\":{\"ref_id\":\"b0\"},\"end\":1433,\"start\":1432}
                bibref = []
                figureref = []
                tableref = []

                if doc["annotations"]["bibref"] is not None:
                    bibref = doc["annotations"]["bibref"]

                    for bib in bibref:
                        bib["start"], bib["end"] = int(bib["start"]), int(bib["end"])
                cite_spans = self.selects_sub_spans(bibref, int(par_span["start"]), int(par_span["end"]))

                # "figureref":"[{\"attributes\":{\"ref_id\":\"fig_0\"},\"end\":3954,\"start\":3945}",
                if doc["annotations"]["figureref"] is not None:
                    figureref = doc["annotations"]["figureref"]
                    for fig in figureref:
                        fig["start"], fig["end"] = int(fig["start"]), int(fig["end"])
                ref_spans = self.selects_sub_spans(figureref, int(par_span["start"]), int(par_span["end"]))
                # "tableref":"[{\"attributes\":{\"ref_id\":\"tab_1\"},\"end\":28061,\"start\":28054}
                if doc["annotations"]["tableref"] is not None:
                    tableref = doc["annotations"]["tableref"]
                    for tab in tableref:
                        tab["start"], tab["end"] = int(tab["start"]), int(tab["end"])
                ref_spans = itertools.chain(
                    ref_spans, self.selects_sub_spans(tableref, int(par_span["start"]), int(par_span["end"]))
                )

                try:
                    section_h = sec_headline_map[int(par_span["start"])]
                except KeyError:
                    section_h = ""

                par_span["start"] = int(par_span["start"])
                par_span["end"] = int(par_span["end"])

                cite_spans = list(cite_spans)
                ref_spans = list(ref_spans)
                # change spans offsets
                for s in itertools.chain(cite_spans, ref_spans):
                    s["start"] = s["start"] - par_span["start"]
                    s["end"] = s["end"] - par_span["start"]

                paragraphs.append(
                    {
                        "type": "paragraph",
                        "section": section_h,
                        "text": doc["text"][par_span["start"]:par_span["end"]],
                        "cite_spans": cite_spans,
                        "ref_spans": sorted(ref_spans, key=lambda x: x["start"]),
                        "start": par_span["start"],
                        "end": par_span["end"]
                    }
                )

        if doc["annotations"]["formula"] is not None:
            for formula_span in doc["annotations"]["formula"]:
                try:
                    section_h = sec_headline_map[int(formula_span["start"])]
                except KeyError:
                    section_h = ""

                paragraphs.append(
                    {
                        "type": "formula",
                        "section": section_h,
                        "text": doc["text"][int(formula_span["start"]):int(formula_span["end"])],
                        "cite_spans": [],
                        "ref_spans": [],
                        "start": int(formula_span["start"]),
                        "end": int(formula_span["end"])
                    }
                )

        return sorted(paragraphs, key=lambda x: int(x["start"]))

    def assemble_bib_entries(self, doc: Dict) -> Dict[str, Dict[str, Any]]:
        """
        Assembles bibliography entries from the document.
            Every bibliography entry will have title, authors, year, matched id

        :param doc: s2orc document content representation (values of 'content' key)
        :return: mapping of bibliography entries
        """

        """ related fields
        "bibauthor":"[{\"end\":11130,\"start\":11120},{\"end\":11142,\"start\":11130},{\"end\":11153,\"start\":11142},{\"end\":11164,\"start\":11153},{\"end\":11175,\"start\":11164},{\"end\":11429,\"start\":11418},{\"end\":11440,\"start\":11429},{\"end\":11454,\"start\":11440},{\"end\":11466,\"start\":11454},{\"end\":11479,\"start\":11466},{\"end\":11489,\"start\":11479},{\"end\":11880,\"start\":11870},{\"end\":11890,\"start\":11880},{\"end\":11901,\"start\":11890},{\"end\":11915,\"start\":11901},{\"end\":11923,\"start\":11915},{\"end\":11935,\"start\":11923},{\"end\":12281,\"start\":12270},{\"end\":12289,\"start\":12281},{\"end\":12299,\"start\":12289},{\"end\":12310,\"start\":12299},{\"end\":12321,\"start\":12310},{\"end\":12333,\"start\":12321},{\"end\":13730,\"start\":13722},{\"end\":13744,\"start\":13730},{\"end\":13758,\"start\":13744},{\"end\":13768,\"start\":13758},{\"end\":14184,\"start\":14173},{\"end\":14197,\"start\":14184},{\"end\":14209,\"start\":14197},{\"end\":14571,\"start\":14561},{\"end\":14580,\"start\":14571},{\"end\":14589,\"start\":14580},{\"end\":14599,\"start\":14589},{\"end\":14942,\"start\":14931},{\"end\":14955,\"start\":14942},{\"end\":14965,\"start\":14955},{\"end\":14976,\"start\":14965},{\"end\":14995,\"start\":14976},{\"end\":15005,\"start\":14995},{\"end\":15387,\"start\":15375},{\"end\":15400,\"start\":15387},{\"end\":15794,\"start\":15779},{\"end\":15807,\"start\":15794},{\"end\":15819,\"start\":15807},{\"end\":15833,\"start\":15819},{\"end\":15845,\"start\":15833},{\"end\":15859,\"start\":15845},{\"end\":16222,\"start\":16210},{\"end\":16238,\"start\":16222},{\"end\":16248,\"start\":16238},{\"end\":16259,\"start\":16248}]",
        "bibauthorfirstname":"[{\"end\":11121,\"start\":11120},{\"end\":11131,\"start\":11130},{\"end\":11143,\"start\":11142},{\"end\":11145,\"start\":11144},{\"end\":11154,\"start\":11153},{\"end\":11165,\"start\":11164},{\"end\":11419,\"start\":11418},{\"end\":11421,\"start\":11420},{\"end\":11430,\"start\":11429},{\"end\":11432,\"start\":11431},{\"end\":11441,\"start\":11440},{\"end\":11443,\"start\":11442},{\"end\":11455,\"start\":11454},{\"end\":11457,\"start\":11456},{\"end\":11467,\"start\":11466},{\"end\":11469,\"start\":11468},{\"end\":11480,\"start\":11479},{\"end\":11482,\"start\":11481},{\"end\":11871,\"start\":11870},{\"end\":11881,\"start\":11880},{\"end\":11891,\"start\":11890},{\"end\":11902,\"start\":11901},{\"end\":11916,\"start\":11915},{\"end\":11924,\"start\":11923},{\"end\":11926,\"start\":11925},{\"end\":12271,\"start\":12270},{\"end\":12273,\"start\":12272},{\"end\":12282,\"start\":12281},{\"end\":12290,\"start\":12289},{\"end\":12300,\"start\":12299},{\"end\":12302,\"start\":12301},{\"end\":12311,\"start\":12310},{\"end\":12322,\"start\":12321},{\"end\":12324,\"start\":12323},{\"end\":13723,\"start\":13722},{\"end\":13731,\"start\":13730},{\"end\":13733,\"start\":13732},{\"end\":13745,\"start\":13744},{\"end\":13747,\"start\":13746},{\"end\":13759,\"start\":13758},{\"end\":14174,\"start\":14173},{\"end\":14176,\"start\":14175},{\"end\":14185,\"start\":14184},{\"end\":14187,\"start\":14186},{\"end\":14198,\"start\":14197},{\"end\":14200,\"start\":14199},{\"end\":14562,\"start\":14561},{\"end\":14572,\"start\":14571},{\"end\":14581,\"start\":14580},{\"end\":14590,\"start\":14589},{\"end\":14935,\"start\":14931},{\"end\":14943,\"start\":14942},{\"end\":14945,\"start\":14944},{\"end\":14956,\"start\":14955},{\"end\":14966,\"start\":14965},{\"end\":14977,\"start\":14976},{\"end\":14996,\"start\":14995},{\"end\":15376,\"start\":15375},{\"end\":15378,\"start\":15377},{\"end\":15388,\"start\":15387},{\"end\":15390,\"start\":15389},{\"end\":15780,\"start\":15779},{\"end\":15782,\"start\":15781},{\"end\":15795,\"start\":15794},{\"end\":15797,\"start\":15796},{\"end\":15808,\"start\":15807},{\"end\":15810,\"start\":15809},{\"end\":15820,\"start\":15819},{\"end\":15822,\"start\":15821},{\"end\":15834,\"start\":15833},{\"end\":15836,\"start\":15835},{\"end\":15846,\"start\":15845},{\"end\":15848,\"start\":15847},{\"end\":16211,\"start\":16210},{\"end\":16213,\"start\":16212},{\"end\":16223,\"start\":16222},{\"end\":16239,\"start\":16238},{\"end\":16241,\"start\":16240},{\"end\":16249,\"start\":16248},{\"end\":16251,\"start\":16250}]",
        "bibauthorlastname":"[{\"end\":11128,\"start\":11122},{\"end\":11140,\"start\":11132},{\"end\":11151,\"start\":11146},{\"end\":11162,\"start\":11155},{\"end\":11173,\"start\":11166},{\"end\":11427,\"start\":11422},{\"end\":11438,\"start\":11433},{\"end\":11452,\"start\":11444},{\"end\":11464,\"start\":11458},{\"end\":11477,\"start\":11470},{\"end\":11487,\"start\":11483},{\"end\":11878,\"start\":11872},{\"end\":11888,\"start\":11882},{\"end\":11899,\"start\":11892},{\"end\":11913,\"start\":11903},{\"end\":11921,\"start\":11917},{\"end\":11933,\"start\":11927},{\"end\":12279,\"start\":12274},{\"end\":12287,\"start\":12283},{\"end\":12297,\"start\":12291},{\"end\":12308,\"start\":12303},{\"end\":12319,\"start\":12312},{\"end\":12331,\"start\":12325},{\"end\":13728,\"start\":13724},{\"end\":13742,\"start\":13734},{\"end\":13756,\"start\":13748},{\"end\":13766,\"start\":13760},{\"end\":14182,\"start\":14177},{\"end\":14195,\"start\":14188},{\"end\":14207,\"start\":14201},{\"end\":14569,\"start\":14563},{\"end\":14578,\"start\":14573},{\"end\":14587,\"start\":14582},{\"end\":14597,\"start\":14591},{\"end\":14940,\"start\":14936},{\"end\":14953,\"start\":14946},{\"end\":14963,\"start\":14957},{\"end\":14974,\"start\":14967},{\"end\":14993,\"start\":14978},{\"end\":15003,\"start\":14997},{\"end\":15385,\"start\":15379},{\"end\":15398,\"start\":15391},{\"end\":15792,\"start\":15783},{\"end\":15805,\"start\":15798},{\"end\":15817,\"start\":15811},{\"end\":15831,\"start\":15823},{\"end\":15843,\"start\":15837},{\"end\":15857,\"start\":15849},{\"end\":16220,\"start\":16214},{\"end\":16236,\"start\":16224},{\"end\":16246,\"start\":16242},{\"end\":16257,\"start\":16252}]",
        "bibentry":"[{\"attributes\":{\"doi\":\"10.1016/j.epsc.2019.101336\",\"id\":\"b0\",\"matched_paper_id\":209253920},\"end\":11414,\"start\":11075},{\"attributes\":{\"id\":\"b1\"},\"end\":11653,\"start\":11416},{\"attributes\":{\"doi\":\"10.1016/j.jpedsurg.2003.12.007\",\"id\":\"b2\"},\"end\":11794,\"start\":11655},{\"attributes\":{\"doi\":\"10.1159/000264281\",\"id\":\"b3\",\"matched_paper_id\":46824664},\"end\":12204,\"start\":11796},{\"attributes\":{\"doi\":\"10.1016/s1470-2045(07)70241-3\",\"id\":\"b4\",\"matched_paper_id\":38559830},\"end\":12587,\"start\":12206},{\"attributes\":{\"id\":\"b5\"},\"end\":12745,\"start\":12589},{\"attributes\":{\"id\":\"b6\"},\"end\":13643,\"start\":12747},{\"attributes\":{\"id\":\"b7\"},\"end\":14046,\"start\":13645},{\"attributes\":{\"doi\":\"10.1148/rg.2016150230\",\"id\":\"b8\",\"matched_paper_id\":25121195},\"end\":14499,\"start\":14048},{\"attributes\":{\"doi\":\"10.5146/tjpath.2013.01149\",\"id\":\"b9\"},\"end\":14833,\"start\":14501},{\"attributes\":{\"doi\":\"10.3390/cancers12030729\",\"id\":\"b10\"},\"end\":15286,\"start\":14835},{\"attributes\":{\"doi\":\"10.1111/j.1365-2559.2008.03110.x\",\"id\":\"b11\",\"matched_paper_id\":205299815},\"end\":15650,\"start\":15288},{\"attributes\":{\"id\":\"b12\",\"matched_paper_id\":11198070},\"end\":16153,\"start\":15652},{\"attributes\":{\"doi\":\"10.1590/S0101-28002011000100014\",\"id\":\"b13\",\"matched_paper_id\":3098497},\"end\":16497,\"start\":16155}]",
        "bibref":"[{\"attributes\":{\"ref_id\":\"b0\"},\"end\":1433,\"start\":1432},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":1572,\"start\":1571},{\"attributes\":{\"ref_id\":\"b3\"},\"end\":1688,\"start\":1687},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":1903,\"start\":1902},{\"attributes\":{\"ref_id\":\"b7\"},\"end\":5672,\"start\":5671},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":5721,\"start\":5720},{\"attributes\":{\"ref_id\":\"b7\"},\"end\":5785,\"start\":5784},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":5847,\"start\":5846},{\"attributes\":{\"ref_id\":\"b3\"},\"end\":5979,\"start\":5978},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":6090,\"start\":6089},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":6262,\"start\":6261},{\"attributes\":{\"ref_id\":\"b8\"},\"end\":7069,\"start\":7068},{\"attributes\":{\"ref_id\":\"b8\"},\"end\":7213,\"start\":7212},{\"attributes\":{\"ref_id\":\"b7\"},\"end\":7535,\"start\":7534},{\"attributes\":{\"ref_id\":\"b9\"},\"end\":7652,\"start\":7651},{\"attributes\":{\"ref_id\":\"b10\"},\"end\":7853,\"start\":7852},{\"attributes\":{\"ref_id\":\"b10\"},\"end\":7972,\"start\":7971},{\"attributes\":{\"ref_id\":\"b10\"},\"end\":8240,\"start\":8239},{\"attributes\":{\"ref_id\":\"b11\"},\"end\":8639,\"start\":8638},{\"attributes\":{\"ref_id\":\"b12\"},\"end\":8872,\"start\":8870},{\"attributes\":{\"ref_id\":\"b13\"},\"end\":9082,\"start\":9080},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":9194,\"start\":9193},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":9314,\"start\":9313},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":9410,\"start\":9409},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":9560,\"start\":9559},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":9805,\"start\":9804}]",
        "bibtitle":"[{\"end\":11118,\"start\":11075},{\"end\":11868,\"start\":11796},{\"end\":12268,\"start\":12206},{\"end\":14171,\"start\":14048},{\"end\":14559,\"start\":14501},{\"end\":15373,\"start\":15288},{\"end\":15777,\"start\":15652},{\"end\":16208,\"start\":16155}]",
        "bibvenue":"[{\"end\":11228,\"start\":11201},{\"end\":11701,\"start\":11687},{\"end\":11968,\"start\":11952},{\"end\":12374,\"start\":12362},{\"end\":12661,\"start\":12589},{\"end\":13061,\"start\":12747},{\"end\":13720,\"start\":13645},{\"end\":14243,\"start\":14230},{\"end\":14642,\"start\":14624},{\"end\":14929,\"start\":14835},{\"end\":15446,\"start\":15432},{\"end\":15869,\"start\":15859},{\"end\":16303,\"start\":16290}]",
        """

        bib_entries = {}
        if doc["annotations"]["bibentry"] is None:
            return bib_entries

        for e in doc["annotations"]["bibentry"]:
            e["start"], e["end"] = int(e["start"]), int(e["end"])
            if doc["annotations"]["bibtitle"] is None:
                continue

            title = list(self.selects_sub_spans(doc["annotations"]["bibtitle"], e["start"], e["end"]))
            if len(title) == 0:
                continue

            title = doc["text"][int(title[0]["start"]):int(title[0]["end"])]

            if doc["annotations"]["bibauthor"] is None or doc["annotations"]["bibauthorfirstname"] is None \
                    or doc["annotations"]["bibauthorlastname"] is None:
                continue

            authors_spans = list(self.selects_sub_spans(doc["annotations"]["bibauthor"], e["start"], e["end"]))
            authors = []
            for a in authors_spans:
                parts = self.materialize_spans(
                    itertools.chain(
                        self.selects_sub_spans(doc["annotations"]["bibauthorfirstname"], a["start"], a["end"]),
                        self.selects_sub_spans(doc["annotations"]["bibauthorlastname"], a["start"], a["end"])
                    ), doc["text"])
                authors.append(" ".join(parts))

            new_bib_entry = {
                "title": title,
                "authors": authors,
                "year": None,  # the year seems to be missing from the data
                "link": int(e["attributes"]["matched_paper_id"]) if "matched_paper_id" in e["attributes"] else None
            }

            if new_bib_entry["link"] is not None:
                try:
                    the_doc = json_loads(self._metadata[new_bib_entry["link"]])
                    new_bib_entry["year"] = the_doc["year"]
                except KeyError:
                    pass

            bib_entries[e["attributes"]["id"]] = new_bib_entry

        return bib_entries

    def assemble_non_plaintext_content(self, doc: Dict) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """
        Assembles non-plain text content for a document (e.g. tables, figures, etc.)

        :param doc: s2orc document content representation (values of 'content' key)
        :return:
            non-plain text
                list of tuples with type and description
            mapping to index in the list of non-plain text content from text identifiers
        """
        #
        non_plaintext_content = []
        non_plaintext_content_index_map = {}

        if doc["annotations"]["figure"]:
            for i, e in enumerate(doc["annotations"]["figure"]):
                if "figurecaption" not in doc["annotations"] or doc["annotations"]["figurecaption"] is None:
                    caption = ""
                else:
                    caption = list(self.selects_sub_spans(doc["annotations"]["figurecaption"], e["start"], e["end"]))
                    if len(caption) > 0:
                        caption = list(self.materialize_spans(caption, doc["text"]))[0]
                    else:
                        caption = ""
                non_plaintext_content_index_map[e["attributes"]["id"]] = len(non_plaintext_content)
                non_plaintext_content.append(("figure", caption))

        if doc["annotations"]["table"]:
            for i, e in enumerate(doc["annotations"]["table"]):
                non_plaintext_content_index_map[f"tab_{i}"] = len(non_plaintext_content)
                non_plaintext_content.append(("table", ""))  # there is no caption for tables

        return non_plaintext_content, non_plaintext_content_index_map

    def assemble_content(self, json_s2orc: Optional[Dict], metadata: Dict) -> Dict[str, Any]:
        """
        Assembles content for a document.
        Content is:
        - abstract
        - body text
        - bibliography
        - non plain text (e.g. tables, figures, etc.)

        :param json_s2orc: s2orc document content representation
        :param metadata: is used to obtain
            title of the document
            abstract of the document
        :return: content
        """

        """
        {
           "corpusid":257292514,
           "externalids":{
              "arxiv":null,
              "mag":null,
              "acl":null,
              "pubmed":"37203959",
              "pubmedcentral":"10231538",
              "dblp":null,
              "doi":"10.31729/jnma.7979"
           },
           "content":{
              "source":{
                 "pdfurls":null,
                 "pdfsha":"a432bcc6aa5c09372ef68d73a4a903bfb5bb81eb",
                 "oainfo":null
              },
              "text":"\nJNMA I VOL 61 I ISSUE 259 I MARCH 2023 259 Free Full Text Articles are Available at www.jnma.com.np CASE\n2023\n\nDrAnkita Simkhada ankitasimkhada@gmail.com@phone:977-9849235922. \nAnkita Simkhada \nDepartment of Pathology, Maharajgunj Medical Campus\nMaharajgunj, KathmanduNepal\n\nRamesh Paudel \nDepartment of Pathology, Maharajgunj Medical Campus\nMaharajgunj, KathmanduNepal\n\nNisha Sharma \nDepartment of Pathology, Maharajgunj Medical Campus\nMaharajgunj, KathmanduNepal\n\n\nDepartment of Pathology\nMaharajgunj Medical Campus\nKathmanduNepal\n\nJNMA I VOL 61 I ISSUE 259 I MARCH 2023 259 Free Full Text Articles are Available at www.jnma.com.np CASE\n\nREPORT J Nepal Med Assoc\n61259202310.31729/jnma.7979CC BY ______________________________________ Correspondence:case reportscongenital mesoblastic nephromakidney neoplasmsnephrectomy\nCongenital mesoblastic nephromas are rare renal tumours that are encountered in paediatric age group. A term female neonate at the end of first week of life presented with bilateral lower limb swelling. On radiological evaluation, ultrasonography revealed an intra-abdominal mass which was managed with radical nephroureterectomy. Histopathological examination confirmed a diagnosis of congenital mesoblastic nephroma of mixed subtype.\n\nINTRODUCTION\n\nWilm's tumour accounts for approximately 90% of paediatric renal tumours and is the most common renal neoplasm in children between one to four years of age. 1 Renal tumours are rare in children less than three months of age. Among neonates, congenital mesoblastic nephroma (CMN), is more common. 2 CMN can be associated with antenatal complications such as polyhydramnios, preterm labour and premature delivery. 3 It is often recognized in the newborn who presents with an abdominal mass. Radical nephrectomy is the treatment of choice with adjuvant chemotherapy recommended for the more aggressive cellular or mixed variants. 4 We report a rare case of congenital mesoblastic nephroma in a term neonate born via normal vaginal delivery.\n\n\nCASE REPORT\n\nA term female neonate at ten days of her life was taken to a peripheral health facility with complaints of bilateral swelling of lower limbs. She weighed 2500 g at birth and was reported to have cried immediately after birth. During the antenatal period, the mother had regular antenatal checkups and had an uneventful pregnancy.\n\nAt the health centre, ultrasonography of lower limbs showed findings suggestive of cellulitis and abdominal ultrasonography showed a large exophytic mass measuring 57x54 mm in the upper pole of the right kidney. She was then referred to a tertiary care centre for further management.\n\nComputed Tomography (CT) scan of the chest and abdomen was done which revealed a heterogeneously enhancing, well defined soft tissue density space occupying lesion in the right abdominal cavity arising from the right kidney, extending from the subhepatic region to the pelvic brim. The tumour was seen crossing the midline and displacing the bowel loops towards the left abdominal cavity. Areas of calcification were not noted. Findings from bilateral lung and mediastinum were normal.\n\nWith these findings, a provisional diagnosis of Wilm's tumour was made and oncology consultation was done following which she was planned for a radical nephrectomy. Preoperative blood investigations including liver, renal function tests and complete blood counts were all normal. She underwent radical right nephroureterectomy, the mass was resected as a whole and sent to the Department of Pathology at another tertiary care centre for histopathological examination.\n\nOn gross examination, the right kidney measured 11x10x9 cm which showed a mass measuring 8x8x7 cm which extended from the upper up to the lower pole of the kidney. On the cut section, the mass was solid, grey white with focal cystic areas. The right ureter, measuring 3.5x0.5x0.5 cm was sent in a separate container and was grossly unremarkable (Figure 1). On microscopic examination, the tumour was arranged predominantly in sheets and consisted of highly cellular areas with densely packed tumour cells as well as focal areas showing interlacing fascicles of tumour cells ( Figure 2).\n\n\nFigure 2. Fascicles of spindle shaped cells are observed (left) along with more cellular areas with ovoid cells (right). (Hematoxylin and eosin stain x 100).\n\nMitotic figures were 5/10 high-power field (HPF). Tumour was limited to the kidney with intact overlying renal capsule and hilar vessels as well as the ureter were free of tumour. Focally, small area of normal renal parenchyma showing glomeruli and tubules was also seen, adjacent to the tumor (Figure 3). Tumour cells had a scant amount of eosinophilic cytoplasm with oval to spindled vesicular nuclei and inconspicuous to small prominent nucleoli in some of the cells (Figure 4).\n\n\nFigure 4. Tumour cells with scant cytoplasm, oval vesicular nuclei and inconspicuous nucleoli. (Hematoxylin and eosin stain x 400).\n\nA diagnosis of congenital mesoblastic nephroma, mixed subtype was made. She was discharged seven days post-surgery with stable vitals. Follow up at the paediatric surgical outpatient department was advised, with a plan to start adjuvant chemotherapy with actinomycin D and vincristine on follow up.\n\nOne year post surgery, the child is doing well with no signs of recurrence on follow up ultrasonography. However, as the condition of the child has improved and the parents are happy about the progress, after discussion with the treating clinicians, they decided to skip chemotherapeutic treatment.\n\n\nDISCUSSION\n\nCMN is an uncommon mesenchymal neoplasm that represents 2-4% of all paediatric renal tumors. 5 It is the most common congenital renal tumour. 2 Around 90% cases are diagnosed within the first year of life. 5 Neonates most commonly present with an intraabdominal mass. 2 Antenatal complications such as polyhydramnios, preterm labour and premature delivery may also be associated in around 71% cases. 3 Paraneoplastic syndromes including hypertension and hypercalcemia, though uncommon, have also been reported. 2 Increased renin synthesis by the tumor cells is thought to be the cause of hypertension whereas secretion of parathormone like proteins is responsible for hypercalcemia. 2 Due to routine use of ultrasonography, most cases are picked up during the antenatal period and the management plan initiated accordingly. In our case, tumour was diagnosed during the postnatal period when the child presented with lower limb swelling and ultrasonographic investigation for the same Free Full Text Articles are Available at www.jnma.com.np incidentally revealed a right sided renal mass.\n\nCongenital mesoblastic nephroma are low grade fibroblastic neoplasms that arise from the infantile renal sinus. These are more common than Wilm's tumor during the neonatal period however the two entities may not be distinguishable radiologically. On imaging, cases of classic CMN appear as homogeneous solid mass with \"ring\" sign of concentric hypoechoic and hyperechoic rims surrounding the tumour. 6 Cellular variant in contrast appear heterogeneously enhancing with fluid filled spaces representing haemorrhage, necrosis and cystic changes. 6 On macroscopic examination, classic CMN exhibit a solid firm grey white tumour with whorled appearance in contrast to the cellular variants in which the mass is soft, cystic with areas of haemorrhage and necrosis. Among the three histological variants of CMN, cellular variant is most common, representing 66% of cases. 5 The classic variant accounts for 24% of the cases and the mixed subtype is least common, accounting for 10% cases. 7 Morphologically, tumours in classic CMN resemble infantile myofibromatosis and are composed of fascicles of fibroblastic cells with tapered nuclei, eosinophilic cytoplasm and a low mitotic activity. 8 In contrast, cellular variant of CMN morphologically has a greater cellularity and resembles infantile fibrosarcoma. 8 Tumour cells arranged in ill formed fascicles give rise to sheet-like pattern of tumour cells. Tumour cells are plump, ovoid, with vesicular nuclei and moderate amount of cytoplasm. Mitotic activity is usually high and areas of necrosis and haemorrhage may be seen. 8 Mixed CMN demonstrates features of both cellular as well as classic CMNs. In keeping with this, our case also showed tumour cells composed of dense areas of ovoid shaped cells in addition to spindle shaped cells arranged in fascicles. Tumour cells in CMN are positive for vimentin, desmin and actin but negative for CD34 and epithelial markers, however diagnosis is primarily based on morphology. 9 Recent genetic studies have shown differences between classic and cellular variants of CMN. The cellular variant shows a translocation (12;15) (p13;q25) with formation of ETV6-NTRK3 gene which is not seen in the cellular variant. 10 The treatment of choice for CMN is radical nephrectomy. Nephrectomy alone is usually sufficient and only around 5% patients develop recurrence that is mainly related to the completeness of tumour resection. 11 Chances of recurrence are higher in patients are older than 3 months of age with the cellular variant of CMN. 4 Recurrences are also common when surgical margins are positive or if there has been an intraoperative tumour rupture. 4 All of these cases as well as cases of relapse are eligible to receive adjuvant chemotherapy. 4 Combinations of vincristine, cyclophosphamide and doxorubincin (VCD), vincristine, doxorubicin and actinomycin D (VDA) have been successfully used. 4 The differential diagnosis for renal tumours in paediatric population commonly include Wilm's tumour, congenital mesoblastic nephroma, clear cell sarcoma of kidney (CSSK), ossifying renal tumour of kidney (ORTI) and malignant rhabdoid tumour. 2 Neuroblastomas can also be a differential diagnosis for congenital tumours and may invade the kidney or rarely even arise from the renal parenchyma. In cases of CMN, rarity of this tumour and unfamiliarity with cases can pose a diagnostic difficulty. Radiological features and morphological findings must be used synergistically to reach a diagnosis and the pathologist must always keep in mind the possibility of CMN in case of a renal tumour in neonates.\n\nCMN are the most common congenital renal neoplasms which are mostly diagnosed in the first year of life. These are low grade neoplasms that carry excellent prognosis with radical nephrectomy. Surgical management is adequate for classic variants however adjuvant chemotherapy is recommended for the more aggressive cellular and mixed variant of CMN.\n\nFigure 1 .\n1Gross section of resected right kidney showing a grey white solid tumour occupying the whole kidney with focal areas of cystic change.\n\nFigure 3 .\n3The tumour (left) kidney (right) interface is seen.(Hematoxylin and eosin stain x 100).\nFree Full Text Articles are Available at www.jnma.com.np\nConsent: JNMA Case Report Consent Form was signed by the patient, and the original article is attached with the patient's chart.Conflict of Interest: None.\nCongenital mesoblastic nephroma: case study. W Kimani, E Ashiundu, P W Saula, M Kimondo, K Keitany, 10.1016/j.epsc.2019.101336J Pediatr Surg Case Reports. 55101336Full Text | DOIKimani W, Ashiundu E, Saula PW, Kimondo M, Keitany K. Congenital mesoblastic nephroma: case study. J Pediatr Surg Case Reports. 2020;55:101336. [Full Text | DOI]\n\n. R D Glick, M J Hicks, J G Nuchtern, D E Wesson, O O Olutoye, D L Cass, Renal tumors in infants less than 6 months of ageGlick RD, Hicks MJ, Nuchtern JG, Wesson DE, Olutoye OO, Cass DL. Renal tumors in infants less than 6 months of age.\n\n. 10.1016/j.jpedsurg.2003.12.007J Pediatr Surg. 394PubMed | Full Text | DOIJ Pediatr Surg. 2004 Apr;39(4):522-5. [PubMed | Full Text | DOI]\n\nThe congenital mesoblastic nephroma: a case report of prenatal diagnosis. B Haddad, J Haziza, C Touboul, M Abdellilah, S Uzan, B J Paniel, 10.1159/000264281Fetal Diagn Ther. 111PubMed | Full Text | DOIHaddad B, Haziza J, Touboul C, Abdellilah M, Uzan S, Paniel BJ. The congenital mesoblastic nephroma: a case report of prenatal diagnosis. Fetal Diagn Ther. 1996 Jan-Feb;11(1):61-6. [PubMed | Full Text | DOI]\n\nPart I: primary malignant non-wilms' renal tumours in children. H U Ahmed, M Arya, G Levitt, P G Duffy, I Mushtaq, N J Sebire, 10.1016/s1470-2045(07)70241-3Lancet Oncol. 88PubMed | Full Text | DOIAhmed HU, Arya M, Levitt G, Duffy PG, Mushtaq I, Sebire NJ. Part I: primary malignant non-wilms' renal tumours in children. Lancet Oncol. 2007 Aug;8(8):730-7. [PubMed | Full Text | DOI]\n\nFree Full Text Articles are Available at www.jnma.com.np © The Author(s). 2023Free Full Text Articles are Available at www.jnma.com.np © The Author(s) 2023.\n\nThe images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in the credit line; if the material is not included under the Creative Commons license, users will need to obtain permission from the license holder to reproduce the material. This work is licensed under a Creative Commons Attribution 4.0 International License. To view a copy of this license, visit https://This work is licensed under a Creative Commons Attribution 4.0 International License. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in the credit line; if the material is not included under the Creative Commons license, users will need to obtain permission from the license holder to reproduce the material. To view a copy of this license, visit https://\n\nWHO classification of tumours of the urinary system and male genital organs. H Moch, P A Humphrey, T M Ulbright, V Reuter, 356Lyon, France: International Agency for Research on Cancer. Full TextMoch H, Humphrey PA, Ulbright TM, Reuter V. WHO classification of tumours of the urinary system and male genital organs. 4th ed. Lyon, France: International Agency for Research on Cancer. p. 356. [Full Text]\n\nRenal tumors of childhood: radiologic-pathologic correlation part 1. The 1st Decade: From the Radiologic Pathology Archives. E M Chung, A R Graeber, R M Conran, 10.1148/rg.2016150230Radiographics. 362PubMed | Full Text | DOIChung EM, Graeber AR, Conran RM. Renal tumors of childhood: radiologic-pathologic correlation part 1. The 1st Decade: From the Radiologic Pathology Archives. Radiographics. 2016 Mar-Apr;36(2):499-522. [PubMed | Full Text | DOI]\n\nCongenital mesoblastic nephroma: a rare pediatric neoplasm. V Mallya, R Arora, K Gupta, U Sharma, 10.5146/tjpath.2013.01149Turk Patoloji Derg. 291PubMed | Full Text | DOIMallya V, Arora R, Gupta K, Sharma U. Congenital mesoblastic nephroma: a rare pediatric neoplasm. Turk Patoloji Derg. 2013;29(1):58-60. [PubMed | Full Text | DOI]\n\nRenal tumors of childhood-a histopathologic pattern-based diagnostic approach. Cancers (Basel). Ahag Ooms, G M Vujanic, E Hooghe, P Collini, A Hermine-Coulomb, C Vokuhl, 10.3390/cancers1203072912729PubMed | Full Text | DOIOoms AHAG, Vujanic GM, D Hooghe E, Collini P, L Hermine-Coulomb A, Vokuhl C, et al. Renal tumors of childhood-a histopathologic pattern-based diagnostic approach. Cancers (Basel). 2020 Mar 19;12(3):729. [PubMed | Full Text | DOI]\n\nPaediatric renal tumours: recent developments, new entities and pathological features. N J Sebire, G M Vujanic, 10.1111/j.1365-2559.2008.03110.xHistopathology. 545PubMed | Full Text | DOISebire NJ, Vujanic GM. Paediatric renal tumours: recent developments, new entities and pathological features. Histopathology. 2009 Apr;54(5):516-28. [PubMed | Full Text | DOI]\n\nETV6-NTRK3 gene fusions and trisomy 11 establish a histogenetic link between mesoblastic nephroma and congenital fibrosarcoma. S R Knezevich, M J Garnett, T J Pysher, J B Beckwith, P E Grundy, P H Sorensen, Cancer Res. 5822PubMed | Full TextKnezevich SR, Garnett MJ, Pysher TJ, Beckwith JB, Grundy PE, Sorensen PH. ETV6-NTRK3 gene fusions and trisomy 11 establish a histogenetic link between mesoblastic nephroma and congenital fibrosarcoma. Cancer Res. 1998 Nov 15;58(22):5046-8. [PubMed | Full Text]\n\nCellular congenital mesoblastic nephroma: case report. L G Santos, S Carvalho Jde, M A Reis, R L Sales, 10.1590/S0101-28002011000100014J Bras Nefrol. 331PubMed | Full Text | DOISantos LG, Carvalho Jde S, Reis MA, Sales RL. Cellular congenital mesoblastic nephroma: case report. J Bras Nefrol. 2011 Mar;33(1):109-12. [PubMed | Full Text | DOI]\n",
              "annotations":{
                 "abstract":"[{\"end\":1259,\"start\":824}]",
                 "author":"[{\"end\":177,\"start\":112},{\"end\":275,\"start\":178},{\"end\":371,\"start\":276},{\"end\":466,\"start\":372},{\"end\":534,\"start\":467}]",
                 "authoraffiliation":"[{\"end\":274,\"start\":195},{\"end\":370,\"start\":291},{\"end\":465,\"start\":386},{\"end\":533,\"start\":468}]",
                 "authorfirstname":"[{\"end\":120,\"start\":114},{\"end\":184,\"start\":178},{\"end\":282,\"start\":276},{\"end\":377,\"start\":372}]",
                 "authorlastname":"[{\"end\":129,\"start\":121},{\"end\":193,\"start\":185},{\"end\":289,\"start\":283},{\"end\":384,\"start\":378}]",
                 "bibauthor":"[{\"end\":11130,\"start\":11120},{\"end\":11142,\"start\":11130},{\"end\":11153,\"start\":11142},{\"end\":11164,\"start\":11153},{\"end\":11175,\"start\":11164},{\"end\":11429,\"start\":11418},{\"end\":11440,\"start\":11429},{\"end\":11454,\"start\":11440},{\"end\":11466,\"start\":11454},{\"end\":11479,\"start\":11466},{\"end\":11489,\"start\":11479},{\"end\":11880,\"start\":11870},{\"end\":11890,\"start\":11880},{\"end\":11901,\"start\":11890},{\"end\":11915,\"start\":11901},{\"end\":11923,\"start\":11915},{\"end\":11935,\"start\":11923},{\"end\":12281,\"start\":12270},{\"end\":12289,\"start\":12281},{\"end\":12299,\"start\":12289},{\"end\":12310,\"start\":12299},{\"end\":12321,\"start\":12310},{\"end\":12333,\"start\":12321},{\"end\":13730,\"start\":13722},{\"end\":13744,\"start\":13730},{\"end\":13758,\"start\":13744},{\"end\":13768,\"start\":13758},{\"end\":14184,\"start\":14173},{\"end\":14197,\"start\":14184},{\"end\":14209,\"start\":14197},{\"end\":14571,\"start\":14561},{\"end\":14580,\"start\":14571},{\"end\":14589,\"start\":14580},{\"end\":14599,\"start\":14589},{\"end\":14942,\"start\":14931},{\"end\":14955,\"start\":14942},{\"end\":14965,\"start\":14955},{\"end\":14976,\"start\":14965},{\"end\":14995,\"start\":14976},{\"end\":15005,\"start\":14995},{\"end\":15387,\"start\":15375},{\"end\":15400,\"start\":15387},{\"end\":15794,\"start\":15779},{\"end\":15807,\"start\":15794},{\"end\":15819,\"start\":15807},{\"end\":15833,\"start\":15819},{\"end\":15845,\"start\":15833},{\"end\":15859,\"start\":15845},{\"end\":16222,\"start\":16210},{\"end\":16238,\"start\":16222},{\"end\":16248,\"start\":16238},{\"end\":16259,\"start\":16248}]",
                 "bibauthorfirstname":"[{\"end\":11121,\"start\":11120},{\"end\":11131,\"start\":11130},{\"end\":11143,\"start\":11142},{\"end\":11145,\"start\":11144},{\"end\":11154,\"start\":11153},{\"end\":11165,\"start\":11164},{\"end\":11419,\"start\":11418},{\"end\":11421,\"start\":11420},{\"end\":11430,\"start\":11429},{\"end\":11432,\"start\":11431},{\"end\":11441,\"start\":11440},{\"end\":11443,\"start\":11442},{\"end\":11455,\"start\":11454},{\"end\":11457,\"start\":11456},{\"end\":11467,\"start\":11466},{\"end\":11469,\"start\":11468},{\"end\":11480,\"start\":11479},{\"end\":11482,\"start\":11481},{\"end\":11871,\"start\":11870},{\"end\":11881,\"start\":11880},{\"end\":11891,\"start\":11890},{\"end\":11902,\"start\":11901},{\"end\":11916,\"start\":11915},{\"end\":11924,\"start\":11923},{\"end\":11926,\"start\":11925},{\"end\":12271,\"start\":12270},{\"end\":12273,\"start\":12272},{\"end\":12282,\"start\":12281},{\"end\":12290,\"start\":12289},{\"end\":12300,\"start\":12299},{\"end\":12302,\"start\":12301},{\"end\":12311,\"start\":12310},{\"end\":12322,\"start\":12321},{\"end\":12324,\"start\":12323},{\"end\":13723,\"start\":13722},{\"end\":13731,\"start\":13730},{\"end\":13733,\"start\":13732},{\"end\":13745,\"start\":13744},{\"end\":13747,\"start\":13746},{\"end\":13759,\"start\":13758},{\"end\":14174,\"start\":14173},{\"end\":14176,\"start\":14175},{\"end\":14185,\"start\":14184},{\"end\":14187,\"start\":14186},{\"end\":14198,\"start\":14197},{\"end\":14200,\"start\":14199},{\"end\":14562,\"start\":14561},{\"end\":14572,\"start\":14571},{\"end\":14581,\"start\":14580},{\"end\":14590,\"start\":14589},{\"end\":14935,\"start\":14931},{\"end\":14943,\"start\":14942},{\"end\":14945,\"start\":14944},{\"end\":14956,\"start\":14955},{\"end\":14966,\"start\":14965},{\"end\":14977,\"start\":14976},{\"end\":14996,\"start\":14995},{\"end\":15376,\"start\":15375},{\"end\":15378,\"start\":15377},{\"end\":15388,\"start\":15387},{\"end\":15390,\"start\":15389},{\"end\":15780,\"start\":15779},{\"end\":15782,\"start\":15781},{\"end\":15795,\"start\":15794},{\"end\":15797,\"start\":15796},{\"end\":15808,\"start\":15807},{\"end\":15810,\"start\":15809},{\"end\":15820,\"start\":15819},{\"end\":15822,\"start\":15821},{\"end\":15834,\"start\":15833},{\"end\":15836,\"start\":15835},{\"end\":15846,\"start\":15845},{\"end\":15848,\"start\":15847},{\"end\":16211,\"start\":16210},{\"end\":16213,\"start\":16212},{\"end\":16223,\"start\":16222},{\"end\":16239,\"start\":16238},{\"end\":16241,\"start\":16240},{\"end\":16249,\"start\":16248},{\"end\":16251,\"start\":16250}]",
                 "bibauthorlastname":"[{\"end\":11128,\"start\":11122},{\"end\":11140,\"start\":11132},{\"end\":11151,\"start\":11146},{\"end\":11162,\"start\":11155},{\"end\":11173,\"start\":11166},{\"end\":11427,\"start\":11422},{\"end\":11438,\"start\":11433},{\"end\":11452,\"start\":11444},{\"end\":11464,\"start\":11458},{\"end\":11477,\"start\":11470},{\"end\":11487,\"start\":11483},{\"end\":11878,\"start\":11872},{\"end\":11888,\"start\":11882},{\"end\":11899,\"start\":11892},{\"end\":11913,\"start\":11903},{\"end\":11921,\"start\":11917},{\"end\":11933,\"start\":11927},{\"end\":12279,\"start\":12274},{\"end\":12287,\"start\":12283},{\"end\":12297,\"start\":12291},{\"end\":12308,\"start\":12303},{\"end\":12319,\"start\":12312},{\"end\":12331,\"start\":12325},{\"end\":13728,\"start\":13724},{\"end\":13742,\"start\":13734},{\"end\":13756,\"start\":13748},{\"end\":13766,\"start\":13760},{\"end\":14182,\"start\":14177},{\"end\":14195,\"start\":14188},{\"end\":14207,\"start\":14201},{\"end\":14569,\"start\":14563},{\"end\":14578,\"start\":14573},{\"end\":14587,\"start\":14582},{\"end\":14597,\"start\":14591},{\"end\":14940,\"start\":14936},{\"end\":14953,\"start\":14946},{\"end\":14963,\"start\":14957},{\"end\":14974,\"start\":14967},{\"end\":14993,\"start\":14978},{\"end\":15003,\"start\":14997},{\"end\":15385,\"start\":15379},{\"end\":15398,\"start\":15391},{\"end\":15792,\"start\":15783},{\"end\":15805,\"start\":15798},{\"end\":15817,\"start\":15811},{\"end\":15831,\"start\":15823},{\"end\":15843,\"start\":15837},{\"end\":15857,\"start\":15849},{\"end\":16220,\"start\":16214},{\"end\":16236,\"start\":16224},{\"end\":16246,\"start\":16242},{\"end\":16257,\"start\":16252}]",
                 "bibentry":"[{\"attributes\":{\"doi\":\"10.1016/j.epsc.2019.101336\",\"id\":\"b0\",\"matched_paper_id\":209253920},\"end\":11414,\"start\":11075},{\"attributes\":{\"id\":\"b1\"},\"end\":11653,\"start\":11416},{\"attributes\":{\"doi\":\"10.1016/j.jpedsurg.2003.12.007\",\"id\":\"b2\"},\"end\":11794,\"start\":11655},{\"attributes\":{\"doi\":\"10.1159/000264281\",\"id\":\"b3\",\"matched_paper_id\":46824664},\"end\":12204,\"start\":11796},{\"attributes\":{\"doi\":\"10.1016/s1470-2045(07)70241-3\",\"id\":\"b4\",\"matched_paper_id\":38559830},\"end\":12587,\"start\":12206},{\"attributes\":{\"id\":\"b5\"},\"end\":12745,\"start\":12589},{\"attributes\":{\"id\":\"b6\"},\"end\":13643,\"start\":12747},{\"attributes\":{\"id\":\"b7\"},\"end\":14046,\"start\":13645},{\"attributes\":{\"doi\":\"10.1148/rg.2016150230\",\"id\":\"b8\",\"matched_paper_id\":25121195},\"end\":14499,\"start\":14048},{\"attributes\":{\"doi\":\"10.5146/tjpath.2013.01149\",\"id\":\"b9\"},\"end\":14833,\"start\":14501},{\"attributes\":{\"doi\":\"10.3390/cancers12030729\",\"id\":\"b10\"},\"end\":15286,\"start\":14835},{\"attributes\":{\"doi\":\"10.1111/j.1365-2559.2008.03110.x\",\"id\":\"b11\",\"matched_paper_id\":205299815},\"end\":15650,\"start\":15288},{\"attributes\":{\"id\":\"b12\",\"matched_paper_id\":11198070},\"end\":16153,\"start\":15652},{\"attributes\":{\"doi\":\"10.1590/S0101-28002011000100014\",\"id\":\"b13\",\"matched_paper_id\":3098497},\"end\":16497,\"start\":16155}]",
                 "bibref":"[{\"attributes\":{\"ref_id\":\"b0\"},\"end\":1433,\"start\":1432},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":1572,\"start\":1571},{\"attributes\":{\"ref_id\":\"b3\"},\"end\":1688,\"start\":1687},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":1903,\"start\":1902},{\"attributes\":{\"ref_id\":\"b7\"},\"end\":5672,\"start\":5671},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":5721,\"start\":5720},{\"attributes\":{\"ref_id\":\"b7\"},\"end\":5785,\"start\":5784},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":5847,\"start\":5846},{\"attributes\":{\"ref_id\":\"b3\"},\"end\":5979,\"start\":5978},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":6090,\"start\":6089},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":6262,\"start\":6261},{\"attributes\":{\"ref_id\":\"b8\"},\"end\":7069,\"start\":7068},{\"attributes\":{\"ref_id\":\"b8\"},\"end\":7213,\"start\":7212},{\"attributes\":{\"ref_id\":\"b7\"},\"end\":7535,\"start\":7534},{\"attributes\":{\"ref_id\":\"b9\"},\"end\":7652,\"start\":7651},{\"attributes\":{\"ref_id\":\"b10\"},\"end\":7853,\"start\":7852},{\"attributes\":{\"ref_id\":\"b10\"},\"end\":7972,\"start\":7971},{\"attributes\":{\"ref_id\":\"b10\"},\"end\":8240,\"start\":8239},{\"attributes\":{\"ref_id\":\"b11\"},\"end\":8639,\"start\":8638},{\"attributes\":{\"ref_id\":\"b12\"},\"end\":8872,\"start\":8870},{\"attributes\":{\"ref_id\":\"b13\"},\"end\":9082,\"start\":9080},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":9194,\"start\":9193},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":9314,\"start\":9313},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":9410,\"start\":9409},{\"attributes\":{\"ref_id\":\"b4\"},\"end\":9560,\"start\":9559},{\"attributes\":{\"ref_id\":\"b1\"},\"end\":9805,\"start\":9804}]",
                 "bibtitle":"[{\"end\":11118,\"start\":11075},{\"end\":11868,\"start\":11796},{\"end\":12268,\"start\":12206},{\"end\":14171,\"start\":14048},{\"end\":14559,\"start\":14501},{\"end\":15373,\"start\":15288},{\"end\":15777,\"start\":15652},{\"end\":16208,\"start\":16155}]",
                 "bibvenue":"[{\"end\":11228,\"start\":11201},{\"end\":11701,\"start\":11687},{\"end\":11968,\"start\":11952},{\"end\":12374,\"start\":12362},{\"end\":12661,\"start\":12589},{\"end\":13061,\"start\":12747},{\"end\":13720,\"start\":13645},{\"end\":14243,\"start\":14230},{\"end\":14642,\"start\":14624},{\"end\":14929,\"start\":14835},{\"end\":15446,\"start\":15432},{\"end\":15869,\"start\":15859},{\"end\":16303,\"start\":16290}]",
                 "figure":"[{\"attributes\":{\"id\":\"fig_0\"},\"end\":10760,\"start\":10613},{\"attributes\":{\"id\":\"fig_1\"},\"end\":10861,\"start\":10761}]",
                 "figurecaption":"[{\"end\":10760,\"start\":10626},{\"end\":10861,\"start\":10774}]",
                 "figureref":"[{\"attributes\":{\"ref_id\":\"fig_0\"},\"end\":3954,\"start\":3945},{\"end\":4184,\"start\":4176},{\"attributes\":{\"ref_id\":\"fig_1\"},\"end\":4652,\"start\":4642},{\"end\":4827,\"start\":4818}]",
                 "formula":"[{\"attributes\":{\"id\":\"formula_0\"},\"end\":8235,\"start\":8200},{\"attributes\":{\"id\":\"formula_1\"},\"end\":8493,\"start\":8432},{\"attributes\":{\"id\":\"formula_2\"},\"end\":8889,\"start\":8876},{\"attributes\":{\"id\":\"formula_3\"},\"end\":9162,\"start\":9110},{\"attributes\":{\"id\":\"formula_4\"},\"end\":9632,\"start\":9572},{\"attributes\":{\"id\":\"formula_5\"},\"end\":9756,\"start\":9682},{\"attributes\":{\"id\":\"formula_6\"},\"end\":10459,\"start\":10357},{\"attributes\":{\"id\":\"formula_7\"},\"end\":10685,\"start\":10574},{\"attributes\":{\"id\":\"formula_8\"},\"end\":11121,\"start\":11049},{\"attributes\":{\"id\":\"formula_9\"},\"end\":11872,\"start\":11644},{\"attributes\":{\"id\":\"formula_10\"},\"end\":12737,\"start\":12612},{\"attributes\":{\"id\":\"formula_11\"},\"end\":13234,\"start\":13182},{\"attributes\":{\"id\":\"formula_12\"},\"end\":13465,\"start\":13424},{\"attributes\":{\"id\":\"formula_13\"},\"end\":13692,\"start\":13615},{\"attributes\":{\"id\":\"formula_14\"},\"end\":14211,\"start\":13982},{\"attributes\":{\"id\":\"formula_15\"},\"end\":14615,\"start\":14509},{\"attributes\":{\"id\":\"formula_16\"},\"end\":15118,\"start\":14868},{\"attributes\":{\"id\":\"formula_17\"},\"end\":15358,\"start\":15337},{\"attributes\":{\"id\":\"formula_18\"},\"end\":15668,\"start\":15497},{\"attributes\":{\"id\":\"formula_19\"},\"end\":16089,\"start\":16059},{\"attributes\":{\"id\":\"formula_20\"},\"end\":16292,\"start\":16215},{\"attributes\":{\"id\":\"formula_21\"},\"end\":16393,\"start\":16327},{\"attributes\":{\"id\":\"formula_22\"},\"end\":17283,\"start\":17187},{\"attributes\":{\"id\":\"formula_23\"},\"end\":17462,\"start\":17400},{\"attributes\":{\"id\":\"formula_24\"},\"end\":17807,\"start\":17643},{\"attributes\":{\"id\":\"formula_25\"},\"end\":18064,\"start\":18003},{\"attributes\":{\"id\":\"formula_26\"},\"end\":18265,\"start\":18184},{\"attributes\":{\"id\":\"formula_27\"},\"end\":18462,\"start\":18343},{\"attributes\":{\"id\":\"formula_28\"},\"end\":18690,\"start\":18600},{\"attributes\":{\"id\":\"formula_29\"},\"end\":18920,\"start\":18788},{\"attributes\":{\"id\":\"formula_30\"},\"end\":19228,\"start\":19186},{\"attributes\":{\"id\":\"formula_31\"},\"end\":19381,\"start\":19292},{\"attributes\":{\"id\":\"formula_32\"},\"end\":19786,\"start\":19760},{\"attributes\":{\"id\":\"formula_33\"},\"end\":19951,\"start\":19858},{\"attributes\":{\"id\":\"formula_34\"},\"end\":20169,\"start\":20088},{\"attributes\":{\"id\":\"formula_35\"},\"end\":20394,\"start\":20313},{\"attributes\":{\"id\":\"formula_36\"},\"end\":21340,\"start\":21319},{\"attributes\":{\"id\":\"formula_37\"},\"end\":21404,\"start\":21377},{\"attributes\":{\"id\":\"formula_38\"},\"end\":21599,\"start\":21538},{\"attributes\":{\"id\":\"formula_39\"},\"end\":21747,\"start\":21694},{\"attributes\":{\"id\":\"formula_40\"},\"end\":21808,\"start\":21747},{\"attributes\":{\"id\":\"formula_41\"},\"end\":21980,\"start\":21907},{\"attributes\":{\"id\":\"formula_42\"},\"end\":22152,\"start\":22058},{\"attributes\":{\"id\":\"formula_43\"},\"end\":22354,\"start\":22247},{\"attributes\":{\"id\":\"formula_44\"},\"end\":22628,\"start\":22451},{\"attributes\":{\"id\":\"formula_45\"},\"end\":22870,\"start\":22707},{\"attributes\":{\"id\":\"formula_46\"},\"end\":22973,\"start\":22923},{\"attributes\":{\"id\":\"formula_47\"},\"end\":23163,\"start\":23081},{\"attributes\":{\"id\":\"formula_48\"},\"end\":23339,\"start\":23230},{\"attributes\":{\"id\":\"formula_49\"},\"end\":25266,\"start\":25228},{\"attributes\":{\"id\":\"formula_50\"},\"end\":25595,\"start\":25535},{\"attributes\":{\"id\":\"formula_51\"},\"end\":25703,\"start\":25668},{\"attributes\":{\"id\":\"formula_52\"},\"end\":26597,\"start\":26370},{\"attributes\":{\"id\":\"formula_53\"},\"end\":31038,\"start\":31001},{\"attributes\":{\"id\":\"formula_54\"},\"end\":31918,\"start\":31871},{\"attributes\":{\"id\":\"formula_56\"},\"end\":37376,\"start\":37359},{\"attributes\":{\"id\":\"formula_0\"},\"end\":8235,\"start\":8200},{\"attributes\":{\"id\":\"formula_1\"},\"end\":8493,\"start\":8432},{\"attributes\":{\"id\":\"formula_2\"},\"end\":8889,\"start\":8876},{\"attributes\":{\"id\":\"formula_3\"},\"end\":9162,\"start\":9110},{\"attributes\":{\"id\":\"formula_4\"},\"end\":9632,\"start\":9572},{\"attributes\":{\"id\":\"formula_5\"},\"end\":9756,\"start\":9682},{\"attributes\":{\"id\":\"formula_6\"},\"end\":10459,\"start\":10357},{\"attributes\":{\"id\":\"formula_7\"},\"end\":10685,\"start\":10574},{\"attributes\":{\"id\":\"formula_8\"},\"end\":11121,\"start\":11049},{\"attributes\":{\"id\":\"formula_9\"},\"end\":11872,\"start\":11644},{\"attributes\":{\"id\":\"formula_10\"},\"end\":12737,\"start\":12612},{\"attributes\":{\"id\":\"formula_11\"},\"end\":13234,\"start\":13182},{\"attributes\":{\"id\":\"formula_12\"},\"end\":13465,\"start\":13424},{\"attributes\":{\"id\":\"formula_13\"},\"end\":13692,\"start\":13615},{\"attributes\":{\"id\":\"formula_14\"},\"end\":14211,\"start\":13982},{\"attributes\":{\"id\":\"formula_15\"},\"end\":14615,\"start\":14509},{\"attributes\":{\"id\":\"formula_16\"},\"end\":15118,\"start\":14868},{\"attributes\":{\"id\":\"formula_17\"},\"end\":15358,\"start\":15337},{\"attributes\":{\"id\":\"formula_18\"},\"end\":15668,\"start\":15497},{\"attributes\":{\"id\":\"formula_19\"},\"end\":16089,\"start\":16059},{\"attributes\":{\"id\":\"formula_20\"},\"end\":16292,\"start\":16215},{\"attributes\":{\"id\":\"formula_21\"},\"end\":16393,\"start\":16327},{\"attributes\":{\"id\":\"formula_22\"},\"end\":17283,\"start\":17187},{\"attributes\":{\"id\":\"formula_23\"},\"end\":17462,\"start\":17400},{\"attributes\":{\"id\":\"formula_24\"},\"end\":17807,\"start\":17643},{\"attributes\":{\"id\":\"formula_25\"},\"end\":18064,\"start\":18003},{\"attributes\":{\"id\":\"formula_26\"},\"end\":18265,\"start\":18184},{\"attributes\":{\"id\":\"formula_27\"},\"end\":18462,\"start\":18343},{\"attributes\":{\"id\":\"formula_28\"},\"end\":18690,\"start\":18600},{\"attributes\":{\"id\":\"formula_29\"},\"end\":18920,\"start\":18788},{\"attributes\":{\"id\":\"formula_30\"},\"end\":19228,\"start\":19186},{\"attributes\":{\"id\":\"formula_31\"},\"end\":19381,\"start\":19292},{\"attributes\":{\"id\":\"formula_32\"},\"end\":19786,\"start\":19760},{\"attributes\":{\"id\":\"formula_33\"},\"end\":19951,\"start\":19858},{\"attributes\":{\"id\":\"formula_34\"},\"end\":20169,\"start\":20088},{\"attributes\":{\"id\":\"formula_35\"},\"end\":20394,\"start\":20313},{\"attributes\":{\"id\":\"formula_36\"},\"end\":21340,\"start\":21319},{\"attributes\":{\"id\":\"formula_37\"},\"end\":21404,\"start\":21377},{\"attributes\":{\"id\":\"formula_38\"},\"end\":21599,\"start\":21538},{\"attributes\":{\"id\":\"formula_39\"},\"end\":21747,\"start\":21694},{\"attributes\":{\"id\":\"formula_40\"},\"end\":21808,\"start\":21747},{\"attributes\":{\"id\":\"formula_41\"},\"end\":21980,\"start\":21907},{\"attributes\":{\"id\":\"formula_42\"},\"end\":22152,\"start\":22058},{\"attributes\":{\"id\":\"formula_43\"},\"end\":22354,\"start\":22247},{\"attributes\":{\"id\":\"formula_44\"},\"end\":22628,\"start\":22451},{\"attributes\":{\"id\":\"formula_45\"},\"end\":22870,\"start\":22707},{\"attributes\":{\"id\":\"formula_46\"},\"end\":22973,\"start\":22923},{\"attributes\":{\"id\":\"formula_47\"},\"end\":23163,\"start\":23081},{\"attributes\":{\"id\":\"formula_48\"},\"end\":23339,\"start\":23230},{\"attributes\":{\"id\":\"formula_49\"},\"end\":25266,\"start\":25228},{\"attributes\":{\"id\":\"formula_50\"},\"end\":25595,\"start\":25535},{\"attributes\":{\"id\":\"formula_51\"},\"end\":25703,\"start\":25668},{\"attributes\":{\"id\":\"formula_52\"},\"end\":26597,\"start\":26370},{\"attributes\":{\"id\":\"formula_53\"},\"end\":31038,\"start\":31001},{\"attributes\":{\"id\":\"formula_54\"},\"end\":31918,\"start\":31871},{\"attributes\":{\"id\":\"formula_56\"},\"end\":37376,\"start\":37359}]",
                 "paragraph":"[{\"end\":2012,\"start\":1275},{\"end\":2357,\"start\":2028},{\"end\":2642,\"start\":2359},{\"end\":3129,\"start\":2644},{\"end\":3598,\"start\":3131},{\"end\":4186,\"start\":3600},{\"end\":4829,\"start\":4348},{\"end\":5263,\"start\":4965},{\"end\":5563,\"start\":5265},{\"end\":6666,\"start\":5578},{\"end\":10262,\"start\":6668},{\"end\":10612,\"start\":10264}]",
                 "publisher":null,
                 "sectionheader":"[{\"end\":1273,\"start\":1261},{\"end\":2026,\"start\":2015},{\"end\":4346,\"start\":4189},{\"end\":4963,\"start\":4832},{\"end\":5576,\"start\":5566},{\"end\":10624,\"start\":10614},{\"end\":10772,\"start\":10762}]",
                 "table":"[{\"end\":47379,\"start\":46914},{\"end\":48010,\"start\":47545},{\"end\":48278,\"start\":48086},{\"end\":48784,\"start\":48392},{\"end\":49367,\"start\":48797},{\"end\":47379,\"start\":46914},{\"end\":48010,\"start\":47545},{\"end\":48278,\"start\":48086},{\"end\":48784,\"start\":48392},{\"end\":49367,\"start\":48797}]",
                 "tableref":"[{\"attributes\":{\"ref_id\":\"tab_1\"},\"end\":28061,\"start\":28054},{\"attributes\":{\"ref_id\":\"tab_2\"},\"end\":28073,\"start\":28066},{\"attributes\":{\"ref_id\":\"tab_1\"},\"end\":28198,\"start\":28179},{\"attributes\":{\"ref_id\":\"tab_3\"},\"end\":32509,\"start\":32502},{\"attributes\":{\"ref_id\":\"tab_4\"},\"end\":35353,\"start\":35346},{\"attributes\":{\"ref_id\":\"tab_5\"},\"end\":35365,\"start\":35358},{\"attributes\":{\"ref_id\":\"tab_4\"},\"end\":35472,\"start\":35465},{\"attributes\":{\"ref_id\":\"tab_5\"},\"end\":35586,\"start\":35579},{\"attributes\":{\"ref_id\":\"tab_1\"},\"end\":28061,\"start\":28054},{\"attributes\":{\"ref_id\":\"tab_2\"},\"end\":28073,\"start\":28066},{\"attributes\":{\"ref_id\":\"tab_1\"},\"end\":28198,\"start\":28179},{\"attributes\":{\"ref_id\":\"tab_3\"},\"end\":32509,\"start\":32502},{\"attributes\":{\"ref_id\":\"tab_4\"},\"end\":35353,\"start\":35346},{\"attributes\":{\"ref_id\":\"tab_5\"},\"end\":35365,\"start\":35358},{\"attributes\":{\"ref_id\":\"tab_4\"},\"end\":35472,\"start\":35465},{\"attributes\":{\"ref_id\":\"tab_5\"},\"end\":35586,\"start\":35579}]",
                 "title":"[{\"end\":105,\"start\":1},{\"end\":639,\"start\":535}]",
                 "venue":"[{\"end\":665,\"start\":641}]"
              }
           },
           "updated":"2023-06-02T05:22:08Z"
        }
        """

        start = time.time()
        try:
            abstract = [
                {"type": "paragraph", "section": "Abstract", "text": metadata["abstract"], "cite_spans": [],
                 "ref_spans": [], "start": None, "end": None}] if metadata["abstract"] is not None else []
        except KeyError:
            abstract = []

        """example of json_abstract:
        {
           "corpusid":244907350,
           "openaccessinfo":{
              "externalids":{
                 "MAG":null,
                 "ACL":null,
                 "DOI":null,
                 "PubMedCentral":null,
                 "ArXiv":null
              },
              "license":null,
              "url":null,
              "status":null
           },
           "abstract":"Hydrodynamic cavitation is an effective means to ensure the intensification of various processes carried out in liquid media. Analysis of the results of existing and specially designed experiments to elucidate the mechanisms of the effect of cavitation on the properties of the liquid, let V.M. Ivčenko and E.D. Malimon allocate following hydrodynamic phenomena by which may be obtained by technological effects: [1]",
           "updated":"2022-02-07T20:04:42.392Z"
        }
        """

        paragraphs = []
        bib_entries = {}
        non_plaintext_content = []
        non_plaintext_content_index_map = {}

        if json_s2orc is not None:

            for k, v in json_s2orc["content"]["annotations"].items():
                # for some reason it is a string
                if v is not None:
                    try:
                        v = json_loads(v)
                        if isinstance(v, list):
                            for i, x in enumerate(v):
                                if isinstance(x, dict):
                                    if "start" in x and "end" in x:
                                        v[i]["start"] = int(v[i]["start"])
                                        v[i]["end"] = int(v[i]["end"])

                        json_s2orc["content"]["annotations"][k] = v
                    except JSONDecodeError:
                        pass

            if len(abstract) == 0 and json_s2orc["content"]["annotations"]["abstract"] is not None:
                # there were observed doubled spans in abstracts
                abstract_spans = sorted(
                    set((x["start"], x["end"]) for x in json_s2orc["content"]["annotations"]["abstract"]),
                    key=lambda x: x[0])

                abstract_spans = [{"start": int(x[0]), "end": int(x[1])} for x in
                                  abstract_spans]  # int() is needed as sometimes the int is a string

                abstract = [
                    {"type": "paragraph", "section": "Abstract", "text": x, "cite_spans": [], "ref_spans": [],
                     "start": abstract_spans[i]["start"], "end": abstract_spans[i]["start"]} for i, x in
                    enumerate(self.materialize_spans(abstract_spans, json_s2orc["content"]["text"]))
                ]

            paragraphs = self.assemble_paragraphs(json_s2orc["content"])

            bib_entries = self.assemble_bib_entries(json_s2orc["content"])

            non_plaintext_content, non_plaintext_content_index_map = \
                self.assemble_non_plaintext_content(json_s2orc["content"])

        hierarchy = Hierarchy(metadata["title"], [])

        if self.stub_mode:
            content_iter = abstract
            for s in content_iter:  # ensures that there is at least one non-empty
                if s["text"]:
                    break
            else:
                content_iter = []
                for s in paragraphs:  # ensures that there is at least one non-empty
                    if s["text"]:
                        content_iter = [s]
                        break
        else:
            content_iter = list(itertools.chain(abstract, paragraphs))

            # dehyphenate
            for i, p in enumerate(content_iter):
                rep_spans, replace_with = self.dehyphenator.replacements(p["text"])

                new_text, new_spans = replace_at(p["text"], rep_spans, replace_with,
                                                 [
                                                     [(c["start"], c["end"]) for c in p["cite_spans"]],
                                                     [(c["start"], c["end"]) for c in p["ref_spans"]]
                                                 ],
                                                 SpanCollisionHandling.SKIP)
                p["text"] = new_text

                for j, c in enumerate(p["cite_spans"]):
                    c["start"], c["end"] = new_spans[0][j]

                for j, c in enumerate(p["ref_spans"]):
                    c["start"], c["end"] = new_spans[1][j]

            # clean titles
            for bi, e in bib_entries.items():
                e["title"] = clean_title(e["title"])

        if not self.stub_mode:
            self.prepare_content_time += time.time() - start

        start = time.time()
        # split into sentences
        sections_sentences = list(
            d.sents for d in self.spacy.pipe((s["text"] for s in content_iter), batch_size=self.spacy_batch_size)
        )

        if not self.stub_mode:
            self.split_into_sentences_time += time.time() - start

        start = time.time()

        bibliography_list = list(bib_entries)
        citations = set()
        last_headline = None
        first_proper_subsection = False
        for i_s, s in enumerate(content_iter):
            act_citations = []

            for c in s["cite_spans"]:
                try:
                    act_citations.append(
                        RefSpan(bibliography_list.index(c["attributes"]["ref_id"]), c["start"], c["end"]))
                    b = bib_entries[c["attributes"]["ref_id"]]
                    bib_id = int(b["link"]) if b["link"] is not None else None
                    if bib_id is not None:
                        citations.add(bib_id)

                except (ValueError, KeyError):
                    # missing bib reference
                    act_citations.append(RefSpan(None, c["start"], c["end"]))

            if last_headline is None or last_headline != s["section"]:
                if s["section"] == "" and not first_proper_subsection:
                    last_headline = s["section"]
                else:
                    if not first_proper_subsection and s["section"] != "" and s["section"] != "Abstract":
                        first_proper_subsection = True
                    last_headline = s["section"]
                    hierarchy.content.append(Hierarchy(headline=s["section"], content=[]))

            try:
                if s["type"] == "formula":
                    h = Hierarchy("formula", TextContent(s["text"], [], []))
                    if len(hierarchy.content) > 0 and not isinstance(hierarchy.content[-1].content, TextContent):
                        hierarchy.content[-1].content.append(h)
                    else:
                        hierarchy.content.append(h)
                    continue
            except KeyError as e:
                print(s, file=sys.stderr, flush=True)
                raise e

            paragraph_hierarchy = Hierarchy(None, content=[])
            for sentence in sections_sentences[i_s]:
                sent_start, sent_end = sentence.start_char, sentence.end_char
                sentences_citations = []

                # tables are total mess in this dataset, there are missing captions
                # sometimes there is not an id for a table in tableref, sometimes there is not even id attribute

                sentences_references = [
                    RefSpan(
                        non_plaintext_content_index_map[r["attributes"]["ref_id"]]
                        if "attributes" in r and "ref_id" in r["attributes"] and r["attributes"][
                            "ref_id"] in non_plaintext_content_index_map
                        else None,
                        r["start"] - sent_start,
                        r["end"] - sent_start
                    )
                    for r in s["ref_spans"] if r["start"] >= sent_start and r["end"] <= sent_end
                ]

                for c in act_citations:
                    if c.start >= sent_start and c.end <= sent_end:
                        c.start = c.start - sent_start
                        c.end = c.end - sent_start
                        sentences_citations.append(c)

                paragraph_hierarchy.content.append(
                    Hierarchy(None, TextContent(
                        s["text"][sent_start:sent_end],
                        citations=sentences_citations,
                        references=sentences_references)
                              )
                )
            if first_proper_subsection or s["section"] == "Abstract":
                hierarchy.content[-1].content.append(paragraph_hierarchy)
            else:
                hierarchy.content.append(paragraph_hierarchy)

        if not self.stub_mode:
            self.create_hierarchy_time += time.time() - start

        return {
            "hierarchy": hierarchy,
            "non_plaintext_content": non_plaintext_content,
            "bibliography": [
                BibEntry(int(b["link"]) if b["link"] is not None else None, b["title"], int(b["year"]) if b["year"] is not None else None,
                         tuple(b["authors"]))
                for b in bib_entries.values()
            ]
        }

    def _get_item(self, item: int) -> Union[Document, Any]:
        """
        Get document from dataset.

        :param item: line number (from zero)
        :return: the document or transformed document
        """
        doc_id = self._ids[item]

        if self.preload_filter is not None and not self.preload_filter(item, doc_id):
            return None

        metadata = self._metadata[doc_id]
        try:
            s2orc = self._s2orc[doc_id]
        except KeyError:
            s2orc = None

        self.cnt += 1
        start_total = time.time()

        start = time.time()
        d_data = self.assemble_metadata(doc_id, json_loads(metadata))

        if not self.stub_mode:
            self.assemble_metadata_time += time.time() - start

        json_s2orc = None
        if s2orc is not None:
            json_s2orc = json_loads(s2orc)

        start = time.time()
        d_data.update(self.assemble_content(json_s2orc, d_data))
        if not self.stub_mode:
            self.assemble_content_time += time.time() - start

        start = time.time()
        if not self.stub_mode:
            # add missing bib entries using citations
            citations_in_bib = set(b.id for b in d_data["bibliography"] if b.id is not None)
            cite_set = set(d_data["citations"])
            for c in cite_set - citations_in_bib:
                try:
                    json_metadata = json_loads(self._metadata[c])
                except KeyError:
                    continue

                d_data["bibliography"].append(
                    BibEntry(
                        c,
                        json_metadata["title"],
                        json_metadata["year"],
                        tuple(a["name"] for a in json_metadata["authors"])
                    )
                )

            d_data["citations"] = sorted(cite_set | citations_in_bib)

        if not self.stub_mode:
            self.add_missing_bib_time += time.time() - start
        # remove extra arguments
        del d_data["abstract"]
        d_data["uncategorized_fields"] = {
            "origin": "s2orc",
        }
        d = Document(**d_data)

        if not self.stub_mode:
            start = time.time()
            enhance_citations_spans(d.hierarchy, interpolate=True)
            self.enhance_citations_time += time.time() - start

        self.total_time += time.time() - start_total
        if self.profile and not self.stub_mode and random.random() < 0.01:
            print(f"cnt: {self.cnt}, assemble_metadata_time: {self.assemble_metadata_time/self.cnt}, "
                  f"assemble_content_time: {self.assemble_content_time/self.cnt}, "
                  f"prepare_content_time: {self.prepare_content_time/self.cnt}, "
                  f"split_into_sentences_time: {self.split_into_sentences_time/self.cnt}, "
                  f"create_hierarchy_time: {self.create_hierarchy_time/self.cnt}, "
                  f"add_missing_bib_time: {self.add_missing_bib_time/self.cnt}, "
                  f"enhance_citations_time: {self.enhance_citations_time/self.cnt}, "
                  f"total_time: {self.total_time/self.cnt}",
                  file=sys.stderr, flush=True)

        return self.apply_transform(d, item)


class OldS2ORCDocumentDataset(RandomLineAccessFile, DocumentDataset):
    """
    S2ORC dataset reader.

    S2ORC is dataset presented in
        http://dx.doi.org/10.18653/v1/2020.acl-main.447
    . The official github is https://github.com/allenai/s2orc.

    Example:

        with S2ORCDocumentDataset("example.jsonl") as dataset:
            print("Document on line 150 is:", dataset[150])

    """
    lock = multiprocessing.Lock()

    def __init__(self, path_to: str, workers: int = 0, chunk_size: int = 10):
        """
        initialization of dataset

        :param path_to: path to dataset
        :param workers: activates multiprocessing and determines number of workers that should be used
            the multiprocessing is used during iteration through whole dataset
        :param chunk_size: chunk size for single process when the multiprocessing is activated
        """
        DocumentDataset.__init__(self)
        super().__init__(path_to)
        self._dirty = True  # we want to return content that is different from the one in original file

        if workers > 0:
            self._lines = multiprocessing.Array(ctypes.c_int64, self._lines, lock=False)

        try:
            self._prepare_spacy()
        except OSError:
            with self.lock:
                spacy.cli.download("en_core_web_sm")
                self._prepare_spacy()

        self._workers = workers
        self.chunk_size = chunk_size
        self.max_chunks_per_worker = 10_000
        self.dehyphenator = DeHyphenator()

    def __iter__(self) -> Generator[Union[Document, Any], None, None]:
        """
        sequence iteration over whole file
        :return: generator of documents or transformed documents when the transformation is activated
        """
        yield from self.iter_range()

    def iter_range(self, f: int = 0, t: Optional[int] = None,
                   unordered: bool = False) -> Generator[Union[Document, Any], None, None]:
        """
        sequence iteration over given range
        :param f: from
        :param t: to
        :param unordered: if True the documents are not returned in order
            might speed up the iteration in multiprocessing mode
        :return: generator of documents or transformed documents when the transformation is activated
        """
        if self.closed:
            raise RuntimeError("Firstly open the file.")

        with nullcontext() if self._workers <= 0 else \
                FactoryFunctorPool(self._workers, DatasetMultProcWorkerFactory(self, self.max_chunks_per_worker),
                                   results_queue_maxsize=1.0, verbose=True, join_timeout=5) as pool:

            m = partial(map, self._get_item) if self._workers <= 0 \
                else partial(pool.imap_unordered if unordered else pool.imap, chunk_size=self.chunk_size)

            if t is None:
                t = len(self)

            for document in m(range(f, t)):
                yield document

    def _prepare_spacy(self):
        """
        Loads the spacy models and selects the components we need.
        """
        self._spacy = spacy.load("en_core_sci_sm")
        self._spacy_stub = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        self._spacy_stub.enable_pipe("senter")

    @property
    def spacy(self) -> spacy.Language:
        """
        Returns spacy model instance.
        """
        if self.stub_mode:
            return self._spacy_stub
        return self._spacy

    @staticmethod
    def convert_author_name(a: Dict) -> str:
        """
        Converts author name from dict format on the input to the str.
        dict format:
            {"first": "Pengqian", "middle": [], "last": "Yu", "suffix": ""}
        string format:
            Pengqian Yu
        :param a: dictionary format of author name
        :return: str format of author name
        """
        return " ".join(x for x in [a["first"]] + a["middle"] + [a["last"]] + [a["suffix"]] if x)

    def _get_item(self, item: int) -> Union[Document, Any]:
        """
        Get document from dataset.

        :param item: line number (from zero)
        :return: the document or transformed document
        """

        # example of document (https://ieeexplore.ieee.org/document/7308013):
        # only interesting parts are shown
        #   {
        #       "paper_id": "18980380",
        #       "title": "Distributionally Robust Counterpart in Markov Decision Processes",
        #       "authors": [
        #           {"first": "Pengqian", "middle": [], "last": "Yu", "suffix": ""},
        #           {"first": "Huan", "middle": [], "last": "Xu", "suffix": ""}
        #       ],
        #       "year": 2016,
        #       "outbound_citations": [
        #           "7229756", "57464058", "9166388", "10603007", "2474018", "8946639", "1537485", "710328",
        #           "10308849", "207242061", "59762877", "15546892", "6103434", "63859912", "16625241", "486400",
        #           "11547182", "18980380", "37925315", "18576331", "24341930", "18980463"
        #       ],
        #       "abstract": [
        #           {
        #               "section": "Abstract",
        #               "text": "This technical note studies Markov decision processes under parameter...",
        #               "cite_spans": [],
        #               "ref_spans": []
        #           },
        #           {
        #               "section": "Abstract",
        #               "text": "Index Terms-Distributional robustness, Markov decision processes, parameter uncertainty.",
        #               "cite_spans": [],
        #               "ref_spans": []
        #           }
        #        ],
        #        "body_text": [
        #           {
        #               "section": "",
        #               "text": ". Illustration of the confidence sets.",
        #               "cite_spans": [],
        #               "ref_spans": []
        #           },
        #           {
        #               "section": "",
        #               "text": "optimizing variable and \u03be is the unknown parameter, distributionally ...",
        #               "cite_spans": [
        #                   {"start": 49, "end": 52, "text": "[1]", "ref_id": "BIBREF0"},
        #                   {"start": 58, "end": 61, "text": "[1]", "ref_id": "BIBREF0"},
        #                   {"start": 845, "end": 849, "text": "[18]", "ref_id": "BIBREF17"}
        #               ],
        #               "ref_spans": [
        #                   {"start": 337, "end": 346, "text": "Fig. 1(a)", "ref_id": "FIGREF2"},
        #                   {"start": 1166, "end": 1175, "text": "Fig. 1(b)", "ref_id": "FIGREF2"}
        #               ]
        #            },
        #           {
        #               "section": "II. PRELIMINARIES",
        #               "text": "Throughout the technical note, we use capital letters to denote matrices, ...",
        #               "cite_spans": [],
        #               "ref_spans": []},
        #           {
        #               "section": "II. PRELIMINARIES",
        #               "text": "A (finite) Markov Decision Process (MDP) is defined as a 6-tuple T, \u03b3, S, A,...",
        #               "cite_spans": [],
        #               "ref_spans": []
        #           },
        #           ...
        #
        #           "bib_entries": {
        #               "BIBREF0": {
        #                   "title": "Distributionally robust Markov decision processes",
        #                   "authors": [
        #                       {"first": "H", "middle": [], "last": "Xu", "suffix": ""},
        #                       {"first": "S", "middle": [], "last": "Mannor", "suffix": ""}
        #                   ],
        #                   "year": 2012,
        #                   "venue": "Math. Oper. Res",
        #                   "link": "7229756"
        #                },
        #                "BIBREF1": {
        #                   "title": "Markov Decision Processes: Discrete Stochastic Dynamic Programming",
        #                   "authors": [
        #                       {"first": "M", "middle": ["L"], "last": "Puterman", "suffix": ""}
        #                   ],
        #                   "year": 2014,
        #                   "venue": "",
        #                   "link": "57464058"
        #                },
        #                ...
        #           },
        #           "ref_entries": {
        #               "FIGREF0": {
        #                   "text": "The condition 1 of Assumption 3 ensures the confidence set with largest ...",
        #                   "type": "figure"
        #               },
        #               "FIGREF1": {
        #                   "text": "Here, K i s * represents the cone dual to K i ...",
        #                   "type": "figure"
        #               },
        #               "TABREF0": {
        #                   "text": "TOTAL DISCOUNTED REWARDS AND COMPUTATIONAL TIMES OF NOMINAL,...",
        #                   "type": "table"
        #               }
        #           }
        #   }
        json_record = json_loads(RandomLineAccessFile._read_line(self, item))

        # "first": "Pengqian", "middle": [], "last": "Yu", "suffix": ""
        authors = [self.convert_author_name(a) for a in json_record["authors"]]

        non_plaintext_content = []
        non_plaintext_content_index_map = {}
        for i, e in enumerate(json_record["ref_entries"].items()):
            non_plaintext_content.append((e[1]["type"], e[1]["text"]))
            non_plaintext_content_index_map[e[0]] = i

        hierarchy = Hierarchy(json_record["title"], [])
        """
        #           {
        #               "section": "",
        #               "text": "optimizing variable and \u03be is the unknown parameter, distributionally ...",
        #               "cite_spans": [
        #                   {"start": 49, "end": 52, "text": "[1]", "ref_id": "BIBREF0"},
        #                   {"start": 58, "end": 61, "text": "[1]", "ref_id": "BIBREF0"},
        #                   {"start": 845, "end": 849, "text": "[18]", "ref_id": "BIBREF17"}
        #               ],
        #               "ref_spans": [
        #                   {"start": 337, "end": 346, "text": "Fig. 1(a)", "ref_id": "FIGREF2"},
        #                   {"start": 1166, "end": 1175, "text": "Fig. 1(b)", "ref_id": "FIGREF2"}
        #               ]
        #            }
        """

        last_headline = None
        citations = set(int(c) for c in json_record["outbound_citations"])

        if self.stub_mode:
            content_iter = list(json_record["abstract"])
            for s in content_iter:  # ensures that there is at least one non-empty
                if s["text"]:
                    break
            else:
                content_iter = []
                for s in json_record["body_text"]:  # ensures that there is at least one non-empty
                    if s["text"]:
                        content_iter = [s]
                        break
        else:
            content_iter = list(itertools.chain(json_record["abstract"], json_record["body_text"]))

            # dehyphenate
            for i, p in enumerate(content_iter):
                rep_spans, replace_with = self.dehyphenator.replacements(p["text"])
                new_text, new_spans = replace_at(p["text"], rep_spans, replace_with,
                                                 [
                                                     [(c["start"], c["end"]) for c in p["cite_spans"]],
                                                     [(c["start"], c["end"]) for c in p["ref_spans"]]
                                                 ],
                                                 SpanCollisionHandling.SKIP)
                p["text"] = new_text

                for j, c in enumerate(p["cite_spans"]):
                    c["start"], c["end"] = new_spans[0][j]

                for j, c in enumerate(p["ref_spans"]):
                    c["start"], c["end"] = new_spans[1][j]

            # clean titles
            for bi, e in json_record["bib_entries"].items():
                e["title"] = clean_title(e["title"])

        # split into sentences
        sections_sentences = list(
            d.sents for d in self.spacy.pipe(s["text"] for s in content_iter)
        )

        bibliography_list = list(json_record["bib_entries"])
        for i_s, s in enumerate(content_iter):
            act_citations = []

            for c in s["cite_spans"]:
                try:
                    act_citations.append(RefSpan(bibliography_list.index(c["ref_id"]), c["start"], c["end"]))
                    b = json_record["bib_entries"][c["ref_id"]]
                    bib_id = int(b["link"]) if b["link"] else None
                    if bib_id is not None:
                        citations.add(bib_id)

                except ValueError:
                    # missing bib reference
                    act_citations.append(RefSpan(None, c["start"], c["end"]))

            if last_headline is None or last_headline != s["section"]:
                last_headline = s["section"]
                hierarchy.content.append(Hierarchy(headline=s["section"], content=[]))

            paragraph_hierarchy = Hierarchy(None, content=[])
            for sentence in sections_sentences[i_s]:
                sent_start, sent_end = sentence.start_char, sentence.end_char
                sentences_citations = []
                sentences_references = [
                    RefSpan(
                        non_plaintext_content_index_map[r["ref_id"]],
                        r["start"] - sent_start,
                        r["end"] - sent_start
                    )
                    for r in s["ref_spans"] if r["start"] >= sent_start and r["end"] <= sent_end
                ]

                for c in act_citations:
                    if c.start >= sent_start and c.end <= sent_end:
                        c.start = c.start - sent_start
                        c.end = c.end - sent_start
                        sentences_citations.append(c)

                paragraph_hierarchy.content.append(
                    Hierarchy(None, TextContent(
                        s["text"][sent_start:sent_end],
                        citations=sentences_citations,
                        references=sentences_references)
                              )
                )
            hierarchy.content[-1].content.append(paragraph_hierarchy)

        mag_id = None
        if json_record["mag_id"] is not None:
            try:
                mag_id = int(json_record["mag_id"].split(",")[0])
            except ValueError:
                pass

        s2orc_id = int(json_record["paper_id"])
        d = Document(
            id=s2orc_id,
            s2orc_id=s2orc_id,
            mag_id=mag_id,
            doi=json_record["doi"],
            title=json_record["title"],
            authors=authors,
            year=None if json_record["year"] is None else int(json_record["year"]),
            fields_of_study=json_record["mag_field_of_study"],
            citations=sorted(citations),
            hierarchy=hierarchy,
            non_plaintext_content=non_plaintext_content,
            bibliography=[
                BibEntry(int(b["link"]) if b["link"] else None, b["title"], int(b["year"]) if b["year"] else None,
                         tuple(self.convert_author_name(a) for a in b["authors"]))
                for b in json_record["bib_entries"].values()
            ],
            uncategorized_fields={
                "origin": "s2orc_original",
            }
        )

        if not self.stub_mode:
            enhance_citations_spans(d.hierarchy, interpolate=True)

        return self.apply_transform(d, item)


class COREDocumentDataset(DocumentDataset):
    """
    CORE dataset reader.

    website: https://core.ac.uk/

    Example:

    >>>dataset=COREDocumentDataset(papers_list, mag_paper_list)
    >>>print("Document with id 150", dataset[150])

    """

    def __init__(self, papers_list: COREPapersList,
                 mag_paper_list: Optional[MAGPapersList],
                 s2orc_paper_list: Optional[PapersList[PapersListRecordWithAllIds]],
                 batch_size: int = 128, workers: int = 0,
                 chunk_size: int = 10):
        """
        Initializes core dataset reader.

        :param papers_list: list of papers in dataset
        :param mag_paper_list: mag paper list for extending references and fields of study
        :param s2orc_paper_list: s2orc paper list for obtaining s2orc ids
        :param batch_size: Maximal number of samples in a batch when searching in paper lists.
        :param workers: you can pass number of workers that will be used for parsing grobid documents in parallel fashion
            when iterating over all dataset
            you do not need to pass it here you can also just set it right before iteration
        :param chunk_size: chunk size for single process when the multiprocessing is activated
        """
        super().__init__()
        self._papers_list = papers_list
        self._mag_paper_list = mag_paper_list
        self.core_2_mag_index, self.mag_2_core_index = None, None
        if mag_paper_list is not None:
            self.core_2_mag_index, self.mag_2_core_index = self._papers_list.to_other_mapping(self._mag_paper_list,
                                                                                              batch_size=batch_size)

        self._s2orc_paper_list = s2orc_paper_list
        self.core_2_s2orc_index = None
        if s2orc_paper_list is not None:
            self.core_2_s2orc_index = self._papers_list.to_other_mapping(self._s2orc_paper_list, batch_size=batch_size,
                                                                         reverse=False)

        self.workers = workers
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.max_chunks_per_worker = 10_000

    @property
    def paths(self) -> List[str]:
        """
        Paths to files that are in this dataset.
        """
        return self._papers_list.get_paths()

    def __iter__(self) -> Generator[Union[Document, Any], None, None]:
        """
        sequence iteration over whole file
        :return: generator of documents or transformed documents when the transformation is activated
        """
        yield from self.iter_range()

    def iter_range(self, f: int = 0, t: Optional[int] = None,
                   unordered: bool = False) -> Generator[Union[Document, Any], None, None]:
        """
        sequence iteration over given range
        :param f: from
        :param t: to
        :param unordered: if True the documents are not returned in order
            might speed up the iteration in multiprocessing mode
        :return: generator of documents or transformed documents when the transformation is activated
        """
        with nullcontext() if self.workers <= 0 else \
                FactoryFunctorPool(self.workers, DatasetMultProcWorkerFactory(self, self.max_chunks_per_worker),
                                   results_queue_maxsize=10.0, verbose=True, join_timeout=5) as pool:
            m = partial(map, self._get_item) if self.workers <= 0 else partial(
                (pool.imap_unordered if unordered else pool.imap), chunk_size=self.chunk_size)
            if t is None:
                t = len(self)

            for document in m(range(f, t)):
                yield document

    def __len__(self):
        return len(self._papers_list)

    def get_core_references_from_mag(self, mag_i: int) -> List[int]:
        """
        Get core ids of referenced documents in MAG.
        Omit documents that are in MAG but not in CORE.

        :param mag_i: index of a document in mag_paper_list
        :return: core ids of referenced documents
        """

        referenced_in_core = []

        for c in self._mag_paper_list[mag_i].references:
            try:
                mag_index = self._mag_paper_list.id_2_index(c)
            except KeyError:
                # unknown mag record
                continue
            c_core_i = self.mag_2_core_index[mag_index]
            if c_core_i is not None:
                referenced_in_core.append(c_core_i)

        return referenced_in_core

    def _get_item(self, selector: int, grobid_doc: Optional[GROBIDDoc] = None) -> Union[Document, Any]:
        """
        Get document on given index.

        :param selector: index of a document
        :param grobid_doc: you may valuntary provide parsed grobid doc when the parsing is done in advance
        :return: document on givne index or transformed document
        """
        if self.preload_filter is not None and not self.preload_filter(selector, selector):
            return None

        if grobid_doc is None:
            grobid_doc = GROBIDDoc(self._papers_list.get_path(selector), self.stub_mode)

        if not self.stub_mode:
            grobid_doc.match_bibliography(selector, self._papers_list, self.batch_size)

        year = grobid_doc.year
        mag_id = None
        doi = None
        fields_of_study = []

        if self.stub_mode:
            citations = []
        else:
            citations = [bib_entry.id for bib_entry in grobid_doc.bibliography.values() if bib_entry.id is not None]

        authors = list(grobid_doc.authors)

        if self._mag_paper_list is not None:
            mag_i = self.core_2_mag_index[selector]
            if mag_i is not None:
                # there is a MAG record, so let's use it for enhancement
                mag_record = self._mag_paper_list[mag_i]  # faster than accessing fields separately
                if year is None:
                    year = mag_record.year

                mag_id = mag_record.id
                doi = mag_record.doi

                fields_of_study = mag_record.fields

                if not self.stub_mode:
                    citations.extend(self.get_core_references_from_mag(mag_i))
                    citations = sorted(set(citations))

                norm_authors = list(frozenset(normalize_and_tokenize_string(a)) for a in authors)
                for a in mag_record.authors:
                    if frozenset(normalize_and_tokenize_string(a)) not in norm_authors:
                        authors.append(a)

        s2orc_id = None

        if self._s2orc_paper_list is not None:
            s2orc_i = self.core_2_s2orc_index[selector]
            if s2orc_i is not None:
                s2orc_record = self._s2orc_paper_list[s2orc_i]
                s2orc_id = s2orc_record.id

        d = Document(
            id=selector,
            s2orc_id=s2orc_id,
            mag_id=mag_id,
            doi=doi,
            title=grobid_doc.title,
            authors=authors,
            year=year,
            fields_of_study=fields_of_study,
            citations=sorted(citations),
            hierarchy=grobid_doc.hierarchy,
            non_plaintext_content=grobid_doc.non_plaintext_content,
            bibliography=list(grobid_doc.bibliography.values()),
            uncategorized_fields={
                "origin": "core"
            }
        )

        enhance_citations_spans(d.hierarchy, interpolate=True)

        return self.apply_transform(d, selector)
