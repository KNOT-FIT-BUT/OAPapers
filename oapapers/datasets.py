# -*- coding: UTF-8 -*-
""""
Created on 21.04.22
Contains classes for reading OAReviews and OARelatedWork datasets.

:author:     Martin Dočekal
"""
import csv
import ctypes
import math
import multiprocessing
import os
import sys
import time
import traceback
from contextlib import nullcontext
from functools import partial
from pathlib import Path
import random
from typing import Dict, Generator, Tuple, List, Union, Any, Optional, Iterable

from windpyutils.files import MapAccessFile
from windpyutils.parallel.own_proc_pools import FunctorWorker, FactoryFunctorPool
from windpyutils.structures.sorted import SortedMap

from oapapers.document import Document, OARelatedWorkDocument
from oapapers.document_datasets import DocumentDataset, DatasetMultProcWorkerFactory
from oapapers.myjson import json_loads


class DatasetMultProcWorker(FunctorWorker):

    def __init__(self, dataset: "OADataset", max_chunks_per_worker: float = math.inf):
        super().__init__(max_chunks_per_worker)
        self.dataset = dataset

        self.cnt = 0
        self.last_time = None
        self.proc_time = 0
        self.wait_time = 0
        self.profile_prob = False

    def __call__(self, i: int) -> Document:
        """
        Obtains document on given index.

        :param i: index of given document
        :return: the document
        """

        self.cnt += 1
        if self.last_time is not None:
            self.wait_time += time.time() - self.last_time
        try:
            wait_time = time.time()
            res = self.dataset[i]
            self.proc_time += time.time() - wait_time

            if self.profile_prob > 0 and random.random() < self.profile_prob:
                print(f"{os.getpid()} SpawnDatasetMultProcWorker: {self.cnt} wait time: {self.wait_time/self.cnt} "
                      f"proc time: {self.proc_time/self.cnt}", file=sys.stderr, flush=True)

            self.last_time = time.time()
            return res
        except Exception as e:
            print(e)
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e

    def begin(self):
        self.dataset.open()
        if self.dataset.transform is not None and hasattr(self.dataset.transform, "begin"):
            self.dataset.transform.begin()

    def end(self):
        self.dataset.close()

        if self.dataset.transform is not None and hasattr(self.dataset.transform, "end"):
            self.dataset.transform.end()


class OAPapersMultProcWorkerFactory(DatasetMultProcWorkerFactory):
    """
    Factory for creating dataset workers.
    """

    def create(self) -> DatasetMultProcWorker:
        return DatasetMultProcWorker(self.dataset, self.max_chunks_per_worker)


class OADataset(MapAccessFile, DocumentDataset):
    """
    Provides access to documents in a OADataset.
    """

    def __init__(self, path_to: str, path_to_mapping: Optional[str] = None, workers: int = 0, chunk_size: int = 10,
                 hierarchy_as_dict: bool = False,
                 lazy_hierarchy: bool = False):
        """
        initialization
        Whole file itself is not loaded into memory.

        :param path_to: path to file
        :param path_to_mapping: Path to file with index in tsv format with following header:
                key\tfile_line_offset
                if the file does not exist the mapping will be created
        :param workers: you can pass number of workers that will be used for parsing when iterating over all dataset
            you do not need to pass it here you can also just set it right before iteration
            WARNING: if multiprocessing is activated, the internal mapping is converted to SortedMap
            Thus the keys and values are sorted and not in the same order as in the original mapping
        :param chunk_size: chunk size for single process when the multiprocessing is activated
        :param hierarchy_as_dict: if True the hierarchy is left as dictionary, otherwise it is converted to Hierarchy
        :param lazy_hierarchy: if True, the hier content is not parsed and instead a lazy hierarchy is created
        """
        if path_to_mapping is not None and Path(path_to_mapping).is_file():
            self.indices_2_id, self.indices_2_offsets, mapping = self.load_mappings(path_to_mapping)
        else:
            self.indices_2_id, self.indices_2_offsets, mapping = self.index_file(path_to)

        DocumentDataset.__init__(self)

        if workers > 0:
            self.indices_2_id = multiprocessing.Array(ctypes.c_int64, self.indices_2_id, lock=False)
            self.indices_2_offsets = multiprocessing.Array(ctypes.c_int64, self.indices_2_offsets, lock=False)
            mapping = SortedMap(mapping)  # save memory
            mapping.keys_storage = multiprocessing.Array(ctypes.c_int64, mapping.keys_storage, lock=False)
            mapping.values_storage = multiprocessing.Array(ctypes.c_int64, mapping.values_storage, lock=False)

        super().__init__(path_to, mapping, int)
        self.workers = workers
        self.chunk_size = chunk_size
        self.max_chunks_per_worker = math.inf
        self.hierarchy_as_dict = hierarchy_as_dict
        self.lazy_hierarchy = lazy_hierarchy

        self.cnt = 0
        self._read_time = 0
        self._json_time = 0
        self._wait_time = 0
        self._parse_doc_time = 0
        self._transform_time = 0
        self._last_time = None
        self.profile_prob = 0

    @staticmethod
    def index_file(path_to: str) -> Tuple[List[int], List[int], Dict[int, int]]:
        """
        Makes index of line offsets.

        :param path_to: Path to file that should be indexed.
        :return:
            the indices to id mapping
            the indices to line offset mapping
            id to line offset mapping
        """

        id_2_offset = {}
        indices_2_id = []
        indices_2_offsets = []
        last_line_offset = 0

        with open(path_to, "rb") as f:
            while line := f.readline():
                record = json_loads(line)
                id_2_offset[record["id"]] = last_line_offset
                indices_2_offsets.append(last_line_offset)
                indices_2_id.append(record["id"])
                last_line_offset = f.tell()

        return indices_2_id, indices_2_offsets, id_2_offset

    @staticmethod
    def load_mappings(p: str) -> Tuple[List[int], List[int], Dict[int, int]]:
        """
        Method for loading key->line offset mapping from tsv file with header key\tfile_line_offset.

        :param p: path to tsv file
        :return:
            indices 2 id mapping
            indices 2 line offset mapping
            id 2 line offset mapping
        """
        res = {}
        indices_2_id = []
        indices_2_offset = []
        with open(p, newline='') as f:
            for r in csv.DictReader(f, delimiter="\t"):
                current_id = int(r["key"])
                res[current_id] = int(r["file_line_offset"])
                indices_2_offset.append(int(r["file_line_offset"]))
                indices_2_id.append(current_id)

        return indices_2_id, indices_2_offset, res

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

        t = min(t, len(self))

        yield from self.iter_selected(range(f, t), unordered=unordered)

    def iter_selected(self, indices: Iterable[int], unordered: bool = False) -> Generator[Document, None, None]:
        """
        sequence iteration over given indices

        :param indices: indices of documents to iterate over
        :param unordered: if True the documents are not returned in order
            might speed up the iteration in multiprocessing mode
        :return: generator of documents or transformed documents when the transformation is activated
        """

        old_file = self.file
        try:
            if self.workers > 0:
                self.file = None  # must be because of spawning processes

            with nullcontext() if self.workers <= 0 else \
                    FactoryFunctorPool(self.workers, OAPapersMultProcWorkerFactory(self, self.max_chunks_per_worker),
                                       work_queue_maxsize=1.0, results_queue_maxsize=10.0, verbose=True,
                                       join_timeout=5) as pool:

                m = partial(map, self.__getitem__) if self.workers <= 0 else \
                    partial((pool.imap_unordered if unordered else pool.imap), chunk_size=self.chunk_size)

                if self.workers == 0 and hasattr(self.transform, "begin"):
                    # for multiprocessing it will be done in the worker
                    self.transform.begin()

                for document in m(indices):
                    yield document
                if self.workers == 0 and hasattr(self.transform, "end"):
                    self.transform.end()

        finally:
            self.file = old_file

    def __contains__(self, item):
        """
        :param item: id of a document
        """
        return item in self.mapping

    def __getitem__(self, selector: Union[int, slice]) -> Union[Document, List[Document]]:
        """
        Get document from dataset.

        :param selector: line number (from zero) or slice
        :return: the document, or list of documents when the slice is used
        """
        return DocumentDataset.__getitem__(self, selector)

    def get_by_id(self, i: int) -> Document:
        """
        Get document from dataset with given id

        :param i: id of a document
        :return: the document
        """
        return Document.from_dict(self._read_json(i), self.hierarchy_as_dict, self.lazy_hierarchy)

    def _get_item(self, selector: int) -> Union[Document, Any]:
        """
        Get document from dataset.

        :param selector: line index of a document
        :return: the document or transformed document
        """
        self.cnt += 1
        if self._last_time is not None:
            self._wait_time += time.time() - self._last_time

        start = time.time()
        self.file.seek(self.indices_2_offsets[selector])
        line = self.file.readline()
        self._read_time += time.time() - start

        start = time.time()
        json_record = json_loads(line)
        self._json_time += time.time() - start

        start = time.time()
        parse_doc = Document.from_dict(json_record, self.hierarchy_as_dict, self.lazy_hierarchy)
        self._parse_doc_time += time.time() - start

        start = time.time()
        transformed = self.apply_transform(parse_doc, selector)
        self._transform_time += time.time() - start

        if self.profile_prob > 0 and random.random() < self.profile_prob:
            print(f"{os.getpid()} {self.cnt} wait: {self._wait_time/self.cnt:.4f} "
                  f"read: {self._read_time/self.cnt:.2f} "
                  f"json: {self._json_time/self.cnt:.2f} "
                  f"parse: {self._parse_doc_time/self.cnt:.2f} "
                  f"transform: {self._transform_time/self.cnt:.2f} ",
                  flush=True, file=sys.stderr)
        self._last_time = time.time()
        return transformed

    def _read_json(self, selector: int) -> Dict:
        """
        Read json representation of a document.

        :param selector: id of a document
        :return: the json representation
        """
        return json_loads(super().__getitem__(selector))


class OARelatedWork(OADataset):
    """
    Class for OARelatedWork dataset. More concretely for the target/related work part.
    """

    def get_by_id(self, i: int) -> OARelatedWorkDocument:
        return OARelatedWorkDocument.from_dict(self._read_json(i), self.hierarchy_as_dict, self.lazy_hierarchy)

    def _get_item(self, selector: int) -> OARelatedWorkDocument:
        d = OARelatedWorkDocument.from_dict(self._read_json(self.indices_2_id[selector]), self.hierarchy_as_dict,
                                            self.lazy_hierarchy)
        return self.apply_transform(d, selector)

    def _read_json(self, selector: int) -> Dict:
        json_record = super()._read_json(selector)
        return json_record
