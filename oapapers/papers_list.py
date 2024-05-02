# -*- coding: UTF-8 -*-
"""
Created on 24.01.22
Representation of list of papers.

:author:     Martin Dočekal
"""
import copy
import csv
import itertools
import math
import multiprocessing
import os
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from multiprocessing.context import BaseContext
from multiprocessing.managers import BaseManager, SharedMemoryManager
from multiprocessing.pool import ThreadPool
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Tuple, List, Union, Set, Sequence, Optional, MutableSequence, Generator, Iterable, \
    Generic, TypeVar, Dict, Type
from xml.etree.ElementTree import ParseError

import faiss
import numpy as np
import scipy
import ijson
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
from windpyutils.files import MutableMemoryMappedRecordFile, Record, MapAccessFile
from windpyutils.generic import Batcher, BatcherIter
from windpyutils.parallel.own_proc_pools import FunctorWorker, FunctorPool
from windpyutils.structures.caches import LFUCache

import oapapers.grobid_doc as grobid_doc
from oapapers.cython.normalization import normalize_and_tokenize_string, initial_and_normalized_authors, \
    normalize_string, normalize_multiple_strings
from oapapers.matching import match_authors_groups
from oapapers.myjson import json_dumps, json_loads
from oapapers.similarities import similarity_score
import sqlite3


@dataclass
class PapersListRecord:
    """
    Single record for paper list.
    """
    __slots__ = ("title", "year", "authors")

    title: str
    year: Optional[int]
    authors: Sequence[str]

    @classmethod
    def load(cls, s: str) -> "PapersListRecord":

        return cls( # taking items separately is faster than iterating over kvitems
            title=next(ijson.items(s, "title")),
            year=next(ijson.items(s, "year")),
            authors=next(ijson.items(s, "authors"))
        )

    def save(self) -> str:
        return json_dumps({
            "title": self.title,
            "year": self.year,
            "authors": self.authors,
        })


@dataclass
class PapersListRecordWithId(PapersListRecord):
    """
    Single record for paper list with id.
    """
    __slots__ = ("id",)
    id: int

    @classmethod
    def load(cls, s: str) -> "PapersListRecordWithId":
        d = json_loads(s)
        return cls(
            id=d["id"],
            title=d["title"],
            year=d["year"],
            authors=d["authors"],
        )

    def save(self) -> str:
        return json_dumps({
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "authors": self.authors,
        })


@dataclass
class PapersListRecordWithAllIds(PapersListRecordWithId):
    """
    Single record for paper list with all ids.
    """
    __slots__ = ("s2orc_id", "mag_id", "doi")
    s2orc_id: Optional[int]  #: s2orc identifier of a document
    mag_id: Optional[int]  #: mag identifier of a document
    doi: Optional[str]  #: Digital Object Identifier

    @classmethod
    def load(cls, s: str) -> "PapersListRecordWithAllIds":
        d = json_loads(s)
        return cls(
            id=d["id"],
            title=d["title"],
            year=d["year"],
            authors=d["authors"],
            s2orc_id=d["s2orc_id"],
            mag_id=d["mag_id"],
            doi=d["doi"],
        )

    def save(self) -> str:
        return json_dumps({
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "authors": self.authors,
            "s2orc_id": self.s2orc_id,
            "mag_id": self.mag_id,
            "doi": self.doi,
        })


class PapersListRecordCachedStub(PapersListRecord):
    """
    Record for cached record papers that is used during initialization and contains only some attributes.
    Is used just as speed up optimization that saves loading from a disk.
    """

    def __init__(self, title: str):
        """
        Initialization of record with preloaded title.

        :param title: preloaded title
        """

        self.title = title

    @property
    def year(self):
        raise RuntimeError("The year property can not be obtained.")

    @property
    def authors(self):
        raise RuntimeError("The authors property can not be obtained.")


R = TypeVar("R", bound=PapersListRecord)


class PapersListRecordMutableMemoryMappedRecordFile(MutableMemoryMappedRecordFile[R]):
    """
    Like mutable MutableMemoryMappedRecordFile, but implements caching of titles for init phase. After
    calling flush_cache it starts to act like ordinary MutableMemoryMappedRecordFile.

    Is used just as speed up optimization that saves loading from a disk.
    """

    def __init__(self, path_to: str, titles: Optional[List[str]], read_cache_size: int = 65536,
                 record_class: Optional[Type[R]] = None, line_offsets: Optional[List[int]] = None):
        """
        :param path_to: path to record file
        :param titles: cached titles
        :param read_cache_size: maximal number of cached records to prevent reading from disk
        :param record_class: class of records
        :param line_offsets: line offsets
        """
        index_p = path_to + ".index"

        if line_offsets is None:
            if Path(index_p).exists():
                line_offsets = self.load_index_file(index_p)

        if record_class is None:
            record_class = PapersListRecord

        super().__init__(path_to, record_class, lines=line_offsets)
        self.titles = titles
        self._read_cache = LFUCache(read_cache_size)

    def __getitem__(self, selector: Union[int, slice, Iterable]) -> Union[
        R, "PapersListRecordMutableMemoryMappedRecordFile[R]"]:
        """
        Get n-th line from file.

        :param n: line index or slice
        :return: n-th line or list with lines subset in case of slice or iterable
        :raise RuntimeError: When the file is not opened.
        :raise IndexError: When the selector is invalid
        """
        if self.closed:
            raise RuntimeError("Firstly open the file.")

        if not isinstance(selector, int):
            return self.__class__(self.path_to, self.titles, self._read_cache.max_size, self.record_class,
                                  self._lines[selector])

        return self._get_item(selector)

    def move_data_to_manager(self, manager: multiprocessing.Manager):
        """
        Moves lines (line offsets or changes) to shared manager.

        :param manager: multiprocessing manager
        """
        self._lines = manager.list(self._lines)

        if self.titles is not None:
            self.titles = manager.list(self.titles)

    @property
    def lines(self) -> Optional[List[Union[int, str]]]:
        """
        It may contain line offsets or line content in form of string representation.
        """
        return self._lines

    @staticmethod
    def load_index_file(path_to: str) -> List[int]:
        """
        Loads index file and returns list of line offsets.

        :param path_to: path to index file
        :return: list of line offsets
        """
        line_offsets = []
        with open(path_to, newline='') as f:
            # skip header
            f.readline()
            for r in f:
                line_offsets.append(int(r.split("\t")[1]))

        return line_offsets

    def flush_cache(self):
        """
        Starts to act like ordinary MutableMemoryMappedRecordFile
        """
        self.titles = None

    def _get_item(self, n: int) -> R:
        if self.titles is None:
            try:
                return self._read_cache[n]
            except KeyError:
                record = super()._get_item(n)
                self._read_cache[n] = record
                return record
        return self.create_stub(n)

    def create_stub(self, n: int) -> PapersListRecordCachedStub:
        """
        Creates stub for record on given index.

        :param n: index of record
        :return: stub for record on given index
        """
        return PapersListRecordCachedStub(self.titles[n])


L = TypeVar("L", bound=PapersListRecord)


class SharedListOfRecords(Sequence[L]):
    """
    Shared list of records. It is used for multiprocessing.
    """

    def __init__(self, records: Union[Sequence[L], Sequence[str]], shared_memory_manager: SharedMemoryManager,
                 record_type: Type[L] = PapersListRecord):
        """
        :param records: list of records or list of json strings that are representing records
        :param shared_memory_manager: shared memory manager for creating shared memory
        :param record_type: type of records
        """

        self._offsets = []
        self._lengths = []

        if isinstance(records[0], str):
            records = [r.encode('utf-8') for r in records]
        else:
            records = [r.save().encode('utf-8') for r in records]

        offset = 0
        for r in records:
            self._offsets.append(offset)
            offset += len(r)
            self._lengths.append(len(r))
        str_form_records = b"".join(records)
        del records

        # copy to shared memory
        self._offsets = shared_memory_manager.ShareableList(self._offsets)
        self._lengths = shared_memory_manager.ShareableList(self._lengths)
        self._shared_memory = shared_memory_manager.SharedMemory(size=len(str_form_records))
        self._shared_memory.buf[:] = str_form_records
        self._record_type = record_type

    def __getitem__(self, item: Union[slice, int]) -> Union[List[L], L]:
        if isinstance(item, slice):
            return [self.parse(i) for i in range(len(self))[item]]

        return self.parse(item)

    def parse(self, item: int) -> L:
        """
        Parses record on given index.

        :param item: index of record
        :return: parsed record
        """
        json_r = bytes(self._shared_memory.buf[self._offsets[item]:self._offsets[item] + self._lengths[item]])

        return self._record_type.load(json_r.decode('utf-8'))

    def __len__(self) -> int:
        return len(self._offsets)


class ObtainFeaturesWorker(FunctorWorker):
    """
    Worker functor for obtaining features for records.
    """

    def __init__(self, records: Sequence[PapersListRecord], vectorizer: HashingVectorizer,
                 max_chunks_per_worker: float = math.inf):
        """
        :param records: records for obtaining features
        :param vectorizer: vectorizer for obtaining features
        :param max_chunks_per_worker: maximal number of chunks per worker
        """
        super().__init__(max_chunks_per_worker)
        self.records = records
        self.vectorizer = vectorizer

    def __call__(self, proc: Tuple[int, int]):
        """
        :param proc: tuple of start and end index of records for processing
        """
        titles = [r.title for r in self.records[proc[0]:proc[1]]]
        return self.vectorizer.transform(titles)


class ObtainNormalizedTitlesWorker(FunctorWorker):
    """
    Worker functor for obtaining normalized titles for records.
    """

    def __init__(self, records: Sequence[PapersListRecord], max_chunks_per_worker: float = math.inf):
        """
        :param records: records for obtaining features
        :param max_chunks_per_worker: maximal number of chunks per worker
        """
        super().__init__(max_chunks_per_worker)
        self.records = records

    def __call__(self, proc: Tuple[int, int]) -> List[str]:
        """
        :param proc: tuple of start and end index of records for processing
        :return: list of normalized titles
        """
        titles = [r.title for r in self.records[proc[0]:proc[1]]]
        return normalize_multiple_strings(titles)


class PapersList(Generic[L]):
    """
    base class for papers list
    It allows approximate search of paper by its title, year, and author.

    :ivar search_workers: Number of workers for searching.  You must use __enter__ and __exit__ methods
        to activate multiprocessing. Otherwise it is not used.
    :vartype search_workers: int
    """

    def __init__(self, records: MutableSequence[L], embed_dim=64, k_neighbors=100, match_threshold=0.75,
                 max_year_diff: int = 0, use_gpus: bool = True, fp16: bool = True, init_workers: int = 0,
                 features_part_size: int = 81_920, allow_gpus: Optional[Sequence[int]] = None,
                 embeddings: Optional[Union[scipy.sparse.spmatrix, SharedMemory]] = None,
                 stub_init: bool = False, cache_size: int = 0, fulltext_search: Optional[str] = None,
                 fulltext_search_uri_connections: bool = False):
        """
        Initialization of paper list.

        :param records: contains:
            title of publication is used for creating features for searching
            year of publication could be None in that case it matches with any year
            authors of publication, at least one author must match during searching
        :param embed_dim: Number of dimensions of title representation obtained be hashing vectorizer.
            This representation is used for nearest neighbors search.
        :param k_neighbors: Number of neighbors for nearest neighbors search of titles.
            is also usef for fulltext search as LIMIT for sqlite query.
        :param match_threshold: Score threshold for matching. All above or equal are ok.
            Matching is used for title and list of authors. For authors at least one of them must have a match.
            To get more info about how the score is calculated se  :py:meth:`~similarities.similarity_score` method.
        :param max_year_diff: Allows to soften the year equality to accept also works that are distant x years from each
            other. E.g. setting max_year_diff to 1 we match also papers with absolute difference 1, which might be
             useful for preprints, which are usually released beforehand.
        :param use_gpus:  If false gpus are not used. Else uses all available gpus in shard mode (splits it).
        :param fp16: true activates fp16 precision for GPU
            Only for gpu
        :param init_workers: you can pass number of workers that will be used for getting features in parallel fashion
        :param features_part_size: maximal number of records per parallel worker for features extraction
            Is used when the parallel processing is used for features extraction.
        :param allow_gpus: list of gpu ids that are allowed to be used. If None all available gpus are used.
        :param embeddings: If you already have embeddings you can pass them here. Otherwise they are computed.
            If you pass shared memory object it is expected that the features are stored as numpy array.
            WARNING: adding new records to the list is not possible when you pass shared memory object
        :param stub_init: If true it is initialized as a stub.
        :param cache_size: Size of cache for storing results of search. If 0 no cache is used.
        :param fulltext_search: Path to file with fulltext search sqlite database. If None fulltext search is not used.
            During initialization (when no records are in list) it checks whether the database is preinitialized.
            If not it will use provided records to initialize it.
        :param fulltext_search_uri_connections: If true it will use URI connections for sqlite.
        :raise ValueError: When the length of provided sequences is not the same.
        """
        self._use_gpus = use_gpus

        self._records: Optional[MutableSequence[L]] = []
        self._stub = stub_init

        self._k_neighbors = k_neighbors
        self._match_threshold = match_threshold
        self._max_year_diff = max_year_diff
        self._all_features = None

        self._vectorizer = None
        self._embed_dim = embed_dim

        self.features_part_size = features_part_size

        self.main_process = os.getpid()
        self._allow_gpus = allow_gpus

        self._lock = multiprocessing.RLock()  # let's make that thread safe

        self._search_workers = 0
        self.multi_search_pool = None
        self.multi_search_pool_threads = None

        self._return_self_on_enter = True

        self._forbidden_adding = False

        self._faiss_res = None
        self._cache = None

        self._title_db = None
        self._title_db_cursor = None
        self.database_path = fulltext_search
        self.database_uri = fulltext_search_uri_connections
        if fulltext_search is not None:
            self._title_db = self.create_database(self.database_path, uri=self.database_uri)
            self._title_db_cursor = self._title_db.cursor()
            self._title_db_for_each_thread = None

        if stub_init:
            self._records = records
        else:
            self._vectorizer = HashingVectorizer(n_features=embed_dim, dtype=np.float32, norm="l2")
            if use_gpus:
                allow_gpus = allow_gpus if allow_gpus is not None else list(range(faiss.get_num_gpus()))

                resources = [faiss.StandardGpuResources() for _ in range(len(allow_gpus))]

                flat_config = []
                for i in allow_gpus:
                    cfg = faiss.GpuIndexFlatConfig()
                    cfg.useFloat16 = fp16
                    cfg.device = i
                    flat_config.append(cfg)

                if len(allow_gpus) == 1:
                    self._faiss_res = faiss.IndexIDMap(faiss.GpuIndexFlatIP(resources[0], embed_dim, flat_config[0]))
                else:
                    indexes = [faiss.IndexIDMap(faiss.GpuIndexFlatIP(resources[i], embed_dim, flat_config[i]))
                               for i in range(len(allow_gpus))]

                    self._faiss_res = faiss.IndexShards(True)
                    # True -> runs single thread per GPU, otherwise it searches sequentially GPU by GPU
                    for sub_index in indexes:
                        self._faiss_res.add_shard(sub_index)

            else:
                self._faiss_res = faiss.IndexIDMap(faiss.IndexFlatIP(embed_dim))

            self._faiss_res.successive_ids = False
            self.__add(records, init_workers, embeddings=embeddings)

        if cache_size > 0:
            self._cache = LFUCache(cache_size)

    def create_database_connections_for_thread(self):
        if self._title_db is not None:
            self._title_db_for_each_thread[threading.get_ident()] = sqlite3.connect(self.database_path,
                                                                                    check_same_thread=False,
                                                                                    uri=self.database_uri)

    @staticmethod
    def create_database(database_path: str, uri: bool = False) -> sqlite3.Connection:
        """
        Creates database of titles for fulltext searching.

        :param database_path: path to database
        :param uri: if true it will use URI connection
        :return: database connection
        """

        title_db = sqlite3.connect(database_path, check_same_thread=False, uri=uri)  # False Ok as we have own lock
        title_db_cursor = title_db.cursor()

        # Create a table to store strings
        title_db_cursor.execute("""
            CREATE TABLE IF NOT EXISTS titles (
                id INTEGER PRIMARY KEY,
                title TEXT
            )
        """)
        title_db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON titles(title)")

        # set case-sensitive search, thus we can use LIKE operator efficiently using index
        title_db_cursor.execute("PRAGMA case_sensitive_like = true")

        return title_db

    def get_records(self) -> MutableSequence[L]:
        """
        Returns records.

        :return: records
        """
        return self._records

    @staticmethod
    def read_records(p: str, workers: int = 0, context: Optional[BaseContext] = None,
                     create_cache: bool = True,
                     read_cache_size: int = 65536,
                     memory_mapped: bool = False,
                     verbose: bool = True,
                     progress_bar_position: int = 0,
                     record_type: Optional[Type[PapersListRecord]] = None,
                     shared_list_for_records: Optional[SharedMemoryManager] = None) -> Union[
        PapersListRecordMutableMemoryMappedRecordFile, List[PapersListRecord], SharedListOfRecords[PapersListRecord]]:
        """
        Reads records from file.

        :param p: path to file
        :param workers: number of workers
        :param context: multiprocessing context
        :param create_cache: if true the cache is created
        :param read_cache_size: maximal number of cached records to prevent reading from disk
        :param memory_mapped: if true it will use memory mapped file, else it will return list of records in memory
        :param verbose: if true it will print progress bar
        :param progress_bar_position: position of progress bar
        :param record_type: type of record to be used for loading
        :param shared_list_for_records: if manager is provided it will use shared memory for records
        :return: list of records
        """

        records = []  # will contain list of records or list of titles if memory_mapped is true and cache is created

        if not memory_mapped or create_cache or shared_list_for_records is not None:
            with (FunctorPool(workers=[RecordReaderWorker(p, return_just_title=memory_mapped,
                                                          record_type=record_type,
                                                          return_json=shared_list_for_records is not None) for _ in
                                       range(workers)],
                              context=context,
                              results_queue_maxsize=10.0) if workers > 0 else nullcontext()) as pool, \
                    MapAccessFile(p, p + ".index") as f:

                if workers == 0:
                    single_process_reader = RecordReaderWorker(p, return_just_title=memory_mapped,
                                                               record_type=record_type,
                                                               return_json=shared_list_for_records is not None)
                    single_process_reader.begin()
                    m = partial(map, single_process_reader)
                else:
                    m = partial(pool.imap, chunk_size=10_000)
                try:
                    for record in tqdm(m(f.mapping.values()), desc="Reading papers list from file", total=len(f),
                                       disable=not verbose, position=progress_bar_position):
                        records.append(record)
                finally:
                    if workers == 0:
                        single_process_reader.end()

        if shared_list_for_records is not None:
            return SharedListOfRecords(records, shared_list_for_records,
                                       record_type=PapersListRecord if record_type is None else record_type)

        if memory_mapped:
            records = PapersListRecordMutableMemoryMappedRecordFile(p,
                                                                    records if create_cache else None,
                                                                    read_cache_size=read_cache_size,
                                                                    record_class=record_type)

        return records

    def flush_cache_of_record_file(self):
        """
        Flushes cache of record file.
        """
        if isinstance(self._records, PapersListRecordMutableMemoryMappedRecordFile):
            self._records.flush_cache()

    def load_records_to_memory(self, workers: int = 0, verbose: bool = True, progress_bar_position: int = 0):
        """
        Loads all records to memory. Instead of the memory mapped file, the records will be loaded to memory.

        :param workers: number of workers
        :param verbose: if true it will print progress bar
        :param progress_bar_position: position of progress bar
        """
        if isinstance(self._records, PapersListRecordMutableMemoryMappedRecordFile):
            self._records = self.read_records(self._records.path_to, workers=workers, verbose=verbose,
                                              progress_bar_position=progress_bar_position)

    @property
    def embeddings(self) -> Optional[scipy.sparse.spmatrix]:
        return self._all_features

    def clear_side_embeddings(self):
        """
        Clears side embeddings. It is useful when you want to save memory.

        Side embeddings are embeddings that are not part of faiss index but are used for making the index.
        """

        self._all_features = None

    def set_search_workers(self, workers: int):
        """
        Sets number of workers for searching.  You must use __enter__ and __exit__ methods
        to activate multiprocessing. Otherwise it is not used.

        :param workers: number of workers
        """
        self._search_workers = workers

    @property
    def search_workers(self):
        return self._search_workers

    def return_self_on_enter(self, yes: int):
        """
        Activates/deactivates returning self on __enter__.
        It is handy to deactivate returning self when this object is in multiprocess manager as otherwise it may
        cause troubles with pickling.

        :param yes: True activates, False deactivates returning self on __enter__
        """
        self._return_self_on_enter = yes

    def __enter__(self):
        if self.search_workers > 0:
            self.multi_search_pool = multiprocessing.Pool(self.search_workers)
            self.multi_search_pool.__enter__()

            self._title_db_for_each_thread = {}
            self.multi_search_pool_threads = ThreadPool(self.search_workers,
                                                        initializer=self.create_database_connections_for_thread)
            self.multi_search_pool_threads.__enter__()

        if isinstance(self._records, PapersListRecordMutableMemoryMappedRecordFile):
            self._records.__enter__()

        if self._return_self_on_enter:
            return self

    def open(self):
        self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.multi_search_pool is not None:
            self.multi_search_pool.__exit__(exc_type, exc_val, exc_tb)
            self.multi_search_pool = None

            self.multi_search_pool_threads.__exit__(exc_type, exc_val, exc_tb)
            self.multi_search_pool_threads = None
            self._title_db_for_each_thread = {}

        if isinstance(self._records, PapersListRecordMutableMemoryMappedRecordFile):
            self._records.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self.__exit__(None, None, None)

    def get_match_threshold(self) -> float:
        return self._match_threshold

    def get_max_year_diff(self) -> int:
        return self._max_year_diff

    def reset(self):
        """
        Resets the index. It removes all records.
        """
        self._all_features = None
        self._records = []
        self._faiss_res.reset()

    def stub(self):
        """
        Converts this list to stub state, which means that it removes all structures that are used for searching.
        Thus, the searching will no longer be possible.
        Useful for saving memory.
        Couldn't use addition after that operation.
        """
        if not self._stub:
            self._all_features = None
            self._faiss_res.reset()
            self._vectorizer = None
            self._faiss_res = None
            self._stub = True

    def __getitem__(self, i: Union[int, Iterable[int]]) -> Union[L, List[L]]:
        """
        Get record on given index.

        :param i: index of a record or list of records
        :return: record on given index
        """

        with self._lock:
            if isinstance(i, int):
                return self._records[i]

            return [self._records[j] for j in i]

    def __setitem__(self, i: int, value: L):
        """
        Sets new record on given index.
        Does not change the index.

        :param i: index for change
        :param value: new record
        """
        with self._lock:
            self._records[i] = value

    def __len__(self):
        return len(self._records)

    @staticmethod
    def map_records_to_list(records: Sequence[PapersListRecord],
                            other: Union[List["PapersList"], "PapersList"],
                            batch_size: int = 128, reverse: bool = True, show_progress_bar: bool = True) -> Union[
        Tuple[
            List[
                Optional[
                    int]],
            List[
                Optional[
                    int]]],
        List[Optional[
            int]]]:
        """
        Creates index mapping to another list and otherwise.

        :param records: the records to map
        :param other: the other list
            You can also provide list of lists, in that case the parallel mode is activated.
            It is expected that each list is in manager.
        :param batch_size: Maximal number of samples in a batch when searching for same papers.
        :param reverse: Whether the reverse index mapping should be generated
        :param show_progress_bar: Whether the progress bar should be shown
        :return: (mapping from this to other one, mapping from other one to this list)
            or just mapping from this to other one when reverse is False
        """

        self_2_other_index: List[Optional[int]] = [None] * len(records)
        self_2_other_index_len = 0
        other_2_self_index: List[Optional[int]] = []

        workers = 0
        if isinstance(other, list):
            workers = len(other)
        else:
            other = [other]

        if reverse:
            other_2_self_index = [None] * len(other[0])

        with (ThreadPool(workers) if workers > 0 else nullcontext()) as pool:

            m = map if workers == 0 else pool.imap

            def search(process: Tuple[int, Iterable[PapersListRecord]]):
                use_i, batch = process
                return other[use_i].batch_search(batch)

            batcher = Batcher(records, batch_size)
            for other_search_res in tqdm(m(search, ((b_i % len(other), b) for b_i, b in enumerate(batcher))),
                                         total=len(batcher),
                                         desc=f"Creating index mapping to another list", unit="batch",
                                         disable=not show_progress_bar):
                for self_i, other_i in zip(
                        range(self_2_other_index_len, self_2_other_index_len + len(other_search_res)),
                        other_search_res):
                    if other_i is not None and reverse:
                        other_2_self_index[other_i] = self_i

                for i in range(len(other_search_res)):
                    self_2_other_index[self_2_other_index_len + i] = other_search_res[i]

                self_2_other_index_len += len(other_search_res)

        if reverse:
            return self_2_other_index, other_2_self_index
        return self_2_other_index

    def to_other_mapping(self, other: Union[List["PapersList"], "PapersList"], batch_size: int = 128,
                         reverse: bool = True) -> Union[
        Tuple[
            List[
                Optional[
                    int]],
            List[
                Optional[
                    int]]],
        List[Optional[
            int]]]:
        """
        Creates index mapping to another list and otherwise.

        :param other: the other list
            You can also provide list of lists, in that case the parallel mode is activated.
            It is expected that each list is in manager.
        :param batch_size: Maximal number of samples in a batch when searching for same papers.
        :param reverse: Whether the reverse index mapping should be generated
        :return: (mapping from this to other one, mapping from other one to this list)
            or just mapping from this to other one when reverse is False
        """

        with self._lock:
            return self.map_records_to_list(self._records, other, batch_size, reverse)

    def add(self, records: MutableSequence[L], workers: int = 0, reset: bool = False,
            embeddings: Optional[Union[scipy.sparse.spmatrix, SharedMemory]] = None):
        """
        Add new records to list.

        :param records: new records that should be added
        :param workers: you can pass number of workers that will be used for getting features in parallel fashion
        :param reset: before adding the new data the index will be reset
            Avoids the potential memory costly realloc on GPU by recreating the index from scratch.
        :param embeddings: you can pass precomputed embeddings
            you can use  ShareMemory only for empty list and also the further adding will be forbidden
        :raise ValueError: When the length of provided sequences is not the same.
        """
        if self._stub:
            raise RuntimeError("Couldn't use add in stub state.")

        if self._forbidden_adding:
            raise RuntimeError("Couldn't use add in forbidden state.")

        with self._lock:
            self.__add(records, workers, reset, embeddings)

    def _extract_features(self, records: Sequence[L], workers: int = 0) -> scipy.sparse.spmatrix:
        """
        Extracts features for knn search.

        :param records: uses title field for creating features
        :param workers: you can pass number of workers that will be used for getting features in parallel fashion
        :return: features
        """

        if workers <= 0:
            return self._vectorizer.transform(r.title for r in records)

        with FunctorPool(workers=[ObtainFeaturesWorker(records, self._vectorizer) for _ in range(workers)],
                         results_queue_maxsize=1.0) if isinstance(records,
                                                                  SharedListOfRecords) else multiprocessing.Pool(
            workers) as pool:

            def batch_boundaries_generator():
                for offset in range(0, len(records), self.features_part_size):
                    yield offset, min(len(records), offset + self.features_part_size)

            if isinstance(records, SharedListOfRecords):
                parts_generator = batch_boundaries_generator()
                m = pool.imap
            else:
                parts_generator = ([records[i].title for i in range(s, e)] for s, e in batch_boundaries_generator())
                m = partial(pool.imap, self._vectorizer.transform)

            all_features = []
            for features in tqdm(m(parts_generator),
                                 desc="Extracting features",
                                 unit="batch",
                                 total=math.ceil(len(records) / self.features_part_size)):
                all_features.append(features)

        return vstack(all_features)

    @staticmethod
    def insert_titles_to_database(db: sqlite3.Connection, records: Sequence[L], workers: int = 0,
                                  batch_size: int = 81_920):
        """
        Inserts titles of records to fulltext database.
        Also performs normalization of titles.

        :param db: database connection
        :param records: records for inserting
        :param workers: number of workers
        :param batch_size: maximal number of records per batch processed by single worker
        """
        workers = max(1, workers)
        db_cursor = db.cursor()
        with FunctorPool(workers=[ObtainNormalizedTitlesWorker(records) for _ in range(workers)],
                         results_queue_maxsize=1.0) as pool:
            def batch_boundaries_generator():
                for offset in range(0, len(records), batch_size):
                    yield offset, min(len(records), offset + batch_size)

            for norm_titles_batch in tqdm(pool.imap(batch_boundaries_generator()),
                                          desc="Normalizing titles for fulltext search",
                                          unit=f"batch ({batch_size})",
                                          total=math.ceil(len(records) / batch_size)):
                norm_titles_batch = [(x,) for x in norm_titles_batch]
                db_cursor.executemany("INSERT INTO titles(title) VALUES (?)", norm_titles_batch)
        db.commit()

    def __add(self, records: MutableSequence[L], workers: int = 0, reset: bool = False,
              embeddings: Optional[Union[scipy.sparse.spmatrix, SharedMemory]] = None):
        """
        Add new records to list.
        This private method is here just because of inheritance. So we can be sure that this method is called and not the
        method of child.

        :param records: new records that should be added
        :param workers: you can pass number of workers that will be used for getting features in parallel fashion
        :param reset: before adding the new data the index will be reset
            Avoids the potential memory costly realloc on GPU by recreating the index from scratch.
        :param embeddings: you can pass precomputed embeddings
            you can use  ShareMemory only for empty list and also the further adding will be forbidden
        :raise ValueError: When the length of provided sequences is not the same.
        """
        if self._cache is not None:
            self._cache.clear()

        add_from_shared_memory = isinstance(embeddings, SharedMemory)
        if add_from_shared_memory and len(self) > 0:
            raise ValueError("You can use shared memory only for empty list.")

        if len(records) > 0:
            if self._title_db is not None:
                is_empty = self._title_db_cursor.execute("SELECT COUNT(*) FROM titles").fetchone()[0] == 0
                if len(self._records) > 0 or is_empty:  # we don't want to add title to preinitialized database
                    # normalize titles and insert them to database
                    self.insert_titles_to_database(self._title_db, records, workers, self.features_part_size)

            ids = np.arange(len(self), len(self) + len(records))

            if len(self._records) == 0:
                self._records = records
            else:
                self._records.extend(records)

            if embeddings is not None:
                features = embeddings
            else:
                features = self._extract_features(records, workers)

            if add_from_shared_memory:
                self._forbidden_adding = True
                features = np.ndarray((len(records), self._embed_dim), dtype=np.float32, buffer=features.buf)
            else:
                self._all_features = features if self._all_features is None else vstack([self._all_features, features])

                if reset:
                    self._faiss_res.reset()
                    features = self._all_features
                    ids = np.arange(0, len(self))

                features = features.toarray()

            self._faiss_res.add_with_ids(features, ids)

    def fulltext_search_title(self, titles: Iterable[str]) -> List[Optional[List[int]]]:
        """
        Searches for title in fulltext database.

        :param titles: titles for search
        :return: list of ids of found records for each title
            None is on positions with no match
        """
        if self._title_db_cursor is None:
            raise RuntimeError("Fulltext search is not available.")
        res = []

        def search(title):
            if self._title_db_for_each_thread is None:
                cursor = self._title_db.cursor()
            else:
                connection = self._title_db_for_each_thread[threading.get_ident()]
                cursor = connection.cursor()
            cursor.execute("SELECT id FROM titles WHERE title LIKE ? LIMIT ?",
                           (f"{normalize_string(title)}%", self._k_neighbors))
            r = cursor.fetchall()
            cursor.close()
            if len(r) == 0:
                return None
            return [x[0] - 1 for x in r]  # -1 because autoincrement starts from 1

        if self.multi_search_pool_threads is None:
            for title in titles:
                res.append(search(title))
        else:
            for title in self.multi_search_pool_threads.imap(search, titles):
                res.append(title)

        return res

    def batch_search_by_title(self, titles: Iterable[str], fulltext: bool = True) -> Union[
        List[List[Optional[int]]], Tuple[List[List[Optional[int]]], List[List[int]]]]:
        """
        The nearest neighbour search by title.

        :param titles: titles for nn search
        :param fulltext: If true it will also search in fulltext database if it is available.
        :return: List of search results in form of searched indices.
            None when index is empty

            or tuple where the second element is list of fulltext search results
        """

        search_vectors = self._vectorizer.transform(titles)
        with self._lock:
            nearest_indices = []

            for i, res_sample in enumerate(
                    self._faiss_res.search(search_vectors.toarray(), self._k_neighbors)[1]):
                nearest_indices.append([
                    None if res is None else int(res) for res in res_sample
                ])

            if fulltext and self._title_db_cursor is not None:
                return nearest_indices, self.fulltext_search_title(titles)
            return nearest_indices

    def batch_search(self, items: Iterable[PapersListRecord], skip_indices: Optional[Sequence[Set[int]]] = None,
                     fulltext: bool = True) \
            -> List[Optional[int]]:
        """
        It firstly searches k nearest neighbors by title, then it filters out all with titles that are not satisfying
        given similarity threshold. Then it continuous with matching by year and
        concludes with matching by authors where at least one author must be similar enough.

        The similarity threshold may be changed. See members of this class.

        :param items: Iterable of papers for approximate search
            If year is None it will match with any year
        :param skip_indices: It allows to specify indices that should be removed from k nearest searched neighbors for
            each item.
            Beware that in the case when a specified index is searched it will not try to find another one for
            substitution.
        :param fulltext: If true it will firstly search in fulltext database if it is available.
        :return: List of search results in form of searched indices. The None is on positions with no match.
        """
        try:
            if self._stub:
                raise RuntimeError("Couldn't search in stub state.")

            with self._lock:
                if fulltext and self._title_db_cursor is not None:
                    fulltext_indices = self.batch_search_fulltext(items, skip_indices)
                    nones = [ite for f, ite in zip(fulltext_indices, items) if f is None]
                    nearest_indices = []
                    if len(nones) > 0:
                        nearest_indices = self.batch_search_nearest(nones, skip_indices)

                    indices = []
                    none_offset = 0
                    for f in fulltext_indices:
                        if f is None:
                            indices.append(nearest_indices[none_offset])
                            none_offset += 1
                        else:
                            indices.append(f)
                else:
                    indices = self.batch_search_nearest(items, skip_indices)

                res = self.batch_filter_search_results(items, indices)

            return res
        except Exception as e:
            print(f"There was an exception in process {multiprocessing.current_process()}", flush=True, file=sys.stderr)
            traceback.print_exc()
            raise e

    def _batch_search_cached(self, items: Iterable[PapersListRecord]) -> List[List[Optional[int]]]:
        """
        It searches k nearest neighbors by title. It uses cache for searching if possible.

        :param items: Iterable of papers for approximate search
        :return: List of search results in form of searched indices. The None is on positions with no match.
        """
        res = []

        not_cached_items = items

        with self._lock:
            if self._cache is not None:
                not_cached_items = []
                for i in items:
                    try:
                        res.append(self._cache[i.title])
                    except KeyError:
                        res.append(None)
                        not_cached_items.append(i)
                if len(not_cached_items) == 0:
                    return res

            search_vectors = self._vectorizer.transform(r.title for r in not_cached_items)
            nearest_indices = []

            for i, res_sample in enumerate(
                    self._faiss_res.search(search_vectors.toarray(), self._k_neighbors)[1]):
                nearest_indices.append([
                    None if res is None else int(res) for res in res_sample
                ])

            if self._cache is None:
                return nearest_indices

            cnt = 0

            for i, r in enumerate(res):
                if r is None:
                    res[i] = nearest_indices[cnt]
                    self._cache[not_cached_items[cnt].title] = nearest_indices[cnt]
                    cnt += 1
        return res

    def batch_search_fulltext(self, items: Iterable[PapersListRecord],
                              skip_indices: Optional[Sequence[Set[int]]] = None,
                              return_records: bool = False) -> Union[
        List[List[Optional[int]]], Tuple[List[List[Optional[int]]], List[List[Optional[PapersListRecord]]]]]:
        """
        It uses fulltext search to search by title.

        :param items: Iterable of papers for fulltext search
        :param skip_indices: It allows to specify indices that should be removed
        :param return_records: If true it will return also records of nearest neighbors
        :return: List of search results in form of searched indices. The None is on positions with no match.
        """
        if self._stub:
            raise RuntimeError("Couldn't search in stub state.")

        records = []

        res = self.fulltext_search_title((r.title for r in items))
        if skip_indices is not None:
            filtered_res = []
            for i, r in enumerate(res):
                if r is None:
                    filtered_res.append(r)
                    continue

                filtered_res.append([x for x in r if x not in skip_indices[i]])

            res = filtered_res
        nearest_indices = res
        if return_records:
            searched_records = []
            for n in nearest_indices:
                if n is None:
                    searched_records.append(None)
                else:
                    searched_records.append([self._records[i] for i in n])
            records.append(searched_records)

        if return_records:
            return nearest_indices, records
        return nearest_indices

    def batch_search_nearest(self, items: Iterable[PapersListRecord],
                             skip_indices: Optional[Sequence[Set[int]]] = None,
                             return_records: bool = False) -> Union[
        List[List[Optional[int]]], Tuple[List[List[Optional[int]]], List[List[Optional[PapersListRecord]]]]]:
        """
        It searches k nearest neighbors by title.

        :param items: Iterable of papers for approximate search
        :param skip_indices: It allows to specify indices that should be removed from k nearest searched neighbors for
        :param return_records: If true it will return also records of nearest neighbors
        :return: List of search results in form of searched indices. The None is on positions with no match.
        """

        try:
            if self._stub:
                raise RuntimeError("Couldn't search in stub state.")
            nearest_indices = []
            records = []
            for i, res_sample in enumerate(
                    self._batch_search_cached(items)):

                nearest_indices.append([])
                records.append([])
                for res in res_sample:
                    if (skip_indices is None) or (res is None) or (res not in skip_indices[i]):
                        nearest_indices[-1].append(res)
                        if return_records:
                            records[-1].append(self._records[res] if res is not None else None)
            if return_records:
                return nearest_indices, records
            return nearest_indices

        except Exception as e:
            print(f"There was an exception in process {multiprocessing.current_process()}", file=sys.stderr, flush=True)
            traceback.print_exc()
            raise e

    def batch_filter_search_results(self, items: Iterable[PapersListRecord],
                                    indices: List[List[Optional[int]]]) -> List[Optional[int]]:
        """
        it filters out all with titles that are not satisfying
        given similarity threshold. Then it continuous with matching by year and
        concludes with matching by authors where at least one author must be similar enough.

        The similarity threshold may be changed. See members of this class.

        :param items: papers for filtering approximate search
        :param indices: searched indices
        :return: List of search results in form of searched indices. The None is on positions with no match.
        """

        try:
            if self._stub:
                raise RuntimeError("Couldn't search in stub state.")

            res = []

            def prepare_sample(s_r, indices_conv):
                nearest_records = {
                    i: self._records[i] for i in indices_conv if i is not None and i >= 0
                }
                return s_r, nearest_records, self._match_threshold, self._max_year_diff

            m = map if self.multi_search_pool is None else partial(self.multi_search_pool.imap, chunksize=10)

            with self._lock:
                for r in m(self.filter_search_results,
                           (prepare_sample(x[0], x[1]) for x in zip(items, indices))):
                    res.append(r)

            return res

        except Exception as e:
            print(f"There was an exception in process {multiprocessing.current_process()}", file=sys.stderr, flush=True)
            traceback.print_exc()
            raise e

    @staticmethod
    def filter_search_results(process: Tuple[PapersListRecord, Dict[int, PapersListRecord], float, int]) -> \
            Optional[int]:
        """
        it filters out all with titles that are not satisfying
        given similarity threshold. Then it continuous with matching by year and
        concludes with matching by authors where at least one author must be similar enough.

        The similarity threshold may be changed. See members of this class.

        :param process: tuple of
            search_record: paper for filtering approximate search
            records: searched records
            match_threshold: similarity threshold
            max_year_diff: maximum year difference
        :return: List of search results in form of searched indices. The None is on positions with no match.
        """
        search_record, records, match_threshold, max_year_diff = process

        title = set(normalize_and_tokenize_string(search_record.title))

        title_sim_scores = []

        for i, r in records.items():
            if search_record.year is None or r.year is None or abs(r.year - search_record.year) <= max_year_diff:
                score = similarity_score(title, set(normalize_and_tokenize_string(r.title)))
                if score >= match_threshold:
                    title_sim_scores.append((i, score))

        if len(title_sim_scores) == 0:
            return None

        all_authors = list(search_record.authors)
        all_authors.extend(itertools.chain.from_iterable(records[i].authors for i, _ in title_sim_scores))

        all_authors_initials, all_authors = initial_and_normalized_authors(all_authors)
        authors_initials, authors = all_authors_initials[:len(search_record.authors)], all_authors[
                                                                                       :len(search_record.authors)]

        authors_offset = len(search_record.authors)
        for i, title_sim_score in sorted(title_sim_scores, key=lambda x: x[1], reverse=True):
            # normalize authors
            paper_list_authors = records[i].authors
            paper_list_authors_initials, paper_list_authors = all_authors_initials[authors_offset:authors_offset + len(
                paper_list_authors)], all_authors[authors_offset:authors_offset + len(paper_list_authors)]
            authors_offset += len(paper_list_authors)

            if match_authors_groups(authors, authors_initials, paper_list_authors, paper_list_authors_initials,
                                    match_threshold):
                # we have a match
                return i

    @classmethod
    def create_shared_embeddings(cls, records: MutableSequence[L], embed_dim=64, workers: int = 0,
                                 features_part_size: int = 81_920) -> SharedMemory:
        """
        It creates shared memory with embeddings of given records.

        :param records: records for which embeddings should be created
        :param embed_dim: dimension of embeddings
        :param workers: number of workers for parallel processing
        :param features_part_size: size of part of features that is processed by one worker
        :return: shared memory with embeddings
        """

        lst_for_embeddings = cls(records=[],
                                 embed_dim=embed_dim,
                                 init_workers=workers,
                                 features_part_size=features_part_size)
        embeddings = lst_for_embeddings._extract_features(records, workers=workers).toarray()
        # move embeddings to shared memory
        shared_memory = SharedMemory(
            create=True,
            size=embeddings.nbytes
        )
        numpy_shared_memory = np.ndarray(embeddings.shape, dtype=embeddings.dtype, buffer=shared_memory.buf)
        numpy_shared_memory[:] = embeddings
        return shared_memory

    @classmethod
    def init_multiple(cls, records: MutableSequence[L], embeddings: Optional[SharedMemory], embed_dim=64,
                      k_neighbors=100,
                      match_threshold: float = 0.75, max_year_diff: int = 0, use_gpus: bool = True,
                      fp16: bool = True,
                      workers: int = 0, features_part_size: int = 81_920,
                      manager: Optional[Union[Sequence["PapersListManager"], "PapersListManager"]] = None,
                      allow_gpus: Optional[Sequence[Sequence[int]]] = None,
                      clear_side_embeddings: bool = False,
                      stub_init: bool = False,
                      search_cache_size: int = 0,
                      load_records_after_init: bool = False,
                      fulltext_search: Optional[str] = None, **aditionals) -> Union["PapersList", List["PapersList"]]:
        """
        Inits multiple lists

        :param records: contains:
            title of publication is used for creating features for searching
            year of publication could be None in that case it matches with any year
            authors of publication, at least one author must match during searching
        :param embeddings: shared memory with embeddings of given records
        :param embed_dim: Number of dimensions of title representation obtained by hashing vectorizer.
            This representation is used for nearest neighbors search.
        :param k_neighbors: Number of neighbors for nearest neighbors search of titles.
        :param match_threshold: Score threshold for matching. All above are ok.
            Matching is used for title and list of authors. For authors at least one of them must have a match.
            To get more info about how the score is calculated se  :func:`~similarities.similarity_score` method.
        :param max_year_diff: Allows to soften the year equality to accept also works that are distant x years from each
            other. E.g. setting max_year_diff to 1 we match also papers with absolute difference 1, which might be
             useful for preprints, which are usually released beforehand.
        :param use_gpus:  If false gpus are not used. Else uses all available gpus in shard mode (splits it).
        :param fp16: true activates fp16 precision for GPU
            Only for gpu
        :param workers: you can pass number of workers that will be used for getting features in parallel fashion
        :param features_part_size: maximal number of records per parallel worker for features extraction
            Is used when the parallel processing is used for features extraction.
        :param manager: Declares whether the papers list should be created through multiprocessing manager.
            the list option is used together with list option of allow_gpus.
        :param allow_gpus: If not None it will create multiple lists with different gpus for each manager.
            You can specify gpus that should be used for each list.
        :param clear_side_embeddings: If true the embeddings are cleared after the list is created.
            If multiple managers are used the side embeddings are not stored at all.
        :param stub_init: If true it is initialized as a stub.
        :param search_cache_size: Size of cache for search results. If 0 the cache is not used.
        :param aditionals: additional parameters for PapersList init
        :param load_records_after_init: If true the records are loaded after init into memory.
        :param fulltext_search: Path to file with fulltext search sqlite database. If None fulltext search is not used.
        :return: paper list
            or multiple lists if allow_gpus is not None
        """

        if allow_gpus is not None and isinstance(manager, PapersListManager):
            raise ValueError("If you want to use allow_gpus you have to specify multiple managers.")

        if allow_gpus is not None and len(allow_gpus) != len(manager):
            raise ValueError("You must specify the same number of managers as allow_gpus groups.")

        if isinstance(manager, PapersListManager):
            manager = [manager]

        papers_lists = []
        gpus_groups = [None] if allow_gpus is None else allow_gpus

        try:
            with (ThreadPool(len(gpus_groups)) if manager is not None else nullcontext()) as pool:
                m = map if manager is None else pool.imap_unordered

                def init_list(proc: Tuple[int, Sequence[int]]):
                    i, gpus = proc
                    cur_workers = (workers // len(gpus_groups))

                    if manager is None:
                        lst = cls(records=records,
                                  embed_dim=embed_dim,
                                  k_neighbors=k_neighbors,
                                  match_threshold=match_threshold,
                                  max_year_diff=max_year_diff,
                                  use_gpus=use_gpus,
                                  fp16=fp16,
                                  init_workers=cur_workers,
                                  features_part_size=features_part_size,
                                  allow_gpus=gpus,
                                  embeddings=embeddings,
                                  stub_init=stub_init,
                                  cache_size=search_cache_size,
                                  fulltext_search=fulltext_search, **aditionals)
                    else:
                        lst = getattr(manager[i], cls.__name__)(records=records,
                                                                embed_dim=embed_dim,
                                                                k_neighbors=k_neighbors,
                                                                match_threshold=match_threshold,
                                                                max_year_diff=max_year_diff,
                                                                use_gpus=use_gpus,
                                                                fp16=fp16,
                                                                init_workers=cur_workers,
                                                                features_part_size=features_part_size,
                                                                allow_gpus=gpus,
                                                                embeddings=embeddings,
                                                                stub_init=stub_init,
                                                                cache_size=search_cache_size,
                                                                fulltext_search=fulltext_search,
                                                                **aditionals)

                        lst.return_self_on_enter(False)

                    if clear_side_embeddings:
                        lst.clear_side_embeddings()

                    if load_records_after_init:
                        lst.load_records_to_memory(workers=cur_workers, progress_bar_position=i)

                    return lst

                for lst in tqdm(m(init_list, enumerate(gpus_groups)), desc="Creating lists", total=len(gpus_groups)):
                    papers_lists.append(lst)
        finally:
            if isinstance(embeddings, SharedMemory):
                embeddings.close()
                embeddings.unlink()

        if allow_gpus is None:
            return papers_lists[0]
        return papers_lists

    @classmethod
    def from_file(cls, p: str, embed_dim=64, k_neighbors=100, match_threshold=0.75, max_year_diff: int = 0,
                  use_gpus: bool = True, fp16: bool = True, workers: int = 0, features_part_size: int = 81_920,
                  manager: Optional[Union[Sequence["PapersListManager"], "PapersListManager"]] = None,
                  allow_gpus: Optional[Sequence[Sequence[int]]] = None,
                  read_cache_size: int = 65536,
                  search_cache_size: int = 0,
                  load_records_after_init: bool = False,
                  record_type: Optional[Type[PapersListRecord]] = None,
                  shared_list_for_records: Optional[SharedMemoryManager] = None,
                  fulltext_search: Optional[str] = None) \
            -> Union["PapersList", list["PapersList"]]:
        """
        Loads papers list from file.

        :param p: path to file
        :param embed_dim: Number of dimensions of title representation obtained by hashing vectorizer.
            This representation is used for nearest neighbors search.
        :param k_neighbors: Number of neighbors for nearest neighbors search of titles.
        :param match_threshold: Score threshold for matching. All above are ok.
            Matching is used for title and list of authors. For authors at least one of them must have a match.
            To get more info about how the score is calculated se  :func:`~similarities.similarity_score` method.
        :param max_year_diff: Allows to soften the year equality to accept also works that are distant x years from each
            other. E.g. setting max_year_diff to 1 we match also papers with absolute difference 1, which might be
             useful for preprints, which are usually released beforehand.
        :param use_gpus:  If false gpus are not used. Else uses all available gpus in shard mode (splits it).
        :param fp16: true activates fp16 precision for GPU
            Only for gpu
        :param workers: you can pass number of workers
        :param features_part_size: maximal number of records per parallel worker for features extraction
            Is used when the parallel processing is used for features extraction.
        :param manager: Declares whether the papers list should be created through multiprocessing manager.
            the list option is used together with list option of allow_gpus.
        :param allow_gpus: If not None it will create multiple lists with different gpus for each manager.
            You can specify gpus that should be used for each list.
        :param read_cache_size: maximal number of cached records to prevent reading from disk when
            MAGMutableMemoryMappedRecordFile will be used
            Used only when the record file is not already loaded
        :param search_cache_size: maximal number of cached searches
        :param load_records_after_init: if true it will load records after init
        :param record_type: type of record to be used for loading
        :param shared_list_for_records: if manager is provided it will use shared memory for records
        :param fulltext_search: Path to file with fulltext search sqlite database. If None fulltext search is not used.
        :return: paper list
        """

        if allow_gpus is not None and isinstance(manager, PapersListManager):
            raise ValueError("If you want to use allow_gpus you have to specify multiple managers.")

        if allow_gpus is not None and len(allow_gpus) != len(manager):
            raise ValueError("You must specify the same number of managers as allow_gpus groups.")

        records = cls.read_records(p, workers=workers, read_cache_size=read_cache_size,
                                   memory_mapped=manager is not None and shared_list_for_records is None,
                                   record_type=record_type,
                                   shared_list_for_records=shared_list_for_records)

        with records if isinstance(records, PapersListRecordMutableMemoryMappedRecordFile) else nullcontext():
            embeddings = cls.create_shared_embeddings(records=records,
                                                      embed_dim=embed_dim,
                                                      workers=workers,
                                                      features_part_size=features_part_size)

        if manager is not None and shared_list_for_records is None:
            records.flush_cache()

        return cls.init_multiple(records=records,
                                 embeddings=embeddings,
                                 embed_dim=embed_dim,
                                 k_neighbors=k_neighbors,
                                 match_threshold=match_threshold,
                                 max_year_diff=max_year_diff,
                                 use_gpus=use_gpus,
                                 fp16=fp16,
                                 workers=workers,
                                 features_part_size=features_part_size,
                                 manager=manager,
                                 allow_gpus=allow_gpus,
                                 search_cache_size=search_cache_size,
                                 load_records_after_init=load_records_after_init,
                                 fulltext_search=fulltext_search)


class RecordReaderWorker(FunctorWorker):
    """
    Worker that reads records from file.
    """

    def __init__(self, file_path: str, return_just_title: bool = False, record_type: Type[PapersListRecord] = None,
                 return_json: bool = False):
        """
        :param file_path: path to file
        :param return_just_title: if true it returns just title of the record
        :param record_type: type of record
        :param return_json: if true it returns json of the record
        """

        super().__init__()
        self._file_path = file_path
        self._file = None
        self._return_just_title = return_just_title
        self._record_type = PapersListRecord if record_type is None else record_type
        self._return_json = return_json

    def begin(self):
        self._file = open(self._file_path, "r")

    def end(self):
        self._file.close()

    def __call__(self, line_offset: int) -> Union[PapersListRecord, str]:
        self._file.seek(line_offset)
        line = self._file.readline()

        if self._return_just_title:
            record = next(ijson.items(ijson.parse(line), "title"))
            return record

        record = self._record_type.load(line)

        if self._return_json:
            return record.save()

        return record


class ScopusPapersList(PapersList):
    """
    Papers list for scopus data.
    """

    @staticmethod
    def read_records(p: str, verbose: bool = True, progress_bar_position: int = 0, *ignored) -> List[PapersListRecord]:
        """
        This method reads records from file.
        It is different from the base method as the scopus format might contain a records that will be skipped durring
        the reading and the base method is not counting with that option.

        :param p: path to file
        :param verbose: if true it will show progress bar
        :param progress_bar_position: position of progress bar
        :param ignored: ignored parameters
        :return: list of records
        """

        # fields description: http://schema.elsevier.com/dtds/document/bkapi/search/SCIDIRSearchViews.htm
        # the authors field is missing and only the first author (dc:creator) is available

        records = []

        with open(p, "rb") as f, tqdm(desc="Reading papers list from file", total=os.path.getsize(p), unit="bytes",
                                      disable=not verbose, position=progress_bar_position) \
                as p_bar:
            last_file_offset = 0

            line = f.readline()
            cnt = 0
            while line:
                cnt += 1
                record = json_loads(line)
                try:
                    if record["dc:title"] and record["prism:coverDate"] and record["dc:creator"]:
                        # not-incomplete record

                        records.append(
                            PapersListRecord(
                                title=record["dc:title"],
                                year=int(record["prism:coverDate"].split("-", maxsplit=1)[0]),
                                # prism:coverDate: Publication date (YYYY-MM-DD)
                                authors=[record["dc:creator"]]
                            )
                        )
                except KeyError:
                    pass

                p_bar.update(f.tell() - last_file_offset)
                last_file_offset = f.tell()
                line = f.readline()

        return records

    @classmethod
    def from_file(cls, p: str, embed_dim=64, k_neighbors=100, match_threshold=0.75, max_year_diff: int = 0,
                  use_gpus: bool = True, fp16: bool = True, workers: int = 0,
                  features_part_size: int = 81_920, fulltext_search: Optional[str] = None) -> "ScopusPapersList":
        """
        Loads papers list from file.
        Omits a record if a title, year, or author/s is missing.

        :param p: path to file
        :param embed_dim: Number of dimensions of title representation obtained by hashing vectorizer.
            This representation is used for nearest neighbors search.
        :param k_neighbors: Number of neighbors for nearest neighbors search of titles.
        :param match_threshold: Score threshold for matching. All above are ok.
            Matching is used for title and list of authors. For authors at least one of them must have a match.
            To get more info about how the score is calculated se  :func:`~similarities.similarity_score` method.
        :param max_year_diff: Allows to soften the year equality to accept also works that are distant x years from each
            other. E.g. setting max_year_diff to 1 we match also papers with absolute difference 1, which might be
             useful for preprints, which are usually released beforehand.
        :param use_gpus:  If false gpus are not used. Else uses all available gpus in shard mode (splits it).
        :param fp16: true activates fp16 precision for GPU
            Only for gpu
        :param workers: you can pass number of workers that will be used for getting features in parallel fashion
        :param features_part_size: maximal number of records per parallel worker for features extraction
            Is used when the parallel processing is used for features extraction.
        :param fulltext_search: Path to file with fulltext search sqlite database. If None fulltext search is not used.
        :return: paper list
        """

        records = cls.read_records(p)

        return cls(records=records,
                   embed_dim=embed_dim,
                   k_neighbors=k_neighbors,
                   match_threshold=match_threshold,
                   max_year_diff=max_year_diff,
                   use_gpus=use_gpus,
                   fp16=fp16,
                   init_workers=workers,
                   features_part_size=features_part_size,
                   fulltext_search=fulltext_search)


@dataclass
class MAGPapersListRecord(PapersListRecord, Record):
    """
    Record for MAG papers list.
    """
    __slots__ = ("id", "references", "fields", "doi", "journal")

    id: int
    references: Sequence[int]
    fields: List[Union[str, Tuple[str, float]]]
    doi: Optional[str]
    journal: Optional[str]

    @classmethod
    def load(cls, s: str) -> "MAGPapersListRecord":
        d = json_loads(s)
        return cls(
            id=d["PaperId"],
            title=d["OriginalTitle"],
            year=d["Year"],
            authors=d["Authors"],
            references=d["References"],
            fields=[x if isinstance(x, str) else tuple(x) for x in d["Fields"]],
            doi=d["Doi"],
            journal=d["Journal"]
        )

    def save(self) -> str:
        return json_dumps({
            "PaperId": self.id,
            "OriginalTitle": self.title,
            "Year": self.year,
            "Authors": self.authors,
            "References": self.references,
            "Fields": self.fields,
            "Doi": self.doi,
            "Journal": self.journal
        })


class MAGPapersListRecordCachedStub(MAGPapersListRecord):
    """
    Record for cached MAG papers that is used during initialization and contains only some attributes.
    Is used just as speed up optimization that saves loading from a disk.
    """

    def __init__(self, i: int, title: str):
        """
        Initialization of MAG record with preloaded id and title.

        :param i: preloaded id
        :param title: preloaded title
        """

        self.id = i
        self.title = title

    @property
    def year(self):
        raise RuntimeError("The year property can not be obtained.")

    @property
    def authors(self):
        raise RuntimeError("The authors property can not be obtained.")

    @property
    def references(self):
        raise RuntimeError("The references property can not be obtained.")

    @property
    def fields(self):
        raise RuntimeError("The fields property can not be obtained.")

    @property
    def doi(self):
        raise RuntimeError("The doi property can not be obtained.")

    @property
    def journal(self):
        raise RuntimeError("The journal property can not be obtained.")


class MAGMutableMemoryMappedRecordFile(PapersListRecordMutableMemoryMappedRecordFile[MAGPapersListRecord]):
    """
    Like mutable MutableMemoryMappedRecordFile, but implements caching of ids and titles for init phase. After
    calling flush_cache it starts to act like ordinary MutableMemoryMappedRecordFile.

    Is used just as speed up optimization that saves loading from a disk.
    """

    def __init__(self, path_to: str, ids: Optional[List[int]], titles: Optional[List[str]],
                 read_cache_size: int = 65536):
        """
        initializing MAG

        :param path_to: path to mag file
        :param ids: cached ids
        :param titles: cached titles
        :param read_cache_size: maximal number of cached records to prevent reading from disk
        """
        super().__init__(path_to, titles, read_cache_size, MAGPapersListRecord)
        self.ids = ids

    def move_data_to_manager(self, manager: multiprocessing.Manager):
        super().move_data_to_manager(manager)
        if self.ids is not None:
            self.ids = manager.list(self.ids)

    def flush_cache(self):
        self.ids = None
        super().flush_cache()

    def create_stub(self, n: int) -> MAGPapersListRecordCachedStub:
        return MAGPapersListRecordCachedStub(self.ids[n], self.titles[n])


class MagRecordReaderWorker(FunctorWorker):
    """
    Worker that reads records from file.
    """

    def __init__(self, path: str):
        """
        :param path: path to file with records
        """
        self.file = None
        self.path = path
        super().__init__()

    def begin(self):
        self.file = open(self.path, "r")

    def end(self):
        self.file.close()

    def __call__(self, file_offset: int) -> Optional[Tuple[int, str]]:
        """
        Reads record from file and returns its id and title.

        :param file_offset: record file offset
        :return: record id and title
            None if record is incomplete
        """
        self.file.seek(file_offset)
        record = MAGPapersListRecord.load(self.file.readline())

        if not record.title or not record.authors:
            return None
        return record.id, record.title


class MAGPapersList(PapersList[MAGPapersListRecord]):
    """
    Papers list for MAG.

    It allows to change some fields and add new records. But beware that not all fields could be changed, just
    those ending with _mutable.
    """

    def __init__(self, records: Optional[MAGMutableMemoryMappedRecordFile], embed_dim=64, k_neighbors=100,
                 match_threshold: float = 0.75, max_year_diff: int = 0, use_gpus: bool = True, fp16: bool = True,
                 init_workers: int = 0, features_part_size: int = 81_920, allow_gpus: Optional[Sequence[int]] = None,
                 init_ids: Optional[List[int]] = None,
                 embeddings: Optional[Union[scipy.sparse.spmatrix, SharedMemory]] = None,
                 stub_init: bool = False,
                 cache_size: int = 0, fulltext_search: Optional[str] = None):
        """
        Initialization of paper list.

        :param records: not opened file with mag records, the opening and closing is handled by this object iself
            Use None when you need to delay the data init.
        :param embed_dim: Number of dimensions of title representation obtained be hashing vectorizer.
            This representation is used for nearest neighbors search.
        :param k_neighbors: Number of neighbors for nearest neighbors search of titles.
        :param match_threshold: Score threshold for matching. All above are ok.
            Matching is used for title and list of authors. For authors at least one of them must have a match.
            To get more info about how the score is calculated se  :func:`~similarities.similarity_score` method.
        :param max_year_diff: Allows to soften the year equality to accept also works that are distant x years from each
            other. E.g. setting max_year_diff to 1 we match also papers with absolute difference 1, which might be
             useful for preprints, which are usually released beforehand.
        :param use_gpus:  If false gpus are not used. Else uses all available gpus in shard mode (splits it).
        :param fp16: true activates fp16 precision for GPU
            Only for gpu
        :param init_workers: you can pass number of workers that will be used for getting features in parallel fashion
        :param features_part_size: maximal number of records per parallel worker for features extraction
            Is used when the parallel processing is used for features extraction.
        :param allow_gpus: list of gpu ids that are allowed to be used. If None all available gpus are used.
        :param init_ids: Allows passing preloaded ids so they will not be loaded during init phase
        :param embeddings: If you already have embeddings you can pass them here. Otherwise they are computed.
        :param stub_init: If true it is initialized as a stub.
        :param cache_size: maximal number of cached search results
        :param fulltext_search: Path to file with fulltext search sqlite database. If None fulltext search is not used.
        """

        if records is None:
            records = []

        with nullcontext() if isinstance(records, list) else records:
            self._init_ids([r.id for r in records] if init_ids is None else init_ids)
            super().__init__(records, embed_dim, k_neighbors, match_threshold, max_year_diff, use_gpus, fp16,
                             init_workers, features_part_size, allow_gpus, embeddings, stub_init, cache_size,
                             fulltext_search=fulltext_search)

    def _init_ids(self, ids: Optional[List[int]] = None):
        """
        Initilization of ids that creates their sorted version.

        :param ids: document ids
        """
        self._ids = ids
        self._sorted_indices = np.argsort(self._ids)
        self._sorted_ids = np.array(self._ids)[self._sorted_indices]

    def __enter__(self):
        self.open()
        if self._return_self_on_enter:
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def paper_with_references(self, i: int) -> Tuple[L, List[int], List[L]]:
        """
        Obtains mag record and it's known references.

        :param i: index of record
        :return:
            first element is mag record
            second indices of references
            records of references that are known
        """

        record = self[i]
        references = [i for i in self.id_2_index(record.references, raise_error=False) if i is not None]
        ref_records = self[references]

        return record, references, ref_records

    def open(self) -> "MAGPapersList":
        """
        opens this list

        :return: Returns the object itself.
        :rtype: MAGPapersList
        """
        super().__enter__()
        self._records: MAGMutableMemoryMappedRecordFile
        self._records.open()
        return self

    def close(self):
        """
        closes this list
        """
        super().__exit__(None, None, None)
        self._records: MAGMutableMemoryMappedRecordFile
        self._records.close()

    @staticmethod
    def _get_index(path: str) -> List[int]:
        """
        Creates list of offsets from mag index file.

        :param path: path to index file
        :return: list of offsets
        """
        res = []
        with open(path, newline='') as f:
            for r in tqdm(csv.DictReader(f, delimiter="\t"), desc="Indexing MAG"):
                res.append(int(r["file_line_offset"]))
        return res

    def add(self, records: MutableSequence[MAGPapersListRecord], workers: int = 0, reset: bool = False,
            *args, **kwargs):
        with self._lock:
            super().add(records, workers, reset)
            # let's handle the id index

            ids = [r.id for r in records]
            self._ids.extend(ids)
            insert_indices = []
            # we must sort the new ids to insert them in sorted order
            # or else we can get a problem like in following example:
            #   np.insert(np.array([3,7,10]), [1,1,1], [6,5,4])
            #   array([ 3,  6,  5,  4,  7, 10])

            new_ind_sorted = np.argsort(ids)
            sorted_new_ids = np.array(ids)[new_ind_sorted]
            for i in sorted_new_ids:
                insert_indices.append(np.searchsorted(self._sorted_ids, i))

            self._sorted_ids = np.insert(self._sorted_ids, insert_indices, sorted_new_ids)
            new_indices = new_ind_sorted + self._sorted_indices.shape[0]
            self._sorted_indices = np.insert(self._sorted_indices, insert_indices, new_indices)

    @property
    def ids(self) -> Sequence:
        return self._ids

    def id_2_index(self, mid: Union[int, Iterable[int]], raise_error: bool = True) -> Union[
        Optional[int], List[Optional[int]]]:
        """
        Translates mag id to index to this list
        uses binary search
            Complexity: O(log len(self.ids))
        :param mid: the id
        :param raise_error: if true raises error if id is not found, otherwise returns None
        :return: index to list
        :raise KeyError for non-existing id
        """
        from_int = False
        if isinstance(mid, int):
            from_int = True
            mid = [mid]

        res = []
        for pos, i in enumerate(np.searchsorted(self._sorted_ids, mid)):
            i = int(i)
            if i != len(self) and self._sorted_ids[i] == mid[pos]:
                res.append(int(self._sorted_indices[i]))
            elif raise_error:
                raise KeyError()
            else:
                res.append(None)

        return res[0] if from_int else res

    @staticmethod
    def get_filtered_record_file(p: str, workers: int = 0,
                                 create_cache: bool = True,
                                 read_cache_size: int = 65536) -> MAGMutableMemoryMappedRecordFile:
        """
        Returns filtered record file with entries containing title and authors.

        :param p: path to file
        :param workers: number of workers
        :param create_cache: if true the cache is created
        :param read_cache_size: maximal number of cached records to prevent reading from disk when
            MAGMutableMemoryMappedRecordFile will be used
        :return: mag file
        """

        record_file = MAGMutableMemoryMappedRecordFile(p, None, None, read_cache_size=read_cache_size)
        ids = []
        titles = []

        with (FunctorPool(workers=[MagRecordReaderWorker(p) for _ in range(workers)],
                          results_queue_maxsize=1.0) if workers > 0 else nullcontext()) as pool, \
                record_file:  # must be after pool init as the pool might use spawn context

            if workers == 0:
                single_process_reader = MagRecordReaderWorker(p)
                m = partial(map, single_process_reader)
            else:
                m = partial(pool.imap, chunk_size=10_000)

            if workers == 0:
                single_process_reader.begin()
            try:
                line_offsets = copy.copy(record_file.lines)  # we need to copy it as we will change it during iteration
                for i, record in tqdm(enumerate(m(reversed(line_offsets))), desc="Filtering MAG list from file",
                                      total=len(record_file)):

                    if record is None:
                        # incomplete record
                        del record_file[len(record_file) - 1 - i]
                    else:
                        record_id, record_title = record
                        if create_cache:
                            ids.append(record_id)
                            titles.append(record_title)
            finally:
                if workers == 0:
                    single_process_reader.end()

            if create_cache:
                record_file.ids = list(reversed(ids))
                record_file.titles = list(reversed(titles))

        return record_file

    def init_from_file(self, p: str, workers: int = 0):
        """
        Fresh data initialization from file.

        :param p: path to file
        :param workers: you can pass number of workers that will be used for getting features in parallel fashion
        """
        with self._lock:
            self.reset()
            self._all_features = None
            self._records = []
            record_file = self.get_filtered_record_file(p, workers=workers)
            self._init_ids(record_file.ids)
            with record_file:
                super().add(record_file, workers=workers)
            self.flush_cache_of_record_file()  # we wanted to just speed up the init phase and save memory for latter

    @classmethod
    def from_file(cls, p: Union[str, MAGMutableMemoryMappedRecordFile], embed_dim=64, k_neighbors=100,
                  match_threshold: float = 0.75, max_year_diff: int = 0, use_gpus: bool = True, fp16: bool = True,
                  workers: int = 0, features_part_size: int = 81_920,
                  manager: Optional[Union[Sequence["PapersListManager"], "PapersListManager"]] = None,
                  allow_gpus: Optional[Sequence[Sequence[int]]] = None,
                  clear_side_embeddings: bool = False,
                  move_record_data_to_manager: Optional[multiprocessing.Manager] = None,
                  read_cache_size: int = 65536,
                  stub_init: bool = False,
                  fulltext_search: Optional[str] = None) -> Union["MAGPapersList", List["MAGPapersList"]]:
        """
        Loads MAG list from file.
        Omits a record if a title or author/s is missing.

        :param p: path to file or already loaded record file
        :param embed_dim: Number of dimensions of title representation obtained by hashing vectorizer.
            This representation is used for nearest neighbors search.
        :param k_neighbors: Number of neighbors for nearest neighbors search of titles.
        :param match_threshold: Score threshold for matching. All above are ok.
            Matching is used for title and list of authors. For authors at least one of them must have a match.
            To get more info about how the score is calculated se  :func:`~similarities.similarity_score` method.
        :param max_year_diff: Allows to soften the year equality to accept also works that are distant x years from each
            other. E.g. setting max_year_diff to 1 we match also papers with absolute difference 1, which might be
             useful for preprints, which are usually released beforehand.
        :param use_gpus:  If false gpus are not used. Else uses all available gpus in shard mode (splits it).
        :param fp16: true activates fp16 precision for GPU
            Only for gpu
        :param workers: you can pass number of workers that will be used for getting features in parallel fashion
        :param features_part_size: maximal number of records per parallel worker for features extraction
            Is used when the parallel processing is used for features extraction.
        :param manager: Declares whether the papers list should be created through multiprocessing manager.
            the list option is used together with list option of allow_gpus.
        :param allow_gpus: If not None it will create multiple lists with different gpus for each manager.
            You can specify gpus that should be used for each list.
        :param clear_side_embeddings: If true the embeddings are cleared after the list is created.
            If multiple managers are used the side embeddings are not stored at all.
        :param move_record_data_to_manager: If not None the record data is moved to the manager before the list is created.
        :param read_cache_size: maximal number of cached records to prevent reading from disk when
            MAGMutableMemoryMappedRecordFile will be used
            Used only when the record file is not already loaded
        :param stub_init: If true it is initialized as a stub.
        :param fulltext_search: Path to file with fulltext search sqlite database. If None fulltext search is not used.
        :return: paper list
            or multiple lists if allow_gpus is not None
        """

        # Mag sample:
        # {
        #   "PaperId": 2401921836,
        #   "OriginalTitle": "Process for the production of rigid PVC foils",
        #   "Year": 1989,
        #   "Authors": ["Peter Wedl", "Kurt Dr. Worschech", "Erwin Fleischer", "Ernst Udo Brand"],
        #   "References": [2402455798, 2811934493, 2867207111],
        #   "Fields": ["Carbon", "Materials science", "Molecule", "Polymer chemistry", "Emulsion"],
        #   "Doi": "",
        #   "Journal": ""
        # }
        if allow_gpus is not None and isinstance(manager, PapersListManager):
            raise ValueError("If you want to use allow_gpus you have to specify multiple managers.")

        if allow_gpus is not None and len(allow_gpus) != len(manager):
            raise ValueError("You must specify the same number of managers as allow_gpus groups.")

        if isinstance(p, str):
            record_file = cls.get_filtered_record_file(p, workers=workers, read_cache_size=read_cache_size)
        else:
            record_file = p

        embeddings = None

        if manager is not None and not isinstance(manager, PapersListManager) and not stub_init:
            with record_file:
                embeddings = cls.create_shared_embeddings(record_file, embed_dim, workers, features_part_size)

        if move_record_data_to_manager is not None:
            record_file.move_data_to_manager(move_record_data_to_manager)

        init_ids = record_file.ids

        if manager is not None and not isinstance(manager, PapersListManager):
            record_file.flush_cache()

        mag_list = cls.init_multiple(record_file, embeddings, embed_dim, k_neighbors, match_threshold, max_year_diff,
                                     use_gpus, fp16, workers, features_part_size, manager, allow_gpus,
                                     clear_side_embeddings,
                                     stub_init, init_ids=init_ids, fulltext_search=fulltext_search)

        for mag in mag_list if isinstance(mag_list, list) else [mag_list]:
            mag.flush_cache_of_record_file()

        return mag_list


class COREPapersList(PapersList):
    """
    Papers list for CORE dataset.
    """

    def __init__(self, paths: List[str], records: MutableSequence[PapersListRecord], embed_dim=64, k_neighbors=100,
                 match_threshold: float = 0.75, max_year_diff: int = 0, use_gpus: bool = True, fp16: bool = True,
                 init_workers: int = 0, features_part_size: int = 81_920, fulltext_search: Optional[str] = None):
        """
        Initialization of paper list.

        :param paths: paths to files with documents
        :param records: contains:
            title of publication is used for creating features for searching
            year of publication could be None in that case it matches with any year
            authors of publication, at least one author must match during searching
        :param embed_dim: Number of dimensions of title representation obtained be hashing vectorizer.
            This representation is used for nearest neighbors search.
        :param k_neighbors: Number of neighbors for nearest neighbors search of titles.
        :param match_threshold: Score threshold for matching. All above are ok.
            Matching is used for title and list of authors. For authors at least one of them must have a match.
            To get more info about how the score is calculated se  :func:`~similarities.similarity_score` method.
        :param max_year_diff: Allows to soften the year equality to accept also works that are distant x years from each
            other. E.g. setting max_year_diff to 1 we match also papers with absolute difference 1, which might be
             useful for preprints, which are usually released beforehand.
        :param use_gpus:  If false gpus are not used. Else uses all available gpus in shard mode (splits it).
        :param fp16: true activates fp16 precision for GPU
            Only for gpu
        :param init_workers: you can pass number of workers that will be used for getting features in parallel fashion
        :param features_part_size: maximal number of records per parallel worker for features extraction
            Is used when the parallel processing is used for features extraction.
        :param fulltext_search: Path to file with fulltext search sqlite database. If None fulltext search is not used.
        """

        super().__init__(records, embed_dim, k_neighbors, match_threshold, max_year_diff, use_gpus, fp16, init_workers,
                         features_part_size, fulltext_search=fulltext_search)
        self.paths = paths

    def get_path(self, i: int) -> str:
        """
        Get path on index i.
        This method exists just for multiprocessing mode when we need a method for mult. proc. manager.

        :param i: index of a path
        :return: path on given index
        """
        return self.paths[i]

    def get_paths(self) -> List[str]:
        """
        Returns paths.
        This method exists just for multiprocessing mode when we need a method for mult. proc. manager.

        :return: all paths
        """
        return self.paths

    @staticmethod
    def parse_grobid_meta_doc(p: str) -> Optional["grobid_doc.GROBIDMetaDoc"]:
        """
        Parse grobid meta document on given path.

        :param p: path to document
        :return: parsed meta document or None on parse errors
        """
        try:
            return grobid_doc.GROBIDMetaDoc(p)
        except ParseError:
            return None

    @staticmethod
    def parse_grobid_meta_doc_with_bib(p: str) -> Optional["grobid_doc.GROBIDMetaDocWitBib"]:
        """
        Parse grobid meta document on given path.

        :param p: path to document
        :return: parsed meta document or None on parse errors
        """
        try:
            return grobid_doc.GROBIDMetaDocWitBib(p)
        except ParseError:
            return None

    @classmethod
    def get_bib_from_meta_doc(cls, p: str) -> List[Tuple[str, Optional[int], Tuple[str, ...]]]:
        """
        Parsed bibliography of grobid document.

        :param p: path to document
        :return: list of tuples
            title, year, authors
            returns empty list when doc parsing fails
        """
        meta_doc = cls.parse_grobid_meta_doc_with_bib(p)
        if meta_doc is None:
            return []
        return list((bib.title, bib.year, bib.authors) for bib in meta_doc.bibliography.values())

    @classmethod
    def get_data_from_dir(cls, p: str, workers: Optional[Pool] = None, chunk_size: int = 10) \
            -> Tuple[List[str], List[PapersListRecord]]:
        """
        Loads CORE metadata from directory.
        Omits a record if a title or author/s is missing. We allow missing year as there is not much papers
        containing it.

        :param p: Path to folder of folders with grobid xmls
        :param workers: you can pass pool of workers that will activate parallel processing
        :param chunk_size: number of documents in chunk when parallel processing is activated
        :return: paths, PapersListRecord for documents in given directory
        """

        paths = []
        records = []

        xml_paths = sorted(str(p) for p in Path(p).rglob("*.xml"))
        proc_m = map if workers is None else partial(workers.imap, chunksize=chunk_size)

        for i, doc in enumerate(
                tqdm(proc_m(cls.parse_grobid_meta_doc, xml_paths), desc="Reading papers from folder",
                     total=len(xml_paths))):
            doc: grobid_doc.GROBIDMetaDoc
            if doc is None or doc.title is None or len(doc.authors) == 0:
                # incomplete record or invalid parsing
                continue

            paths.append(xml_paths[i])
            records.append(PapersListRecord(
                title=doc.title,
                year=doc.year,
                authors=doc.authors
            ))

        return paths, records

    @classmethod
    def from_dir(cls, p: str, embed_dim=64, k_neighbors=100, match_threshold: float = 0.75, max_year_diff: int = 0,
                 use_gpus: bool = True, fp16: bool = True, workers: int = 0, chunk_size: int = 10,
                 features_part_size: int = 81_920, manager: Optional["PapersListManager"] = None) -> "COREPapersList":
        """
        Loads CORE list from directory.
        Omits a record if a title or author/s is missing. We allow missing year as there is not much papers
        containing it.

        :param p: Path to folder of folders with grobid xmls
        :param embed_dim: Number of dimensions of title representation obtained by hashing vectorizer.
            This representation is used for nearest neighbors search.
        :param k_neighbors: Number of neighbors for nearest neighbors search of titles.
        :param match_threshold: Score threshold for matching. All above are ok.
            Matching is used for title and list of authors. For authors at least one of them must have a match.
            To get more info about how the score is calculated se  :func:`~similarities.similarity_score` method.
        :param max_year_diff: Allows to soften the year equality to accept also works that are distant x years from each
            other. E.g. setting max_year_diff to 1 we match also papers with absolute difference 1, which might be
             useful for preprints, which are usually released beforehand.
        :param use_gpus:  If false gpus are not used. Else uses all available gpus in shard mode (splits it).
        :param fp16: true activates fp16 precision for GPU
            Only for gpu
        :param workers: you can pass number of workers that will activate parallel processing
        :param chunk_size: number of documents in chunk when parallel processing is activated
        :param features_part_size: maximal number of records per parallel worker for features extraction
            Is used when the parallel processing is used.
        :param manager: Declares whether the papers list should be created through multiprocessing manager.
        :return: paper list
        """

        with multiprocessing.get_context("spawn").Pool(workers) if workers > 0 else nullcontext() as pool:
            paths, records = cls.get_data_from_dir(p, pool if workers > 0 else None, chunk_size)

        return (cls if manager is None else manager.COREPapersList)(paths=paths,
                                                                    records=records,
                                                                    embed_dim=embed_dim,
                                                                    k_neighbors=k_neighbors,
                                                                    match_threshold=match_threshold,
                                                                    max_year_diff=max_year_diff,
                                                                    use_gpus=use_gpus,
                                                                    fp16=fp16,
                                                                    init_workers=workers,
                                                                    features_part_size=features_part_size)

    def bib_generator(self, workers: Optional[Pool] = None, chunk_size: int = 10) -> \
            Generator[Tuple[int, Tuple[str, Optional[int], List[str]]], None, None]:
        """
        Generate bib entries from files.

        :param workers: you can pass pool of workers that will activate parallel processing
        :param chunk_size: number of documents in chunk when parallel processing is activated
        :return: Generates tuple in form of:
            doc index, (bib title, bib year, bib authors)
        """
        if workers is not None:
            proc_m = partial(workers.imap, chunksize=chunk_size)
        else:
            proc_m = map

        for i, doc_bib in enumerate(proc_m(self.get_bib_from_meta_doc, self.paths)):
            for bib in doc_bib:
                yield i, bib

    def identify_references(self, paper_list: Optional[PapersList] = None, workers: Optional[Union[Pool, int]] = None,
                            chunk_size: int = 10, batch_size: int = 128) -> List[List[int]]:
        """
        Identifies each bibliographic entry with index in a paper list

        :param paper_list: You can provide paper list that will be used for matching bib entry. If None
            the current one is used.
        :param workers: you can pass pool of workers that will activate parallel processing
        :param chunk_size: number of documents in chunk when parallel processing is activated
        :param batch_size: Maximal number of samples in a batch when searching for same papers.
        :return: list of matched references for each document
        """

        if paper_list is None:
            paper_list = self

        core_doc_2_references: List[List[int]] = [[] for _ in range(len(self))]

        with multiprocessing.get_context("spawn").Pool(workers) if isinstance(workers,
                                                                              int) and workers > 0 else nullcontext() as pool:
            workers = pool if isinstance(workers, int) and workers > 0 else None

            for b in tqdm(BatcherIter(self.bib_generator(workers, chunk_size), batch_size=batch_size),
                          desc=f"Mapping references", unit="batch"):
                doc_indices = []
                bib = []
                for ind, (title, year, authors) in b:
                    doc_indices.append(ind)
                    bib.append(PapersListRecord(title, year, authors))

                for i, search_res in enumerate(paper_list.batch_search(bib)):
                    if search_res is not None:
                        core_doc_2_references[doc_indices[i]].append(search_res)

        return core_doc_2_references


class PapersListManager(BaseManager):
    pass


PapersListManager.register('PapersList', PapersList,
                           exposed=["__getitem__", "__setitem__", "__len__", "__enter__", "__exit__",
                                    "return_self_on_enter", "to_other_mapping", "add", "batch_search",
                                    "batch_search_nearest_by_index", "batch_search_nearest",
                                    "batch_filter_search_results", "stub",
                                    "get_match_threshold", "get_max_year_diff", "set_search_workers",
                                    "clear_side_embeddings", "load_records_to_memory", "get_records",
                                    "fulltext_search_title"]
                           )

PapersListManager.register('MAGPapersList', MAGPapersList,
                           exposed=["__getitem__", "__setitem__", "__len__",
                                    "to_other_mapping", "add", "batch_search",
                                    "batch_search_nearest_by_index", "batch_search_nearest",
                                    "batch_filter_search_results", "stub",
                                    "id_2_index", "flush_cache_of_record_file", "__enter__", "__exit__",
                                    "return_self_on_enter", "from_file", "init_from_file",
                                    "get_match_threshold", "get_max_year_diff", "set_search_workers",
                                    "clear_side_embeddings", "paper_with_references", "load_records_to_memory",
                                    "get_records", "fulltext_search_title"]
                           )

PapersListManager.register('COREPapersList', COREPapersList,
                           exposed=["__getitem__", "__setitem__", "__len__", "__enter__", "__exit__",
                                    "return_self_on_enter", "to_other_mapping", "add", "batch_search",
                                    "batch_search_nearest_by_index", "batch_search_nearest",
                                    "batch_filter_search_results", "stub", "identify_references", "get_paths",
                                    "get_path", "from_dir", "get_match_threshold", "get_max_year_diff",
                                    "set_search_workers", "clear_side_embeddings", "load_records_to_memory",
                                    "get_records", "fulltext_search_title"]
                           )
