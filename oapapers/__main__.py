# -*- coding: UTF-8 -*-
""""
Created on 24.01.22

:author:     Martin Dočekal
"""
import argparse
import atexit
import csv
import ctypes
import json
import logging
import math
import multiprocessing
import os
import random
import re
import sys
import time
import traceback
from contextlib import nullcontext
from functools import partial
from json import JSONDecodeError
from multiprocessing import active_children
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.pool import ThreadPool
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TextIO, Set, Tuple, Dict, List, AbstractSet, Optional, Sequence, Union, Callable, Any, Iterable, \
    Mapping, Collection, Container

import faiss
import numpy as np
import patoolib
import spacy
from cachetools import LFUCache
from tqdm import tqdm
from windpyutils.args import ExceptionsArgumentParser, ArgumentParserError
from windpyutils.files import MapAccessFile
from windpyutils.generic import BatcherIter
from windpyutils.parallel.own_proc_pools import FunctorPool, FunctorWorker
from windpyutils.structures.sorted import SortedSet, SortedMap

from oapapers.bib_entry import Bibliography, BibEntry
from oapapers.citation_spans import CitationStyle, \
    add_missing_harvard_style_citations, match_unk_citation_spans_with_bib, identify_citation_style_of_doc
from oapapers.cython.normalization import initial_and_normalized_authors, normalize_authors, \
    normalize_and_tokenize_string
from oapapers.datasets import OADataset, OARelatedWork
from oapapers.document import Document, ABSTRACT_REGEX, OARelatedWorkDocument
from oapapers.document_datasets import S2ORCDocumentDataset, DocumentDataset, COREDocumentDataset, \
    OldS2ORCDocumentDataset
from oapapers.filters import CombinedFilter, NumberOfSectionsFilter, SecNonEmptyHeadlinesFilter, Filter, \
    NumberOfTextPartsInSectionFilter, NumberOfCitationsFilter, FullRecordFilter, CouldEstablishMultLvlHierFilter, \
    IsValidAfterHierPruneFilter, FilterWithID, HasHeadlineFilter, CitationsFracFilter, CitationsGroupsFracFilter, \
    FractionOfCitedDocumentsWithMultiSectionContentFilter
from oapapers.hierarchy import Hierarchy
from oapapers.myjson import json_dumps, json_loads
from oapapers.papers_list import ScopusPapersList, MAGPapersList, COREPapersList, PapersList, \
    MAGPapersListRecord, PapersListManager, PapersListRecordWithId
from oapapers.stats import DocumentsStats, RelatedWorkStats
from oapapers.utils import SharedSortedMapOfSequencesOfIntegers


class ArgumentsManager(object):
    """
    Parsers arguments for script.
    """

    @classmethod
    def parse_args(cls):
        """
        Performs arguments parsing.

        :param cls: arguments class
        :returns: Parsed arguments.
        """

        parser = ExceptionsArgumentParser(
            description="Creation of OAReviews (open access reviews) dataset. It is a dataset created for "
                        "multi-document summarization. The task is to create a summary in form of scientfific survey/review from "
                        "referenced documents and provided title (or structure) of target review.")

        subparsers = parser.add_subparsers()

        extract_s2orc_parser = subparsers.add_parser("extract_s2orc",
                                                     help="Extracts S2ORC archives into indexed metadata and fulltext jsonl files. The output may be used for create_s2orc_metadata_parser and convert_s2orc commands.")
        extract_s2orc_parser.add_argument("original",
                                          help="Path to release directory with two subfolders fulltexts, containing s2orc, papers, abstracts, citations archvies. "
                                               "The results will be saved to this directory.", type=str)
        extract_s2orc_parser.add_argument("-w", "--workers",
                                          help="Values grater than zero activates multiprocessing. "
                                               "It is number of additional workers."
                                               "You can use -1 for using all cpus.",
                                          type=int,
                                          default=0)
        extract_s2orc_parser.add_argument("--subfolders",
                                          help="Names of subfolders in original directory that will be used for "
                                               "extraction. "
                                               "By default these are papers, abstracts, citations, fulltexts."
                                               "The citations has special place as it is processed differently.",
                                          nargs="+",
                                          type=str,
                                          required=False,
                                          default=["papers", "abstracts", "citations", "s2orc"])
        extract_s2orc_parser.add_argument("--no_decompression",
                                          help="If set then no decompression will be performed. "
                                               "It is useful if you already have decompressed files in original directory.",
                                          action="store_true")
        extract_s2orc_parser.set_defaults(func=extract_s2orc)

        create_s2orc_metadata_parser = subparsers.add_parser("create_s2orc_metadata",
                                                             help="Creates metadata for S2ORC dataset. "
                                                                  "The output may be used for convert_s2orc command.")
        create_s2orc_metadata_parser.add_argument("papers",
                                                  help="Path to papers jsonl file containing papers metadata.",
                                                  type=str)
        create_s2orc_metadata_parser.add_argument("abstracts", help="Path to jsonl file with abstracts.", type=str)
        create_s2orc_metadata_parser.add_argument("citation_graph", help="Path to jsonl file with citation_graph.",
                                                  type=str)
        create_s2orc_metadata_parser.add_argument("result",
                                                  help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                                  type=str)
        create_s2orc_metadata_parser.add_argument("-w", "--workers",
                                                  help="Values grater than zero activates multiprocessing. "
                                                       "It is number of additional workers."
                                                       "You can use -1 for using all cpus.",
                                                  type=int,
                                                  default=0)
        create_s2orc_metadata_parser.set_defaults(func=create_s2orc_metadata)

        create_s2orc_records_parser = subparsers.add_parser("create_s2orc_records",
                                                            help="Creates records for S2ORC dataset. ")
        create_s2orc_records_parser.add_argument("metadata",
                                                 help="Path to metadata jsonl file",
                                                 type=str)
        create_s2orc_records_parser.add_argument("result",
                                                 help="Path to file where records will be saved. It will save index of that file on the same path but with .index extension.",
                                                 type=str)
        create_s2orc_records_parser.add_argument("-w", "--workers",
                                                 help="Values grater than zero activates multiprocessing. "
                                                      "It is number of additional workers."
                                                      "You can use -1 for using all cpus.",
                                                 type=int,
                                                 default=0)
        create_s2orc_records_parser.set_defaults(func=create_s2orc_records)

        convert_s2orc_parser = subparsers.add_parser("convert_s2orc",
                                                     help="Converts S2ORC to OAReviews document format. Leaves only records with title, year, authors, and content.")

        convert_s2orc_parser.add_argument("metadata", help="Path to papers jsonl file containing papers metadata.",
                                          type=str)
        convert_s2orc_parser.add_argument("s2orc", help="Path to jsonl file containing fulltexts.", type=str)
        convert_s2orc_parser.add_argument("result",
                                          help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                          type=str)
        convert_s2orc_parser.add_argument("-w", "--workers",
                                          help="Values grater than zero activates multiprocessing. "
                                               "It is number of additional workers."
                                               "You can use -1 for using all cpus.",
                                          type=int,
                                          default=0)
        convert_s2orc_parser.add_argument("-g", "--gpu",
                                          help="Activates usage of GPU.",
                                          action="store_true",
                                          required=False)
        convert_s2orc_parser.add_argument("-f", "--from_i",
                                          help="Processing interval start. (line number of first document)",
                                          type=int,
                                          default=0)
        convert_s2orc_parser.add_argument("-t", "--to_i",
                                          help="Processing interval end. (line number after last document)",
                                          type=int,
                                          default=None
                                          )
        convert_s2orc_parser.set_defaults(func=convert_s2orc)

        convert_s2orc_old_parser = subparsers.add_parser("convert_s2orc_old",
                                                         help="Converts Old (original) S2ORC to OAReviews document format. Leaves only records with title, year, authors, and content.")
        convert_s2orc_old_parser.add_argument("original", help="Path to original dataset.", type=str)
        convert_s2orc_old_parser.add_argument("result",
                                              help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                              type=str)
        convert_s2orc_old_parser.add_argument("-w", "--workers",
                                              help="Values grater than zero activates multiprocessing. "
                                                   "It is number of additional workers."
                                                   "You can use -1 for using all cpus.",
                                              type=int,
                                              default=0)
        convert_s2orc_old_parser.add_argument("-g", "--gpu",
                                              help="Activates usage of GPU.",
                                              action="store_true",
                                              required=False)
        convert_s2orc_old_parser.set_defaults(func=convert_s2orc_old)

        convert_core_parser = subparsers.add_parser("convert_core",
                                                    help="Converts CORE to OAReviews document format. Leaves only records with title, year, authors, and content.")
        convert_core_parser.add_argument("original", help="Path to original dataset.", type=str)
        convert_core_parser.add_argument("result",
                                         help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                         type=str)
        convert_core_parser.add_argument("-m", "--mag",
                                         help="Path to mag .jsonl dataset. Is used for getting fields of study and for extending references.",
                                         type=str,
                                         required=False)
        convert_core_parser.add_argument("-s", "--s2orc",
                                         help="Path to s2orc .jsonl records dataset. Is used for obtaining ids.",
                                         type=str,
                                         required=False)

        convert_core_parser.add_argument("-b", "--batch", help="Number of samples in a batch for GPU.", type=int,
                                         default=128)

        convert_core_parser.add_argument("-w", "--workers",
                                         help="Values grater than zero activates multiprocessing. "
                                              "It is number of additional workers."
                                              "You can use -1 for using all cpus.",
                                         type=int,
                                         default=0)
        convert_core_parser.add_argument("-g", "--gpu",
                                         help="Activates usage of GPU.",
                                         action="store_true",
                                         required=False)
        convert_core_parser.set_defaults(func=convert_core)

        multi_lvl_hier_parser = subparsers.add_parser("multi_lvl_hier",
                                                      help="Converts OAPapers hierarchy from flat to multi-level.")
        multi_lvl_hier_parser.add_argument("original", help="path to OAPapers", type=str)
        multi_lvl_hier_parser.add_argument("result",
                                           help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                           type=str)
        multi_lvl_hier_parser.add_argument("-d", "--discard",
                                           help="Activates discarding of papers for which the multi-level hierarchy could not be obtained.",
                                           action="store_true", required=False)
        multi_lvl_hier_parser.add_argument("-w", "--workers",
                                           help="Values grater than zero activates multiprocessing. "
                                                "It is number of additional workers."
                                                "You can use -1 for using all cpus.",
                                           type=int,
                                           default=0)
        multi_lvl_hier_parser.add_argument("-m", "--multi",
                                           help="Path to file where ids of papers with multi-level hierarchy will be saved.",
                                           type=str, required=False)
        multi_lvl_hier_parser.add_argument("-f", "--from_index", help="The processing will start from this index",
                                           type=int, default=0, required=False)
        multi_lvl_hier_parser.add_argument("-t", "--to_index", help="The processing will end before this index",
                                           type=int, default=None, required=False)
        multi_lvl_hier_parser.set_defaults(func=multi_lvl_hier)

        pruning_hier_parser = subparsers.add_parser("pruning_hier", help="Prunes OAPapers hierarchy. "
                                                                         "Will remove documents that will became empty.")
        pruning_hier_parser.add_argument("original", help="path to OAPapers", type=str)
        pruning_hier_parser.add_argument("result",
                                         help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                         type=str)
        pruning_hier_parser.add_argument("-e", "--empty-headlines",
                                         help="Removes all sections with empty headline.",
                                         action="store_true", required=False)
        pruning_hier_parser.add_argument("-n", "--no-text",
                                         help="Removes all sections without text content (in whole sub-hierarchy).",
                                         action="store_true", required=False)
        pruning_hier_parser.add_argument("-p", "--plain_latin",
                                         help="Minimal coverage of plain latin characters in a section headline.",
                                         type=float, required=False)
        pruning_hier_parser.add_argument("-t", "--named_text_blocks",
                                         help="Removes hierarchies having one of targeted headlines and directly "
                                              "containing text block. The headlines are: figure",
                                         action="store_true", required=False)
        pruning_hier_parser.add_argument("-w", "--workers",
                                         help="Values grater than zero activates multiprocessing. "
                                              "It is number of additional workers."
                                              "You can use -1 for using all cpus.",
                                         type=int,
                                         default=0)
        pruning_hier_parser.add_argument("--from_index", help="The processing will start from this index", type=int,
                                         default=0, required=False)
        pruning_hier_parser.add_argument("--to_index", help="The processing will end before this index", type=int,
                                         default=None, required=False)
        pruning_hier_parser.add_argument("-f", "--filter_ids",
                                         help="Whether to filter out ids of removed documents.",
                                         action="store_true", required=False)
        pruning_hier_parser.set_defaults(func=pruning_hier)

        extension_subset_parser = subparsers.add_parser("extension_subset",
                                                        help="Creates a subset of OAPapers dataset with documents from second dataset that are not present in the first on."
                                                             "Whether two documents are the same is decided according to heuristic that uses title, year, and authors. "
                                                             "When two documents are considered the same only the document from first dataset is in the result. "
                                                             "It will also resolve new references in bibliography."
                                                        )
        extension_subset_parser.add_argument("first", help="Path to first dataset.", type=str)
        extension_subset_parser.add_argument("second", help="Path to second dataset.", type=str)
        extension_subset_parser.add_argument("result",
                                             help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                             type=str)
        extension_subset_parser.add_argument("--match_threshold",
                                             help="Score threshold for matching. All above or equal are match.",
                                             type=float,
                                             default=0.75)
        extension_subset_parser.add_argument("--max_year_diff",
                                             help="Maximal difference in years for matching.",
                                             type=int,
                                             default=2.0)
        extension_subset_parser.add_argument("-b", "--batch", help="Number of samples in a batch for GPU.", type=int,
                                             default=8192)
        extension_subset_parser.add_argument("-w", "--workers",
                                             help="Values grater than zero activates multiprocessing. "
                                                  "It is number of additional workers."
                                                  "You can use -1 for using all cpus.",
                                             type=int,
                                             default=0)
        extension_subset_parser.set_defaults(func=extension_subset)

        extend_ids_parser = subparsers.add_parser("extend_ids",
                                                  help="Changes ids in second dataset to ids that will be subsequent to the last id in the first dataset"
                                                  )
        extend_ids_parser.add_argument("first", help="Path to first dataset.", type=str)
        extend_ids_parser.add_argument("second", help="Path to second dataset.", type=str)
        extend_ids_parser.add_argument("result",
                                       help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                       type=str)
        extend_ids_parser.add_argument("-w", "--workers",
                                       help="Values grater than zero activates multiprocessing. "
                                            "It is number of additional workers."
                                            "You can use -1 for using all cpus.",
                                       type=int,
                                       default=0)
        extend_ids_parser.set_defaults(func=extend_ids)

        extend_parser = subparsers.add_parser("extend",
                                              help="Extends a dataset in OAReviews document format with documents from second dataset. "
                                                   "The second must contain only documents that are not in first, use extension_subset for that. "
                                                   "It will also resolve new references in bibliography."
                                              )
        extend_parser.add_argument("first", help="Path to first dataset.", type=str)
        extend_parser.add_argument("second", help="Path to second dataset.", type=str)
        extend_parser.add_argument("result",
                                   help="Path to file where documents will be saved. It will save index of that file on the same path but with .index extension.",
                                   type=str)
        extend_parser.add_argument("--match_threshold",
                                   help="Score threshold for matching. All above or equal are match.",
                                   type=float,
                                   default=0.75)
        extend_parser.add_argument("--max_year_diff",
                                   help="Maximal difference in years for matching.",
                                   type=int,
                                   default=2.0)
        extend_parser.add_argument("-b", "--batch", help="Number of samples in a batch for GPU.", type=int,
                                   default=8192)
        extend_parser.add_argument("-d", "--docs",
                                   help="Number of documents processed together in single process. It allows to make bibliography batches among documents.",
                                   type=int,
                                   default=256)
        extend_parser.add_argument("-w", "--workers",
                                   help="Values grater than zero activates multiprocessing. "
                                        "It is number of additional workers."
                                        "You can use -1 for using all cpus.",
                                   type=int,
                                   default=0)
        extend_parser.set_defaults(func=extend)

        related_work = subparsers.add_parser("related_work", help="Creation of OARelatedWork dataset.")
        related_work.add_argument("documents", help="Path to dataset with documents in OAPapers format.", type=str)
        related_work.add_argument("reviews",
                                  help="Path to file where reviews will be saved. It will save index of that file on the same path but with .index extension.",
                                  type=str)
        related_work.add_argument("references",
                                  help="Path to file where referenced documents in reviews will be saved. It will save index of that file on the same path but with .index extension.",
                                  type=str)
        related_work.add_argument("--references_dataset",
                                  help="By default the data source for references is the same as for reviews, but you can use different one using this argument. WARNING: both dataset must use the same ids because of identification.",
                                  type=str)
        related_work.add_argument("-w", "--workers",
                                  help="Values grater than zero activates multiprocessing. "
                                       "It is number of additional workers."
                                       "You can use -1 for using all cpus.",
                                  type=int,
                                  default=0)
        related_work.add_argument("-u", "--unordered",
                                  help="The order of documents will be ignored. Allows more effective processing.",
                                  action="store_true")
        related_work.set_defaults(func=create_related_work)

        related_work_filter_parser = subparsers.add_parser("related_work_filter",
                                                           help="Filtration of OARelatedWork dataset.")
        related_work_filter_parser.add_argument("related_work", help="Path to dataset file with related work.",
                                                type=str)
        related_work_filter_parser.add_argument("references", help="Path to dataset file with references.", type=str)
        related_work_filter_parser.add_argument("res_related_work",
                                                help="Path to file where related work will be saved. Prints stats at the stdout. It will save index of that file on the same path but with .index extension.",
                                                type=str)
        related_work_filter_parser.add_argument("res_references",
                                                help="Path to file where referenced documents in reviews will be saved. It will save index of that file on the same path but with .index extension.",
                                                type=str)
        related_work_filter_parser.add_argument("--sec-non-empty-headlines-ref",
                                                help="Filters out all referenced documents with a section with empty headline.",
                                                action="store_true")

        related_work_filter_parser.add_argument("--has-abstract-rev",
                                                help="Filters out all reviews without abstract that have less than given number of text parts.",
                                                type=int, default=2)

        related_work_filter_parser.add_argument("--has-abstract-ref",
                                                help="Filters out all referenced documents without abstract that have less than given number of text parts.",
                                                type=int, default=2)

        related_work_filter_parser.add_argument("--min-cit", help="Minimal number of citations in a review.", type=int,
                                                default=0)
        related_work_filter_parser.add_argument("--max-cit", help="Maximal number of citations in a review.", type=int,
                                                default=math.inf)

        related_work_filter_parser.add_argument("--min-cit-frac",
                                                help="Minimal fraction of known citation spans in a review.",
                                                type=float,
                                                default=0)
        related_work_filter_parser.add_argument("--max-cit-frac",
                                                help="Maximal fraction of known citation spans in a review.",
                                                type=float,
                                                default=1.0)

        related_work_filter_parser.add_argument("--min-cit-group-frac",
                                                help="Minimal fraction of known citation span groups in a review.",
                                                type=float,
                                                default=0)
        related_work_filter_parser.add_argument("--max-cit-group-frac",
                                                help="Maximal fraction of known citation span groups in a review.",
                                                type=float,
                                                default=1.0)

        related_work_filter_parser.add_argument("--min-sec-rev",
                                                help="Minimal number of sub-sections in related work section in target document.",
                                                type=int, default=0)
        related_work_filter_parser.add_argument("--max-sec-rev",
                                                help="Maximal number of sub-sections in related work section in target document.",
                                                type=int,
                                                default=math.inf)
        related_work_filter_parser.add_argument("--min-sec-ref",
                                                help="Minimal number of sections in a referenced document.",
                                                type=int, default=0)
        related_work_filter_parser.add_argument("--max-sec-ref",
                                                help="Maximal number of sections in a referenced document.",
                                                type=int, default=math.inf)

        related_work_filter_parser.add_argument("--min-par-rev",
                                                help="Minimal number of text parts in related work section in target document.",
                                                type=int,
                                                default=0)
        related_work_filter_parser.add_argument("--max-par-rev",
                                                help="Maximal number of text parts in related work section in target document.",
                                                type=int,
                                                default=math.inf)
        related_work_filter_parser.add_argument("--min-par-ref",
                                                help="Minimal number of text parts in a section in a referenced document.",
                                                type=int,
                                                default=0)
        related_work_filter_parser.add_argument("--max-par-ref",
                                                help="Maximal number of text parts in a section in a referenced document.",
                                                type=int,
                                                default=math.inf)
        related_work_filter_parser.add_argument("--min-fraction-of-cited-documents-with-multi-section-content-filter",
                                                help="Minimal fraction of cited documents with multi section content in a review.",
                                                type=float,
                                                default=0)
        related_work_filter_parser.add_argument("--max-fraction-of-cited-documents-with-multi-section-content-filter",
                                                help="Maximal fraction of cited documents with multi section content in a review.",
                                                type=float,
                                                default=1.0)
        related_work_filter_parser.add_argument("-w", "--workers",
                                                help="Values grater than zero activates multiprocessing. "
                                                     "It is number of additional workers."
                                                     "You can use -1 for using all cpus.",
                                                type=int,
                                                default=0)
        related_work_filter_parser.set_defaults(func=filter_related_work)

        filter_features_parser = subparsers.add_parser("filter_features",
                                                       help="Prints tsv containing features that are used by filters on stdout.")
        filter_features_parser.add_argument("dataset", help="Path to dataset file", type=str)
        filter_features_parser.add_argument("--references",
                                            help="Path to dataset file with references. It will use the same as for dataset argument if there is no given.",
                                            type=str)
        filter_features_parser.add_argument("-rw", help="The input has related work format.", action="store_true")
        filter_features_parser.add_argument("-w", "--workers",
                                            help="Values grater than zero activates multiprocessing. "
                                                 "It is number of additional workers."
                                                 "You can use -1 for using all cpus.",
                                            type=int,
                                            default=0)
        filter_features_parser.set_defaults(func=create_filter_features)

        stats_parser = subparsers.add_parser("stats", help="Prints stats on stdout.")
        stats_parser.add_argument("dataset", help="Path to dataset file", type=str)
        stats_parser.add_argument("--rw", help="The input has related work format.", action="store_true")
        stats_parser.add_argument("--references",
                                  help="Path to dataset file with references. If the rw is activated and references"
                                       "are provided it will make RelatedWorkStats instead of DocumentsStats.",
                                  type=str, default=None)
        stats_parser.add_argument("-w", "--workers",
                                  help="Values grater than zero activates multiprocessing. "
                                       "It is number of additional workers."
                                       "You can use -1 for using all cpus.",
                                  type=int,
                                  default=0)
        stats_parser.set_defaults(func=create_stats)

        enhance_mag_with_core_parser = subparsers.add_parser("enhance_mag_with_core",
                                                             help="Enhances mag full record with CORE data. "
                                                                  "Not all fields will be filled. Only id, title, year if it is known, authors and references.")
        enhance_mag_with_core_parser.add_argument("mag", help="Path to original mag .jsonl dataset.", type=str)
        enhance_mag_with_core_parser.add_argument("core", help="Path to folder of folders with grobid xmls.", type=str)
        enhance_mag_with_core_parser.add_argument("result", help="Path where results will be saved.", type=str)
        enhance_mag_with_core_parser.add_argument("-b", "--batch", help="Number of samples in a batch for GPU.",
                                                  type=int,
                                                  default=8192)
        enhance_mag_with_core_parser.add_argument("-w", "--workers",
                                                  help="Values grater than zero activates multiprocessing. "
                                                       "It is number of additional workers."
                                                       "You can use -1 for using all cpus.",
                                                  type=int,
                                                  default=0)
        enhance_mag_with_core_parser.add_argument("-i", "--identified_references",
                                                  help="MAINLY FOR DEBUGGING PURPOSES. "
                                                       "In case the file on given path exists it will be loaded and used"
                                                       "as identified CORE bibliographies. It always saves identified "
                                                       "references to this given file as they could be updated when the"
                                                       "identified references file is just from the first stage.",
                                                  type=str,
                                                  default=None)
        enhance_mag_with_core_parser.set_defaults(func=extend_mag_with_core)

        create_papers_2_mag_mapping_parser = subparsers.add_parser("create_papers_2_mag_mapping",
                                                                   help="Creates mapping from papers indices to mag indices.")

        create_papers_2_mag_mapping_parser.add_argument("dataset", help="Path to OAPapers dataset.", type=str)
        create_papers_2_mag_mapping_parser.add_argument("mag", help="Path to mag in .jsonl format.", type=str)
        create_papers_2_mag_mapping_parser.add_argument("-b", "--batch", help="Number of samples in a batch for GPU.",
                                                        type=int,
                                                        default=8192)
        create_papers_2_mag_mapping_parser.add_argument("-w", "--workers",
                                                        help="Values grater than zero activates multiprocessing. "
                                                             "It is number of additional workers."
                                                             "You can use -1 for using all cpus.",
                                                        type=int,
                                                        default=0)
        create_papers_2_mag_mapping_parser.add_argument("--force_gpu_split",
                                                        help="By default the index has multiple copies on each GPU. You can force to split the index"
                                                             "among GPUs to reduce memory usage.",
                                                        action="store_true")
        create_papers_2_mag_mapping_parser.set_defaults(func=create_papers_2_mag_mapping)

        enhance_papers_with_mag_parser = subparsers.add_parser("enhance_papers_with_mag",
                                                               help="Enhances papers with mag informations.")
        enhance_papers_with_mag_parser.add_argument("dataset", help="Path to OAPapers dataset.",
                                                    type=str)
        enhance_papers_with_mag_parser.add_argument("mag", help="Path to mag in .jsonl format.", type=str)
        enhance_papers_with_mag_parser.add_argument("to_mag",
                                                    help="Papers indices to mag indices. It is the output of create_papers_2_mag_mapping.",
                                                    type=str)
        enhance_papers_with_mag_parser.add_argument("result", help="Path where results will be saved.", type=str)
        enhance_papers_with_mag_parser.add_argument("-j", "--just_references",
                                                    help="Whether just the citations should be updated.",
                                                    action="store_true")
        enhance_papers_with_mag_parser.add_argument("--match_threshold",
                                                    help="Score threshold for matching. All above or equal are match.",
                                                    type=float,
                                                    default=0.75)
        enhance_papers_with_mag_parser.add_argument("-w", "--workers",
                                                    help="Values grater than zero activates multiprocessing. "
                                                         "It is number of additional workers."
                                                         "You can use -1 for using all cpus.",
                                                    type=int,
                                                    default=0)
        enhance_papers_with_mag_parser.set_defaults(func=enhance_papers_with_mag)

        deduplication_parser = subparsers.add_parser("deduplication",
                                                     help="Performs dataset deduplication.")
        deduplication_parser.add_argument("dataset", help="Path to OAPapers dataset.", type=str)
        deduplication_parser.add_argument("result", help="Path where results will be saved.", type=str)
        deduplication_parser.add_argument("-b", "--batch", help="Number of samples in a batch for GPU.",
                                          type=int,
                                          default=128)

        deduplication_parser.add_argument("--match_threshold",
                                          help="Score threshold for matching. All above or equal are match.",
                                          type=float,
                                          default=1.0)
        deduplication_parser.add_argument("--max_year_diff",
                                          help="Allows to soften the year equality to accept also works that are "
                                               "distant x years from each other. E.g. setting max_year_diff to 1 we "
                                               "match also papers with absolute difference 1, which might be useful "
                                               "for preprints, which are usually released beforehand.",
                                          type=int,
                                          default=2)
        deduplication_parser.add_argument("-w", "--workers",
                                          help="Values grater than zero activates multiprocessing. "
                                               "It is number of additional workers."
                                               "You can use -1 for using all cpus.",
                                          type=int,
                                          default=0)
        deduplication_parser.set_defaults(func=deduplication)

        identify_bibliography_spans_parser = subparsers.add_parser("identify_bibliography",
                                                                   help="Tries to identify bibliography with unknown "
                                                                        "reference, also updates citation spans.")
        identify_bibliography_spans_parser.add_argument("dataset", help="Path to OAPapers dataset.", type=str)
        identify_bibliography_spans_parser.add_argument("result", help="Path where results will be saved.", type=str)
        identify_bibliography_spans_parser.add_argument("-b", "--batch", help="Number of samples in a batch for GPU.",
                                                        type=int,
                                                        default=8192)
        identify_bibliography_spans_parser.add_argument("-d", "--docs",
                                                        help="Number of documents processed together in single process. It allows to make bibliography batches among documents.",
                                                        type=int,
                                                        default=256)
        identify_bibliography_spans_parser.add_argument("--match_threshold",
                                                        help="Score threshold for matching. All above or equal are match.",
                                                        type=float,
                                                        default=0.75)
        identify_bibliography_spans_parser.add_argument("--max_year_diff",
                                                        help="Allows to soften the year equality to accept also works that are "
                                                             "distant x years from each other. E.g. setting max_year_diff to 1 we "
                                                             "match also papers with absolute difference 1, which might be useful "
                                                             "for preprints, which are usually released beforehand.",
                                                        type=int,
                                                        default=2)
        identify_bibliography_spans_parser.add_argument("-w", "--workers",
                                                        help="Values grater than zero activates multiprocessing. "
                                                             "It is number of additional workers."
                                                             "You can use -1 for using all cpus.",
                                                        type=int,
                                                        default=0)
        identify_bibliography_spans_parser.add_argument("-f", "--from_i",
                                                        help="Processing interval start. (line number of first document)",
                                                        type=int,
                                                        default=0)
        identify_bibliography_spans_parser.add_argument("-t", "--to_i",
                                                        help="Processing interval end. (line number after last document)",
                                                        type=int,
                                                        default=None)
        identify_bibliography_spans_parser.add_argument("-s", "--search",
                                                        help="Search for bibliography in this dataset. Be default it searches in the same dataset.",
                                                        type=str,
                                                        default=None)
        identify_bibliography_spans_parser.add_argument("--force_gpu_split",
                                                        help="By default the index has multiple copies on each GPU. You can force to split the index"
                                                             "among GPUs to reduce memory usage.",
                                                        action="store_true")
        identify_bibliography_spans_parser.add_argument("--title_db",
                                                        help="Title database that should be used for fulltext_search",
                                                        type=str,
                                                        default=None)
        identify_bibliography_spans_parser.set_defaults(func=identify_bibliography)

        enrich_bibliography_from_citation_graph_parser = subparsers.add_parser(
            "enrich_bibliography_from_citation_graph",
            help="Tries to identify bibliography with unknown reference, using citation graph.")
        enrich_bibliography_from_citation_graph_parser.add_argument("dataset",
                                                                    help="Path to OAPapers dataset that should be updated.",
                                                                    type=str)
        enrich_bibliography_from_citation_graph_parser.add_argument("result",
                                                                    help="Path where results will be saved.",
                                                                    type=str)
        enrich_bibliography_from_citation_graph_parser.add_argument("citation_graph",
                                                                    help="Path to json file with citation graph.",
                                                                    type=str)
        enrich_bibliography_from_citation_graph_parser.add_argument("-i", "--id",
                                                                    help="Name of id field that corresponds to id in citation graph.",
                                                                    type=str,
                                                                    default="s2orc_id")
        enrich_bibliography_from_citation_graph_parser.add_argument("-s", "--search",
                                                                    help="By default it searches for OA id in the same dataset. "
                                                                         "You can specify path to dataset where it should search instead.",
                                                                    type=str,
                                                                    default=None)
        enrich_bibliography_from_citation_graph_parser.add_argument("-w", "--workers",
                                                                    help="Values grater than zero activates multiprocessing. "
                                                                         "It is number of additional workers."
                                                                         "You can use -1 for using all cpus.",
                                                                    type=int,
                                                                    default=0)
        enrich_bibliography_from_citation_graph_parser.add_argument("--title_match_threshold",
                                                                    help="Score threshold for matching titles. All above or equal are match.",
                                                                    type=float,
                                                                    default=0.75)
        enrich_bibliography_from_citation_graph_parser.add_argument("--authors_match_threshold",
                                                                    help="Score threshold for matching authors. All above or equal are match.",
                                                                    type=float,
                                                                    default=0.75)
        enrich_bibliography_from_citation_graph_parser.add_argument("--year_diff_threshold",
                                                                    help="Allows to soften the year equality to accept also works that are "
                                                                         "distant x years from each other. E.g. setting max_year_diff to 1 we "
                                                                         "match also papers with absolute difference 1, which might be useful "
                                                                         "for preprints, which are usually released beforehand.",
                                                                    type=int,
                                                                    default=2)
        enrich_bibliography_from_citation_graph_parser.add_argument("-f", "--from_i",
                                                                    help="Processing interval start. (line number of first document)",
                                                                    type=int,
                                                                    default=0)
        enrich_bibliography_from_citation_graph_parser.add_argument("-t", "--to_i",
                                                                    help="Processing interval end. (line number after last document)",
                                                                    type=int,
                                                                    default=None)
        enrich_bibliography_from_citation_graph_parser.set_defaults(func=enrich_bibliography_from_citation_graph)

        identify_citation_spans_parser = subparsers.add_parser("identify_citation_spans",
                                                               help="Identifies citation spans in dataset.")
        identify_citation_spans_parser.add_argument("dataset", help="Path to dataset file", type=str)
        identify_citation_spans_parser.add_argument("result", help="Path where results will be saved.", type=str)
        identify_citation_spans_parser.add_argument("-w", "--workers",
                                                    help="Values grater than zero activates multiprocessing. "
                                                         "It is number of additional workers."
                                                         "You can use -1 for using all cpus.",
                                                    type=int,
                                                    default=0)

        identify_citation_spans_parser.set_defaults(func=identify_citation_spans)

        discard_documents_parser = subparsers.add_parser("discard_documents",
                                                         help="Discards documents that do not have given ids. Also handles update of citations.")
        discard_documents_parser.add_argument("dataset", help="Path to dataset file", type=str)
        discard_documents_parser.add_argument("result", help="Path where results will be saved.", type=str)
        discard_documents_parser.add_argument("ids", help="Path to file with ids of documents to keep.", type=str)
        discard_documents_parser.add_argument("-w", "--workers",
                                              help="Values grater than zero activates multiprocessing. "
                                                   "It is number of additional workers."
                                                   "You can use -1 for using all cpus.",
                                              type=int,
                                              default=0)
        discard_documents_parser.set_defaults(func=discard_documents)

        create_references_parser = subparsers.add_parser("create_references",
                                                         help="Selects all referenced documents in given dataset from another dataset. Doesn't perform filtration of ids in documents.")
        create_references_parser.add_argument("documents",
                                              help="Path to dataset with papers which are citing documents you want to select.",
                                              type=str)
        create_references_parser.add_argument("results",
                                              help="Path to file where results will be saved",
                                              type=str)
        create_references_parser.add_argument("references",
                                              help="Path to file with references.",
                                              type=str)
        create_references_parser.add_argument("-w", "--workers",
                                              help="Values grater than zero activates multiprocessing. "
                                                   "It is number of additional workers."
                                                   "You can use -1 for using all cpus.",
                                              type=int,
                                              default=0)
        create_references_parser.set_defaults(func=create_references)

        sort_dataset_parser = subparsers.add_parser("sort_dataset", help="Sorts dataset by ids.")
        sort_dataset_parser.add_argument("dataset",
                                         help="Path to dataset with papers that should be sorted.",
                                         type=str)
        sort_dataset_parser.add_argument("results",
                                         help="Path to file where results will be saved",
                                         type=str)
        sort_dataset_parser.set_defaults(func=sort_dataset)

        make_rw_train_val_test_splits_parser = subparsers.add_parser("make_rw_train_val_test_splits",
                                                                     help="Makes train, val and test splits.")
        make_rw_train_val_test_splits_parser.add_argument("dataset",
                                                          help="Path to dataset with papers that should be used for splits.",
                                                          type=str)
        make_rw_train_val_test_splits_parser.add_argument("references",
                                                          help="Path to dataset with referenced papers.",
                                                          type=str)
        make_rw_train_val_test_splits_parser.add_argument("train",
                                                          help="Path to file where train split will be saved.",
                                                          type=str)
        make_rw_train_val_test_splits_parser.add_argument("val",
                                                          help="Path to file where val split will be saved.",
                                                          type=str)
        make_rw_train_val_test_splits_parser.add_argument("test",
                                                          help="Path to file where test split will be saved.",
                                                          type=str)
        make_rw_train_val_test_splits_parser.add_argument("--test_prop",
                                                          help="Proportion of test set. It is proportion from a subset"
                                                               "of papers citing just multi section papers.",
                                                          type=float,
                                                          default=0.5)
        make_rw_train_val_test_splits_parser.add_argument("--val_prop",
                                                          help="Proportion of validation set. It is proportion from a subset"
                                                               "of papers citing just multi section papers.",
                                                          type=float,
                                                          default=0.3)
        make_rw_train_val_test_splits_parser.add_argument("-w", "--workers",
                                                          help="Values grater than zero activates multiprocessing. "
                                                               "It is number of additional workers."
                                                               "You can use -1 for using all cpus.",
                                                          type=int,
                                                          default=0)
        make_rw_train_val_test_splits_parser.add_argument("--fixed_seed",
                                                          help="If set, randomseed will be fixed.",
                                                          action="store_true")
        make_rw_train_val_test_splits_parser.set_defaults(func=make_rw_train_val_test_splits)


        create_title_database_parser = subparsers.add_parser("create_title_database",
                                                             help="It creates normalized titles database for given dataset.")
        create_title_database_parser.add_argument("dataset",
                                                  help="Path to dataset with documents in OAPapers format.",
                                                  type=str)
        create_title_database_parser.add_argument("database",
                                                  help="Path where the database will be saved.",
                                                  type=str)
        create_title_database_parser.add_argument("-w", "--workers",
                                                  help="Values grater than zero activates multiprocessing. "
                                                       "It is number of additional workers."
                                                       "You can use -1 for using all cpus.",
                                                  type=int,
                                                  default=0)
        create_title_database_parser.add_argument("-b", "--batch",
                                                  help="Number of samples in a batch for a worker.",
                                                  default=81_920)
        create_title_database_parser.add_argument("--scopus",
                                                  help="Signalizes that the dataset has scopus format.",
                                                  action="store_true")
        create_title_database_parser.set_defaults(func=create_title_database)


        merge_intervals_parser = subparsers.add_parser("merge_intervals",
                                                       help="When processing by intervals you can use this function to merge them.")

        merge_intervals_parser.add_argument("results",
                                            help="Path to file where results will be saved",
                                            type=str)

        merge_intervals_parser.add_argument("intervals",
                                            help="Paths to files with intervals.",
                                            type=str,
                                            nargs="+")

        merge_intervals_parser.set_defaults(func=merge_intervals)

        filter_missing_ids_parser = subparsers.add_parser("filter_missing_ids",
                                                          help="Filters ids of documents that are not in dataset from citation and bibtexts.")
        filter_missing_ids_parser.add_argument("dataset",
                                               help="Path to dataset with documents in OAPapers format.",
                                               type=str)
        filter_missing_ids_parser.add_argument("results",
                                               help="Path to file where results will be saved",
                                               type=str)
        filter_missing_ids_parser.add_argument("-w", "--workers",
                                               help="Values grater than zero activates multiprocessing. "
                                                    "It is number of additional workers."
                                                    "You can use -1 for using all cpus.",
                                               type=int,
                                               default=0)
        filter_missing_ids_parser.set_defaults(func=filter_missing_ids)

        convert_back_from_rw_parser = subparsers.add_parser("convert_back_from_rw",
                                                            help="Converts dataset back from related work format.")
        convert_back_from_rw_parser.add_argument("dataset",
                                                 help="Path to dataset with documents in OAPapers format.",
                                                 type=str)
        convert_back_from_rw_parser.add_argument("results",
                                                 help="Path to file where results will be saved",
                                                 type=str)
        convert_back_from_rw_parser.add_argument("-w", "--workers",
                                                 help="Values grater than zero activates multiprocessing. "
                                                      "It is number of additional workers."
                                                      "You can use -1 for using all cpus.",
                                                 type=int,
                                                 default=0)
        convert_back_from_rw_parser.set_defaults(func=convert_back_from_rw)

        convert_back_to_rw_parser = subparsers.add_parser("convert_back_to_rw",
                                                          help="Converts dataset back to related work format.")
        convert_back_to_rw_parser.add_argument("dataset",
                                               help="Path to dataset with documents in OAPapers format.",
                                               type=str)
        convert_back_to_rw_parser.add_argument("orig",
                                               help="Path to corresponding dataset with documents in related work format which will be used to obtain related work positions in hiearachy.",
                                               type=str)
        convert_back_to_rw_parser.add_argument("results",
                                               help="Path to file where results will be saved",
                                               type=str)
        convert_back_to_rw_parser.add_argument("-w", "--workers",
                                               help="Values grater than zero activates multiprocessing. "
                                                    "It is number of additional workers."
                                                    "You can use -1 for using all cpus.",
                                               type=int,
                                               default=0)
        convert_back_to_rw_parser.set_defaults(func=convert_back_to_rw)

        if len(sys.argv) < 2:
            parser.print_help()
            return None
        try:

            parsed = parser.parse_args()

        except ArgumentParserError as e:
            parser.print_help()
            print("\n" + str(e), file=sys.stdout, flush=True)
            return None

        return parsed


def get_ids_of_filter_passing_documents(dataset: DocumentDataset, f: Filter, verbose: bool = False) -> List[int]:
    """
    Get ids of documents that pass given filter.

    :param dataset: dataset of documents which you want to filter
    :param f: filter that should be used
    :param verbose: Whether a notice should be printed when a document doesn't pass provided filter
    :return: document ids passing filter
    """
    res = []

    dataset.transform = FilterWithID(f)
    saved_chunk_size = None
    saved_max_chunks_per_worker = None
    if hasattr(dataset, "chunk_size"):
        saved_chunk_size = dataset.chunk_size
        dataset.chunk_size = 1000
        saved_max_chunks_per_worker = dataset.max_chunks_per_worker
        dataset.max_chunks_per_worker = 1000

    for passes, doc_id in tqdm(dataset.iter_range(unordered=True), desc="Pre-filtering documents", total=len(dataset)):
        if passes:
            res.append(doc_id)
        elif verbose:
            print(f"The document with id {doc_id} doesn't passed provided filter.")

    if saved_chunk_size is not None:
        dataset.chunk_size = saved_chunk_size
        dataset.max_chunks_per_worker = saved_max_chunks_per_worker

    dataset.transform = None
    return res


class NoTransform:
    """
    No transform.
    """

    def __call__(self, doc: Document) -> Document:
        return doc


class FilterCitationsTransform:
    """
    filters citations
    """

    def __init__(self, allowed_ids: AbstractSet[int]):
        """
        :param allowed_ids: ids that can be left
        """

        self.allowed_ids = allowed_ids

    def __call__(self, doc: Document) -> Document:
        doc.filter_citations(self.allowed_ids)
        return doc


class FilterTransform:
    """
    Uses transform and filter on a document.
    """

    def __init__(self, transform: Callable[[Document], Document], f: Filter, transform_first: bool = True,
                 return_doc: bool = True):
        """
        :param transform: transform of a document
        :param f: filter of a document
        :param transform_first: whether transform should be applied before filter
        :param return_doc: whether the document should be returned
        """
        self.transform = transform
        self.filter = f
        self.transform_first = transform_first
        self.return_doc = return_doc

    def __call__(self, doc: Document) -> Tuple[int, Union[bool, Optional[str]]]:
        """
        Applies transform and filter on a document.
        :param doc: document
        :return:
            id of the document
            transformed document or None if the document doesn't pass the filter
                if return_doc is False, it returns false if the document doesn't pass the filter and true otherwise
        """
        if self.transform_first:
            doc = self.transform(doc)

        if not self.filter(doc):
            return doc.id, None if self.return_doc else False

        if not self.transform_first:
            doc = self.transform(doc)

        if self.return_doc:
            return doc.id, str(doc)
        return doc.id, True


def filter_and_print(documents: DocumentDataset, f: Filter, res_f: TextIO, res_index_f: TextIO,
                     ids_in_dataset: Optional[AbstractSet[int]] = None):
    """
    Filters documents and prints them.

    :param documents: dataset of documents
    :param f: filter for filtrating documents
    :param res_f: where the result should be written
    :param res_index_f: when the results index should be written
    :param ids_in_dataset: All ids that are in the dataset. It is used to filter citation to non-existing documents.
        If None, the filter is not used.
    """

    index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
    index_writer.writeheader()

    old = documents.transform

    t = NoTransform() if ids_in_dataset is None else FilterCitationsTransform(ids_in_dataset)
    documents.transform = FilterTransform(t, f)

    for d_id, d in tqdm(documents, desc="Filtering"):
        if d is not None:
            index_writer.writerow({"key": d_id, "file_line_offset": res_f.tell()})
            print(d, file=res_f)

    documents.transform = old


def write_references(path_to: str, dataset: OADataset, referenced_documents: AbstractSet[int],
                     ids_in_dataset: Optional[AbstractSet[int]] = None):
    """
    Writes referenced documents.

    :param path_to: Path to file where the documents should be saved. Writes also index file to file with name
        path_to.index .
    :param dataset: Already opened dataset of documents.
    :param referenced_documents: ids of documents that should be written.
    :param ids_in_dataset: All ids that are in the dataset. It is used to filter out citation of non-existing documents.
        If None no filtration is involved.
    """
    sorted_ids = sorted(referenced_documents, key=lambda d_id: dataset.mapping[d_id])  # sort according to file offset

    with open(path_to, "w") as references_f, open(path_to + ".index", "w") as references_index_f:
        index_writer = csv.DictWriter(references_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()
        for doc_id in tqdm(sorted_ids, "Writing references"):

            index_writer.writerow({"key": doc_id, "file_line_offset": references_f.tell()})

            if ids_in_dataset is None:
                offset = dataset.mapping[doc_id]
                dataset.file.seek(offset)
                document = dataset.file.readline().rstrip("\n")
            else:
                document = dataset.get_by_id(doc_id)
                # leave only citations of known documents
                document.filter_citations(ids_in_dataset)
            # we should do normalization again as some spans might GET UNK
            print(document, file=references_f)


class ConvertTransform:
    """
    Transformation for document conversion.
    """

    def __init__(self, ids_trans: Mapping[int, int]):
        """
        :param ids_trans: Mapping from old ids to new ids.
        """
        self.ids_trans = ids_trans

    def __call__(self, document: Document) -> Tuple[int, Optional[str]]:
        """
        Conversion of document.

        :param document: document to be converted
        :return: tuple (new document id, new document text representation)
        """
        if document.id in self.ids_trans:
            document.filter_citations(self.ids_trans)  # make sure that the references are really in dataset
            document.translate_ids(self.ids_trans)
            return document.id, str(document)

        return document.id, None


def convert_dataset(dataset: DocumentDataset, res_path: str, verbose: bool = False, f: int = 0,
                    t: Optional[int] = None) -> SortedSet[int]:
    """
    Converts given dataset into OA format and saves it onto given path.
    Only records with title, authors, and content are saved.

    :param dataset: The dataset that should be converted.
    :param res_path: Path to file with results. Also, the *.index file will be created next to the result file.
    :param verbose: Whether a notice should be printed when a document doesn't pass provided filter
    :param f: Start of interval of documents to be converted.
    :param t: End of interval of documents to be converted.
    :return: original ids of written documents
    """

    with open(res_path, "w") as res_f, open(res_path + ".index", "w") as res_index_f:
        # we want to make sure that references papers are really in dataset
        dataset.stub_mode = True
        all_ids = get_ids_of_filter_passing_documents(dataset, FullRecordFilter(), verbose=verbose)
        dataset.stub_mode = False
        all_ids = sorted(all_ids)
        # must be sorted as we want to have same translations even when processing by
        # parts (f and t is set), this is not otherwise guaranteed as get_ids_of_filter_passing_documents is using
        # multiprocessing with unordered=True

        ids_trans = SortedMap(zip(all_ids, range(len(all_ids))))
        del all_ids

        if hasattr(dataset, "workers") and dataset.workers > 0:
            ids_trans.keys_storage = multiprocessing.Array(ctypes.c_int64, ids_trans.keys_storage, lock=False)
            ids_trans.values_storage = multiprocessing.Array(ctypes.c_int64, ids_trans.values_storage, lock=False)

        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        dataset.transform = ConvertTransform(ids_trans)
        dataset.preload_filter = lambda _, d_id: d_id in ids_trans

        t = len(dataset) if t is None else t
        for r in tqdm(dataset.iter_range(f, t, unordered=True), desc=f"Converting documents", unit="doc", total=t - f):
            if r is None:
                continue
            doc_id, doc_repr = r
            if doc_repr is None:
                continue
            index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
            print(doc_repr, file=res_f)

        return SortedSet(ids_trans.keys())


def create_citation_graph(part: str) -> str:
    """
    Creates citation graph from given part of dataset.

    :param part: path to part of dataset
    :return: citation graph in json format
    """

    graph = {}
    with open(part) as part_f:
        for line in part_f:
            record = json.loads(line)

            if record["citingcorpusid"] is None or record["citedcorpusid"] is None:
                # I have seen citedcorpusid with None in the dataset
                continue

            citingcorpusid, citedcorpusid = int(record["citingcorpusid"]), int(record["citedcorpusid"])
            try:
                graph[citingcorpusid].append(citedcorpusid)
            except KeyError:
                graph[citingcorpusid] = [citedcorpusid]

    return json_dumps(graph)


def extract_s2orc(args: argparse.Namespace):
    """
    Extracts S2ORC dataset

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    workers = max(1, workers)

    # find all .gz archives in metadata folder and extract them

    def decompress_archive(task_data):
        p, path_to = task_data
        path_to = Path(path_to)
        res = patoolib.extract_archive(p, outdir=path_to.parent, verbosity=-1, interactive=False)
        # rename file to .jsonl
        Path(path_to.with_suffix("")).rename(path_to)

    def find_and_decompress(folder_path: Path, decompress_archive_flag: bool = True):
        all_archives_mapping = []
        with ThreadPool(workers) as pool:
            for p in folder_path.glob("*.gz"):
                # remove .gz from the end and add .jsonl
                all_archives_mapping.append((str(p), str(p.with_suffix(".jsonl"))))

            if decompress_archive_flag:
                for _ in tqdm(pool.imap_unordered(decompress_archive, all_archives_mapping),
                              total=len(all_archives_mapping), desc=f"Decompressing {folder_path.name}"):
                    ...

        return all_archives_mapping

    def merge_and_create_index(parts: Iterable[str], path_to_merged: str):
        with open(path_to_merged, "w") as merged_f, open(path_to_merged + ".index", "w") as merged_index_f:
            index_writer = csv.DictWriter(merged_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
            index_writer.writeheader()
            for part_path in tqdm(parts, desc="Merging"):
                with open(part_path) as part_f:
                    for i, line in enumerate(part_f):
                        try:
                            record = json.loads(line)
                        except json.decoder.JSONDecodeError as e:
                            print(f"Error while parsing line {i} in {part_path}", file=sys.stderr, flush=True)
                            raise e
                        index_writer.writerow({"key": record["corpusid"], "file_line_offset": merged_f.tell()})
                        print(json.dumps(record), file=merged_f)

    orig_path = Path(args.original)

    # merge and create index
    for folder in args.subfolders:
        archives = find_and_decompress(orig_path / folder, not args.no_decompression)

        if folder == "citations":
            graph = {}
            with multiprocessing.Pool(workers) as pool:
                for g in tqdm(pool.imap_unordered(create_citation_graph, (p[1] for p in archives)),
                              total=len(archives), desc="Creating citation graph"):
                    g = json.loads(g)
                    for k, v in g.items():
                        try:
                            graph[k].extend(v)
                        except KeyError:
                            graph[k] = v

            with open(str(orig_path / f"{folder}.json"), "w") as f:
                json.dump(graph, f)
        else:
            merge_and_create_index((p[1] for p in archives), str(orig_path / f"{folder}.jsonl"))


class S2ORCMetadataCreator(FunctorWorker):
    """
    Creates metadata for S2ORC dataset.
    It will use the papers format as base and add abstracts and citing papers fields.
    """

    def __init__(self, papers_path: str, abstracts_path: str):
        super().__init__()
        self.papers_path = papers_path
        self.abstracts_path = abstracts_path
        self.papers = None
        self.abstracts = None

    def begin(self):
        self.papers = open(self.papers_path, "r")
        self.abstracts = open(self.abstracts_path, "r")

    def end(self):
        self.papers.close()
        self.abstracts.close()

    def __call__(self, proc: Tuple[int, Optional[int], Optional[List[int]]]) -> Tuple[int, str]:
        self.papers.seek(proc[0])
        paper = json_loads(self.papers.readline())
        abstract = None
        if proc[1] is not None:
            self.abstracts.seek(proc[1])
            abstract = json_loads(self.abstracts.readline())["abstract"]

        paper["abstract"] = abstract
        paper["citing"] = [] if proc[2] is None else proc[2]

        return int(paper["corpusid"]), json_dumps(paper)


def create_s2orc_metadata(args: argparse.Namespace):
    """
    Creates metadata for S2ORC dataset.

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    workers = max(1, workers)

    logging.debug("Loading papers")
    papers = MapAccessFile(args.papers, args.papers + ".index", int)

    logging.debug("Loading abstracts")
    abstracts = MapAccessFile(args.abstracts, args.abstracts + ".index", int)

    logging.debug("Loading citation graph")

    def load_integer_keys_hook(dct):
        return {int(k): tuple(v) for k, v in dct.items()}

    with open(args.citation_graph) as f:
        citation_graph = json.load(f, object_hook=load_integer_keys_hook)

    logging.debug("Creating metadata")
    with FunctorPool([S2ORCMetadataCreator(args.papers, args.abstracts) for _ in range(workers)],
                     results_queue_maxsize=1.0) as pool:
        with open(args.result, "w") as f, open(args.result + ".index", "w") as index_f:
            def generator():
                for p_id, p_offset in papers.mapping.items():
                    yield p_offset, abstracts.mapping.get(p_id), citation_graph.get(p_id)

            index_writer = csv.DictWriter(index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
            index_writer.writeheader()

            for res_id, res in tqdm(pool.imap_unordered(generator(), chunk_size=10000),
                                    total=len(papers.mapping), desc="Creating metadata"):
                index_writer.writerow({"key": res_id, "file_line_offset": f.tell()})
                print(res, file=f)


class S2ORCRecordCreator(FunctorWorker):
    """
    Creates records for S2ORC dataset.
    """

    def __init__(self, metadata_path: str):
        super().__init__()
        self.metadata_path = metadata_path
        self.metadata = None

    def begin(self):
        self.metadata = open(self.metadata_path, "r")

    def end(self):
        self.metadata.close()

    def __call__(self, offset: int) -> Tuple[int, Optional[str]]:
        self.metadata.seek(offset)
        metadata = json_loads(self.metadata.readline())
        if metadata["title"] is None or metadata["authors"] is None or len(metadata["authors"]) == 0:
            return int(metadata["corpusid"]), None

        return int(metadata["corpusid"]), json_dumps({
            "id": int(metadata["corpusid"]),
            "title": metadata["title"],
            "year": metadata["year"],
            "authors": [a["name"] for a in metadata["authors"]]
        })


def create_s2orc_records(args: argparse.Namespace):
    """
    Creates record file for S2ORC dataset.

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    workers = max(1, workers)

    logging.debug("Loading metadata")
    papers = MapAccessFile(args.metadata, args.metadata + ".index", int)

    with FunctorPool([S2ORCRecordCreator(args.metadata) for _ in range(workers)], results_queue_maxsize=1.0) as pool:
        with open(args.result, "w") as f, open(args.result + ".index", "w") as index_f:
            index_writer = csv.DictWriter(index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
            index_writer.writeheader()

            for res_id, res in tqdm(pool.imap_unordered(papers.mapping.values(), chunk_size=10000),
                                    total=len(papers.mapping), desc="Creating records"):
                if res is None:
                    continue
                index_writer.writerow({"key": res_id, "file_line_offset": f.tell()})
                print(res, file=f)


def convert_s2orc(args: argparse.Namespace):
    """
    Converts S2ORC dataset to OAReviews format.

    :param args: user arguments
    """

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    if args.gpu:
        spacy.require_gpu()
    else:
        spacy.require_cpu()

    with S2ORCDocumentDataset(args.metadata, args.s2orc, workers) as dataset:
        convert_dataset(dataset, args.result, True, args.from_i, args.to_i)


def convert_s2orc_old(args: argparse.Namespace):
    """
    Converts old version of S2ORC dataset to OAReviews format.

    :param args: user arguments
    """

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    if args.gpu:
        spacy.require_gpu()
    else:
        spacy.require_cpu()

    with OldS2ORCDocumentDataset(args.original, workers) as dataset:
        convert_dataset(dataset, args.result, True)


def convert_core(args: argparse.Namespace):
    """
    Converts CORE dataset to OAReviews format.

    :param args: user arguments
    """
    mp_context = multiprocessing.get_context('spawn')  # we need that because of CUDA
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    spacy.require_cpu()

    manager = None
    allow_gpus = None
    if workers > 0:
        manager = PapersListManager(ctx=CONTEXT_SPAWN)
        allow_gpus = [[i for i in range(faiss.get_num_gpus())]]

    with nullcontext() if manager is None else manager:
        mag_list = None
        if args.mag:
            mag_list = MAGPapersList.from_file(args.mag, workers=workers,
                                               manager=[manager] if manager is not None else None,
                                               allow_gpus=allow_gpus,
                                               use_gpus=args.gpu)
            if isinstance(mag_list, list):
                mag_list = mag_list[0]
            mag_list.return_self_on_enter(False)  # otherwise may cause problems in mult. proc. mode

        s2orc_list = None
        if args.s2orc:
            s2orc_list = PapersList.from_file(args.s2orc,
                                              workers=workers,
                                              manager=[manager] if manager is not None else None,
                                              allow_gpus=allow_gpus,
                                              record_type=PapersListRecordWithId,
                                              use_gpus=args.gpu)
            if isinstance(s2orc_list, list):
                s2orc_list = s2orc_list[0]
            s2orc_list.return_self_on_enter(False)

        papers_list = COREPapersList.from_dir(args.original, workers=workers, manager=manager)
        with nullcontext() if mag_list is None else mag_list, \
                nullcontext() if s2orc_list is None else s2orc_list:
            dataset = COREDocumentDataset(papers_list, mag_list, s2orc_list, batch_size=args.batch, workers=workers)

            # lets save memory, as we no longer need search abilities, because we have precomputed the mappings
            if mag_list is not None:
                mag_list.stub()

            all_paths = set(str(p) for p in Path(args.original).rglob("*.xml"))
            dataset_paths = dataset.paths
            for p in all_paths.difference(dataset_paths):
                print(f"The document on path {p} was omitted (parsing error or missing data field).")

            written_ids = convert_dataset(dataset, args.result, False)
            written_ids.add(len(dataset))  # ensures that the last ones will be printed
            before = 0
            for i in written_ids:
                for x in range(before, i):
                    print(f"The document on path {dataset_paths[x]} was omitted (parsing error or missing data field).")
                before = i + 1


class Flat2MultiTransform:
    """
    Functor that makes flat 2 multi hierarchy transformations.
    """

    def __init__(self, ids: Optional[AbstractSet[int]] = None):
        """
        :param ids: ids of documents that are in the resulting dataset
            is used to filter citations (filtered only when the conversion to multi-lvl hierarchy is performed
            successfully)
        """
        self.ids = ids

        self.cnt = 0
        self.wait_time = 0
        self.convert_time = 0
        self.filter_time = 0
        self.str_convert_time = 0
        self.last_time = None
        self.profile = False

    def __call__(self, document: Document) -> Tuple[int, Optional[str]]:
        """
        conversion of document to multi-lvl hierarchy

        :param document: document for transformation
        :return:
            the document id
            the json document representation or None if the transformation failed
        """

        if self.last_time is not None:
            self.wait_time += time.time() - self.last_time
        self.cnt += 1
        try:
            start = time.time()

            success = document.hierarchy.flat_2_multi()
            self.convert_time += time.time() - start

            start = time.time()
            if not success:
                return document.id, None

            if self.ids is not None:
                document.filter_citations(self.ids)

            self.filter_time += time.time() - start

            start = time.time()
            doc_str = str(document)
            self.str_convert_time += time.time() - start

            if self.profile and self.cnt % 100 == 0:
                print(f"{os.getpid()} Flat 2 multi transform: {self.cnt} documents processed, "
                      f"{self.wait_time / self.cnt:.2f}s waiting, {self.convert_time / self.cnt:.2f}s converting, "
                      f"{self.filter_time / self.cnt:.2f}s filtering, {self.str_convert_time / self.cnt:.2f}s str converting",
                      flush=True, file=sys.stderr)

            self.last_time = time.time()
            return document.id, doc_str
        except Exception as e:
            print(e)
            traceback.print_exception(type(e), e, e.__traceback__)


def multi_lvl_hier(args: argparse.Namespace):
    """
    Converts OAPapers hierarchy from flat to multi-level.

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    successful_ids = []

    with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f, \
            OADataset(args.original, args.original + ".index", workers=workers) as dataset, \
            open(args.original, "r") as dataset_file:

        # we want to make sure that references papers are really in dataset
        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        dataset.chunk_size = 100

        from_index = args.from_index
        to_index = len(dataset) if args.to_index is None else args.to_index

        if from_index > to_index:
            raise ValueError("The from index must be smaller than the to index.")

        if from_index < 0:
            raise ValueError("The from index must be non-negative.")

        if to_index > len(dataset):
            raise ValueError(
                f"The to index must be smaller than the number of documents ({len(dataset)}) in the dataset.")

        if args.discard:
            all_ids = SortedSet(get_ids_of_filter_passing_documents(dataset, CouldEstablishMultLvlHierFilter()))
            dataset.transform = Flat2MultiTransform(all_ids)

            for d_id, document in tqdm(dataset.iter_range(from_index, to_index), desc="Converting", unit="doc",
                                       total=to_index - from_index):
                if document is not None:
                    successful_ids.append(d_id)
                    index_writer.writerow({"key": d_id, "file_line_offset": res_f.tell()})
                    print(document, file=res_f)
                else:
                    print(f"discarding document with id {d_id}")
        else:
            dataset.transform = Flat2MultiTransform()

            for i, (d_id, document) in tqdm(
                    zip(range(len(dataset))[from_index:to_index], dataset.iter_range(from_index, to_index)),
                    desc="Converting", unit="doc", total=to_index - from_index):

                f_offset = dataset.indices_2_offsets[i]
                if document is not None:
                    successful_ids.append(d_id)
                else:
                    print(f"can't obtain hierarchy in document with id {d_id}")
                    dataset_file.seek(f_offset)
                    document = dataset_file.readline().rstrip()

                index_writer.writerow({"key": d_id, "file_line_offset": res_f.tell()})
                print(document, file=res_f)

    if args.multi:
        with open(args.multi, "w") as f:
            print(json_dumps(successful_ids), file=f)


class FilterByIDsTransform:
    """
    Functor that filters documents and references by their ids.
    """

    def __init__(self, ids: Optional[AbstractSet[int]] = None):
        """
        :param ids: ids of documents that are in the resulting dataset
        """
        self.ids = ids

    def __call__(self, document: Document) -> Tuple[int, Optional[str]]:
        """
        filter document by ids

        :param document: document for transformation
        :return:
            the document id
            the json document representation or None if the transformation failed
        """
        try:
            document.filter_citations(self.ids)
            return document.id, str(document)
        except Exception as e:
            print(e)
            traceback.print_exception(type(e), e, e.__traceback__)


def discard_documents(args: argparse.Namespace):
    """
    Discards documents according to given ids. It will only keep documents that have ids in the given file.

    :param args: user arguments
    """

    with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f, \
            OADataset(args.original, args.original + ".index", workers=args.workers,
                      hierarchy_as_dict=True) as dataset:

        # we want to make sure that references papers are really in dataset
        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        dataset.chunk_size = 100

        with open(args.ids, "r") as ids_f:
            ids = []
            for line in ids_f:
                ids.extend(json_loads(line))

            ids = SortedSet(ids)

        dataset.transform = FilterByIDsTransform(ids)

        for i, (d_id, document) in tqdm(enumerate(dataset), desc="Discarding", unit="doc", total=len(dataset)):
            if d_id in ids:
                index_writer.writerow({"key": d_id, "file_line_offset": res_f.tell()})
                print(document, file=res_f)
            else:
                print(f"discarding document with id {d_id}")


class PruneHier:
    """
    Functor that prunes the hierarchy.
    """
    regex_latin = re.compile(r"^LATIN (CAPITAL|SMALL) LETTER .$", re.IGNORECASE | re.UNICODE)

    def __init__(self, ids: Optional[AbstractSet[int]], empty_headlines: bool, plain_latin: Optional[bool],
                 no_text: bool, named_text_blocks: bool):
        """
        :param ids: ids of documents that are in the resulting dataset
            is used to filter citations (filtered only when the conversion to multi-lvl hierarchy is performed
            successfully)

            If None then the ids are not filtered.
        :param empty_headlines: Removes all sections with empty headline.
        :param plain_latin: Minimal coverage of plain latin characters in a section headline.
        :param no_text: Removes all sections without text content (in whole sub-hierarchy).
        :param named_text_blocks: Removes certain named text blocks:
            figures
        """
        self.ids = ids
        self.empty_headlines = empty_headlines
        self.plain_latin = plain_latin
        self.no_text = no_text
        self.named_text_blocks = named_text_blocks

    def __call__(self, document: Document) -> Tuple[int, Optional[str]]:
        """
        Prunes the document hierarchy.

        :param document: document to transform
        :return:
            the document id
            the json document representation or None when it should be discarded
        """
        if self.ids is not None and document.id not in self.ids:
            return document.id, None

        if self.empty_headlines:
            document.hierarchy.prune_empty_headlines_nodes()
        if self.plain_latin:
            document.hierarchy.prune_according_to_name_assigned_to_chars_in_headline(self.regex_latin,
                                                                                     self.plain_latin)
        if self.no_text:
            document.hierarchy.prune_nodes_without_text_content()

        if self.named_text_blocks:
            document.hierarchy.prune_named_text_blocks({"figure"})

        if self.ids is not None:
            document.filter_citations(self.ids)  # make sure that the references are really in dataset
        return document.id, str(document)


def pruning_hier(args: argparse.Namespace):
    """
    Prunes OAPapers hierarchy. Will remove documents that will become empty.

    :param args: user arguments
    """

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f, \
            OADataset(args.original, args.original + ".index", workers=workers) as dataset:

        from_index = args.from_index
        to_index = len(dataset) if args.to_index is None else args.to_index

        if from_index > to_index:
            raise ValueError("The from index must be smaller than the to index.")

        if from_index < 0:
            raise ValueError("The from index must be non-negative.")

        if to_index > len(dataset):
            raise ValueError(
                f"The to index must be smaller than the number of documents ({len(dataset)}) in the dataset.")

        # we want to make sure that references papers are really in dataset
        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        all_ids = None  # no filtering by default

        if args.filter_ids:
            all_ids = SortedSet(get_ids_of_filter_passing_documents(dataset,
                                                                    IsValidAfterHierPruneFilter(
                                                                        args.empty_headlines,
                                                                        args.no_text,
                                                                        (PruneHier.regex_latin,
                                                                         args.plain_latin) if args.plain_latin else None,
                                                                        copy_doc=False
                                                                    )))

            if workers > 0:
                all_ids.values = multiprocessing.Array(ctypes.c_int64, all_ids.values, lock=False)

        dataset.transform = PruneHier(all_ids, args.empty_headlines, args.plain_latin, args.no_text,
                                      args.named_text_blocks)

        # seems that multiprocessing is not much helpful in next steps, good point to look at it at the future
        for d_id, document_json in tqdm(dataset.iter_range(from_index, to_index), desc="Pruning",
                                        total=to_index - from_index, unit="doc"):
            if document_json is not None:
                index_writer.writerow({"key": d_id, "file_line_offset": res_f.tell()})
                print(document_json, file=res_f)


def get_last_line(p: str) -> str:
    """
    Get last line from file on given path.

    :param p: path to file
    :return: last line
    """

    with open(p, 'rb') as f:
        try:
            # I'm not interested in the last character, because when the file finishes with \n I don't want to
            # return empty string. If it is other character then I don't mind.
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            # single line file
            f.seek(0)
        return f.readline().decode()


class TranslateIdsTransform:
    """
    Functor that translates ids of documents.
    """

    def __init__(self, translation: Dict[int, int], return_original_id: bool = False, return_doc: bool = False):
        """
        :param translation: mapping from old id to new id
        :param return_original_id: if True then the original id is returned
        :param return_doc: if True then the document object is returned
        """
        self.translation = translation
        self.return_original_id = return_original_id
        self.return_doc = return_doc

    def __call__(self, document: Document) -> Union[
        Tuple[int, Optional[str]], Tuple[int, int, Optional[str]], Document]:
        """
        Translates the document id.

        :param document: document to transform
        :return:
            the document id
            the original document id (if return_original_id is True)
            the json document representation or None when it should be discarded
        """
        orig_id = document.id
        document.translate_ids(self.translation)

        if self.return_doc:
            return document

        if self.return_original_id:
            return document.id, orig_id, str(document)
        return document.id, str(document)


def extension_subset(args: argparse.Namespace):
    """
    Selects subset of documents from OAPapers dataset that is not present in another dataset.

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    managers = None
    allow_gpus = None
    if workers > 0:
        managers = ContextList(
            [PapersListManager(ctx=CONTEXT_SPAWN) for _ in range(max(1, min(workers, faiss.get_num_gpus())))])
        allow_gpus = [[i] for i in range(len(managers))]

    with nullcontext() if managers is None else managers:
        first_papers_list = PapersList.from_file(args.first,
                                                 match_threshold=args.match_threshold,
                                                 max_year_diff=args.max_year_diff,
                                                 workers=workers,
                                                 manager=managers,
                                                 allow_gpus=allow_gpus)
        second_papers_list = PapersList.from_file(args.second,
                                                  match_threshold=args.match_threshold,
                                                  max_year_diff=args.max_year_diff,
                                                  workers=workers)

        if not isinstance(first_papers_list, list):
            first_papers_list = [first_papers_list]

        with ContextList(first_papers_list), second_papers_list:
            second_2_first = second_papers_list.to_other_mapping(first_papers_list, batch_size=args.batch,
                                                                 reverse=False)

        del second_papers_list
        del first_papers_list

    with OADataset(args.second, args.second + ".index", workers=workers) as second_dataset:

        with open(args.result, "w+") as res_f, open(args.result + ".index", "w") as res_index_f:
            index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
            index_writer.writeheader()

            for i, in_first in enumerate(tqdm(second_2_first, desc="Collecting missing documents")):
                if in_first is None:
                    index_writer.writerow({
                        "key": second_dataset.indices_2_id[i],
                        "file_line_offset": res_f.tell()
                    })

                    second_dataset.file.seek(second_dataset.indices_2_offsets[i])
                    line = second_dataset.file.readline()
                    res_f.write(line)


def extend_ids(args: argparse.Namespace):
    """
    Changes ids in second dataset to ids that will be subsequent to the last id in the first dataset
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    with open(args.first + ".index", "r") as f:
        def iterator():
            for record in tqdm(csv.reader(f, delimiter="\t"), desc="Searching max id"):
                try:
                    yield int(record[0])
                except ValueError:
                    # this one is probably the headline
                    pass

        first_empty_id = max(i for i in iterator()) + 1
    id_cnt = first_empty_id
    logging.log(logging.INFO, f"First new document will have id {first_empty_id}")

    with OADataset(args.second, args.second + ".index", workers=workers) as second_dataset:
        ids_trans = {}
        for id_in_second in tqdm(second_dataset.indices_2_id,
                                 desc="Creating id translation mapping for extension dataset"):
            ids_trans[id_in_second] = id_cnt
            id_cnt += 1

        ids_trans = SortedMap(ids_trans)

        if workers > 0:
            ids_trans.keys_storage = multiprocessing.Array(ctypes.c_int64, ids_trans.keys_storage, lock=False)
            ids_trans.values_storage = multiprocessing.Array(ctypes.c_int64, ids_trans.values_storage,
                                                             lock=False)

        with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f:
            index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
            index_writer.writeheader()
            second_dataset.transform = TranslateIdsTransform(ids_trans)

            for doc_id, doc_str in tqdm(second_dataset, desc="Translating ids"):
                index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
                print(doc_str, file=res_f)


def extend(args: argparse.Namespace):
    """
    Merges two datasets in OAReviews document format.

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    managers = None
    allow_gpus = None
    if workers > 0:
        managers = ContextList(
            [PapersListManager(ctx=CONTEXT_SPAWN) for _ in range(max(1, min(workers, faiss.get_num_gpus())))])
        allow_gpus = [[i] for i in range(len(managers))]

    shared_records_list_manager = SharedMemoryManager() if managers is not None else None

    with nullcontext() if managers is None else managers, \
            nullcontext() if shared_records_list_manager is None else shared_records_list_manager:

        first_papers_list = PapersList.from_file(args.first,
                                                 match_threshold=args.match_threshold,
                                                 max_year_diff=args.max_year_diff,
                                                 workers=workers,
                                                 manager=managers,
                                                 allow_gpus=allow_gpus,
                                                 shared_list_for_records=shared_records_list_manager)

        if not isinstance(first_papers_list, list):
            first_papers_list = [first_papers_list]

        with ContextList(first_papers_list):

            with open(args.first + ".index", "r") as f:
                def iterator():
                    for record in tqdm(csv.reader(f, delimiter="\t"), desc="Searching max id"):
                        try:
                            yield int(record[0])
                        except ValueError:
                            # this one is probably the headline
                            pass

                first_empty_id = max(i for i in iterator()) + 1
            id_cnt = first_empty_id
            logging.log(logging.INFO, f"First new document will have id {first_empty_id}")

            first_dataset = OADataset(args.first, args.first + ".index", workers=workers)
            second_dataset = OADataset(args.second, args.second + ".index", workers=workers)

            ids_trans = {}
            for id_in_second in tqdm(second_dataset.indices_2_id,
                                     desc="Creating id translation mapping for extension dataset"):
                ids_trans[id_in_second] = id_cnt
                id_cnt += 1

            ids_trans = SortedMap(ids_trans)

            if workers > 0:
                ids_trans.keys_storage = multiprocessing.Array(ctypes.c_int64, ids_trans.keys_storage, lock=False)
                ids_trans.values_storage = multiprocessing.Array(ctypes.c_int64, ids_trans.values_storage,
                                                                 lock=False)

            with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f:
                index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
                index_writer.writeheader()
                second_dataset.transform = TranslateIdsTransform(ids_trans, return_doc=True)

                extension_paper_list = PapersList.from_file(args.second,
                                                            workers=workers,
                                                            manager=managers, allow_gpus=allow_gpus,
                                                            match_threshold=args.match_threshold,
                                                            max_year_diff=args.max_year_diff,
                                                            load_records_after_init=True)
                if isinstance(extension_paper_list, PapersList):
                    extension_paper_list = [extension_paper_list]

                # firstly we will print the first dataset and resolve bibliography in it

                first_dataset.hierarchy_as_dict = True
                second_dataset.hierarchy_as_dict = True
                indices_2_ids = [ids_trans[i] for i in second_dataset.indices_2_id]
                identified_citations = identify_bibliography_for_dataset(first_dataset, extension_paper_list,
                                                                         indices_2_id=indices_2_ids,
                                                                         res_file=res_f, index_writer=index_writer,
                                                                         workers=workers,
                                                                         batch_size=args.docs,
                                                                         batch_size_search=args.batch)

                print(f"identified citations in first dataset: {identified_citations}", file=sys.stderr, flush=True)

                if isinstance(first_papers_list, PapersList):
                    first_papers_list = [first_papers_list]

                identified_citations = identify_bibliography_for_dataset(second_dataset, first_papers_list,
                                                                         indices_2_id=first_dataset.indices_2_id,
                                                                         res_file=res_f, index_writer=index_writer,
                                                                         workers=workers,
                                                                         batch_size=args.docs,
                                                                         batch_size_search=args.batch)

                print(f"identified citations in for extension subset of second dataset: {identified_citations}",
                      file=sys.stderr, flush=True)


def extend_mag_with_core(args: argparse.Namespace):
    """
    Extends mag full record with CORE data.

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(workers) if workers > 0 else nullcontext() as pool:
        core_list = COREPapersList.from_dir(args.core, workers=workers)
        mag_list = MAGPapersList.from_file(args.mag, workers=workers)

        with mag_list:
            if args.identified_references and os.path.isfile(args.identified_references):
                core_doc_2_mag_references = []
                with open(args.identified_references, "r") as f:
                    for line in f:
                        core_doc_2_mag_references.append([int(x) for x in line.split()])
            else:
                core_doc_2_mag_references = core_list.identify_references(mag_list, pool, batch_size=args.batch)
                with open(args.identified_references, "w") as f:
                    for references in core_doc_2_mag_references:
                        print(" ".join(str(r) for r in references), file=f)

            core_2_mag_index = core_list.to_other_mapping(mag_list, args.batch, False)
            next_mag_id = 0 if len(mag_list.ids) == 0 else max(mag_list.ids) + 1

            # we do not want to alter index on every new record so it will be firstly stored and then it will be
            # added into mag list
            add_records = []

            for i, mag_index in (pb := tqdm(enumerate(core_2_mag_index),
                                            desc=f"Adding pass of CORE ({len(add_records)} for addition)",
                                            total=len(core_2_mag_index))):
                if mag_index is None:
                    core_record = core_list[i]
                    add_records.append(MAGPapersListRecord(
                        id=i,  # temporally set it to core, because it is used in disambiguation step
                        title=core_record.title,
                        year=core_record.year,
                        authors=core_record.authors,
                        references=tuple(mag_list.ids[r_index] for r_index in core_doc_2_mag_references[i]),
                        fields=[],
                        doi=None,
                        journal=None
                    ))
                    pb.set_description(f"Adding pass of CORE ({len(add_records)} for addition)")

            if len(add_records):
                # not sure whether the core dataset is deduplicated so let's make sure that we are not adding
                # two same papers to mag and we choose only the first one
                deduplicated_list = PapersList([])
                for d_i, record in enumerate(pb := tqdm(add_records, desc=f"Disambiguation (0 uniques)",
                                                        total=len(add_records))):
                    record: MAGPapersListRecord
                    search_res = deduplicated_list.batch_search([record])[0]
                    if search_res is None:
                        new_mag_id = next_mag_id + d_i
                        new_mag_index = len(mag_list) + len(deduplicated_list)
                        deduplicated_list.add([record])
                        pb.set_description(f"Disambiguation ({len(deduplicated_list)} uniques)")
                    else:
                        same_record: MAGPapersListRecord = deduplicated_list[search_res]
                        new_mag_index = len(mag_list) + search_res
                        new_mag_id = same_record.id

                    core_2_mag_index[record.id] = new_mag_index
                    record.id = new_mag_id

                logging.log(logging.INFO, "updating references with new MAG records")
                # update references with new mag records
                core_doc_2_new_mag_references = core_list.identify_references(deduplicated_list, pool,
                                                                              batch_size=args.batch)

                for i, new_indices in enumerate(core_doc_2_new_mag_references):
                    core_doc_2_mag_references[i].extend(len(mag_list) + x for x in new_indices)

                with open(args.identified_references, "w") as f:
                    for references in core_doc_2_mag_references:
                        print(" ".join(str(r) for r in references), file=f)

                mag_list.add(list(deduplicated_list), reset=True)

            updated_cnt = 0

            for i, mag_index in (pb := tqdm(enumerate(core_2_mag_index),
                                            desc=f"Update pass of CORE ({updated_cnt} updated)",
                                            total=len(core_2_mag_index))):
                mag_record = mag_list[mag_index]
                reference_ids = set(mag_list.ids[r_index] for r_index in core_doc_2_mag_references[i])
                known_references = set(mag_record.references)
                if any(rid not in known_references for rid in reference_ids):
                    mag_record.references = tuple(known_references | reference_ids)
                    mag_list[mag_index] = mag_record
                    updated_cnt += 1
                    pb.set_description(f"Update pass of CORE ({updated_cnt} updated)")

            with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f:
                index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
                index_writer.writeheader()

                for record in tqdm(mag_list, desc="Writing new MAG"):
                    index_writer.writerow({"key": record.id, "file_line_offset": res_f.tell()})
                    res = {
                        "PaperId": record.id,
                        "OriginalTitle": record.title,
                        "Year": record.year,
                        "Authors": record.authors,
                        "References": record.references,
                        "Fields": record.fields,
                        "Doi": record.doi,
                        "Journal": record.journal
                    }
                    print(json_dumps(res), file=res_f)


class GivenFunctorWorker(FunctorWorker):
    """
    It will use given functor.
    """

    def __init__(self, functor: Callable[[Any], Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.functor = functor

    def begin(self):
        if hasattr(self.functor, "begin"):
            self.functor.begin()

    def end(self):
        if hasattr(self.functor, "end"):
            self.functor.end()

    def __call__(self, data):
        return self.functor(data)


class MagEnhanceTransform:
    """
    Enhances papers with MAG information.
    """

    SHARED_MEMORY_BLOCK_NAME_FOR_PAPERS_2_MAG = "mag_enhance_transform_papers_2_mag"
    SHARED_MEMORY_BLOCK_NAME_FOR_MAG_2_PAPERS_ID = "mag_enhance_transform_mag_2_papers_id"

    def __init__(self, mag: MapAccessFile, mag_indices_2_ids: Sequence[int], mag_ids_2_indices: Mapping,
                 mapping_file: str, dataset: str,
                 just_references: bool = False,
                 match_threshold: float = 0.8, workers: int = 1):
        """
        :param map: MAG papers
        :param mag_indices_2_ids: list of MAG indices that correspond to papers
        :param mag_ids_2_indices: mapping from MAG ids to indices
        :param mapping_file: file with mapping from papers to MAG
        :param dataset: path to dataset
        positions that correspond to MAG indices
        :param just_references: Whether just the citations should be updated.
        :param match_threshold: Score threshold for matching. All above or equal are match.
        :param workers: World size, how many transforms will be used in parallel. The 1 means no parallelism.
        """
        self.mag = mag
        self.mag_indices_2_ids = mag_indices_2_ids
        self.mag_ids_2_indices = mag_ids_2_indices
        self.mapping_file = mapping_file
        self.dataset = dataset
        self._dataset_file_descriptor = None
        self.papers_2_mag = None
        self.mag_2_papers_id = None
        self.papers_2_mag_sm = None
        self.mag_2_papers_id_sm = None
        self.just_references = just_references
        self.match_threshold = match_threshold

        self.workers = workers
        self.init_barrier = multiprocessing.Barrier(workers)
        self.init_completed = multiprocessing.Value("i", -1)  # will contain number of papers
        self.sm_required_cnt = multiprocessing.Value("i", 0)  # will contain number of processes using shared memory
        self.main_process = multiprocessing.Value("i", -1)

    @staticmethod
    def create_shared_memory(name, size: int) -> SharedMemory:
        """
        Creates shared memory block with given name and size.
        If block already exists, it will be deleted and recreated.

        :param name: name of shared memory block
        :param size: size of shared memory block
        :return: SharedMemory object
        """

        try:
            return SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            sm = SharedMemory(name=name, create=False)

            sm.close()
            sm.unlink()
            return SharedMemory(name=name, create=True, size=size)

    def begin(self):
        # called in multiprocessing context when a worker is started
        random.seed(time.time() + os.getpid())
        if self.workers == 1:
            logging.log(logging.INFO, "Loading OA2MAG mapping")
            self.papers_2_mag, self.mag_2_papers_id = self._load_mappings()
        else:
            with self.main_process.get_lock():
                if self.main_process.value == -1:
                    self.main_process.value = os.getpid()

            with self.sm_required_cnt.get_lock():
                self.sm_required_cnt.value += 1

            if self.main_process.value != os.getpid():
                self.init_barrier.wait()

                if self.init_completed.value >= 0:
                    self.papers_2_mag_sm = SharedMemory(name=self.SHARED_MEMORY_BLOCK_NAME_FOR_PAPERS_2_MAG)
                    self.mag_2_papers_id_sm = SharedMemory(name=self.SHARED_MEMORY_BLOCK_NAME_FOR_MAG_2_PAPERS_ID)

                    self.papers_2_mag = np.ndarray(
                        shape=(self.init_completed.value,), dtype=np.int32, buffer=self.papers_2_mag_sm.buf
                    )
                    self.mag_2_papers_id = np.ndarray(
                        shape=(len(self.mag),), dtype=np.int32, buffer=self.mag_2_papers_id_sm.buf
                    )

            else:
                papers_2_mag, mag_2_papers_id = self._load_mappings()
                self.papers_2_mag = np.array(papers_2_mag, dtype=np.int32)
                self.mag_2_papers_id = np.array(mag_2_papers_id, dtype=np.int32)

                self.papers_2_mag_sm = self.create_shared_memory(
                    name=self.SHARED_MEMORY_BLOCK_NAME_FOR_PAPERS_2_MAG,
                    size=self.papers_2_mag.nbytes
                )

                sm = np.ndarray(self.papers_2_mag.shape, dtype=self.papers_2_mag.dtype, buffer=self.papers_2_mag_sm.buf)
                sm[:] = self.papers_2_mag[:]
                self.papers_2_mag = sm

                self.mag_2_papers_id_sm = self.create_shared_memory(
                    name=self.SHARED_MEMORY_BLOCK_NAME_FOR_MAG_2_PAPERS_ID,
                    size=self.mag_2_papers_id.nbytes
                )

                sm = np.ndarray(self.mag_2_papers_id.shape, dtype=self.mag_2_papers_id.dtype,
                                buffer=self.mag_2_papers_id_sm.buf)
                sm[:] = self.mag_2_papers_id[:]
                self.mag_2_papers_id = sm

                self.init_completed.value = self.papers_2_mag.shape[0]
                self.init_barrier.wait()

        self._dataset_file_descriptor = open(self.dataset, "r")

    def _load_mappings(self) -> Tuple[np.array, np.array]:
        """
        Loads mappings from file.

        :return: Tuple of two arrays: papers_2_mag and mag_2_papers_id
        """
        papers_2_mag, mag_2_papers = [], [-1] * len(self.mag)

        with open(self.mapping_file, "r") as f:
            for line in f:
                paper_index, mag_index = len(papers_2_mag), -1 if line.startswith("None") else int(line)
                papers_2_mag.append(mag_index)
                if mag_index != -1:
                    mag_2_papers[mag_index] = paper_index

        with OADataset(self.dataset, self.dataset + ".index") as ds:
            mag_2_papers_id = [
                -1 if i == -1 else ds.indices_2_id[i] for i in tqdm(mag_2_papers, desc="Index to id")
            ]

        return np.array(papers_2_mag, dtype=np.int32), np.array(mag_2_papers_id, dtype=np.int32)

    def end(self):

        if self.workers != 1:
            self.papers_2_mag_sm.close()
            self.mag_2_papers_id_sm.close()

            with self.sm_required_cnt.get_lock():
                self.sm_required_cnt.value -= 1
                if self.sm_required_cnt.value == 0:
                    self.papers_2_mag_sm.unlink()
                    self.mag_2_papers_id_sm.unlink()

        self._dataset_file_descriptor.close()

    def __call__(self, process: Tuple[int, int]) -> Tuple[int, str]:
        """
        Enhances document with MAG information.

        :param process:
            file offset of the document
            document index
        :return:
            document id
            document string representation
        """

        # load document
        offset, i = process
        self._dataset_file_descriptor.seek(offset)
        doc = Document.from_dict(json_loads(self._dataset_file_descriptor.readline()), hierarchy_as_dict=True)

        doc.deduplicate_authors()
        mag_index = int(self.papers_2_mag[i])
        if mag_index != -1:
            # let's enhance it
            mag_line = self.mag[self.mag_indices_2_ids[mag_index]]
            mag_record = MAGPapersListRecord.load(mag_line)

            ref_indices = []
            ref_records = []
            for ref_id in mag_record.references:
                if ref_id is not None:
                    try:
                        ref_indices.append(self.mag_ids_2_indices[ref_id])
                    except KeyError:
                        continue
                    ref_records.append(MAGPapersListRecord.load(self.mag[ref_id]))

            if not self.just_references:
                known_authors = set(frozenset(a) for a in normalize_authors(doc.authors))
                for a, a_norm in zip(mag_record.authors, normalize_authors(mag_record.authors)):
                    a_norm = frozenset(a_norm)
                    if a_norm not in known_authors:
                        doc.authors.append(a)

                if doc.year is None:
                    doc.year = mag_record.year

                doc.mag_id = mag_record.id

                if mag_record.doi is not None:
                    doc.doi = mag_record.doi

                if mag_record.fields is not None:
                    if doc.fields_of_study is None:
                        doc.fields_of_study = mag_record.fields
                    else:
                        field_names = set(f if isinstance(f, str) else f[0] for f in doc.fields_of_study)
                        for f in mag_record.fields:
                            field_name = f if isinstance(f, str) else f[0]
                            if field_name not in field_names:
                                doc.fields_of_study.append(f)

            if len(ref_indices) > 0:

                for mag_ind in ref_indices:
                    oa_id = int(self.mag_2_papers_id[mag_ind])
                    if oa_id == -1:
                        # missing mag paper in oa
                        continue
                    if oa_id not in doc.citations:
                        doc.citations.append(oa_id)
                doc.citations = sorted(doc.citations)

                # update bibliography
                bib = Bibliography(doc.bibliography, authors_match_threshold=sys.float_info.min)

                for ref_i, ref_record in zip(ref_indices, ref_records):
                    bib_paper_id = int(self.mag_2_papers_id[ref_i])
                    if bib_paper_id == -1:
                        bib_paper_id = None
                    try:
                        init_norm_authors, norm_authors = initial_and_normalized_authors(ref_record.authors)
                        bib_i = bib.index_prenorm(set(normalize_and_tokenize_string(ref_record.title)), norm_authors,
                                                  init_norm_authors, ref_record.year)
                        bib_record = doc.bibliography[bib_i]

                        if bib_record.id is None:
                            bib_record.id = bib_paper_id

                        if bib_record.year is None:
                            bib_record.year = ref_record.year

                        if len(bib_record.authors) == 0:
                            bib_record.authors = tuple(ref_record.authors)

                    except ValueError:
                        # not found
                        entry = BibEntry(bib_paper_id, ref_record.title, ref_record.year, tuple(ref_record.authors))
                        doc.bibliography.append(entry)
                        bib.append(entry)

        return doc.id, str(doc)


class ContextList(Sequence):
    """
    Allows to use with statement on list of objects.
    """

    def __init__(self, s: Sequence):
        self.s = s

    def __getitem__(self, index: int):
        return self.s[index]

    def __len__(self) -> int:
        return len(self.s)

    def __enter__(self):
        for x in self.s:
            x.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for x in self.s:
            x.__exit__(exc_type, exc_val, exc_tb)


CONTEXT_SPAWN = multiprocessing.get_context("spawn")


class MagIdReader(FunctorWorker, CONTEXT_SPAWN.Process):
    def __init__(self, file_path: str):
        """
        :param file_path: path to file
        """

        super().__init__()
        self._file_path = file_path
        self._file = None

    def begin(self):
        self._file = open(self._file_path, "r")

    def end(self):
        self._file.close()

    def __call__(self, line_offset: int) -> int:
        self._file.seek(line_offset)
        record = json_loads(self._file.readline())

        return record["mag_id"]


def oa_papers_2_mag_mapping(oa_papers: str, mags: List[MAGPapersList], batch: int, workers: int) \
        -> List[Optional[int]]:
    """
    Creates mapping from oa papers to mag papers.

    :param oa_papers: path to oa papers
    :param mags: mag lists
        multiple copies of mag list are used to speed up searching
    :param batch: batch size for searching in mag list
    :param workers: number of workers
    :return: oa_2_mag
    """

    records = PapersList.read_records(oa_papers, workers=workers)

    first_mag = mags[0]
    oa_2_mag = [-1] * len(records)

    unk_records = []
    unk_records_indices = []

    with MapAccessFile(oa_papers, oa_papers + ".index") as f, \
            (FunctorPool(workers=[MagIdReader(oa_papers) for _ in range(workers)], context=CONTEXT_SPAWN,
                         results_queue_maxsize=2.0) if workers > 0 else nullcontext()) as pool:

        if workers == 0:
            single_process_reader = MagIdReader(oa_papers)
            single_process_reader.begin()
            m = partial(map, single_process_reader)
        else:
            m = partial(pool.imap, chunk_size=1000)

        try:
            for i, mag_id in enumerate(
                    tqdm(m(f.mapping.values()), desc="Mapping oa to mag with known mag ids", total=len(f.mapping))):
                mag_index = None
                if mag_id is not None:
                    # reuse the id from dataset
                    try:
                        mag_index = first_mag.id_2_index(mag_id)
                    except KeyError:
                        # unknown mag record
                        pass

                if mag_index is None:
                    unk_records.append(records[i])
                    unk_records_indices.append(i)
                else:
                    oa_2_mag[i] = mag_index
        finally:
            if workers == 0:
                single_process_reader.end()

    if unk_records:
        # try to find the mag record using title, year and authors

        mag_indices = PapersList.map_records_to_list(unk_records, mags, reverse=False, batch_size=batch)

        for i, mag_index in enumerate(tqdm(mag_indices, desc="Mapping the rest using search results")):
            if mag_index is not None:
                oa_2_mag[unk_records_indices[i]] = mag_index

    return oa_2_mag


def create_papers_2_mag_mapping(args: argparse.Namespace):
    """
    Creates mapping from oa papers to mag papers.

    :param args: user arguments
    """

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    managers = None
    allow_gpus = None
    if workers > 0:
        if args.force_gpu_split:
            managers = ContextList([PapersListManager(ctx=CONTEXT_SPAWN)])
            allow_gpus = [[i for i in range(faiss.get_num_gpus())]]
        else:
            managers = ContextList(
                [PapersListManager(ctx=CONTEXT_SPAWN) for _ in range(max(1, min(workers, faiss.get_num_gpus())))])
            allow_gpus = [[i] for i in range(len(managers))]

    with nullcontext() if managers is None else managers:
        mags = MAGPapersList.from_file(args.mag, workers=workers, manager=managers, allow_gpus=allow_gpus)

        if not isinstance(mags, list):
            mags = [mags]

        for m in mags:
            m.return_self_on_enter(False)  # otherwise may cause problems in mult. proc. mode

        with ContextList(mags):
            papers_2_mag = oa_papers_2_mag_mapping(args.dataset, mags, args.batch, workers=workers)

        for mag_index in papers_2_mag:
            print(mag_index)


def enhance_papers_with_mag(args: argparse.Namespace):
    """
    Enhances papers with mag information.

    :param args: user arguments
    """

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    logging.log(logging.INFO, "Loading MAG index")
    mag_indices_2_ids, _, mag_mapping = OADataset.load_mappings(args.mag + ".index")

    with MapAccessFile(args.mag, mag_mapping, key_type=int) as mag_index_f:
        mag_index_f.mapping = SortedMap(mag_index_f.mapping)
        mag_ids_2_indices = SortedMap((mag_id, i) for i, mag_id in enumerate(mag_indices_2_ids))

        if workers > 0:
            mag_index_f.mapping.keys_storage = multiprocessing.Array(ctypes.c_int64, mag_index_f.mapping.keys_storage,
                                                                     lock=False)
            mag_index_f.mapping.values_storage = multiprocessing.Array(ctypes.c_int64,
                                                                       mag_index_f.mapping.values_storage, lock=False)
            mag_indices_2_ids = multiprocessing.Array(ctypes.c_int64, mag_indices_2_ids, lock=False)
            mag_ids_2_indices.keys_storage = multiprocessing.Array(ctypes.c_int64, mag_ids_2_indices.keys_storage,
                                                                   lock=False)
            mag_ids_2_indices.values_storage = multiprocessing.Array(ctypes.c_int64, mag_ids_2_indices.values_storage,
                                                                     lock=False)

        transform = MagEnhanceTransform(mag_index_f,
                                        mag_indices_2_ids,
                                        mag_ids_2_indices,
                                        args.to_mag,
                                        args.dataset,
                                        args.just_references,
                                        args.match_threshold,
                                        workers=max(workers, 1))

        p_workers = [GivenFunctorWorker(transform) for _ in range(max(workers, 1))]
        with FunctorPool(workers=p_workers, context=CONTEXT_SPAWN,
                         results_queue_maxsize=10.0) if workers > 0 else nullcontext() as pool:
            logging.log(logging.INFO, "Loading OA index")
            with OADataset(args.dataset,
                           args.dataset + ".index") as dataset:  # do not set workers > 0 here, we are counting on original positions
                if workers > 0:
                    m = partial(pool.imap, chunk_size=1000)
                else:
                    m = partial(map, transform)
                    p_workers[0].begin()
                try:
                    with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f:
                        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"],
                                                      delimiter="\t")
                        index_writer.writeheader()
                        for doc_id, doc in tqdm(m((offset, i) for i, offset in enumerate(dataset.mapping.values())),
                                                total=len(dataset), desc=f"Enhancing dataset", unit="doc"):
                            index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
                            print(doc, file=res_f)
                finally:
                    if workers == 0:
                        p_workers[0].end()


class RelatedWorkTransform:
    """
    Transforms documents containing related work sections. If the document doesn't have related work section it will
    transform it into a None.
    """

    def __init__(self, references_ids: SortedSet[int]):
        """
        :param references_ids: ids of documents that might be references
        """
        self.related_work_regex = re.compile(r"^((^|\s|\()((((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|"
                                             r"[0-9]+|[a-z])(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+"
                                             r"|[a-z]))*\.?)($|\s|\)))?\s*((related\s+works?)|(related\s+literature)|"
                                             r"(theoretical\s+background)|(literature\s+review))\s*$", re.IGNORECASE)

        self.background_regex = re.compile(r"^((^|\s|\()((((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|"
                                           r"[0-9]+|[a-z])(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+"
                                           r"|[a-z]))*\.?)($|\s|\)))?\s*background\s*$", re.IGNORECASE)

        self.introduction_regex = re.compile(r"^((^|\s|\()((((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|"
                                             r"[0-9]+|[a-z])(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+"
                                             r"|[a-z]))*\.?)($|\s|\)))?\s*introduction\s*$", re.IGNORECASE)

        self.references_ids = references_ids

    def __call__(self, document: Document) -> Optional[Tuple[int, str, Set[int]]]:
        """
        Transformation of document into form that is suitable for creating related work dataset.

        :param document: document to transform
        :return:
            None if the document doesn't contain related work section

            else
                id of document
                json of document for related work dataset
                ids of references
        """

        # we need documents with title, abstract and related work section
        # we will search in depth 1 as we want to speed up things and also prevent unwanted recursion when
        # constructing new tree as there was one document named Abstract which causes that a related work section
        # was inside the abstract section and the hierarchy was not tree anymore
        related_work_section = document.hierarchy.get_part(self.related_work_regex, max_h=2, min_depth=1, max_depth=1,
                                                           return_path=True)

        # check whether we can use background section
        if not related_work_section:
            introduction_section = document.hierarchy.get_part(self.introduction_regex, max_h=1, min_depth=1,
                                                               max_depth=1)
            if introduction_section:
                introduction_section = introduction_section[0]
                # we want the introduction to be present as sometimes background is writen as introduction
                background_section = document.hierarchy.get_part(self.background_regex, max_h=2, min_depth=1,
                                                                 max_depth=1, return_path=True)
                if len(background_section) == 1 and introduction_section.height == background_section[0][0].height and \
                        not introduction_section.get_part(self.background_regex, max_h=2):
                    # we don't want the background to be part of introduction as in some cases it was not
                    # literature review
                    related_work_section = background_section

        if len(related_work_section) == 1 and any(
                len(t_c.text) > 0 for t_c in related_work_section[0][0].text_content()):
            related_work_section, related_work_section_path = related_work_section[0]

            abstract_section = document.hierarchy.get_part(ABSTRACT_REGEX, max_h=1, min_depth=1, max_depth=1,
                                                           return_path=True)

            if abstract_section and any(len(t_c.text) > 0 for t_c in abstract_section[0][0].text_content()):
                abstract_section, abstract_section_path = abstract_section[0]
                abstract_section.headline = "Abstract"

                # remove related work section from hierarchy and make sure that abstract is first section
                if Hierarchy.serial_numbers_sparsity(abstract_section_path, related_work_section_path) >= 0:
                    document.hierarchy.remove(related_work_section_path)
                    document.hierarchy.remove(abstract_section_path)
                else:
                    document.hierarchy.remove(abstract_section_path)
                    document.hierarchy.remove(related_work_section_path)

                document.hierarchy.content.insert(0, abstract_section)

                # we want only citations from related work section
                citations = set()

                for t_c in related_work_section.text_content():
                    for c in t_c.citations:
                        if c.index is not None:
                            try:
                                cited_id = document.bibliography[c.index].id
                            except IndexError as e:
                                print(document.id)
                                raise e
                            if cited_id in self.references_ids:
                                citations.add(cited_id)

                document.citations = sorted(citations)

                doc_res = document.asdict()
                doc_res["related_work"] = related_work_section.asdict()
                doc_res["related_work_orig_path"] = related_work_section_path

                referenced_documents = set()
                for bib_entry in document.bibliography:
                    if bib_entry.id is not None and bib_entry.id in self.references_ids:
                        referenced_documents.add(bib_entry.id)

                return document.id, json_dumps(doc_res), referenced_documents

        return None


def create_related_work(args: argparse.Namespace):
    """
    Creates related work oriented OAReviews dataset.

    :param args: user arguments
    """

    referenced_documents = []

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    with OADataset(args.documents, args.documents + ".index", workers=workers, lazy_hierarchy=True) as dataset, \
            open(args.reviews, "w") as res_f, \
            open(args.reviews + ".index", "w") as res_index_f, \
            (
                    OADataset(args.references_dataset, args.references_dataset + ".index")
                    if args.references_dataset else dataset
            ) as references_dataset:
        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        dataset.transform = RelatedWorkTransform(SortedSet(references_dataset.mapping.keys()))
        dataset.chunk_size = 100

        for doc_res in tqdm(dataset.iter_range(unordered=args.unordered), total=len(dataset),
                            desc="Collecting"):
            if doc_res is not None:
                doc_id, json_res, references = doc_res

                for i in references:
                    referenced_documents.append(i)

                index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
                print(json_res, file=res_f)

        dataset.transform = None  # in case it is the same as references_dataset
        write_references(args.references, references_dataset, SortedSet(referenced_documents))


def filter_oa_dataset_with_reviews(reviews: DocumentDataset, references: DocumentDataset, reviews_filter: Filter,
                                   references_filter: Filter, res_reviews: str, res_references: str):
    """
    Filters OA dataset.

    :param reviews: the dataset of reviews
    :param references: references mentioned in reviews
    :param reviews_filter: filter for filtering reviews
    :param references_filter: filters that will be used for filtering references
    :param res_reviews: path where filtered reviews should be saved
    :param res_references: path where filtered references should be saved
    """

    with open(res_reviews, "w") as res_f, open(res_reviews + ".index", "w") as res_index_f, \
            open(res_references, "w") as ref_res_f, open(res_references + ".index", "w") as ref_res_index_f:
        # we want to make sure that referenced papers are really in dataset even after filtration
        ids_of_filtered_references = SortedSet(get_ids_of_filter_passing_documents(references, references_filter))

        filter_and_print(reviews, reviews_filter, res_f, res_index_f, ids_of_filtered_references)

        filter_and_print(references, references_filter, ref_res_f, ref_res_index_f, None)


def filter_related_work(args: argparse.Namespace):
    """
    Filters related work oriented OAReviews dataset.

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    references_filter = CombinedFilter([NumberOfSectionsFilter(args.min_sec_ref, args.max_sec_ref),
                                        NumberOfTextPartsInSectionFilter(args.min_par_ref, args.max_par_ref),
                                        ])
    if args.sec_non_empty_headlines_ref:
        references_filter.filters.append(SecNonEmptyHeadlinesFilter())

    if args.has_abstract_ref:
        references_filter.filters.append(HasHeadlineFilter(ABSTRACT_REGEX, args.has_abstract_ref, 1, 1))

    reviews_filter = CombinedFilter([NumberOfCitationsFilter(args.min_cit, args.max_cit),
                                     CitationsFracFilter(args.min_cit_frac, args.max_cit_frac, True),
                                     CitationsGroupsFracFilter(args.min_cit_group_frac, args.max_cit_group_frac, True),
                                     NumberOfSectionsFilter(args.min_sec_rev, args.max_sec_rev, True),
                                     NumberOfTextPartsInSectionFilter(args.min_par_rev, args.max_par_rev, True),
                                     ])

    if args.has_abstract_rev:
        reviews_filter.filters.append(HasHeadlineFilter(ABSTRACT_REGEX, args.has_abstract_rev, 1, 1))

    with OARelatedWork(args.related_work, args.related_work + ".index", workers=workers) as reviews, \
            OADataset(args.references, args.references + ".index", workers=workers) as references:

        reviews_filter.filters.append(FractionOfCitedDocumentsWithMultiSectionContentFilter(references,
                                                                                            args.min_fraction_of_cited_documents_with_multi_section_content_filter,
                                                                                            args.max_fraction_of_cited_documents_with_multi_section_content_filter))

        filter_oa_dataset_with_reviews(reviews, references, reviews_filter, references_filter, args.res_related_work,
                                       args.res_references)


def create_filter_features(args: argparse.Namespace):
    """
    Prints tsv containing features that are used by filters on stdout.

    :param args: user arguments
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    postfix = ' in related work' if args.rw else ''
    writer = csv.DictWriter(sys.stdout, fieldnames=["id", f"non empty headlines{postfix}", f"citations{postfix}",
                                                    f"in content known citations fractions{postfix}",
                                                    f"in content known citations groups fractions{postfix}",
                                                    f"in content known citations paragraph fractions{postfix}",
                                                    f"in content unk bib. entry citations fractions{postfix}",
                                                    f"fraction of cited documents with multi-section content{postfix}",
                                                    "known bibliography entries fraction",
                                                    f"sections{postfix}",
                                                    f"paragraphs{postfix}",
                                                    f"min text parts in a section{postfix}",
                                                    f"max text parts in a section{postfix}",
                                                    "has abstract headline",
                                                    f"number of text parts{postfix}",
                                                    f"citations groups with at least one cited multi-section paper fractions{postfix}",
                                                    f"paragraphs with at least one cited multi-section paper fractions{postfix}",
                                                    f"number of target words{postfix}",
                                                    f"number of abstract words",
                                                    f"number of input words",
                                                    f"number of abstract text parts",
                                                    f"number of input text parts",
                                                    f"citation spans",
                                                    f"citation groups",
                                                    ], delimiter="\t")
    writer.writeheader()

    has_headline_filter = HasHeadlineFilter(ABSTRACT_REGEX, min_text_parts=1, min_depth=1, max_depth=1)

    ref_path = args.references if args.references else args.dataset

    with (OARelatedWork if args.rw else OADataset)(args.dataset, args.dataset + ".index", workers=workers) as dataset, \
            OADataset(ref_path, ref_path + ".index") as references:

        references_cache = LFUCache(10_000)

        def trans(d: Union[Document, OARelatedWorkDocument]):
            min_text_parts, max_text_parts = math.inf, -math.inf

            known_content_cit = 0
            all_content_cit = 0
            unk_bib_entry_content_cit = 0

            known_content_cit_groups = 0
            multi_sec_content_cit_groups = 0
            all_content_cit_groups = 0

            known_content_cit_paragraphs = 0  # number of paragraphs containing at least one known citation
            known_content_cit_paragraphs_covered_by_multi_sec_paper = 0  # number of paragraphs containing at least one known citation that is multi-section paper
            all_content_cit_paragraphs = 0  # number of paragraphs containing at least one identified citation

            hierarchy = d.related_work if args.rw else d.hierarchy
            content = hierarchy.content if args.rw else hierarchy.content
            for s in content:
                cnt = 0

                for p in s.nodes_with_height(1):
                    cit_paragraph = False
                    known_cit_paragraph = False
                    known_cit_paragraph_covered_by_multi_sec_paper = False
                    for t in p.text_content():
                        cnt += 1
                        previous_c = None
                        previous_c_hit = False
                        previous_c_hit_multi_section = False
                        for c in t.citations:
                            all_content_cit += 1
                            cit_paragraph = True
                            unk_bib_entry_content_cit += c.index is None
                            if previous_c is None or not re.match(r"^\W*$", t.text[previous_c.end: c.start]):
                                # new cit. group
                                known_content_cit_groups += previous_c_hit
                                multi_sec_content_cit_groups += previous_c_hit_multi_section
                                all_content_cit_groups += 1
                                previous_c_hit = False
                                previous_c_hit_multi_section = False

                            if c.index is not None and d.bibliography[c.index].id is not None:
                                known_content_cit += 1
                                previous_c_hit = True
                                known_cit_paragraph = True

                                if not previous_c_hit_multi_section and len(
                                        references.get_by_id(d.bibliography[c.index].id).hierarchy.content) > 1:
                                    previous_c_hit_multi_section = True
                                    known_cit_paragraph_covered_by_multi_sec_paper = True

                            previous_c = c
                        known_content_cit_groups += previous_c_hit
                        multi_sec_content_cit_groups += previous_c_hit_multi_section

                    known_content_cit_paragraphs += known_cit_paragraph
                    known_content_cit_paragraphs_covered_by_multi_sec_paper += known_cit_paragraph_covered_by_multi_sec_paper
                    all_content_cit_paragraphs += cit_paragraph

                if cnt < min_text_parts:
                    min_text_parts = cnt
                if cnt > max_text_parts:
                    max_text_parts = cnt

            hier_for_query_words = d.abstract

            number_of_abstract_words = 0
            number_of_abstract_text_parts = 0
            if hier_for_query_words is not None:
                number_of_abstract_words = sum(len(t.text.split()) for t in hier_for_query_words.text_content()) + sum(
                    len(n.headline.split()) for n in hier_for_query_words.nodes_with_height(2))
                number_of_abstract_text_parts = sum(1 for _ in hier_for_query_words.text_content())

            cited_doc_with_mutl_sec_cont = 0
            number_of_input_words = 0
            number_of_input_text_parts = 0

            for i in d.citations:
                if i in references:
                    try:
                        add_number_of_input_text_parts, add_number_of_input_words, add_cited_doc_with_mutl_sec_cont = references_cache[i]
                    except KeyError:
                        ref = references.get_by_id(i)
                        add_number_of_input_text_parts = sum(1 for _ in ref.hierarchy.text_content())
                        add_number_of_input_words = sum(len(t.text.split()) for t in ref.hierarchy.text_content()) + sum(
                            len(n.headline.split()) for n in ref.hierarchy.nodes_with_height(2))
                        add_cited_doc_with_mutl_sec_cont = len(ref.hierarchy.content) > 1
                        references_cache[i] = (add_number_of_input_text_parts, add_number_of_input_words, add_cited_doc_with_mutl_sec_cont)

                    number_of_input_text_parts += add_number_of_input_text_parts
                    number_of_input_words += add_number_of_input_words
                    cited_doc_with_mutl_sec_cont += add_cited_doc_with_mutl_sec_cont

            return {
                "id": d.id,
                f"non empty headlines{postfix}": all(s.headline for s in content),
                f"citations{postfix}": len(d.citations),
                f"in content known citations fractions{postfix}":
                    0 if all_content_cit == 0 else known_content_cit / all_content_cit,
                f"in content known citations groups fractions{postfix}":
                    0 if all_content_cit_groups == 0 else known_content_cit_groups / all_content_cit_groups,
                f"in content known citations paragraph fractions{postfix}":
                    0 if all_content_cit_paragraphs == 0 else known_content_cit_paragraphs / all_content_cit_paragraphs,
                f"in content unk bib. entry citations fractions{postfix}": 0 if all_content_cit == 0 else unk_bib_entry_content_cit / all_content_cit,
                f"fraction of cited documents with multi-section content{postfix}": cited_doc_with_mutl_sec_cont / len(
                    d.citations) if len(d.citations) > 0 else 0,
                "known bibliography entries fraction":
                    0 if len(d.bibliography) == 0 else sum(b.id is not None for b in d.bibliography) / len(
                        d.bibliography),
                f"sections{postfix}": sum(1 for _ in hierarchy.sections()),
                f"paragraphs{postfix}": sum(1 for _ in hierarchy.nodes_with_height(1)),
                f"min text parts in a section{postfix}": min_text_parts,
                f"max text parts in a section{postfix}": max_text_parts,
                "has abstract headline": has_headline_filter(d),
                f"number of text parts{postfix}": sum(1 for _ in hierarchy.text_content()),
                f"citations groups with at least one cited multi-section paper fractions{postfix}": 0 if all_content_cit_groups == 0 else multi_sec_content_cit_groups / all_content_cit_groups,
                f"paragraphs with at least one cited multi-section paper fractions{postfix}": 0 if all_content_cit_paragraphs == 0 else known_content_cit_paragraphs_covered_by_multi_sec_paper / all_content_cit_paragraphs,
                f"number of target words{postfix}": sum(len(t.text.split()) for t in hierarchy.text_content()) + sum(
                    len(n.headline.split()) for n in hierarchy.nodes_with_height(2)),
                f"number of abstract words": number_of_abstract_words,
                f"number of input words": number_of_input_words,
                f"number of abstract text parts": number_of_abstract_text_parts,
                f"number of input text parts": number_of_input_text_parts,
                f"citation spans": sum(1 for _ in hierarchy.citation_spans()),
                f"citation groups": all_content_cit_groups
            }

        dataset.transform = trans
        dataset.chunk_size = 100
        for res in tqdm(dataset, desc="Filter features"):
            writer.writerow(res)


class CreateStatsWorker(FunctorWorker):
    """
    Multiprocessing functor that creates statistics for given subset of documents.
    """

    def __init__(self, dataset: str, rw: bool = False, references: Optional[str] = None):
        """
        :param dataset: path to dataset
        :param rw: true if the dataset is in related work format
        :param references: path to references
            if rw is true and references are provided it will use RelatedWorkStats instead of DocumentStats
        """
        super().__init__()
        self.dataset_path = dataset
        self.rw = rw
        self.dataset = None

        self.references_path = references
        self.references = None

    def begin(self):
        self.dataset = open(self.dataset_path, "r")
        if self.references_path:
            self.references = OADataset(self.references_path, self.references_path + ".index")
            self.references.open()

    def end(self):
        self.dataset.close()
        self.dataset = None
        if self.references_path:
            self.references.close()
            self.references = None

    def __call__(self, offset: List[int]) -> DocumentsStats:
        """
        Creates statistics for given range of documents.
        :param offset: list of document offsets
        :return: statistics
        """

        if self.references is not None and self.rw:
            stats = RelatedWorkStats(self.references)
        else:
            stats = DocumentsStats()

        for o in offset:
            self.dataset.seek(o)
            line = json_loads(self.dataset.readline())
            d = OARelatedWorkDocument.from_dict(line) if self.rw else Document.from_dict(line)
            stats.process(d)

        if isinstance(stats, RelatedWorkStats):
            # because of pickling
            stats.references = None

        return stats


def create_stats(args: argparse.Namespace):
    """
    Prints tsv containing features that are used by filters on stdout.

    :param args: user arguments
    """
    RANGE_SIZE = 1000
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    if args.rw and args.references is not None:
        stats = RelatedWorkStats()
    else:
        stats = DocumentsStats()

    with (OARelatedWork if args.rw else OADataset)(args.dataset, args.dataset + ".index") as dataset:
        if workers > 0:
            workers = [CreateStatsWorker(args.dataset, args.rw, args.references) for _ in range(workers)]

            with FunctorPool(workers) as pool:
                gen_chunks = ([dataset.indices_2_offsets[i] for i in range(x, min(x + RANGE_SIZE, len(dataset)))]
                              for x in range(0, len(dataset), RANGE_SIZE))
                for s in tqdm(pool.imap_unordered(gen_chunks),
                              desc="Processing documents",
                              unit="chunk", total=math.ceil(len(dataset) / RANGE_SIZE)):
                    stats.update(s)
        else:
            references = nullcontext()
            if args.rw and args.references is not None:
                references = OADataset(args.references, args.references + ".index")
                stats = RelatedWorkStats(references)

            with references:

                for d in tqdm(dataset, desc="Reading documents", unit="doc"):
                    stats.process(d)

    print(stats, file=sys.stdout)


def create_stats(args: argparse.Namespace):
    """
    Prints tsv containing features that are used by filters on stdout.

    :param args: user arguments
    """
    RANGE_SIZE = 1000
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    if args.rw and args.references is not None:
        stats = RelatedWorkStats()
    else:
        stats = DocumentsStats()

    with (OARelatedWork if args.rw else OADataset)(args.dataset, args.dataset + ".index") as dataset:
        if workers > 0:
            workers = [CreateStatsWorker(args.dataset, args.rw, args.references) for _ in range(workers)]

            with FunctorPool(workers) as pool:
                gen_chunks = ([dataset.indices_2_offsets[i] for i in range(x, min(x + RANGE_SIZE, len(dataset)))]
                              for x in range(0, len(dataset), RANGE_SIZE))
                for s in tqdm(pool.imap_unordered(gen_chunks),
                              desc="Processing documents",
                              unit="chunk", total=math.ceil(len(dataset) / RANGE_SIZE)):
                    stats.update(s)
        else:
            references = nullcontext()
            if args.rw and args.references is not None:
                references = OADataset(args.references, args.references + ".index")
                stats = RelatedWorkStats(references)

            with references:

                for d in tqdm(dataset, desc="Reading documents", unit="doc"):
                    stats.process(d)

    print(stats, file=sys.stdout)


def deduplication(args: argparse.Namespace):
    """
    Performs dataset deduplication.

    :param args: user arguments
    """

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    papers_list = PapersList.from_file(args.dataset, match_threshold=args.match_threshold,
                                       max_year_diff=args.max_year_diff,
                                       workers=workers)

    papers_list.set_search_workers(workers)

    with OADataset(args.dataset, args.dataset + ".index", workers=workers) as dataset, \
            open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f:
        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()
        with papers_list:
            # following code is optimized with assumption that the duplicates are really rare
            ids_trans = {}

            # structures for storing representatives of duplicates equivalence classes
            duplicates_eq_classes_representatives = PapersList([])
            duplicates_eq_classes_representatives_ids = []

            for b in tqdm(BatcherIter((({i} for i in range(len(papers_list))), papers_list), batch_size=args.batch),
                          desc="Searching duplicates", total=math.ceil(len(papers_list) / args.batch), unit="batch"):
                for i, x in enumerate(papers_list.batch_search(b[1], skip_indices=b[0])):
                    ele_ind = next(iter(b[0][i]))
                    if x is None:
                        # no duplicate whatsoever, this should trigger most of the time
                        act_id = dataset.indices_2_id[ele_ind]
                        ids_trans[act_id] = act_id
                    else:
                        # we have found duplicate pair, but there could be a whole group of duplicates, lets check whether
                        # there is already established equivalence class if not let's establish it
                        act_id = dataset.indices_2_id[ele_ind]
                        duplicate_id = dataset.indices_2_id[x]
                        if act_id in ids_trans or duplicate_id in ids_trans:
                            # equivalence class already exists, and we can get equivalence class representative id from id
                            # translation mapping

                            if act_id in ids_trans:
                                ids_trans[duplicate_id] = ids_trans[act_id]
                            else:
                                ids_trans[act_id] = ids_trans[duplicate_id]
                            # There is also the case when both are there. In that case it is just shuffle of two same
                            # values. So who cares?
                        else:
                            # we need to search the equivalence class representative in the list
                            record = [b[1][i]]
                            eq_class_index = duplicates_eq_classes_representatives.batch_search(record)[0]
                            if eq_class_index is None:
                                # let's establish a new class
                                duplicates_eq_classes_representatives.add(record)
                                duplicates_eq_classes_representatives_ids.append(act_id)
                                ids_trans[act_id] = act_id
                                ids_trans[duplicate_id] = act_id
                            else:
                                ids_trans[act_id] = duplicates_eq_classes_representatives_ids[eq_class_index]
                                ids_trans[duplicate_id] = duplicates_eq_classes_representatives_ids[eq_class_index]

        ids_trans = SortedMap(ids_trans)
        if workers > 0:
            ids_trans.keys_storage = multiprocessing.Array(ctypes.c_int64, ids_trans.keys_storage, lock=False)
            ids_trans.values_storage = multiprocessing.Array(ctypes.c_int64, ids_trans.values_storage, lock=False)

        del papers_list
        del duplicates_eq_classes_representatives
        del duplicates_eq_classes_representatives_ids

        def transform(d):
            d: Document
            representative_id = ids_trans[d.id]
            d_id = d.id
            if representative_id == d.id:
                # representative
                d.translate_ids(ids_trans)
                d = str(d)
            else:
                d = None

            return d_id, representative_id, d

        dataset.transform = transform
        # go through and print just the single representative for each equivalence class
        dup_cnt = 0
        for doc_id, r_id, doc in (
                p_bar := tqdm(dataset, desc=f"Writing results ({dup_cnt} duplicates removed)", total=len(dataset))):

            if doc is None:
                dup_cnt += 1
                p_bar.desc = f"Writing results ({dup_cnt} duplicates)"
                print(f"removing duplicate with {doc_id} id, its representative is {r_id}")

            else:
                # representative
                index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
                print(doc, file=res_f)


class IdentifyBib:
    """
    Functor for identifying bibliography.
    """

    def __init__(self, dataset: OADataset, papers_list: Sequence[PapersList], indices_2_id: Sequence[int],
                 batch_size: int):
        """
        :param dataset: dataset to process
        :param papers_list: list that will be used to search for bibliography
            you can pass multiple list to better load balance
        :param indices_2_id: mapping from paper list index to document id
        :param batch_size: maximal batch size used for searching references
        """
        self.dataset = dataset
        self.papers_list = papers_list
        self.indices_2_id = indices_2_id
        self.batch_size = batch_size

        self.total_cnt = 0
        self.wait_time = 0
        self.total_time = 0
        self.document_parse_time = 0
        self.identify_time = 0
        self.start_wait_time = 0
        self.profile_prob = 0

    def begin(self):
        # called in multiprocessing context when a worker is started
        random.seed(time.time() + os.getpid())
        self.start_wait_time = time.time()
        self.dataset.__enter__()

    def end(self):
        self.dataset.__exit__(None, None, None)

    def __call__(self, indices: Iterable[int]) -> Tuple[List[int], List[str], List[int]]:
        """
        Identifies bibliography in the document.

        :param indices: batch of indices of documents in the dataset
        :return: tuple containing
            list of document ids,
            list of strings version of documents
            list of number of newly identified citations for each document
        """
        self.wait_time += time.time() - self.start_wait_time
        start_t = time.time()
        papers_list = random.choice(self.papers_list)

        docs = []

        for i in indices:
            self.total_cnt += 1
            t = time.time()
            try:
                docs.append(self.dataset[i])
            except JSONDecodeError as e:
                print(f"Error while parsing document {i}: {e}")
                self.dataset.file.seek(self.dataset.indices_2_offsets[i])
                print(f"Line: {self.dataset.file.readline()}", file=sys.stderr, flush=True)
                raise e

            self.document_parse_time += time.time() - t

        t = time.time()
        act_new_cite_cnt = Document.identify_bibliography_docs(docs, papers_list, self.indices_2_id, self.batch_size)
        self.identify_time += time.time() - t

        ids = []
        strs = []
        for doc in docs:
            ids.append(doc.id)
            strs.append(str(doc))

        self.total_time += time.time() - start_t

        if self.profile_prob > 0 and random.random() < self.profile_prob:
            print(
                f"total_cnt: {self.total_cnt}, avg total_time: {self.total_time / self.total_cnt}, avg document_parse_time: {self.document_parse_time / self.total_cnt}, avg identify_time: {self.identify_time / self.total_cnt}, avg wait_time: {self.wait_time / self.total_cnt}",
                flush=True, file=sys.stderr)

        self.start_wait_time = time.time()
        return ids, strs, act_new_cite_cnt


def identify_bibliography_for_dataset(dataset: OADataset, paper_lists: Sequence[PapersList],
                                      indices_2_id: Sequence[int],
                                      res_file: TextIO, index_writer: csv.DictWriter,
                                      workers: int, identify_for: Optional[Collection[int]] = None,
                                      batch_size: int = 1000,
                                      batch_size_search: int = 1000):
    """
    Identifies bibliography for the dataset and writes the results to the res_file.

    :param dataset: dataset to process
    :param paper_lists: list of papers lists to use for searching bibliography
    :param indices_2_id: mapping from paper list index to document id
    :param res_file: file to write the results to
    :param index_writer: csv writer for writing the index (headline will not be written)
    :param workers: number of workers to use for searching bibliography
    :param identify_for: indices of documents to identify bibliography for, if None, all documents will be processed
    :param batch_size: maximal batch size used for loading documents
       We are processing multiple documents in a batch as we want to aggregate bibliography entries that we are searching for.
    :param batch_size_search: maximal batch size used for searching references
    """
    transform = IdentifyBib(dataset, paper_lists, indices_2_id, batch_size_search)
    p_workers = [GivenFunctorWorker(transform) for _ in range(max(workers, 1))]

    new_cite_cnt = 0

    with FunctorPool(workers=p_workers, results_queue_maxsize=10.0) if workers > 0 else nullcontext() as pool, \
            ContextList(paper_lists):

        if workers > 0:
            m = partial(pool.imap)
        else:
            m = partial(map, transform)
            p_workers[0].begin()

        identify_for = identify_for if identify_for is not None else range(len(dataset))

        try:
            for doc_ids, doc_strs, act_new_cite_cnts in (
                    p_bar := tqdm(
                        m(BatcherIter(identify_for, batch_size)),
                        desc=f"Identifying | new citations 0",
                        total=math.ceil(len(identify_for) / batch_size), unit=f"batch {batch_size}")):

                for doc_id, doc_str, act_new_cite_cnt in zip(doc_ids, doc_strs, act_new_cite_cnts):
                    index_writer.writerow({"key": doc_id, "file_line_offset": res_file.tell()})
                    print(doc_str, file=res_file)
                    new_cite_cnt += act_new_cite_cnt
                p_bar.desc = f"Identifying | new citations {new_cite_cnt}"
        finally:
            if workers == 0:
                p_workers[0].end()
    return new_cite_cnt


def identify_bibliography(args: argparse.Namespace):
    """
    Tries to identify bibliography.

    :param args: user arguments
    """

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    managers = None
    allow_gpus = None
    if workers > 0:
        if args.force_gpu_split:
            managers = ContextList([PapersListManager(ctx=CONTEXT_SPAWN)])
            allow_gpus = [[i for i in range(faiss.get_num_gpus())]]
        else:
            managers = ContextList(
                [PapersListManager(ctx=CONTEXT_SPAWN) for _ in range(max(1, min(workers, faiss.get_num_gpus())))])
            allow_gpus = [[i] for i in range(len(managers))]

    shared_records_list_manager = SharedMemoryManager() if managers is not None else None
    with nullcontext() if managers is None else managers, \
            nullcontext() if shared_records_list_manager is None else shared_records_list_manager:
        papers_list = PapersList.from_file(args.dataset if args.search is None else args.search,
                                           match_threshold=args.match_threshold,
                                           max_year_diff=args.max_year_diff,
                                           workers=workers,
                                           manager=managers,
                                           allow_gpus=allow_gpus,
                                           search_cache_size=0,
                                           shared_list_for_records=shared_records_list_manager,
                                           fulltext_search=args.title_db
                                           )

        if not isinstance(papers_list, list):
            papers_list = [papers_list]

        logging.log(logging.INFO, f"Loading dataset {args.dataset}")

        dataset = OADataset(args.dataset, args.dataset + ".index", workers=workers)

        if args.search is not None:
            logging.log(logging.INFO, f"Loading dataset {args.search}")
            search_dataset = OADataset(args.search, args.search + ".index", workers=workers)
            indices_2_id = search_dataset.indices_2_id
        else:
            indices_2_id = dataset.indices_2_id

        if workers > 0:
            logging.log(logging.INFO, f"Saving indices_2_id to shared memory")
            indices_2_id = multiprocessing.Array(ctypes.c_int64, indices_2_id, lock=False)

        from_i = args.from_i if args.from_i is not None else 0
        to_i = args.to_i if args.to_i is not None else len(dataset)

        dataset.hierarchy_as_dict = True
        with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f:
            index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
            index_writer.writeheader()
            new_cite_cnt = identify_bibliography_for_dataset(dataset, papers_list, indices_2_id, res_f,
                                                             index_writer, workers,
                                                             (range(from_i, to_i)), args.docs, args.batch)

            print(f"new citations {new_cite_cnt}")

class IdentifyCitSpansFunctor:
    """
    Functor for identifying citations spans in documents. This means that it will find new spans or matches existing
    ones with a bibliographic entry.

    """

    def __init__(self, dataset: str):
        """
        :param dataset: path to dataset
        """

        self.dataset = dataset
        self._file_dataset = None

    def begin(self):
        """
        Initializes dataset.
        """
        self._file_dataset = open(self.dataset, "r")

    def end(self):
        self._file_dataset.close()

    def __call__(self, offset: int) -> Tuple[int, str, int]:
        """
        Identifies citations spans in given document.

        :param offset: document offset in file
        :return: tuple containing document id, string version of document, and number of newly identified citations
            spans
        """

        self._file_dataset.seek(offset)
        doc = Document.from_dict(json_loads(self._file_dataset.readline()))

        new_cite_cnt = 0
        if identify_citation_style_of_doc(doc) == CitationStyle.HARVARD:
            matched_cit_span_cnt = sum(c.index is not None for c in doc.citation_spans())
            add_missing_harvard_style_citations(doc)
            match_unk_citation_spans_with_bib(doc)
            new_cite_cnt = sum(c.index is not None for c in doc.citation_spans()) - matched_cit_span_cnt

        return doc.id, str(doc), new_cite_cnt


def identify_citation_spans(args: argparse.Namespace):
    """
   Identifies citation spans in dataset.

   :param args: user arguments
   """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()
    transform = IdentifyCitSpansFunctor(args.dataset)

    with FunctorPool(workers=[GivenFunctorWorker(transform) for _ in range(workers)],
                     results_queue_maxsize=2.0) as pool:
        with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f, \
                OADataset(args.dataset, args.dataset + ".index") as dataset:

            index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
            index_writer.writeheader()

            new_identified_cite = 0
            if workers > 0:
                m = partial(pool.imap, chunk_size=1000)
            else:
                m = partial(map, transform)
                transform.begin()
            try:
                for d_id, document_json, new_cite in (pbar := tqdm(m(dataset.indices_2_offsets),
                                                                   desc="Identifying citation spans",
                                                                   total=len(dataset))):
                    index_writer.writerow({"key": d_id, "file_line_offset": res_f.tell()})
                    print(document_json, file=res_f)
                    new_identified_cite += new_cite
                    pbar.set_postfix({"new_identified_cite": new_identified_cite})
            finally:
                if workers == 0:
                    transform.end()


class CitedIdsTransform:
    """
    Get all ids from bibliography.
    """

    def __call__(self, document: Document) -> List[int]:
        return [ref.id for ref in document.bibliography if ref.id is not None]


def create_references(args: argparse.Namespace):
    """
    Selects all referenced documents in given dataset from another dataset.
    Doesn't perform filtration of ids in documents.

    :param args: user arguments
    """

    referenced_documents = []
    with OADataset(args.documents, args.documents + ".index", workers=multiprocessing.cpu_count(),
                   hierarchy_as_dict=False) as dataset:
        dataset.transform = CitedIdsTransform()
        dataset.chunk_size = 100

        for cited in tqdm(dataset.iter_range(unordered=True), desc="Collecting references", total=len(dataset)):
            referenced_documents.extend(cited)

    referenced_documents = SortedSet(referenced_documents)

    with OADataset(args.references, args.references + ".index") as dataset:
        write_references(args.results, dataset, referenced_documents)


def sort_dataset(args: argparse.Namespace):
    """
    Sorts dataset by ids.

    :param args: user arguments
    """

    with OADataset(args.dataset, args.dataset + ".index", workers=multiprocessing.cpu_count()) as dataset, \
            open(args.results, "w") as res_f, \
            open(args.results + ".index", "w") as res_index_f:
        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        for doc_id in tqdm(sorted(dataset.mapping.keys()), desc="Writing sorted"):
            dataset.file.seek(dataset.mapping[doc_id])
            line = dataset.file.readline()
            index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
            print(line, file=res_f, end="")


class ExtractAbstractTransform:

    def __call__(self, document: Document) -> Optional[str]:
        abstract = document.abstract()
        if abstract is not None:
            record = {
                "id": document.id,
                "title": document.title,
                "doi": document.doi,
                "abstract": " ".join(t.text for t in abstract.text_content()),
            }
            return json_dumps(record)


def make_rw_train_val_test_splits(args: argparse.Namespace):
    """
    Makes train/val/test splits for related work section

    :param args: user arguments
    """
    assert args.test_prop + args.val_prop < 1.0
    papers_citing_multi_section = []

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    with OARelatedWork(args.dataset, args.dataset + ".index", workers=workers, lazy_hierarchy=True) as dataset, \
            OADataset(args.references, args.references + ".index") as references_dataset:
        dataset.transform = FilterTransform(NoTransform(),
                                            FractionOfCitedDocumentsWithMultiSectionContentFilter(references_dataset,
                                                                                                  1.0, 1.0),
                                            return_doc=False)
        dataset.chunk_size = 100

        for did, passing in tqdm(dataset, desc="Collecting citing multi section papers"):
            if passing:
                papers_citing_multi_section.append(did)

        print(f"Found {len(papers_citing_multi_section)} papers citing multi section papers", file=sys.stderr,
              flush=True)

        if args.fixed_seed:
            random.seed(42)

        random.shuffle(papers_citing_multi_section)

        val_size = int(len(papers_citing_multi_section) * args.val_prop)
        test_size = int(len(papers_citing_multi_section) * args.test_prop)

        print(f"{val_size} will be used for validation, {test_size} for test, "
              f"and {len(papers_citing_multi_section) - val_size - test_size} for train", file=sys.stderr, flush=True)

        val = set(papers_citing_multi_section[:val_size])
        test = set(papers_citing_multi_section[val_size:val_size + test_size])
        # the rest is train

        with open(args.train, "w") as train_f, open(args.train + ".index", "w") as train_f_index, \
                open(args.val, "w") as val_f, open(args.val + ".index", "w") as val_f_index, \
                open(args.test, "w") as test_f, open(args.test + ".index", "w") as test_f_index:

            train_writer = csv.DictWriter(train_f_index, fieldnames=["key", "file_line_offset"], delimiter="\t")
            train_writer.writeheader()
            val_writer = csv.DictWriter(val_f_index, fieldnames=["key", "file_line_offset"], delimiter="\t")
            val_writer.writeheader()
            test_writer = csv.DictWriter(test_f_index, fieldnames=["key", "file_line_offset"], delimiter="\t")
            test_writer.writeheader()

            # dataset file pointer to begining
            dataset.file.seek(0)
            for did, line in tqdm(zip(dataset.indices_2_id, dataset.file), desc="Writing train/val/test splits",
                                  total=len(dataset)):
                if did in val:
                    val_writer.writerow({"key": did, "file_line_offset": val_f.tell()})
                    print(line, file=val_f, end="")
                elif did in test:
                    test_writer.writerow({"key": did, "file_line_offset": test_f.tell()})
                    print(line, file=test_f, end="")
                else:
                    train_writer.writerow({"key": did, "file_line_offset": train_f.tell()})
                    print(line, file=train_f, end="")


class RepairS2ORCIdTransform:
    """
    Assigns s2orc ids to documents.
    """

    def __init__(self, mapping_2_s2orc_id: Sequence[int], missing_only: bool = False,
                 trans_indices: Optional[Sequence[int]] = None, index_offset_correction: int = 0):
        """
        :param mapping_2_s2orc_id: mapping from document INDEX to s2orc id
        :param missing_only: if True, only documents without s2orc id will be repaired
        :param trans_indices: translation from document index to index in the mapping
        :param index_offset_correction: correction of index every received index will be decreased by this value
            it is substracted before the translation
        """
        self.mapping_2_s2orc_id = mapping_2_s2orc_id
        self.missing_only = missing_only
        self.trans_indices = trans_indices
        self.index_offset_correction = index_offset_correction

    def __call__(self, doc: Document, doc_index: int) -> Tuple[int, str, bool]:
        prev = doc.s2orc_id
        use_i = doc_index - self.index_offset_correction
        if doc.s2orc_id is None or not self.missing_only:
            if self.trans_indices is not None:
                use_i = self.trans_indices[use_i]
                if use_i == -1:
                    raise ValueError(f"The document index {doc_index} is not in the translation index.")
            doc.s2orc_id = self.mapping_2_s2orc_id[use_i]
            if doc.s2orc_id == -1:
                doc.s2orc_id = None
        return doc.id, str(doc), prev != doc.s2orc_id


def create_title_database(args: argparse.Namespace):
    """
    Creates database with normalized titles.
    """
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    with SharedMemoryManager() as shared_records_list_manager:
        if args.scopus:
            records = ScopusPapersList.read_records(args.dataset)
        else:
            records = PapersList.read_records(args.dataset,
                                              workers=workers,
                                              record_type=PapersListRecordWithId,
                                              shared_list_for_records=shared_records_list_manager)

        db = PapersList.create_database(args.database)
        PapersList.insert_titles_to_database(db, records, workers=workers, batch_size=args.batch)

def merge_intervals(args: argparse.Namespace):
    """
    When processing by intervals you can use this function to marge them.
    :param args: user arguments
    """

    with open(args.results, "wb") as res_f, open(args.results + ".index", "w") as res_index_f:
        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        for file in tqdm(args.intervals, desc="Merging files"):
            with open(file + ".index") as f:
                index_reader = csv.DictReader(f, delimiter="\t")
                offset = res_f.tell()
                for record in index_reader:
                    record["file_line_offset"] = int(record["file_line_offset"]) + offset
                    index_writer.writerow(record)

            with open(file, "rb") as f:
                for block in iter(partial(f.read, 512 * 1024), b''):
                    res_f.write(block)


class FilterIdsTransform:
    def __init__(self, leave: Container[int]):
        self.leave = leave

    def __call__(self, doc: Document) -> Tuple[int, str]:
        doc.filter_citations(self.leave)
        return doc.id, str(doc)


def filter_missing_ids(args: argparse.Namespace):
    """
    Filters ids that are not in the dataset from citation and bibliography.
    """

    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    if workers == 0:
        workers = 1

    with OADataset(args.dataset, args.dataset + ".index", workers=workers) as dataset, \
            open(args.results, "w") as res_f, open(args.results + ".index", "w") as res_index_f:

        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        all_ids = SortedSet(dataset.mapping.keys())

        if workers > 0:
            all_ids.values = multiprocessing.Array(ctypes.c_int64, all_ids.values, lock=False)

        dataset.hierarchy_as_dict = True
        dataset.transform = FilterIdsTransform(all_ids)

        for doc_id, doc in tqdm(dataset.iter_range(0, None, unordered=True), total=len(dataset), desc="Filtering"):
            index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
            print(doc, file=res_f)

def convert_back_from_rw(args: argparse.Namespace):
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    if workers == 0:
        workers = 1

    with OARelatedWork(args.dataset, args.dataset + ".index", workers=workers) as dataset, \
            open(args.results, "w") as res_f, open(args.results + ".index", "w") as res_index_f:
        dataset.transform = lambda doc: (doc.id, str(doc.convert_back_to_doc()))

        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        for doc_id, doc in tqdm(dataset, desc="Converting"):
            index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
            print(doc, file=res_f)


def convert_back_to_rw(args: argparse.Namespace):
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    if workers == 0:
        workers = 1

    with OADataset(args.dataset, args.dataset + ".index", workers=workers) as dataset, \
            OARelatedWork(args.orig, args.orig + ".index", workers=workers) as orig, \
            open(args.results, "w") as res_f, open(args.results + ".index", "w") as res_index_f:

        orig.hierarchy_as_dict = True

        def transform(doc: Document) -> OARelatedWorkDocument:
            orig_doc = orig.get_by_id(doc.id)

            related_work_section = doc.hierarchy.get(orig_doc.related_work_orig_path)
            doc.hierarchy.remove(orig_doc.related_work_orig_path)

            doc_res = doc.asdict()
            doc_res["related_work"] = related_work_section.asdict()
            doc_res["related_work_orig_path"] = orig_doc.related_work_orig_path

            citations = set()
            for t_c in related_work_section.text_content():
                for c in t_c.citations:
                    if c.index is not None:
                        cited_id = doc.bibliography[c.index].id
                        if cited_id is not None:
                            citations.add(cited_id)

            doc_res["citations"] = sorted(citations)

            doc_res = OARelatedWorkDocument.from_dict(doc_res)

            return doc_res.id, str(doc_res)

        dataset.transform = transform

        index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
        index_writer.writeheader()

        for doc_id, doc in tqdm(dataset, desc="Converting"):
            index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
            print(doc, file=res_f)


class EnrichBibliographyUsingGraphTransform:
    """
    Uses citation graph to enrich bibliography.
    """

    def __init__(self, graph: Mapping[int, List[int]], search_in: str, graph_trans_offsets: Mapping[int, int],
                 id_field: str = "s2orc id", title_match_threshold: float = 0.75,
                 authors_match_threshold: float = 0.75, year_diff_threshold: int = 0):
        """
        :param graph: citation graph
        :param search_in: path to dataset to search in for referenced documents, it used to obtain title, authors, and year for the bibliography
        :param graph_trans_offsets: mapping from graph ids to offsets in search_in
        :param id_field: document id field that contains graph ids
        :param title_match_threshold: threshold for title similarity when finding bibliography entry
        :param authors_match_threshold: threshold for authors similarity when finding bibliography entry
        :param year_diff_threshold: the maximal difference in years for year match
        """
        self.graph = graph
        self.graph_trans_offsets = graph_trans_offsets
        self.search_in = search_in
        self.dataset_file = None
        self.id_field = id_field
        self.title_match_threshold = title_match_threshold
        self.authors_match_threshold = authors_match_threshold
        self.year_diff_threshold = year_diff_threshold

    def begin(self):
        self.dataset_file = open(self.search_in, "r")

    def end(self):
        self.dataset_file.close()

    def __call__(self, doc: Document) -> Tuple[int, int, int, str]:
        """
        Enriches bibliography of document.

        :param doc: document to enrich
        :return: tuple of (document id, number of new bib. entries, number of updated bib. entries, dccument with enriched bibliography)
        """

        # get references
        try:
            references = self.graph[getattr(doc, self.id_field)]
        except KeyError:
            return doc.id, 0, 0, str(doc)

        # get file offsets of references
        offsets_of_references = []
        for ref in references:
            try:
                offsets_of_references.append(self.graph_trans_offsets[ref])
            except KeyError:
                continue

        if len(offsets_of_references) == 0:
            return doc.id, 0, 0, str(doc)

        # read referenced records
        ref_records = []
        for offset in offsets_of_references:
            self.dataset_file.seek(offset)
            ref_records.append(PapersListRecordWithId.load(self.dataset_file.readline()))

        bib = Bibliography(doc.bibliography,
                           title_match_threshold=self.title_match_threshold,
                           authors_match_threshold=self.authors_match_threshold,
                           year_diff_threshold=self.year_diff_threshold)

        new_entries, updated_entries = 0, 0
        for r in ref_records:
            try:
                bib_index = bib.index(r.title, r.authors, r.year)
                bib_record = doc.bibliography[bib_index]

                # update existing entry
                if bib_record.id is None:
                    bib_record.id = r.id
                    updated_entries += 1

                if bib_record.year is None:
                    bib_record.year = r.year

                if len(bib_record.authors) == 0:
                    bib_record.authors = tuple(r.authors)

            except ValueError:
                # not found
                entry = BibEntry(r.id, r.title, r.year, tuple(r.authors))
                doc.bibliography.append(entry)
                bib.append(entry)
                new_entries += 1

        if new_entries > 0 or updated_entries > 0:
            doc.citations = sorted(set(b.id for b in doc.bibliography if b.id is not None))

        return doc.id, new_entries, updated_entries, str(doc)


def enrich_bibliography_from_citation_graph(args: argparse.Namespace):
    workers = args.workers
    if workers == -1:
        workers = multiprocessing.cpu_count()

    # obtain mapping from graph ids to OA ids
    search_in = args.search if args.search is not None else args.dataset
    with (OADataset(search_in, search_in + ".index", workers=workers) as search_in_dataset):
        search_in_dataset.transform = lambda doc, index: (getattr(doc, args.id), index)
        search_in_dataset.hierarchy_as_dict = True
        search_in_dataset.chunk_size = 10_000

        graph_ids_to_oa_offsets = []
        for graph_id, doc_index in tqdm(search_in_dataset.iter_range(0, None, unordered=True),
                                        desc="Collecting mapping from graph ids to OA ids",
                                        total=len(search_in_dataset)):
            if graph_id is not None:
                graph_ids_to_oa_offsets.append((graph_id, search_in_dataset.indices_2_offsets[doc_index]))

        graph_ids_to_oa_offsets = SortedMap(graph_ids_to_oa_offsets)
        if workers > 0:
            graph_ids_to_oa_offsets.keys_storage = multiprocessing.Array(ctypes.c_int64,
                                                                         graph_ids_to_oa_offsets.keys_storage,
                                                                         lock=False)
            graph_ids_to_oa_offsets.values_storage = multiprocessing.Array(ctypes.c_int64,
                                                                           graph_ids_to_oa_offsets.values_storage,
                                                                           lock=False)

        logging.log(logging.INFO, "Loading citation graph")

        with open(args.citation_graph, "r") as f:
            g = json.load(f)
            g = [(int(k), v) for k, v in g.items()]

        if workers > 0:
            g = SharedSortedMapOfSequencesOfIntegers(g)  # creates one big shared array for all sequences
        else:
            g = SortedMap(g)

        with open(args.result, "w") as res_f, open(args.result + ".index", "w") as res_index_f, \
                OADataset(args.dataset, args.dataset + ".index", workers=workers) as dataset:
            index_writer = csv.DictWriter(res_index_f, fieldnames=["key", "file_line_offset"], delimiter="\t")
            index_writer.writeheader()

            new_cite_cnt = 0
            updated_cite_cnt = 0
            dataset.transform = EnrichBibliographyUsingGraphTransform(
                graph=g,
                search_in=search_in,
                graph_trans_offsets=graph_ids_to_oa_offsets,
                id_field=args.id,
                title_match_threshold=args.title_match_threshold,
                authors_match_threshold=args.authors_match_threshold,
                year_diff_threshold=args.year_diff_threshold
            )
            from_index = args.from_i
            to_index = len(dataset) if args.to_i is None else args.to_i
            for doc_id, new_cites, updated_cites, doc in (p_bar := tqdm(dataset.iter_range(from_index, to_index),
                                                                        desc=f"Identifying | new bib. entries 0 | updated bib. entries 0",
                                                                        total=to_index - from_index)):
                new_cite_cnt += new_cites
                updated_cite_cnt += updated_cites
                index_writer.writerow({"key": doc_id, "file_line_offset": res_f.tell()})
                print(doc, file=res_f)
                p_bar.desc = f"Identifying | new bib. entries {new_cite_cnt} | updated bib. entries {updated_cite_cnt}"

            logging.log(logging.INFO, f"Found {new_cite_cnt} new citations")
            logging.log(logging.INFO, f"Updated {updated_cite_cnt} citations")

def kill_children():
    """
    Kills all subprocesses created by multiprocessing module.
    """

    for p in active_children():
        p.terminate()


def main():
    atexit.register(kill_children)
    logging.basicConfig(format='%(process)d: %(levelname)s : %(asctime)s : %(message)s', level=logging.INFO)
    args = ArgumentsManager.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
