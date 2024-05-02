# -*- coding: UTF-8 -*-
""""
Created on 24.01.22
Contains representation of a document.

:author:     Martin Dočekal
"""
import copy
import itertools
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional, Generator, Sequence, Callable, Union, Container, \
    Mapping

from windpyutils.generic import BatcherIter

from oapapers.bib_entry import BibEntry
from oapapers.cython.normalization import normalize_and_tokenize_string
from oapapers.hierarchy import Hierarchy, TextContent, RefSpan
from oapapers.myjson import json_dumps
from oapapers.papers_list import PapersList, PapersListRecord

ABSTRACT_REGEX = re.compile(r"^((^|\s|\()((((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|"
                            r"[0-9]+|[a-z])(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+"
                            r"|[a-z]))*\.?)($|\s|\)))?\s*abstract\s*$", re.IGNORECASE)


@dataclass
class Document:
    """
    Representation of a document.
    """

    __slots__ = (
        "id", "s2orc_id", "mag_id", "doi", "title", "authors", "year", "fields_of_study", "citations", "hierarchy",
        "bibliography", "non_plaintext_content", "uncategorized_fields"
    )

    id: int  #: identifier of a document
    s2orc_id: Optional[int]  #: s2orc identifier of a document
    mag_id: Optional[int]  #: mag identifier of a document
    doi: Optional[str]  #: Digital Object Identifier
    title: str  #: the title of a document
    authors: List[str]  #: authors of given document
    year: Optional[int]  #: year when the document was published
    fields_of_study: List[Union[str, Tuple[str, float]]]  # fields of study, such as "Mathematics", "Physics" ...
    citations: List[int]  #: identifiers of referenced documents
    hierarchy: Union[Hierarchy, Dict]  #: hierarchical representation of a document
    bibliography: List[BibEntry]
    non_plaintext_content: List[
        Tuple[str, str]]  #: list of tuple in form of content type (figure, table) and description
    uncategorized_fields: Dict[str, Any]  #: uncategorized fields

    def asdict(self) -> Dict[str, Any]:
        """
        Creates dictionary representation of this document.

        :return: dictionary representation of this document
        """
        # dataclasses.asdict is too slow
        res = {
            "id": self.id,
            "s2orc_id": self.s2orc_id,
            "mag_id": self.mag_id,
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "fields_of_study": self.fields_of_study,
            "citations": self.citations,
            "hierarchy": self.hierarchy.asdict() if isinstance(self.hierarchy, Hierarchy) else self.hierarchy,
            "bibliography": [b.asdict() for b in self.bibliography],
            "non_plaintext_content": self.non_plaintext_content
        }
        res.update(self.uncategorized_fields)
        return res

    def __str__(self):
        return json_dumps(self.asdict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any],
                  hierarchy_as_dict: bool = False,
                  lazy_hierarchy: bool = False) -> "Document":
        """
        Creates this data class from dictionary.

        :param d: the dictionary used of instantiation
        :param hierarchy_as_dict: if True the hierarchy is left as dictionary, otherwise it is converted to Hierarchy
        :param lazy_hierarchy: if True, the hier content is not parsed and instead a lazy hierarchy is created
        :return: create document
        """

        # Impl. detail. I've firstly used from_dict(data_class=Document, data=json_record), but it was too slow.

        if hierarchy_as_dict:
            hierarchy = d["hierarchy"]
        else:
            hierarchy = Hierarchy.from_dict(d["hierarchy"], lazy=lazy_hierarchy)

        return cls(
            id=d["id"],
            s2orc_id=d["s2orc_id"] if "s2orc_id" in d else None,    # backward compatibility
            mag_id=d["mag_id"],
            doi=d["doi"],
            title=d["title"],
            authors=d["authors"],
            year=d["year"],
            fields_of_study=[x if isinstance(x, str) else tuple(x) for x in d["fields_of_study"]] if d["fields_of_study"] is not None else [],
            citations=d["citations"],
            hierarchy=hierarchy,
            non_plaintext_content=[tuple(x) for x in d["non_plaintext_content"]],
            bibliography=[BibEntry.from_dict(b) for b in d["bibliography"]],
            uncategorized_fields={k: v for k, v in d.items() if k not in {"id", "s2orc_id", "mag_id", "doi", "title",
                                                                         "authors", "year", "fields_of_study",
                                                                         "citations", "hierarchy", "bibliography",
                                                                         "non_plaintext_content"}}
        )

    def citation_spans(self) -> Generator[RefSpan, None, None]:
        """
        Generation of all citations spans in hierarchy.
        It iterates text content from the left most one to the right most one, but it does not guarantee left to right
        positioning of citation inside a single text content.

        :return: generator of citation spans
        """
        yield from self.hierarchy.citation_spans()

    def filter_citations(self, leave: Container[int]):
        """
        Removes all citation and sets None to id member in bibliographies that are not in leave.

        :param leave: ids that can be left
        """
        self.citations = [i for i in self.citations if i in leave]

        for bib_entry in self.bibliography:
            if bib_entry.id not in leave:
                bib_entry.id = None

    def text_content(self, parent_condition: Optional[Callable[["Hierarchy"], bool]] = None) \
            -> Generator[TextContent, None, None]:
        """
        Generates text contents in hierarchy from the left most one to the right most one.

        :param parent_condition: optional filter that allows to set up condition on parent of text content
            True passes
        :return: generator of TextContent
        """
        yield from self.hierarchy.text_content(parent_condition)

    def fix_span_boundaries(self):
        """
        Fixed (citation, reference) span boundaries.

        There were observed spans missing associated brackets:
            This bactrophorine ( Fig. 1 ) is only the third described species of its genus. Two others have been
            described from Northern Honduras (Rehn 1938 )
            "cite_spans": [{"start": 134, "end": 144, "text": "(Rehn 1938", "ref_id": "BIBREF4"}],
            "ref_spans": [{"start": 21, "end": 27, "text": "Fig. 1", "ref_id": "FIGREF0"}]}

        """
        start_brackets = {"(", "[", "{", "<"}
        end_brackets = {")", "]", "}", ">"}
        for t_c in self.text_content():
            text = t_c.text

            for s in itertools.chain(t_c.citations, t_c.references):
                if text[s.start] not in start_brackets:  # fix start
                    for i in range(s.start - 1, -1, -1):
                        if text[i].isspace() or text[i] in start_brackets:
                            if text[i] in start_brackets:
                                s.start = i
                                break
                        else:
                            break

                if text[s.end] not in end_brackets:  # fix end
                    for i in range(s.end + 1, len(text)):
                        if text[i].isspace() or text[i] in end_brackets:
                            if text[i] in end_brackets:
                                s.end = i + 1
                        else:
                            break

    def normalize_citation_spans(self):
        """
        Converts citations to unified form.
        """

        self.normalize_spans(False)

    def normalize_reference_spans(self):
        """
        Convert reference spans referencing to non-plaintext content to unified form.
        """
        self.normalize_spans(True)

    def normalize_spans(self, is_ref: Optional[bool] = None):
        """
        Converts spans to unified form.

        :param is_ref:  True means that we are normalizing reference spans. False means citation spans.
            None means both.
        """

        for t_c in self.text_content():
            offset_diff = 0
            for span_type, span in sorted(
                    itertools.chain([(False, c) for c in t_c.citations], [(True, r) for r in t_c.references]),
                    # add types
                    key=lambda x: x[1].start):
                span.start += offset_diff
                span.end += offset_diff

                if is_ref is None or is_ref == span_type:
                    try:
                        new_text = self.normalize_single_span(t_c.text, span, span_type)
                        offset_diff += len(new_text) - len(t_c.text)
                        t_c.text = new_text
                    except Exception as e:
                        print(f"I'm not able to normalize span {span} of type {span_type} in {self.id}. "
                              f"Skipping due to: {e}", flush=True, file=sys.stderr)

                        search_in = t_c.references if span_type else t_c.citations
                        search_in.remove(span)

    @staticmethod
    def normalize_single_span(text: str, span: RefSpan, is_ref: bool) -> str:
        """
        Replace citation/reference span with normalized form.

        :param text: text containing given span
        :param span: span that should be normalized in input text
            May alternate start and end.
        :param is_ref: True means that we are normalizing reference span. False means citation span.
            They differ in their form.
            Reference span is:
             [REF:{span.id}]
            Citation span is
                [CITE:{'UNK' if span.index is None else span.index}]
        :return: Original input text with normalized span
        """

        while span.start > 0 and text[span.start - 1].isspace():
            span.start -= 1
        while span.end < len(text) and text[span.end].isspace():
            span.end += 1

        new_span = ""
        left_whitespace = 0
        if span.start > 0:
            new_span = " "
            left_whitespace = 1

        if is_ref:
            new_span += f"[REF:{span.index}]"
        else:
            new_span += f"[CITE:{'UNK' if span.index is None else span.index}]"

        right_whitespace = 0
        if span.end < len(text):
            new_span += " "
            right_whitespace = 1

        new_txt = text[:span.start] + new_span + text[span.end:]
        span.end = span.start + len(new_span) - right_whitespace
        span.start += left_whitespace

        return new_txt

    def translate_ids(self, m: Mapping[int, int]):
        """
        Using mapping m (old -> new) translates document id and referenced documents ids.

        :param m: mapping from olds ids to new ones
        """

        self.id = m[self.id]
        self.citations = sorted(set(m[c] for c in self.citations))

        for bib_entry in self.bibliography:
            if bib_entry.id is not None:
                bib_entry.id = m[bib_entry.id]

    def deduplicate_authors(self):
        """
        Performs deduplication of authors. The first one in a list is used when duplicates are found.
        Uses their normalized version.
        """
        norm_a = {frozenset(normalize_and_tokenize_string(a)) for a in self.authors}
        authors = []
        for a in self.authors:
            norm = frozenset(normalize_and_tokenize_string(a))
            if norm in norm_a:
                authors.append(a)
                norm_a.remove(norm)
        self.authors = authors

    def identify_bibliography(self, papers_list: PapersList, indices_2_id: Sequence[int], batch_size: int = 128) -> int:
        """
        Tries to identify bibliography using provided list of papers.

        :param papers_list: list of papers used for searching unknown referenced literature
        :param indices_2_id: mapping from paper list index to document id
        :param batch_size: maximal batch size used for searching references
        :return: number of newly found cited documents
        """

        return self.identify_bibliography_docs([self], papers_list, indices_2_id, batch_size)[0]

    @classmethod
    def identify_bibliography_docs(cls, docs: Sequence["Document"], papers_list: PapersList, indices_2_id: Sequence[int],
                                   batch_size: int = 128) -> List[int]:
        """
        Identifies bibliographies of provided documents.

        WARNING: It is using PapersList.filter_search_nearest_results instead of object's method.

        :param docs: documents to identify bibliographies for
        :param papers_list: list of papers used for searching unknown referenced literature
        :param indices_2_id: mapping from paper list index to document id
        :param batch_size: maximal batch size used for searching references
        :return: number of newly found cited documents for each document
        """

        to_search = {}

        for i, doc in enumerate(docs):
            for target, bib_entry in enumerate(doc.bibliography):
                if bib_entry.id is None:
                    to_search[(i, target)] = PapersListRecord(bib_entry.title, bib_entry.year, bib_entry.authors)

        new_ones_cnt = [0] * len(docs)

        if len(to_search) > 0:
            search_res = {}
            for bib_targets, records in BatcherIter((to_search.keys(), to_search.values()), batch_size):
                # find nearest and then filter them in this process as we do not want to exhaust the manager process
                filtered_res = papers_list.batch_search(records)

                # try to search unknown ones
                for target, res in zip(bib_targets, filtered_res):
                    if res is None:
                        continue
                    act_id = indices_2_id[res]
                    if act_id != docs[target[0]].id:
                        search_res[target] = act_id

            if len(search_res) > 0:
                bib_ids = [set(b.id for b in doc.bibliography if b.id is not None) for doc in docs]
                bib_ids_orig_cnt = [len(b) for b in bib_ids]

                for target, identified_id in search_res.items():
                    doc = docs[target[0]]
                    if identified_id not in bib_ids[target[0]]:
                        doc.bibliography[target[1]].id = identified_id
                        bib_ids[target[0]].add(identified_id)

                for i, doc in enumerate(docs):
                    doc.citations = sorted(set(b.id for b in doc.bibliography if b.id is not None))
                    new_ones_cnt[i] = len(doc.citations) - bib_ids_orig_cnt[i]

        return new_ones_cnt

    @property
    def abstract(self) -> Optional[Hierarchy]:
        """
        Tries to extract abstract from document hierarchy.

        :return: abstract or None if not found
        """
        abstract_section = self.hierarchy.get_part(ABSTRACT_REGEX, max_h=1, min_depth=1, max_depth=1)

        if abstract_section and any(len(t_c.text) > 0 for t_c in abstract_section[0].text_content()):
            return abstract_section[0]


@dataclass
class OARelatedWorkDocument(Document):
    """
    Data class for target documents in OARelatedWork dataset.
    """
    __slots__ = ("related_work", "related_work_orig_path")
    related_work: Hierarchy
    related_work_orig_path: Tuple[int, ...]

    def asdict(self) -> Dict[str, Any]:
        """
        Creates dictionary representation of this document.

        :return: dictionary representation of this document
        """
        # dataclasses.asdict is too slow

        res = {
            "id": self.id,
            "s2orc_id": self.s2orc_id,
            "mag_id": self.mag_id,
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "fields_of_study": self.fields_of_study,
            "citations": self.citations,
            "hierarchy": self.hierarchy.asdict() if isinstance(self.hierarchy, Hierarchy) else self.hierarchy,
            "related_work": self.related_work.asdict(),
            "related_work_orig_path": self.related_work_orig_path,
            "bibliography": [b.asdict() for b in self.bibliography],
            "non_plaintext_content": self.non_plaintext_content
        }
        res.update(self.uncategorized_fields)
        return res

    @classmethod
    def from_dict(cls, d: Dict[str, Any], hierarchy_as_dict: bool = False,
                  lazy_hierarchy: bool = False) -> "OARelatedWorkDocument":
        """
        Creates this data class from dictionary.

        :param d: the dictionary used of instantiation
        :param hierarchy_as_dict: if True the hierarchy is left as dictionary, otherwise it is converted to Hierarchy
        :param lazy_hierarchy: if True, the hier content is not parsed and instead a lazy hierarchy is created
        :return: create document
        """

        # Impl. detail. I've firstly used from_dict(data_class=Document, data=json_record), but it was too slow.

        if hierarchy_as_dict:
            hierarchy = d["hierarchy"]
        else:
            hierarchy = Hierarchy.from_dict(d["hierarchy"], lazy_hierarchy)

        return cls(
            id=d["id"],
            s2orc_id=d["s2orc_id"] if "s2orc_id" in d else None,  # backward compatibility
            mag_id=d["mag_id"],
            doi=d["doi"],
            title=d["title"],
            authors=d["authors"],
            year=d["year"],
            fields_of_study=[x if isinstance(x, str) else tuple(x) for x in d["fields_of_study"]] if d["fields_of_study"] is not None else [],
            citations=d["citations"],
            hierarchy=hierarchy,
            related_work=Hierarchy.from_dict(d["related_work"]),
            related_work_orig_path=tuple(d["related_work_orig_path"]),
            non_plaintext_content=[tuple(x) for x in d["non_plaintext_content"]],
            bibliography=[BibEntry.from_dict(b) for b in d["bibliography"]],
            uncategorized_fields={k: v for k, v in d.items() if k not in {"id", "s2orc_id", "mag_id", "doi", "title",
                                                                         "authors", "year", "fields_of_study",
                                                                         "citations", "hierarchy", "related_work",
                                                                         "related_work_orig_path", "bibliography",
                                                                         "non_plaintext_content"}}
        )

    @property
    def abstract(self) -> Hierarchy:
        return self.hierarchy.content[0]

    def convert_back_to_doc(self) -> Document:
        """
        Converts this document back to Document.
        The hierarchy must not be dict.

        :return: converted document
        """

        hierarchy = copy.deepcopy(self.hierarchy)
        hierarchy.insert(self.related_work_orig_path, self.related_work)

        citations = sorted(set(b.id for b in self.bibliography if b.id is not None))

        return Document(
            id=self.id,
            s2orc_id=self.s2orc_id,
            mag_id=self.mag_id,
            doi=self.doi,
            title=self.title,
            authors=self.authors,
            year=self.year,
            fields_of_study=self.fields_of_study,
            citations=citations,
            hierarchy=hierarchy,
            bibliography=self.bibliography,
            non_plaintext_content=self.non_plaintext_content,
            uncategorized_fields=self.uncategorized_fields
        )




