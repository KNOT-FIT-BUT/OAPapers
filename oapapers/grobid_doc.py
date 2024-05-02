# -*- coding: UTF-8 -*-
""""
Created on 16.05.22

:author:     Martin Dočekal
"""
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import Lock
from typing import Union, Dict, Tuple, List
import dateutil.parser as dparser

import spacy
from windpyutils.generic import Batcher

import oapapers.papers_list as papers_list_module
from oapapers.bib_entry import BibEntry
from oapapers.hierarchy import Hierarchy, TextContent, RefSpan
from oapapers.text import SpanCollisionHandling, DeHyphenator, clean_title


class GROBIDMetaDoc:
    """
    Reader for meta information of GROBID xml representation of a document.

    :ivar title: document title
    :vartype title: Optional[str]
    :ivar year: year when a document was published
    :vartype year: Optional[int]
    :ivar authors: list of authors
    :vartype authors: Tuple[str]
    """

    PREFIX = "{http://www.tei-c.org/ns/1.0}"
    PREFIX_XML = "{http://www.w3.org/XML/1998/namespace}"
    MIN_YEAR = datetime.now().year - 200    # lest than that will be set to None
    MAX_YEAR = datetime.now().year   # more than that will be set to None

    def __init__(self, source: Union[str, ET.Element]):
        """
        Reads GROBID xml representation of a document.

        :param source: path to xml file
            or parsed xml
        """

        if isinstance(source, str):
            tree = ET.parse(source)
            source = tree.getroot()

        file_desc = source.find(f"{self.PREFIX}teiHeader/{self.PREFIX}fileDesc")

        self.title = None
        self.year = None
        self.authors = []

        try:
            self.title = file_desc.find(f"{self.PREFIX}titleStmt/{self.PREFIX}title").text
        except AttributeError:
            pass
        try:
            date = file_desc.find(f"{self.PREFIX}publicationStmt/{self.PREFIX}date")
            self.year = dparser.parse(date.attrib["when"] if "when" in date.attrib else date.text, fuzzy=True,
                                      ignoretz=True).year
            if not (self.MIN_YEAR <= self.year <= self.MAX_YEAR):
                self.year = None

        except (ValueError, AttributeError, dparser.ParserError, OverflowError):
            pass

        for author_name in file_desc.findall(f"{self.PREFIX}sourceDesc/{self.PREFIX}biblStruct/"
                                             f"{self.PREFIX}analytic/{self.PREFIX}author/{self.PREFIX}persName"):
            try:
                self.authors.append(self.parse_name(author_name))
            except TypeError:
                # there is probably no text in author's name
                pass

        self.authors = tuple(self.authors)

    @classmethod
    def parse_name(cls, pers_name: ET.Element) -> str:
        """
        Extracts person name from persName element.

        :param pers_name: persName element
        :return: parsed name
        """

        return " ".join(e.text
                        for e in pers_name if e.tag == f"{cls.PREFIX}forename" or e.tag == f"{cls.PREFIX}surname")


def remove_prefix(s: str, p: str) -> str:
    """
    Removes prefix from string if it has that prefix.

    :param s: string for prefix removal
    :param p: prefix that should be removed
    :return: string without given prefix
    """

    if s.startswith(p):
        return s[len(p):]
    return s


class GROBIDMetaDocWitBib(GROBIDMetaDoc):
    """
    Same as GROBIDMetaDoc but also parses bibliography.

    :ivar title: document title
    :vartype title: Optional[str]
    :ivar year: year when a document was published
    :vartype year: Optional[int]
    :ivar authors: list of authors
    :vartype authors: Tuple[str]
    :ivar bibliography: document's bibliography
    :vartype bibliography: Dict[str, BibEntry]
    """

    def __init__(self, source: Union[str, ET.Element]):
        """
        Reads GROBID xml representation of a document.

        :param source: path to xml file
            or parsed xml
        """

        if isinstance(source, str):
            tree = ET.parse(source)
            source = tree.getroot()

        super().__init__(source)
        self.bibliography = self.parse_bibliography(source)
        self.bibliography_list = list(self.bibliography)

    @classmethod
    def parse_bibliography(cls, source: ET.Element) -> Dict[str, BibEntry]:
        """
        Parses bibliography.
        Without matching it with ids.
        Leaves only bibliography with known authors and title.

        :param source: xml representation of document
        :return: dictionary representing bibliography
            bib id from document -> BibEntry
        """
        list_bibl = source.find(f"{cls.PREFIX}text/{cls.PREFIX}back/{cls.PREFIX}div/{cls.PREFIX}listBibl")
        if list_bibl is None:
            return {}

        res = {}
        for bib_ele in list_bibl:
            ele_analytic = bib_ele.find(f"{cls.PREFIX}analytic")
            if ele_analytic is None:
                ele_analytic = bib_ele.find(f"{cls.PREFIX}monogr")

            if ele_analytic is None:
                continue

            title = ele_analytic.find(f"{cls.PREFIX}title")
            if title is None:
                continue
            title = title.text
            if title is None:
                continue

            title = clean_title(title)
            if not title:
                continue

            authors = tuple(" ".join(x.strip() for x in a.itertext()).strip() for a in ele_analytic.findall(f"{cls.PREFIX}author"))
            if len(authors) == 0:
                continue

            year = bib_ele.find(f"{cls.PREFIX}monogr/{cls.PREFIX}imprint/{cls.PREFIX}date[@type='published']")
            if year is not None:
                try:
                    year = dparser.parse(year.attrib["when"]
                                         if "when" in year.attrib else year.text, fuzzy=True, ignoretz=True).year
                except (KeyError, ValueError, AttributeError, dparser.ParserError, OverflowError):
                    # invalid number or missing attribute
                    year = None

            try:
                bib_id = bib_ele.attrib[f"{cls.PREFIX_XML}id"]
            except KeyError:
                # missing attribute
                continue

            if bib_id is not None:
                res[bib_id] = BibEntry(None, title, year, authors)

        return res


class GROBIDDoc(GROBIDMetaDocWitBib):
    """
    Reader for GROBID xml representation of a document.
    Provides way (:func:`GROBIDDoc.match_bibliography`) to match bibliography with PaperList to obtain ids.

    :ivar doc_id: id of document
    :vartype doc_id: int
    :ivar title: document title
    :vartype title: Optional[str]
    :ivar year: year when a document was published
    :vartype year: Optional[int]
    :ivar authors: list of authors
    :vartype authors: Tuple[str]
    :ivar hierarchy: content in document's body in hierarchical format
    :vartype hierarchy: Hierarchy
    :ivar bibliography: document's bibliography
    :vartype bibliography: Dict[str, BibEntry]
    :ivar non_plaintext_content: non plaintext content in document's body (tables, figures)
        in form of tuple (type, description)
    :vartype non_plaintext_content: List[Tuple[str, str]
    """

    _spacy = None
    _spacy_stub = None
    lock = Lock()

    def __init__(self, source: Union[str, ET.Element], stub_init: bool = False):
        """
        Reads GROBID xml representation of a document.

        :param source: path to xml file
            or parsed xml
        :param stub_init: True activates stub mode which provides stub documents.
            Stub documents are documents that might not contain whole content, but just short "preview".
        """
        self.stub = stub_init

        if isinstance(source, str):
            tree = ET.parse(source)
            source = tree.getroot()

        super().__init__(source)

        try:
            self._prepare_spacy()
        except OSError:
            with self.lock:
                spacy.cli.download("en_core_web_sm")
                self._prepare_spacy()

        self.dehyphenator = DeHyphenator()

        self.non_plaintext_content, self._non_plaintext_content_mapping = \
            self._parse_non_plaintext_content(source.find(f"{self.PREFIX}text/{self.PREFIX}body"))
        self.hierarchy = self._parse_abstract_and_body(self.title, source)

    def _prepare_spacy(self):
        """
        Loads the spacy models and selects the components we need.
        It reuses spacy class instance if it exists.
        """
        if self.__class__._spacy is None:
            self.__class__._spacy = spacy.load("en_core_sci_sm")

        if self.__class__._spacy_stub is None:
            self.__class__._spacy_stub = spacy.blank("en")
            self.__class__._spacy_stub.add_pipe("sentencizer")

    @property
    def spacy(self) -> spacy.Language:
        """
        Returns spacy model instance.
        """
        if self.stub:
            return self.__class__._spacy_stub
        return self.__class__._spacy

    def match_bibliography(self, doc_id: int, papers_list: "papers_list_module.PapersList",
                           batch_size: int = 128):
        """
        Matches bibliography with referenced document ids.


        :param doc_id: id of document in paper_list to make sure that we are not referencing to itself.
            If by accident the searched reference is this document the reference will be marked as unknown (None).
        :param papers_list: Papers list for searching of references documents. It assumes that the index and id of a
            document in papers list and dataset are the same
        :param batch_size: Maximal number of samples in a batch when searching in paper lists.
        """

        bib_ref_ids = []
        bib_to_search = []
        for ref_id, bib in self.bibliography.items():
            bib_ref_ids.append(ref_id)
            bib_to_search.append(papers_list_module.PapersListRecord(bib.title, bib.year, bib.authors))

        if len(bib_to_search) > 0:
            search_res = []
            for b in Batcher(bib_to_search, batch_size):
                search_res.extend(papers_list.batch_search(b))

            for bib_ref_id, doc_in_dataset_id in zip(bib_ref_ids, search_res):
                doc_in_dataset_id = None if doc_in_dataset_id is None else int(doc_in_dataset_id)
                self.bibliography[bib_ref_id].id = None if doc_in_dataset_id == doc_id else doc_in_dataset_id
                # we don't want to reference to itself

    def _parse_non_plaintext_content(self, body: ET.Element) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """
        Parses non-plaintext content from a document's body.

        :param body: body of documet
        :return: non-plaintext content
            mapping from ref_key to non plaintext content index
        """

        content = []
        map_key_index = {}
        for element in body:
            if f"{self.PREFIX}figure" == element.tag:
                try:
                    map_key_index[element.attrib[f"{self.PREFIX_XML}id"]] = len(content)
                    desc = ""
                    head_ele = element.find(f"{self.PREFIX}head")
                    if head_ele is not None:
                        desc = " ".join(head_ele.itertext())

                    desc_ele = element.find(f"{self.PREFIX}figDesc")

                    if desc_ele is not None:
                        x = " ".join(desc_ele.itertext())
                        if len(x) > 0:
                            desc += f" {x}" if len(desc) > 0 else x
                    content.append((
                        "table" if "type" in element.attrib and element.attrib["type"] == "table" else "figure",
                        desc
                    ))
                except KeyError:
                    # missing attribute
                    continue
        return content, map_key_index

    def _parse_abstract_and_body(self, title: str, source: ET.Element) -> Hierarchy:
        """
        Parses document abstract and body.

        :param title: title of document
        :param source: the document
        :return: parsed document hierarchy
        :raise RuntimeError: invalid error
        """

        hier = Hierarchy(title, [])

        abstract = source.find(f"{self.PREFIX}teiHeader/{self.PREFIX}profileDesc/{self.PREFIX}abstract")
        if abstract is not None:
            h = self._proc_section(abstract)
            h.headline = "Abstract"
            hier.content.append(h)
            if self.stub and hier.has_text_content:
                return hier

        body = source.find(f"{self.PREFIX}text/{self.PREFIX}body")
        first_proper_section = False
        if body is not None:
            for element in source.find(f"{self.PREFIX}text/{self.PREFIX}body"):
                if f"{self.PREFIX}div" == element.tag:
                    # text section
                    sec = self._proc_section(element)

                    if not first_proper_section and sec.headline == "" and sec.height == 2:
                        # there some initial paragraphs that are not part of any section

                        for p in sec.content:
                            hier.content.append(p)
                    else:
                        first_proper_section = True
                        hier.content.append(sec)
                    if self.stub and hier.has_text_content:
                        # first is enough
                        return hier

        return hier

    def _proc_section(self, section: ET.Element) -> Hierarchy:
        """
        Processes single section from a grobid xml document.

        :param section: element representing section
        :return: hierarchy representing section
        """
        sec_hier = Hierarchy("", [])
        content = []
        # as we want to split sentences, and splitting sentences is faster when we do it for whole section than for
        # every paragraph separately, we firstly get all the text and then the sentences will be parsed and the
        # hierarchy will be created from them

        for div_ele in section:
            if f"{self.PREFIX}head" == div_ele.tag:
                # section headline
                if "n" in div_ele.attrib:
                    sec_hier.headline += div_ele.attrib["n"] + " "
                if div_ele.text is not None:
                    sec_hier.headline += div_ele.text

            elif f"{self.PREFIX}p" == div_ele.tag:
                # new paragraph
                content.append(self._proc_paragraph(div_ele))
            else:
                # I've seen another tag "formula" so this might be it. But I've written it for any other tag,
                # just in case.
                content.append(Hierarchy(remove_prefix(div_ele.tag, self.PREFIX), TextContent(
                    "" if div_ele.text is None else div_ele.text,
                    [],
                    []
                )))

        if not self.stub:
            # dehyphenate
            for i, p in enumerate(content):
                if not isinstance(p, Hierarchy):
                    t = TextContent(p[0], p[1], p[2])
                    spans, replace_with = self.dehyphenator.replacements(t.text)
                    t.replace_at(spans, replace_with, SpanCollisionHandling.SKIP)
                    content[i] = (t.text, t.citations, t.references)

        # get sentences
        sections_sentences = list(
            d.sents for d in self.spacy.pipe(p[0] for p in content if not isinstance(p, Hierarchy))
        )

        # fill the hier
        iter_sent = -1
        for c in content:
            if isinstance(c, Hierarchy):
                sec_hier.content.append(c)
                continue
            iter_sent += 1

            paragraph_hierarchy = Hierarchy(None, content=[])
            for sentence in sections_sentences[iter_sent]:
                sent_start, sent_end = sentence.start_char, sentence.end_char

                cite_spans = []
                ref_spans = []
                for store_to, filter_from in zip([cite_spans, ref_spans], [c[1], c[2]]):
                    for ref_span in filter_from:
                        if ref_span.start >= sent_start and ref_span.end <= sent_end:
                            ref_span.start = ref_span.start - sent_start
                            ref_span.end = ref_span.end - sent_start
                            store_to.append(ref_span)

                paragraph_hierarchy.content.append(
                    Hierarchy(None, TextContent(c[0][sent_start:sent_end], citations=cite_spans, references=ref_spans))
                )
            sec_hier.content.append(paragraph_hierarchy)
        return sec_hier

    def _proc_paragraph(self, p: ET.Element) -> Tuple[str, List[RefSpan], List[RefSpan]]:
        """
        Processes single paragraph. Handles citations and references.

        :param p: element representing paragraph
        :return: Plaintext in paragraph
            list of citation spans
            list of reference spans
        """

        whole_txt = "" if p.text is None else p.text
        cite_spans = []
        ref_spans = []

        for ref_span in p:
            start_offset = len(whole_txt)
            if ref_span.text is not None:
                whole_txt += ref_span.text
            end_offset = len(whole_txt)
            if ref_span.tail is not None:
                whole_txt += ref_span.tail

            if ref_span.text is not None and len(ref_span.text) > 0:
                try:
                    att_type = ref_span.attrib["type"]
                except KeyError:
                    continue
                try:
                    target = remove_prefix(ref_span.attrib["target"], "#")
                except KeyError:
                    target = None

                if att_type == "bibr":
                    try:
                        span = RefSpan(self.bibliography_list.index(target), start_offset, end_offset)
                    except ValueError:
                        # this bib was filtered out
                        span = RefSpan(None, start_offset, end_offset)

                    cite_spans.append(span)
                elif att_type == "table" or att_type == "figure":
                    try:
                        target_id = self._non_plaintext_content_mapping[target]
                    except KeyError:
                        # referencing to non-parsed non-plaintext content
                        target_id = None

                    span = RefSpan(target_id, start_offset, end_offset)
                    ref_spans.append(span)

        return whole_txt, cite_spans, ref_spans
