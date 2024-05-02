# -*- coding: UTF-8 -*-
""""
Created on 04.02.22

:author:     Martin Dočekal
"""
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from io import StringIO
from typing import Optional, Tuple, Callable, Dict

from windpyutils.structures.span_set import SpanSet, SpanSetOverlapsEqRelation
from windpyutils.visual.text import print_histogram, print_buckets_histogram

from oapapers.datasets import OADataset
from oapapers.document import Document, OARelatedWorkDocument
from oapapers.citation_spans import CitationStyle, identify_citation_style, HARVARD_RE, VANCOUVER_PARENTHESIS_RE, \
    VANCOUVER_SQUARE_BRACKETS_RE, NORMALIZED_RE, identify_citation_style_of_doc
from oapapers.hierarchy import Hierarchy


class Stats(ABC):
    """
    Base class for statistics
    """

    @abstractmethod
    def update(self, other: "DocumentsStats"):
        """
        It will add values from other statistics to this statistics.

        :param other: Other statistics.
        """
        ...

    @abstractmethod
    def process(self, d: Document):
        """
        Add new document to statistics.

        :param d: the document for processing
        """
        ...

    @abstractmethod
    def __str__(self):
        """
        String representation of statistics.

        :return: String representation of statistics.
        """
        ...

    @staticmethod
    def histogram_to_str(title: str, hist: Dict, **kwargs) -> str:
        """
        Converts histogram to string.

        :param title: Title of histogram.
        :param hist: Histogram.
        :param kwargs: Additional arguments for print_histogram.
        :return: String representation of histogram.
        """
        res = title + "\n"

        if len(hist):
            s = StringIO()

            if isinstance(next(iter(hist.keys())), str):
                print_histogram(sorted(hist.items(), key=lambda x: x[1], reverse=True), file=s, **kwargs)
            else:
                if None in hist:
                    hist[-1] = hist[None]
                    del hist[None]
                print_buckets_histogram(hist, file=s, **kwargs)

            res += s.getvalue()
        else:
            res += "nothing\n"

        return res


class DocumentsStats(Stats):
    """
    Manages statistics about documents in a dataset.
    """

    HARVARD = re.compile(
        r"\b(?!(?:Although|Also)\b)(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:, *(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))",
        re.MULTILINE)
    VANCOUVER_PARENTHESIS = re.compile(
        r"(\(\s*(([0-9]+)|[0-9]+\s*-\s*[0-9]+|[0-9]+(\s*,\s*[0-9]+)+)\s*\))")  # couldn't use something like(\(([0-9]+(,|-|)\s*)+\)) as it causes catastrophic backtracking
    VANCOUVER_SQUARE_BRACKETS = re.compile(r"(\[\s*(([0-9]+)|[0-9]+\s*-\s*[0-9]+|[0-9]+(\s*,\s*[0-9]+)+)\s*\])")
    NORMALIZED = re.compile(r"(\[CITE:(UNK|[0-9]+)])")

    def __init__(self):
        """
        initialization of statistics
        """

        self.num_of_documents = 0
        self.num_of_documents_with_all_known_citations = 0  # a reference was found in a text but is not associated with id
        self.num_of_documents_with_non_referenced_citations = 0  # a citation from citations field is not referenced in text
        self.num_of_documents_with_non_empty_headlines = 0
        self.num_of_documents_with_suspicion_for_missing_reference = 0  # suspicion that a reference was not found in a text

        self.num_of_citation_spans = 0

        self.num_of_bibliography_entries = 0
        self.num_of_bibliography_entries_with_id = 0

        self.hist_of_years = defaultdict(int)
        self.hist_of_num_of_authors_per_document = defaultdict(int)
        self.hist_of_num_of_top_lvl_sections_per_document = defaultdict(int)
        self.hist_of_num_of_sections_per_document = defaultdict(int)
        self.hist_of_num_of_paragraphs_per_document = defaultdict(int)
        self.hist_of_num_of_sentences_per_document = defaultdict(
            int)  # formals are counted as sentences, but not paragraphs
        self.hist_of_num_of_words_per_document = defaultdict(int)
        self.hist_of_num_of_chars_per_document = defaultdict(int)
        self.hist_of_num_of_chars_per_document_with_headlines = defaultdict(int)
        self.hist_of_num_of_text_parts_per_top_lvl_section = defaultdict(int)
        self.hist_of_text_parts_len_chars = defaultdict(int)
        self.hist_of_num_of_citations_per_document = defaultdict(int)
        self.hist_of_num_of_non_plaintexts_per_document = defaultdict(int)
        self.hist_of_fields = defaultdict(int)
        self.hist_of_citations_styles = defaultdict(int)

    def update(self, other: "DocumentsStats"):
        """
        It will add values from other statistics to this statistics.

        :param other: Other statistics.
        """

        self.num_of_documents += other.num_of_documents
        self.num_of_documents_with_all_known_citations += other.num_of_documents_with_all_known_citations
        self.num_of_documents_with_non_referenced_citations += other.num_of_documents_with_non_referenced_citations
        self.num_of_documents_with_non_empty_headlines += other.num_of_documents_with_non_empty_headlines
        self.num_of_documents_with_suspicion_for_missing_reference += other.num_of_documents_with_suspicion_for_missing_reference

        self.num_of_citation_spans += other.num_of_citation_spans

        self.num_of_bibliography_entries += other.num_of_bibliography_entries
        self.num_of_bibliography_entries_with_id += other.num_of_bibliography_entries_with_id

        # update histograms by adding values from other histograms
        histograms_attributes = [a for a in dir(self) if a.startswith("hist_of_") and not callable(getattr(self, a))]

        for h in histograms_attributes:
            for k, v in getattr(other, h).items():
                getattr(self, h)[k] += v

    def process(self, d: Document, normalize_spans: bool = False):
        """
        Add new document to statistics.

        :param d: the document for processing
        :param normalize_spans: Converts cite/ref spans to unified form. Usefull when you want to get character related
            stats that are counting with normalized spans and it does not corrupts the statistics about citation styles.
        """

        self.num_of_documents += 1
        self.hist_of_years[d.year] += 1
        self.hist_of_num_of_authors_per_document[len(d.authors)] += 1

        self.num_of_citation_spans += sum(1 for _ in d.hierarchy.citation_spans())

        for s in d.hierarchy.content:
            self.hist_of_num_of_text_parts_per_top_lvl_section[len(list(s.text_content()))] += 1
        self.hist_of_num_of_citations_per_document[len(d.citations)] += 1
        self.hist_of_num_of_non_plaintexts_per_document[len(d.non_plaintext_content)] += 1

        if d.fields_of_study is not None:

            f_names = [f if isinstance(f, str) else f[0] for f in d.fields_of_study]
            for f in set(x.lower() for x in f_names):
                self.hist_of_fields[f] += 1

        all_known_citations = 1
        non_referenced_citations = set(d.citations)
        non_empty_headline = 1
        suspicion_for_missing_reference = 0

        cit_style = identify_citation_style_of_doc(d)
        self.hist_of_citations_styles[cit_style.name] += 1

        cit_re: Optional[re.Pattern] = {
            CitationStyle.UNKNOWN: None,
            CitationStyle.HARVARD: HARVARD_RE,
            CitationStyle.VANCOUVER_PARENTHESIS: VANCOUVER_PARENTHESIS_RE,
            CitationStyle.VANCOUVER_SQUARE_BRACKETS: VANCOUVER_SQUARE_BRACKETS_RE,
            CitationStyle.NORMALIZED: NORMALIZED_RE
        }[cit_style]

        if normalize_spans:
            # should be after cit styles and before char related stats
            d.normalize_spans()

        chars_cnt = 0
        words_cnt = 0
        sentences_cnt = 0
        paragraph_cnt = 0

        # let's get the document lengths in chars, sentences, paragraphs

        for p in d.hierarchy.sections(min_height=1):  # paragraphs
            if p.headline is None:
                paragraph_cnt += 1

        for s in d.hierarchy.text_content():  # sentences (also formulas)
            sentences_cnt += 1
            chars_cnt += len(s.text)
            words_cnt += len(s.text.split())
            self.hist_of_text_parts_len_chars[len(s.text)] += 1

        self.hist_of_num_of_chars_per_document[chars_cnt] += 1
        chars_cnt += sum(len(sub_hier.headline) for sub_hier in d.hierarchy.pre_order() if sub_hier.headline)
        self.hist_of_num_of_chars_per_document_with_headlines[chars_cnt] += 1

        self.hist_of_num_of_words_per_document[words_cnt] += 1
        self.hist_of_num_of_sentences_per_document[sentences_cnt] += 1
        self.hist_of_num_of_paragraphs_per_document[paragraph_cnt] += 1
        self.hist_of_num_of_sections_per_document[sum(1 for _ in d.hierarchy.sections())] += 1
        self.hist_of_num_of_top_lvl_sections_per_document[len(d.hierarchy.content)] += 1

        for section in d.hierarchy.content:
            for p in section.text_content():
                known_citation_spans_starts = []
                known_citation_spans_ends = []

                for c in p.citations:
                    known_citation_spans_starts.append(c.start)
                    known_citation_spans_ends.append(c.end)

                    if c.index is None or d.bibliography[c.index].id is None:
                        all_known_citations = 0

                    if c.index is not None:
                        try:
                            non_referenced_citations.remove(d.bibliography[c.index].id)
                        except KeyError:
                            pass

                if cit_re is not None and not suspicion_for_missing_reference:
                    # still need to check those

                    known_citation_spans = SpanSet(known_citation_spans_starts, known_citation_spans_ends, True,
                                                   SpanSetOverlapsEqRelation())

                    for match in cit_re.finditer(p.text):
                        if match.span() not in known_citation_spans:
                            suspicion_for_missing_reference = 1

            if not section.headline:
                non_empty_headline = 0

        self.num_of_documents_with_all_known_citations += all_known_citations
        self.num_of_documents_with_non_referenced_citations += min(1, len(non_referenced_citations))
        self.num_of_documents_with_non_empty_headlines += non_empty_headline
        self.num_of_documents_with_suspicion_for_missing_reference += suspicion_for_missing_reference

        self.num_of_bibliography_entries += len(d.bibliography)
        self.num_of_bibliography_entries_with_id += sum(1 for e in d.bibliography if e.id is not None)

    def __str__(self):
        res = "Number of documents\t" + str(self.num_of_documents) + "\n"
        res += "Number of documents with all known citations\t" + str(
            self.num_of_documents_with_all_known_citations) + "\n"
        res += "Number of documents with non referenced citations\t" + str(
            self.num_of_documents_with_non_referenced_citations) + "\n"
        res += "Number of documents with just non empty headlines\t" + str(
            self.num_of_documents_with_non_empty_headlines) + "\n"
        res += "Number of documents with suspicion for at least one missing reference\t" + str(
            self.num_of_documents_with_suspicion_for_missing_reference) + "\n"

        res += "Number of citation spans\t" + str(self.num_of_citation_spans) + "\n"

        res += "Number of bibliography entries\t" + str(self.num_of_bibliography_entries) + "\n"
        res += "Number of bibliography entries with id\t" + str(self.num_of_bibliography_entries_with_id) + "\n"

        for title, hist, args in [
            ("Histogram of years", self.hist_of_years, {}),
            ("Histogram of authors per document", self.hist_of_num_of_authors_per_document, {}),
            ("Histogram of chars per document", self.hist_of_num_of_chars_per_document,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of chars per document with headlines", self.hist_of_num_of_chars_per_document_with_headlines,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of words per document", self.hist_of_num_of_words_per_document,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of sentences per document", self.hist_of_num_of_sentences_per_document, {}),
            ("Histogram of paragraphs per document", self.hist_of_num_of_paragraphs_per_document, {}),
            ("Histogram of sections per document", self.hist_of_num_of_sections_per_document, {}),
            ("Histogram of top lvl sections per document", self.hist_of_num_of_top_lvl_sections_per_document, {}),
            ("Histogram of text parts per top lvl section", self.hist_of_num_of_text_parts_per_top_lvl_section, {}),
            ("Histogram of chars per text part", self.hist_of_text_parts_len_chars,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of citations per document", self.hist_of_num_of_citations_per_document, {}),
            ("Histogram of non-plaintexts per document", self.hist_of_num_of_non_plaintexts_per_document, {}),
            ("Histogram of fields of study", self.hist_of_fields, {}),
            ("Histogram of citation styles", self.hist_of_citations_styles, {}),
        ]:
            res += self.histogram_to_str(title, hist, **args)

        return res


class RelatedWorkStats(Stats):
    """
    Statistics about related work
    """

    def __init__(self, references: Optional[OADataset] = None):
        """
        Initialize statistics.

        :param references: dataset that is used to obtain cited documents
            You need to pass the references if you want to process documents
        """
        self.references = references

        self.num_of_targets = 0
        self.num_of_citations = 0
        self.target_vocabulary = set()
        self.abstracts_vocabulary = set()
        self.all_input_content_vocabulary = set()

        self.hist_of_num_of_sections_per_target = defaultdict(int)  # including root section
        self.hist_of_num_of_paragraphs_per_target = defaultdict(int)
        self.hist_of_num_of_sentences_per_target = defaultdict(
            int)  # formals are counted as sentences, but not paragraphs
        self.hist_of_num_of_words_per_target = defaultdict(int)
        self.hist_of_num_of_chars_per_target = defaultdict(int)

        self.hist_of_num_of_sections_per_input_abstracts = defaultdict(int) # including root section
        self.hist_of_num_of_paragraphs_per_input_abstracts = defaultdict(int)
        self.hist_of_num_of_sentences_per_input_abstracts = defaultdict(
            int)  # formals are counted as sentences, but not paragraphs, headlines are not counted as sentences
        self.hist_of_num_of_words_per_input_abstracts = defaultdict(int)  # headlines are not counted in
        self.hist_of_num_of_chars_per_input_abstracts = defaultdict(int)  # headlines are not counted in

        self.hist_of_num_of_sections_per_input_all_content = defaultdict(int) # including root section
        self.hist_of_num_of_paragraphs_per_input_all_content = defaultdict(int)
        self.hist_of_num_of_sentences_per_input_all_content = defaultdict(
            int)  # formals are counted as sentences, but not paragraphs
        self.hist_of_num_of_words_per_input_all_content = defaultdict(int)  # headlines are not counted in
        self.hist_of_num_of_chars_per_input_all_content = defaultdict(int)  # headlines are not counted in

    def update(self, other: "RelatedWorkStats"):
        """
        It will add values from other statistics to this statistics.

        :param other: Other statistics.
        """
        self.num_of_targets += other.num_of_targets
        self.num_of_citations += other.num_of_citations
        self.target_vocabulary |= other.target_vocabulary
        self.abstracts_vocabulary |= other.abstracts_vocabulary
        self.all_input_content_vocabulary |= other.all_input_content_vocabulary

        # update histograms by adding values from other histograms
        histograms_attributes = [a for a in dir(self) if
                                 a.startswith("hist_of_") and not callable(getattr(self, a))]

        for h in histograms_attributes:
            for k, v in getattr(other, h).items():
                getattr(self, h)[k] += v

    def process(self, d: OARelatedWorkDocument):
        """
        Add new document to statistics.

        :param d: the document for processing
        """

        self.num_of_targets += 1
        self.num_of_citations += len(d.citations)

        related_work_stats = self.hier_stats(d.related_work)
        self.target_vocabulary |= set(related_work_stats[0])
        self.hist_of_num_of_sections_per_target[related_work_stats[1]] += 1
        self.hist_of_num_of_paragraphs_per_target[related_work_stats[2]] += 1
        self.hist_of_num_of_sentences_per_target[related_work_stats[3]] += 1
        self.hist_of_num_of_words_per_target[related_work_stats[4]] += 1
        self.hist_of_num_of_chars_per_target[related_work_stats[5]] += 1

        abstracts_stats = self.references_hier_stats(d, lambda doc: doc.abstract)
        self.abstracts_vocabulary |= set(abstracts_stats[0])
        self.hist_of_num_of_sections_per_input_abstracts[abstracts_stats[1]] += 1
        self.hist_of_num_of_paragraphs_per_input_abstracts[abstracts_stats[2]] += 1
        self.hist_of_num_of_sentences_per_input_abstracts[abstracts_stats[3]] += 1
        self.hist_of_num_of_words_per_input_abstracts[abstracts_stats[4]] += 1
        self.hist_of_num_of_chars_per_input_abstracts[abstracts_stats[5]] += 1

        all_content_stats = self.references_hier_stats(d, lambda doc: doc.hierarchy)
        self.all_input_content_vocabulary |= set(all_content_stats[0])
        self.hist_of_num_of_sections_per_input_all_content[all_content_stats[1]] += 1
        self.hist_of_num_of_paragraphs_per_input_all_content[all_content_stats[2]] += 1
        self.hist_of_num_of_sentences_per_input_all_content[all_content_stats[3]] += 1
        self.hist_of_num_of_words_per_input_all_content[all_content_stats[4]] += 1
        self.hist_of_num_of_chars_per_input_all_content[all_content_stats[5]] += 1

    def references_hier_stats(self, d: OARelatedWorkDocument,
                              hier_selector: Callable[[Document], Hierarchy]) -> Tuple[
        set[str], int, int, int, int, int]:
        """
        Get aggregated statistics about hierarchies in references.

        :param d: the document which citations fields will be used to obtain references
        :param hier_selector: function which selects hierarchy from document
        :return: set of words, number of sections, number of paragraphs, number of sentences, number of words, number of chars
        """

        vocabulary, sections, paragraphs, sentences, words, chars = set(), 0, 0, 0, 0, 0

        for c in d.citations:
            doc = self.references.get_by_id(c)
            h = hier_selector(doc)
            assert h is not None, f"Document {doc.id} has no hierarchy"
            h_stats = self.hier_stats(h)
            vocabulary |= set(h_stats[0])

            sections += h_stats[1]
            paragraphs += h_stats[2]
            sentences += h_stats[3]
            words += h_stats[4]
            chars += h_stats[5]

        return vocabulary, sections, paragraphs, sentences, words, chars

    def hier_stats(self, hier: Hierarchy) -> Tuple[set[str], int, int, int, int, int]:
        """
        Get statistics about hierarchy.

        :param hier: hierarchy
        :return: set of words, number of sections, number of paragraphs, number of sentences, number of words, number of chars
        """
        vocabulary = set()
        sections = 0
        paragraphs = 0
        sentences = 0
        words_cnt = 0
        chars_cnt = 0

        for p in hier.sections(min_height=1):  # paragraphs
            if p.headline is None:
                paragraphs += 1

        for s in hier.text_content():  # sentences (also formulas)
            sentences += 1
            chars_cnt += len(s.text)
            words = s.text.split()
            words_cnt += len(words)
            vocabulary |= set(w.lower() for w in words)

        sections += sum(1 for _ in hier.sections()) + 1  # +1 for root section

        return vocabulary, sections, paragraphs, sentences, words_cnt, chars_cnt

    def __str__(self):
        res = f"Number of targets: {self.num_of_targets}\n"
        res += f"Number of citations: {self.num_of_citations}\n"
        res += f"Average number of citations per target: {self.num_of_citations / self.num_of_targets}\n"
        res += f"Target vocabulary size: {len(self.target_vocabulary)}\n"
        res += f"Abstracts vocabulary size: {len(self.abstracts_vocabulary)}\n"
        res += f"All input content vocabulary size: {len(self.all_input_content_vocabulary)}\n"

        names = ["sections", "paragraphs", "sentences", "words", "chars"]

        for per_what, hist_group in [
            ("targets", [
                self.hist_of_num_of_sections_per_target, self.hist_of_num_of_paragraphs_per_target,
                self.hist_of_num_of_sentences_per_target, self.hist_of_num_of_words_per_target,
                self.hist_of_num_of_chars_per_target,
            ]),
            ("input abstracts", [
                self.hist_of_num_of_sections_per_input_abstracts, self.hist_of_num_of_paragraphs_per_input_abstracts,
                self.hist_of_num_of_sentences_per_input_abstracts, self.hist_of_num_of_words_per_input_abstracts,
                self.hist_of_num_of_chars_per_input_abstracts,
            ]),
            ("all input content", [
                self.hist_of_num_of_sections_per_input_all_content, self.hist_of_num_of_paragraphs_per_input_all_content,
                self.hist_of_num_of_sentences_per_input_all_content, self.hist_of_num_of_words_per_input_all_content,
                self.hist_of_num_of_chars_per_input_all_content,
            ]),
        ]:
            for n, h in zip(names, hist_group):
                total = sum(v*cnt for v, cnt in h.items())
                res += f"Total number of {n} in {per_what}: {total}\n"
                res += f"Avg number of {n} in {per_what} per target: {total / self.num_of_targets}\n\n"

        for title, hist, args in [
            ("Histogram of number of sections per target", self.hist_of_num_of_sections_per_target, {}),
            ("Histogram of number of paragraphs per target", self.hist_of_num_of_paragraphs_per_target, {}),
            ("Histogram of number of sentences per target", self.hist_of_num_of_sentences_per_target, {}),
            ("Histogram of number of words per target", self.hist_of_num_of_words_per_target,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of number of chars per target", self.hist_of_num_of_chars_per_target,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of number of sections per input abstracts", self.hist_of_num_of_sections_per_input_abstracts, {}),
            ("Histogram of number of paragraphs per input abstracts", self.hist_of_num_of_paragraphs_per_input_abstracts, {}),
            ("Histogram of number of sentences per input abstracts", self.hist_of_num_of_sentences_per_input_abstracts, {}),
            ("Histogram of number of words per input abstracts", self.hist_of_num_of_words_per_input_abstracts,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of number of chars per input abstracts", self.hist_of_num_of_chars_per_input_abstracts,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of number of sections per input all content", self.hist_of_num_of_sections_per_input_all_content, {}),
            ("Histogram of number of paragraphs per input all content", self.hist_of_num_of_paragraphs_per_input_all_content, {}),
            ("Histogram of number of sentences per input all content", self.hist_of_num_of_sentences_per_input_all_content, {}),
            ("Histogram of number of words per input all content", self.hist_of_num_of_words_per_input_all_content,
             {"buckets": 40, "bucket_size_int": True}),
            ("Histogram of number of chars per input all content", self.hist_of_num_of_chars_per_input_all_content,
             {"buckets": 40, "bucket_size_int": True}),
        ]:
            res += self.histogram_to_str(title, hist, **args)

        return res


