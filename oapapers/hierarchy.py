# -*- coding: UTF-8 -*-
""""
Created on 17.08.22

:author:     Martin Dočekal
"""
import enum
import itertools
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, List, Dict, Any, Generator, Pattern, Callable, Tuple, Sequence, MutableSequence, \
    AbstractSet

from windpyutils.generic import roman_2_int, int_2_roman
from windpyutils.structures.maps import ImmutIntervalMap

from oapapers.myjson import json_dumps
from oapapers.text import SpanCollisionHandling, replace_at


@dataclass
class RefSpan:
    """
    Referencing span
    """

    __slots__ = ("index", "start", "end")

    index: Optional[int]
    """
    identifier of referenced entity
    it should be index to non_plaintext_content or bibliography
    null means that the source is unknown
    """
    start: int  #: span start offset
    end: int  #: span end offset (not inclusive)

    def asdict(self) -> Dict[str, Any]:
        """
        Converts this data class to dictionary.

        :return: dictionary representation of this data class
        """
        # dataclasses.asdict is too slow
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RefSpan":
        """
        Creates this data class from dictionary.

        :param d: the dictionary used of instantiation
        :return: create RefSpan
        """
        return RefSpan(
            index=d["index"],
            start=d["start"],
            end=d["end"]
        )


@dataclass
class TextContent:
    """
    Text content of a document.
    """

    __slots__ = ("text", "citations", "references")

    text: str  #: text content of a part
    citations: List[RefSpan]  #: list of citation
    references: List[RefSpan]  #: list of references to images, graphs, tables, ...

    def asdict(self):
        """
        Converts this data class to dictionary.

        :return: dictionary representation of this data class
        """
        # dataclasses.asdict is too slow
        return {
            "text": self.text,
            "citations": [c.asdict() for c in self.citations],
            "references": [r.asdict() for r in self.references]
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TextContent":
        """
        Creates this data class from dictionary.

        :param d: the dictionary used of instantiation
        :return: create TextContent
        """
        return TextContent(
            text=d["text"],
            citations=[RefSpan.from_dict(c) for c in d["citations"]],
            references=[RefSpan.from_dict(r) for r in d["references"]]
        )

    def replace_at(self, spans: Sequence[Tuple[int, int]], replace_with: Sequence[str],
                   collisions: SpanCollisionHandling = SpanCollisionHandling.RAISE):
        """
        Replaces text at given spans with given text.
        It also handles citation and references.

        :param spans: starts and ends of disjunctive spans to be replaced
        :param replace_with: new text on given spans
        :param collisions: defines how to handle collisions with citations and references
        :raise ValueError: if the number of spans and replace_with is not the same
        :raise ValueError: if there are spans that collide with citations or references and collisions is set to RAISE
        """

        text, updated_spans = replace_at(self.text, spans, replace_with,
                                         [
                                             [(s.start, s.end) for s in self.citations],
                                             [(s.start, s.end) for s in self.references]
                                         ], collisions)

        self.text = text

        new_citations = []
        for i, s in enumerate(self.citations):
            if updated_spans[0][i] is not None:
                s.start, s.end = updated_spans[0][i]
                new_citations.append(s)
        self.citations = new_citations

        new_references = []
        for i, s in enumerate(self.references):
            if updated_spans[1][i] is not None:
                s.start, s.end = updated_spans[1][i]
                new_references.append(s)
        self.references = new_references


class PositionFromEnd:
    """
    Serial number that is used for assigning position from an unknown end.
    Is used for just saying that this suppose to be the last section, the second last section etc.
    """

    def __init__(self, pos_from_end: int):
        """
        initialization

        :param pos_from_end: defines the position from end
        """
        self.pos_from_end = pos_from_end

    def __str__(self):
        return str(self.pos_from_end)

    def __repr__(self):
        return f"PositionFromEnd({self.pos_from_end})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.pos_from_end == other.pos_from_end
        return False

    def __lt__(self, other):
        """
        The position from end is greater than anything, but the other position from end, then it is smaller when
        it has grater position from end than the other.
        """

        if isinstance(other, self.__class__):
            return self.pos_from_end > other.pos_from_end
        return False

    def __gt__(self, other):
        """
        The position from end is greater than anything, but the other position from end, then it is greater when
        it has smaller position from end than the other.
        """
        return not (self < other) and self != other

    def __sub__(self, other):
        """
        Subtracting anything else than position from end returns inf
        """
        if isinstance(other, self.__class__):
            return other.pos_from_end - self.pos_from_end  # it's the other way around because we are going backwards
        return math.inf


class SerialNumberFormat(Enum):
    """
    Defines all formats of serial numbers.
    """

    UNKNOWN = enum.auto()  # e.g. for cases when it was induced from text such as in case of Introduction -> 1
    ARABIC = enum.auto()  # ordinary numbers 1 2 3
    ROMAN = enum.auto()  # good old fashioned roman numbers: I, V ...
    LATIN = enum.auto()  # latin character: a, b, c


@dataclass
class Hierarchy:
    """
    Representation of a document in form of hierarchy.

    Example of a document with two sections where each section has 2 paragraphs with 2 sentences

    headline    Title of a document
    content
        headline Section 1 headline
        content
            headline None
            content
                headline None
                content First sentence of first paragraph in first section.

                headline None
                content Second sentence of first paragraph in first section.

            headline None
            content
                headline None
                content First sentence of second paragraph in first section.

                headline None
                content Second sentence of second paragraph in first section.

        headline Section 2 headline
        content
            headline None
            content
                headline None
                content First sentence of first paragraph in second section.

                headline None
                content Second sentence of first paragraph in second section.

            headline None
            content
                headline None
                content First sentence of second paragraph in second section.

                headline None
                content Second sentence of second paragraph in second section.
    """
    __slots__ = ("headline", "content")

    headline: Optional[str]  #: headline of a part
    content: Union[List["Hierarchy"], TextContent]  # content of a part could contain another parts or a text content

    SECTION_NUMBER_REGEX = re.compile(r"(^|\s|\()"
                                      r"(?P<serial>(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+|[a-z])"
                                      r"("
                                      r"(\s+\.\s+(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+))|"  # not |[a-z] to prevent matching text part
                                      r"(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+|[a-z]))"
                                      r")*\.?)"
                                      r"($|\s|\))",
                                      re.IGNORECASE | re.ASCII)
    SECTION_NUMBER_REGEX_STRICT = re.compile(r"(^|(part|chapter|section)\W+|^\W+)"
                                             r"(?P<serial>(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+|[a-z])"
                                             r"("
                                             r"(\s+\.\s+(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+))|"  # not |[a-z] to prevent matching text part
                                             r"(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+|[A-Z]))"
                                             r")*\.?(?=(\W|$)))",
                                             re.IGNORECASE | re.ASCII)
    SECTION_NUMBER_REGEX_STRICT_WITHOUT_ABC = re.compile(r"(^|(part|chapter|section)\W+|^\W+)"
                                                         r"(?P<serial>(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+)"
                                                         r"("
                                                         r"(\s+\.\s+(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+))|"
                                                         r"(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+))"
                                                         r")*\.?(?=(\W|$)))",
                                                         re.IGNORECASE | re.ASCII)

    INTRO_HEADLINES_REGEX = re.compile(r"\W*\s*introduction\s*\W*", re.IGNORECASE | re.ASCII)
    INTRO_HEADLINES_STICKY_REGEX = re.compile(r"^[1I]?\W*\s*introduction\s*.*$", re.IGNORECASE | re.ASCII)
    MIDDLE_HEADLINES_REGEX = re.compile(r"(part|chapter|section)\W+"
                                        r"(?P<serial>(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+|[a-z])"
                                        r"("
                                        r"(\s+\.\s+(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+))|"  # not |[a-z] to prevent matching text part
                                        r"(\.(((X{1,3}(IX|IV|V?I{0,3}))|((IX|IV|I{1,3}|VI{0,3})))|[0-9]+|[A-Z]))"
                                        r")*\.?(?=(\W|$)))", re.IGNORECASE | re.ASCII)
    INTRO_AND_MIDDLE_HEADLINES_STICKY_REGEX = re.compile(
        rf"(({INTRO_HEADLINES_STICKY_REGEX.pattern})|({MIDDLE_HEADLINES_REGEX.pattern}))", re.IGNORECASE | re.ASCII)
    CONCLUSION_HEADLINES_REGEX = re.compile(r"\W*\s*(conclusion|discussion|conclusions)\W*\s*",
                                            re.IGNORECASE | re.ASCII)
    CONCLUSION_HEADLINES_STICKY_REGEX = re.compile(
        r"^(.*\s)?(?P<conclusion>(conclusion|discussion|conclusions))(\s.+)?$", re.IGNORECASE | re.ASCII)

    APPENDIX_HEADLINES_STICKY_REGEX = re.compile(r"^(.*\s)?(?P<appendix>(appendix))(\s.+)?$", re.IGNORECASE | re.ASCII)
    ACKNOWLEDGEMENT_HEADLINES_STICKY_REGEX = re.compile(r"^(.*\s)?(?P<acknowledgement>(acknowledgement))(\s.+)?$",
                                                        re.IGNORECASE | re.ASCII)
    REFERENCES_HEADLINES_STICKY_REGEX = re.compile(r"^(.*\s)?(?P<references>(references))(\s.+)?$",
                                                   re.IGNORECASE | re.ASCII)

    ENDINGS_STICKY_REGEX = re.compile(
        rf"(({CONCLUSION_HEADLINES_STICKY_REGEX.pattern})|({APPENDIX_HEADLINES_STICKY_REGEX.pattern})|"
        rf"({ACKNOWLEDGEMENT_HEADLINES_STICKY_REGEX.pattern})|({REFERENCES_HEADLINES_STICKY_REGEX.pattern}))",
        re.IGNORECASE | re.ASCII)

    ALL_HEADLINES_REGEX = re.compile(
        rf"(({INTRO_HEADLINES_REGEX.pattern})|({MIDDLE_HEADLINES_REGEX.pattern})|({CONCLUSION_HEADLINES_REGEX}))",
        re.IGNORECASE | re.ASCII)

    # takes numbering only at the start of a headline

    def __str__(self):
        return json_dumps(self.asdict())

    def remove(self, path: Sequence[int]):
        """
        Removes a sub-hierarchy from this hierarchy.

        :param path: the path to the sub-hierarchy
        """
        if len(path) == 1:
            del self.content[path[0]]
        else:
            self.content[path[0]].remove(path[1:])

    def asdict(self) -> Dict[str, Any]:
        """
        Converts this data class to dictionary.

        :return: dictionary representation of this data class
        """
        # dataclasses.asdict(self) is too slow
        return {
            "headline": self.headline,
            "content": self.content.asdict() if isinstance(self.content, TextContent) else [h.asdict() for h in self.content]
        }

    @staticmethod
    def from_dict(d: Dict[str, Any], lazy: bool = False, lazy_children: bool = False) -> "Hierarchy":
        """
        Creates this data class from dictionary.

        :param d: the dictionary used of instantiation
        :param lazy: if True, the content is not parsed and instead a lazy hierarchy is created
        :param lazy_children: if True, the children are not parsed and instead lazy hierarchies are created
        :return: create hierarchy
        """
        if lazy:
            return LazyHierarchy(d)

        content_dic = d["content"]

        content = [Hierarchy.from_dict(h, lazy_children) for h in content_dic] \
            if isinstance(content_dic, list) else TextContent.from_dict(content_dic)

        return Hierarchy(
            headline=d["headline"],
            content=content
        )

    def text_content(self, parent_condition: Optional[Callable[["Hierarchy"], bool]] = None) \
            -> Generator[TextContent, None, None]:
        """
        Generates text contents in hierarchy from the left most one to the right most one.

        :param parent_condition: optional filter that allows to set up condition on parent of text content
            True passes
        :return: generator of TextContent
        """

        for h in self.pre_order():
            if isinstance(h.content, TextContent) and (parent_condition is None or parent_condition(h)):
                yield h.content

    def citation_spans(self) -> Generator[RefSpan, None, None]:
        """
        Generation of all citations spans in hierarchy.
        It iterates text content from the left most one to the right most one, but it does not guarantee left to right
        positioning of citation inside a single text content.

        :return: generator of citation spans
        """
        for text in self.text_content():
            for cit in text.citations:
                yield cit

    @property
    def height(self) -> int:
        """
        Height of hierarchy.
        """
        if isinstance(self.content, TextContent) or len(self.content) == 0:
            return 0
        return max(h.height + 1 for h in self.content)

    @property
    def has_text_content(self) -> bool:
        """
        True if in this whole hierarchy is at least one text content with non empty text
        """

        return any(len(t_c.text) > 0 for t_c in self.text_content())

    def sections(self, min_height: int = 2) -> Generator["Hierarchy", None, None]:
        """
        Generates sections in hierarchy. Doesn't generate itself.

        :param min_height: sub-hierarchy is considered as section when it has at least such height
        :return: generator of sections in pre-order fashion
        """

        iterator = iter(self.pre_order())
        next(iterator)  # not interested in itself

        for h in iterator:
            if h.height >= min_height:
                yield h

    def nodes_with_height(self, height: int) -> Generator["Hierarchy", None, None]:
        """
        Generates nodes with given height in pre-order fashion.

        :param height: height of nodes that are supposed to be generated
        :return: generator of nodes with given height
        """

        for n in self.pre_order():
            if n.height == height:
                yield n

    def prune_empty_headlines_nodes(self, min_height: int = 2):
        """
        Removes all sub-nodes with empty headline that have at least given height.

        :param min_height: sub-hierarchy is considered when it has at least such height
        """

        self.prune_nodes_with_at_least_given_height(lambda c: c.headline is not None and len(c.headline) > 0,
                                                    min_height)

    def prune_according_to_name_assigned_to_chars_in_headline(self, r: Pattern, coverage: float, min_height: int = 2):
        """
        Removes all sub-nodes with lower coverage of valid characters in headline that have at least given height.

        Every node with headline None is pruned.

        :param r: regex pattern for results of unicodedata.name
        :param coverage: minimal proportion of characters matching given pattern to pass this filter
        :param min_height: sub-hierarchy is considered when it has at least such height
        """
        if not (0 <= coverage <= 1):
            raise ValueError("Coverage must be in [0,1].")

        def predicate(n: Hierarchy) -> bool:
            return n.headline is not None and self.chars_name_coverage(n.headline, r) >= coverage

        self.prune_nodes_with_at_least_given_height(predicate, min_height)

    def prune_nodes_with_at_least_given_height(self, predicate: Callable[["Hierarchy"], bool], min_height: int = 2):
        """
        Prunes all nodes (sub-hierarchies) not satisfying predicated having at least given height.

        :param predicate: predicate that returns for each node whether it should stay (True) or be pruned (False)
        :param min_height: sub-hierarchy is considered when it has at least such height
        """

        if self.height <= min_height:
            return

        to_process = [self]

        while to_process:
            parent = to_process.pop(-1)
            if isinstance(parent.content, list):
                # children might have lower height than parent.height -1,
                # because parents height is max among children + 1
                parent.content = [c for c in parent.content if c.height < min_height or predicate(c)]
                for child_hierarchy in parent.content:
                    if child_hierarchy.height > min_height:
                        # is parent of at least one child that should be investigated
                        to_process.append(child_hierarchy)

    @staticmethod
    def chars_name_coverage(characters: str, r: Pattern) -> float:
        """
        Percentage of characters which assigned name by unicodedata.name satisfies given pattern.
        for empty strings the coverage is 0

        :param characters: characters for coverage
        :param r: pattern for character name
        :return: coverage
        """
        if len(characters) == 0:
            return 0
        return sum(r.match(unicodedata.name(c, "")) is not None for c in characters) / len(characters)

    def prune_nodes_without_text_content(self):
        """
        Removes all sub-nodes without text content (in whole sub-hierarchy).
        """
        to_process = [self]

        while to_process:
            h = to_process.pop(-1)
            if isinstance(h.content, list):
                h.content = [c for c in h.content if c.has_text_content]
                for sub_hierarchy in h.content:
                    to_process.append(sub_hierarchy)

    def prune_named_text_blocks(self, headlines: AbstractSet[str], lower_case: bool = True):
        """
        removes all hierarchies, which have given headline, and directly contain text content

        :param headlines: headlines of text blocks to be removed
        :param lower_case: if True, headlines are converted to lower case
        """

        to_process = [self]

        while to_process:
            h = to_process.pop(-1)
            if isinstance(h.content, list):
                new_content = []
                for c in h.content:
                    if not isinstance(c.content, TextContent):
                        new_content.append(c)
                        continue

                    headline = c.headline
                    if headline is not None and lower_case:
                        headline = headline.lower()

                    if headline not in headlines:
                        new_content.append(c)

                h.content = new_content
                for sub_hierarchy in h.content:
                    to_process.append(sub_hierarchy)

    def paths_to_nodes_with_height(self, height: int, on_path: Optional[List["Hierarchy"]] = None) \
            -> Generator[List["Hierarchy"], None, None]:
        """
        Generates paths from root to nodes with given height in pre-order fashion.

        :param height: height of nodes
        :param on_path: path tracing of parents
        :return: generates paths from roots to nodes with given height
             path is represented by sequence of nodes
        """
        self_height = self.height

        act_path = [self] if on_path is None else on_path + [self]

        if self_height == height:
            yield act_path
        elif self_height > height and isinstance(self.content, list):
            for h in self.content:
                yield from h.paths_to_nodes_with_height(height, act_path)

    def pre_order(self) -> Generator["Hierarchy", None, None]:
        """
        Iterates all sub-hierarchies in pre-order like order.

        And yes, it also generates itself.

        :return: generator of sub-hierarchies
        """

        to_process = [self]

        while to_process:
            h = to_process.pop(-1)
            yield h
            if isinstance(h.content, list):
                for sub_hierarchy in reversed(h.content):
                    to_process.append(sub_hierarchy)

    def get_part(self, headline_re: Pattern, max_h: float = math.inf, min_depth: float = 0,
                 max_depth: float = math.inf, return_path: bool = False) -> Union[List["Hierarchy"], List[Tuple[List["Hierarchy"], Tuple[int]]]]:
        """
        Searches in hierarchy for given headline and returns the whole sub-hierarchy associated to it.

        If a hierarchy with matching headline contains sub hierarchy with headline that also matches, it returns just
        the parent hierarchy.

        :param headline_re: compiled regex that will be used for headline matching
        :param max_h: maximal number of matching hierarchies after which the search is stopped
        :param min_depth: minimal depth of a node (root has zero depth)
        :param max_depth: maximal depth of a node
        :param return_path: if True, returns also path to the hierarchy
            path is represented by sequence of indices of sub-hierarchies
        :return: all searched hierarchies.
            If return_path is True, returns also path to the hierarchy
        """
        to_process = [(0, self, ())]

        res = []

        while to_process:
            depth, h, path = to_process.pop(-1)

            if h.headline is not None and depth >= min_depth:
                if headline_re.match(h.headline):
                    res.append((h, path) if return_path else h)
                    if len(res) >= max_h:
                        break
                    continue

            if isinstance(h.content, list) and depth < max_depth:
                for i, sub_hierarchy in zip(range(len(h.content)-1, -1, -1), reversed(h.content)):
                    to_process.append((depth + 1, sub_hierarchy, path + (i,)))

        return res

    @staticmethod
    def convert_letter_to_int(letter: str) -> int:
        """
        Converts letter character used as section counter to its integer representation.

        :param letter: CAPITAL letter for conversion
        :return: converted letter to text
        """
        assert 0 < len(letter) < 2
        return 1 + (ord(letter) - ord('A'))  # we don't start with zero in our numerical system, but from 1

    def guess_serial_number(self, strict: bool = False, num_format: bool = False) \
            -> Optional[Union[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[SerialNumberFormat, ...]]]]:
        """
        Tries to guess serial number from hierarchy headline.
        A serial multi-lvl number haves levels separated with . (voluntary . at the end) and each level consists
        of serial number (could be roman (at top lvl), also a letter (not at top lvl) is allowed).
        Example of  three lvl serial number:
            II.2.1

        The guess is not authoritative. E.g. it may extract serial number II from headline: World War II
        If you want a solution that is able to utilize context and therefore provide more reliable guess use:
            :func:`~document.Hierarchy.guess_content_serial_numbers`

        :param strict: Whether the strict regex for guessing should be used
        :param num_format: whether the format of serial number should also be returned
        :return:
            Serial number in form of tuple from the top most lvl to the lowest one
            Serial number in form of tuple from the top most lvl to the lowest one and format of all numbers
            None if there is no serial number that can be guessed from headline.
        """
        if self.headline is None:
            return None

        return self.guess_serial_number_from_string(self.headline, strict, num_format)

    @classmethod
    def guess_serial_number_from_string(cls, headline: str, strict: bool = False, num_format: bool = False) \
            -> Optional[Union[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[SerialNumberFormat, ...]]]]:
        """
        Tries to guess serial number from hierarchy headline.
        A serial multi-lvl number haves levels separated with . (voluntary . at the end) and each level consists
        of serial number (could be roman (at top lvl), also a letter (not at top lvl) is allowed).
        Example of  three lvl serial number:
            II.2.1

        The guess is not authoritative. E.g. it may extract serial number II from headline: World War II
        If you want a solution that is able to utilize context and therefore provide more reliable guess use:
            :func:`~document.Hierarchy.guess_content_serial_numbers`

        :param headline: Headline for guessing
        :param strict: Whether the strict regex for guessing should be used
        :param num_format: whether the format of serial number should also be returned
        :return:
            Serial number in form of tuple from the top most lvl to the lowest one
            Serial number in form of tuple from the top most lvl to the lowest one and format of all numbers
            None if there is no serial number that can be guessed from headline.
        """

        use_re = cls.SECTION_NUMBER_REGEX_STRICT if strict else cls.SECTION_NUMBER_REGEX
        s = use_re.search(headline)
        if not s:
            if cls.INTRO_HEADLINES_REGEX.match(headline):
                return ((1,), (SerialNumberFormat.UNKNOWN,)) if num_format else (1,)
            return None

        split_res: List[Union[str, int]] = s.group("serial").rstrip(".").split(".")
        num_formats = []

        for i, v in enumerate(split_res):
            v = v.strip()

            try:
                split_res[i] = int(v)
                num_formats.append(SerialNumberFormat.ARABIC)
            except ValueError:
                v = v.upper()
                # is not integer, must be roman or a letter
                try:
                    # let's first try the ROMAN
                    # see that for some cases this may incorrectly match latter, so the a contextual repair should
                    # follow this
                    if any(r in v for r in {"L", "C", "D", "M"}):
                        # exclude big numbers as it is not probable that they are section roman numbers
                        raise KeyError
                    split_res[i] = roman_2_int(v)
                    num_formats.append(SerialNumberFormat.ROMAN)
                except KeyError:
                    # letter it is
                    num_formats.append(SerialNumberFormat.LATIN)
                    split_res[i] = cls.convert_letter_to_int(v)

        return (tuple(split_res), tuple(num_formats)) if num_format else tuple(split_res)

    @classmethod
    def correct_the_roman_format_mismatch(cls, ser_nums: List[Optional[Tuple[int, ...]]],
                                          ser_nums_format: List[Tuple[SerialNumberFormat, ...]]) \
            -> Tuple[List[Tuple[int, ...]], List[Tuple[SerialNumberFormat, ...]]]:
        """
        In some cases a letter could be incorrectly identified as roman number by guess_serial_number method.
        This method, in contrast to mentioned method, takes into consideration the context in which the serial number is
        so it can identify whether a roman number occurs in series of letters.

        :param ser_nums: serial numbers of sections
        :param ser_nums_format: formats of provided serial numbers
        :return: corrected numbers and formats
        """

        if len(ser_nums) < 2:
            # we need at least some context
            return ser_nums, ser_nums_format

        for i, (n, f) in enumerate(zip(ser_nums, ser_nums_format)):
            # let's go through sequence and detect whether the ROMAN ones fits into their context
            if SerialNumberFormat.ROMAN in f:
                if not (i == 0 or cls.serial_number_is_subsequent(ser_nums[i - 1], n, ser_nums_format[i - 1], f)) or \
                        not (i == len(ser_nums) - 1 or cls.serial_number_is_subsequent(n, ser_nums[i + 1], f,
                                                                                       ser_nums_format[i + 1])):
                    # it seems that the roman is not fitting into its context, lets check whether the letter will work
                    # fit better
                    num_list = list(n)
                    format_list = list(f)

                    before_num_list, before_format_list = None, None
                    if i > 0 and ser_nums[i - 1] is not None:
                        before_num_list = list(ser_nums[i - 1])
                        before_format_list = list(ser_nums_format[i - 1])

                    after_num_list, after_format_list = None, None
                    if i < len(ser_nums) - 1 and ser_nums[i + 1] is not None:
                        after_num_list = list(ser_nums[i + 1])
                        after_format_list = list(ser_nums_format[i + 1])

                    for p, p_f in enumerate(f):
                        if p_f == SerialNumberFormat.ROMAN:
                            try:
                                num_list[p] = cls.convert_letter_to_int(int_2_roman(n[p]))
                                format_list[p] = SerialNumberFormat.LATIN
                                # check whether the letter fits better into context, let's make the check just on
                                # prefix as the rest may not be converted yet

                                if after_num_list is not None and \
                                        len(after_format_list) > p and after_format_list[p] == SerialNumberFormat.ROMAN:
                                    # let's make one step forward and try to convert that to latin
                                    after_num_list[p] = cls.convert_letter_to_int(int_2_roman(ser_nums[i + 1][p]))
                                    after_format_list[p] = SerialNumberFormat.LATIN

                                if (i == 0 or cls.serial_number_is_subsequent(
                                        ser_nums[i - 1], tuple(num_list),
                                        ser_nums_format[i - 1], tuple(format_list))) and \
                                        (i == len(ser_nums) - 1 or cls.serial_number_is_subsequent(
                                            tuple(num_list), None if after_num_list is None else tuple(after_num_list),
                                            tuple(format_list),
                                            None if after_num_list is None else tuple(after_format_list))):
                                    # the letter fits better
                                    continue
                                elif (
                                        i > 0 and before_format_list is not None and p < len(
                                    before_format_list)  # can compare with left
                                        and before_format_list[p] == SerialNumberFormat.LATIN
                                        and before_num_list[p] <= num_list[p]  # the new latin is not breaking order
                                        and (n[p] < before_num_list[p] or  # order was broken or better fit
                                             n[p] - before_num_list[p] > num_list[p] - before_num_list[p])
                                ) or (
                                        i < len(ser_nums) - 1 and after_format_list is not None and p < len(
                                    after_format_list)  # can compare with right
                                        and after_format_list[p] == SerialNumberFormat.LATIN
                                        and num_list[p] <= after_num_list[p]  # the new latin is not breaking order
                                        and (after_num_list[p] < n[p] or  # order was broken or better fit
                                             after_num_list[p] - n[p] > after_num_list[p] - num_list[p])):
                                    # previous context test may sometimes fail on the reset section counter phenomena
                                    # e.g.: [(1,), (1,), (2,), (100,), (500,), (2,)]
                                    #   [(ARABIC,), (LATIN,), (LATIN,), ROMAN,), (ROMAN,), (ARABIC,)]
                                    #
                                    # the result would be:
                                    #   [(1,), (1,), (2,), (3,), (500,), (2,)]
                                    # so we checked whether the difference will lower at least from one side
                                    continue

                            except AssertionError:
                                # ok never mind this one translates to more than one letter
                                pass
                            num_list[p] = n[p]
                            format_list[p] = f[p]

                    ser_nums[i], ser_nums_format[i] = tuple(num_list), tuple(format_list)

        return ser_nums, ser_nums_format

    @classmethod
    def search_endings(cls, content: Sequence["Hierarchy"],
                       ser_num: MutableSequence[Optional[Tuple[Union[int, PositionFromEnd], ...]]],
                       sticky: MutableSequence[int]) -> None:
        """
        Searches headlines that seems like those that are used at the end of a paper and haven't been used in sticky
        so far.
        This method works in place thus the passed structures could be changed.

        :param content: content of a hierarchy
        :param ser_num: (could be changed) series of serial numbers for each item in content member variable
        :param sticky: (could be changed) indices of sticky sequence numbers, those that couldn't be changed and their
            serial number is highly probable identified correctly
            should be sorted in ascending order
        """

        not_seen_ends = {"conclusion", "appendix", "acknowledgement", "references"}
        # check whether we have already seen some of them
        for s in sticky:
            m = cls.ENDINGS_STICKY_REGEX.match(content[s].headline)
            if m:
                for k, v in m.groupdict().items():
                    # remove the triggered ones
                    if v is not None:
                        try:
                            not_seen_ends.remove(k)
                        except KeyError:
                            # its here multiple times
                            ...

        search_endings_in = content[sticky[-1] + 1:] if len(sticky) > 0 else content
        from_end_pos_cnt = 0
        new_sticky = []  # will be saved there first as we are going in reverse order
        for i, c in enumerate(reversed(search_endings_in)):
            if c.headline is not None:
                m = cls.ENDINGS_STICKY_REGEX.match(c.headline)
                if m:
                    m_groups = m.groupdict()

                    if any(m_groups[e] is not None for e in not_seen_ends):
                        for k, v in m_groups.items():
                            # remove the triggered ones
                            if v is not None:
                                try:
                                    not_seen_ends.remove(k)
                                except KeyError:
                                    # its here multiple times
                                    ...

                        index = len(content) - 1 - i
                        if ser_num[index] is None:  # serial number is not assigned
                            # (PositionFromEnd(1),) we are leaving one position from end for appendix
                            ser_num[index] = (PositionFromEnd(from_end_pos_cnt),)
                        new_sticky.append(index)
                        from_end_pos_cnt += 1
                        if len(not_seen_ends) == 0:
                            return
        sticky.extend(reversed(new_sticky))

    @staticmethod
    def match_cap_gap_filler(content: Sequence["Hierarchy"],
                             ser_num: MutableSequence[Optional[Tuple[Union[int, PositionFromEnd], ...]]],
                             ser_nums_formats: MutableSequence[Tuple[SerialNumberFormat, ...]],
                             anchors: Sequence[int]) -> None:
        """
        tries to fill gaps by matching capitalization of None-ones if capitalization is exclusive for top lvl

        This method works in place thus the passed structures could be changed.

        :param content: content of a hierarchy
        :param ser_num: (could be changed) series of serial numbers for each item in content member variable
        :param ser_nums_formats: (could be changed) formats of serial numbers for each item in content member variable
        :param anchors: sections that should be considered as the main document structure
        """

        if len(anchors) > 0 and any(ser_num[s] is not None and len(ser_num[s]) > 1 for s in anchors) \
                and all((ser_num[s] is None) or (len(ser_num[s]) == 1 and content[s].headline.isupper()) or
                        (len(ser_num[s]) > 1 and not content[s].headline.isupper()) for s in anchors):

            for f, s in zip(anchors, anchors[1:]):
                f_num = ser_num[f][0]
                s_num = ser_num[s][0]
                if f + 1 < s and ser_nums_formats[f][0] == ser_nums_formats[s][0] and \
                        isinstance(ser_num[f][0], int) and isinstance(ser_num[s][0], int) and \
                        f_num + 1 < s_num:
                    # there is something between and there is also a gap
                    gap_cnt = ser_num[f][0] + 1
                    num_format = ser_nums_formats[f][0]
                    for g_i in range(f + 1, s):
                        if ser_num[g_i] is None and content[g_i].headline.isupper():
                            ser_num[g_i] = (gap_cnt,)
                            ser_nums_formats[g_i] = (num_format,)
                            gap_cnt += 1
                            if gap_cnt >= s_num:
                                break

            # handle the ending
            if isinstance(ser_num[anchors[-1]][0], int):
                gap_cnt = ser_num[anchors[-1]][0] + 1
                num_format = ser_nums_formats[anchors[-1]][0]
                for g_i in range(anchors[-1] + 1, len(ser_num)):
                    if ser_num[g_i] is None and content[g_i].headline.isupper():
                        ser_num[g_i] = (gap_cnt,)
                        ser_nums_formats[g_i] = (num_format,)
                        gap_cnt += 1

    def _parse_serial_numbers(self, strict: bool = False) -> \
            Tuple[List[Optional[Tuple[Union[int, PositionFromEnd], ...]]], List[Tuple[SerialNumberFormat, ...]]]:
        """
        Parses serial numbers, with their format, of items in content.

        :param strict: Whether the strict regex for guessing should be used
        :return: serial numbers for content and their format
        """
        ser_num = []
        ser_nums_formats = []

        for h in self.content:
            res = h.guess_serial_number(strict, True)
            if res is None:
                ser_num.append(None)
                ser_nums_formats.append((SerialNumberFormat.UNKNOWN,))
                continue
            n, f = res
            ser_num.append(n)
            ser_nums_formats.append(f)

        return self.correct_the_roman_format_mismatch(ser_num, ser_nums_formats)

    def guess_content_serial_numbers(self, strict: bool = False, ret_sticky: bool = False, max_sparsity: int = 3) \
            -> Union[
                Tuple[
                    List[Optional[Tuple[Union[int, PositionFromEnd], ...]]], Sequence[Tuple[SerialNumberFormat, ...]]],
                Tuple[List[Optional[Tuple[Union[int, PositionFromEnd], ...]]], Sequence[Tuple[SerialNumberFormat, ...]],
                List[int]]
            ]:
        """
        Guesses all serial numbers for content in this hierarchy.

        :param strict: Whether the strict regex for guessing should be used
        :param ret_sticky: Whether the sticky marks should be returned
            It marks those that serial number is highly probable identified correctly.
            The name sticky comes from that fact that they are used when building anchor as those points that must be in
            it.
        :param max_sparsity: Maximal sparsity used for :func:`document.Hierarchy.search_longest_sparse_subsequent`
        :return: serial numbers for content and their format
            and sticky marks on demand
        """

        ser_num, ser_nums_formats = self._parse_serial_numbers(strict)

        sticky = [i for i, c in enumerate(self.content) if c.headline is not None and
                  self.INTRO_AND_MIDDLE_HEADLINES_STICKY_REGEX.match(c.headline) and ser_num[i] is not None]

        self.search_endings(self.content, ser_num, sticky)

        anchors = self.search_longest_sparse_subsequent(ser_num, ser_nums_formats, sticky, convert_to_intervals=False,
                                                        max_sparsity=max_sparsity)
        self.match_cap_gap_filler(self.content, ser_num, ser_nums_formats, anchors)

        ser_num, ser_nums_formats = self.repair_reset_counter_ser_nums(ser_num, sticky, ser_nums_formats, 1)

        return (ser_num, ser_nums_formats, sticky) if ret_sticky else (ser_num, ser_nums_formats)

    @staticmethod
    def serial_numbers_sparsity(f: Tuple[Union[int, PositionFromEnd], ...],
                                s: Tuple[Union[int, PositionFromEnd], ...]) -> int:
        """
        Calculates sparsity of two serial numbers. The second must be greater or the sparsity is always -1.

        In theory between any two consecutive serial numbers there could be infinitely many numbers. Thus, we define
        the sparsity in different way by making an assumption about the structure that could exist between these
        two given nodes. In each case we assume that the unknown structure is minimalistic as possible.

        Examples:
            f=(1,), s=(4,)
            Assuming: 1 -> 2 -> 3 -> 4
            Sparsity: 2

            f=(1,), s=(1, 1)
            Assuming: 1 -> 1.1
            Sparsity: 0

            f=(1,1) s=(4,2)
            Assuming: 1.1 -> 2 -> 3 -> 4 -> 4.1 -> 4.2
            Sparsity: 4

            f=(1,), s=(3, 3)
            Assuming: 1 -> 2 -> 3 -> 3.1 -> 3.2 -> 3.3
            Sparsity: 4

            f=(3, 1, 2), s=(4,2)
            Assuming: 3.1.2 -> 4 -> 4.1 -> 4.2
            Sparsity: 2

            f=(4, 2, 2), s=(4,2,4,2)
            Assuming: 4.2.2 -> 4.2.3 -> 4.2.4 -> 4.2.4.1 -> 4,2,4,2
            Sparsity: 3


        There is also special number inf which means that it is subsequent of any (no matter the format),
        but has no subsequents by its own on given hierarchy level. The sparsity on given hierarchy level is zero,
        therefore it is considered as direct subsequent.

        :param f: first serial number
        :param s: second serial number
        :return: sparsity or -1 when the second is not greater than first
        """

        steps = 0
        while len(s) and f < s:
            act_steps = s[0] - f[0]
            if act_steps == math.inf:
                act_steps = 1
            steps += act_steps
            if act_steps == 0 and len(f) > 1:
                # we are removing shared prefix
                f = f[1:]
            else:
                # we are moving to another section
                f = (0,)
            s = s[1:]

        return steps - 1

    @classmethod
    def serial_number_is_subsequent(cls, f: Optional[Tuple[Union[int, PositionFromEnd], ...]],
                                    s: Optional[Tuple[Union[int, PositionFromEnd], ...]],
                                    f_format: Optional[Tuple[SerialNumberFormat, ...]] = None,
                                    s_format: Optional[Tuple[SerialNumberFormat, ...]] = None,
                                    max_sparsity: int = 0) -> bool:
        """
        Decides whether given number s is subsequent of f.

        There is also special number inf which means that it is subsequent of any, but has no subsequents by its own
        on given hierarchy level. It was developed for appendix: (inf,).

        :param f: first serial number
        :param s: second serial number
        :param f_format: voluntary you can also provide format of first serial number in that case it will force
            that the subsequents also haves the same format on shared parts
        :param s_format: voluntary you can also provide format of first serial number in that case it will force
            that the subsequents also haves the same format on shared parts
        :param max_sparsity: defines how sparse it can be in terms of missing items between two elements on the
            same hierarchy level
        :return: True s is subsequent of f.
        """
        if f is None:
            return False

        return s is not None and f < s and cls.serial_numbers_sparsity(f, s) <= max_sparsity and \
            (
                    f_format is None or
                    s_format is None or
                    cls.serial_number_format_match(f_format, s_format)
            )

    @classmethod
    def serial_number_is_direct_subsequent(cls, f: Optional[Tuple[Union[int, PositionFromEnd], ...]],
                                           s: Optional[Tuple[Union[int, PositionFromEnd], ...]],
                                           f_format: Optional[Tuple[SerialNumberFormat, ...]] = None,
                                           s_format: Optional[Tuple[SerialNumberFormat, ...]] = None) -> bool:
        """
        This is just alias for serial_number_is_subsequent with forced sparsity to 0.

        :param f: first serial number
        :param s: second serial number
        :param f_format: voluntary you can also provide format of first serial number in that case it will force
            that the subsequents also haves the same format on shared parts
        :param s_format: voluntary you can also provide format of first serial number in that case it will force
            that the subsequents also haves the same format on shared parts
        :return: True s is direct subsequent of f.
        """
        return cls.serial_number_is_subsequent(f, s, f_format, s_format, max_sparsity=0)

    @staticmethod
    def serial_number_format_match(f: Tuple[SerialNumberFormat, ...], s: Tuple[SerialNumberFormat, ...]):
        """
        Two serial numbers match when they share the same prefix of length that is equal to the smaller one.
        Also the unknown type matches with any.

        :param f: first serial number for comparison
        :param s: second serial number for comparison
        :return: True when the formats are matching
        """
        min_len = min(len(f), len(s))
        f_shared, s_shared = f[:min_len], s[:min_len]
        return all(
            s == o or s == SerialNumberFormat.UNKNOWN or o == SerialNumberFormat.UNKNOWN
            for s, o in zip(f_shared, s_shared)
        )

    @classmethod
    def search_longest_sparse_subsequent(cls, ser_num: Sequence[Optional[Tuple[Union[int, PositionFromEnd], ...]]],
                                         ser_num_formats: Optional[
                                             Sequence[Tuple[SerialNumberFormat, ...]]] = None,
                                         sticky: Optional[Sequence[int]] = None,
                                         convert_to_intervals: bool = True,
                                         max_sparsity: int = 3) \
            -> List[Union[Tuple[int, int], int]]:
        """
        It selects the longest sequence of intervals that are subsequent, but it may be sparse, which means,
            that another interval may be between.

            works on block lvl of direct subsequents on the same lvl

        Example:
            >>> ser_num = [(1,), (1, 1), None, (1, 2),(1, 2, 1)]
            >>> Hierarchy.search_longest_sparse_subsequent(ser_num)
            [(0, 2), (3, 5)]

        :param ser_num: serial numbers of sections
        :param ser_num_formats: if provided than it will force subsequent matching that consideres also format of
            serial numbers
        :param sticky: List of sections indices that must be in final sequence.
            this may cause that an empty sequence is returned
        :param convert_to_intervals: whether individual indices should be returned or intervals
        :param max_sparsity: defines how sparse it can be in terms of missing items between two elements on the
            same hierarchy level
        :return: longest sequence of intervals that are subsequent or individual indices
            at least two subsequent numbers must be found otherwise empty list is returned
        """
        if all(x is None for x in ser_num):
            return []

        if sticky is None:
            sticky = []

        all_subsequents: List[Optional[List[int]]] = [None for _ in range(len(ser_num))]
        for i in reversed(range(len(ser_num))):
            # we iterate in reverse, because due to the transitive property of order relation subsequents of
            # an element x are also subsequents of elements before x for which the x is subsequent
            all_current = [[i]]
            for look_at in range(i + 1, len(ser_num)):
                if ser_num_formats is None:
                    is_subsequent = cls.serial_number_is_subsequent(ser_num[i], ser_num[look_at],
                                                                    max_sparsity=max_sparsity)
                else:
                    is_subsequent = cls.serial_number_is_subsequent(ser_num[i], ser_num[look_at], ser_num_formats[i],
                                                                    ser_num_formats[look_at], max_sparsity=max_sparsity)

                if is_subsequent:
                    new_one = [i] + all_subsequents[look_at]
                    if len(sticky) > 0:
                        skip = False
                        for s in sticky:
                            if i <= s and s not in new_one:
                                skip = True
                        if skip:
                            continue
                    all_current.append(new_one)
            all_subsequents[i] = max(all_current, key=lambda elements: len(elements))

        all_subsequents = [x for x in all_subsequents if all(s in x for s in sticky)]
        if len(all_subsequents) == 0:
            return []
        the_longest = max(all_subsequents, key=lambda elements: len(elements))

        if len(the_longest) < 2:
            # just single node
            return []

        if not convert_to_intervals:
            return the_longest

        # convert to intervals
        longest_intervals = []

        s, e, = the_longest[0], the_longest[0] + 1
        for ele in the_longest[1:]:
            if e == ele:
                e += 1
            else:
                longest_intervals.append((s, e))
                s = ele
                e = ele + 1
        longest_intervals.append((s, e))
        return longest_intervals

    @classmethod
    def fix_alone_number_headlines(cls, sections: Sequence["Hierarchy"], anchor_map: ImmutIntervalMap):
        """
        Fixes headlines that are containing only the serial number and the rest of headline is in content.
        Examples (GROBID format):
        1. example
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="4.1.1.">Relevant International Declarations</head>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>4.1.2.</head><p>Case Study One: ...e Effective. "</p>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="4.1.3.">Case Study Two: The International ...</head>
        2. example
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="2.4.3.">What is the Technology Readiness Level of GEHB?
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>Technology Readiness Levels in the National Aeronautics
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>Technology Readiness Levels in the National Aeronautics
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="9.">Actual system ʹflight provenʹ through suc- cessful
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>2.4.4.</head><p>Psychotropic Drugs as the ...</p>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="2.4.5.">Does it Make any Sense to Think...</head>

        :param sections: sections for headline fix
            the modification is done in place of that argument
        :param anchor_map: mapping that is used for deciding which section is in anchor
            anchor: intervals of sections that should be considered as the main document structure
        """

        for i, s in enumerate(sections):
            if s.headline is None:
                continue
            search_res = cls.SECTION_NUMBER_REGEX_STRICT.search(s.headline)
            if search_res is not None and search_res.span() == (0, len(s.headline)) and i in anchor_map and \
                    not isinstance(s.content, TextContent) and len(s.content) > 0:
                # obtain first text part and move it to the headline if appropriate, delete its hierarchy part in case
                # its empty
                path_to: List[Union[Hierarchy, TextContent]] = [s]
                while isinstance(path_to[-1], Hierarchy):
                    if isinstance(path_to[-1].content, TextContent):
                        path_to.append(path_to[-1].content)
                        break
                    if len(path_to[-1].content) > 0:
                        path_to.append(path_to[-1].content[0])
                    else:
                        break

                if isinstance(path_to[-1], TextContent) and len(path_to[-1].citations) == 0 and \
                        len(path_to[-1].references) == 0:
                    # modify
                    s.headline = s.headline + " " + path_to[-1].text
                    # remove from hierarchy
                    for h in reversed(path_to):
                        if isinstance(h, Hierarchy) and not isinstance(h.content, TextContent):
                            h.content = h.content[1:]
                            # check whether the hierarchy becomes empty and should be removed
                            if len(h.content) > 0:
                                # no removal is needed
                                break

    @classmethod
    def merge_split_headlines(cls, sections: Sequence["Hierarchy"],
                              ser_num: Sequence[Optional[Tuple[Union[int, PositionFromEnd], ...]]],
                              anchor_map: ImmutIntervalMap) -> List["Hierarchy"]:
        """
        Merges split headline together.
        Examples of split headlines (GROBID format):

        1. example
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>CHAPTER II</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>REVIEW OF ENERGY STORAGE DEVICES FOR POWER</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>ELECTRONICS APPLICATIONS</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="2.1.">Introduction</head><p>Power electronics applications
        2. example - This one contains two headlines for the same section, but it was probably split intentionally
            so we will not merge it
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>PART (I) INTRODUCTION</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="1.">THE GOALS AND THE STRUCTURE OF THIS BOOK</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="1.1.">The Goals of this Book</head></div>
        3.example
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>CHAPTER II</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>LOW DENSITY PARITY CHECK CODES</head><p>This chapter presents
            <div xmlns="http://www.tei-c.org/ns/1.0"><head n="2.1">Product Accumulate Codes:</head><p>PA Codes
        4.example
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>CHAPTER II</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>LOW DENSITY PARITY CHECK CODES</head><p>This chapter presents
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>CHAPTER IV</head><p>PA Codes

        :param sections: sections for merging
            the sections might be modified by this method
        :param ser_num: serial numbers of sections strictly identified
        :param anchor_map: mapping that is used for deciding which section is in anchor
            anchor: intervals of sections that should be considered as the main document structure
        :return: merged sections
        """
        ser_num_not_strict = [h.guess_serial_number(False) for h in sections]
        new_sections = []
        i = 0
        while i < len(ser_num_not_strict):
            s_n = ser_num_not_strict[i]
            if s_n is not None and len(sections[i].content) == 0:
                j = i + 1
                # skip empty sections without number
                while j < len(sections) and len(sections[j].content) == 0 and ser_num[j] is None:
                    j += 1

                if j < len(sections):
                    if j in anchor_map:
                        # this part coverages 1. and 2. example
                        if s_n == ser_num[j]:  # 2. example
                            # no merging
                            pass
                        elif cls.serial_number_is_direct_subsequent(s_n, ser_num[j]):  # 1. example
                            # the headline is probably associated headline of next anchor
                            h = sections[i]
                            h.headline = " ".join(sections[x].headline for x in range(i, j))
                            new_sections.append(h)
                            i = j
                            continue
                    elif len(sections[j].content) > 0 and j + 1 in anchor_map and \
                            cls.serial_number_is_direct_subsequent(s_n, ser_num[j + 1]):
                        # that part coverages 3. example
                        h = sections[j]
                        h.headline = " ".join(sections[x].headline for x in range(i, j + 1))
                        new_sections.append(h)
                        i = j + 1
                        continue
                    elif i in anchor_map and ser_num[j] is None:  # 4. example
                        h = sections[j]
                        h.headline = " ".join(sections[x].headline for x in range(i, j + 1))
                        new_sections.append(h)
                        i = j + 1
                        continue

            new_sections.append(sections[i])
            i += 1
        return new_sections

    @classmethod
    def split_merged_headlines(cls, sections: Sequence["Hierarchy"],
                               ser_num: Sequence[Optional[Tuple[Union[int, PositionFromEnd], ...]]],
                               ser_num_format: Sequence[Tuple[SerialNumberFormat, ...]],
                               ) -> List["Hierarchy"]:
        """
        Splits of wrongly merged headlines.
        Examples of split headlines (GROBID format):

        1. example
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>6. Conclusions and future work</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>Appendix A. Notational representation of social argument</head></div>
            <div xmlns="http://www.tei-c.org/ns/1.0"><head>B. Operational semantics </head></div>

        :param sections: sections for merging
            the sections might be modified by this method
        :param ser_num: serial numbers of sections strictly identified
        :param ser_num_format: is used for subsequent matching that consideres also format of  serial numbers
        :return: split sections
        """
        new_sections = []
        for i, s in enumerate(sections):
            new_sections.append(s)

            if s.headline is None:
                continue

            if ser_num[i] == (PositionFromEnd(0),):
                m = cls.APPENDIX_HEADLINES_STICKY_REGEX.match(s.headline)
                if m:
                    start, end = m.span("appendix")
                    if end < len(s.headline.rstrip()):
                        new_s = Hierarchy(s.headline[end:].strip(), s.content)
                        s.headline = s.headline[:end].strip()
                        s.content = []
                        new_sections.append(new_s)
                        continue

            matches = list(cls.SECTION_NUMBER_REGEX.finditer(s.headline))
            if len(matches) == 2:
                start, end = matches[1].span()
                new_headline = s.headline[start:].strip()
                num, form = cls.guess_serial_number_from_string(new_headline, num_format=True)
                if cls.serial_number_is_direct_subsequent(ser_num[i], num, ser_num_format[i], form):
                    new_s = Hierarchy(new_headline, s.content)
                    s.headline = s.headline[:start].strip()
                    s.content = []
                    new_sections.append(new_s)

        return new_sections

    @classmethod
    def repair_reset_counter_ser_nums(cls, ser_nums: List[Optional[Tuple[int, ...]]], sticky: Sequence[int],
                                      ser_nums_format: MutableSequence[Tuple[SerialNumberFormat, ...]],
                                      hier_lvl: int) -> Tuple[List[Optional[Tuple[int, ...]]],
    MutableSequence[Tuple[SerialNumberFormat, ...]]]:
        """
        Tries to identify subsections with reset counter and changes serial number in a way that it fits
        into hierarchy.

        It also changes the serial numbers format

        Example:
            PART (I) INTRODUCTION
            1. THE GOALS AND THE STRUCTURE OF THIS BOOK
            1.1. The Goals of this Book
            2. THE GOALS AND THE STRUCTURE OF THIS BOOK
            2.1. The Goals of this Book
            PART II. second section

            REPAIRED:
            PART (I) INTRODUCTION
            1.1. THE GOALS AND THE STRUCTURE OF THIS BOOK
            1.1.1. The Goals of this Book
            1.2. THE GOALS AND THE STRUCTURE OF THIS BOOK
            1.2.1. The Goals of this Book
            PART II. second section

            Format change:
                PART (I) INTRODUCTION
                1.1. THE GOALS AND THE STRUCTURE OF THIS BOOK   # ARABIC, ARABIC
                1.1.1. The Goals of this Book   # ARABIC, ARABIC, ARABIC
                1.2. THE GOALS AND THE STRUCTURE OF THIS BOOK   # ARABIC, ARABIC
                1.2.1. The Goals of this Book   # ARABIC, ARABIC, ARABIC
                PART II. second section

                REPAIRED:
                PART (I) INTRODUCTION
                1.1. THE GOALS AND THE STRUCTURE OF THIS BOOK   # ROMAN, ARABIC
                1.1.1. The Goals of this Book   # ROMAN, ARABIC, ARABIC
                1.2. THE GOALS AND THE STRUCTURE OF THIS BOOK   # ROMAN, ARABIC
                1.2.1. The Goals of this Book   # ROMAN, ARABIC, ARABIC
                PART II. second section

        :param ser_nums: serial numbers for alterations
        :param sticky: indices of sticky sequence numbers, those that couldn't be changed and their serial number is
            highly probable identified correctly
            should be sorted in ascending order
            if empty returns the originals (not the copy of it)
        :param ser_nums_format: formats of serial numbers
        :param hier_lvl: lvl of hierarchy on which we want to work
            as this is recursive algorithm we need to know on which hierarchy level we are
        :return: altered serial numbers and their formats
        """

        if len(sticky) == 0:
            return ser_nums, ser_nums_format

        ser_nums = list(ser_nums)
        ser_nums_format = list(ser_nums_format)

        act_lvl_sticky = [s for s in sticky if len(ser_nums[s]) == hier_lvl]
        anchor = cls.search_longest_sparse_subsequent(ser_nums, ser_nums_format,
                                                      sticky=act_lvl_sticky, convert_to_intervals=False)
        anchor_act_lvl = [a for a in anchor if len(ser_nums[a]) == hier_lvl]

        if len(anchor_act_lvl) == 0:
            return ser_nums, ser_nums_format

        # go through sections between anchors and between last anchor and end
        if anchor_act_lvl[-1] == len(ser_nums) - 1:
            # last anchor is at the very end

            starts = anchor_act_lvl[:-1]
            ends = anchor_act_lvl[1:]
        else:
            starts = anchor_act_lvl[:-1] + [anchor_act_lvl[-1]]
            ends = anchor_act_lvl[1:] + [len(ser_nums)]

        for start, end in zip(starts, ends):
            start += 1
            between = ser_nums[start:end]
            if all(b is None for b in between):
                # there are no subsections with serial numbers
                continue
            sticky_between = [s - start for s in sticky if start <= s < end]
            anchor_between = cls.search_longest_sparse_subsequent(between, ser_nums_format[start:end],
                                                                  sticky=sticky_between,
                                                                  convert_to_intervals=False)
            if len(anchor_between) > 0:
                min_len_anch = min(len(between[a]) for a in anchor_between)
                if min_len_anch <= hier_lvl:
                    # seems that the section counter is reset in this subsection, so we need to add parent serial number

                    parent = ser_nums[start - 1]
                    parent_format = ser_nums_format[start - 1]
                    for a in anchor_between:
                        ser_nums[start + a] = tuple(itertools.chain(parent, ser_nums[start + a]))
                        ser_nums_format[start + a] = tuple(itertools.chain(parent_format, ser_nums_format[start + a]))
                else:
                    # check the format
                    # there is a known problem that the grobid provides serial numbers but without their format and
                    # we insert those numbers into section headline in arabic even though it is in "roman context" as
                    # the previous is not working with hierarchical context

                    parent_format = ser_nums_format[start - 1]
                    if parent_format[hier_lvl - 1] == SerialNumberFormat.UNKNOWN:
                        # let's try the other end
                        if end == len(ser_nums_format):
                            # ok there is none, so lets try to check whether the majority is ROMAN
                            roman_cnt = sum(
                                ser_nums_format[x][hier_lvl - 1] == SerialNumberFormat.ROMAN for x in starts
                            )
                            if roman_cnt >= len(starts) / 2:
                                parent_format = list(parent_format)
                                parent_format[hier_lvl - 1] = SerialNumberFormat.ROMAN
                        else:
                            parent_format = ser_nums_format[end]

                    if parent_format[hier_lvl - 1] == SerialNumberFormat.ROMAN:
                        for a in anchor_between:
                            if ser_nums_format[start + a][0] == SerialNumberFormat.ARABIC:
                                ser_nums_format[start + a] = tuple(itertools.chain([SerialNumberFormat.ROMAN],
                                                                                   ser_nums_format[start + a][1:]))

                ser_nums[start:end], ser_nums_format[start:end] = cls.repair_reset_counter_ser_nums(
                    ser_nums[start:end], sticky_between, ser_nums_format[start:end], hier_lvl + 1)

        return ser_nums, ser_nums_format

    def flat_2_multi(self) -> bool:
        """
        Tries to guess from section structure the underlying multi lvl hierarchy. It assumes that the headlines
        contains serial numbering that can be used for guessing of this structure.
        Such a serial multi-lvl number haves levels separated with . (voluntary . at the end) and each level consists
        of serial number (could be roman (at top lvl), also a letter (not at top lvl) is allowed).
        Example of  three lvl serial number:
            II.2.1

        Illustrative example of conversion:
            I.
            II.
            II.1
            II.2
            II.2.1
            III.
            III.1
            III.2
            IV
            V

            Converts to:
                I
                II.
                    II.1
                    II.2
                        II.2.1
                III.
                    III.1
                    III.2
                IV
                V
        WARNING: It removes empty sections or sections without a headline.

        If there is no numbering found it is assumed that the hierarchy is shallow and no change is performed
        (also returns True). Example of shallow:
            Abstract
            Introduction
            Methods
            Subjects
            Measurement of confounding factors
            Statistical analyses
            Results
            Discussion

        :return: True when the multi-lvl hierarchy was established, or it assumes that the hierarchy is shallow
            False on failure. The original hierarchy will probably be in unstable state.
        """
        if not isinstance(self.content, list):
            return False

        if len(self.content) < 2 or \
                sum(h.headline is not None and self.SECTION_NUMBER_REGEX_STRICT_WITHOUT_ABC.search(h.headline) is not None for h in self.content) <= 1:
            # too few headlines with numbering prefix
            return True

        ser_num, ser_nums_formats, sticky = self.guess_content_serial_numbers(True, True)  # parsed serial numbers

        # let's select sticky section, those which are highly probable anchors

        # try to search the main structure line by merging interval of subsequents together
        anchor_one = self.search_longest_sparse_subsequent(ser_num, ser_num_formats=ser_nums_formats, sticky=sticky)
        if len(anchor_one) == 0:
            return False

        anchor_one_map = ImmutIntervalMap({(s, e - 1): True for s, e in anchor_one})
        # is used for deciding which are on the main (anchor) one

        # let's try to make some repairs
        self.fix_alone_number_headlines(self.content, anchor_one_map)

        self.content = self.split_merged_headlines(self.content, ser_num, ser_nums_formats)
        # refresh serial numbers
        ser_num, ser_nums_formats, sticky = self.guess_content_serial_numbers(True, True)

        self.content = self.merge_split_headlines(self.content, ser_num, anchor_one_map)

        # refresh serial numbers
        ser_num, ser_nums_formats, sticky = self.guess_content_serial_numbers(True, True)

        anchor_one = self.search_longest_sparse_subsequent(ser_num, ser_num_formats=ser_nums_formats, sticky=sticky)

        if len(anchor_one) == 0 or sum(e - s for s, e in anchor_one) <= 3:
            # we require at least 3 anchor elements to be more confident that the true underlying structure was
            # captured
            return False

        anchor_one_map = ImmutIntervalMap({(s, e - 1): True for s, e in anchor_one})

        # construct the multi-level hierarchy

        # let's create list of contents for current hierarchy lvls and lets initialize it with empty list that will
        # contain the newly created hierarchy
        # this structure allows us to have information about how deep in hier we are and also it stores contents of
        # active parents
        cur_anchor_hier: List[List[Hierarchy]] = [[]]
        cur_anchor_hier_numbers = [None]

        last_anchor_ser_num_format = None
        for i, section in enumerate(self.content):
            is_anchor = i in anchor_one_map
            # firstly check, according to anchor, whether the hierarchy should be changed
            if is_anchor:
                if last_anchor_ser_num_format is not None and \
                        not self.serial_number_format_match(last_anchor_ser_num_format, ser_nums_formats[i]):
                    # the number format is not matching level wise
                    return False

                last_anchor_ser_num_format = ser_nums_formats[i]

                if len(cur_anchor_hier_numbers) == len(ser_num[i]):  # the same level
                    cur_anchor_hier_numbers[-1] = ser_num[i]
                elif len(cur_anchor_hier_numbers) < len(ser_num[i]):  # deeper in hierarchy
                    cur_anchor_hier_numbers.append(ser_num[i])
                else:  # upper in hierarchy
                    cur_anchor_hier_numbers = cur_anchor_hier_numbers[:len(ser_num[i])]
                    cur_anchor_hier_numbers[-1] = ser_num[i]
                cur_anchor_hier = cur_anchor_hier[:len(ser_num[i])]

            # check the hier
            for n_index in range(1, len(cur_anchor_hier_numbers)):
                parent_n = cur_anchor_hier_numbers[n_index - 1]
                child_n = cur_anchor_hier_numbers[n_index]
                if parent_n is None or len(parent_n) >= len(child_n) or any(x != y for x, y in zip(parent_n, child_n)):
                    return False

            # stick it to the last anchor or at the top level in case of anchor
            cur_anchor_hier[-1].append(section)
            if is_anchor:
                cur_anchor_hier.append(section.content)

        self.content = cur_anchor_hier[0]  # it should contain the established hierarchy
        return True

    def insert(self, pos: Sequence[int], item: "Hierarchy"):
        """
        Inserts the item at the given position.

        :param pos: position where to insert the item
        :param item: item to insert
        """
        if len(pos) == 0:
            raise ValueError("The position cannot be empty.")
        if len(pos) == 1:
            self.content.insert(pos[0], item)
        else:
            self.content[pos[0]].insert(pos[1:], item)

    def get(self, pos: Sequence[int]) -> "Hierarchy":
        """
        Returns the item at the given position.

        :param pos: position of the item
        :return: item at the given position
        """
        if len(pos) == 0:
            raise ValueError("The position cannot be empty.")
        if len(pos) == 1:
            return self.content[pos[0]]
        else:
            return self.content[pos[0]].get(pos[1:])


class LazyHierarchy(Hierarchy):
    """
    The content is loaded only when it is needed.
    """

    def __init__(self, d: Dict):
        """
        :param d: dictionary representation of a hierarchy
        """
        self.d = d
        self._hierarchy = None

    def __getattr__(self, attr):
        if self._hierarchy is None:
            self._hierarchy = Hierarchy.from_dict(self.d, lazy_children=True)
        return getattr(self._hierarchy, attr)
