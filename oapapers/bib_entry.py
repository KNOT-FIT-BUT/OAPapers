# -*- coding: UTF-8 -*-
""""
Created on 05.09.22

:author:     Martin Dočekal
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Sequence, AbstractSet

from oapapers.cython.normalization import normalize_and_tokenize_string, \
    convert_authors_to_initials_normalized_version, \
    initial_and_normalized_authors, normalize_authors
from oapapers.matching import match_authors_groups
from oapapers.similarities import similarity_score


@dataclass
class BibEntry:
    __slots__ = ("id", "title", "year", "authors")

    id: Optional[int]  # this is id of document in dataset may be None if this document is not in it
    title: str
    year: Optional[int]
    authors: Tuple[str, ...]

    def asdict(self) -> Dict[str, Any]:
        """
        Converts this data class to dictionary.

        :return: dictionary representation of this data class
        """
        # dataclasses.asdict is too slow
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "authors": self.authors
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BibEntry":
        """
        Creates this data class from dictionary.

        :param d: the dictionary used of instantiation
        :return: bib entry
        """
        return cls(d["id"], d["title"], d["year"], tuple(d["authors"]))


@dataclass
class Bibliography:
    """
    Bibliography of a document.
    Useful for matching bib. entries.
    """

    def __init__(self, entries: Sequence[BibEntry], title_match_threshold: float = 0.75,
                 authors_match_threshold: float = 0.5, year_diff_threshold: int = 0):
        """
        :param entries: bibliography entries
        :param title_match_threshold: the minimal similarity score for title match
        :param authors_match_threshold: the minimal similarity score for authors match
        :param year_diff_threshold: the maximal difference in years for year match
        """

        self.entries = []
        self.title_match_threshold = title_match_threshold
        self.authors_match_threshold = authors_match_threshold
        self.year_diff_threshold = year_diff_threshold

        # normalized versions
        self._title_normalized = []
        self._authors_normalized = []
        self._authors_initials = []

        for e in entries:
            self.append(e)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        return self.entries[item]

    def __iter__(self):
        return iter(self.entries)

    @property
    def titles_normalized(self) -> Sequence[AbstractSet[str]]:
        """
        Normalized titles of all entries.

        :return: normalized titles
        """
        return self._title_normalized

    @property
    def authors_normalized(self) -> Sequence[Sequence[Counter[str]]]:
        """
        Normalized authors of all entries.

        :return: normalized authors
        """
        return self._authors_normalized

    @property
    def authors_initials(self) -> Sequence[Sequence[Counter[str]]]:
        """
        Normalized authors in initials form of all entries.

        :return: normalized authors in initials form
        """
        return self._authors_initials

    def append(self, entry: BibEntry):
        self.entries.append(entry)
        self._title_normalized.append(set(normalize_and_tokenize_string(entry.title)))
        init_norm_authors, norm_authors = initial_and_normalized_authors(entry.authors)
        self._authors_normalized.append(norm_authors)
        self._authors_initials.append(init_norm_authors)

    def index(self, title: Optional[str], authors: Optional[Sequence[str]], year: Optional[int],
              elimination_method: bool = True) -> int:
        """
        Finds index of the entry in bibliography using fuzzy matching.

        :param title: title
        :param authors: authors
        :param year: year
        :param elimination_method: if True then elimination method is used when there is no match with year
        :return: index of the entry
        :raises ValueError: if no match is found
        """
        init_norm_authors, norm_authors = None, None
        if authors is not None:
            init_norm_authors, norm_authors = initial_and_normalized_authors(authors)

        return self.index_prenorm(set(normalize_and_tokenize_string(title)) if title is not None else None,
                                  norm_authors,
                                  init_norm_authors,
                                  year,
                                  elimination_method)

    def index_prenorm(self, title: Optional[AbstractSet[str]],
                      authors: Optional[Sequence[Counter[str]]],
                      authors_initials: Optional[Sequence[Counter[str]]],
                      year: Optional[int],
                      elimination_method: bool = True) -> int:
        """
        Finds index of the entry in bibliography using fuzzy matching on prenormalized data.

        :param title: normalized title
        :param authors: normalized authors
        :param authors_initials: normalized authors in intials form
        :param year: year
        :param elimination_method: if True then elimination method is used when there is no match with year
        :return: index of the entry
        :raises ValueError: if no match is found
        """

        non_year_matches = []  # it allows to use elimination method, when there is unknown year in bibliography
        for i in range(len(self.entries)):

            if title is not None and not self._match_title(self._title_normalized[i], title):
                continue

            if authors is not None and not match_authors_groups(self._authors_normalized[i], self._authors_initials[i],
                                                                authors, authors_initials,
                                                                self.authors_match_threshold):
                continue

            non_year_matches.append(i)

            if (year is not None and self.entries[i].year is not None and
                    abs(self.entries[i].year - year) > self.year_diff_threshold):
                continue

            return i

        if elimination_method:
            # try elimination method on year
            if year is not None and len(non_year_matches) > 0:
                if len(non_year_matches) == 1:
                    return non_year_matches[0]
                else:
                    # if there is only one bib. entry with None
                    bib_with_none = None
                    for i in non_year_matches:
                        if self.entries[i].year is None:
                            if bib_with_none is None:
                                bib_with_none = i
                            else:
                                break
                    else:
                        if bib_with_none is not None:
                            return bib_with_none

        raise ValueError("No match found.")

    def _match_title(self, a: Optional[AbstractSet[str]], b: Optional[AbstractSet[str]]) -> bool:
        """
        Determines whether two titles match.

        :param a: one title
        :param b: another title
        :return: True if they match
        """
        return similarity_score(a, b) >= self.title_match_threshold
