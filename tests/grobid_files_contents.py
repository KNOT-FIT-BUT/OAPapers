# -*- coding: UTF-8 -*-
""""
Created on 25.05.22

:author:     Martin Dočekal
"""
from oapapers.hierarchy import Hierarchy, TextContent, RefSpan
from oapapers.bib_entry import BibEntry

grobid_titles = [
    "DIAMOND STORAGE RING APERTURES",
    "Beam Lifetime Studies for the SLS Storage Ring",
    "Apertures for Injection"
]

grobid_years = [2010, 1999, 2002]
grobid_years_not_matched = [None, 1999, None]
grobid_authors = [
    ("N Wyles", "J Jones", "H Owen", "J Varley"),
    ("M Böge", "A Streun"),
    ("S Tazzari",)
]
grobid_hierarchy = [
    Hierarchy(
        headline="DIAMOND STORAGE RING APERTURES",
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
                                    "This paper discusses the factors contributing to the choice of beam stay-clear (BSC) in DIAMOND.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "The lifetime which results from this definition is then calculated to ensure that it is adequate for the operation of the machine.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "The BSC is defined using a semi-empirical approach by calculating the requirement for a momentum acceptance of at least 4% at all locations around the lattice.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "Allowance for injection and closed orbit errors also contribute to the overall BSC.",
                                    [], []
                                )
                            ),
                        ]
                    )
                ]
            ),
            Hierarchy(
                headline="1 FACTORS DETERMINING APERTURES",
                content=[]
            ),
            Hierarchy(
                headline="1.1 Momentum Acceptance",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence [1].",
                                    [RefSpan(0, 24, 27)], []
                                )
                            )
                        ]
                    )
                ]
            ),
            Hierarchy(
                headline="1.2 Injection",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence [2].",
                                    [RefSpan(1, 23, 26)], []
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
                                    "This is first sentence.",
                                    [], []
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
                                    "This is first sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence.",
                                    [], []
                                )
                            )
                        ]
                    ),
                ]
            ),
            Hierarchy(
                headline="APERTURE REQUIREMENTS",
                content=[]
            ),
            Hierarchy(
                headline="2.1 Contingency",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence Table 1.",
                                    [], [
                                        RefSpan(2, 23, 30)
                                    ]
                                )
                            )
                        ]
                    )
                ]
            ),
        ]
    ),
    Hierarchy(
        headline="Beam Lifetime Studies for the SLS Storage Ring",
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
                                    "This is first sentence of abstract.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence of abstract.",
                                    [], []
                                )
                            ),
                        ]
                    )
                ]
            ),
            Hierarchy(
                headline="1 FACTORS DETERMINING APERTURES",
                content=[]
            ),
            Hierarchy(
                headline="1.1 Momentum Acceptance",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence [1].",
                                    [RefSpan(0, 24, 27)], []
                                )
                            )
                        ]
                    )
                ]
            )
        ]
    ),
    Hierarchy(
        headline="Apertures for Injection",
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
                                    "This is first sentence of abstract.",
                                    [], []
                                )
                            ),
                        ]
                    )
                ]
            ),
            Hierarchy(
                headline="First headline",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is third sentence.",
                                    [], []
                                )
                            )
                        ]
                    )
                ]
            )
        ]
    ),
]

grobid_hierarchy_not_matched = [
    Hierarchy(
        headline="DIAMOND STORAGE RING APERTURES",
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
                                    "This paper discusses the factors contributing to the choice of beam stay-clear (BSC) in DIAMOND.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "The lifetime which results from this definition is then calculated to ensure that it is adequate for the operation of the machine.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "The BSC is defined using a semi-empirical approach by calculating the requirement for a momentum acceptance of at least 4% at all locations around the lattice.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "Allowance for injection and closed orbit errors also contribute to the overall BSC.",
                                    [], []
                                )
                            ),
                        ]
                    )
                ]
            ),
            Hierarchy(
                headline="1 FACTORS DETERMINING APERTURES",
                content=[]
            ),
            Hierarchy(
                headline="1.1 Momentum Acceptance",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence [1].",
                                    [RefSpan(0, 24, 27)], []
                                )
                            )
                        ]
                    )
                ]
            ),
            Hierarchy(
                headline="1.2 Injection",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence [2].",
                                    [RefSpan(1, 23, 26)], []
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
                                    "This is first sentence.",
                                    [], []
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
                                    "This is first sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence.",
                                    [], []
                                )
                            )
                        ]
                    ),
                ]
            ),
            Hierarchy(
                headline="APERTURE REQUIREMENTS",
                content=[]
            ),
            Hierarchy(
                headline="2.1 Contingency",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence Table 1.",
                                    [], [
                                        RefSpan(2, 23, 30)
                                    ]
                                )
                            )
                        ]
                    )
                ]
            ),
        ]
    ),
    Hierarchy(
        headline="Beam Lifetime Studies for the SLS Storage Ring",
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
                                    "This is first sentence of abstract.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence of abstract.",
                                    [], []
                                )
                            ),
                        ]
                    )
                ]
            ),
            Hierarchy(
                headline="1 FACTORS DETERMINING APERTURES",
                content=[]
            ),
            Hierarchy(
                headline="1.1 Momentum Acceptance",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence [1].",
                                    [RefSpan(0, 24, 27)], []
                                )
                            )
                        ]
                    )
                ]
            )
        ]
    ),
    Hierarchy(
        headline="Apertures for Injection",
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
                                    "This is first sentence of abstract.",
                                    [], []
                                )
                            ),
                        ]
                    )
                ]
            ),
            Hierarchy(
                headline="First headline",
                content=[
                    Hierarchy(
                        headline=None,
                        content=[
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is first sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is second sentence.",
                                    [], []
                                )
                            ),
                            Hierarchy(
                                headline=None,
                                content=TextContent(
                                    "This is third sentence.",
                                    [], []
                                )
                            )
                        ]
                    )
                ]
            )
        ]
    ),
]
grobid_non_plaintext = [
    [
        ("figure", "Figure 2 : Description of figure 2."),
        ("figure", "Figure 3 : Description of figure 3."),
        ("table", "Table 1 : Error table for misalignments of the storage ring.")
    ],
    [],
    []
]
grobid_bibliography = [
    {
        "b0": BibEntry(1, "Beam Lifetime Studies for the SLS Storage Ring", 1999, ("M Böge", "A Streun")),
        "b1": BibEntry(None, "Accelerateurs circulaires de particules", 1966, ("H Bruck",))
    },
    {
        "b0": BibEntry(2, "Apertures for Injection", None, ("S Tazzari",)),
        "b1": BibEntry(None, "Accelerateurs circulaires de particules", 1966, ("H Bruck",)),
    },
    {
        "b0": BibEntry(None, "Accelerateurs circulaires de particules", 1966, ("H Bruck",)),
    }
]

grobid_bibliography_not_matched = [
    {
        "b0": BibEntry(None, "Beam Lifetime Studies for the SLS Storage Ring", 1999, ("M Böge", "A Streun",)),
        "b1": BibEntry(None, "Accelerateurs circulaires de particules", 1966, ("H Bruck",))
    },
    {
        "b0": BibEntry(None, "Apertures for Injection", None, ("S Tazzari",)),
        "b1": BibEntry(None, "Accelerateurs circulaires de particules", 1966, ("H Bruck",)),
    },
    {
        "b0": BibEntry(None, "Accelerateurs circulaires de particules", 1966, ("H Bruck",)),
    }
]

title_100085 = grobid_titles[0]
year_100085 = grobid_years[0]
authors_100085 = grobid_authors[0]
hierarchy_100085 = grobid_hierarchy[0]

non_plaintext_100085 = grobid_non_plaintext[0]

bibliography_100085 = grobid_bibliography[0]
