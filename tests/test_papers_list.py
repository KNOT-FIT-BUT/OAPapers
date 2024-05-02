# -*- coding: UTF-8 -*-
""""
Created on 02.02.22

:author:     Martin Dočekal
"""
import copy
import multiprocessing
import os
import unittest
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from shutil import copyfile, copytree, rmtree
from typing import List, Optional
from unittest import TestCase

from windpyutils.parallel.own_proc_pools import FunctorWorker, FunctorPool

from oapapers.papers_list import PapersList, ScopusPapersList, MAGPapersList, COREPapersList, \
    PapersListRecord, MAGPapersListRecord, PapersListManager, MAGMutableMemoryMappedRecordFile, SharedListOfRecords

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_PATH = os.path.join(SCRIPT_PATH, "tmp")


class PaperListManagerConsumer(FunctorWorker):
    def __init__(self, p_list: PapersList):
        super().__init__()
        self.list = p_list

    def __call__(self, items: List[PapersListRecord]) -> List[Optional[int]]:
        return self.list.batch_search(items)


class TestSharedListOfRecords(TestCase):

    def setUp(self) -> None:
        self.records = [
            PapersListRecord("On regular polytopes", 2012, ["Luis J. Boya"]),
            PapersListRecord(
                "Spectral calculations of isotropic turbulence: Efficient removal of aliasing interactions", 1971,
                ["Patterson G."]),
            PapersListRecord("Self-consistent relativistic beam equilibria", 1972, ["Rensink M.E."]),
            PapersListRecord("Energy transfer to ions from an unneutralized electron beam", 1973, ["Widner M."]),
            PapersListRecord("Expanding actions: minimality and ergodicity", 2013, ["Pablo G. Barrientos"]),
            PapersListRecord("Initial ionization in a helium shock wave", 1971, ["Kalra S.P."]),
            PapersListRecord("Thermoconvective stability of ferrofluids", 1979, ["Lalas D."]),
            PapersListRecord(
                "Parametric dependence of the density in a mirror-confined plasma subject to an ion-cyclotron instability",
                2000, ["Foote J.H."]),
            PapersListRecord("On cascade decays of squarks at the LHC in NLO QCD", 2013, ["Wolfgang Hollik"]),
        ]
        self.manager = SharedMemoryManager()
        self.manager.start()
        self.shared_list = SharedListOfRecords(copy.deepcopy(self.records), self.manager)

    def tearDown(self) -> None:
        self.manager.shutdown()

    def test_len(self):
        self.assertEqual(len(self.records), len(self.shared_list))

    def test_getitem(self):
        for i, r in enumerate(self.records):
            self.assertEqual(r, self.shared_list[i])

    def test_getitem_slice(self):
        self.assertEqual(self.records[2:5], self.shared_list[2:5])
        self.assertEqual(self.records[2:], self.shared_list[2:])
        self.assertEqual(self.records[:5], self.shared_list[:5])
        self.assertEqual(self.records[:], self.shared_list[:])
        self.assertEqual(self.records[2:5:2], self.shared_list[2:5:2])


class TestPapersList(TestCase):
    def setUp(self) -> None:
        self.titles = [
            "JokeMeter at SemEval-2020 Task 7: Convolutional humor",
            "R2-D2: A Modular Baseline for Open-Domain Question Answering",
            "Attention Is All You Need",
            "A Logical Calculus of Ideas Immanent in Nervous Activity",
            "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"
        ]

        self.years = [2020, 2021, 2017, 1943, 2021]
        self.authors = [
            ["Martin Docekal", "Martin Fajcik", "Josef Jon", "Pavel Smrz"],
            ["Martin Fajcik", "Martin Docekal", "Karel Ondrej", "Pavel Smrz"],
            [
                "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez",
                "Lukasz Kaiser", "Illia Polosukhin"],
            ["McCulloch, Warren", "Walter Pitts"],
            ["Michael M. Bronstein", "Joan Bruna", "Taco Cohen", "Petar Veličković"]
        ]
        self.records = list(PapersListRecord(t, y, a) for t, y, a in zip(self.titles, self.years, self.authors))
        self.list = PapersList(self.records)

    def test_batch_search_exact(self):
        expected = [i for i in range(len(self.titles))]
        res = self.list.batch_search(self.records)
        self.assertListEqual(expected, res)

        # reversed
        res = self.list.batch_search(list(reversed(self.records)))
        self.assertListEqual(list(reversed(expected)), res)

    def test_batch_search_exact_skip(self):
        expected = [None for _ in range(len(self.titles))]
        res = self.list.batch_search(self.records, skip_indices=[{i} for i in range(len(self.list))])
        self.assertListEqual(expected, res)

        records = [
            PapersListRecord("JokeMeter at SemEval-2020 Task 7: Convolutional humor", 2020,
                             ["Martin Docekal", "Martin Fajcik", "Josef Jon", "Pavel Smrz"]),
            PapersListRecord("JokeMeter at SemEval-2020 Task 7: Convolutional humor", 2020,
                             ["Martin Docekal", "Martin Fajcik", "Josef Jon", "Pavel Smrz"]),
        ]
        new_list = PapersList(records)
        expected = [1, 0]
        res = new_list.batch_search(records, skip_indices=[{i} for i in range(len(self.list))])
        self.assertListEqual(expected, res)

    def test_batch_search_empty(self):
        expected = [None for _ in range(len(self.titles))]
        res = PapersList([]).batch_search(self.records)
        self.assertListEqual(expected, res)

    def test_add(self):
        partial_list = PapersList(self.records[:3])
        partial_list.add(self.records[3:])

        expected = [i for i in range(len(self.titles))]
        res = partial_list.batch_search(self.records)
        self.assertListEqual(expected, res)

        # reversed
        res = partial_list.batch_search(list(reversed(self.records)))
        self.assertListEqual(list(reversed(expected)), res)

    def test_add_reset(self):
        partial_list = PapersList(self.records[:3])
        partial_list.add(self.records[3:], reset=True)

        expected = [i for i in range(len(self.titles))]
        res = partial_list.batch_search(self.records)
        self.assertListEqual(expected, res)

        # reversed
        res = partial_list.batch_search(list(reversed(self.records)))
        self.assertListEqual(list(reversed(expected)), res)

    def test_batch_search_approximate_title(self):
        records = [PapersListRecord(
            "Geometric Deep Learning Grids Groups Graphs Geodesics Gauges",
            2021,
            ["Michael M. Bronstein", "Joan Bruna", "Taco Cohen", "Petar Veličković"]
        )]

        self.assertListEqual(self.list.batch_search(records), [4])

    def test_batch_search_approximate_authors(self):
        records = [PapersListRecord(
            "A Logical Calculus of Ideas Immanent in Nervous Activity",
            1943,
            ["Warren McCulloch"]
        )]

        self.assertListEqual(self.list.batch_search(records), [3])

    def test_batch_search_unknown_title(self):
        records = [PapersListRecord(
            "The Communist Manifesto",
            2020,
            ["Martin Docekal", "Martin Fajcik", "Josef Jon", "Pavel Smrz"]
        )]

        self.assertListEqual(self.list.batch_search(records), [None])

    def test_batch_search_unknown_record(self):
        records = [PapersListRecord(
            "The Communist Manifesto",
            1848,
            ["Karl Marx"]
        )]

        self.assertListEqual(self.list.batch_search(records), [None])

    def test_batch_search_different_year(self):
        records = [PapersListRecord(
            "JokeMeter at SemEval-2020 Task 7: Convolutional humor",
            1848,
            ["Martin Docekal", "Martin Fajcik", "Josef Jon", "Pavel Smrz"]
        )]

        self.assertListEqual(self.list.batch_search(records), [None])

    def test_batch_search_different_authors(self):
        records = [PapersListRecord(
            "JokeMeter at SemEval-2020 Task 7: Convolutional humor",
            2020,
            ["Karl Marx"]
        )]

        self.assertListEqual(self.list.batch_search(records), [None])

    def test_to_other_mapping(self):
        other_list = PapersList(self.records[3:])
        other_list.add([PapersListRecord("Some title", 2100, ["Author A"])])
        self_2_other, other_2_self = self.list.to_other_mapping(other_list)

        self.assertSequenceEqual([None, None, None, 0, 1], self_2_other)
        self.assertSequenceEqual([3, 4, None], other_2_self)

    def test_to_other_mapping_without_reverse(self):
        other_list = PapersList(self.records[3:])
        other_list.add([PapersListRecord("Some title", 2100, ["Author A"])])
        self_2_other = self.list.to_other_mapping(other_list, reverse=False)

        self.assertSequenceEqual([None, None, None, 0, 1], self_2_other)


class TestPapersListStatic(TestCase):

    def test_filter_search_nearest_results(self):
        search_for = PapersListRecord("Automatic generation of textual summaries from neonatal intensive care data", 2007,
                                      ["F Portet", "E Reiter", "J Hunter", "S Sripada"])
        records = {
            0: PapersListRecord("Some title", 2020, ["K Authorovic", "S King"]),
            1: PapersListRecord("Automatic generation of textual summaries from neonatal intensive care data", 2007,
                                ["François Portet", "Ehud Reiter", "Jim Hunter", "Somayajulu Sripada"]),
            2: PapersListRecord("Completly something else", 2000, ["Author Name", "L Carlsen"]),
        }

        self.assertEqual(1, PapersList.filter_search_results((search_for, records, 0.85, 1)))


class TestPaperListFulltext(TestPapersList):
    def setUp(self) -> None:
        super().setUp()
        self.list = PapersList(self.records, fulltext_search="file::memory:?cache=shared",
                                            fulltext_search_uri_connections=True)


class TestPapersListParallel(TestPapersList):
    def setUp(self) -> None:
        super().setUp()

        self.manager = PapersListManager(ctx=multiprocessing.get_context("spawn")).__enter__()
        self.list = self.manager.PapersList(self.records, init_workers=multiprocessing.cpu_count(),
                                            features_part_size=1)
        self.list.return_self_on_enter(False)
        self.list.set_search_workers(multiprocessing.cpu_count())
        self.list.__enter__()

    def tearDown(self) -> None:
        super(TestPapersListParallel, self).tearDown()
        self.list.__exit__(None, None, None)
        self.manager.__exit__(None, None, None)

    def test_to_other_mapping(self):
        other_list = self.manager.PapersList(self.records[3:])
        other_list.add([PapersListRecord("Some title", 2100, ["Author A"])])
        self_2_other, other_2_self = self.list.to_other_mapping(other_list)

        self.assertSequenceEqual([None, None, None, 0, 1], self_2_other)
        self.assertSequenceEqual([3, 4, None], other_2_self)

    def test_to_other_mapping_without_reverse(self):
        other_list = self.manager.PapersList(self.records[3:])
        other_list.add([PapersListRecord("Some title", 2100, ["Author A"])])
        self_2_other = self.list.to_other_mapping(other_list, reverse=False)

        self.assertSequenceEqual([None, None, None, 0, 1], self_2_other)


class TestPapersListFulltextParallel(TestPapersListParallel):
    def setUp(self) -> None:
        TestPapersList.setUp(self)

        self.manager = PapersListManager(ctx=multiprocessing.get_context("spawn")).__enter__()
        self.list = self.manager.PapersList(self.records, init_workers=multiprocessing.cpu_count(),
                                            features_part_size=1, fulltext_search="file::memory:?cache=shared",
                                            fulltext_search_uri_connections=True)
        self.list.return_self_on_enter(False)
        self.list.set_search_workers(multiprocessing.cpu_count())
        self.list.__enter__()


REFERENCES_LIST_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/references.jsonl")
REFERENCES_LIST_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/references.jsonl")


class TestPapersListOnOAPapers(TestCase):
    def setUp(self) -> None:
        self.clear_tmp()
        copyfile(REFERENCES_LIST_FIXTURES_PATH, REFERENCES_LIST_TMP_PATH)
        copyfile(REFERENCES_LIST_FIXTURES_PATH+".index", REFERENCES_LIST_TMP_PATH + ".index")

    def tearDown(self) -> None:
        self.clear_tmp()

    @staticmethod
    def clear_tmp():
        for f in Path(TMP_PATH).glob('*'):
            if not str(f).endswith("placeholder"):
                os.remove(f)

    def test_from_file(self):
        papers_list = PapersList.from_file(REFERENCES_LIST_TMP_PATH)
        records = [
            PapersListRecord("This is not a review", 2011, ["Pablo G. Barrientos", "Abbas Fakhari"]),
            PapersListRecord("This is not a review: part two", 2013, ["Wolfgang Hollik", "Jonas M. Lindert"])
        ]

        self.assertSequenceEqual(records, papers_list)


SCOPUS_REVIEW_LIST_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/scopus_review_list.jsonl")
SCOPUS_REVIEW_LIST_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/scopus_review_list.jsonl")


class TestScopusReviewList(TestCase):
    def setUp(self) -> None:
        self.clear_tmp()
        copyfile(SCOPUS_REVIEW_LIST_FIXTURES_PATH, SCOPUS_REVIEW_LIST_TMP_PATH)

    def tearDown(self) -> None:
        self.clear_tmp()

    @staticmethod
    def clear_tmp():
        for f in Path(TMP_PATH).glob('*'):
            if not str(f).endswith("placeholder"):
                os.remove(f)

    def test_load(self):
        scopus_list = ScopusPapersList.from_file(SCOPUS_REVIEW_LIST_TMP_PATH)

        records = [
            PapersListRecord("On regular polytopes", 2012, ["Luis J. Boya"]),
            PapersListRecord(
                "Spectral calculations of isotropic turbulence: Efficient removal of aliasing interactions", 1971,
                ["Patterson G."]),
            PapersListRecord("Self-consistent relativistic beam equilibria", 1972, ["Rensink M.E."]),
            PapersListRecord("Energy transfer to ions from an unneutralized electron beam", 1973, ["Widner M."]),
            PapersListRecord("Expanding actions: minimality and ergodicity", 2013, ["Pablo G. Barrientos"]),
            PapersListRecord("Initial ionization in a helium shock wave", 1971, ["Kalra S.P."]),
            PapersListRecord("Thermoconvective stability of ferrofluids", 1979, ["Lalas D."]),
            PapersListRecord(
                "Parametric dependence of the density in a mirror-confined plasma subject to an ion-cyclotron instability",
                2000, ["Foote J.H."]),
            PapersListRecord("On cascade decays of squarks at the LHC in NLO QCD", 2013, ["Wolfgang Hollik"]),
        ]
        self.assertSequenceEqual(records, scopus_list)


MAG_REVIEW_LIST_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/mag_review_list.jsonl")
MAG_REVIEW_LIST_INDEX_FIXTURES_PATH = os.path.join(SCRIPT_PATH, "fixtures/mag_review_list.jsonl.index")
MAG_REVIEW_LIST_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/mag_review_list.jsonl")
MAG_REVIEW_LIST_INDEX_TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/mag_review_list.jsonl.index")

MAG_IDS = [
            2401921836, 2257060365, 4302047, 4597021, 4997732, 5285776, 6521977, 6848608, 7690530, 7996524
]


class TestMAGMutableMemoryMappedRecordFileWorker(FunctorWorker):

    def __init__(self, file: MAGMutableMemoryMappedRecordFile):
        super().__init__()
        self.file = file

    def __call__(self, d):
        return self.file[d]


class TestMAGMutableMemoryMappedRecordFile(TestCase):
    def setUp(self) -> None:
        self.file_cached = MAGPapersList.get_filtered_record_file(MAG_REVIEW_LIST_FIXTURES_PATH,
                                                                    create_cache=True,
                                                                    read_cache_size=3)
        self.file = MAGPapersList.get_filtered_record_file(MAG_REVIEW_LIST_FIXTURES_PATH,
                                                           create_cache=False,
                                                           read_cache_size=3)
        self.file_cached.open()
        self.file.open()

    def tearDown(self) -> None:
        self.file_cached.close()
        self.file.close()

    def test_get_record(self):
        self.assertSequenceEqual(MAG_IDS, [r.id for r in self.file_cached])

        self.assertSequenceEqual(MAG_IDS, [r.id for r in self.file])

    def test_get_record_multiprocessing(self):
        if multiprocessing.cpu_count() < 2:
            self.skipTest("Not enough CPUs to run multiprocessing test")
        with FunctorPool([TestMAGMutableMemoryMappedRecordFileWorker(self.file),
                          TestMAGMutableMemoryMappedRecordFileWorker(self.file)]) as pool:

            res = pool.imap(range(len(self.file)))
            self.assertSequenceEqual(MAG_IDS, [r.id for r in res])


class TestMAGPapersList(TestCase):
    def setUp(self) -> None:
        self.clear_tmp()
        copyfile(MAG_REVIEW_LIST_FIXTURES_PATH, MAG_REVIEW_LIST_TMP_PATH)
        copyfile(MAG_REVIEW_LIST_INDEX_FIXTURES_PATH, MAG_REVIEW_LIST_INDEX_TMP_PATH)
        self.mag_list = MAGPapersList.from_file(MAG_REVIEW_LIST_TMP_PATH)
        self.mag_list.open()

    def tearDown(self) -> None:
        self.mag_list.close()
        self.clear_tmp()

    @staticmethod
    def clear_tmp():
        for f in Path(TMP_PATH).glob('*'):
            if not str(f).endswith("placeholder"):
                os.remove(f)

    def test_load(self):
        """
        self.ids = ids
        self.references = references
        self.fields = fields
        self.dois = dois
        self.journals = journals
        """

        # the tests are attribute oriented because I was lazy to refactor it to record oriented version

        self.assertSequenceEqual(MAG_IDS, [r.id for r in self.mag_list])

        self.assertSequenceEqual([
            [2402455798, 2811934493, 2867207111],
            [1503348259, 1579257318, 1589752559, 1718432802, 1979650864, 2028524154, 2038037493, 2049802244, 2064466058,
             2101988878, 2499173879, 3113485264, 3205942148],
            [2407581550, 2833216158, 2876883237, 3144949113],
            [935068232, 1778722766, 1858878092, 2275420322, 2832771534],
            [1547041076, 2244216745, 2278143506, 2400509832],
            [1549339326, 1633531190, 1907361524, 1911844720, 1929025191, 2110472320, 2120240929, 2148014849],
            [876146, 6120285, 16261977, 23263222, 27559830, 28306257, 33361307, 54639937, 58760378, 73061653, 81321950,
             90809697, 96038589, 97984479, 106182960, 109494222, 112423369, 113837940, 114820422, 114941099, 117649794,
             125686373, 165262406, 167346141, 170481330, 180201837, 181948513, 187936974, 189916858, 197598611,
             199950535, 246071305, 261101351, 306182738, 315406601, 944631940, 951012466, 969145342, 995345588,
             1001454761, 1005563936, 1032835116, 1039374373, 1039753658, 1481515093, 1483371674, 1488166440, 1492746620,
             1493923481, 1495600641, 1497213276, 1497307384, 1505504347, 1505836564, 1506877155, 1508082761, 1510935459,
             1511835412, 1512883779, 1513020749, 1515319592, 1518540237, 1519240019, 1521113539, 1521187002, 1521696060,
             1524726928, 1525864834, 1527218828, 1529795546, 1537757300, 1537988972, 1539050034, 1540426766, 1540929249,
             1542512267, 1544179742, 1546876329, 1548085437, 1551044755, 1554817848, 1556791360, 1556795311, 1557349111,
             1557495970, 1560007472, 1562075484, 1564423392, 1564762074, 1566773414, 1567555710, 1576050782, 1576516419,
             1577318533, 1577693168, 1579370120, 1581086474, 1583668188, 1585045098, 1588975114, 1598650259, 1600148414,
             1600975112, 1603526996, 1607181643, 1607505103, 1615369879, 1636998272, 1645682329, 1649672561, 1671708705,
             1706899274, 1722276212, 1725201191, 1727405702, 1774893397, 1795271396, 1808149543, 1814173270, 1832525852,
             1842121164, 1853746159, 1854225135, 1859314236, 1868071268, 1879966223, 1892606334, 1897278789, 1902804281,
             1913998207, 1937505447, 1940414603, 1941661341, 1942404046, 1963005981, 2084799144, 2096325248, 2109665450,
             2118968614, 2129928720, 2135179828, 2139008586, 2144242489, 2147341278, 2153685447, 2155896693, 2158870561,
             2201062973, 2203630595, 2207355993, 2208979075, 2209989938, 2210187035, 2214129456, 2216714461, 2219240252,
             2225493308, 2226056756, 2232239542, 2232570970, 2240282392, 2240740409, 2242862076, 2242918661, 2243276628,
             2243495742, 2243521797, 2246931322, 2247689575, 2253630413, 2254202797, 2254343364, 2255313203, 2260271272,
             2261503916, 2264285415, 2264676474, 2269229607, 2270546819, 2274109746, 2274182051, 2274621943, 2275168419,
             2276876030, 2277178277, 2277254709, 2278408367, 2278639517, 2282830279, 2283857612, 2285067637, 2286553655,
             2286800249, 2287121140, 2287588557, 2288408567, 2288943982, 2294409168, 2294856869, 2300607484, 2301572622,
             2305375219, 2306002748, 2394580172, 2395540063, 2395895079, 2396316756, 2397139074, 2397297720, 2399042686,
             2399250941, 2399414123, 2400424904, 2400567843, 2402183906, 2404653053, 2404939318, 2407232256, 2407727339,
             2408612493, 2411362290, 2417021215, 2417498339, 2417622862, 2840035403, 2866306331, 2881090920, 3104124123,
             3104716380, 3106281808, 3139682597, 3140621862, 3140968216, 3141879328, 3142885647, 3143785944, 3144081109,
             3145025836, 3145335015, 3147458966, 3147788503, 3148991663, 3149752456, 3150007680, 3150765513, 3151529336,
             3151973142, 3152082660],
            [1593225443, 1663824761, 2408534236, 2820249185, 3101667206],
            [56871931, 78629751, 183508782, 204605788, 281223549, 1534814175, 1571904061, 1581796649, 2242097667,
             2262833217, 2394943148, 2395503731, 2402015347, 2404425648, 2615359577, 2849927002, 2859281351,
             3106189056],
            [2401921833, 2257060365]
        ], [list(r.references) for r in self.mag_list])

        self.assertSequenceEqual([
            ["Carbon", "Materials science", "Molecule", "Polymer chemistry", "Emulsion"],
            ["Reuse", "Inheritance (object-oriented programming)", "Extension (predicate logic)", "Datalog",
             "Computer science", "Data model", "Development (topology)", "Rule-based system", "sort",
             "Programming language"],
            ["Layer (electronics)", "Mechanical engineering", "Engineering drawing", "Cartridge", "Process (computing)",
             "Triboelectric effect", "Engineering"],
            ["Authentication", "Computer science", "Biometrics", "World Wide Web", "Computer security"],
            ["Composite material", "Materials science", "Cement", "Mortar", "Silicate"],
            ["Composite material", "Porosity", "Dispersion (chemistry)", "Materials science", "Phase (matter)",
             "Polymer", "Solvent", "Casting"],
            ["Safety shutoff valve", "Purge", "Control engineering", "Arc (geometry)", "Mechanical engineering",
             "Nozzle", "Rotation", "Turret", "Precipitation", "Flow (psychology)", "Engineering"],
            ["Chemistry", "Precipitation (chemistry)", "Chromatography", "Food science", "Flavor"],
            ["Boss", "Structural engineering", "Flange", "Rotation", "Engineering"],
            ["Styrene", "Halogen", "Ring (chemistry)", "Chemistry", "Polymer chemistry", "Alkoxy group", "Indoline",
             "Aryl", "Radical", "Alkyl"],

        ], [list(r.fields) for r in self.mag_list])

        self.assertSequenceEqual([
            None, "10.3233/978-1-58603-957-8-354", None, None, None, None, None, None, None, None
        ], [r.doi for r in self.mag_list])

        self.assertSequenceEqual([
            None, None, None, None, None, None, None, None, None, "Just for purpose of texting"
        ], [r.journal for r in self.mag_list])

        self.assertSequenceEqual([
            "Process for the production of rigid PVC foils",
            "Inheritance and Polymorphism in Datalog: an experience in Model Management",
            "Curved developer amount controlling member, developing apparatus, and process cartridge using the same",
            "System and method for authenticating a meeting",
            "Method for repairing and restoring deteriorated cement-containing inorganic material",
            "Biocompatible, porous material, method for its production and use of the same",
            "Arc adjustable rotary sprinkler having full-circle operation and automatic matched precipitation",
            "Method for making comminuted meats and extenders",
            "Universal hub assembly",
            "Process for mass-dyeing of thermoplastics: polystyrene and styrene copolymers with indoline methine dyes",
        ], [r.title for r in self.mag_list])

        self.assertSequenceEqual([
            1989, 2009, 1996, 2007, 1983, 2001, 2013, 1978, 1996, 1983,
        ], [r.year for r in self.mag_list])
        self.assertSequenceEqual([
            ["Peter Wedl", "Kurt Dr. Worschech", "Erwin Fleischer", "Ernst Udo Brand"],
            ["Giorgio Gianforme", "Paolo Atzeni"],
            ["Arihiro Yamamoto", "Kentaro Niwano", "Masahiro Watabe"],
            ["Youval Rasin", "Florian Klinger"],
            ["Toshihiko Shimizu"],
            ["Kjell Nilsson"],
            ["Jorge Alfredo Duenas Lebron", "Derek Michael Nations", "Kenneth J. Skripkar"],
            ["Nicholas Melachouris"],
            ["Clint Berscheid"],
            ["Bernhard Wehling"],
        ], [list(r.authors) for r in self.mag_list])

    def test_id_2_index(self):
        search_ids = [2401921836, 2257060365, 4302047, 4597021, 4997732, 5285776, 6521977, 6848608, 7690530, 7996524]
        self.assertSequenceEqual([i for i in range(len(search_ids))], [self.mag_list.id_2_index(i) for i in search_ids])

    def test_id_2_index_iterable(self):
        search_ids = [2401921836, 2257060365, 4302047, 4597021, 4997732, 5285776, 6521977, 6848608, 7690530, 7996524]
        self.assertSequenceEqual([i for i in range(len(search_ids))], self.mag_list.id_2_index(search_ids))

    def test_id_2_index_unknown_id(self):
        for mid in [5597021, 3, 9999999]:
            with self.assertRaises(KeyError):
                _ = self.mag_list.id_2_index(mid)

            self.assertIsNone(self.mag_list.id_2_index(mid, raise_error=False))

    def test_id_2_index_unknown_id_iterable(self):
        search_ids = [2401921836, 5597021, 4302047]
        with self.assertRaises(KeyError):
            _ = self.mag_list.id_2_index(search_ids)
        self.assertSequenceEqual([0, None, 2], self.mag_list.id_2_index(search_ids, raise_error=False))

    def test_id_2_after_add(self):
        len_before = len(self.mag_list)
        self.mag_list.add([MAGPapersListRecord(id=8690530, title="New title", year=1999, authors=["Author Name"],
                                               references=[], fields=[], doi=None, journal=None)])
        self.assertEqual(len_before, self.mag_list.id_2_index(8690530))

        records = [
            MAGPapersListRecord(id=2, title="New title", year=1999, authors=["Author Name"], references=[],
                                fields=[], doi=None, journal=None),
            MAGPapersListRecord(id=1, title="New title 2", year=1998, authors=["Author Name"], references=[],
                                fields=[], doi=None, journal=None)
        ]
        self.mag_list.add(records)
        self.assertEqual(len_before + 1, self.mag_list.id_2_index(2))
        self.assertEqual(len_before + 2, self.mag_list.id_2_index(1))

    def test_search_after_add(self):
        len_before = len(self.mag_list)
        self.mag_list.add([MAGPapersListRecord(id=8690530, title="New title", year=1999, authors=["Author Name"],
                                               references=[], fields=[], doi=None, journal=None)])
        self.assertSequenceEqual([len_before], self.mag_list.batch_search([
            PapersListRecord("New title", 1999, ["Author Name"])
        ]))

    def test_paper_with_references(self):
        mag_record, ref_indices, ref_records = self.mag_list.paper_with_references(9)

        self.assertEqual(
            MAGPapersListRecord(
                id=7996524,
                title="Process for mass-dyeing of thermoplastics: polystyrene and styrene copolymers with indoline methine dyes",
                year=1983,
                authors=["Bernhard Wehling"],
                references=[2401921833, 2257060365],
                fields=["Styrene", "Halogen", "Ring (chemistry)", "Chemistry", "Polymer chemistry", "Alkoxy group", "Indoline", "Aryl", "Radical", "Alkyl"],
                doi=None,
                journal="Just for purpose of texting"
            )
            , mag_record)

        self.assertSequenceEqual([1], ref_indices)
        self.assertSequenceEqual([
            MAGPapersListRecord(
                id=2257060365,
                title="Inheritance and Polymorphism in Datalog: an experience in Model Management",
                year=2009,
                authors=["Giorgio Gianforme", "Paolo Atzeni"],
                references=[1503348259, 1579257318, 1589752559, 1718432802, 1979650864, 2028524154, 2038037493, 2049802244, 2064466058, 2101988878, 2499173879, 3113485264, 3205942148],
                fields=["Reuse", "Inheritance (object-oriented programming)", "Extension (predicate logic)", "Datalog", "Computer science", "Data model", "Development (topology)", "Rule-based system", "sort", "Programming language"],
                doi="10.3233/978-1-58603-957-8-354",
                journal=None
            )
        ], ref_records)

class TestMAGPapersListInitFromFile(TestMAGPapersList):
    def setUp(self) -> None:
        self.clear_tmp()
        copyfile(MAG_REVIEW_LIST_FIXTURES_PATH, MAG_REVIEW_LIST_TMP_PATH)
        copyfile(MAG_REVIEW_LIST_INDEX_FIXTURES_PATH, MAG_REVIEW_LIST_INDEX_TMP_PATH)
        self.mag_list = MAGPapersList(None)
        self.mag_list.init_from_file(MAG_REVIEW_LIST_TMP_PATH)
        self.mag_list.open()


GROBID_PATH = os.path.join(SCRIPT_PATH, "fixtures/grobid")
GROBID_TMP_PATH = os.path.join(TMP_PATH, "grobid")


class TestCOREPapersList(TestCase):
    def setUp(self) -> None:
        self.clear_tmp()
        copytree(GROBID_PATH, GROBID_TMP_PATH)

    def tearDown(self) -> None:
        self.clear_tmp()

    @staticmethod
    def clear_tmp():
        if os.path.isdir(GROBID_TMP_PATH):
            rmtree(GROBID_TMP_PATH)

    def test_from_dir(self):
        core_list = COREPapersList.from_dir(GROBID_TMP_PATH)
        records = [
            PapersListRecord("DIAMOND STORAGE RING APERTURES", None, ("N Wyles", "J Jones", "H Owen", "J Varley")),
            PapersListRecord("Beam Lifetime Studies for the SLS Storage Ring", 1999, ("M Böge", "A Streun")),
            PapersListRecord("Apertures for Injection", None, ("S Tazzari",))
        ]

        self.assertSequenceEqual(records, core_list)

    def test_from_dir_parallel(self):
        if multiprocessing.cpu_count() == 1:
            self.skipTest("Skipping as there is only one core.")
            return

        core_list = COREPapersList.from_dir(GROBID_TMP_PATH, workers=multiprocessing.cpu_count(), features_part_size=1)

        records = [
            PapersListRecord("DIAMOND STORAGE RING APERTURES", None, ("N Wyles", "J Jones", "H Owen", "J Varley")),
            PapersListRecord("Beam Lifetime Studies for the SLS Storage Ring", 1999, ("M Böge", "A Streun")),
            PapersListRecord("Apertures for Injection", None, ("S Tazzari",))
        ]

        self.assertSequenceEqual(records, core_list)

    def test_identify_references(self):
        core_list = COREPapersList.from_dir(GROBID_TMP_PATH)
        records = [
            PapersListRecord("Apertures for Injection", None, ("S Tazzari",)),
            PapersListRecord("Beam Lifetime Studies for the SLS Storage Ring", 1999, ("M Böge", "A Streun")),
            PapersListRecord("unknown title", None, ("author of unknown",)),
            PapersListRecord("unknown title 2", None, ("author of unknown 2",)),
        ]
        identify_with_list = PapersList(records)
        self.assertListEqual(
            [
                [1],
                [0],
                []
            ],
            core_list.identify_references(identify_with_list)
        )


if __name__ == '__main__':
    unittest.main()
