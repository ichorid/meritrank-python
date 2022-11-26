import random

import networkx as nx
import pytest
from _pytest.python_api import approx

from meritrank_python.rank import RandomWalk, PosWalk, WalksStorage, IncrementalPageRank


@pytest.fixture
def simple_graph():
    return nx.DiGraph(
        {0: {1: {'weight': 1}, 2: {'weight': 1}},
         1: {2: {'weight': 1}}}
    )


def test_pagerank(simple_graph):
    random.seed(1)
    # rank = nx.pagerank(simple_graph, personalization={0: 1})
    # print(rank)
    ipr = IncrementalPageRank(simple_graph)
    ipr.calculate(0)
    ranks = ipr.get_ranks(0)
    assert ranks[0] == approx(0.453, 0.01)
    assert ranks[1] == approx(0.199, 0.01)
    assert ranks[2] == approx(0.348, 0.01)


def test_pagerank_incremental(simple_graph):
    ipr = IncrementalPageRank(simple_graph)
    ipr.calculate(0)
    ranks1 = ipr.get_ranks(0)

    ipr.add_edge(2, 3)
    ranks2 = ipr.get_ranks(0)


def test_random_walk_uuid():
    assert RandomWalk().uuid != RandomWalk().uuid


def test_pos_walk():
    w = PosWalk(123, RandomWalk())
    assert w.pos == 123
    assert w.walk == []


def test_walks_storage():
    s = WalksStorage()
    walk = RandomWalk([100, 200, 300])
    s.add_walk(walk)
