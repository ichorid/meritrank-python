import networkx as nx
import pytest

from meritrank_python.rank import RandomWalk, PosWalk, WalksStorage, IncrementalPageRank


@pytest.fixture
def simple_graph():
    return nx.DiGraph(
        {0: {1: {'weight': 1}, 2: {'weight': 1}},
         1: {2: {'weight': 1}}}
    )


def test_pagerank(simple_graph):
    rank = nx.pagerank(simple_graph, personalization={0: 1})
    ipr = IncrementalPageRank(simple_graph)
    ipr.calculate(0)
    assert ipr.get_ranks(0) == rank
    print(rank)
    print(ipr.get_ranks(0))


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
