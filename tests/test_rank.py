import random
from operator import itemgetter

import networkx as nx
import pytest
from _pytest.python_api import approx
from networkx import scale_free_graph

from meritrank_python.rank import RandomWalk, PosWalk, WalksStorage, IncrementalPageRank, NodeDoesNotExist


def top_items(d, num_items=3):
    return dict(sorted(d.items(), key=itemgetter(1), reverse=True)[:num_items])


def assert_ranking_approx(d1, d2, num_top_items=5, precision=0.1):
    """
    Compare the first num_items of two ranking dicts with given precision
    """

    assert top_items(d1, num_top_items) == approx(top_items(d2, num_top_items), rel=precision)


def test_assert_ranking_approx():
    assert_ranking_approx({1: 0.3}, {1: 0.31})


@pytest.fixture
def rng():
    random.seed(1)


@pytest.fixture
def simple_graph():
    return nx.DiGraph(
        {0: {1: {'weight': 1}, 2: {'weight': 1}},
         1: {2: {'weight': 1}}}
    )


def get_scale_free_graph(count):
    graph = scale_free_graph(count)
    weighted_graph = nx.DiGraph(seed=123)
    for edge in graph.edges():
        weighted_graph.add_edge(edge[0], edge[1], weight=1.0)
    return weighted_graph


def test_pagerank(simple_graph):
    # rank = nx.pagerank(simple_graph, personalization={0: 1})
    # print(rank)
    ipr = IncrementalPageRank(simple_graph)
    ipr.calculate(0)
    ranks = ipr.get_ranks(0)
    assert ranks[0] == approx(0.453, 0.01)
    assert ranks[1] == approx(0.199, 0.01)
    assert ranks[2] == approx(0.348, 0.01)


def test_pagerank_incremental(simple_graph):
    ipr1 = IncrementalPageRank(simple_graph)
    ipr1._graph.add_edge(2, 3, weight=1.0)
    ipr1.calculate(0)
    ranks_simple = ipr1.get_ranks(0)

    ipr2 = IncrementalPageRank(simple_graph)
    ipr2.calculate(0)
    ipr2.add_edge(2, 3)
    ranks_incremental = ipr2.get_ranks(0)

    assert ranks_incremental == ranks_simple


def test_pagerank_incremental_basic():
    graph = nx.DiGraph(
        {0: {1: {'weight': 1}}}
    )
    ipr1 = IncrementalPageRank(graph)
    ipr1._graph.add_edge(0, 2, weight=1.0)
    ipr1.calculate(0)
    ranks_simple = ipr1.get_ranks(0)

    ipr2 = IncrementalPageRank(graph)
    ipr2.calculate(0)
    ipr2.add_edge(0, 2)
    ranks_incremental = ipr2.get_ranks(0)

    assert ranks_incremental == ranks_simple


def test_calculate_nonexistent_node(simple_graph):
    ipr1 = IncrementalPageRank()
    with pytest.raises(NodeDoesNotExist):
        ipr1.calculate(0)


def test_pagerank_incremental_from_empty_graph():
    ipr1 = IncrementalPageRank()
    ipr1._graph.add_node(0)
    ipr1.calculate(0)
    ipr1.add_edge(0, 1, weight=1.0)
    ipr1.add_edge(0, 2, weight=1.0)
    print(ipr1.get_ranks(0))


def test_pagerank_incremental_big(rng):
    graph = get_scale_free_graph(1000)
    ipr1 = IncrementalPageRank(graph, max_iter=1000)
    ipr1.calculate(0)
    ranks_simple = ipr1.get_ranks(0)

    ipr2 = IncrementalPageRank(max_iter=1000)
    ipr2._graph.add_node(0)
    ipr2.calculate(0)
    for edge in graph.edges():
        ipr2.add_edge(edge[0], edge[1], weight=1.0)
    ranks_incremental = ipr2.get_ranks(0)

    assert_ranking_approx(ranks_simple, ranks_incremental)


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
