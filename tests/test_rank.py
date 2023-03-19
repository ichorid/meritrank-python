import random
from operator import itemgetter
from unittest.mock import Mock

import networkx as nx
import pytest
from _pytest.python_api import approx
from networkx import scale_free_graph

from meritrank_python.rank import RandomWalk, PosWalk, WalkStorage, \
    IncrementalPageRank, NodeDoesNotExist


def top_items(d, num_items=3):
    return dict(sorted(d.items(), key=itemgetter(1), reverse=True)[:num_items])


def assert_ranking_approx(d1, d2, num_top_items=5, precision=0.2):
    """
    Compare the first num_items of two ranking dicts with given precision
    """

    assert top_items(d1, num_top_items) == approx(top_items(d2, num_top_items),
                                                  rel=precision)


def test_assert_ranking_approx():
    assert_ranking_approx({1: 0.3}, {1: 0.31})


@pytest.fixture(autouse=True)
def rng():
    random.seed(1)


@pytest.fixture
def simple_graph():
    return (
        {0: {1: {'weight': 1}, 2: {'weight': 1}},
         1: {2: {'weight': 1}}}
    )


@pytest.fixture
def simple_graph_negative():
    return (
        {0: {1: {'weight': 1}, 2: {'weight': -2}},
         1: {2: {'weight': 1}}}
    )


def get_scale_free_graph(count):
    graph = scale_free_graph(count)
    weighted_graph = nx.DiGraph(seed=123)
    for edge in graph.edges():
        weighted_graph.add_edge(edge[0], edge[1], weight=1.0)
    return weighted_graph


def test_update_walk_penalties():
    empty_walk = RandomWalk()
    walk = RandomWalk([1, 2, 3, 4, 5])
    walk_repeated_node = RandomWalk([1, 2, 3, 2, 4])

    empty_negs = {}
    single_neg = {4: -1.0}
    two_negs = {3: -1.0, 4: -1.0}
    neg_not_in_walk = {100500: -1000.0}

    f = RandomWalk.calculate_penalties

    assert f(empty_walk, empty_negs) == {}
    assert f(walk, empty_negs) == {}
    assert f(walk, single_neg) == {1: -1.0, 2: -1.0, 3: -1.0}
    assert f(walk, two_negs) == {1: -2.0, 2: -2.0, 3: -1.0, }
    assert f(walk, neg_not_in_walk) == {}
    assert f(walk_repeated_node, single_neg) == {1: -1.0, 2: -1.0, 3: -1.0}
    assert f(walk_repeated_node, two_negs) == {1: -2.0, 2: -2.0, 3: -1.0}


def test_pagerank(simple_graph):
    ipr = IncrementalPageRank(simple_graph)
    ipr.calculate(0)
    ranks = ipr.get_ranks(0)
    assert ranks == approx({1: 0.354, 2: 0.645}, 0.1)

    # Test limiting the results by count
    ranks = ipr.get_ranks(0, limit=1)
    print(ranks)
    assert ranks[2] == approx(0.645, 0.1)
    assert len(ranks) == 1


def test_meritrank_negative(simple_graph_negative):
    ipr = IncrementalPageRank(simple_graph_negative)
    ipr.calculate(0)
    ranks = ipr.get_ranks(0)
    assert ranks == approx({1: 0.354, 2: 0.645}, 0.1)


def test_pagerank_incremental(simple_graph):
    orig_graph = {2: {3: {"weight": 1.0}}} | simple_graph
    ipr1 = IncrementalPageRank(orig_graph)
    ipr1.calculate(0)
    ranks_simple = ipr1.get_ranks(0)

    ipr2 = IncrementalPageRank(simple_graph)
    ipr2.calculate(0)
    ipr2.add_edge(2, 3)
    ranks_incremental = ipr2.get_ranks(0)

    assert ranks_incremental == approx(ranks_simple, 0.1)


def test_get_node_edges(simple_graph):
    ipr1 = IncrementalPageRank(simple_graph)
    assert ipr1.get_node_edges(0) == [(0, 1, 1), (0, 2, 1)]


def test_pagerank_incremental_basic():
    graph = {0: {1: {'weight': 1}}}
    graph_updated = {0: {1: {'weight': 1}, 2: {'weight': 1}}}
    ipr1 = IncrementalPageRank(graph_updated)
    ipr1.calculate(0)
    ranks_simple = ipr1.get_ranks(0)

    ipr2 = IncrementalPageRank(graph)
    ipr2.calculate(0)
    ipr2.add_edge(0, 2)
    ranks_incremental = ipr2.get_ranks(0)

    assert ranks_incremental == approx(ranks_simple, 0.1)


def test_calculate_nonexistent_node(simple_graph):
    ipr1 = IncrementalPageRank()
    with pytest.raises(NodeDoesNotExist):
        ipr1.calculate(0)


def test_pagerank_incremental_from_empty_graph():
    ipr1 = IncrementalPageRank(graph={0: {}})
    ipr1.calculate(0)
    ipr1.add_edge(0, 1, weight=1.0)
    ipr1.add_edge(0, 2, weight=1.0)
    print(ipr1.get_ranks(0))


def test_pagerank_incremental_big():
    graph = get_scale_free_graph(1000)
    ipr1 = IncrementalPageRank(graph)
    ipr1.calculate(0)
    ranks_simple = ipr1.get_ranks(0)

    ipr2 = IncrementalPageRank(graph={0: {}})
    ipr2.calculate(0)
    for edge in graph.edges():
        ipr2.add_edge(edge[0], edge[1], weight=1.0)
    ranks_incremental = ipr2.get_ranks(0)

    assert_ranking_approx(ranks_simple, ranks_incremental)


def test_drop_walks():
    s = WalkStorage()

    # Add and drop walks from a single node
    walk0 = RandomWalk([0, 1, 2, 3, 4])
    walk00 = RandomWalk([0, 2, 3, 4, 1])
    s.add_walk(walk0)
    s.add_walk(walk00)
    s.drop_walks_from_node(0)
    assert walk0.uuid not in s.get_walks(0)
    assert walk00.uuid not in s.get_walks(0)

    # Add two walks mentioning the same node, test that dropping
    # the walk from the first node does not affect the other walks
    # going through it
    walk4 = RandomWalk([4, 3, 2, 1, 0])
    s.add_walk(walk0)
    s.add_walk(walk4)
    s.drop_walks_from_node(0)
    assert walk4.uuid in s.get_walks(4)


def test_random_walk_uuid():
    assert RandomWalk().uuid != RandomWalk().uuid


def test_pos_walk():
    w = PosWalk(123, RandomWalk())
    assert w.pos == 123
    assert w.walk == []


def test_walks_storage():
    s = WalkStorage()
    walk = RandomWalk([100, 200, 300])
    s.add_walk(walk)


def test_load_graph_from_persist_store(simple_graph):
    stor = Mock()
    stor.get_graph_and_calc_commands = lambda: (simple_graph, {0: 10})
    ipr1 = IncrementalPageRank(persistent_storage=stor)
    assert ipr1.get_node_edges(0) == [(0, 1, 1), (0, 2, 1)]
    assert ipr1.get_node_score(0, 1) == 0.4


def test_persist_edge_and_calc_commands():
    stor = Mock()
    stor.get_graph_and_calc_commands = lambda: ({}, {})
    ipr1 = IncrementalPageRank(persistent_storage=stor)
    ipr1.add_edge(0, 1, weight=1.0)
    stor.put_edge.assert_called_with(0, 1, 1.0)

    ipr1.calculate(0, 2)
    stor.put_rank_calc_command.assert_called_with(0, 2)
