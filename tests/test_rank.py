import logging
import random
from operator import itemgetter

import networkx as nx
import pytest
from _pytest.python_api import approx
from networkx import scale_free_graph

from meritrank_python.rank import RandomWalk, PosWalk, WalkStorage, \
    IncrementalMeritRank, NodeDoesNotExist, SelfReferenceNotAllowed

IncrementalMeritRank.ASSERT = True

def top_items(d, num_items=3):
    return dict(sorted(d.items(), key=itemgetter(1), reverse=True)[:num_items])


def assert_ranking_approx(d1, d2, num_top_items=5, precision=0.1):
    """
    Compare the first num_items of two ranking dicts with given precision
    """

    assert top_items(d1, num_top_items) == approx(top_items(d2, num_top_items),
                                                  rel=precision)


def test_assert_ranking_approx():
    assert_ranking_approx({'1': 0.3}, {'1': 0.31})


@pytest.fixture(autouse=True)
def rng():
    random.seed('1')


@pytest.fixture
def simple_graph():
    return (
        {'0': {'1': {'weight': 1.0}, '2': {'weight': 1.0}},
         '1': {'2': {'weight': 1.0}}}
    )


@pytest.fixture
def simple_graph_negative():
    return (
        {'0': {'1': {'weight': 1.0},
               '2': {'weight': 1.0},
               '3': {'weight': -1.0}},
         '1': {'2': {'weight': 1.0}},
         '2': {'3': {'weight': 1.0}}
         }
    )


@pytest.fixture
def spi(simple_graph_negative):
    ipr = IncrementalMeritRank(simple_graph_negative)
    ipr.calculate('0')
    return ipr


def get_scale_free_graph(count):
    graph = scale_free_graph(count)
    weighted_graph = nx.DiGraph(seed=123)
    for edge in graph.edges():
        weighted_graph.add_edge(str(edge[0]), str(edge[1]), weight=1.0)
    return weighted_graph


def test_logging(caplog, simple_graph):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    ipr = IncrementalMeritRank(simple_graph, logger=logger)
    ipr.calculate('0')

    assert caplog.records[0].levelname == 'INFO'


def test_update_walk_penalties():
    empty_walk = RandomWalk()
    walk = RandomWalk(['1', '2', '3', '4', '5'])
    walk_repeated_node = RandomWalk(['1', '2', '3', '2', '4'])

    empty_negs = {}
    single_neg = {'4': -1.0}
    two_negs = {'3': -1.0, '4': -1.0}
    neg_not_in_walk = {100500: -1000.0}

    f = RandomWalk.calculate_penalties

    assert f(empty_walk, empty_negs) == {}
    assert f(walk, empty_negs) == {}
    assert f(walk, single_neg) == {'1': -1.0, '2': -1.0, '3': -1.0, '4': -1.0}
    assert f(walk, two_negs) == {'1': -2.0, '2': -2.0, '3': -2.0, '4': -1.0}
    assert f(walk, neg_not_in_walk) == {}
    assert f(walk_repeated_node, single_neg) == {'1': -1.0, '2': -1.0, '3': -1.0,
                                                 '4': -1.0}
    assert f(walk_repeated_node, two_negs) == {'1': -2.0, '2': -2.0, '3': -2.0,
                                               '4': -1.0}


def test_correct_removal_of_repeating_nodes_in_walk():
    # If there is a chain of nodes recursively pointing at each other,
    # there can be walks with repeated nodes, e.g. "abcbcbcbc". For such
    # walks, special care should be taken to correctly remove from the walks
    # storage all the references belonging to the nodes from the repeating
    # section, all the while not screwing up the bookkeeping
    graph = {'0': {'1': {'weight': 1.0}},
             '1': {'2': {'weight': 1.0}, '0': {'weight': 1.0}},
             '2': {'1': {'weight': 1.0}},
             }
    ipr = IncrementalMeritRank(graph)
    ipr.calculate('0')
    # ACHTUNG! This sequential deletion of edges is here
    # to check a particularly nasty bug!
    # DO NOT TRY TO OPTIMIZE THIS TEST!
    ipr.add_edge('1', '2', weight=0.0)
    ipr.add_edge('0', '1', weight=0.0)
    # Every connection of '0' to external nodes was deleted, so the resulting
    # rankings dict should be empty
    assert ipr.get_ranks('0') == {'0': 1.0}


def test_correct_removal_of_repeating_nodes2():
    graph = {'0': {'2': {'weight': 1.0}},
             '2': {'0': {'weight': 1.0}, '1': {'weight': 1.0}},
             '1': {'2': {'weight': 1.0}}}
    ipr = IncrementalMeritRank(graph)
    ipr.calculate('0')
    # ACHTUNG! This sequential deletion of edges is here
    # to check a particularly nasty bug!
    # DO NOT TRY TO OPTIMIZE THIS TEST!
    ipr.add_edge('1', '2', 0.0)
    ipr.add_edge('1', '0', 1.0)
    ipr.add_edge('1', '0', 0.0)
    ipr.add_edge('2', '0', 0.0)
    ipr.add_edge('2', '1', 0.0)
    assert ipr.get_ranks('0') == approx({'0': 0.53, '2': 0.46}, 0.1)


def test_pagerank(simple_graph):
    ipr = IncrementalMeritRank(simple_graph)
    ipr.calculate('0')
    ranks = ipr.get_ranks('0')
    assert ranks == approx({'0': 0.453, '1': 0.193, '2': 0.35}, 0.1)

    # Test limiting the results by count
    ranks = ipr.get_ranks('0', limit=2)
    assert len(ranks) == 2
    assert ranks['2'] == approx(0.35, 0.1)


def test_add_edge_zz(spi):
    # Empty add - noop
    orig = spi.get_ranks('0')
    spi.add_edge('0', '4', weight=0.0)
    assert spi.get_ranks('0') == orig


def test_add_edge_zp(spi):
    # '2' adds a new friend 4, and because it shares some gratitude
    # with it, 4 gets ahead of '3' and '1'
    spi.add_edge('2', '4', weight=1.0)
    assert spi.get_ordered_peers('0') == ['0', '2', '4', '1', '3']


def test_add_edge_zn(spi):
    # Negative add
    # '1' adds a new friend 4, but then '0' votes 4 down,
    # so '1' gets demoted because now paths going through it lead to
    # '0''s new enemy 4
    spi.add_edge('1', '4', weight=1.0)
    spi.add_edge('0', '4', weight=-1.5)
    assert spi.get_ordered_peers('0') == ['0', '2', '1', '3', '4']


def test_add_edge_pz(spi):
    # Remove positive edge
    # '2' breaks his friendship with '3', and then '2' becomes '0''s best friend
    # because '2' is still voted up by '1', and '2' gets no more penalties
    # from befriending '0''s enemy '3'
    spi.add_edge('2', '3', weight=0.0)
    assert spi.get_ordered_peers('0') == ['0', '2', '1']


def test_add_edge_pp(spi):
    # Change weight of a positive edge
    # '0' increases his opinion about '2' tenfold, so now
    # '2' becomes '0''s best friend despite '2''s compromising connection with '3'
    spi.add_edge('0', '2', weight=10)
    assert spi.get_ordered_peers('0') == ['0', '2', '1', '3']


def test_add_edge_pn(spi):
    # Change positive edge to negative edge
    # '0' changes its opinion about '1' to negative, so
    # now '2' becomes '0''s best friend
    # Peculiarly, '3' becomes a better in '0''s eyes then '1', because
    # '3' is at least voted by '2', who is valued by '0'
    spi.add_edge('0', '1', weight=-1.0)
    assert spi.get_ordered_peers('0') == ['0', '2', '3']


def test_basic_negative(spi):
    # Just test that the order of the test graph is as we expect
    assert spi.get_ordered_peers('0') == ['0', '1', '2', '3']


def test_add_edge_nz(spi):
    # Remove negative edge
    # '0' removes his negative opinion about '3', so now '2' is '0''s best friend
    spi.add_edge('0', '3', weight=0.0)
    assert spi.get_ordered_peers('0') == ['0', '2', '3', '1']


def test_get_ordered_peers_limit(spi):
    assert len(spi.get_ordered_peers('0', limit=2)) == 2


def test_add_edge_np(spi):
    # Change negative edge to positive edge
    # '0' changes its negative opinion about '3' to positive, so
    # now '3' becomes '0''s best friend, and '2' second best,
    # because the vote-chain now looks like '1'->'2'->'3'
    # without any penalties
    spi.add_edge('0', '3', weight=1.0)
    assert spi.get_ordered_peers('0') == ['0', '3', '2', '1']


def test_add_edge_nn(spi):
    # Change the weight of a negative edge
    # First, '1' gets a new friend 4, and '0' hates 4, so '1' gets
    # lower than '2' on the list of '0''s preferences.
    spi.add_edge('1', '4', weight=1.0)
    spi.add_edge('0', '4', weight=-5.0)
    assert spi.get_ordered_peers('0') == ['2', '3', '0', '1', '4']
    # However, after that '0' pulls the world of hate upon '3',
    # which results in '1', '2' and '3' pushed to the bottom of '0''s priorities
    spi.add_edge('0', '3', weight=-100)
    assert spi.get_ordered_peers('0') == ['4', '1', '2', '3', '0']


def test_add_edge_commutativity(simple_graph):
    ipr1 = IncrementalMeritRank(simple_graph)
    ipr1.calculate('0')
    ipr1.add_edge('0', '3', weight=-1.0)
    ipr1.add_edge('1', '3', weight=1.0)

    ipr2 = IncrementalMeritRank(simple_graph)
    ipr2.calculate('0')
    ipr2.add_edge('1', '3', weight=1.0)
    ipr2.add_edge('0', '3', weight=-1.0)

    assert ipr1.get_ranks('0') == approx(ipr2.get_ranks('0'), 0.1)


def test_meritrank_negative(spi):
    assert spi.get_ranks('0') == approx({'0': 0.118, '1': 0.04, '2': 0.041, '3': 0.00},
                                        0.1)


def test_pagerank_incremental(simple_graph):
    orig_graph = {'2': {'3': {"weight": 1.0}}} | simple_graph
    ipr1 = IncrementalMeritRank(orig_graph)
    ipr1.calculate('0')
    ranks_simple = ipr1.get_ranks('0')

    ipr2 = IncrementalMeritRank(simple_graph)
    ipr2.calculate('0')
    ipr2.add_edge('2', '3')
    ranks_incremental = ipr2.get_ranks('0')

    assert ranks_incremental == approx(ranks_simple, 0.1)


def test_get_node_edges(simple_graph):
    ipr1 = IncrementalMeritRank(simple_graph)
    assert ipr1.get_node_edges('0') == [('0', '1', 1.0), ('0', '2', 1.0)]

def test_get_node_edges_no_node(simple_graph):
    ipr1 = IncrementalMeritRank(simple_graph)
    assert pytest.raises(NodeDoesNotExist, ipr1.get_node_edges, "aaa")


def test_pagerank_incremental_basic():
    graph = {'0': {'1': {'weight': 1.0}}}
    graph_updated = {'0': {'1': {'weight': 1.0}, '2': {'weight': 1.0}}}
    ipr1 = IncrementalMeritRank(graph_updated)
    ipr1.calculate('0')
    ranks_simple = ipr1.get_ranks('0')

    ipr2 = IncrementalMeritRank(graph)
    ipr2.calculate('0')
    ipr2.add_edge('0', '2')
    ranks_incremental = ipr2.get_ranks('0')

    assert ranks_incremental == approx(ranks_simple, 0.1)


def test_calculate_nonexistent_node(simple_graph):
    ipr1 = IncrementalMeritRank()
    with pytest.raises(NodeDoesNotExist):
        ipr1.calculate('0')


def test_forbid_self_reference_edges():
    graph = {'0': {'0': {'weight': 1.0}}}
    with pytest.raises(SelfReferenceNotAllowed):
        IncrementalMeritRank(graph)

    ipr1 = IncrementalMeritRank()
    with pytest.raises(SelfReferenceNotAllowed):
        ipr1.add_edge('0', '0')


def test_pagerank_incremental_from_empty_graph():
    ipr1 = IncrementalMeritRank(graph={'0': {}})
    ipr1.calculate('0')
    ipr1.add_edge('0', '1', weight=1.0)
    ipr1.add_edge('0', '2', weight=1.0)


def test_pagerank_incremental_add_loop():
    graph_orig = {'0': {'1': {'weight': -1.0}, '2': {'weight': 1.0}},
                  '2': {'1': {'weight': 1.0}}}
    graph_updated = {'0': {'1': {'weight': -1.0}, '2': {'weight': 1.0}},
                     '2': {'1': {'weight': 1.0}, '0': {'weight': 1.0}}}
    ipr1 = IncrementalMeritRank(graph_updated)
    ipr1.calculate('0')

    ipr2 = IncrementalMeritRank(graph_orig)
    ipr2.calculate('0')
    ipr2.add_edge('2', '0', 1.0)
    assert ipr2.get_ranks('0') == approx(ipr1.get_ranks('0'), 0.1)


def test_pagerank_incremental_add_edge_from_loop_source():
    graph_orig = {'0': {'1': {'weight': -1.0}, '2': {'weight': 1.0}},
                  '2': {'0': {'weight': 1.0}}}
    graph_updated = {'0': {'1': {'weight': -1.0}, '2': {'weight': 1.0}},
                     '2': {'1': {'weight': 1.0}, '0': {'weight': 1.0}}}
    ipr1 = IncrementalMeritRank(graph_updated)
    ipr1.calculate('0')

    ipr2 = IncrementalMeritRank(graph_orig)
    ipr2.calculate('0')
    ipr2.add_edge('2', '1', 1.0)
    assert ipr2.get_ranks('0') == approx(ipr1.get_ranks('0'), 0.1)


def test_pagerank_incremental_delete_edge():
    graph_orig = {'0': {'1': {'weight': -1.0}, '2': {'weight': 1.0}},
                  '2': {'3': {'weight': 1.0}, '1': {'weight': 1.0}}}
    graph_updated = {'0': {'1': {'weight': -1.0}, '2': {'weight': 1.0}},
                     '2': {'1': {'weight': 1.0}}}
    ipr1 = IncrementalMeritRank(graph_updated)
    ipr1.calculate('0')

    ipr2 = IncrementalMeritRank(graph_orig)
    ipr2.calculate('0')
    ipr2.add_edge('2', '3', 0.0)
    assert ipr2.get_ranks('0') == approx(ipr1.get_ranks('0'), 0.1)


def test_pagerank_incremental_delete_loop():
    graph_orig = {'0': {'1': {'weight': -1.0}, '2': {'weight': 1.0}},
                  '2': {'0': {'weight': 1.0}, '1': {'weight': 1.0}}}
    graph_updated = {'0': {'1': {'weight': -1.0}, '2': {'weight': 1.0}},
                     '2': {'1': {'weight': 1.0}}}
    ipr1 = IncrementalMeritRank(graph_updated)
    ipr1.calculate('0')

    ipr2 = IncrementalMeritRank(graph_orig)
    ipr2.calculate('0')
    ipr2.add_edge('2', '0', 0.0)
    assert ipr2.get_ranks('0') == approx(ipr1.get_ranks('0'), 0.1)


@pytest.mark.skip(reason="slow and fails for non-obvious reasons")
def test_pagerank_incremental_big():
    graph = get_scale_free_graph(1000)
    graph = graph.reverse()
    # Remove self-references - we don't tolerate that
    for node in graph.nodes():
        if graph.has_edge(node, node):
            graph.remove_edge(node, node)

    ipr1 = IncrementalMeritRank(graph)
    ipr1.calculate('0')

    ipr3 = IncrementalMeritRank(graph={'0': {}})
    ipr3.calculate('0')
    lll = list(graph.edges())
    random.shuffle(lll)
    for edge in lll:
        ipr3.add_edge(edge[0], edge[1], weight=1.0)
    assert_ranking_approx(ipr1.get_ranks('0'), ipr3.get_ranks('0'), precision=0.1)

def test_many_egos_change_edge(simple_graph_negative):
    ipr = IncrementalMeritRank()
    ipr.add_edge("a", "b")
    ipr.calculate("a", num_walks=1)
    ipr.calculate("b", num_walks=1)

    ipr.add_edge("a","b", 2.0)



def test_drop_walks():
    s = WalkStorage()

    # Add and drop walks from a single node
    walk0 = RandomWalk(['0', '1', '2', '3', '4', '1'])
    walk00 = RandomWalk(['0', '2', '3', '4', '1', '2'])
    s.add_walk(walk0)
    s.add_walk(walk00)
    s.drop_walks_from_node('0')
    assert walk0.uuid not in s.get_walks('0')
    assert walk00.uuid not in s.get_walks('0')

    # Add two walks mentioning the same node, test that dropping
    # the walk from the first node does not affect the other walks
    # going through it
    walk4 = RandomWalk(['4', '3', '2', '1', '0'])
    s.add_walk(walk0)
    s.add_walk(walk4)
    s.drop_walks_from_node('0')
    assert walk4.uuid in s.get_walks('4')


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
