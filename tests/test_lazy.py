from meritrank_python.lazy import LazyMeritRank

from .test_rank import simple_graph


def test_get_ranks(simple_graph):
    mr = LazyMeritRank(simple_graph)
    assert mr.get_ranks("0") is not None


def test_get_node_score(simple_graph):
    mr = LazyMeritRank(simple_graph)
    assert mr.get_node_score("0", "1") is not None
