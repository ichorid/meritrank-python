import logging

from meritrank_python.lazy import LazyMeritRank

from .test_rank import simple_graph


def test_get_ranks(simple_graph):
    mr = LazyMeritRank(simple_graph)
    assert mr.get_ranks("0") is not None


def test_get_node_score(simple_graph):
    mr = LazyMeritRank(simple_graph)
    assert mr.get_node_score("0", "1") is not None


def test_calculate(simple_graph):
    # Make sure that calculate properly counts as ego-adding operation
    mr = LazyMeritRank(simple_graph)
    mr.calculate("0")
    assert ("0") in mr.egos


def test_logging(caplog, simple_graph):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    ipr = LazyMeritRank(simple_graph, logger=logger)
    ipr.get_ranks("0")

    assert caplog.records[0].levelname == 'INFO'
