import networkx as nx
import pytest


@pytest.fixture
def simple_graph():
    return nx.DiGraph(
        {0: {1: {'weight': 1}, 2: {'weight': 1}},
         1: {2: {'weight': 1}}}
    )


def test_pagerank(simple_graph):
    rank = nx.pagerank(simple_graph, personalization={0: 1})
    print (rank)


