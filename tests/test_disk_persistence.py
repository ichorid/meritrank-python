import pytest

from meritrank_python.disk_persistence import GraphPersistentStore


@pytest.fixture
def stor(tmp_path_factory):
    store = GraphPersistentStore(
        str(tmp_path_factory.mktemp("test_disk_persistence") / "test_db.dbm"))
    yield store
    store.close()


def test_load_graph(tmp_path_factory):
    path = str(
        tmp_path_factory.mktemp("test_disk_persistence") / "test_db.dbm")
    stor = GraphPersistentStore(path)
    stor.put_edge(0, 1, 1.0)
    stor.close()

    stor = GraphPersistentStore(path)
    graph = stor.get_graph()
    assert graph == {0: {1: {'weight': 1.0}}}


def test_remove_edge(stor):
    stor.put_edge(0, 1, 1.0)
    stor.put_edge(0, 2, 1.0)
    stor.put_edge(0, 1, 0.0)
    assert stor.get_graph() == {0: {2: {'weight': 1.0}}}

    stor.put_edge(0, 1, 1.0)
    assert stor.get_graph() == {0: {1: {'weight': 1.0}, 2: {'weight': 1.0}}}

    stor.put_edge(0, 3, 0.0)
    assert stor.get_graph() == {0: {1: {'weight': 1.0}, 2: {'weight': 1.0}}}
