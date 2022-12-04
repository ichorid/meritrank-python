import pytest

from meritrank_python.disk_persistence import GraphPersistentStore


@pytest.fixture
def stor(tmp_path_factory):
    store = GraphPersistentStore(str(tmp_path_factory.mktemp("test_disk_persistence") / "test_db.dbm"))
    yield store
    store.close()


def test_load_graph_and_commands(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("test_disk_persistence") / "test_db.dbm")
    stor = GraphPersistentStore(path)
    stor.put_edge("a", "b", 1.0)
    stor.put_rank_calc_command("a", 10)
    stor.close()

    stor = GraphPersistentStore(path)
    graph, commands = stor.get_graph_and_calc_commands()
    assert graph.edges[("a", "b")]["weight"] == 1.0
    assert commands == {"a": 10}