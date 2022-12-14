from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from meritrank_python.asgi import create_meritrank_app


@pytest.fixture()
def mrank():
    return Mock()


@pytest.fixture()
def client(mrank):
    return TestClient(
        app=create_meritrank_app(
            rank_instance=mrank,
            persistent_storage=Mock()))

def test_complete_init_with_default_values():
    assert TestClient(app=create_meritrank_app()).get("/edge/0/1").status_code == 200


def test_get_walks_count(mrank, client):
    mrank.get_walks_count_for_node = lambda *_: 0
    response = client.get("/walks_count/0")
    assert response.status_code == 200
    assert response.json() == 0
