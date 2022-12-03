from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from meritrank_python.asgi import create_meritrank_app


@pytest.fixture()
def mrank():
    return Mock()


@pytest.fixture()
def client(mrank):
    return TestClient(app=create_meritrank_app(mrank))


def test_get_walks_count(mrank, client):
    mrank.get_walks_count_for_node = lambda *_: 0
    response = client.get("/walks_count/0")
    assert response.status_code == 200
    assert response.json() == 0
