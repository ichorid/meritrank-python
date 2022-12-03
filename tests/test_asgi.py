from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meritrank_python.asgi import MeritRankRoutes


@pytest.fixture()
def mrank():
    return Mock()


@pytest.fixture()
def client(mrank):
    app = FastAPI()
    user_routes = MeritRankRoutes(mrank)
    app.include_router(user_routes.router)
    return TestClient(app=app)


def test_get_walks_count(mrank, client):
    mrank.get_walks_count_for_node = lambda *_: 0
    response = client.get("/walks_count/0")
    assert response.status_code == 200
    assert response.json() == 0
