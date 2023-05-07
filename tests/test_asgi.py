from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meritrank_python.asgi import create_meritrank_app, MeritRankRoutes


@pytest.fixture()
def mrank():
    return Mock()


@pytest.fixture()
def rank_routes(mrank):
    return MeritRankRoutes(mrank, Mock())


@pytest.fixture()
def client(rank_routes):
    app = FastAPI()
    app.include_router(rank_routes.router)
    return TestClient(app=app)


def test_complete_init_with_default_values():
    assert TestClient(app=create_meritrank_app()).get(
        "/edge/0/1").status_code == 200


def test_get_node_score(mrank, rank_routes, client):
    result = 0.999
    mrank.get_node_score = lambda *_: result
    response = client.get("/node_score/0/1")
    assert response.status_code == 200
    assert response.json() == result
