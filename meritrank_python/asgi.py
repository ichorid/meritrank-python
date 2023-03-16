from classy_fastapi import Routable, get, put
from fastapi import FastAPI
from pydantic import BaseModel

from meritrank_python.disk_persistence import GraphPersistentStore
from meritrank_python.rank import NodeId, IncrementalPageRank


class Edge(BaseModel):
    src: NodeId
    dest: NodeId
    weight: float = 1.0


class MeritRankRoutes(Routable):
    def __init__(self, rank: IncrementalPageRank) -> None:
        super().__init__()
        self.__rank = rank

    @get("/edge/{src}/{dest}")
    async def get_edge(self, src: NodeId, dest: NodeId):
        if (weight := self.__rank.get_edge(src, dest)) is not None:
            return Edge(src=src, dest=dest, weight=weight)

    @put("/edge")
    async def put_edge(self, edge: Edge):
        self.__rank.add_edge(edge.src, edge.dest, edge.weight)
        return {"message": f"Added edge {edge.src} -> {edge.dest} "
                           f"with weight {edge.weight}"}

    @put("/walks_count/{src}")
    async def put_walks_count(self, src: NodeId, count: int):
        self.__rank.calculate(src, count)

    @get("/walks_count/{src}")
    async def get_walks_count(self, src: NodeId):
        return self.__rank.get_walks_count_for_node(src)

    @get("/scores/{src}")
    async def get_scores(self, src: NodeId, limit: int | None = None):
        return self.__rank.get_ranks(src, limit=limit)

    @get("/node_score/{src}/{dest}")
    async def get_node_score(self, src: NodeId, dest: NodeId):
        return self.__rank.get_node_score(src, dest)

    @get("/node_edges/{node}")
    async def get_node_edges(self, node: NodeId):
        return self.__rank.get_node_edges(node)


def create_meritrank_app(rank_instance=None, persistent_storage=None):
    app = FastAPI()
    user_routes = MeritRankRoutes(
        rank_instance or IncrementalPageRank(persistent_storage=
            persistent_storage or GraphPersistentStore()))
    app.include_router(user_routes.router)
    return app
