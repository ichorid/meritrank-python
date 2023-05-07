from classy_fastapi import Routable, get, put
from fastapi import FastAPI
from pydantic import BaseModel

from meritrank_python.disk_persistence import GraphPersistentStore
from meritrank_python.rank import NodeId, IncrementalMeritRank


class Edge(BaseModel):
    src: NodeId
    dest: NodeId
    weight: float = 1.0


class MeritRankRoutes(Routable):
    def __init__(self,
                 rank: IncrementalMeritRank,
                 persistent_storage: GraphPersistentStore = None) -> None:
        super().__init__()
        self.__rank = rank
        self.__persistent_storage = persistent_storage
        # The set of egos for which MeritRank has already been calculated
        self.__egos = set()

    @get("/edge/{src}/{dest}")
    async def get_edge(self, src: NodeId, dest: NodeId):
        if (weight := self.__rank.get_edge(src, dest)) is not None:
            return Edge(src=src, dest=dest, weight=weight)

    @put("/edge")
    async def put_edge(self, edge: Edge):
        if self.__persistent_storage is not None:
            self.__persistent_storage.put_edge(edge.src,
                                               edge.dest,
                                               edge.weight)
        self.__rank.add_edge(edge.src, edge.dest, edge.weight)
        return {"message": f"Added edge {edge.src} -> {edge.dest} "
                           f"with weight {edge.weight}"}

    @get("/scores/{ego}")
    async def get_scores(self, ego: NodeId, limit: int | None = None):
        self.__maybe_add_ego(ego)
        return self.__rank.get_ranks(ego, limit=limit)

    @get("/node_score/{ego}/{dest}")
    async def get_node_score(self, ego: NodeId, dest: NodeId):
        self.__maybe_add_ego(ego)
        return self.__rank.get_node_score(ego, dest)

    @get("/node_edges/{node}")
    async def get_node_edges(self, node: NodeId):
        return self.__rank.get_node_edges(node)

    def __maybe_add_ego(self, ego):
        if ego not in self.__egos:
            self.__rank.calculate(ego)


def create_meritrank_app():
    persistent_storage = GraphPersistentStore()
    rank_instance = IncrementalMeritRank(persistent_storage.get_graph())
    user_routes = MeritRankRoutes(rank_instance, persistent_storage)

    app = FastAPI()
    app.include_router(user_routes.router)
    return app
