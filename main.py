from classy_fastapi import Routable, get, put
from fastapi import FastAPI
from pydantic import BaseModel

from meritrank_python.rank import NodeId, IncrementalPageRank


class Edge(BaseModel):
    src: NodeId
    dest: NodeId
    weight: float = 1.0


class UserRoutes(Routable):
    def __init__(self, rank: IncrementalPageRank) -> None:
        super().__init__()
        self._rank = rank

    @get("/edge/{src}/{dest}")
    async def get_edge(self, src: NodeId, dest: NodeId):
        if (weight := self._rank.get_edge(src, dest)) is not None:
            return Edge(src=src, dest=dest, weight=weight)

    @put("/edge")
    async def put_edge(self, edge: Edge):
        self._rank.add_edge(edge.src, edge.dest, edge.weight)
        return {"message": f"Added edge {edge.src} -> {edge.dest} "
                           f"with weight {edge.weight}"}

    @put("/walks_count/{src}")
    async def put_walks_count(self, src: NodeId, count: int):
        self._rank.calculate(src, count)

    @get("/walks_count/{src}")
    async def get_walks_count(self, src: NodeId):
        return self._rank.get_walks_count_for_node(src)

    @get("/scores/{src}")
    async def get_scores(self, src: NodeId, count: int | None = None):
        return self._rank.get_ranks(src, count=count)

    @get("/node_score/{src}/{dest}")
    async def get_node_score(self, src: NodeId, dest: NodeId):
        return self._rank.get_node_score(src, dest)


app = FastAPI()
user_routes = UserRoutes(IncrementalPageRank())
app.include_router(user_routes.router)
