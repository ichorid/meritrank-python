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
        return None

    @put("/calculate")
    async def calculate(self, ego_nodes: dict[NodeId, int]):
        for node, num_walks in ego_nodes.items():
            self._rank.calculate(node, num_walks)

    @get("/scores/{src}")
    async def get_scores(self, src: NodeId, count: int | None = None):
        return self._rank.get_ranks(src, count=count)

    @put("/edge")
    async def put_edge(self, edge: Edge):
        self._rank.add_edge(edge.src, edge.dest, edge.weight)
        return {"message": f"Added edge {edge.src} -> {edge.dest} "
                           f"with weight {edge.weight}"}


app = FastAPI()
user_routes = UserRoutes(IncrementalPageRank())
app.include_router(user_routes.router)
