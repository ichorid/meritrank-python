from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx


class NodeId(int):
    pass


class RandomWalk(List[NodeId]):
    pass


@dataclass
class IndexedWalk:
    ind: int
    walk: RandomWalk


class WalksStorage:
    def __init__(self):
        self._walks = Dict[NodeId:List[IndexedWalk]]

    def add_walk(self, new_walk: RandomWalk):
        for pos, node in enumerate(new_walk):
            # To prevent linear scanning, we store the position of the
            # node in the walk.
            new_entry = IndexedWalk(pos, new_walk)
            self._walks[node] = self._walks.get(node, []).append(new_entry)


class IncrementalPageRank:
    def __init__(self, graph=None):
        self._graph = nx.DiGraph(graph)
        self._walks = WalksStorage()

    def add_edge(self, src: NodeId, dst: NodeId, weight: float = 1.0):
        pass

    def calc_rank(self, node):
        pass
