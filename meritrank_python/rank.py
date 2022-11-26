import random
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx


# TODO: use __slots__ to speed up stuff:
# https://tommyseattle.com/tech/python-class-dict-named-tuple-memory-and-perf.html

class NodeId(int):
    pass


class RandomWalk(List[NodeId]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uuid = uuid.uuid4()


@dataclass
class PosWalk:
    __slots__ = ['pos', 'walk']
    pos: int
    walk: RandomWalk


class WalksStorage:

    def __init__(self):
        self._walks: Dict[NodeId, Dict[uuid.UUID, PosWalk]] = {}

    def add_walk(self, walk: RandomWalk):
        for pos, node in enumerate(walk):
            # Get existing walks for the node. If there is none,
            # create an empty dictionary for those
            walks_with_node = self._walks[node] = self._walks.get(node, {})

            # Avoid storing the same walk twice in case it visits the node
            # more than once. We're only interested in the first visit, because
            # in the case of an incremental update, the part of the walk that occurs
            # after just the first visit to the node becomes invalid.
            if walk.uuid in walks_with_node:
                continue

            # To prevent linear scanning, we store the node's position in the walk
            walks_with_node[walk.uuid] = PosWalk(pos, walk)

    def get_walk(self, node):
        return self._walks.get(node)


class IncrementalPageRank:
    def __init__(self, graph=None):
        self._graph = nx.DiGraph(graph)
        self._walks = WalksStorage()
        self.alpha = 0.85
        self.max_iter = 100
        self._personal_hits: Dict[NodeId, Counter] = {}

    def calculate(self, src: NodeId):
        counter = self._personal_hits[src] = Counter()
        for _ in range(0, self.max_iter):
            walk = self.perform_walk(src)
            counter.update(walk)
            self._walks.add_walk(walk)

    def get_node_score(self, src: NodeId, dst: NodeId):
        counter = self._personal_hits[src]
        # TODO: optimize by caching total?
        return counter[dst] / counter.total()

    def get_ranks(self, src: NodeId):
        counter = self._personal_hits[src]
        total = counter.total()
        return {node: count / total for node, count in counter.items()}

    def perform_walk(self, start_node):
        walk = RandomWalk([start_node])
        while ((neighbours := list(self._graph.neighbors(node := walk[-1])))
               and random.random() <= self.alpha):
            weights = [self._graph[node][nbr]['weight'] for nbr in neighbours]
            walk.append(random.choices(neighbours, weights=weights, k=1)[0])
        return walk

    def add_edge(self, src: NodeId, dst: NodeId, weight: float = 1.0):
        pass

    def calc_rank(self, node):
        pass
