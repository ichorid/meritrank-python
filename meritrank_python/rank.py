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

    def invalidate_walks_through_node(self, node: NodeId) -> List[RandomWalk]:
        invalidated_walks = []
        for uid, pos_walk in self._walks[node].items():
            pos, walk = pos_walk.pos, pos_walk.walk
            # For every node mentioned in the invalidated subsequence,
            # remove the corresponding entries in the bookkeeping dict
            invalidated_subsequence = walk[pos:]
            # FIXME: MOVE COUNTER?
            for affected_node in invalidated_subsequence:
                # if there is a self-reference (a loop) in the walk, don't
                # remove it here, for not to shoot ourselves in the foot.
                # Also, don't try to remove the walk twice from the same node
                if (
                        affected_node != node and
                        affected_node in self._walks and
                        uid in self._walks[affected_node]):
                    self._walks[affected_node].pop(uid)

            # Remove the invalidated subsequence from the walk
            del walk[pos:]
            invalidated_walks.append(walk)
        return invalidated_walks

    def get_walks(self, node):
        return self._walks.get(node)


class IncrementalPageRank:
    def __init__(self, graph=None, max_iter: int = 100):
        self._graph = nx.DiGraph(graph)
        self._walks = WalksStorage()
        self._personal_hits: Dict[NodeId, Counter] = {}
        self.alpha = 0.85
        self.max_iter = max_iter

    def calculate(self, src: NodeId):
        counter = self._personal_hits[src] = Counter()
        for _ in range(0, self.max_iter):
            walk = self.perform_walk(src)
            counter.update(walk)
            self._walks.add_walk(walk)

    def get_node_score(self, src: NodeId, dst: NodeId):
        counter = self._personal_hits[src]
        # TODO: optimize by caching the total?
        return counter[dst] / counter.total()

    def get_ranks(self, src: NodeId):
        counter = self._personal_hits[src]
        total = counter.total()
        return {node: count / total for node, count in counter.items()}

    def perform_walk(self, start_node: NodeId) -> RandomWalk:
        return self.continue_walk(RandomWalk([start_node]))

    def continue_walk(self, walk: RandomWalk) -> RandomWalk:
        while ((neighbours := list(self._graph.neighbors(node := walk[-1])))
               and random.random() <= self.alpha):
            weights = [self._graph[node][nbr]['weight'] for nbr in neighbours]
            walk.append(random.choices(neighbours, weights=weights, k=1)[0])
        return walk

    def add_edge(self, src: NodeId, dst: NodeId, weight: float = 1.0):
        self._graph.add_edge(src, dst, weight=weight)
        invalidated_walks = self._walks.invalidate_walks_through_node(src)
        for walk in invalidated_walks:
            self.continue_walk(walk)
            # FIXME !!!!!!!!!!!!!! continuation
            self._walks.add_walk(walk)

    def calc_rank(self, node):
        pass
