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


class NodeDoesNotExist(Exception):
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

    def add_walk(self, walk: RandomWalk, start_pos: int = 0):
        for pos, node in enumerate(walk):
            if pos < start_pos:
                continue
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

    def invalidate_walks_through_node(self, invalidated_node: NodeId) -> \
            List[Tuple[RandomWalk, RandomWalk]]:
        invalidated_walks = []
        for uid, pos_walk in self._walks.get(invalidated_node, {}).items():
            pos, walk = pos_walk.pos, pos_walk.walk
            # For every node mentioned in the invalidated subsequence,
            # remove the corresponding entries in the bookkeeping dict
            invalidated_fragment = walk[pos + 1:]
            for affected_node in invalidated_fragment:
                # if a node is encountered in the walk more than once, don't
                # remove it, for not to shoot ourselves in the foot.
                # Also, don't try to remove the walk twice from the same node
                if (affected_node != invalidated_node and
                        affected_node in self._walks and
                        uid in self._walks[affected_node]):
                    self._walks[affected_node].pop(uid)

            # Remove the invalidated subsequence from the walk
            del walk[pos + 1:]
            invalidated_walks.append((walk, invalidated_fragment))
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
        if not self._graph.has_node(src):
            raise NodeDoesNotExist

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
        walk = RandomWalk([start_node])
        self.continue_walk(walk)
        return walk

    def continue_walk(self, walk: RandomWalk):
        while ((neighbours := list(self._graph.neighbors(node := walk[-1])))
               and random.random() <= self.alpha):
            weights = [self._graph[node][nbr]['weight'] for nbr in neighbours]
            walk.append(random.choices(neighbours, weights=weights, k=1)[0])

    def add_edge(self, src: NodeId, dst: NodeId, weight: float = 1.0):
        self._graph.add_edge(src, dst, weight=weight)
        invalidated_walks = self._walks.invalidate_walks_through_node(src)
        for (walk, invalidated_fragment) in invalidated_walks:
            # Subtract the nodes in the invalidated sequence from the hit counter
            # for the starting node of the invalidated walk.
            starting_node = walk[0]
            counter = self._personal_hits[starting_node]
            counter.subtract(invalidated_fragment)

            # Finish the invalidated walk. The stochastic nature of the random walk
            # allows us to complete the walk by just continuing it until it stops naturally.
            new_fragment_start = len(walk)
            self.continue_walk(walk)
            counter.update(walk[new_fragment_start:])

            self._walks.add_walk(walk, start_pos=new_fragment_start)

    def calc_rank(self, node):
        pass
