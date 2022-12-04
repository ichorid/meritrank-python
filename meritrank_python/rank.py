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

    def walks_starting_from_node(self, src: NodeId) -> list[RandomWalk]:
        """ Returns the walks starting from the given node"""
        return [pos_walk.walk for pos_walk in self._walks.get(src, {}).values() if pos_walk.pos == 0]

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

    def drop_walks_from_node(self, node: NodeId):
        for walk in self.walks_starting_from_node(node):
            for affected_node in walk:
                del self._walks[affected_node][walk.uuid]
        self._walks.get(node, {}).clear()

    def invalidate_walks_through_node(self, invalidated_node: NodeId) -> \
            List[Tuple[RandomWalk, RandomWalk]]:
        if (walks := self._walks.get(invalidated_node)) is None:
            return []
        invalidated_walks = []
        for uid, pos_walk in walks.items():
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
    def __init__(self, graph=None, persistent_storage=None):
        self._persistent_storage = persistent_storage
        # FIXME: graph vs persistent_storage options
        rank_calc_commands = None
        if self._persistent_storage is not None:
            graph, rank_calc_commands = self._persistent_storage.get_graph_and_calc_commands()

        self._graph = nx.DiGraph(graph)
        self._walks = WalksStorage()
        self._personal_hits: Dict[NodeId, Counter] = {}
        self.alpha = 0.85

        if rank_calc_commands is not None:
            for node, num_walks in rank_calc_commands.items():
                self.calculate(node, num_walks)

    def get_walks_count_for_node(self, src: NodeId):
        return len(self._walks.walks_starting_from_node(src))

    def calculate(self, src: NodeId, num_walks: int = 1000):
        """
        Calculate the PageRank of the given source node.
        If there are already walks for the node, drop them and calculate anew.
        :param src: The source node to calculate the PageRank for.
        :param num_walks: The number of walks that should be used
        """

        if self._persistent_storage is not None:
            self._persistent_storage.put_rank_calc_command(src, num_walks)
        self._walks.drop_walks_from_node(src)

        if not self._graph.has_node(src):
            raise NodeDoesNotExist

        counter = self._personal_hits[src] = Counter()
        for _ in range(0, num_walks):
            walk = self.perform_walk(src)
            counter.update(walk)
            self._walks.add_walk(walk)

    def get_node_score(self, src: NodeId, dest: NodeId):
        counter = self._personal_hits[src]
        # TODO: optimize by caching the total?
        return counter[dest] / counter.total()

    def get_ranks(self, src: NodeId, count=None):
        counter = self._personal_hits[src]
        total = counter.total()
        sorted_ranks = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:count]
        return {node: count / total for node, count in sorted_ranks}

    def perform_walk(self, start_node: NodeId) -> RandomWalk:
        walk = RandomWalk([start_node])
        self.continue_walk(walk)
        return walk

    def continue_walk(self, walk: RandomWalk):
        while ((neighbours := list(self._graph.neighbors(node := walk[-1])))
               and random.random() <= self.alpha):
            weights = [self._graph[node][nbr]['weight'] for nbr in neighbours]
            walk.append(random.choices(neighbours, weights=weights, k=1)[0])

    def get_edge(self, src: NodeId, dest: NodeId) -> float | None:
        if not self._graph.has_edge(src, dest):
            return None
        return self._graph[src][dest]['weight']

    def get_node_edges(self, node: NodeId) -> list[tuple[NodeId, NodeId, float]] | None:
        if not self._graph.has_node(node):
            return None
        return list(self._graph.edges(node, data='weight'))

    def _persist_edge(self, src: NodeId, dest: NodeId, weight: float = 1.0):
        if self._persistent_storage is not None:
            self._persistent_storage.put_edge(src, dest, weight)

    def add_edge(self, src: NodeId, dest: NodeId, weight: float = 1.0):
        self._graph.add_edge(src, dest, weight=weight)
        self._persist_edge(src, dest, weight)
        invalidated_walks = self._walks.invalidate_walks_through_node(src)
        for (walk, invalidated_fragment) in invalidated_walks:
            # Subtract the nodes in the invalidated sequence from the hit counter
            # for the starting node of the invalidated walk.
            starting_node = walk[0]
            counter = self._personal_hits[starting_node]
            counter.subtract(invalidated_fragment)

            # Finish the invalidated walk. The stochastic nature of random walks
            # allows us to complete a walk by just continuing it until it stops naturally.
            new_fragment_start = len(walk)
            self.continue_walk(walk)
            counter.update(walk[new_fragment_start:])

            self._walks.add_walk(walk, start_pos=new_fragment_start)
