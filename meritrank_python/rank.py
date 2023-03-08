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


class WalkStorage:

    def __init__(self):
        self.__walks: Dict[NodeId, Dict[uuid.UUID, PosWalk]] = {}

    def walks_starting_from_node(self, src: NodeId) -> list[RandomWalk]:
        """ Returns the walks starting from the given node"""
        return [pos_walk.walk for pos_walk in self.__walks.get(src, {}).values() if pos_walk.pos == 0]

    def add_walk(self, walk: RandomWalk, start_pos: int = 0):
        for pos, node in enumerate(walk):
            if pos < start_pos:
                continue
            # Get existing walks for the node. If there is none,
            # create an empty dictionary for those
            walks_with_node = self.__walks[node] = self.__walks.get(node, {})

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
                del self.__walks[affected_node][walk.uuid]
        self.__walks.get(node, {}).clear()

    def invalidate_walks_through_node(self, invalidated_node: NodeId) -> \
            List[Tuple[RandomWalk, RandomWalk]]:
        if (walks := self.__walks.get(invalidated_node)) is None:
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
                        affected_node in self.__walks and
                        uid in self.__walks[affected_node]):
                    self.__walks[affected_node].pop(uid)

            # Remove the invalidated subsequence from the walk
            del walk[pos + 1:]
            invalidated_walks.append((walk, invalidated_fragment))
        return invalidated_walks

    def get_walks(self, node):
        return self.__walks.get(node)


class IncrementalPageRank:
    def __init__(self, graph=None, persistent_storage=None):
        self.__persistent_storage = persistent_storage
        # FIXME: graph vs persistent_storage options
        rank_calc_commands = None
        if self.__persistent_storage is not None:
            graph, rank_calc_commands = self.__persistent_storage.get_graph_and_calc_commands()

        self.__graph = nx.DiGraph(graph)
        self.__walks = WalkStorage()
        self.__personal_hits: Dict[NodeId, Counter] = {}
        self.alpha = 0.85

        if rank_calc_commands is not None:
            for node, num_walks in rank_calc_commands.items():
                self.calculate(node, num_walks)

    def get_walks_count_for_node(self, src: NodeId):
        return len(self.__walks.walks_starting_from_node(src))

    def calculate(self, src: NodeId, num_walks: int = 1000):
        """
        Calculate the PageRank of the given source node.
        If there are already walks for the node, drop them and calculate anew.
        :param src: The source node to calculate the PageRank for.
        :param num_walks: The number of walks that should be used
        """

        if self.__persistent_storage is not None:
            self.__persistent_storage.put_rank_calc_command(src, num_walks)
        self.__walks.drop_walks_from_node(src)

        if not self.__graph.has_node(src):
            raise NodeDoesNotExist

        counter = self.__personal_hits[src] = Counter()
        for _ in range(0, num_walks):
            walk = self.perform_walk(src)
            counter.update(walk[1:])
            self.__walks.add_walk(walk)

    def get_node_score(self, src: NodeId, dest: NodeId):
        counter = self.__personal_hits[src]
        # TODO: optimize by caching the total?
        assert src not in counter
        return counter[dest] / counter.total()

    def get_ranks(self, src: NodeId, count=None):
        counter = self.__personal_hits[src]
        total = counter.total()
        sorted_ranks = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:count]
        return {node: count / total for node, count in sorted_ranks}

    def perform_walk(self, start_node: NodeId) -> RandomWalk:
        walk = RandomWalk([start_node])
        walk.extend(self.generate_walk_segment(start_node, stop_nodes={start_node}))
        return walk

    def generate_walk_segment(self, start_node: NodeId, stop_nodes: {NodeId} = None) -> List[NodeId]:
        node = start_node
        walk = []
        while ((neighbours := list(self.__graph.neighbors(node))) and
               random.random() <= self.alpha):
            neighbours_filtered = []
            weights = []
            for nbr in neighbours:
                # Only walk through positive edges
                if (weight := self.__graph[node][nbr]['weight']) > 0:
                    neighbours_filtered.append(nbr)
                    weights.append(weight)
            next_step = random.choices(neighbours_filtered, weights=weights, k=1)[0]
            if next_step in stop_nodes:
                break
            walk.append(next_step)
            node = next_step
        return walk

    def get_edge(self, src: NodeId, dest: NodeId) -> float | None:
        if not self.__graph.has_edge(src, dest):
            return None
        return self.__graph[src][dest]['weight']

    def get_node_edges(self, node: NodeId) -> list[tuple[NodeId, NodeId, float]] | None:
        if not self.__graph.has_node(node):
            return None
        return list(self.__graph.edges(node, data='weight'))

    def __persist_edge(self, src: NodeId, dest: NodeId, weight: float = 1.0):
        if self.__persistent_storage is not None:
            self.__persistent_storage.put_edge(src, dest, weight)

    def add_edge(self, src: NodeId, dest: NodeId, weight: float = 1.0):
        self.__graph.add_edge(src, dest, weight=weight)
        self.__persist_edge(src, dest, weight)
        invalidated_walks = self.__walks.invalidate_walks_through_node(src)
        for (walk, invalidated_fragment) in invalidated_walks:
            # Subtract the nodes in the invalidated sequence from the hit counter
            # for the starting node of the invalidated walk.
            starting_node = walk[0]
            counter = self.__personal_hits[starting_node]
            counter.subtract(invalidated_fragment)

            # Finish the invalidated walk. The stochastic nature of random walks
            # allows us to complete a walk by just continuing it until it stops naturally.
            new_fragment_start = len(walk)
            walk.extend(self.generate_walk_segment(walk[-1], stop_nodes={starting_node}))
            if walk[0] in set(walk[1:]):
                pass

            counter.update(walk[new_fragment_start:])

            self.__walks.add_walk(walk, start_pos=new_fragment_start)
