import random
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypeAlias

import networkx as nx

# TODO: use __slots__ to speed up stuff:
# https://tommyseattle.com/tech/python-class-dict-named-tuple-memory-and-perf.html

NodeId: TypeAlias = int


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
        return [pos_walk.walk for pos_walk in
                self.__walks.get(src, {}).values() if pos_walk.pos == 0]

    def add_walk(self, walk: RandomWalk, start_pos: int = 0):
        for pos, node in enumerate(walk):
            if pos < start_pos:
                continue
            # Get existing walks for the node. If there is none,
            # create an empty dictionary for those
            walks_with_node = self.__walks[node] = self.__walks.get(node, {})

            # Avoid storing the same walk twice in case it visits the node
            # more than once. We're only interested in the first visit,
            # because in the case of an incremental update, the part of the
            # walk that occurs after just the first visit to the node
            # becomes invalid.
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
            invalidated_segment = walk[pos + 1:]
            for affected_node in invalidated_segment:
                # if a node is encountered in the walk more than once, don't
                # remove it, for not to shoot ourselves in the foot.
                # Also, don't try to remove the walk twice from the same node
                if (affected_node != invalidated_node and
                        affected_node in self.__walks and
                        uid in self.__walks[affected_node]):
                    self.__walks[affected_node].pop(uid)

            # Remove the invalidated subsequence from the walk
            del walk[pos + 1:]
            invalidated_walks.append((walk, invalidated_segment))
        return invalidated_walks

    def get_walks(self, node):
        return self.__walks.get(node)


def calculate_walk_penalties(
        segment: List[NodeId], neg_weights: Dict[NodeId, float]) -> (
        Dict[NodeId, float], Dict[NodeId, float], float):
    negs_encountered: Dict[NodeId, float] = {}
    penalties_update: Dict[NodeId, float] = {}
    acc_penalty = 0.0
    for step in reversed(segment):
        if acc_penalty:
            penalties_update[step] = acc_penalty
        if penalty := neg_weights.get(step):
            negs_encountered[step] = penalty
            acc_penalty += penalty
    return negs_encountered, penalties_update, acc_penalty


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
        self.__neg_hits: Dict[NodeId, Dict[NodeId, set[NodeId]]] = {}
        self.alpha = 0.85

        if rank_calc_commands is not None:
            for node, num_walks in rank_calc_commands.items():
                self.calculate(node, num_walks)

    def get_walks_count_for_node(self, src: NodeId):
        return len(self.__walks.walks_starting_from_node(src))

    def calculate(self, ego: NodeId, num_walks: int = 1000):
        """
        Calculate the PageRank from the perspective of the given node.
        If there are already walks for the node, drop them and calculate anew.
        :param ego: The source node to calculate the PageRank for.
        :param num_walks: The number of walks that should be used
        """
        if self.__persistent_storage is not None:
            self.__persistent_storage.put_rank_calc_command(ego, num_walks)
        self.__walks.drop_walks_from_node(ego)

        if not self.__graph.has_node(ego):
            raise NodeDoesNotExist

        counter = self.__personal_hits[ego] = Counter()
        for _ in range(0, num_walks):
            walk = self.perform_walk(ego)
            counter.update(walk[1:])
            self.__walks.add_walk(walk)

    def get_node_score(self, ego: NodeId, target: NodeId):
        counter = self.__personal_hits[ego]
        # TODO: optimize by caching the result?
        assert ego not in counter
        hits = counter[target]

        # TODO: normalize the negative hits?
        hits_penalized = hits - self.__neg_hits.get(ego, {}).get(target, 0)
        return hits_penalized / counter.total()

    def get_ranks(self, ego: NodeId, limit=None):
        counter = self.__personal_hits[ego]
        total = counter.total()
        sorted_ranks = sorted(counter.items(), key=lambda x: x[1],
                              reverse=True)[:limit]
        return {node: hits / total for node, hits in sorted_ranks}

    def perform_walk(self, start_node: NodeId) -> RandomWalk:
        walk = RandomWalk([start_node])
        new_segment, stop_reason = self.__generate_walk_segment(
            start_node, stop_nodes={start_node})
        walk.extend(new_segment)
        return walk

    def __neighbours_weighted(self, node: NodeId, positive=True) -> Dict[
        NodeId, float]:
        neighbours = {}
        for nbr in self.__graph.neighbors(node):
            # Only return positive/negative neighbours
            weight = self.__graph[node][nbr]['weight']
            if weight == 0:
                continue
            if positive and weight > 0 or not positive and weight < 0:
                neighbours[nbr] = weight
        return neighbours

    def __generate_walk_segment(self, start_node: NodeId,
                                stop_nodes: {NodeId} = None) -> (
            List[NodeId], NodeId | None):
        node = start_node
        walk = []
        stop_reason = None
        while ((neighbours := self.__neighbours_weighted(node))
               and random.random() <= self.alpha):
            peers, weights = zip(*neighbours.items())
            next_step = random.choices(peers, weights=weights, k=1)[0]
            if stop_reason := next_step in stop_nodes:
                break
            walk.append(next_step)
            node = next_step
        return walk, stop_reason

    def get_edge(self, src: NodeId, dest: NodeId) -> float | None:
        if not self.__graph.has_edge(src, dest):
            return None
        return self.__graph[src][dest]['weight']

    def get_node_edges(self, node: NodeId) -> list[tuple[
        NodeId, NodeId, float]] | None:
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
        for (walk, invalidated_segment) in invalidated_walks:
            # Subtract the nodes in the invalidated sequence from the hit
            # counter for the starting node of the invalidated walk.
            ego = walk[0]
            counter = self.__personal_hits[ego]
            counter.subtract(invalidated_segment)

            # Finish the invalidated walk. The stochastic nature of random
            # walks allows us to complete a walk by just continuing it until
            # it stops naturally.
            new_segment_start = len(walk)
            new_segment, stop_reason = self.__generate_walk_segment(
                walk[-1], stop_nodes={ego})
            self.__update_negative_hits(walk, new_segment, invalidated_segment)
            walk.extend(new_segment)

            counter.update(walk[new_segment_start:])
            self.__walks.add_walk(walk, start_pos=new_segment_start)

    def __update_negative_hits(self, walk: RandomWalk,
                               new_segment: List[NodeId],
                               invalidated_segment: List[NodeId]):
        # FIXME: case for changes in negs set (removal or change of a neg)
        # Penalty calculation logic:
        # 1. The penalty is accumulated by walking backwards from the last
        # node in the segment.
        # 2. If a node is encountered in the walk more than once, its penalty
        # is updated to the highest current accumulated penalty
        # 3. If a penalty-inducing node (called a "neg" for short)
        # is encountered more than once, its effect is not accumulated.
        #
        # In a sense, every neg in the walk produces a "tag", so each node
        # in the walk leading up to a given neg is "tagged" by it,
        # and then each "tagged" node is penalized according
        # to the weight of the "tags" associated with the negs.
        #
        # Example:
        # nodes D and F both are negs of weight 1
        # node B is repeated twice in positions 2-3
        #         ◄─+tag F────────┐
        #         ◄─+tag D──┐     │
        #        ┌──────────┴─────┴────┐
        #        │ A  B  B  D  E  F  G │
        #        └─────────────────────┘
        # Resulting penalties for the nodes:
        # node     A  -  B  D  E  F  G
        # "tags"   DF -  DF F  F
        # penalty  2  -  2  1  1  0  0
        #
        ego = walk[0]
        ego_negs = self.__neighbours_weighted(ego, positive=False)
        if not (set(ego_negs.keys()) &
                set(walk + new_segment + invalidated_segment)):
            return
        ego_neg_hits = self.__neg_hits[ego] = self.__neg_hits.get(ego, {})

        # Note this can be further optimized by memorizing the positions of
        # the negs during the walk generation.

        # First, clean the invalidated negs from the original walk
        old_negs, old_penalties, total_penalty_to_remove = calculate_walk_penalties(
            invalidated_segment, ego_negs)
        for node, penalty in old_penalties:
            ego_neg_hits[node] -= penalty

        # Next, pass through the new segment,
        # collect new negs and simultaneously apply them
        new_negs, new_penalties, total_penalty_to_add = calculate_walk_penalties(
            new_segment, ego_negs)
        for node, penalty in new_penalties:
            ego_neg_hits[node] = ego_neg_hits.get(node, 0) + penalty

        net_penalty = total_penalty_to_add - total_penalty_to_remove
        # Now pass through the unchanged walk segment,
        # both removing the old penalties and adding the new ones
        for step in reversed(walk[1:]):
            # ACTHUNG! if there is an encounter of a neg to remove in the
            # walk segment to keep, we must re-add that neg to the walk,
            # from that point.
            net_penalty += old_negs.pop(step, 0)

            ego_neg_hits[step] += net_penalty
