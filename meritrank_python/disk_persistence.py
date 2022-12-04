import dbm
import pickle
from enum import IntEnum

import networkx as nx

from meritrank_python.rank import NodeId

DEFAULT_DB_FILENAME = "meritrank_graph.dbm"

class EntryType(IntEnum):
    GRAPH_EDGE = 10
    RANK_CALC_COMMAND = 20


class GraphPersistentStore:
    def __init__(self, db_filename: str | None = None):
        self._db = dbm.open(db_filename or DEFAULT_DB_FILENAME, "cs")

    def put_edge(self, src: NodeId, dst: NodeId, weight: float):
        self._db[pickle.dumps((EntryType.GRAPH_EDGE, (src, dst)))] = pickle.dumps(weight)

    def close(self):
        self._db.close()

    def put_rank_calc_command(self, node: NodeId, num_walks: int):
        self._db[pickle.dumps((EntryType.RANK_CALC_COMMAND, node))] = pickle.dumps(num_walks)

    def get_graph_and_calc_commands(self):
        graph = nx.DiGraph()
        commands = {}
        k = self._db.firstkey()
        while k is not None:
            entry_type, entry_key = pickle.loads(k)
            if entry_type == EntryType.GRAPH_EDGE:
                weight = pickle.loads(self._db[k])
                graph.add_edge(*entry_key, weight=weight)
            elif entry_type == EntryType.RANK_CALC_COMMAND:
                commands[entry_key] = pickle.loads(self._db[k])
            k = self._db.nextkey(k)
        return graph, commands
