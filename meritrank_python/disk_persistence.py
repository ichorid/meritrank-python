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
        self.__db = dbm.open(db_filename or DEFAULT_DB_FILENAME, "cs")

    def put_edge(self, src: NodeId, dst: NodeId, weight: float):
        self.__db[pickle.dumps((EntryType.GRAPH_EDGE, (src, dst)))] = pickle.dumps(weight)

    def remove_edge(self, src: NodeId, dst: NodeId):
        # TODO
        pass

    def close(self):
        self.__db.close()

    def put_rank_calc_command(self, node: NodeId, num_walks: int):
        self.__db[pickle.dumps((EntryType.RANK_CALC_COMMAND, node))] = pickle.dumps(num_walks)

    def get_graph_and_calc_commands(self):
        graph = nx.DiGraph()
        commands = {}
        k = self.__db.firstkey()
        while k is not None:
            entry_type, entry_key = pickle.loads(k)
            if entry_type == EntryType.GRAPH_EDGE:
                weight = pickle.loads(self.__db[k])
                graph.add_edge(*entry_key, weight=weight)
            elif entry_type == EntryType.RANK_CALC_COMMAND:
                commands[entry_key] = pickle.loads(self.__db[k])
            k = self.__db.nextkey(k)
        return graph, commands
