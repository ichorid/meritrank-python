import dbm
import pickle

from meritrank_python.rank import NodeId

DEFAULT_DB_FILENAME = "meritrank_graph.dbm"


class GraphPersistentStore:
    def __init__(self, db_filename: str | None = None):
        self.__db = dbm.open(db_filename or DEFAULT_DB_FILENAME, "cs")

    def put_edge(self, src: NodeId, dst: NodeId, weight: float):
        key = pickle.dumps((src, dst))
        if weight != 0.0:
            self.__db[key] = pickle.dumps(weight)
        elif key in self.__db:
            # Zero weight means deleted edge
            del self.__db[key]

    def close(self):
        self.__db.close()

    def get_graph(self):
        graph = {}
        k = self.__db.firstkey()
        while k is not None:
            src, dst = pickle.loads(k)
            weight = pickle.loads(self.__db[k])
            # TODO: add warning if encountering a zero-weight edge
            graph.setdefault(src, {}).setdefault(dst, {}).update(
                {'weight': weight})
            k = self.__db.nextkey(k)
        return graph
