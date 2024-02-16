from meritrank_python.rank import IncrementalMeritRank, NodeId


class LazyMeritRank(IncrementalMeritRank):
    """
    This is a basic lazy-calculated variant of IncrementalMeritRank.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.num_walks = kwargs.pop("num_walks", None)
        super().__init__(*args, **kwargs)
        # The set of egos for which MeritRank has already been calculated
        self.__egos = set()

    @property
    def egos(self):
        return self.__egos


    def __maybe_add_ego(self, ego, num_walks: int = None) -> bool:
        if ego not in self.__egos:
            self.logger.debug("LazyMeritRank: Adding new ego: %s", ego)
            self.__egos.add(ego)
            super().calculate(ego, num_walks or self.num_walks)
            return True
        return False

    def calculate(self, ego: NodeId, num_walks: int = None):
        self.__maybe_add_ego(ego, num_walks or self.num_walks) or \
           super().calculate(ego, num_walks or self.num_walks)

    def get_ranks(self, ego, *args, **kwargs):
        self.__maybe_add_ego(ego)
        return super().get_ranks(ego, *args, **kwargs)

    def get_node_score(self, ego, *args, **kwargs):
        self.__maybe_add_ego(ego)
        return super().get_node_score(ego, *args, **kwargs)

