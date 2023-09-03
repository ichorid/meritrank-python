Copyright: Vadim Bulavintsev (GPL v2)

# MeritRank Python implementation

This repository contains the Python implementation for the incremental version of the MeritRank 
scoring system (which is inspired by personalized PageRank).


## Usage example
```python
from meritrank_python.rank import IncrementalMeritRank

pr = IncrementalMeritRank()

pr.add_edge(0, 1, )
pr.add_edge(0, 2, weight=0.5)
pr.add_edge(1, 2, weight=2.0)

# Initalize calculating rank from the standpoint of node "0"
pr.calculate(0)

# Get the score for node "1" from the standpoint of the node "0" 
print(pr.get_node_score(0, 1))

# Add another edge: note that the scores are automatically recalculated
pr.add_edge(2, 1, weight=3.0)
print(pr.get_node_score(0, 1))

```

## Known issues and limitations
* The bookkeeping algorithm for the incremental 
addition-deletion of edges is pretty complex.  
Initial tests show its results are equivalent to non-incremental version,
at least for all possible transitions between all possible meaningful 3- and 4-nodes graphs.
Nonetheless, it is hard to predict how the thing will work in real-life scenarios.
