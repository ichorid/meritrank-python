Copyright: Vadim Bulavintsev (GPL v2)

# MeritRank Python implementation

This repository contains the Python implementation for the incremental version of the MeritRank 
scoring system (which is inspired by personalized PageRank). The implementation is broken down into several modules,
namely:
* `meritrank_python.rank`: incremental PageRank with in-memory data structures
* `meritrank_python.disk_persistence`: and `mdb`-based persistence layer for storing the rank state on disk
* `meritrank_python.asgi`: a FastAPI-based API for the ranking system (persists data to disk by default)

### Persistence
Two types of data are saved to disk on corresponding events:
* edges - on `put_edge` event
* the number of required walks for given nodes - on `put_walks_count` event

On restart, the API will load the edges data and the commands to redo the walks
for given nodes. Effectively, this will recompute the ranking and
the auxillary structures (e.g. bookkeeping for incremental ranks).
The logic here is that restarts occur rarely, and it is much easier and more efficient
to persist just the edges and the walk-starting nodes.
With the default configuration, the persistence layer will persist data 
into `meritrank_graph.dbm` in the working dir (the repo dir).

## Usage example
```python
from meritrank_python.rank import IncrementalPageRank

pr = IncrementalPageRank()

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

## HTTP API usage (ASGI)
The basic usage is covered in the test suite. 
To run the FastAPI-based ASGI implementation:
```commandline
poetry install
poetry shell
uvicorn meritrank_python.asgi:create_meritrank_app --reload --factory
```
If all runs fine, you should be able to point your browser 
to `http://127.0.0.1:8000/docs`, see the autogenerated Swagger documentation
and experiment with the API in-browser. Note the basic run options will persist the 
database on disk 

## Known issues and limitations
* Currently, the only implemented algorithm is incremental personal PageRank. 
* The `NodeID` type is `int` - should be changed to something more general, e.g. `bytes`
* No security/authorization

##
* The next step should be implementing other types of ranks (MeritRank, Personal Hitting Time)
and enable them in as dependency injections.
* After that, we should add some networking layer, e.g. IPv8
* At that point, we should add security in the form of signature verification