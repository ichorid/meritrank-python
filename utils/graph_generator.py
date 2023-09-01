from itertools import permutations

import networkx as nx
from _pytest.python_api import approx
from networkx.algorithms.approximation import traveling_salesman_problem

from meritrank_python.rank import IncrementalMeritRank


class TestGraphListBuilder:

    def __init__(self, num_vertices=3, weights=(-1, 0, 1)):
        self.num_vertices = num_vertices
        self.weights = weights

        # Define the possible edge weights
        self.edges = list(permutations(range(self.num_vertices), 2))

    def _gen_perm(self, result, prefix=None, level=0):
        if prefix is None:
            prefix = []
        if level == 0:
            result.append(prefix)
            return
        for w in self.weights:
            new_prefix = list(prefix)
            new_prefix.append(w)
            self._gen_perm(result, prefix=new_prefix, level=level - 1)

    def generate_weights_permutations(self):
        output = []
        self._gen_perm(output, level=len(self.edges))
        print(len(output))
        return output

    def graph_sig(self, graph):
        res = []
        for edge in self.edges:
            if graph.has_edge(*edge):
                res.append(graph.edges[edge]['weight'])
            else:
                res.append(0)
        return tuple(res)

    def generate_all_possible_graphs(self, weights_combinations):
        output = {}
        for weights_var in weights_combinations:
            graph = nx.DiGraph()
            for ind, edge in enumerate(self.edges):
                w = weights_var[ind]
                if w != 0:
                    graph.add_edge(*edge, weight=w)

            nx.set_node_attributes(graph,
                                   {n: (n == 0) for n in
                                    range(self.num_vertices)},
                                   name='ego')
            output[self.graph_sig(graph)] = graph
        return output

    def merge_symmetric_graphs(self, input_graphs):
        output = {}
        for sig, graph in input_graphs.items():
            progress = len(output)
            if progress % 100 == 0:
                print(progress)

            graph_exists = False
            for existing_graph in output.values():
                if nx.is_isomorphic(
                        graph,
                        existing_graph,
                        edge_match=lambda a, b: a['weight'] == b['weight'],
                        node_match=lambda a, b: a['ego'] == b['ego']
                ):
                    graph_exists = True
            if graph_exists:
                continue
            output[self.graph_sig(graph)] = graph
        return output

    def generate_graphs_transitions(self, input_graphs):

        steps = []
        for edge in self.edges:
            for weight in self.weights:
                steps.append(edge + (weight,))

        total_transitions = 0
        transitions = {}
        for sig, graph in input_graphs.items():
            for step in steps:
                graph_copy = nx.DiGraph(graph)
                edge = step[:2]
                weight = step[2]
                if graph_copy.has_edge(*edge):
                    graph_copy.remove_edge(*edge)
                if weight != 0:
                    graph_copy.add_edge(*edge, weight=weight)
                new_sig = self.graph_sig(graph_copy)
                if new_sig not in input_graphs:
                    continue
                transitions_for_sig = transitions[sig] = transitions.get(sig,
                                                                         {})
                # Limit the destination graph to the list of source graphs
                if new_sig != sig:
                    transitions_for_sig[step] = new_sig
                    total_transitions += 1

        # pprint (transitions)
        print(total_transitions)
        return transitions

    def remove_nonego_negs(self, weights_combinations_list):
        output = []
        for weights in weights_combinations_list:
            skip = False
            for ind, edge in enumerate(self.edges):
                w = weights[ind]
                if w < 0 and edge[0] != 0:
                    skip = True
                    break
            if skip:
                continue
            output.append(weights)
        return output

    def remove_unconnected_to_ego(self, weights_combinations_list):
        output = []
        for weights in weights_combinations_list:
            has_positive_edge_from_ego = False

            for ind, edge in enumerate(self.edges):
                w = weights[ind]
                if w > 0 and edge[0] == 0:
                    has_positive_edge_from_ego = True
                    break
            if has_positive_edge_from_ego:
                output.append(weights)
        return output


builder = TestGraphListBuilder(3)
perms = builder.generate_weights_permutations()
perms_filtered = builder.remove_unconnected_to_ego(
    builder.remove_nonego_negs(perms))
all_possible_graphs = builder.generate_all_possible_graphs(perms_filtered)
merged = builder.merge_symmetric_graphs(all_possible_graphs)
transitions = builder.generate_graphs_transitions(merged)
print(f"Num graphs: {len(merged)}")
# print(transitions)
sigs = sorted(merged.keys())
sigs_index = {sig: ind for ind, sig in enumerate(sigs)}
# pprint (sigs_index)

transitions_graph = nx.DiGraph()

for sig, trans_dict in transitions.items():
    src = sigs_index[sig]
    for change, result in trans_dict.items():
        dst = sigs_index[result]
        transitions_graph.add_edge(src, dst, change=change)

# print (transitions_graph.edges())
print(f"Num transitions: {len(transitions_graph.edges())}")
print(f"Transitions graph strongly connected: "
      f"{nx.is_strongly_connected(transitions_graph)}")

travel_path = traveling_salesman_problem(nx.Graph(transitions_graph))
travel_path = travel_path + list(reversed(travel_path))[1:]

edges_taken = set()
for i in range(len(travel_path)):
    edges_taken.add(tuple(travel_path[i: i + 2]))
    # edges_taken.update("-".join([str(x) for x in travel_path[i: i + 2]]))
    # edges_taken.update(tuple(travel_path[i: i + 2]))

print(f"Steps travel: {len(travel_path)} unique {len(edges_taken)}")

calculated_results = {}
for sig, graph in merged.items():
    ipr = IncrementalMeritRank(graph)
    ipr.calculate(0)
    # print(sig, ipr.get_ranks(0))
    calculated_results[sig] = ipr.get_ranks(0)
    pass

start_graph = merged[sigs[travel_path[0]]]
ipr = IncrementalMeritRank(start_graph)
ipr.calculate(0)

stepped_results = {sigs[travel_path[0]]: ipr.get_ranks(0)}
for i in range(len(travel_path) - 1):
    current_step = travel_path[i]
    next_step = travel_path[i + 1]
    orig_results = ipr.get_ranks(0)
    change = transitions_graph[current_step][next_step]['change']
    orig_graph = ipr.get_graph()
    ipr.add_edge(change[0], change[1], weight=change[2])
    stepped_results[sigs[next_step]] = (ipr.get_ranks(0))
    print(sigs[current_step])
    print(change)
    print(sigs[next_step])
    print(orig_graph)
    print(ipr.get_graph())
    print(orig_results)
    print(calculated_results[sigs[current_step]])
    print(stepped_results[sigs[next_step]])
    print(calculated_results[sigs[next_step]])
    print(flush=True)
    cres = calculated_results[sigs[next_step]]
    sres = stepped_results[sigs[next_step]]
    print (
        ("CRES", ) +
        change +
        sigs[next_step] +
        tuple(cres.get(v, 0.0) for v in range(builder.num_vertices)) +
        tuple(sres.get(v, 0.0) for v in range(builder.num_vertices)))

    if not stepped_results[sigs[next_step]] == approx(calculated_results[sigs[next_step]], 0.1):
        print ('WARN')

print(builder.edges)

# assert stepped_results == approx(calculated_results)

print(stepped_results)
print(calculated_results)
