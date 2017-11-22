import numpy as np
from collections import OrderedDict


class Graph():
    def __init__(self):
        self.nodes = OrderedDict()
        self.links = OrderedDict()
        self.constraints = list()

    def add_node(self, node, domain):
        self.nodes[node] = domain

    def add_uni_directed_link(self, node_a, node_b):
        self.links[node_a] = node_b


class Backtracking():
    def __init__(self, graph):
        self.graph = graph
        self.assignment = OrderedDict()
        self.unassigned = OrderedDict()
        self.assignment = self.graph.nodes.fromkeys(self.graph.nodes)
        self.unassigned = self.graph.nodes.fromkeys(self.graph.nodes)

    def backtrack(self):
        if all(value or value == 0 for value in self.assignment.itervalues()):
            return self.assignment
        var = self.select_unassigned_variable()
        for value in self.unassigned[var]:
            self.unassigned[var].remove(value)
            if self.constraints(var, value):
                self.assignment[var] = value
                result = self.backtrack()
                if not result or result == 0:
                    return value
                self.assignment[var] = None
        return var

    def select_unassigned_variable(self):
        # from assignment select key with empty domain
        for key, domain in self.unassigned.iteritems():
            if not domain:
                # refresh domain
                self.unassigned[key] = self.graph.nodes[key]
                return key

    def constraints(self, var, value):
        # all different
        temp = dict(self.assignment)
        del temp['U']
        seen = set()
        seen.add(value)
        all_diff = True
        if var != 'U':
            all_diff = not any(val in seen and val != None or seen.add(val) for val in temp.itervalues())

        # check sum
        true_sum = True
        if all(value or value == 0 for value in temp.itervalues()):
            a = self.assignment['A']
            b = self.assignment['B']
            c = self.assignment['C']
            if a + b != c + 10*value:
                true_sum = False

        return all_diff and true_sum


if __name__ == '__main__':
    graph = Graph()

    # domains
    A = np.arange(10).tolist()
    B = np.arange(10).tolist()
    C = np.arange(10).tolist()
    U = np.arange(2).tolist()

    # nodes
    graph.add_node('C', C)
    graph.add_node('B', B)
    graph.add_node('A', A)
    graph.add_node('U', U)

    # uni directed links
    graph.add_uni_directed_link('C', 'B')
    graph.add_uni_directed_link('B', 'A')
    graph.add_uni_directed_link('A', 'U')

    # find the solution
    assignment = Backtracking(graph).backtrack()
    print(assignment)
