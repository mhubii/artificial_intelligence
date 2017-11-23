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
        self.assignment = self.assignment.fromkeys(self.graph.nodes.keys())

    def backtrack(self):
        if self.complete():
            return self.assignment
        var = self.select_unassigned_variable()
        for val in self.order_domain_values(var):
            if self.consistent(var, val):
                self.assignment[var] = val
                result = self.backtrack()
                if result is not self.failure():
                    return result
                self.assignment[var] = None
        return self.failure()

    def forward_checking(self, var):
        temp = self.graph.nodes[var]
        for ass_val in self.assignment.itervalues():
            if var is not 'U' and ass_val is not None:
                temp.remove(ass_val)
        return temp

    def complete(self):
        if all(val is not None for val in self.assignment.itervalues()):
            return self.assignment['A'] + self.assignment['B'] == self.assignment['C'] + 10*self.assignment['U']
        else:
            return False

    def select_unassigned_variable(self):
        for key, val in self.assignment.iteritems():
            if val is None:
                return key

    def order_domain_values(self, var):
        feasible = self.forward_checking(var)
        return feasible

    def consistent(self, var, val):
        unique = True
        sum = True
        if var is not 'U':
            seen = set()
            seen.add(val)
            temp = dict(self.assignment)
            del temp['U']
            unique = not any(value in seen and value is not None or seen.add(value) for value in temp.itervalues())
        if var is 'U':
            sum = (self.assignment['A'] + self.assignment['B'] == self.assignment['C'] + 10*val)
        return unique and sum

    def failure(self):
        return 'failure'


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
