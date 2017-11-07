"""
    Artificial Intelligence
    Prof. Bjoern Ommer
    WS17/18

    Ex02
"""


class Graph:
    def __init__(self):
        self.graph = {}

    def add_node(self, name):
        self.graph[name] = {}

    def add_link(self, start, end, cost):
        self.graph[start][end] = cost


class Treesearch:
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.visited_nodes = [start]
        self.cost_to_reach_node = graph[start]
        self.goal_not_reached = True

        # do some minimal checks
        if self.start == self.goal:
            self.goal_not_reached = False

    def optimal_path(self):
        while self.goal_not_reached:
            # find cheapest not visited step
            next_cheapest_step = min(self.cost_to_reach_node, key=self.cost_to_reach_node.get)

            if any(node == next_cheapest_step for node in self.visited_nodes):
                
            else

            # take cheapest step


            # store path and store visited nodes

            # return path if goal reached







if __name__ == '__main__':
    # build the graph
    map_of_romanian = Graph()

    map_of_romanian.add_node(name='Arad')
    map_of_romanian.add_link(start='Arad', end='Sibiu', cost=140)
    map_of_romanian.add_link(start='Arad', end='Zerind', cost=65)
    map_of_romanian.add_link(start='Arad', end='Lala', cost=70)

    print(map_of_romanian.graph)

    print(min(map_of_romanian.graph['Arad'], key=map_of_romanian.graph['Arad'].get))

    # find the cheapest path
