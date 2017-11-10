"""
    Artificial Intelligence
    Prof. Bjoern Ommer
    WS17/18

    Ex02 Tree Search

"""


class Graph:
    def __init__(self):
        self.graph = {}

    def add_node(self, name):
        self.graph[name] = {}

    def add_bidirectional_link(self, node_a, node_b, cost):
        self.graph[node_a][node_b] = cost
        self.graph[node_b][node_a] = cost

    def add_unidirectional_link(self, start, end, cost):
        self.graph[start][end] = cost


def tree_search(graph, root, goal):
    visited = [root]
    frontier = graph[root].copy()

    # find optimal path
    while True:
        # take the cheapest step
        next_node = min(frontier, key=frontier.get)

        # go to next_node
        current_node = next_node
        visited.append(next_node)

        # end tree search and return path if goal was reached
        if current_node == goal:
            break
        else:
            for node in graph[current_node]:
                tmp_frontier = frontier[current_node] + graph[current_node][node]

                # only explore non-explored nodes. This works because it is for
                # sure that you've been there for less already
                if not any(node == vis_node for vis_node in visited):
                    # expand frontier if node not expanded yet
                    if not frontier.get(node):
                        frontier[node] = frontier[current_node] + graph[current_node][node]
                    # reset frontier if cheaper
                    elif frontier.get(node) and tmp_frontier <= frontier[node]:
                        frontier[node] = frontier[current_node] + graph[current_node][node]

        # delete visited node from frontier
        del frontier[current_node]

    return frontier[current_node], visited


def backtracking(graph, goal, root, cost):
    paths = [[goal]]
    visited = [[]]
    it = 0
    tracked_cost = [0]

    while True:
        valid_path = False
        tracked_cost.append(0)

        print(paths[it])

        for node in graph[paths[it][-1]]:
            if not any(paths[it] == paths[k] for k in range(it)):
                # next iteration
                it += 1
                visited.append([])

                if not any(node == vis for vis in visited[it]):
                    # track all paths
                    paths[it].append(node)
                    visited[it].append(node)
                    valid_path = True
                    break


        # if valid step calc cost
        if valid_path and tracked_cost[it] < cost:
            tracked_cost[it] += graph[paths[it][-2]][paths[it][-1]]

            # if path found break and return
            if tracked_cost[it] == cost and paths[it][-1] == root:
                break

        # else go one step back
        else:
            # keep everything expect the last step
            paths.append(paths[it][:-1])

    return paths[-1][::-1]


if __name__ == '__main__':
    # build the graph
    map_of_romanian = Graph()

    # add nodes
    map_of_romanian.add_node('Arad')
    map_of_romanian.add_node('Timisoara')
    map_of_romanian.add_node('Zerind')
    map_of_romanian.add_node('Oradea')
    map_of_romanian.add_node('Sibiu')
    map_of_romanian.add_node('Lugoj')
    map_of_romanian.add_node('Mehadia')
    map_of_romanian.add_node('Drobeta')
    map_of_romanian.add_node('Craiova')
    map_of_romanian.add_node('Rimnicu Vilcea')
    map_of_romanian.add_node('Pitesti')
    map_of_romanian.add_node('Fagaras')
    map_of_romanian.add_node('Bucharest')
    map_of_romanian.add_node('Giurgiu')
    map_of_romanian.add_node('Urziceni')
    map_of_romanian.add_node('Hirsova')
    map_of_romanian.add_node('Eforie')
    map_of_romanian.add_node('Vaslui')
    map_of_romanian.add_node('Iasi')
    map_of_romanian.add_node('Neamt')

    # add nodes
    map_of_romanian.add_bidirectional_link('Arad', 'Zerind', 75)
    map_of_romanian.add_bidirectional_link('Arad', 'Sibiu', 140)
    map_of_romanian.add_bidirectional_link('Zerind', 'Oradea', 71)
    map_of_romanian.add_bidirectional_link('Oradea', 'Sibiu', 151)
    map_of_romanian.add_bidirectional_link('Arad', 'Timisoara', 118)
    map_of_romanian.add_bidirectional_link('Timisoara', 'Lugoj', 111)
    map_of_romanian.add_bidirectional_link('Lugoj', 'Mehadia', 70)
    map_of_romanian.add_bidirectional_link('Mehadia', 'Drobeta', 75)
    map_of_romanian.add_bidirectional_link('Drobeta', 'Craiova', 120)
    map_of_romanian.add_bidirectional_link('Craiova', 'Rimnicu Vilcea', 146)
    map_of_romanian.add_bidirectional_link('Craiova', 'Pitesti', 138)
    map_of_romanian.add_bidirectional_link('Rimnicu Vilcea', 'Sibiu', 80)
    map_of_romanian.add_bidirectional_link('Rimnicu Vilcea', 'Pitesti', 97)
    map_of_romanian.add_bidirectional_link('Pitesti', 'Bucharest', 101)
    map_of_romanian.add_bidirectional_link('Sibiu', 'Fagaras', 99)
    map_of_romanian.add_bidirectional_link('Fagaras', 'Bucharest', 211)
    map_of_romanian.add_bidirectional_link('Bucharest', 'Giurgiu', 90)
    map_of_romanian.add_bidirectional_link('Bucharest', 'Urziceni', 85)
    map_of_romanian.add_bidirectional_link('Urziceni', 'Hirsova', 98)
    map_of_romanian.add_bidirectional_link('Hirsova', 'Eforie', 86)
    map_of_romanian.add_bidirectional_link('Urziceni', 'Vaslui', 142)
    map_of_romanian.add_bidirectional_link('Vaslui', 'Iasi', 92)
    map_of_romanian.add_bidirectional_link('Iasi', 'Neamt', 86)

    # find the cheapest path
    cost, visited = tree_search(map_of_romanian.graph, root='Timisoara', goal='Bucharest')
    #path = backtracking(map_of_romanian.graph, goal='Bucharest', root='Timisoara', cost=cost)

    print(cost, visited)
    #print(path)
