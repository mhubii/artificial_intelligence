import numpy as np
import Tkinter as tk
from networkx import DiGraph
import copy
from mini_max import minimax_decision


class TicTacToe:
    def __init__(self):
        self.board = np.full((3, 3), '', dtype=str)
        self.turn = ['x', 'o']
        self.graph = DiGraph()
        self.id = 0
        self.current_id = 0

    def make_move_by_id(self, id):
        self.current_id = id
        self.board = self.game_tree().node()[id]['state']
        self.turn = self.turn[::-1]

    def make_move(self, x, y):
        self.board[x, y] = self.turn[0]
        prev_id = self.current_id
        self.current_id = self.find_id(prev_id, self.board)
        self.turn = self.turn[::-1]

    def game_tree(self):
        self.graph.add_node(self.id, state=self.board, terminal=False, utility='')
        unexpanded = list()
        unexpanded.append(self.id)
        for depths in range(10):
            new_nodes = []
            for node in unexpanded:
                if not self.graph.node()[node]['terminal']:
                    moves, player = self.possible_moves(self.graph.node()[node]['state'])
                    for move in moves:
                        self.id += 1
                        new_nodes.append(self.id)
                        terminal, winner = self.terminal_test(move)
                        if winner == 'o':
                            utility = 1
                        elif winner == 'x':
                            utility = -1
                        else:
                            utility = 0
                        self.graph.add_node(self.id, state=move, terminal=terminal, utility=utility, player=player)
                        self.graph.add_edge(node, self.id)
            unexpanded = new_nodes

        return self.graph

    def possible_moves(self, state):
        moves = []
        for index, item in np.ndenumerate(state):
            if not item:
                move = copy.deepcopy(state)
                move[index] = self.turn[0]
                moves.append(move)
        player = self.turn
        self.turn = self.turn[::-1]
        return moves, player

    def terminal_test(self, state):
        # check horizontally and vertically
        for new_state in [state, np.transpose(state)]:
            for row in new_state:
                if len(set(row)) == 1 and row[0] != '':
                    return True, row[0]
        # check diagonally
        if len(set([state[i, i] for i in range(3)])) == 1 and state[0, 0] != '':
            return True, state[0, 0]
        if len(set([state[i, 2 - i] for i in range(3)])) == 1 and state[0, 2] != '':
            return True, state[0, 2]
        # check if full
        if all(filled for filled in np.nditer(state)):
            return True, 'draw'
        else:
            return False, None

    def find_id(self, prev_id, new_state):
        # find the id of a new state
        for succeessor in self.graph.successors(prev_id):
            if np.array_equal(self.graph.node()[succeessor]['state'], new_state):
                return succeessor


class SimpleTableInput(tk.Frame):
    def __init__(self, parent, rows, columns):
        tk.Frame.__init__(self, parent)

        self._entry = {}
        self.rows = rows
        self.columns = columns

        # register a command to use for validation
        vcmd = (self.register(self._validate), "%P")

        # create the table of widgets
        for row in range(self.rows):
            for column in range(self.columns):
                index = (row, column)
                e = tk.Entry(self, validate="key", validatecommand=vcmd)
                e.grid(row=row, column=column, stick="nsew")
                self._entry[index] = e
        # adjust column weights so they all expand equally
        for column in range(self.columns):
            self.grid_columnconfigure(column, weight=1)
        # designate a final, empty row to fill up any extra space
        self.grid_rowconfigure(rows, weight=1)

    def get(self):
        """

        Return a list of lists, containing the data in the table
        """
        result = np.empty((3, 3), dtype=str)
        for row in range(self.rows):
            for column in range(self.columns):
                index = (row, column)
                result[row, column] = self._entry[index].get()
        return result

    def set(self, x, y):
        """

        Assign new value to table
        """
        self._entry[x, y].insert(0, 'o')

    def set_board(self, board):
        for index, value in np.ndenumerate(board):
            self._entry[index].delete(0, 'end')
            self._entry[index].insert(0, value)

    def _validate(self, P):
        """
        Perform input validation.

        Allow only an empty value, or a value that can be converted to a str
        """
        if P.strip() == "":
            return True

        try:
            f = str(P)
        except ValueError:
            self.bell()
            return False
        return True


class Play(tk.Frame):
    def __init__(self, gui, game):
        tk.Frame.__init__(self, gui)
        self.table = SimpleTableInput(self, 3, 3)
        self.submit = tk.Button(self, text="Submit x", command=self.on_submit)
        self.table.pack(side="top", fill="both", expand=True)
        self.submit.pack(side="bottom")
        self.game = game

    def on_submit(self):
        # register new state with table.get()
        game.board = self.table.get()
        id = game.find_id(game.current_id, game.board)
        game.current_id = id

        # react to new state using mini max
        new_id = minimax_decision(game.current_id, game)
        game.make_move_by_id(new_id)
        self.table.set_board(game.board)


if __name__ == '__main__':
    # game
    game = TicTacToe()
    game.game_tree()

    # gui for the game
    gui = tk.Tk()
    Play(gui, game).pack(side="top", fill="both", expand=True)
    gui.mainloop()
