argmax = max
argmin = min

def minimax_decision(id, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states."""

    def max_value(id):
        state = game.graph.node()[id]['state']
        terminal, _ = game.terminal_test(state)

        if terminal:
            return game.graph.node()[id]['utility']
        v = -float('inf')
        for a in game.graph.successors(id):
            v = max(v, min_value(a))
        return v

    def min_value(id):
        state = game.graph.node()[id]['state']
        terminal, _ = game.terminal_test(state)

        if terminal:
            return game.graph.node()[id]['utility']
        v = float('inf')
        for a in game.graph.successors(id):
            v = min(v, max_value(a))
        return v

    # Body of minimax_decision:
    return argmin(game.graph.successors(id),
                  key=lambda id: max_value(id))

