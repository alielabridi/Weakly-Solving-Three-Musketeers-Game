"""Games, or Adversarial Search (Chapter 5)"""

from collections import namedtuple
import random

from utils import argmax

infinity = float('inf')
GameState = namedtuple('GameState', 'to_move, utility, board, moves')
memo_max = {}
memo_min = {}
memo = {}
# ______________________________________________________________________________
# Minimax Search


def minimax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. """

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minimax_decision:
    return argmax(game.actions(state),
                  key=lambda a: min_value(game.result(state, a)))

# ______________________________________________________________________________


def alphabeta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""
    # return the best move from that state
    print "alpha beta exploring"
    print state.board
    if(state.board in memo):
        print "no need to go alpha-beta"
        return memo[state.board]
    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta):
        print "max_value exploring"
        print state.board
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            if(a in memo_max):
                print("memo max used in " + a)
                v = memo_max[a]
            else:
                v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        print "min_value exploring"
        print state.board
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            if(a in memo_min):
                print("memo min used in " + a)
                v = memo_min[a]
            else:
                v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_cutoff_search:
    best_score = -infinity
    beta = infinity
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    print "best action"
    print best_action
    memo[state.board] = best_action 
    return best_action

# ______________________________________________________________________________
# Players for Games


def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move_string = input('Your move? ')
    try:
        move = eval(move_string)
    except NameError:
        move = move_string
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state))


def alphabeta_player(game, state):
    return alphabeta_search(state, game)


# ______________________________________________________________________________
# Some Sample Games


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        print "initial state of the board"
        print state
        while True:
            for player in players:
                print ("player " + state.to_move)
                move = player(self, state)
                print move
                state = self.result(state, move)
                if self.terminal_test(state):
                    print("end state of the board")
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

class NIM(Game):
    def __init__(self):
        self.initial = GameState(to_move='0', utility=0, board=(2,3,0), moves=[])

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        list_moves = []
        for x in range (0,state.board[0]):
                list_moves.append((x,state.board[1],state.board[2]))
        for x in range (0,state.board[2]):
                list_moves.append((state.board[0],state.board[1],x))
        for x in range (0,state.board[1]):
                list_moves.append((state.board[0],x,state.board[2]))
        return list_moves

    def result(self, state, move):
        return GameState(to_move=('1' if state.to_move == '0' else '1'),
                         utility=self.compute_utility(state.board, move, state.to_move),
                         board=move, moves=[])

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == '0' else -state.utility

    def terminal_test(self, state):
        return state.utility != 0 or state.board == None

    def to_move(self, state):
        return '1' if state.to_move == '0' else '1'

    def compute_utility(self, board, move, to_move):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if(board[0]+board[1]+board[2] != 0): return 0
        if(to_move == '1'): return -1
        return 1



# Creating the game instances
NIM_game = NIM()


def gen_state(to_move='X', x_positions=[], o_positions=[], h=3, v=3, k=3):
    """Given whose turn it is to move, the positions of X's on the board, the
    positions of O's on the board, and, (optionally) number of rows, columns
    and how many consecutive X's or O's required to win, return the corresponding
    game state"""

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) \
        - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)


def test_minimax_decision():
    assert minimax_decision('A', f52) == 'a1'
    assert minimax_decision('B', f52) == 'b1'
    assert minimax_decision('C', f52) == 'c1'
    assert minimax_decision('D', f52) == 'd3'


def test_alphabeta_search():
    assert alphabeta_search('A', f52) == 'a1'
    assert alphabeta_search('B', f52) == 'b1'
    assert alphabeta_search('C', f52) == 'c1'
    assert alphabeta_search('D', f52) == 'd3'

    state = gen_state(to_move='X', x_positions=[(1, 1), (3, 3)],
                      o_positions=[(1, 2), (3, 2)])
    assert alphabeta_search(state, ttt) == (2, 2)

    state = gen_state(to_move='O', x_positions=[(1, 1), (3, 1), (3, 3)],
                      o_positions=[(1, 2), (3, 2)])
    assert alphabeta_search(state, ttt) == (2, 2)

    state = gen_state(to_move='O', x_positions=[(1, 1)],
                      o_positions=[])
    assert alphabeta_search(state, ttt) == (2, 2)

    state = gen_state(to_move='X', x_positions=[(1, 1), (3, 1)],
                      o_positions=[(2, 2), (3, 1)])
    assert alphabeta_search(state, ttt) == (1, 3)


def test_random_tests():
    #assert Fig52Game().play_game(alphabeta_player, alphabeta_player) == 3

    # The player 'X' (one who plays first) in TicTacToe never loses:
    assert ttt.play_game(alphabeta_player, random_player) >= 0 

    # The player 'X' (one who plays first) in TicTacToe never loses:
    #assert ttt.play_game(alphabeta_player, random_player) >= 0

#test_random_tests()
print("Starting the program...")

#stateNIM = GameState(to_move='0', utility=0, board=[20,2,1], moves=[])
print ("winner is " + str(NIM_game.play_game(alphabeta_player,alphabeta_player)))
#print alphabeta_search(stateNIM,NIM_game)
#NIM_game.play_game(alphabeta_search(stateNIM, NIM_game),query_player )