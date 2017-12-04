from collections import namedtuple
import random
from copy import deepcopy
from sys import getsizeof
from random import randint
import profile
import copy


from utils import argmax
import time

infinity = float('inf')
GameState = namedtuple('GameState', 'to_move, utility, board, moves')
memo_maximizer_max_value = {}
memo_maximizer_min_value = {}
memo_minimizer_max_value = {}
memo_minimizer_min_value = {}
memorization = 1
count_memo_minimizer_max_value = 0
count_memo_minimizer_min_value = 0
count_memo_maximizer_max_value = 0
count_memo_maximizer_min_value = 0
count_alpha_prunning = 0
count_beta_prunning = 0
max_depth_maximizer = 0
max_depth_minimizer = 0
printing_interval = 10
initial_board = [\
    ['|','-','-','-','-','-','|'],\
    ['|','G','G','G','G','M','|'],\
    ['|','G','G','G','G','G','|'],\
    ['|','G','G','M','G','G','|'],\
    ['|','G','G','G','G','G','|'],\
    ['|','M','G','G','G','G','|'],\
    ['|','-','-','-','-','-','|']]
ZobristTable = [\
                [[0,0],[0,0],[0,0],[0,0],[0,0]],
                [[0,0],[0,0],[0,0],[0,0],[0,0]],
                [[0,0],[0,0],[0,0],[0,0],[0,0]],
                [[0,0],[0,0],[0,0],[0,0],[0,0]],
                [[0,0],[0,0],[0,0],[0,0],[0,0]]\
                ]
def init_table_zobrist():
    for i in range (0,5):
        for j in range(0,5):
            for k in range(0,2):
                #print (i)
                #print(j)
                #print(k)
                ZobristTable[i][j][k] = randint(0,2**64)
                # print (ZobristTable[i][j][k])
def compute_hash(board):
    h = 0
    for i in range (1,6):
        for j in range(1,6):
            if(board[i][j] != ' '):
                piece = 0 if board[i][j] == 'M' else 1
                h ^= ZobristTable[i-1][j-1][piece]
    return h

def board_piece_comparaison(p1,p2):
    piece_value = {'M': 3 , 'G' : 2, ' ': 1}
    if piece_value[p1] > piece_value[p2] : return 1
    elif piece_value[p1] < piece_value[p2] : return -1
    return 0 

def least_lexicographical_board_hash(board):
    return compute_hash(board)
    least_lexicographical_board = board
    for rep in range(0,3):
        exitFlag = False
        board_rotation = list(zip(*board[::-1])) 
        
        for i in range(1,6):
            for j in range(1,6):
                comparaison_value = board_piece_comparaison(least_lexicographical_board[i][j], board_rotation[i][j])
                if(comparaison_value == 1):
                    exitFlag = True
                    break
                elif(comparaison_value == -1):
                    exitFlag = True
                    least_lexicographical_board = board_rotation
                    break
            if(exitFlag): break
        board = board_rotation
    return compute_hash(least_lexicographical_board)

initial_state_standard_three_musketeers = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
startprint = time.time()
start = time.time()

def within_board_range(x,y):
    return not(x >= 5 or y >= 5 or y < 0 or x < 0)


def reinitiliaze_vars():
    memo_maximizer_max_value = {}
    memo_maximizer_min_value = {}
    memo_minimizer_max_value = {}
    memo_minimizer_min_value = {}
    count_memo_minimizer_max_value = 0
    count_memo_minimizer_min_value = 0
    count_memo_maximizer_max_value = 0
    count_memo_maximizer_min_value = 0
    count_alpha_prunning = 0
    count_beta_prunning
    max_depth_maximizer = 0
    max_depth_minimizer = 0
    init_table_zobrist()

def print_info_vars():
    print ("memorization = " + str(memorization))
    print ("count_memo_minimizer_max_value = " + str(count_memo_minimizer_max_value))
    print ("count_memo_minimizer_min_value = " + str(count_memo_minimizer_min_value))
    print ("count_memo_maximizer_max_value = " + str(count_memo_maximizer_max_value))
    print ("count_memo_maximizer_min_value = " + str(count_memo_maximizer_min_value))
    print ("count_alpha_prunning = " + str(count_alpha_prunning))
    print ("count_beta_prunning = " + str(count_beta_prunning))
    print ("max_depth_maximizer = " + str(max_depth_maximizer))
    print ("max_depth_minimizer = " + str(max_depth_minimizer))
    print ("memo_maximizer_max_value = " + str(len(memo_maximizer_max_value)))
    print ("memo_maximizer_min_value = " + str(len(memo_maximizer_min_value)))
    print ("memo_minimizer_max_value = " + str(len(memo_minimizer_max_value)))
    print ("memo_minimizer_min_value = " + str(len(memo_minimizer_min_value)))
    print("time spent: " + str(time.time() - start))
    return
# ______________________________________________________________________________

def max(a,b):
    return a if a>b else b

def alphabeta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""
    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta,depth):
        #print("max "+str(depth))

        global startprint
        global printing_interval
        if(time.time() - startprint > printing_interval):
            startprint = time.time()
            print_info_vars()


        global count_memo_maximizer_max_value
        global count_memo_minimizer_max_value
        global max_depth_minimizer
        global max_depth_maximizer
        global memo_maximizer_max_value
        global memo_minimizer_max_value
        global count_beta_prunning

        #print "depth in max: " + str(depth) 
        if game.terminal_test(state):
            # print "--------------------"
            # for r in state.board:
            #     print r
            # print "utility = " + str(game.utility(state, player))
            # print "--------------------"
            #print("utility " + str(game.utility(state, player)) + str(depth))
            return game.utility(state, player)
        v = -infinity
        player_about_to_move = 'M' if state.to_move == 'G' else 'G'
        statehash = least_lexicographical_board_hash(state.board)
        if(player_about_to_move == 'M' and statehash in memo_maximizer_max_value and memorization):
            count_memo_maximizer_max_value = count_memo_maximizer_max_value + 1
            #print("memo used")

            #print "used memorization in memo_maximizer_max_value #= " + str(count_memo_maximizer_max_value)
            return memo_maximizer_max_value[statehash]
        elif(player_about_to_move == 'G' and statehash in memo_minimizer_max_value and memorization):
            count_memo_minimizer_max_value = count_memo_minimizer_max_value + 1
            #print("memo used")

            #print "used memorization in memo_minimizer_min_value #=" + str(count_memo_minimizer_min_value)
            return memo_minimizer_max_value[statehash]

        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta,depth+1)) 
            #print ("depth max " + str(depth)+player+str(v))
            if v >= beta or (v == 1 and player == 'M') :
                count_beta_prunning = count_beta_prunning + 1
                if(player_about_to_move == 'M' and memorization): 
                    max_depth_maximizer = max(depth+1,max_depth_maximizer)
                    memo_maximizer_max_value[statehash] = v
                elif(player_about_to_move == 'G' and memorization):
                    max_depth_minimizer = max(depth+1, max_depth_minimizer)
                    memo_minimizer_max_value[statehash] = v
                #print("utility " + str(v)+ str(depth))
                return v
            alpha = max(alpha, v)
        if(player_about_to_move == 'M' and memorization): 
            max_depth_maximizer = max(depth+1,max_depth_maximizer)
            memo_maximizer_max_value[statehash] = v
        elif(player_about_to_move == 'G' and memorization):
            max_depth_minimizer = max(depth+1, max_depth_minimizer)
            memo_minimizer_max_value[statehash] = v
        #print("utility " + str(v) + str(depth))
        return v

    def min_value(state, alpha, beta,depth):
        #print("min " + str(depth))

        global startprint
        global printing_interval
        if(time.time() - startprint > printing_interval):
            startprint = time.time()
            print_info_vars()


        global count_memo_maximizer_min_value
        global count_memo_minimizer_min_value
        global max_depth_minimizer
        global max_depth_maximizer
        global memo_maximizer_min_value
        global memo_minimizer_min_value
        global count_alpha_prunning

        #print "depth in min: " + str(depth) 
        if game.terminal_test(state):
            # print "--------------------"
            # for r in state.board:
            #     print r
            # print "utility = " + str(game.utility(state, player))
            # print "--------------------"
            return game.utility(state, player)
        statehash = least_lexicographical_board_hash(state.board)
        v = infinity
        player_about_to_move = 'M' if state.to_move == 'G' else 'G'
        if(player_about_to_move == 'M' and statehash in memo_maximizer_min_value and memorization):
            count_memo_maximizer_min_value = count_memo_maximizer_min_value + 1
            #print("memo used")
            #print "used memorization in memo_maximizer_min_value #=" + str(count_memo_maximizer_min_value)
            return memo_maximizer_min_value[statehash]
        elif(player_about_to_move == 'G' and statehash in memo_minimizer_min_value and memorization):
            count_memo_minimizer_min_value = count_memo_minimizer_min_value + 1
            #print("memo used")
            #print "used memorization in memo_minimizer_min_value #=" + str(count_memo_minimizer_min_value)
            return memo_minimizer_min_value[statehash]

        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta,depth+1))
            if v <= alpha or (v == 1 and player == 'M'):
                count_alpha_prunning = count_alpha_prunning + 1
                if(player_about_to_move == 'M' and memorization): 
                    max_depth_maximizer = max(depth+1,max_depth_maximizer)
                    memo_maximizer_min_value[statehash] = v
                elif(player_about_to_move == 'G' and memorization):
                    max_depth_minimizer = max(depth+1, max_depth_minimizer)
                    memo_minimizer_min_value[statehash] = v
                return v
            beta = min(beta, v)

        if(player_about_to_move == 'M' and memorization): 
            max_depth_maximizer = max(depth+1,max_depth_maximizer)
            memo_maximizer_min_value[statehash] = v
        elif(player_about_to_move == 'G' and memorization):
            max_depth_minimizer = max(depth+1, max_depth_minimizer)
            memo_minimizer_min_value[statehash] = v
        return v

    # Body of alphabeta_cutoff_search:
    best_score = -infinity
    beta = infinity
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta,0)
        # print ("comparing the value v = " + str(v) + " and best_score " + str(best_score))
        # print ("with action")
        # for r in a:
        #     print (r)
        
        if( v == 1 and player == 'M' or v == -1 and player == 'G'):
            best_score = v
            best_action = a
            break
        if v > best_score:
            best_score = v
            best_action = a
        # if v == 1 and player == 'M':
        #     print("found it")
        #     best_score = v
        #     best_action = a
        #     break
        #print "best score so far is " + str(best_score)
    print ("-----")
    print ("to_move " + player + " utility = " + str(best_score))
    for r in best_action:
        print (r)
    print ("-----")
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
        #return not self.actions(state)
        raise NotImplementedError
        

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, initial_state = initial_state_standard_three_musketeers, *players):
        """Play an n-person, move-alternating game."""
        state = initial_state
        print ("initial state of the board")
        for r in state.board:
            print (r)
        while True:
            for player in players:
                move = player(self, state)
                #print "the move chosen by the player " + state.to_move 
                state = self.result(state, move)
                #print "utility 2 " + str(state.utility)
                #print self.display(state)
                #print ""
                #print ""
                #print ""
                #print ""
                if self.terminal_test(state):
                    print("end state of the board")
                    self.display(state)
                    return self.utility(state, self.to_move(initial_state))



class NIM(Game):
    def __init__(self):
        self.initial = GameState(to_move='0', utility=0, board=(9,9,9), moves=[])

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
        return GameState(to_move=('1' if state.to_move == '0' else '0'),
                         utility=self.compute_utility(state.board, move, state.to_move),
                         board=move, moves=[])

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return self.compute_utility(state.board, [], state.to_move) if player == '0' else -self.compute_utility(state.board, [], state.to_move)

    def terminal_test(self, state):
        return self.compute_utility(state.board, [], state.to_move) != 0 or state.board == None

    def to_move(self, state):
        return '1' if state.to_move == '0' else '0'

    def compute_utility(self, board, move, to_move):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if(board[0]+board[1]+board[2] != 0): return 0
        if(to_move == '1'): return -1
        return 1


Orthogonal_moves = [(1,0),(-1,0),(0,-1),(0,1)]
# initial_board = (\
#     (' ',' ',' ',' ','M'),\
#     (' ',' ','G',' ',' '),\
#     (' ',' ','M',' ',' '),\
#     (' ',' ',' ',' ',' '),\
#     ('M',' ',' ',' ',' '))

# initial_board = (\
#     (' ',' ',' ',' ',' '),\
#     (' ','M','G','M',' '),\
#     (' ',' ','M',' ',' '),\
#     (' ',' ',' ',' ',' '),\
#     (' ',' ',' ',' ',' '))

class ThreeMusketeers(Game):
    def __init__(self):
        self.Musketeers_positions = []

    def terminal_test(self, state):
        return 0 if state.utility == 0 else 1
        
    def to_move(self, state):
        return state.to_move

    def actions(self, state):
        List = []
        board = []
        count = 0
        if(state.to_move == 'M' and count <= 3):
            for x  in  range(1,6):
                for y in range(1,6):
                    if(state.board[x][y] == 'M'):
                        count += 1
                        for (i,j) in Orthogonal_moves:
                                if(state.board[x+i][y+j] == 'G'):
                                    board =  [[i for i in row] for row in state.board]
                                    board[x][y] = ' '
                                    board[x+i][y+j] = 'M'
                                    List.append(board)

        elif(state.to_move == 'G'):
            for i  in  range(1,6):
                for j in range(1,6):
                    if(state.board[i][j] == 'G'):
                        for (x,y) in Orthogonal_moves:
                                if(state.board[i+x][j+y] == ' '):
                                    board =  [[i for i in row] for row in state.board]
                                    board[i][j] = ' '
                                    board[i+x][j+y] = 'G'
                                    List.append(board)
        del board
        return List

    def result(self, state, move):
        return GameState(to_move=('M' if state.to_move == 'G' else 'G'),
                         utility=self.compute_utility(move, {},state.to_move),
                         board=move, moves={})

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'M' else -state.utility

    def compute_utility(self, board, move, to_move):
        """If 'M' wins with this move, return 1; if 'G' wins return -1; else return 0."""
        #print self.Musketeers_positions
        # same row(Guardmen winning)
        self.Musketeers_positions = []
        for i  in  range(1,6):
            for j in range(1,6):
                if(board[i][j] == 'M'):
                    self.Musketeers_positions.append((i,j))

        if(self.Musketeers_positions[0][0] == self.Musketeers_positions[1][0] and
         self.Musketeers_positions[1][0] == self.Musketeers_positions[2][0]):
            return -1
        # same column (Gardsmen winning)
        if(self.Musketeers_positions[0][1] == self.Musketeers_positions[1][1] and
         self.Musketeers_positions[1][1] == self.Musketeers_positions[2][1]):
            #print "this 2 -1"
            return -1
        #still possible moves (nobody wins yet!)
        for (x,y) in self.Musketeers_positions:
            for (i,j) in Orthogonal_moves:
                    if(board[x+i][y+j] == 'G'):
                        #print "this (0)"
                        return 0
        #print "this (1)"
        #print to_move
        # no possible move (Musketeers winning)
        return 1

    def display(self, state):
        for r in state.board:
            print (r)

# Creating the game instances
NIM_game = NIM()
ThreeMusketeers_game = ThreeMusketeers()


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

def test_random_tests():

    # The player 'X' (one who plays first) in TicTacToe never loses:
    assert ttt.play_game(alphabeta_player, random_player) >= 0 


def unit_testing():
    print("+++++++++++++++++++++")
    # initial_board = (\
    # ('M',' ','G',' ','M'),\
    # (' ',' ','G',' ',' '),\
    # (' ',' ','G',' ','G'),\
    # (' ',' ','M',' ',' '),\
    # (' ',' ',' ',' ',' '))
    # initial_state = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
    # #Guardmen winning
    # assert(ThreeMusketeers_game.play_game(initial_state,alphabeta_player,alphabeta_player) == -1)

    # print("+++++++++++++++++++++")
    # initial_board = (\
    # ('M',' ','G',' ','M'),\
    # (' ',' ','G',' ',' '),\
    # (' ',' ','G',' ','G'),\
    # (' ',' ','G',' ',' '),\
    # (' ',' ','M',' ',' '))
    # initial_state = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
    # #Guardmen winning
    # assert(ThreeMusketeers_game.play_game(initial_state,alphabeta_player,alphabeta_player) == -1)

    # print("+++++++++++++++++++++")
    # initial_board = (\
    # ('M',' ',' ',' ','G'),\
    # (' ',' ',' ',' ',' '),\
    # ('G','G','G','G','M'),\
    # (' ',' ',' ',' ',' '),\
    # ('M',' ','G',' ',' '))
    # initial_state = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
    # #Guardmen winning
    # assert(ThreeMusketeers_game.play_game(initial_state,alphabeta_player,alphabeta_player) == -1)

    print("+++++++++++++++++++++")
    initial_board = [\
    ['|','-','-','-','-','-','|'],\
    ['|',' ',' ',' ',' ','M','|'],\
    ['|',' ',' ','G',' ',' ','|'],\
    ['|',' ',' ','M',' ',' ','|'],\
    ['|',' ',' ',' ',' ',' ','|'],\
    ['|','M',' ',' ',' ',' ','|'],\
    ['|','-','-','-','-','-','|']]
    initial_state = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
    initial_Musketeers_positions = [(1,5),(3,3),(5,1)]
    #Musketeers winning
    assert(ThreeMusketeers_game.play_game(initial_state,alphabeta_player,alphabeta_player) == 1)

    print("+++++++++++++++++++++")
    initial_board = [\
        ['|','-','-','-','-','-','|'],\
        ['|',' ',' ',' ',' ',' ','|'],\
        ['|',' ','M','G','M',' ','|'],\
        ['|',' ',' ','M',' ',' ','|'],\
        ['|',' ',' ',' ',' ',' ','|'],\
        ['|',' ',' ',' ',' ',' ','|'],\
        ['|','-','-','-','-','-','|']]
    initial_state = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
    #Musketeers winning
    assert(ThreeMusketeers_game.play_game(initial_state,alphabeta_player,alphabeta_player) == 1)



    # print("+++++++++++++++++++++")
    # initial_board = (\
    #     (' ',' ',' ',' ',' '),\
    #     (' ',' ',' ',' ',' '),\
    #     (' ',' ',' ',' ',' '),\
    #     (' ',' ','M',' ',' '),\
    #     ('M',' ','G',' ','M'))
    # initial_state = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
    # #Musketeers winning
    # assert(ThreeMusketeers_game.play_game(initial_state,alphabeta_player,alphabeta_player) == -1)

    print("+++++++++++++++++++++")
    initial_board = [\
    ['|','-','-','-','-','-','|'],\
    ['|',' ',' ',' ',' ','G','|'],\
    ['|',' ','G','M','G',' ','|'],\
    ['|','G','G','M','G','G','|'],\
    ['|','M','G','G','G','G','|'],\
    ['|',' ',' ','G',' ',' ','|'],\
    ['|','-','-','-','-','-','|']]
    initial_state = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
    #Musketeers winning
    assert(ThreeMusketeers_game.play_game(initial_state,alphabeta_player,alphabeta_player) == 1)
    return
    print("+++++++++++++++++++++")
    initial_board = [\
    [' ',' ','M',' ','G'],\
    [' ','G',' ','G',' '],\
    ['M',' ','G',' ',' '],\
    ['M','G',' ',' ',' '],\
    [' ',' ','G',' ',' ']]
    initial_state = GameState(to_move = 'M', utility = 0, board = initial_board, moves = {})
    #Musketeers winning
    assert(ThreeMusketeers_game.play_game(initial_state,alphabeta_player,alphabeta_player) == 1)

    print ("All unit tests succeeded")


def main():
    #test_random_tests()
    print("Starting the program...")
    # reinitiliaze_vars()
    # start = time.time()
    # unit_testing()
    # print_info_vars()
    # end = time.time()
    # print("execution time for unit testing: " + str(end - start))
    #stateNIM = GameState(to_move='0', utility=0, board=[20,2,1], moves=[])
    #print ("winner is " + str(NIM_game.play_game(alphabeta_player,alphabeta_player)))
    reinitiliaze_vars()
    print (alphabeta_search(initial_state_standard_three_musketeers,ThreeMusketeers_game))
    start = time.time()
    print ("winner is " + str(ThreeMusketeers_game.play_game(initial_state_standard_three_musketeers,alphabeta_player,alphabeta_player)))
    end = time.time()
    print("execution time for getting the result: " + str(end - start))
    print_info_vars()

    #NIM_game.play_game(alphabeta_search(stateNIM, NIM_game),query_player )


#main()
profile.run('main()')
